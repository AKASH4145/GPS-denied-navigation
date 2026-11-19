r"""
03b_error_analysis.py

Looks for a systematic offset in the position errors of accepted frames.
- World frame (east/north): a constant offset here points to the satellite map being shifted.
- Drone frame (forward/right, using heading Phi1): a constant offset here points to the camera not
  pointing straight down, or to a delay between the photo and the GPS log (the drone flies facing
  forward at about 13 m/s, so a lag shows up as a forward or backward offset).
The frame where the errors cluster tightest around their mean is the one the offset lives in.

The correction is tested honestly: the offset is fitted on the first half of the flight and applied
to the second half, and the other way round. No frame is corrected using an offset fitted on itself.

Run from the project root (gps-denied-nav):
    python scripts\03b_error_analysis.py
    python scripts\03b_error_analysis.py --stage 02_sift --tag sift_s1000_off2
Saves error_offsets.png next to the results CSV.
"""

import argparse
import csv
import math
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geo_utils import ROOT, haversine_m, load_drone_csv

M_PER_DEG_LAT = 110574.0   # metres per degree of latitude (near 32 deg N)
M_PER_DEG_LON_EQ = 111320.0  # metres per degree of longitude at the equator, scaled by cos(lat)


def cv_corrected(x, y):
    """Error size after removing a mean offset fitted on the other half of the flight."""
    half = len(x) // 2
    a, b = slice(0, half), slice(half, None)
    out = np.empty(len(x))
    fits = []
    for fit, apply in ((a, b), (b, a)):
        mx, my = x[fit].mean(), y[fit].mean()
        fits.append((mx, my))
        out[apply] = np.hypot(x[apply] - mx, y[apply] - my)
    return out, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="03_lightglue")
    ap.add_argument("--tag", default="superpoint_mpp0.5_s1000_off2")
    ap.add_argument("--loc", default="03")
    ap.add_argument("--good-m", type=float, default=25)
    args = ap.parse_args()

    folder = ROOT / "outputs" / args.stage / args.tag
    with open(folder / f"results_{args.loc}.csv", newline="") as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r["status"] == "ok"]
    n_all = len(all_rows)
    phi = {r["num"]: r["phi1"] for r in load_drone_csv(args.loc)}

    num = np.array([int(r["num"]) for r in rows])
    gt_lat = np.array([float(r["gt_lat"]) for r in rows])
    gt_lon = np.array([float(r["gt_lon"]) for r in rows])
    est_lat = np.array([float(r["est_lat"]) for r in rows])
    est_lon = np.array([float(r["est_lon"]) for r in rows])

    # Error vector (estimate minus GPS) in metres, east and north
    d_e = (est_lon - gt_lon) * M_PER_DEG_LON_EQ * np.cos(np.radians(gt_lat))
    d_n = (est_lat - gt_lat) * M_PER_DEG_LAT

    # Same error in the drone's frame. Heading is clockwise from north, as confirmed earlier.
    h = np.radians([phi[n] for n in num])
    fwd = d_e * np.sin(h) + d_n * np.cos(h)
    right = d_e * np.cos(h) - d_n * np.sin(h)

    # Split frames by flight direction: same way as the first frame, or roughly opposite
    same_dir = np.cos(h - h[0]) > 0

    # Ground speed at each frame from the GPS log (neighbouring photos), to test lag vs tilt
    all_rows = load_drone_csv(args.loc)
    idx = {r["num"]: i for i, r in enumerate(all_rows)}

    def gps_speed(n):
        i = idx[n]
        a, b = all_rows[max(i - 1, 0)], all_rows[min(i + 1, len(all_rows) - 1)]
        dt = (datetime.fromisoformat(b["date"]) - datetime.fromisoformat(a["date"])).total_seconds()
        return haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) / dt if dt > 0 else np.nan

    speeds = np.array([gps_speed(n) for n in num])
    height = np.mean([r["height"] for r in all_rows])

    raw = np.hypot(d_e, d_n)
    world_cv, world_fits = cv_corrected(d_e, d_n)
    body_cv, body_fits = cv_corrected(right, fwd)

    def spread(x, y):
        return float(np.sqrt(x.var() + y.var()))   # RMS distance from the mean offset

    print(f"{folder.relative_to(ROOT)}: {len(rows)} accepted of {n_all} frames\n")
    print(f"World frame: mean offset east {d_e.mean():+.1f} m, north {d_n.mean():+.1f} m | "
          f"spread around it {spread(d_e, d_n):.1f} m")
    print(f"  fitted on each half: " + ", ".join(f"({e:+.1f} E, {n:+.1f} N)" for e, n in world_fits))
    print(f"Drone frame: mean offset forward {fwd.mean():+.1f} m, right {right.mean():+.1f} m | "
          f"spread around it {spread(right, fwd):.1f} m")
    print(f"  fitted on each half: " + ", ".join(f"({f_:+.1f} fwd, {r_:+.1f} right)" for r_, f_ in body_fits))
    print("(the frame with the smaller spread is where the offset really is; the two halves should agree)\n")

    print("Forward offset by flight direction (timing lag or camera tilt?):")
    for sel in (same_dir, ~same_dir):
        if not sel.any():
            continue
        heading = np.degrees(np.arctan2(np.sin(h[sel]).mean(), np.cos(h[sel]).mean())) % 360
        f_mean, spd = fwd[sel].mean(), np.nanmedian(speeds[sel])
        print(f"  heading ~{heading:3.0f} deg: forward {f_mean:+5.1f} m | ground speed {spd:4.1f} m/s | "
              f"as a lag {f_mean / spd:4.2f} s | as a tilt {math.degrees(math.atan(f_mean / height)):4.2f} deg")
    print("(similar lag in both directions -> timing lag; similar tilt, i.e. similar metres -> camera tilt)\n")

    print(f"{'correction':<30} | {'median error':>12} | {'within ' + str(int(args.good_m)) + ' m (of all frames)':>26}")
    print("-" * 76)
    for name, e in (("none", raw), ("world offset, cross-validated", world_cv),
                    ("drone offset, cross-validated", body_cv)):
        within = int((e <= args.good_m).sum())
        print(f"{name:<30} | {np.median(e):>10.1f} m | {within:>5}/{n_all} ({100 * within / n_all:.1f}%)")

    # ---------- plot ----------
    groups = [(same_dir, "tab:blue"), (~same_dir, "tab:orange")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, x, y, xl, yl, title in (
        (axes[0], d_e, d_n, "east error (m)", "north error (m)", "World frame"),
        (axes[1], right, fwd, "right error (m)", "forward error (m)", "Drone frame"),
    ):
        for sel, color in groups:
            if sel.any():
                heading = np.degrees(np.arctan2(np.sin(h[sel]).mean(), np.cos(h[sel]).mean())) % 360
                ax.scatter(x[sel], y[sel], color=color, s=18, alpha=0.8,
                           label=f"heading ~{heading:.0f} deg ({sel.sum()} frames)")
        ax.plot(x.mean(), y.mean(), "k+", ms=16, mew=2, label=f"mean ({x.mean():+.1f}, {y.mean():+.1f})")
        ax.add_patch(plt.Circle((0, 0), args.good_m, fill=False, ls="--", color="gray"))
        ax.axhline(0, color="lightgray", lw=0.8)
        ax.axvline(0, color="lightgray", lw=0.8)
        lim = max(45, np.abs(np.concatenate([x, y])).max() * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{title}, spread {spread(x, y):.1f} m")
        ax.legend(loc="upper left", fontsize=8)
    png = folder / "error_offsets.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    print(f"\nSaved {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()