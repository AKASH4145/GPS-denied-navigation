r"""
04_track.py

Step 3: visual navigation along the whole flight. Instead of locating each frame from scratch,
each search window is centred on a prediction from the previous position:
    predicted position = last position + speed x time since last frame, along the compass heading
Speed is learned from the drone's own past visual fixes (no GPS). If a frame doesn't match, the
drone "coasts" on the prediction and the next search window gets bigger so it can re-acquire.

Same matcher and accept checks as 03_lightglue.py, plus:
- The ~14 m forward bias found by 03b_error_analysis.py is removed with --fwd-offset-m.
  That value was fitted on frames 3, 8, 13, ... so those frames are flown through but not
  counted in the scores here.
- Min inliers is 15 (03 used 4). In tracking, one wrong fix corrupts every later prediction, so
  coasting is safer than accepting a weak match. The 03 sweep showed this costs almost nothing.

Run from the project root (gps-denied-nav):
    python scripts\04_track.py --max-frames 30      # quick check, about 5 minutes
    python scripts\04_track.py                       # every 2nd frame of the flight, about 1 hour
Outputs go to outputs\04_track\<tag>\
"""

import argparse
import csv
import importlib
import math
import time
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightglue import LightGlue

from geo_utils import ROOT, DATA, haversine_m, load_drone_csv, load_metric_map, rotate_keep_all

# Reuse the matcher from step 2 and the preprocessing/drawing from the SIFT script
_lg = importlib.import_module("03_lightglue")
match, FEATURES = _lg.match, _lg.FEATURES
_sift = importlib.import_module("02_sift_baseline")
to_gray, save_viz, DRONE_FLAGS = _sift.to_gray, _sift.save_viz, _sift.DRONE_FLAGS


def is_calibration_frame(num):
    """Frames 3, 8, 13, ... were used to fit --fwd-offset-m, so they're not scored."""
    return (num - 1) % 5 == 2


def localize(r, cx, cy, win_m, ctx, args):
    """Match one drone frame against a search window centred on (cx, cy) in map pixels."""
    geo = ctx["geo"]
    hx, hy = win_m / 2 / geo.mpp_x, win_m / 2 / geo.mpp_y
    x0, y0 = int(max(0, cx - hx)), int(max(0, cy - hy))
    x1, y1 = int(min(geo.W, cx + hx)), int(min(geo.H, cy + hy))
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None                                   # window is off the map

    drone = cv2.imread(str(DATA / args.loc / "drone" / r["filename"]), DRONE_FLAGS[args.drone_reduce])
    if drone is None:
        return None
    gsd = 2 * r["height"] * math.tan(math.radians(args.hfov_deg) / 2) / drone.shape[1]
    f = gsd / geo.mpp_x
    drone = cv2.resize(drone, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    drone_up, M = rotate_keep_all(drone, -r["phi1"], return_matrix=True)
    mask = rotate_keep_all(np.full(drone.shape[:2], 255, np.uint8), -r["phi1"])
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8))

    res = match(to_gray(drone_up, ctx["clahe"]), mask, ctx["sat_gray"][y0:y1, x0:x1],
                ctx["ext_d"], ctx["ext_s"], ctx["matcher"])
    out = {"res": res, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "drone_up": drone_up,
           "ok": False, "est_xy": None, "footprint": None, "scale": None, "rot": None}
    if res["H"] is None:
        return out

    H = res["H"]
    A = H[:2, :2]
    scale = math.sqrt(abs(np.linalg.det(A)))
    rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    c = H @ np.array([drone_up.shape[1] / 2, drone_up.shape[0] / 2, 1.0])
    h0, w0 = drone.shape[:2]
    corners = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)
    out.update({
        "est_xy": (c[0], c[1]), "scale": scale, "rot": rot,
        "footprint": cv2.perspectiveTransform(cv2.transform(corners, M).astype(np.float32), H).reshape(-1, 2),
        "ok": len(res["inliers"]) >= args.min_inliers and 0.75 < scale < 1.33 and abs(rot) < args.max_rot,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="03")
    ap.add_argument("--step", type=int, default=2, help="use every Nth frame (2 = about 190 m apart)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--prior-m", type=float, default=200, help="error of the starting position")
    ap.add_argument("--init-win-m", type=float, default=1000, help="search window until speed is known")
    ap.add_argument("--win-m", type=float, default=700, help="search window while tracking")
    ap.add_argument("--grow", type=float, default=1.4, help="window growth factor after each missed frame")
    ap.add_argument("--max-win-m", type=float, default=1200, help="largest window (CPU memory limit)")
    ap.add_argument("--max-speed", type=float, default=40, help="ignore speed estimates above this (m/s)")
    ap.add_argument("--fwd-offset-m", type=float, default=14.3, help="forward bias to remove (from 03b)")
    ap.add_argument("--sat-mpp", type=float, default=0.5)
    ap.add_argument("--drone-reduce", type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument("--hfov-deg", type=float, default=49.2)
    ap.add_argument("--features", default="superpoint", choices=list(FEATURES))
    ap.add_argument("--max-kp-drone", type=int, default=2048)
    ap.add_argument("--max-kp-sat", type=int, default=4096)
    ap.add_argument("--max-rot", type=float, default=30)
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--viz-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or f"{args.features}_step{args.step}_win{args.win_m:g}_fwd{args.fwd_offset_m:g}"
    out = ROOT / "outputs" / "04_track" / tag
    (out / "viz").mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    Feat = FEATURES[args.features]
    sat_bgr, geo, region = load_metric_map(args.loc, args.sat_mpp)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ctx = {"geo": geo, "clahe": clahe, "sat_gray": to_gray(sat_bgr, clahe),
           "ext_d": Feat(max_num_keypoints=args.max_kp_drone).eval(),
           "ext_s": Feat(max_num_keypoints=args.max_kp_sat).eval(),
           "matcher": LightGlue(features=args.features).eval()}

    rows = load_drone_csv(args.loc)[::args.step]
    if args.max_frames:
        rows = rows[:args.max_frames]
    print(f"[{tag}] Location {args.loc} ({region}): {len(rows)} frames | window {args.win_m:.0f} m "
          f"(start {args.init_win_m:.0f} m, max {args.max_win_m:.0f} m) | forward bias removed {args.fwd_offset_m} m")

    # Starting position: GPS of the first frame plus a random error of up to --prior-m
    rng = np.random.default_rng(args.seed)
    gx, gy = geo.px(rows[0]["lat"], rows[0]["lon"])
    rad, th = args.prior_m * math.sqrt(rng.random()), 2 * math.pi * rng.random()
    pos = np.array([gx + rad * math.cos(th) / geo.mpp_x, gy - rad * math.sin(th) / geo.mpp_y])

    speed = None                 # m/s, learned from visual fixes only
    last_fix, last_fix_t = None, None
    t_prev = None
    win = args.init_win_m
    records = []

    for i, r in enumerate(rows):
        t0 = time.time()
        t = datetime.fromisoformat(r["date"])
        h = math.radians(r["phi1"])

        # ---------- predict ----------
        if t_prev is None or speed is None:
            pred = pos.copy()
        else:
            dt = (t - t_prev).total_seconds()
            pred = pos + np.array([speed * dt * math.sin(h) / geo.mpp_x, -speed * dt * math.cos(h) / geo.mpp_y])

        # ---------- measure ----------
        loc = localize(r, pred[0], pred[1], win, ctx, args)
        used_win = win
        if loc is not None and loc["ok"]:
            est = np.array([loc["x0"] + loc["est_xy"][0], loc["y0"] + loc["est_xy"][1]])
            # The match lands about fwd-offset metres ahead of the true position, so step back
            est[0] -= args.fwd_offset_m * math.sin(h) / geo.mpp_x
            est[1] += args.fwd_offset_m * math.cos(h) / geo.mpp_y
            if last_fix is not None:
                dts = (t - last_fix_t).total_seconds()
                dist = math.hypot((est[0] - last_fix[0]) * geo.mpp_x, (est[1] - last_fix[1]) * geo.mpp_y)
                if dts > 0 and dist / dts < args.max_speed:
                    v = dist / dts
                    speed = v if speed is None else 0.5 * speed + 0.5 * v
            last_fix, last_fix_t = est.copy(), t
            pos, status = est, "fix"
            win = args.win_m if speed is not None else args.init_win_m
        else:
            pos, status = pred, "coast"
            win = min(win * args.grow, args.max_win_m)
        t_prev = t

        # ---------- score against GPS (never used by the tracker) ----------
        est_lat, est_lon = geo.latlon(pos[0], pos[1])
        p_lat, p_lon = geo.latlon(pred[0], pred[1])
        err = haversine_m(r["lat"], r["lon"], est_lat, est_lon)
        pred_err = haversine_m(r["lat"], r["lon"], p_lat, p_lon)
        n_inl = len(loc["res"]["inliers"]) if loc else 0
        rec = {"num": r["num"], "filename": r["filename"], "scored": int(not is_calibration_frame(r["num"])),
               "status": status, "gt_lat": r["lat"], "gt_lon": r["lon"],
               "est_lat": round(est_lat, 7), "est_lon": round(est_lon, 7), "err_m": round(err, 1),
               "pred_err_m": round(pred_err, 1), "win_m": round(used_win), "n_inliers": n_inl,
               "speed_mps": round(speed, 2) if speed is not None else "", "time_s": round(time.time() - t0, 1)}
        records.append(rec)
        print(f"[{i + 1}/{len(rows)}] {r['filename']}: {status:5s} | window {used_win:5.0f} m | inliers {n_inl:4d} "
              f"| prediction off {pred_err:6.1f} m | error {err:6.1f} m | {rec['time_s']}s")

        if loc is not None and i % args.viz_every == 0:
            save_viz(out / "viz" / f"{args.loc}_{r['num']:04d}.jpg", loc["drone_up"],
                     sat_bgr[loc["y0"]:loc["y1"], loc["x0"]:loc["x1"]], loc["res"],
                     (geo.px(r["lat"], r["lon"])[0] - loc["x0"], geo.px(r["lat"], r["lon"])[1] - loc["y0"]),
                     loc["est_xy"] if loc["ok"] else None, loc["footprint"])

    # ---------- results ----------
    csv_path = out / f"track_{args.loc}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    sc = [r for r in records if r["scored"]]
    n = len(sc)
    errs = np.array([r["err_m"] for r in sc])
    fix_errs = np.array([r["err_m"] for r in sc if r["status"] == "fix"])
    streak, longest = 0, 0
    for r in records:
        streak = streak + 1 if r["status"] == "coast" else 0
        longest = max(longest, streak)

    print(f"\n===== Tracking [{tag}], location {args.loc}: {len(records)} frames flown, {n} scored =====")
    print(f"Visual fixes: {len(fix_errs)}/{n} ({100 * len(fix_errs) / n:.1f}%) | coasted: {n - len(fix_errs)}")
    for t_ in (25, 50, 100):
        print(f"Position within {t_:3d} m: {int((errs <= t_).sum())}/{n} ({100 * (errs <= t_).sum() / n:.1f}%)")
    print(f"Median error: all frames {np.median(errs):.1f} m | visual fixes only "
          f"{np.median(fix_errs) if len(fix_errs) else float('nan'):.1f} m | worst {errs.max():.1f} m")
    print(f"Median prediction error (how far off the motion model was before matching): "
          f"{np.median([r['pred_err_m'] for r in sc]):.1f} m")
    print(f"Longest run of missed frames: {longest} | frames more than 100 m off: {int((errs > 100).sum())}")
    print(f"Average time per frame: {np.mean([r['time_s'] for r in records]):.1f} s")

    # Map of the GPS path vs the tracked path
    scale = 2500 / max(geo.W, geo.H)
    thumb = cv2.resize(sat_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gps_px = np.int32([np.array(geo.px(r["gt_lat"], r["gt_lon"])) * scale for r in records])
    est_px = np.int32([np.array(geo.px(r["est_lat"], r["est_lon"])) * scale for r in records])
    cv2.polylines(thumb, [gps_px.reshape(-1, 1, 2)], False, (0, 255, 0), 3)      # GPS: green
    cv2.polylines(thumb, [est_px.reshape(-1, 1, 2)], False, (0, 0, 255), 2)      # tracked: red
    for r, p in zip(records, est_px):
        if r["status"] == "coast":
            cv2.circle(thumb, tuple(int(v) for v in p), 6, (0, 255, 255), 2)       # coasted: yellow
    cv2.imwrite(str(out / "track_map.jpg"), thumb)

    fig, ax = plt.subplots(figsize=(12, 4))
    nums = np.array([r["num"] for r in records])
    e_all = np.array([r["err_m"] for r in records])
    coast = np.array([r["status"] == "coast" for r in records])
    ax.plot(nums, e_all, color="lightgray", lw=1)
    ax.scatter(nums[~coast], e_all[~coast], s=10, color="tab:blue", label="visual fix")
    ax.scatter(nums[coast], e_all[coast], s=18, color="tab:orange", label="coasting (no match)")
    ax.axhline(25, ls="--", color="gray", lw=1)
    ax.set_xlabel("frame number")
    ax.set_ylabel("position error (m)")
    ax.set_title(f"Tracking error along the flight, median {np.median(errs):.1f} m")
    ax.legend()
    fig.savefig(out / "error_along_flight.png", dpi=120, bbox_inches="tight")

    print(f"Saved {csv_path.relative_to(ROOT)}, track_map.jpg and error_along_flight.png in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
