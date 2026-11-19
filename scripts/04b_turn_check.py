r"""
04b_turn_check.py

Looks at the GPS log and the tracker's output around the turns, to see why the motion prediction
jumped hundreds of metres there. For each frame it prints the time gap and distance from the previous
photo, the actual direction of travel from GPS, the compass heading, and what the tracker did.

Run from the project root (gps-denied-nav):
    python scripts\04b_turn_check.py
    python scripts\04b_turn_check.py --frames 93-99,189-197
"""

import argparse
import csv
import math
from datetime import datetime

import numpy as np

from geo_utils import ROOT, haversine_m, load_drone_csv


def bearing_deg(lat1, lon1, lat2, lon2):
    """Direction of travel, clockwise from north (same convention as Phi1)."""
    d_e = (lon2 - lon1) * math.cos(math.radians(lat1))
    d_n = lat2 - lat1
    return math.degrees(math.atan2(d_e, d_n)) % 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="03")
    ap.add_argument("--frames", default="93-99,189-197", help="frame ranges to print, e.g. 93-99,189-197")
    ap.add_argument("--tag", default="superpoint_step2_win700_fwd14.3", help="tracking run to compare")
    args = ap.parse_args()

    rows = load_drone_csv(args.loc)
    by_num = {r["num"]: r for r in rows}

    track = {}
    track_path = ROOT / "outputs" / "04_track" / args.tag / f"track_{args.loc}.csv"
    if track_path.exists():
        with open(track_path, newline="") as f:
            track = {int(r["num"]): r for r in csv.DictReader(f)}

    # Whole-flight overview of time gaps and step sizes
    times = [datetime.fromisoformat(r["date"]) for r in rows]
    gaps = np.array([(b - a).total_seconds() for a, b in zip(times[:-1], times[1:])])
    steps = np.array([haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) for a, b in zip(rows[:-1], rows[1:])])
    print(f"Time between photos: min {gaps.min():.0f} s, median {np.median(gaps):.0f} s, max {gaps.max():.0f} s")
    print(f"Distance between photos: median {np.median(steps):.0f} m, max {steps.max():.0f} m")
    big = [(rows[i + 1]["num"], gaps[i], steps[i]) for i in range(len(gaps)) if gaps[i] > 2 * np.median(gaps)]
    print(f"Gaps longer than twice the median: {len(big)}")
    for num, g, s in big[:20]:
        print(f"  before frame {num}: {g:.0f} s, {s:.0f} m")

    # Frame by frame around the chosen ranges
    wanted = []
    for part in args.frames.split(","):
        a, b = (int(x) for x in part.split("-"))
        wanted.extend(range(a, b + 1))

    print(f"\n{'frame':>5} | {'gap s':>5} | {'step m':>6} | {'GPS speed':>9} | {'travel dir':>10} | "
          f"{'Phi1':>6} | {'dir - Phi1':>10} | {'tracker':>7} | {'speed used':>10} | {'pred off m':>10}")
    print("-" * 105)
    for num in wanted:
        r = by_num.get(num)
        p = by_num.get(num - 1)
        if r is None or p is None:
            continue
        gap = (datetime.fromisoformat(r["date"]) - datetime.fromisoformat(p["date"])).total_seconds()
        step = haversine_m(p["lat"], p["lon"], r["lat"], r["lon"])
        travel = bearing_deg(p["lat"], p["lon"], r["lat"], r["lon"])
        crab = (travel - r["phi1"] + 180) % 360 - 180
        t = track.get(num, {})
        print(f"{num:>5} | {gap:>5.0f} | {step:>6.0f} | {step / gap if gap else 0:>7.1f}/s | {travel:>9.0f}° | "
              f"{r['phi1'] % 360:>5.0f}° | {crab:>+9.0f}° | {t.get('status', '-'):>7} | "
              f"{t.get('speed_mps', '-'):>10} | {t.get('pred_err_m', '-'):>10}")


if __name__ == "__main__":
    main()
