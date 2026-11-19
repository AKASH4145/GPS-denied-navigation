r"""
02b_accept_sweep.py

Re-scores a results CSV from 02_sift_baseline.py with different accept rules, without re-running
any matching. Shows the trade-off: a looser rule accepts more correct frames, but also more wrong ones.

Run from the project root (gps-denied-nav):
    python scripts\02b_accept_sweep.py
    python scripts\02b_accept_sweep.py --tag v1
"""

import argparse
import csv

from geo_utils import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="02_sift", help="folder in outputs, e.g. 02_sift or 03_lightglue")
    ap.add_argument("--tag", default="mpp0.5_similarity_scaled", help="run subfolder inside the stage folder")
    ap.add_argument("--loc", default="03")
    ap.add_argument("--scale-lo", type=float, default=0.75)
    ap.add_argument("--scale-hi", type=float, default=1.33)
    ap.add_argument("--max-rot", type=float, default=30)
    ap.add_argument("--good-m", type=float, default=25, help="error counted as correct")
    ap.add_argument("--bad-m", type=float, default=100, help="error counted as wrong")
    args = ap.parse_args()

    path = ROOT / "outputs" / args.stage / args.tag / f"results_{args.loc}.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)

    fitted = []
    for r in rows:
        if r["err_m"] == "":
            continue
        fitted.append({"err": float(r["err_m"]), "inl": int(r["n_inliers"]),
                       "scale": float(r["scale"]), "rot": float(r["rot_deg"])})

    def geo_ok(r):
        return args.scale_lo < r["scale"] < args.scale_hi and abs(r["rot"]) < args.max_rot

    reachable = sum(r["err"] <= args.good_m for r in fitted)
    print(f"{path.relative_to(ROOT)}: {n} frames, {reachable} within {args.good_m:.0f} m before any accept rule")
    print(f"Accept rule: inliers >= k AND scale in ({args.scale_lo}, {args.scale_hi}) AND |rotation| < {args.max_rot} deg\n")
    print(f"{'k':>3} | {'accepted':>8} | {'correct':>7} | {'wrong':>5} | {'precision':>9} | {'recall':>6} | "
          f"{'wrong without scale/rot check':>29}")
    print("-" * 85)
    for k in range(2, 21):
        acc = [r for r in fitted if r["inl"] >= k and geo_ok(r)]
        good = sum(r["err"] <= args.good_m for r in acc)
        bad = sum(r["err"] > args.bad_m for r in acc)
        bad_no_geo = sum(r["err"] > args.bad_m for r in fitted if r["inl"] >= k)
        prec = 100 * good / len(acc) if acc else 0.0
        print(f"{k:>3} | {len(acc):>8} | {good:>7} | {bad:>5} | {prec:>8.1f}% | {100 * good / n:>5.1f}% | "
              f"{bad_no_geo:>29}")
    print(f"\ncorrect = error <= {args.good_m:.0f} m, wrong = error > {args.bad_m:.0f} m, "
          f"precision = correct / accepted, recall = correct / all {n} frames")


if __name__ == "__main__":
    main()
