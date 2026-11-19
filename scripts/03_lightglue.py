r"""
03_lightglue.py

Same pipeline as 02_sift_baseline.py (search window, scaling to the map's metres per pixel, north-up
rotation, similarity fit, scale/rotation accept rule), with SIFT + ratio test swapped for SuperPoint
keypoints matched by LightGlue. So any difference in the results comes from the matcher.

Notes:
- Runs on CPU. The first run downloads the SuperPoint and LightGlue weights (a few MB).
- SuperPoint's weights are licensed for non-commercial use only. Fine for a portfolio project;
  --features disk is a permissively licensed alternative.
- Default search window is 1000 m (02 defaulted to 1500 m) to keep CPU memory and time reasonable.
  For a fair comparison, run SIFT with --search-m 1000 too.

Run from the project root (gps-denied-nav):
    python scripts\03_lightglue.py --max-frames 10     # quick check of speed and memory
    python scripts\03_lightglue.py --offset 2          # held-out frames, same set as the SIFT number
Outputs go to outputs\03_lightglue\<tag>\
"""

import argparse
import csv
import importlib
import math
import time

import cv2
import numpy as np
import torch
from lightglue import DISK, LightGlue, SuperPoint
from lightglue.utils import rbd

from geo_utils import ROOT, DATA, haversine_m, load_drone_csv, load_metric_map, rotate_keep_all

# Reuse the CLAHE preprocessing and match drawing from the SIFT script, so both are identical
_sift = importlib.import_module("02_sift_baseline")
to_gray, save_viz, DRONE_FLAGS = _sift.to_gray, _sift.save_viz, _sift.DRONE_FLAGS

FEATURES = {"superpoint": SuperPoint, "disk": DISK}


def extract(extractor, gray):
    img = torch.from_numpy(gray).float()[None] / 255.0     # (1, H, W) in [0, 1]
    return extractor.extract(img, resize=None)             # resize=None: keep our metres-per-pixel scale


def keep_inside(feats, mask):
    """Drop keypoints that fall on the black border left by the rotation."""
    kp = feats["keypoints"][0]
    x = kp[:, 0].round().long().clamp(0, mask.shape[1] - 1)
    y = kp[:, 1].round().long().clamp(0, mask.shape[0] - 1)
    keep = torch.from_numpy(mask)[y, x] > 0
    out = dict(feats)
    for k in ("keypoints", "keypoint_scores", "descriptors"):
        if k in out:
            out[k] = out[k][:, keep]
    return out


def match(drone_gray, drone_mask, sat_gray, ext_d, ext_s, matcher):
    res = {"kp_d": [], "kp_s": [], "good": [], "inliers": [], "H": None}
    f0 = keep_inside(extract(ext_d, drone_gray), drone_mask)
    f1 = extract(ext_s, sat_gray)
    if f0["keypoints"].shape[1] < 4 or f1["keypoints"].shape[1] < 4:
        return res

    out = rbd(matcher({"image0": f0, "image1": f1}))
    m = out["matches"]                                      # (K, 2) index pairs
    if len(m) < 4:
        return res
    p0 = rbd(f0)["keypoints"][m[:, 0]].numpy()
    p1 = rbd(f1)["keypoints"][m[:, 1]].numpy()

    # cv2 objects so the SIFT drawing code works unchanged
    res["kp_d"] = [cv2.KeyPoint(float(x), float(y), 1) for x, y in p0]
    res["kp_s"] = [cv2.KeyPoint(float(x), float(y), 1) for x, y in p1]
    res["good"] = [cv2.DMatch(i, i, 0) for i in range(len(p0))]

    A, inl = cv2.estimateAffinePartial2D(p0.reshape(-1, 1, 2), p1.reshape(-1, 1, 2),
                                         method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if A is not None:
        res["H"] = np.vstack([A, [0.0, 0.0, 1.0]])
        res["inliers"] = [g for g, keep in zip(res["good"], inl.ravel()) if keep]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="03")
    ap.add_argument("--step", type=int, default=5, help="use every Nth frame (1 = all)")
    ap.add_argument("--offset", type=int, default=0,
                    help="start at this frame index; offset 0 was used for tuning, use 1-4 for held-out frames")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many (0 = no limit)")
    ap.add_argument("--search-m", type=float, default=1000, help="side of the search window in metres")
    ap.add_argument("--prior-m", type=float, default=200, help="max offset of the window centre from GPS")
    ap.add_argument("--sat-mpp", type=float, default=0.5, help="metres per pixel used for matching")
    ap.add_argument("--drone-reduce", type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument("--hfov-deg", type=float, default=49.2, help="camera horizontal field of view")
    ap.add_argument("--features", default="superpoint", choices=list(FEATURES))
    ap.add_argument("--max-kp-drone", type=int, default=2048)
    ap.add_argument("--max-kp-sat", type=int, default=4096)
    ap.add_argument("--max-rot", type=float, default=30)
    ap.add_argument("--min-inliers", type=int, default=4)
    ap.add_argument("--viz-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or (f"{args.features}_mpp{args.sat_mpp:g}_s{args.search_m:g}"
                       + (f"_off{args.offset}" if args.offset else ""))
    out = ROOT / "outputs" / "03_lightglue" / tag
    (out / "viz").mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    Feat = FEATURES[args.features]
    ext_d = Feat(max_num_keypoints=args.max_kp_drone).eval()
    ext_s = Feat(max_num_keypoints=args.max_kp_sat).eval()
    matcher = LightGlue(features=args.features).eval()

    sat_bgr, geo, region = load_metric_map(args.loc, args.sat_mpp)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sat_gray = to_gray(sat_bgr, clahe)
    rng = np.random.default_rng(args.seed)   # same seed as 02, so the search windows are the same

    rows = load_drone_csv(args.loc)[args.offset::args.step]
    if args.max_frames:
        rows = rows[:args.max_frames]
    print(f"[{tag}] Location {args.loc} ({region}): {len(rows)} frames | "
          f"search {args.search_m:.0f} m, prior {args.prior_m:.0f} m | CPU threads {torch.get_num_threads()}")

    records = []
    for i, r in enumerate(rows):
        t0 = time.time()
        gx, gy = geo.px(r["lat"], r["lon"])

        rad = args.prior_m * math.sqrt(rng.random())
        th = 2 * math.pi * rng.random()
        cx = gx + rad * math.cos(th) / geo.mpp_x
        cy = gy - rad * math.sin(th) / geo.mpp_y
        hx, hy = args.search_m / 2 / geo.mpp_x, args.search_m / 2 / geo.mpp_y
        x0, y0 = int(max(0, cx - hx)), int(max(0, cy - hy))
        x1, y1 = int(min(geo.W, cx + hx)), int(min(geo.H, cy + hy))

        drone = cv2.imread(str(DATA / args.loc / "drone" / r["filename"]), DRONE_FLAGS[args.drone_reduce])
        if drone is None:
            print(f"  could not read {r['filename']}, skipping")
            continue
        gsd = 2 * r["height"] * math.tan(math.radians(args.hfov_deg) / 2) / drone.shape[1]
        f = gsd / geo.mpp_x
        drone = cv2.resize(drone, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)

        drone_up, M = rotate_keep_all(drone, -r["phi1"], return_matrix=True)
        mask = rotate_keep_all(np.full(drone.shape[:2], 255, np.uint8), -r["phi1"])
        mask = cv2.erode(mask, np.ones((15, 15), np.uint8))

        res = match(to_gray(drone_up, clahe), mask, sat_gray[y0:y1, x0:x1], ext_d, ext_s, matcher)

        rec = {"num": r["num"], "filename": r["filename"], "gt_lat": r["lat"], "gt_lon": r["lon"],
               "est_lat": "", "est_lon": "", "err_m": "", "n_good": len(res["good"]),
               "n_inliers": len(res["inliers"]), "scale": "", "rot_deg": "", "footprint_m": "",
               "status": "fail"}
        est_xy, footprint = None, None

        if res["H"] is not None:
            H = res["H"]
            A = H[:2, :2]
            scale = math.sqrt(abs(np.linalg.det(A)))
            rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
            c = H @ np.array([drone_up.shape[1] / 2, drone_up.shape[0] / 2, 1.0])
            u, v = c[0], c[1]
            est_lat, est_lon = geo.latlon(x0 + u, y0 + v)
            err = haversine_m(r["lat"], r["lon"], est_lat, est_lon)

            h0, w0 = drone.shape[:2]
            corners = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)
            footprint = cv2.perspectiveTransform(cv2.transform(corners, M).astype(np.float32), H).reshape(-1, 2)
            est_xy = (u, v)

            ok = (len(res["inliers"]) >= args.min_inliers and 0.75 < scale < 1.33
                  and abs(rot) < args.max_rot)
            rec.update({"est_lat": round(est_lat, 7), "est_lon": round(est_lon, 7), "err_m": round(err, 1),
                        "scale": round(scale, 3), "rot_deg": round(rot, 1),
                        "footprint_m": round(w0 * scale * geo.mpp_x, 0),
                        "status": "ok" if ok else "fail"})

        records.append(rec)
        err_txt = f"{rec['err_m']} m" if rec["err_m"] != "" else "-"
        print(f"[{i + 1}/{len(rows)}] {r['filename']}: {rec['status']:4s} | matches {rec['n_good']:4d} "
              f"| inliers {rec['n_inliers']:4d} | error {err_txt} | {time.time() - t0:.1f}s")

        if i % args.viz_every == 0:
            save_viz(out / "viz" / f"{args.loc}_{r['num']:04d}.jpg", drone_up, sat_bgr[y0:y1, x0:x1],
                     res, (gx - x0, gy - y0), est_xy, footprint)

    csv_path = out / f"results_{args.loc}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    n = len(records)
    ok = [r for r in records if r["status"] == "ok"]
    errs = np.array([r["err_m"] for r in ok], dtype=float)
    all_errs = np.array([r["err_m"] for r in records if r["err_m"] != ""], dtype=float)
    print(f"\n===== LightGlue [{tag}], location {args.loc}, {n} frames =====")
    print(f"Accepted: {len(ok)}/{n} ({100 * len(ok) / n:.1f}%)")
    for t in (25, 50, 100):
        print(f"Accepted and within {t:3d} m: {int((errs <= t).sum())}/{n} ({100 * (errs <= t).sum() / n:.1f}%)")
    print(f"Within 25 m before the accept check: {int((all_errs <= 25).sum())}/{n}")
    if len(ok):
        print(f"Median error of accepted frames: {np.median(errs):.1f} m")
        print(f"Accepted but more than 100 m off (confident and wrong): {int((errs > 100).sum())}")
        print(f"Median inliers: {np.median([r['n_inliers'] for r in ok]):.0f} | "
              f"median leftover rotation: {np.median([abs(r['rot_deg']) for r in ok]):.1f} deg | "
              f"median footprint width: {np.median([r['footprint_m'] for r in ok]):.0f} m")
    print(f"Saved {csv_path.relative_to(ROOT)} and match images in {(out / 'viz').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
