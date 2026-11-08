"""
02_sift_baseline.py

Baseline visual localisation: find each drone photo on the satellite map with SIFT + RANSAC,
read the lat/lon off the map, and compare it with the GPS label (ground truth).

Assumptions (state these in the README too):
- Heading is known (drone compass, Phi1 in the CSV), so each drone image is rotated north-up first.
- A rough position prior is known: the search window is centred on a point up to --prior-m metres
  from the true GPS position in a random direction, not on the GPS itself.
- The camera points roughly straight down, so the image centre is taken as the drone's position.

Run from the project root (gps-denied-nav):
    python scripts\02_sift_baseline.py                 # every 5th frame of location 03
    python scripts\02_sift_baseline.py --step 1        # all 768 frames
Outputs go to outputs\02_sift\
"""

import argparse
import csv
import math
import time

import cv2
import numpy as np

from geo_utils import ROOT, DATA, haversine_m, load_drone_csv, load_metric_map, rotate_keep_all

OUT = ROOT / "outputs" / "02_sift"

DRONE_FLAGS = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
               4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def to_gray(img_bgr, clahe):
    # CLAHE evens out contrast, which helps with haze in the drone photos and dark satellite areas
    return clahe.apply(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))


def match(drone_gray, drone_mask, sat_gray, sift, ratio):
    kp_d, des_d = sift.detectAndCompute(drone_gray, drone_mask)
    kp_s, des_s = sift.detectAndCompute(sat_gray, None)
    res = {"kp_d": kp_d, "kp_s": kp_s, "good": [], "inliers": [], "H": None}
    if des_d is None or des_s is None or len(kp_d) < 4 or len(kp_s) < 4:
        return res

    # Lowe's ratio test: keep a match only if it's clearly better than the second-best option
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_d, des_s, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    res["good"] = good
    if len(good) < 4:
        return res

    src = np.float32([kp_d[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is not None:
        res["H"] = H
        res["inliers"] = [m for m, keep in zip(good, mask.ravel()) if keep]
    return res


def save_viz(path, drone_up, sat_crop, res, gt_xy, est_xy, footprint):
    sat = sat_crop.copy()
    if footprint is not None:
        cv2.polylines(sat, [np.int32(footprint)], True, (255, 0, 255), 3)    # matched footprint
    cv2.circle(sat, (int(gt_xy[0]), int(gt_xy[1])), 14, (0, 255, 0), 3)      # GPS label: green
    if est_xy is not None:
        cv2.drawMarker(sat, (int(est_xy[0]), int(est_xy[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 36, 3)                               # estimate: red
    shown = res["inliers"] if res["inliers"] else res["good"][:50]
    img = cv2.drawMatches(drone_up, res["kp_d"], sat, res["kp_s"], shown, None,
                          matchColor=(0, 255, 255),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    scale = 1800 / img.shape[1]
    img = cv2.resize(img, (1800, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="03")
    ap.add_argument("--step", type=int, default=5, help="use every Nth frame (1 = all)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many (0 = no limit)")
    ap.add_argument("--search-m", type=float, default=1500, help="side of the search window in metres")
    ap.add_argument("--prior-m", type=float, default=200, help="max offset of the window centre from GPS")
    ap.add_argument("--sat-mpp", type=float, default=1.0, help="satellite resolution used for matching")
    ap.add_argument("--drone-reduce", type=int, default=4, choices=[1, 2, 4, 8],
                    help="load drone images at 1/N size")
    ap.add_argument("--nfeatures", type=int, default=5000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--viz-every", type=int, default=10, help="save a match image every N frames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "viz").mkdir(exist_ok=True)

    sat_bgr, geo, region = load_metric_map(args.loc, args.sat_mpp)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sat_gray = to_gray(sat_bgr, clahe)
    sift = cv2.SIFT_create(nfeatures=args.nfeatures)
    rng = np.random.default_rng(args.seed)

    rows = load_drone_csv(args.loc)[::args.step]
    if args.max_frames:
        rows = rows[:args.max_frames]
    print(f"Location {args.loc} ({region}): {len(rows)} frames | map {geo.W} x {geo.H} px "
          f"at {geo.mpp_x:.2f} x {geo.mpp_y:.2f} m/px | search {args.search_m:.0f} m, prior {args.prior_m:.0f} m")

    records = []
    for i, r in enumerate(rows):
        t0 = time.time()
        gx, gy = geo.px(r["lat"], r["lon"])

        # Search window centred on a point near (not at) the true position
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
        # North-up: rotate clockwise by Phi1 (cv2 angle = -Phi1). Checked by eye on frames 1 and 300.
        drone_up, M = rotate_keep_all(drone, -r["phi1"], return_matrix=True)
        mask = rotate_keep_all(np.full(drone.shape[:2], 255, np.uint8), -r["phi1"])
        mask = cv2.erode(mask, np.ones((15, 15), np.uint8))   # ignore keypoints on the black border

        res = match(to_gray(drone_up, clahe), mask, sat_gray[y0:y1, x0:x1], sift, args.ratio)

        rec = {"num": r["num"], "filename": r["filename"], "gt_lat": r["lat"], "gt_lon": r["lon"],
               "est_lat": "", "est_lon": "", "err_m": "", "n_good": len(res["good"]),
               "n_inliers": len(res["inliers"]), "scale": "", "rot_deg": "", "footprint_m": "",
               "status": "fail"}
        est_xy, footprint = None, None

        if res["H"] is not None:
            H = res["H"]
            A = H[:2, :2] / H[2, 2]
            scale = math.sqrt(abs(np.linalg.det(A)))           # drone px -> satellite px
            rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))    # leftover rotation, ~0 if heading is right
            c = H @ np.array([drone_up.shape[1] / 2, drone_up.shape[0] / 2, 1.0])
            u, v = c[0] / c[2], c[1] / c[2]
            est_lat, est_lon = geo.latlon(x0 + u, y0 + v)
            err = haversine_m(r["lat"], r["lon"], est_lat, est_lon)

            h0, w0 = drone.shape[:2]
            corners = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)
            footprint = cv2.perspectiveTransform(cv2.transform(corners, M).astype(np.float32), H).reshape(-1, 2)
            est_xy = (u, v)

            ok = len(res["inliers"]) >= args.min_inliers and 0.1 < scale < 10
            rec.update({"est_lat": round(est_lat, 7), "est_lon": round(est_lon, 7), "err_m": round(err, 1),
                        "scale": round(scale, 3), "rot_deg": round(rot, 1),
                        "footprint_m": round(w0 * scale * geo.mpp_x, 0),
                        "status": "ok" if ok else "fail"})

        records.append(rec)
        err_txt = f"{rec['err_m']} m" if rec["err_m"] != "" else "-"
        print(f"[{i + 1}/{len(rows)}] {r['filename']}: {rec['status']:4s} | inliers {rec['n_inliers']:4d} "
              f"| error {err_txt} | {time.time() - t0:.1f}s")

        if i % args.viz_every == 0:
            gt_local = (gx - x0, gy - y0)
            save_viz(OUT / "viz" / f"{args.loc}_{r['num']:04d}.jpg", drone_up, sat_bgr[y0:y1, x0:x1],
                     res, gt_local, est_xy, footprint)

    # ---------- results ----------
    csv_path = OUT / f"results_{args.loc}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    n = len(records)
    ok = [r for r in records if r["status"] == "ok"]
    errs = np.array([r["err_m"] for r in ok], dtype=float)
    print(f"\n===== SIFT baseline, location {args.loc}, {n} frames =====")
    print(f"Matched (>= {args.min_inliers} inliers): {len(ok)}/{n} ({100 * len(ok) / n:.1f}%)")
    for t in (25, 50, 100):
        print(f"Within {t:3d} m: {int((errs <= t).sum())}/{n} ({100 * (errs <= t).sum() / n:.1f}%)")
    if len(ok):
        print(f"Median error of matched frames: {np.median(errs):.1f} m")
        print(f"Matched but more than 100 m off (confident and wrong): {int((errs > 100).sum())}")
        print(f"Median inliers: {np.median([r['n_inliers'] for r in ok]):.0f} | "
              f"median leftover rotation: {np.median([abs(r['rot_deg']) for r in ok]):.1f} deg | "
              f"median footprint width: {np.median([r['footprint_m'] for r in ok]):.0f} m")
    print(f"Saved {csv_path.relative_to(ROOT)} and match images in {(OUT / 'viz').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
