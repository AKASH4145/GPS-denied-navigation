"""
01_check_data.py

First sanity check for UAV-VisLoc. No localisation yet, just making sure the data lines up.
1. Reads the satellite map bounds and the drone CSV for one location.
2. Prints basic stats (image count, altitude, spacing between frames, map resolution).
3. Draws the drone flight path on a downscaled copy of the satellite map.
4. Saves one drone image next to the satellite crop at its GPS position, with the drone
   image also rotated by +Phi1 and -Phi1, so we can see by eye which rotation makes it
   match the north-up satellite map.

Run from the project root (gps-denied-nav):
    python scripts\01_check_data.py
    python scripts\01_check_data.py --loc 03 --frame 120

Outputs go to outputs\01_check\
"""

import argparse
import csv
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.errors
from rasterio.enums import Resampling
from rasterio.windows import Window

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "01_check"

EARTH_R = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lat/lon points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def load_map_bounds(loc):
    # In the example subset the file is named "satellite_ coordinates_range.csv" (with a space),
    # so match it loosely instead of hard-coding the name.
    files = sorted(DATA.glob("satellite*coordinates_range.csv"))
    if not files:
        raise FileNotFoundError(f"No satellite coordinates CSV found in {DATA}")
    with open(files[0], newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            if Path(row["mapname"]).stem == loc:
                keys = ("LT_lat_map", "LT_lon_map", "RB_lat_map", "RB_lon_map")
                return {k: float(row[k]) for k in keys}, row.get("region", "")
    raise ValueError(f"Location {loc} not found in {files[0].name}")


def load_drone_csv(loc):
    path = DATA / loc / f"{loc}.csv"
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): v.strip() for k, v in r.items()}
            rows.append({
                "num": int(r["num"]),
                "filename": r["filename"],
                "date": r["date"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "height": float(r["height"]),
                "phi1": float(r["Phi1"]),
            })
    return rows


def latlon_to_px(lat, lon, b, width, height):
    # Linear mapping between the map corners. Google Earth maps are Web Mercator, but over a
    # few km the difference from linear in latitude is well under 1 m, so this is fine here.
    x = (lon - b["LT_lon_map"]) / (b["RB_lon_map"] - b["LT_lon_map"]) * width
    y = (b["LT_lat_map"] - lat) / (b["LT_lat_map"] - b["RB_lat_map"]) * height
    return x, y


def to_bgr(arr):
    """rasterio gives (bands, h, w) in RGB order; OpenCV wants (h, w, 3) in BGR."""
    arr = arr[:3]
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    img = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def rotate_keep_all(img, angle_deg):
    """Rotate counter-clockwise by angle_deg without cropping the corners."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh))


def resize_h(img, h):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_AREA)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="03", help="location folder, e.g. 03")
    ap.add_argument("--frame", type=int, default=1, help="'num' of the drone image for the pair check")
    ap.add_argument("--crop-m", type=float, default=800, help="side of the satellite crop in metres")
    ap.add_argument("--thumb", type=int, default=2500, help="longest side of the track overview in px")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    b, region = load_map_bounds(args.loc)
    rows = load_drone_csv(args.loc)
    drone_dir = DATA / args.loc / "drone"
    tif_path = DATA / args.loc / f"satellite{args.loc}.tif"

    # ---------- drone CSV stats ----------
    print(f"Location {args.loc} ({region})")
    print(f"  Map bounds: lat {b['RB_lat_map']:.6f} to {b['LT_lat_map']:.6f}, "
          f"lon {b['LT_lon_map']:.6f} to {b['RB_lon_map']:.6f}")

    missing = [r["filename"] for r in rows if not (drone_dir / r["filename"]).exists()]
    inside = [r for r in rows
              if b["RB_lat_map"] <= r["lat"] <= b["LT_lat_map"]
              and b["LT_lon_map"] <= r["lon"] <= b["RB_lon_map"]]
    steps = [haversine_m(a["lat"], a["lon"], c["lat"], c["lon"]) for a, c in zip(rows[:-1], rows[1:])]
    heights = [r["height"] for r in rows]

    print(f"  Drone images in CSV: {len(rows)} | missing on disk: {len(missing)} | inside map: {len(inside)}")
    print(f"  Dates: {rows[0]['date']} to {rows[-1]['date']}")
    print(f"  Height: min {min(heights):.1f} m, max {max(heights):.1f} m, mean {np.mean(heights):.1f} m")
    if steps:
        print(f"  Distance between frames: median {np.median(steps):.1f} m, max {max(steps):.1f} m, "
              f"total path {sum(steps) / 1000:.2f} km")

    # ---------- satellite map ----------
    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        n_bands = min(src.count, 3)
        bands = list(range(1, n_bands + 1))
        map_w_m = haversine_m(b["LT_lat_map"], b["LT_lon_map"], b["LT_lat_map"], b["RB_lon_map"])
        map_h_m = haversine_m(b["LT_lat_map"], b["LT_lon_map"], b["RB_lat_map"], b["LT_lon_map"])
        mpp_x, mpp_y = map_w_m / W, map_h_m / H

        print(f"\nSatellite map {tif_path.name}")
        print(f"  Size: {W} x {H} px | bands: {src.count} | dtype: {src.dtypes[0]}")
        print(f"  CRS inside the tif: {src.crs} | internal overviews: {src.overviews(1)}")
        print(f"  Ground size from CSV bounds: {map_w_m / 1000:.2f} km x {map_h_m / 1000:.2f} km")
        print(f"  Metres per pixel: x {mpp_x:.3f}, y {mpp_y:.3f}  (x is smaller by cos(lat) when the tif is EPSG:4326)")

        # Downscaled overview with the flight path drawn on it
        scale = args.thumb / max(W, H)
        tw, th = int(W * scale), int(H * scale)
        print("  Reading downscaled map, this can take a minute or two...")
        thumb = to_bgr(src.read(indexes=bands, out_shape=(n_bands, th, tw), resampling=Resampling.average))

        pts = np.array([latlon_to_px(r["lat"], r["lon"], b, tw, th) for r in rows], dtype=np.int32)
        cv2.polylines(thumb, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 2)
        for r, (x, y) in zip(rows, pts):
            cv2.circle(thumb, (int(x), int(y)), 3, (0, 200, 255), -1)
            if r["num"] % 50 == 0:
                cv2.putText(thumb, str(r["num"]), (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(thumb, tuple(int(v) for v in pts[0]), 10, (0, 255, 0), -1)   # start: green
        cv2.circle(thumb, tuple(int(v) for v in pts[-1]), 10, (0, 0, 255), -1)  # end: red
        track_path = OUT / f"{args.loc}_track.jpg"
        cv2.imwrite(str(track_path), thumb)
        print(f"  Saved {track_path.relative_to(ROOT)}")

        # Full-resolution satellite crop around one drone image's GPS position
        sel = next((r for r in rows if r["num"] == args.frame), None)
        if sel is None:
            raise ValueError(f"Frame {args.frame} not in {args.loc}.csv")
        cx, cy = latlon_to_px(sel["lat"], sel["lon"], b, W, H)
        half = int(args.crop_m / ((mpp_x + mpp_y) / 2) / 2)
        col0, row0 = max(0, int(cx) - half), max(0, int(cy) - half)
        col1, row1 = min(W, int(cx) + half), min(H, int(cy) + half)
        crop = to_bgr(src.read(indexes=bands, window=Window(col0, row0, col1 - col0, row1 - row0)))
        cv2.drawMarker(crop, (int(cx - col0), int(cy - row0)), (0, 0, 255), cv2.MARKER_CROSS, 60, 4)

    # ---------- drone vs satellite pair ----------
    drone = cv2.imread(str(drone_dir / sel["filename"]), cv2.IMREAD_REDUCED_COLOR_4)
    if drone is None:
        raise FileNotFoundError(drone_dir / sel["filename"])

    panel_h = 600
    panels = [
        label(resize_h(crop, panel_h), f"satellite {args.crop_m:.0f} m, north up"),
        label(resize_h(drone, panel_h), f"drone {sel['filename']} raw"),
        label(resize_h(rotate_keep_all(drone, sel["phi1"]), panel_h), f"rotated +Phi1 ({sel['phi1']:.1f})"),
        label(resize_h(rotate_keep_all(drone, -sel["phi1"]), panel_h), "rotated -Phi1"),
    ]
    pair_path = OUT / f"{args.loc}_pair_{sel['num']:04d}.jpg"
    cv2.imwrite(str(pair_path), cv2.hconcat(panels))
    print(f"\nFrame {sel['num']}: lat {sel['lat']:.6f}, lon {sel['lon']:.6f}, "
          f"height {sel['height']:.1f} m, Phi1 {sel['phi1']:.1f} deg")
    print(f"  Saved {pair_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
