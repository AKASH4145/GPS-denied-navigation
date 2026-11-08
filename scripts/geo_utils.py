"""
geo_utils.py

Shared helpers for the UAV-VisLoc scripts: loading the CSVs, converting between lat/lon and
map pixels, rotating images, and building a cached satellite map with square pixels in metres.
"""

import csv
import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.errors
from rasterio.enums import Resampling

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"   # inside data/, so it's gitignored too

EARTH_R = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lat/lon points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def load_map_bounds(loc):
    # The bounds file in the example subset has a space in its name, so match it loosely.
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


class MapGeo:
    """Linear lat/lon <-> pixel mapping between the map corners.
    Exact for these maps, because the tif is EPSG:4326 (every pixel is the same size in degrees)."""

    def __init__(self, bounds, width, height):
        self.b, self.W, self.H = bounds, width, height
        b = bounds
        mid_lat = (b["LT_lat_map"] + b["RB_lat_map"]) / 2
        self.mpp_x = haversine_m(mid_lat, b["LT_lon_map"], mid_lat, b["RB_lon_map"]) / width
        self.mpp_y = haversine_m(b["LT_lat_map"], b["LT_lon_map"], b["RB_lat_map"], b["LT_lon_map"]) / height

    def px(self, lat, lon):
        b = self.b
        x = (lon - b["LT_lon_map"]) / (b["RB_lon_map"] - b["LT_lon_map"]) * self.W
        y = (b["LT_lat_map"] - lat) / (b["LT_lat_map"] - b["RB_lat_map"]) * self.H
        return x, y

    def latlon(self, x, y):
        b = self.b
        lon = b["LT_lon_map"] + x / self.W * (b["RB_lon_map"] - b["LT_lon_map"])
        lat = b["LT_lat_map"] - y / self.H * (b["LT_lat_map"] - b["RB_lat_map"])
        return lat, lon


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


def rotate_keep_all(img, angle_deg, return_matrix=False):
    """Rotate counter-clockwise by angle_deg without cropping the corners.
    The centre of the input stays at the centre of the output."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, M, (nw, nh))
    return (out, M) if return_matrix else out


def load_metric_map(loc, mpp=1.0):
    """Satellite map resampled so each pixel is mpp x mpp metres on the ground.
    The original tif has pixels about 0.25 m wide and 0.30 m tall, so this also removes the
    east-west stretch. Built once from the tif (a minute or two) and cached as a PNG."""
    bounds, region = load_map_bounds(loc)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"satellite{loc}_{mpp:g}m.png"

    if cache.exists():
        img = cv2.imread(str(cache))
    else:
        tif = DATA / loc / f"satellite{loc}.tif"
        with rasterio.open(tif) as src:
            native = MapGeo(bounds, src.width, src.height)
            out_w = round(native.W * native.mpp_x / mpp)
            out_h = round(native.H * native.mpp_y / mpp)
            print(f"Building {cache.name} ({out_w} x {out_h} px). One-time step, takes a minute or two...")
            n = min(src.count, 3)
            arr = src.read(indexes=list(range(1, n + 1)), out_shape=(n, out_h, out_w),
                           resampling=Resampling.average)
        img = to_bgr(arr)
        cv2.imwrite(str(cache), img)

    geo = MapGeo(bounds, img.shape[1], img.shape[0])
    return img, geo, region
