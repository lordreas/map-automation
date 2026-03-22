from gpx2pdf import build_elevation_map
import argparse
import numpy as np
import heapq
import os
import tempfile
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import get_context
import logging
import time
import struct
from typing import Callable, Optional

import matplotlib
matplotlib.use("TkAgg")  # show interactive window
import matplotlib.pyplot as plt

import trimesh
from shapely.geometry import Polygon

from pyproj import CRS, Transformer
from skimage.measure import find_contours
from tqdm.auto import tqdm

try:
    from scipy.spatial import cKDTree as KDTree  # optional
except Exception:
    KDTree = None


def _ceil_div(a: float, b: float) -> int:
    return int(np.ceil(a / b))


def _mm_to_px_x(mm: float, final_width_mm: float, img_width_px: int) -> int:
    return int(round(mm * (img_width_px / float(final_width_mm))))


def _mm_to_px_y(mm: float, final_height_mm: float, img_height_px: int) -> int:
    return int(round(mm * (img_height_px / float(final_height_mm))))


def compute_corridors_1d(total_px: int, starts_px: list[int], overlap_px: int, axis: str) -> list[np.ndarray]:
    """
    Returns one bool mask per corridor for a 1D tiling.
    - For columns (axis='x'): mask shape (H, W) is created later; here return (W,) masks.
    - For rows    (axis='y'): return (H,) masks.
    Corridor i is the overlap region between tile i-1 and tile i: [start_i, start_i + overlap)
    """
    if overlap_px <= 0:
        return []

    corridors = []
    for i in range(1, len(starts_px)):
        c0 = int(starts_px[i])
        c1 = int(min(total_px, c0 + overlap_px))
        if c1 <= c0:
            continue
        m = np.zeros((total_px,), dtype=bool)
        m[c0:c1] = True
        corridors.append(m)
    return corridors


def plan_tile_starts_mm(total_mm: float, bed_mm: float, overlap_mm: float) -> list[float]:
    """
    Choose the minimum tile count needed to span `total_mm`, then center the
    repeating bed/overlap pattern so the outer visible strips are symmetric.

    The first start may be slightly negative and the last tile may end slightly
    past `total_mm`; downstream code clips each tile to the real map extent.
    """
    if total_mm <= 0:
        raise ValueError("total dimension must be > 0")
    if bed_mm <= 0:
        raise ValueError("bed dimension must be > 0")
    if overlap_mm < 0:
        raise ValueError("overlap_width_mm must be >= 0")
    if overlap_mm >= bed_mm:
        raise ValueError("overlap_width_mm must be < bed dimension")

    step = bed_mm - overlap_mm
    tile_count = max(1, int(np.ceil(max(float(total_mm) - float(overlap_mm), 0.0) / float(step))))
    coverage_mm = float(bed_mm) + float(tile_count - 1) * float(step)
    excess_mm = max(0.0, coverage_mm - float(total_mm))
    start0_mm = -0.5 * excess_mm
    return [start0_mm + float(i) * float(step) for i in range(tile_count)]


class Timer:
    def __init__(self, label: str, logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger or logging.getLogger(__name__)
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - (self.t0 or time.perf_counter())
        self.logger.info("%s took %.3fs", self.label, dt)


def parse_args():
    ap = argparse.ArgumentParser(description="Generate corridor masks for splitting an elevation map into printable tiles.")
    # bbox for build_elevation_map (same data source you already use)
    ap.add_argument("--lat-min", default=43.5, type=float)
    ap.add_argument("--lat-max", default=48.4, type=float)
    ap.add_argument("--lon-min", default=4.9, type=float)
    ap.add_argument("--lon-max", default=16.4, type=float)

    # all physical parameters in mm
    ap.add_argument("--final-width-mm",   default=1500, type=float, help="Assembled final print width (mm)")
    ap.add_argument("--final-height-mm",  default=1000,  type=float, help="Assembled final print height (mm)")
    ap.add_argument("--bed-width-mm",     default=320,  type=float, help="Printer bed width (mm)")
    ap.add_argument("--bed-height-mm",    default=320,  type=float, help="Printer bed height (mm)")
    ap.add_argument("--overlap-width-mm", default=50,   type=float, help="Overlap width (mm)")
    ap.add_argument("--elevation-interval-m", default=50.0, type=float, help="Contour interval in meters")
    ap.add_argument(
        "--elevation-fine-threshold-m",
        default=300.0,
        type=float,
        help="Use the fine contour interval up to and including this elevation (m).",
    )
    ap.add_argument(
        "--elevation-fine-interval-m",
        default=10.0,
        type=float,
        help="Fine contour interval in meters used below the threshold.",
    )
    ap.add_argument(
        "--reversal-split-mm",
        default=20.0,
        type=float,
        help="Only split a polyline after this much reverse-travel along the corridor axis is accumulated (mm).",
    )
    ap.add_argument(
        "--neighbor-radius-mm",
        default=2.0,
        type=float,
        help="If >0, add extra graph edges between nodes within this radius (mm on the printed/output map).",
    )
    ap.add_argument(
        "--skip-alpha",
        default=2.0,
        type=float,
        help="Only add a proximity edge (u,v) if shortest-path distance in the current subgraph is > alpha * direct distance.",
    )
    ap.add_argument(
        "--workers",
        default=12,
        type=int,
        help="Number of worker processes (0 => conservative default). One corridor is processed per worker.",
    )
    ap.add_argument(
        "--mesh-export-workers",
        default=1,
        type=int,
        help="Number of worker processes used for per-tile mesh export.",
    )
    ap.add_argument(
        "--print-scale",
        default=1.0,
        type=float,
        help="Uniform scale factor applied to nozzle-based resolution, fitting clearance, and final exported mesh vertices.",
    )
    ap.add_argument(
        "--export-meshes",
        nargs="*",
        default=None,
        metavar="TY_TX",
        help="Export STL meshes. With no tile ids, export all tiles. With tile ids like 02_03 02_01, export only that subset.",
    )
    ap.add_argument(
        "--mesh-out-dir",
        default="",
        type=str,
        help="Output directory for STL meshes (default: current working directory).",
    )
    ap.add_argument(
        "--bottom-thickness-mm",
        default=10,
        type=float,
        help="Extra solid thickness added below the minimum height (mm).",
    )
    ap.add_argument(
        "--desired-height-mm",
        default=50.0,
        type=float,
        help="Relief height above the bottom thickness (mm). Heights are normalized into [bottom_thickness_mm, bottom_thickness_mm + desired_height_mm] after the nonlinear mapping.",
    )
    ap.add_argument(
        "--height-exponent",
        default=0.7,
        type=float,
        help="Exponent applied to normalized heights during mesh generation. 1.0 keeps linear scaling.",
    )
    ap.add_argument(
        "--fitting-clearance",
        default=0.15,
        type=float,
        help="Total clearance gap between tiles (mm). Edges are offset by half this amount.",
    )
    ap.add_argument(
        "--nozzle-diameter-mm",
        default=0.4,
        type=float,
        help="Approximate nozzle diameter used to choose a pre-processing raster resolution (mm).",
    )
    ap.add_argument(
        "--resampling-interval-mm",
        dest="resampling_interval_mm",
        default=2.0,
        type=float,
        help="Maximum segment length used when resampling graph edges on the printed/output map (mm).",
    )
    return ap.parse_args()


def choose_local_crs(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> CRS:
    """
    Pick a local, meter-based CRS for a lon/lat bbox.
    Prefer UTM for small/medium extents in its validity range; otherwise use local AEQD.
    """
    lat0 = 0.5 * (lat_min + lat_max)
    lon0 = 0.5 * (lon_min + lon_max)
    lat_span = abs(lat_max - lat_min)
    lon_span = abs(lon_max - lon_min)

    # UTM is generally fine for typical print-sized regions and avoids "custom CRS" strings.
    utm_ok = (-80.0 <= lat0 <= 84.0) and (lon_span <= 6.0) and (lat_span <= 6.0)
    if utm_ok:
        zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
        epsg = (32600 + zone) if lat0 >= 0 else (32700 + zone)
        return CRS.from_epsg(epsg)

    # Robust fallback: local tangent plane-ish projection centered on the bbox.
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )


def project_bbox_to_local_m(lat_min: float, lat_max: float, lon_min: float, lon_max: float, crs_local: CRS):
    """
    Project bbox corners to local meters; return (minx, miny, maxx, maxy, width_m, height_m).
    """
    t = Transformer.from_crs("EPSG:4326", crs_local, always_xy=True)
    corners_lon = np.array([lon_min, lon_max, lon_max, lon_min], dtype=float)
    corners_lat = np.array([lat_min, lat_min, lat_max, lat_max], dtype=float)
    xs, ys = t.transform(corners_lon, corners_lat)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    return minx, miny, maxx, maxy, (maxx - minx), (maxy - miny)


def fit_mm_bbox_preserve_aspect_ratio(bbox_w_mm: float, bbox_h_mm: float, aspect_w_over_h: float):
    """
    Fit to a mm bounding box without stretching, given a target aspect ratio (w/h).
    Returns (effective_w_mm, effective_h_mm).
    """
    if bbox_w_mm <= 0 or bbox_h_mm <= 0:
        raise ValueError("final width/height must be > 0")
    if aspect_w_over_h <= 0:
        raise ValueError("aspect ratio must be > 0")

    bbox_ar = bbox_w_mm / float(bbox_h_mm)
    if bbox_ar >= aspect_w_over_h:
        eff_h = float(bbox_h_mm)
        eff_w = eff_h * float(aspect_w_over_h)
    else:
        eff_w = float(bbox_w_mm)
        eff_h = eff_w / float(aspect_w_over_h)
    return eff_w, eff_h


def _uniform_filter_reflect(arr: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if size <= 1:
        return arr.copy()
    pad = int(size // 2)
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=np.float32(0.0))
    integral = np.cumsum(np.cumsum(integral, axis=0, dtype=np.float32), axis=1, dtype=np.float32)
    sums = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    sums *= np.float32(1.0 / float(size * size))
    return sums.astype(np.float32, copy=False)


def _pad_for_half_sampling(arr: np.ndarray, fill_edge: bool = True) -> np.ndarray:
    arr = np.asarray(arr)
    pad_h = int(arr.shape[0] % 2)
    pad_w = int(arr.shape[1] % 2)
    if pad_h == 0 and pad_w == 0:
        return arr
    if fill_edge:
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)


def _masked_block_median(blocks: np.ndarray, valid_blocks: np.ndarray) -> np.ndarray:
    h2, _, w2, _ = blocks.shape
    flat = blocks.transpose(0, 2, 1, 3).reshape(h2, w2, 4)
    valid_flat = valid_blocks.transpose(0, 2, 1, 3).reshape(h2, w2, 4)
    counts = valid_flat.sum(axis=-1)
    safe = np.where(valid_flat, flat, np.float32(np.inf))
    safe.sort(axis=-1)
    lo_idx = np.clip((counts - 1) // 2, 0, 3)[..., None]
    hi_idx = np.clip(counts // 2, 0, 3)[..., None]
    lo = np.take_along_axis(safe, lo_idx, axis=-1)[..., 0]
    hi = np.take_along_axis(safe, hi_idx, axis=-1)[..., 0]
    med = np.float32(0.5) * (lo + hi)
    med[counts == 0] = np.nan
    return med.astype(np.float32, copy=False)


def _clamp_nonnegative(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    neg_mask = np.isfinite(arr) & (arr < 0)
    if not np.any(neg_mask):
        return arr
    if not arr.flags.writeable:
        arr = np.array(arr, copy=True)
    np.maximum(arr, 0, out=arr)
    return arr


def _replace_hgt_voids_with_nan(arr: np.ndarray, void_threshold: float = -32000.0) -> np.ndarray:
    """
    Convert HGT/NASADEM-style void sentinels to NaN before downsampling.

    The merge step can return integer rasters with `-32768` void samples on
    outer borders or tile seams. Those should remain invalid during the
    min/max pyramid instead of being clamped to sea level first.
    """
    arr = np.asarray(arr)
    void_mask = np.isfinite(arr) & (arr <= float(void_threshold))
    if not np.any(void_mask):
        return arr
    out = np.asarray(arr, dtype=np.float32).copy()
    out[void_mask] = np.nan
    return out


def _build_contour_levels(
    crop_min: float,
    crop_max: float,
    *,
    fine_threshold_m: float,
    fine_interval_m: float,
    coarse_interval_m: float,
) -> np.ndarray:
    eps = 1e-9
    levels: list[np.ndarray] = []

    if crop_min <= fine_threshold_m + eps:
        fine_stop = min(crop_max, fine_threshold_m)
        fine_start = np.floor(crop_min / fine_interval_m) * fine_interval_m
        fine_levels = np.arange(fine_start, fine_stop + fine_interval_m, fine_interval_m, dtype=float)
        fine_levels = fine_levels[fine_levels <= fine_threshold_m + eps]
        if fine_levels.size:
            levels.append(fine_levels)

    if crop_max > fine_threshold_m + eps:
        coarse_start = np.ceil((fine_threshold_m + eps) / coarse_interval_m) * coarse_interval_m
        coarse_start = max(coarse_start, np.floor(crop_min / coarse_interval_m) * coarse_interval_m)
        coarse_levels = np.arange(coarse_start, crop_max + coarse_interval_m, coarse_interval_m, dtype=float)
        coarse_levels = coarse_levels[coarse_levels > fine_threshold_m + eps]
        if coarse_levels.size:
            levels.append(coarse_levels)

    if not levels:
        return np.array([], dtype=float)
    return np.unique(np.concatenate(levels))


def _premerge_output_shape_for_nozzle(
    eff_width_mm: float,
    eff_height_mm: float,
    nozzle_diameter_mm: float,
    oversample_factor: float = 2.0,
) -> tuple[int, int]:
    target_mm_per_px = float(nozzle_diameter_mm) / 2.0
    if target_mm_per_px <= 0.0:
        raise ValueError("--nozzle-diameter-mm must be > 0")
    premerge_mm_per_px = target_mm_per_px / max(float(oversample_factor), 1.0)
    out_w = max(1, int(np.ceil(float(eff_width_mm) / premerge_mm_per_px)))
    out_h = max(1, int(np.ceil(float(eff_height_mm) / premerge_mm_per_px)))
    return out_h, out_w


def downsample_minmax_half(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    valid = np.isfinite(arr)
    if not np.any(valid):
        new_h = max(1, (arr.shape[0] + 1) // 2)
        new_w = max(1, (arr.shape[1] + 1) // 2)
        return np.full((new_h, new_w), np.nan, dtype=np.float32)

    arr_pad = np.asarray(_pad_for_half_sampling(arr, fill_edge=True), dtype=np.float32)
    valid_pad = _pad_for_half_sampling(valid, fill_edge=False)
    del arr

    if not np.all(valid_pad):
        global_fill = np.float32(np.nanmedian(arr_pad[valid_pad]))
        valid_f = valid_pad.astype(np.float32, copy=False)
        mean_vals = _uniform_filter_reflect(np.where(valid_pad, arr_pad, np.float32(0.0)), size=25)
        mean_mask = _uniform_filter_reflect(valid_f, size=25)
        np.divide(
            mean_vals,
            np.maximum(mean_mask, np.float32(1e-12)),
            out=mean_vals,
            where=mean_mask > np.float32(1e-12),
        )
        mean_vals[mean_mask <= np.float32(1e-12)] = global_fill
        np.copyto(arr_pad, mean_vals, where=~valid_pad)
        del mean_vals, mean_mask, valid_f

    smooth = _uniform_filter_reflect(arr_pad, size=25)
    lap_large = smooth - arr_pad

    h2 = arr_pad.shape[0] // 2
    w2 = arr_pad.shape[1] // 2
    blocks = arr_pad.reshape(h2, 2, w2, 2)
    valid_blocks = valid_pad.reshape(h2, 2, w2, 2)
    any_valid = valid_blocks.any(axis=(1, 3))

    if np.all(valid_blocks):
        bmin = blocks.min(axis=(1, 3))
        bmax = blocks.max(axis=(1, 3))
    else:
        bmin = np.min(np.where(valid_blocks, blocks, np.float32(np.inf)), axis=(1, 3))
        bmax = np.max(np.where(valid_blocks, blocks, np.float32(-np.inf)), axis=(1, 3))
    bmed = _masked_block_median(blocks, valid_blocks)
    bmin[~any_valid] = np.nan
    bmax[~any_valid] = np.nan

    lap_blocks = lap_large.reshape(h2, 2, w2, 2).mean(axis=(1, 3))
    abs_lap = np.abs(lap_blocks[any_valid])
    threshold = float(np.percentile(abs_lap, 25)) if abs_lap.size else 0.0

    out = np.where(
        lap_blocks > threshold,
        bmin,
        np.where(lap_blocks < -threshold, bmax, bmed),
    ).astype(np.float32, copy=False)
    out[~any_valid] = np.nan
    return out


def downsample_elevation_map_for_nozzle(
    elev_map: np.ndarray,
    *,
    eff_width_mm: float,
    eff_height_mm: float,
    nozzle_diameter_mm: float,
) -> tuple[np.ndarray, int]:
    elev_map = np.asarray(elev_map, dtype=np.float32)
    if nozzle_diameter_mm <= 0:
        raise ValueError("--nozzle-diameter-mm must be > 0")

    target_mm_per_px = float(nozzle_diameter_mm) / 2.0
    out = elev_map
    steps = 0

    while min(out.shape) >= 2:
        next_h = max(1, (out.shape[0] + 1) // 2)
        next_w = max(1, (out.shape[1] + 1) // 2)
        next_mm_per_px_x = float(eff_width_mm) / float(next_w)
        next_mm_per_px_y = float(eff_height_mm) / float(next_h)
        if next_mm_per_px_x > target_mm_per_px or next_mm_per_px_y > target_mm_per_px:
            break
        out = downsample_minmax_half(out)
        steps += 1

    return out, steps


@dataclass
class RasterContext:
    elev_map: np.ndarray
    img_h: int
    img_w: int
    bbox_w_m: float
    bbox_h_m: float
    eff_width_mm: float
    eff_height_mm: float

    @property
    def mm_per_px_x(self) -> float:
        return float(self.eff_width_mm) / float(max(1, self.img_w))

    @property
    def mm_per_px_y(self) -> float:
        return float(self.eff_height_mm) / float(max(1, self.img_h))

    @property
    def mppx(self) -> float:
        return float(self.bbox_w_m) / float(max(1, self.img_w - 1))

    @property
    def mppy(self) -> float:
        return float(self.bbox_h_m) / float(max(1, self.img_h - 1))

    @property
    def extent_m(self) -> tuple[float, float, float, float]:
        return (0.0, float(self.bbox_w_m), float(self.bbox_h_m), 0.0)


@dataclass
class TilingPlan:
    corridors: list[dict]
    col_starts_mm: list[float]
    row_starts_mm: list[float]
    col_starts_px: list[int]
    row_starts_px: list[int]
    overlap_px_x: int
    overlap_px_y: int


@dataclass
class IntersectionMetrics:
    img_h: int
    img_w: int
    bbox_w_m: float
    bbox_h_m: float
    eff_width_mm: float
    eff_height_mm: float
    mm_per_px_x: float
    mm_per_px_y: float
    mppx: float
    mppy: float


def _px_to_m_x(x_px: np.ndarray, img_w: int, bbox_w_m: float) -> np.ndarray:
    denom = float(max(1, img_w - 1))
    return (np.asarray(x_px, dtype=float) / denom) * float(bbox_w_m)


def _px_to_m_y(y_px: np.ndarray, img_h: int, bbox_h_m: float) -> np.ndarray:
    denom = float(max(1, img_h - 1))
    return (np.asarray(y_px, dtype=float) / denom) * float(bbox_h_m)


def _clip_tile_interval_mm(start_mm: float, bed_mm: float, total_mm: float) -> tuple[float, float]:
    """Clip one nominal tile interval to the real map extent in mm."""
    lo_mm = max(0.0, float(start_mm))
    hi_mm = min(float(total_mm), float(start_mm) + float(bed_mm))
    return lo_mm, hi_mm


def _interval_mm_to_px_bounds(lo_mm: float, hi_mm: float, total_mm: float, total_px: int) -> tuple[int, int]:
    """Convert a continuous mm interval to inclusive raster coverage via floor/ceil."""
    if hi_mm <= lo_mm:
        return 0, 0
    px_per_mm = float(total_px) / float(max(total_mm, 1e-9))
    lo_px = max(0, int(np.floor(float(lo_mm) * px_per_mm)))
    hi_px = min(int(total_px), int(np.ceil(float(hi_mm) * px_per_mm)))
    return lo_px, hi_px


def _axis_interval(points_rc: np.ndarray, axis: str) -> tuple[float, float]:
    # points_rc: (N,2) as [row, col]
    a = points_rc[:, 0] if axis == "y" else points_rc[:, 1]
    return float(np.nanmin(a)), float(np.nanmax(a))


def _clip_polyline_to_axis_range(points_rc: np.ndarray, axis: str, lo: float, hi: float, eps: float = 1e-9) -> list[np.ndarray]:
    """
    Clip a polyline to the constraint lo <= axis <= hi (axis is 'x' or 'y').
    Returns list of polyline pieces (each (M,2) with M>=2).
    Uses linear interpolation where segments cross the boundaries.
    """
    if points_rc.shape[0] < 2:
        return []

    ai = 0 if axis == "y" else 1
    pts = np.asarray(points_rc, dtype=float)

    def interp(p0, p1, t):
        return p0 + t * (p1 - p0)

    pieces = []
    cur = []

    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        a0 = float(p0[ai])
        a1 = float(p1[ai])

        # Solve for t in [0,1] where a(t) in [lo,hi]
        if abs(a1 - a0) < eps:
            # constant axis on this segment
            if lo - eps <= a0 <= hi + eps:
                # entire segment inside
                if not cur:
                    cur = [p0]
                cur.append(p1)
            else:
                # outside: finalize current piece
                if len(cur) >= 2:
                    pieces.append(np.vstack(cur))
                cur = []
            continue

        t_lo = (lo - a0) / (a1 - a0)
        t_hi = (hi - a0) / (a1 - a0)
        t0 = min(t_lo, t_hi)
        t1 = max(t_lo, t_hi)

        seg_t0 = max(0.0, t0)
        seg_t1 = min(1.0, t1)

        if seg_t1 < seg_t0 - eps:
            # no intersection
            if len(cur) >= 2:
                pieces.append(np.vstack(cur))
            cur = []
            continue

        # segment portion inside is [seg_t0, seg_t1] (inclusive)
        q0 = interp(p0, p1, seg_t0)
        q1 = interp(p0, p1, seg_t1)

        if not cur:
            cur = [q0]
        else:
            if np.linalg.norm(cur[-1] - q0) > 1e-6:
                cur.append(q0)
        cur.append(q1)

        # If the inside portion ends before the segment ends, we exit the band -> cut piece
        if seg_t1 < 1.0 - eps:
            if len(cur) >= 2:
                pieces.append(np.vstack(cur))
            cur = []

    if len(cur) >= 2:
        pieces.append(np.vstack(cur))
    return pieces


def _subtract_shadow_and_clip(points_rc: np.ndarray, axis: str, shadow_lo: float, shadow_hi: float) -> list[np.ndarray]:
    """
    Remove the axis-shadow interval [shadow_lo, shadow_hi] from the polyline by clipping to the remaining
    axis ranges: (-inf, shadow_lo] and [shadow_hi, +inf). In practice, we clip to the polyline's own axis
    bounds for numerical stability.
    """
    a0, a1 = _axis_interval(points_rc, axis)
    if shadow_lo <= a0 and shadow_hi >= a1:
        return []  # fully shadowed

    # remaining ranges (intersection with [a0,a1])
    ranges = []
    left_hi = min(a1, shadow_lo)
    right_lo = max(a0, shadow_hi)
    if left_hi > a0:
        ranges.append((a0, left_hi))
    if a1 > right_lo:
        ranges.append((right_lo, a1))

    out = []
    for lo, hi in ranges:
        out.extend(_clip_polyline_to_axis_range(points_rc, axis, lo, hi))
    return out


def reduce_segments_by_shadow(segments_rc: list[np.ndarray], axis: str) -> list[np.ndarray]:
    """
    Greedy selection:
      - pick segment with longest axis projection ("shadow")
      - remove segments fully contained in that shadow
      - for partial overlaps, clip away the shadowed part and keep the remainder
      - repeat until none remain
    Returns the list of selected (remaining) segments (in the order picked).
    """
    # normalize + drop tiny segments
    remaining = [np.asarray(s, dtype=float) for s in segments_rc if s is not None and np.asarray(s).shape[0] >= 2]
    selected = []

    while remaining:
        intervals = [(_axis_interval(s, axis), s) for s in remaining]
        # pick max shadow length
        best_idx = max(range(len(intervals)), key=lambda i: (intervals[i][0][1] - intervals[i][0][0]))
        (best_lo, best_hi), best_seg = intervals[best_idx]

        selected.append(best_seg)

        new_remaining = []
        for (lo, hi), seg in intervals:
            if seg is best_seg:
                continue
            # fully encapsulated -> drop
            if best_lo <= lo and hi <= best_hi:
                continue
            # partial/no overlap -> clip away best shadow if overlapping
            if hi <= best_lo or lo >= best_hi:
                new_remaining.append(seg)
                continue
            # partial overlap -> remove shadowed portion(s)
            clipped_pieces = _subtract_shadow_and_clip(seg, axis, best_lo, best_hi)
            new_remaining.extend([p for p in clipped_pieces if p.shape[0] >= 2])

        remaining = new_remaining

    return selected


def split_polyline_monotonic_axis(
    points_rc: np.ndarray,
    axis: str,
    mm_per_px: float,
    reversal_split_mm: float,
    eps: float = 1e-12,
) -> list[np.ndarray]:
    """
    Split a polyline into pieces that are *approximately 1-to-1* along the corridor axis, but with hysteresis.

    Instead of splitting immediately on direction reversal, we allow some backtracking.
    We split only after the accumulated travel *against* the previous direction reaches `reversal_split_mm`.

    The split point is interpolated within the segment where the threshold is crossed.
    """
    pts = np.asarray(points_rc, dtype=float)
    if pts.shape[0] < 2:
        return []
    if mm_per_px <= 0:
        raise ValueError("mm_per_px must be > 0")
    if reversal_split_mm < 0:
        raise ValueError("reversal_split_mm must be >= 0")

    ai = 0 if axis == "y" else 1

    pieces: list[np.ndarray] = []
    cur: list[np.ndarray] = [pts[0].copy()]

    last_sign = 0  # -1 / +1; 0 means "not established yet"
    reverse_accum_mm = 0.0

    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        da = float(p1[ai] - p0[ai])

        # zero delta along axis => doesn't affect monotonicity; keep it
        if abs(da) <= eps:
            cur.append(p1.copy())
            continue

        sign = 1 if da > 0 else -1

        if last_sign == 0:
            last_sign = sign
            reverse_accum_mm = 0.0
            cur.append(p1.copy())
            continue

        if sign == last_sign:
            # normal forward travel resets reverse accumulator
            reverse_accum_mm = 0.0
            cur.append(p1.copy())
            continue

        # reversal: accumulate reverse distance
        seg_mm = abs(da) * float(mm_per_px)

        if reversal_split_mm == 0.0:
            # behave like "split immediately" (but still continuous)
            cut = p0.copy()
            if np.linalg.norm(cur[-1] - cut) > 1e-9:
                cur.append(cut)
            if len(cur) >= 2:
                pieces.append(np.vstack(cur))
            cur = [cut.copy(), p1.copy()]
            last_sign = sign
            reverse_accum_mm = 0.0
            continue

        if reverse_accum_mm + seg_mm < reversal_split_mm - 1e-9:
            reverse_accum_mm += seg_mm
            cur.append(p1.copy())
            continue

        # threshold crossed within this segment -> interpolate cut point
        needed_mm = max(0.0, reversal_split_mm - reverse_accum_mm)
        t = needed_mm / seg_mm  # in (0, 1]
        t = float(np.clip(t, 0.0, 1.0))
        cut = p0 + t * (p1 - p0)

        if np.linalg.norm(cur[-1] - cut) > 1e-9:
            cur.append(cut.copy())
        if len(cur) >= 2:
            pieces.append(np.vstack(cur))

        # start new piece continuing from cut
        cur = [cut.copy(), p1.copy()]
        last_sign = sign
        reverse_accum_mm = 0.0

    if len(cur) >= 2:
        pieces.append(np.vstack(cur))
    return pieces


def split_segments_monotonic_axis(
    segments_rc: list[np.ndarray],
    axis: str,
    mm_per_px: float,
    reversal_split_mm: float,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for seg in segments_rc:
        if seg is None:
            continue
        out.extend(
            split_polyline_monotonic_axis(
                seg,
                axis=axis,
                mm_per_px=mm_per_px,
                reversal_split_mm=reversal_split_mm,
            )
        )
    return [s for s in out if s is not None and s.shape[0] >= 2]


def _euclid_len_rc(p0_rc: np.ndarray, p1_rc: np.ndarray) -> float:
    dr = float(p1_rc[0] - p0_rc[0])
    dc = float(p1_rc[1] - p0_rc[1])
    return float(np.hypot(dr, dc))


def _quantize_rc(p_rc: np.ndarray, q: float) -> tuple[int, int]:
    # q=2 => 0.5px bins
    return (int(round(float(p_rc[0]) * q)), int(round(float(p_rc[1]) * q)))


def _metric_len_rc(p0_rc: np.ndarray, p1_rc: np.ndarray, mppx: float, mppy: float) -> float:
    dr_m = float(p1_rc[0] - p0_rc[0]) * float(mppy)
    dc_m = float(p1_rc[1] - p0_rc[1]) * float(mppx)
    return float(np.hypot(dr_m, dc_m))


def _coords_metric_xy(coords_rc: list[np.ndarray], mppx: float, mppy: float) -> np.ndarray:
    # returns (N,2) with x=col*mppx, y=row*mppy (meters)
    cols = np.array([p[1] for p in coords_rc], dtype=float)
    rows = np.array([p[0] for p in coords_rc], dtype=float)
    return np.column_stack([cols * float(mppx), rows * float(mppy)])


def _pixel_edge_step_m(mppx: float, mppy: float) -> float:
    steps = [float(v) for v in (mppx, mppy) if np.isfinite(v) and float(v) > 0.0]
    return min(steps) if steps else 0.0


def _add_resampled_edge(
    coords_rc: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    u: int,
    v: int,
    *,
    mppx: float,
    mppy: float,
    max_step_m: float,
):
    if u == v:
        return

    max_step_m = float(max_step_m)
    if max_step_m <= 0.0:
        w_m = _metric_len_rc(coords_rc[u], coords_rc[v], mppx=mppx, mppy=mppy)
        adj[u].append((v, float(w_m)))
        adj[v].append((u, float(w_m)))
        return

    p0 = np.asarray(coords_rc[u], dtype=float)
    p1 = np.asarray(coords_rc[v], dtype=float)
    total_m = _metric_len_rc(p0, p1, mppx=mppx, mppy=mppy)
    segments = max(1, int(np.ceil(total_m / max_step_m)))
    prev = int(u)

    for step in range(1, segments):
        t = float(step) / float(segments)
        point_rc = ((1.0 - t) * p0) + (t * p1)
        nid = len(coords_rc)
        coords_rc.append(np.asarray(point_rc, dtype=float))
        adj[nid] = []
        w_m = _metric_len_rc(coords_rc[prev], coords_rc[nid], mppx=mppx, mppy=mppy)
        adj[prev].append((nid, float(w_m)))
        adj[nid].append((prev, float(w_m)))
        prev = nid

    w_m = _metric_len_rc(coords_rc[prev], coords_rc[v], mppx=mppx, mppy=mppy)
    adj[prev].append((v, float(w_m)))
    adj[v].append((prev, float(w_m)))


def build_graph_from_polylines(
    polylines_rc: list[np.ndarray],
    mppx: float,
    mppy: float,
    quantize_q: float = 2.0,
    resample_step_m: Optional[float] = None,
):
    """
    Build an undirected weighted graph from polylines in (row,col).
    Edge weights are metric lengths (meters), not pixel lengths.
    """
    key_to_id: dict[tuple[int, int], int] = {}
    coords: list[np.ndarray] = []
    adj: dict[int, list[tuple[int, float]]] = {}
    max_step_m = (
        float(resample_step_m)
        if resample_step_m is not None and np.isfinite(resample_step_m) and float(resample_step_m) > 0.0
        else 0.0
    )

    def get_node_id(p_rc: np.ndarray) -> int:
        key = _quantize_rc(p_rc, quantize_q)
        nid = key_to_id.get(key)
        if nid is None:
            nid = len(coords)
            key_to_id[key] = nid
            coords.append(np.asarray(p_rc, dtype=float))
            adj[nid] = []
        return nid

    for pl in polylines_rc:
        pts = np.asarray(pl, dtype=float)
        if pts.shape[0] < 2:
            continue
        prev = get_node_id(pts[0])
        for k in range(1, pts.shape[0]):
            cur = get_node_id(pts[k])
            _add_resampled_edge(
                coords,
                adj,
                prev,
                cur,
                mppx=mppx,
                mppy=mppy,
                max_step_m=max_step_m,
            )
            prev = cur

    return coords, adj, key_to_id


def connected_components(adj: dict[int, list[tuple[int, float]]]) -> list[list[int]]:
    seen = set()
    comps: list[list[int]] = []
    for start in adj.keys():
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _w in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def add_proximity_edges(
    coords_rc: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    mppx: float,
    mppy: float,
    radius_m: float,
):
    """
    Add undirected edges between nodes within `radius_m` in full 2D metric space (x/y), with metric weights.
    Uses KDTree if available; falls back to brute force.
    """
    if radius_m is None or float(radius_m) <= 0.0:
        return
    if len(coords_rc) < 2:
        return

    xy = _coords_metric_xy(coords_rc, mppx=mppx, mppy=mppy)
    r = float(radius_m)

    # avoid duplicates
    added: set[tuple[int, int]] = set()

    def add_edge(u: int, v: int):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in added:
            return
        added.add((a, b))
        dx = float(xy[u, 0] - xy[v, 0])
        dy = float(xy[u, 1] - xy[v, 1])
        w_m = float(np.hypot(dx, dy))
        adj[u].append((v, w_m))
        adj[v].append((u, w_m))

    if KDTree is not None:
        tree = KDTree(xy)
        for u, v in tree.query_pairs(r):
            add_edge(int(u), int(v))
    else:
        # brute force fallback
        n = xy.shape[0]
        for u in range(n):
            du = xy[u]
            for v in range(u + 1, n):
                dx = float(du[0] - xy[v, 0])
                dy = float(du[1] - xy[v, 1])
                if (dx * dx + dy * dy) <= (r * r):
                    add_edge(u, v)


def add_proximity_edges_within_component_knn(
    coords_rc: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    node_ids: list[int],
    mppx: float,
    mppy: float,
    radius_m: float,
    k: int,
):
    """
    Add proximity edges *within* one connected component only.
    To avoid O(pairs), connect each node to up to k nearest neighbors (2D metric) that are within radius_m.
    """
    if radius_m is None or float(radius_m) <= 0.0:
        return
    if k is None or int(k) <= 0:
        return
    if len(node_ids) < 2:
        return

    ids = np.asarray(node_ids, dtype=int)
    xy = _coords_metric_xy([coords_rc[i] for i in ids], mppx=mppx, mppy=mppy)  # (n,2) meters
    r = float(radius_m)
    k = int(k)

    added: set[tuple[int, int]] = set()

    def add_edge(global_u: int, global_v: int, w_m: float):
        if global_u == global_v:
            return
        a, b = (global_u, global_v) if global_u < global_v else (global_v, global_u)
        if (a, b) in added:
            return
        added.add((a, b))
        adj[global_u].append((global_v, float(w_m)))
        adj[global_v].append((global_u, float(w_m)))

    if KDTree is not None:
        tree = KDTree(xy)
        # query k+1 because the nearest neighbor is the point itself (dist=0)
        dists, idxs = tree.query(xy, k=min(k + 1, xy.shape[0]))
        # normalize shapes for k==0 / k==1
        dists = np.atleast_2d(dists)
        idxs = np.atleast_2d(idxs)

        for local_i in range(xy.shape[0]):
            gi = int(ids[local_i])
            for t in range(1, idxs.shape[1]):  # skip self
                local_j = int(idxs[local_i, t])
                d = float(dists[local_i, t])
                if not np.isfinite(d) or d <= 0.0 or d > r:
                    continue
                gj = int(ids[local_j])
                add_edge(gi, gj, d)
    else:
        # brute-force fallback (still bounded by k)
        for local_i in range(xy.shape[0]):
            gi = int(ids[local_i])
            diff = xy - xy[local_i]
            d2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
            order = np.argsort(d2)
            picked = 0
            for local_j in order[1:]:
                d = float(np.sqrt(d2[local_j]))
                if d <= 0.0:
                    continue
                if d > r:
                    break
                gj = int(ids[int(local_j)])
                add_edge(gi, gj, d)
                picked += 1
                if picked >= k:
                    break


def add_proximity_edges_between_components_knn(
    coords_rc: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    comp_a: list[int],
    comp_b: list[int],
    mppx: float,
    mppy: float,
    radius_m: float,
    k: int,
):
    """
    Add shortcut edges *between* two different components only.
    Bounded: for each node in the smaller component, connect to up to k nearest nodes in the other component,
    but only if within radius_m (2D metric distance).
    """
    if radius_m is None or float(radius_m) <= 0.0:
        return
    if k is None or int(k) <= 0:
        return
    if not comp_a or not comp_b:
        return

    # Always query from smaller -> larger
    if len(comp_a) <= len(comp_b):
        src_ids = np.asarray(comp_a, dtype=int)
        dst_ids = np.asarray(comp_b, dtype=int)
    else:
        src_ids = np.asarray(comp_b, dtype=int)
        dst_ids = np.asarray(comp_a, dtype=int)

    if src_ids.size == 0 or dst_ids.size == 0:
        return

    # metric XY for both sets
    src_xy = _coords_metric_xy([coords_rc[i] for i in src_ids], mppx=mppx, mppy=mppy)
    dst_xy = _coords_metric_xy([coords_rc[i] for i in dst_ids], mppx=mppx, mppy=mppy)

    r = float(radius_m)
    k = int(min(int(k), dst_xy.shape[0]))

    added: set[tuple[int, int]] = set()

    def add_edge(u: int, v: int, w_m: float):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in added:
            return
        added.add((a, b))
        adj[u].append((v, float(w_m)))
        adj[v].append((u, float(w_m)))

    if KDTree is not None:
        tree = KDTree(dst_xy)
        # query k nearest in destination for each source node
        dists, idxs = tree.query(src_xy, k=k)
        dists = np.atleast_2d(dists)
        idxs = np.atleast_2d(idxs)

        for si in range(src_ids.size):
            u = int(src_ids[si])
            for t in range(idxs.shape[1]):
                d = float(dists[si, t])
                if not np.isfinite(d) or d <= 0.0 or d > r:
                    continue
                v = int(dst_ids[int(idxs[si, t])])
                add_edge(u, v, d)
    else:
        # brute fallback: still bounded by k via sorting
        for si in range(src_ids.size):
            u = int(src_ids[si])
            diff = dst_xy - src_xy[si]
            d2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
            order = np.argsort(d2)
            picked = 0
            for j in order:
                d = float(np.sqrt(d2[int(j)]))
                if d <= 0.0:
                    continue
                if d > r:
                    break
                v = int(dst_ids[int(j)])
                add_edge(u, v, d)
                picked += 1
                if picked >= k:
                    break


def connect_components_by_nearest_mst(
    coords: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    mppx: float,
    mppy: float,
    resample_step_m: Optional[float] = None,
):
    """
    Connect disjoint components with exactly (k-1) bridges (MST-style), using nearest points in 2D metric space.
    NOTE: This function ONLY adds bridging edges between components (no proximity shortcuts).
    """
    comps = [set(comp) for comp in connected_components(adj)]
    if len(comps) <= 1:
        return

    xy_all = _coords_metric_xy(coords, mppx=mppx, mppy=mppy)

    def add_bridge(u: int, v: int):
        _add_resampled_edge(
            coords,
            adj,
            u,
            v,
            mppx=mppx,
            mppy=mppy,
            max_step_m=(
                float(resample_step_m)
                if resample_step_m is not None and np.isfinite(resample_step_m) and float(resample_step_m) > 0.0
                else 0.0
            ),
        )

    while len(comps) > 1:
        best = (float("inf"), None, None, None, None)  # dist, i, j, u, v

        for i in range(len(comps)):
            ids_i = np.fromiter(comps[i], dtype=int)
            if ids_i.size == 0:
                continue
            xy_i = xy_all[ids_i]

            for j in range(i + 1, len(comps)):
                ids_j = np.fromiter(comps[j], dtype=int)
                if ids_j.size == 0:
                    continue
                xy_j = xy_all[ids_j]

                if KDTree is not None:
                    tree_j = KDTree(xy_j)
                    dists, nn = tree_j.query(xy_i, k=1)
                    kmin = int(np.argmin(dists))
                    d = float(dists[kmin])
                    u = int(ids_i[kmin])
                    v = int(ids_j[int(nn[kmin])])
                else:
                    d2 = (xy_i[:, None, 0] - xy_j[None, :, 0]) ** 2 + (xy_i[:, None, 1] - xy_j[None, :, 1]) ** 2
                    ia, ib = np.unravel_index(int(np.argmin(d2)), d2.shape)
                    d = float(np.sqrt(d2[ia, ib]))
                    u = int(ids_i[int(ia)])
                    v = int(ids_j[int(ib)])

                if d < best[0]:
                    best = (d, i, j, u, v)

        d, i, j, u, v = best
        if i is None:
            break

        add_bridge(u, v)
        comps[i].update(comps[j])
        comps.pop(j)


def dijkstra_path(adj: dict[int, list[tuple[int, float]]], start: int, goal: int) -> list[int]:
    dist = {start: 0.0}
    prev: dict[int, int] = {}
    pq = [(0.0, start)]
    seen = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == goal:
            break
        for v, w in adj.get(u, []):
            nd = d + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist:
        return []

    # reconstruct
    path = [goal]
    cur = goal
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def extend_graph_to_axis_borders(
    coords: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    axis: str,
    border_min: float,
    border_max: float,
    mppx: float,
    mppy: float,
    resample_step_m: Optional[float] = None,
    eps: float = 1e-6,
) -> tuple[int, int]:
    """
    Ensure there are nodes exactly on border_min/border_max along the corridor axis (in *pixel* axis units).
    If needed, extend the current min/max axis node straight to the border (orthogonal coord unchanged).
    Edge weights are metric lengths (meters).
    Returns (start_node_id_at_min_border, end_node_id_at_max_border).
    """
    ai = 0 if axis == "y" else 1

    axis_vals = np.array([p[ai] for p in coords], dtype=float)
    min_id = int(np.argmin(axis_vals))
    max_id = int(np.argmax(axis_vals))
    min_val = float(axis_vals[min_id])
    max_val = float(axis_vals[max_id])

    def add_node_and_edge(from_id: int, new_axis_val: float) -> int:
        new_p = np.asarray(coords[from_id], dtype=float).copy()
        new_p[ai] = float(new_axis_val)
        nid = len(coords)
        coords.append(new_p)
        adj[nid] = []
        _add_resampled_edge(
            coords,
            adj,
            from_id,
            nid,
            mppx=mppx,
            mppy=mppy,
            max_step_m=(
                float(resample_step_m)
                if resample_step_m is not None and np.isfinite(resample_step_m) and float(resample_step_m) > 0.0
                else 0.0
            ),
        )
        return nid

    start_id = min_id
    end_id = max_id

    if abs(min_val - float(border_min)) > eps:
        start_id = add_node_and_edge(min_id, float(border_min))
    if abs(max_val - float(border_max)) > eps:
        end_id = add_node_and_edge(max_id, float(border_max))

    return start_id, end_id


def add_proximity_edges_within_component_radius(
    coords_rc: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    node_ids: list[int],
    mppx: float,
    mppy: float,
    radius_m: float,
    alpha: float = 2.0,
    resample_step_m: Optional[float] = None,
):
    """
    Over the provided node set, add an undirected edge between candidate node pairs within radius_m.

    For node pairs already connected in the snapshot graph, skip "path-neighborhood" edges:

      Add (u,v) only if d_graph(u,v) > alpha * d_direct(u,v),

    where d_graph is shortest-path distance in the CURRENT subgraph (snapshot before adding new edges)
    and d_direct is 2D Euclidean metric distance.

    For node pairs in different snapshot components, add the edge directly if they are within radius_m.
    Accepted shortcut edges are resampled so later reduction passes can reason about them instead of
    seeing one long hop.
    """
    if radius_m is None or float(radius_m) <= 0.0:
        return
    if not node_ids or len(node_ids) < 2:
        return
    alpha = float(alpha)
    if alpha <= 0:
        return

    comp_set = set(int(i) for i in node_ids)

    # Snapshot adjacency restricted to the provided node set so decisions don't change while we add edges.
    adj0: dict[int, list[tuple[int, float]]] = {}
    for u in comp_set:
        adj0[u] = [(v, float(w)) for (v, w) in adj.get(u, []) if v in comp_set]

    comp_index: dict[int, int] = {}
    for comp_i, comp in enumerate(connected_components(adj0)):
        for u in comp:
            comp_index[int(u)] = int(comp_i)

    ids = np.asarray(list(comp_set), dtype=int)
    # stable order for local indexing
    ids.sort()
    xy = _coords_metric_xy([coords_rc[i] for i in ids], mppx=mppx, mppy=mppy)  # (n,2) meters
    r = float(radius_m)
    eps = 1e-9
    pixel_step_m = (
        float(resample_step_m)
        if resample_step_m is not None and np.isfinite(resample_step_m) and float(resample_step_m) > 0.0
        else _pixel_edge_step_m(mppx, mppy)
    )

    added: set[tuple[int, int]] = set()

    def add_edge(u: int, v: int):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in added:
            return
        added.add((a, b))
        _add_resampled_edge(
            coords_rc,
            adj,
            u,
            v,
            mppx=mppx,
            mppy=mppy,
            max_step_m=pixel_step_m,
        )

    def dijkstra_limited(start: int, cutoff: float) -> dict[int, float]:
        dist: dict[int, float] = {start: 0.0}
        pq: list[tuple[float, int]] = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > cutoff + eps:
                break
            if d != dist.get(u, None):
                continue
            for v, w in adj0.get(u, []):
                nd = d + float(w)
                if nd <= cutoff + eps and nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    if KDTree is not None:
        tree = KDTree(xy)
        for li in range(xy.shape[0]):
            # candidate neighbors within radius (local indices)
            neigh = tree.query_ball_point(xy[li], r)
            if not neigh:
                continue

            # prepare same-component targets (only lj>li to avoid duplicates)
            targets: list[tuple[int, float]] = []
            max_direct = 0.0
            u = int(ids[li])
            comp_u = comp_index.get(u, -1)
            for lj in neigh:
                lj = int(lj)
                if lj <= li:
                    continue
                dx = float(xy[li, 0] - xy[lj, 0])
                dy = float(xy[li, 1] - xy[lj, 1])
                d_direct = float(np.hypot(dx, dy))
                if d_direct <= 0.0 or d_direct > r + eps:
                    continue
                v = int(ids[lj])
                if comp_index.get(v, -1) != comp_u:
                    add_edge(u, v)
                    continue
                targets.append((v, d_direct))
                max_direct = max(max_direct, d_direct)

            if not targets:
                continue

            cutoff = alpha * max_direct
            dist_u = dijkstra_limited(u, cutoff=cutoff)

            for v, d_direct in targets:
                d_graph = dist_u.get(v, float("inf"))
                if d_graph > alpha * d_direct + eps:
                    add_edge(u, v)
    else:
        # brute-force fallback
        n = xy.shape[0]
        for li in range(n):
            targets: list[tuple[int, float]] = []
            max_direct = 0.0
            u = int(ids[li])
            comp_u = comp_index.get(u, -1)
            for lj in range(li + 1, n):
                dx = float(xy[li, 0] - xy[lj, 0])
                dy = float(xy[li, 1] - xy[lj, 1])
                d_direct = float(np.hypot(dx, dy))
                if d_direct <= 0.0 or d_direct > r + eps:
                    continue
                v = int(ids[lj])
                if comp_index.get(v, -1) != comp_u:
                    add_edge(u, v)
                    continue
                targets.append((v, d_direct))
                max_direct = max(max_direct, d_direct)

            if not targets:
                continue

            cutoff = alpha * max_direct
            dist_u = dijkstra_limited(u, cutoff=cutoff)

            for v, d_direct in targets:
                d_graph = dist_u.get(v, float("inf"))
                if d_graph > alpha * d_direct + eps:
                    add_edge(u, v)


def build_separating_line_for_corridor(
    elev_map: np.ndarray,
    corridor: dict,
    *,
    img_h: int,
    img_w: int,
    bbox_w_m: float,
    bbox_h_m: float,
    eff_width_mm: float,
    eff_height_mm: float,
    interval_m: float,
    fine_threshold_m: float,
    fine_interval_m: float,
    reversal_split_mm: float,
    neighbor_radius_mm: float,
    resampling_interval_mm: float,
    skip_alpha: float,
):
    """
    Build ONE separating line for ONE corridor.
    Returns (rows_full_px, cols_full_px) arrays for plotting (full-image pixel coords), or None.
    """
    c = corridor
    crop = elev_map[c["y0"] : c["y1"], c["x0"] : c["x1"]]
    if crop.size == 0:
        return None

    crop_min = float(np.nanmin(crop))
    crop_max = float(np.nanmax(crop))
    if not np.isfinite(crop_min) or not np.isfinite(crop_max):
        return None

    levels = _build_contour_levels(
        crop_min,
        crop_max,
        fine_threshold_m=float(fine_threshold_m),
        fine_interval_m=float(fine_interval_m),
        coarse_interval_m=float(interval_m),
    )
    if levels.size == 0:
        return None

    fill_val = crop_min - 1e6
    crop_filled = np.nan_to_num(crop, nan=fill_val, posinf=fill_val, neginf=fill_val)

    all_segments_crop_rc: list[np.ndarray] = []
    for level in levels:
        cs = find_contours(crop_filled, float(level))
        for contour in cs:
            if contour is None or contour.shape[0] < 2:
                continue
            all_segments_crop_rc.append(np.asarray(contour, dtype=float))

    axis = "y" if c["kind"] == "col" else "x"
    axis_mm_per_px = (eff_height_mm / float(img_h)) if axis == "y" else (eff_width_mm / float(img_w))

    all_segments_crop_rc = split_segments_monotonic_axis(
        all_segments_crop_rc,
        axis=axis,
        mm_per_px=axis_mm_per_px,
        reversal_split_mm=float(reversal_split_mm),
    )

    selected_segments_crop_rc = reduce_segments_by_shadow(all_segments_crop_rc, axis=axis)
    if not selected_segments_crop_rc:
        return None

    # meters-per-pixel for the full raster (metric edge weights)
    mppx = float(bbox_w_m) / float(max(1, img_w - 1))
    mppy = float(bbox_h_m) / float(max(1, img_h - 1))
    meters_per_mm_w = float(bbox_w_m) / float(max(1e-9, eff_width_mm))
    meters_per_mm_h = float(bbox_h_m) / float(max(1e-9, eff_height_mm))
    meters_per_mm = 0.5 * (meters_per_mm_w + meters_per_mm_h)
    resample_step_m = float(resampling_interval_mm) * meters_per_mm

    coords, adj, _key_to_id = build_graph_from_polylines(
        selected_segments_crop_rc,
        mppx=mppx,
        mppy=mppy,
        quantize_q=2.0,
        resample_step_m=resample_step_m,
    )
    if not coords:
        return None

    # neighbor radius specified in mm on output map -> meters in-world
    neighbor_radius_m = float(neighbor_radius_mm) * meters_per_mm

    add_proximity_edges_within_component_radius(
        coords,
        adj,
        node_ids=list(adj.keys()),
        mppx=mppx,
        mppy=mppy,
        radius_m=neighbor_radius_m,
        alpha=float(skip_alpha),
        resample_step_m=resample_step_m,
    )

    connect_components_by_nearest_mst(coords, adj, mppx=mppx, mppy=mppy, resample_step_m=resample_step_m)

    crop_h, crop_w = crop.shape
    border_min = 0.0
    border_max = float((crop_h - 1) if axis == "y" else (crop_w - 1))
    start_id, end_id = extend_graph_to_axis_borders(
        coords,
        adj,
        axis=axis,
        border_min=border_min,
        border_max=border_max,
        mppx=mppx,
        mppy=mppy,
        resample_step_m=resample_step_m,
    )

    path_ids = dijkstra_path(adj, start_id, end_id)
    if not path_ids:
        return None

    # path nodes are in crop coords -> shift to full image coords
    rows_full = np.array([coords[n][0] for n in path_ids], dtype=float) + float(c["y0"])
    cols_full = np.array([coords[n][1] for n in path_ids], dtype=float) + float(c["x0"])
    return rows_full, cols_full


def _polyline_to_cut_col_at_row(rows: np.ndarray, cols: np.ndarray, img_h: int) -> np.ndarray:
    """For a mostly-y-monotone polyline: return cut_col[r] for r=0..img_h-1 (float)."""
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    order = np.argsort(rows)
    rows = rows[order]
    cols = cols[order]
    # collapse duplicates in rows by averaging
    ur, inv = np.unique(rows, return_inverse=True)
    uc = np.zeros_like(ur, dtype=float)
    cnt = np.zeros_like(ur, dtype=float)
    for i, k in enumerate(inv):
        uc[k] += cols[i]
        cnt[k] += 1.0
    uc = uc / np.maximum(cnt, 1.0)
    rr = np.arange(img_h, dtype=float)
    return np.interp(rr, ur, uc, left=uc[0], right=uc[-1])


def _polyline_to_cut_row_at_col(rows: np.ndarray, cols: np.ndarray, img_w: int) -> np.ndarray:
    """For a mostly-x-monotone polyline: return cut_row[c] for c=0..img_w-1 (float)."""
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    order = np.argsort(cols)
    cols = cols[order]
    rows = rows[order]
    uc, inv = np.unique(cols, return_inverse=True)
    ur = np.zeros_like(uc, dtype=float)
    cnt = np.zeros_like(uc, dtype=float)
    for i, k in enumerate(inv):
        ur[k] += rows[i]
        cnt[k] += 1.0
    ur = ur / np.maximum(cnt, 1.0)
    cc = np.arange(img_w, dtype=float)
    return np.interp(cc, uc, ur, left=ur[0], right=ur[-1])


def generate_full_mesh(
    elev_map: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
    bottom_thickness_mm: float,
    desired_height_mm: float,
    height_exponent: float = 1.0,
    norm_h_min: Optional[float] = None,
    norm_h_max: Optional[float] = None,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
) -> trimesh.Trimesh:
    """
    Inputs:
    - `elev_map`: 2D elevation raster.
    - `mm_per_px_x`, `mm_per_px_y`: physical XY spacing per raster sample.
    - `bottom_thickness_mm`, `desired_height_mm`: relief thickness controls.
    - `height_exponent`: optional nonlinear scaling on normalized heights.
    - `norm_h_min`, `norm_h_max`: optional global normalization range.
    - `origin_x_mm`, `origin_y_mm`: XY offset for cropped local meshes.

    Outputs:
    - Returns one watertight terrain solid mesh in mm coordinates.
    """
    h, w = elev_map.shape
    
    # Normalize heights
    valid_mask = np.isfinite(elev_map)
    if not np.any(valid_mask):
        raise ValueError("Elevation map has no valid data")

    if norm_h_min is None or norm_h_max is None:
        h_min = float(np.nanmin(elev_map[valid_mask]))
        h_max = float(np.nanmax(elev_map[valid_mask]))
    else:
        h_min = float(norm_h_min)
        h_max = float(norm_h_max)
    denom = max(1e-12, h_max - h_min)
    
    # Normalize to [0, 1]
    t = (elev_map - h_min) / denom
    t[~valid_mask] = 0.0 # Handle NaNs by setting to min height
    np.clip(t, 0.0, 1.0, out=t)
    if abs(float(height_exponent) - 1.0) > 1e-9:
        np.power(t, float(height_exponent), out=t)
    
    # Map to mm Z
    z_values = bottom_thickness_mm + t * desired_height_mm
    
    # Create grid of X, Y in mm
    x_idx = np.arange(w)
    y_idx = np.arange(h)
    xv, yv = np.meshgrid(x_idx, y_idx)
    
    x_mm = origin_x_mm + (xv * mm_per_px_x)
    y_mm = origin_y_mm + (yv * mm_per_px_y)
    
    # Vertices (H*W, 3)
    vertices_top = np.column_stack((x_mm.ravel(), y_mm.ravel(), z_values.ravel()))
    
    # Create faces for the grid
    # Quads: (r, c), (r, c+1), (r+1, c+1), (r+1, c)
    # Split into two triangles
    
    # Top surface faces (Counter-Clockwise for +Z normal)
    # Grid indices (H-1, W-1)
    r = np.arange(h - 1)
    c = np.arange(w - 1)
    rv, cv = np.meshgrid(r, c, indexing='ij')
    
    v00 = rv * w + cv
    v01 = rv * w + (cv + 1)
    v10 = (rv + 1) * w + cv
    v11 = (rv + 1) * w + (cv + 1)
    
    # Triangles: (v00, v11, v10) and (v00, v01, v11)
    # v00(0,0)->v11(1,1)->v10(0,1) => (1,1)x(-1,0) = (0,0,1) +Z
    # v00(0,0)->v01(1,0)->v11(1,1) => (1,0)x(0,1) = (0,0,1) +Z
    f1 = np.stack((v00, v11, v10), axis=-1).reshape(-1, 3)
    f2 = np.stack((v00, v01, v11), axis=-1).reshape(-1, 3)
    faces_top = np.vstack((f1, f2))
    
    # Create solid block: Add bottom vertices at Z=0
    vertices_bottom = vertices_top.copy()
    vertices_bottom[:, 2] = 0.0
    
    # Offset for bottom vertices
    offset = h * w
    
    # Bottom faces (Clockwise for -Z normal)
    # (b00, b10, b11) and (b00, b11, b01)
    b00 = v00 + offset
    b01 = v01 + offset
    b10 = v10 + offset
    b11 = v11 + offset
    
    # b00(0,0)->b10(0,1)->b11(1,1) => (0,1)x(1,0) = (0,0,-1) -Z
    # b00(0,0)->b11(1,1)->b01(1,0) => (1,1)x(0,-1) = (0,0,-1) -Z
    f3 = np.stack((b00, b10, b11), axis=-1).reshape(-1, 3)
    f4 = np.stack((b00, b11, b01), axis=-1).reshape(-1, 3)
    faces_bottom = np.vstack((f3, f4))
    
    # Side faces
    # Top edge (r=0): v0(c) -> v0(c+1) -> b0(c+1) -> b0(c)
    # Normal should be -Y (since y increases with r)
    c_edge = np.arange(w - 1)
    top_v0 = c_edge
    top_v1 = c_edge + 1
    top_b0 = top_v0 + offset
    top_b1 = top_v1 + offset
    
    f_top_side_1 = np.stack((top_v0, top_b0, top_b1), axis=-1)
    f_top_side_2 = np.stack((top_v0, top_b1, top_v1), axis=-1)
    
    # Bottom edge (r=h-1): Normal +Y
    bot_v0 = (h - 1) * w + c_edge
    bot_v1 = bot_v0 + 1
    bot_b0 = bot_v0 + offset
    bot_b1 = bot_v1 + offset
    
    f_bot_side_1 = np.stack((bot_v0, bot_v1, bot_b1), axis=-1)
    f_bot_side_2 = np.stack((bot_v0, bot_b1, bot_b0), axis=-1)
    
    # Left edge (c=0): Normal -X
    r_edge = np.arange(h - 1)
    left_v0 = r_edge * w
    left_v1 = (r_edge + 1) * w
    left_b0 = left_v0 + offset
    left_b1 = left_v1 + offset
    
    f_left_side_1 = np.stack((left_v0, left_v1, left_b1), axis=-1)
    f_left_side_2 = np.stack((left_v0, left_b1, left_b0), axis=-1)
    
    # Right edge (c=w-1): Normal +X
    right_v0 = r_edge * w + (w - 1)
    right_v1 = (r_edge + 1) * w + (w - 1)
    right_b0 = right_v0 + offset
    right_b1 = right_v1 + offset
    
    f_right_side_1 = np.stack((right_v0, right_b0, right_b1), axis=-1)
    f_right_side_2 = np.stack((right_v0, right_b1, right_v1), axis=-1)
    
    all_vertices = np.vstack((vertices_top, vertices_bottom))
    all_faces = np.vstack((
        faces_top, faces_bottom,
        f_top_side_1, f_top_side_2,
        f_bot_side_1, f_bot_side_2,
        f_left_side_1, f_left_side_2,
        f_right_side_1, f_right_side_2
    ))
    
    # process=True ensures vertices are merged and normals are calculated
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=True)
    
    # Ensure it's a volume
    if not mesh.is_volume:
        mesh.fix_normals()
        
    return mesh


def create_extruded_tool(
    boundary_points: np.ndarray,
    z_min: float,
    z_max: float
) -> trimesh.Trimesh:
    """
    Inputs:
    - `boundary_points`: 2D polygon boundary in XY mesh coordinates.
    - `z_min`, `z_max`: vertical extent of the cutting prism.

    Outputs:
    - Returns a watertight prism mesh suitable for boolean intersection.
    """
    # Ensure closed polygon
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])

    poly = Polygon(boundary_points)
    if not poly.is_valid:
        repaired = poly.buffer(0)
        if repaired.is_empty:
            raise ValueError("Boundary polygon could not be repaired")
        if repaired.geom_type == "Polygon":
            poly = repaired
        elif hasattr(repaired, "geoms"):
            polys = [g for g in repaired.geoms if g.geom_type == "Polygon"]
            if not polys:
                raise ValueError("Boundary repair did not produce a polygon")
            poly = max(polys, key=lambda p: p.area)
        else:
            raise ValueError("Boundary repair produced unsupported geometry")
    # Simplify slightly to reduce vertex count if needed, but separating lines are critical
    # poly = poly.simplify(0.1, preserve_topology=True) 
    
    height = z_max - z_min
    # extrude_polygon creates a mesh from 0 to height
    mesh = trimesh.creation.extrude_polygon(poly, height)
    
    # Shift to z_min
    mesh.apply_translation([0, 0, z_min])
    
    return mesh


def offset_polyline(points: np.ndarray, dist: float, constrain_axis: int = None) -> np.ndarray:
    """
    Offset a 2D polyline by `dist`.
    dist > 0: Shift Right (relative to travel).
    dist < 0: Shift Left.
    constrain_axis: 0 (x) or 1 (y). If set, endpoints are constrained to move only along the OTHER axis.
    """
    if len(points) < 2 or abs(dist) < 1e-9:
        return points.copy()
    
    # Calculate segment normals
    # segments: (N-1, 2)
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0 # avoid div/0
    tangents = segments / lengths
    
    # Normal to the right: (dx, dy) -> (dy, -dx)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    
    # Vertex normals (average of adjacent segment normals)
    v_normals = np.zeros_like(points)
    v_normals[0] = normals[0]
    v_normals[-1] = normals[-1]
    v_normals[1:-1] = normals[:-1] + normals[1:]
    
    # Normalize
    vn_lens = np.linalg.norm(v_normals, axis=1, keepdims=True)
    vn_lens[vn_lens == 0] = 1.0
    v_normals = v_normals / vn_lens
    
    # Apply offset
    new_points = points + v_normals * dist
    
    # Constrain endpoints
    if constrain_axis is not None:
        for idx, seg_idx in [(0, 0), (-1, -1)]:
            n = normals[seg_idx]
            t_vec = tangents[seg_idx]
            
            # If tangent is not perpendicular to constraint axis
            if abs(t_vec[constrain_axis]) > 1e-6:
                # t = - (dist * n[constrain_axis]) / t_vec[constrain_axis]
                t = - (dist * n[constrain_axis]) / t_vec[constrain_axis]
                new_pos = points[idx] + dist * n + t * t_vec
                new_points[idx] = new_pos

    return new_points


def save_lines_obj(path: str, lines: list[np.ndarray], z: float = 0.0):
    """Write one or more 2D polylines to an OBJ file for debugging."""
    with open(path, "w") as f:
        f.write("# Corridor lines\n")
        vertex_offset = 1
        for line in lines:
            # line is (N, 2)
            for x, y in line:
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
            
            # write line indices
            # l v1 v2 v3 ...
            indices = range(vertex_offset, vertex_offset + len(line))
            f.write("l " + " ".join(map(str, indices)) + "\n")
            vertex_offset += len(line)


def save_graph_obj(
    path: str,
    coords_rc: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    *,
    mm_per_px_x: float,
    mm_per_px_y: float,
    z: float = 0.0,
):
    """Write an intersection search graph to OBJ as points plus line segments."""
    with open(path, "w") as f:
        f.write("# Intersection search graph\n")
        for point_rc in coords_rc:
            x_mm = float(point_rc[1]) * float(mm_per_px_x)
            y_mm = float(point_rc[0]) * float(mm_per_px_y)
            f.write(f"v {x_mm:.4f} {y_mm:.4f} {float(z):.4f}\n")

        seen: set[tuple[int, int]] = set()
        for u, neighbors in adj.items():
            for v, _w in neighbors:
                a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                f.write(f"l {a + 1} {b + 1}\n")


def _mesh_intersection(mesh: trimesh.Trimesh, tool: trimesh.Trimesh, preferred_engine: Optional[str] = "manifold"):
    if preferred_engine:
        try:
            return mesh.intersection(tool, engine=preferred_engine)
        except Exception:
            pass
    return mesh.intersection(tool)


def _flip_points_across_x_axis(points_xy: np.ndarray, total_height_mm: float) -> np.ndarray:
    flipped = np.asarray(points_xy, dtype=float).copy()
    flipped[:, 1] = float(total_height_mm) - flipped[:, 1]
    return flipped


def _flip_mesh_across_x_axis(mesh: trimesh.Trimesh, total_height_mm: float) -> trimesh.Trimesh:
    """Mirror a mesh across the X axis and keep it in positive Y coordinates."""
    mesh = mesh.copy()
    mesh.apply_scale([1.0, -1.0, 1.0])
    mesh.apply_translation([0.0, float(total_height_mm), 0.0])
    return mesh


def _empty_mesh_for_tile(tx: int, ty: int) -> trimesh.Trimesh:
    """Create an empty mesh placeholder and record the tile coordinate in metadata."""
    mesh = trimesh.Trimesh(
        vertices=np.zeros((0, 3), dtype=float),
        faces=np.zeros((0, 3), dtype=np.int64),
        process=False,
    )
    mesh.metadata["tile_coord"] = (int(tx), int(ty))
    return mesh


def _normalize_tile_coords(
    tile_coords: Optional[list[tuple[float, float]]],
    *,
    nx: int,
    ny: int,
) -> list[tuple[int, int]]:
    """Normalize optional tile coordinates into unique integer `(tx, ty)` pairs."""
    if tile_coords is None:
        return [(tx, ty) for ty in range(int(ny)) for tx in range(int(nx))]

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for coord in tile_coords:
        if len(coord) != 2:
            raise ValueError(f"Invalid tile coordinate {coord!r}; expected a 2-tuple like (2, 3).")
        tx = int(coord[0])
        ty = int(coord[1])
        if tx < 0 or tx >= int(nx) or ty < 0 or ty >= int(ny):
            raise ValueError(f"Tile coordinate {(tx, ty)!r} is out of range for a {nx}x{ny} tiling.")
        key = (tx, ty)
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def cut_mesh_into_tiles(
    mesh: trimesh.Trimesh,
    horizontal_cutlines: list[np.ndarray],
    vertical_cutlines: list[np.ndarray],
    clearance: float,
    tile_coords: Optional[list[tuple[float, float]]] = None,
) -> list[trimesh.Trimesh]:
    """
    Inputs:
    - `mesh`: watertight source mesh to split.
    - `horizontal_cutlines`: ordered top-to-bottom list of 2D cut polylines.
    - `vertical_cutlines`: ordered left-to-right list of 2D cut polylines.
    - `clearance`: total gap between neighboring tiles in mesh XY units.
    - `tile_coords`: optional list of `(tx, ty)` tile coordinates to extract.

    Outputs:
    - Returns the cut tile meshes in row-major order when `tile_coords` is `None`,
      or in the same order as the requested coordinates otherwise.
    - Each returned mesh stores its `(tx, ty)` coordinate in `mesh.metadata["tile_coord"]`.
    """
    if mesh.is_empty:
        raise ValueError("Input mesh is empty")

    horizontal = [np.asarray(line, dtype=float) for line in horizontal_cutlines]
    vertical = [np.asarray(line, dtype=float) for line in vertical_cutlines]
    nx = len(vertical) + 1
    ny = len(horizontal) + 1
    requested_tiles = _normalize_tile_coords(tile_coords, nx=nx, ny=ny)

    bounds = np.asarray(mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("Input mesh bounds are invalid")

    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    xy_span = max(float(max_x - min_x), float(max_y - min_y), 1.0)
    xy_pad = max(float(clearance), 0.005 * xy_span, 0.5)
    z_pad = max(1.0, 0.05 * max(float(max_z - min_z), 1.0))

    left_x = float(min_x) - xy_pad
    right_x = float(max_x) + xy_pad
    top_y = float(min_y) - xy_pad
    bottom_y = float(max_y) + xy_pad
    half_clearance = float(clearance) / 2.0
    z_min = float(min_z) - z_pad
    z_max = float(max_z) + z_pad

    # The outer tool boundaries sit slightly outside the source mesh so boolean
    # operations do not coincide exactly with an existing vertical wall.
    top_edge = np.array([[left_x, top_y], [right_x, top_y]], dtype=float)
    bottom_edge = np.array([[left_x, bottom_y], [right_x, bottom_y]], dtype=float)
    left_edge = np.array(  [[left_x, top_y], [left_x, bottom_y]], dtype=float)
    right_edge = np.array([[right_x, top_y], [right_x, bottom_y]], dtype=float)

    row_slices: dict[int, trimesh.Trimesh] = {}
    for ty in sorted({ty for _tx, ty in requested_tiles}):
        top_line = top_edge if ty == 0 else offset_polyline(horizontal[ty - 1], -half_clearance, constrain_axis=0)
        bot_line = bottom_edge if ty == ny - 1 else offset_polyline(horizontal[ty], half_clearance, constrain_axis=0)
        row_tool = create_extruded_tool(np.vstack([top_line, bot_line[::-1]]), z_min, z_max)
        row_slices[ty] = _mesh_intersection(mesh, row_tool, preferred_engine="manifold")

    cut_meshes: list[trimesh.Trimesh] = []
    for tx, ty in requested_tiles:
        row_slice = row_slices[ty]
        if row_slice.is_empty:
            cut_meshes.append(_empty_mesh_for_tile(tx, ty))
            continue

        left_line = left_edge if tx == 0 else offset_polyline(vertical[tx - 1], half_clearance, constrain_axis=1)
        right_line = right_edge if tx == nx - 1 else offset_polyline(vertical[tx], -half_clearance, constrain_axis=1)
        col_tool = create_extruded_tool(np.vstack([left_line, right_line[::-1]]), z_min, z_max)
        tile_mesh = _mesh_intersection(row_slice, col_tool, preferred_engine="manifold")
        if tile_mesh.is_empty:
            cut_meshes.append(_empty_mesh_for_tile(tx, ty))
            continue

        tile_mesh.metadata["tile_coord"] = (int(tx), int(ty))
        cut_meshes.append(tile_mesh)

    return cut_meshes


def _worker_build_corridor_line(args):
    """
    Multiprocessing worker: open elev_map via memmap and compute one corridor line.
    Returns dict with both pixel polyline and meter polyline for downstream splitting/plotting.
    """
    (
        corridor,
        mmap_path,
        shape,
        dtype_str,
        img_h,
        img_w,
        bbox_w_m,
        bbox_h_m,
        eff_width_mm,
        eff_height_mm,
        interval_m,
        fine_threshold_m,
        fine_interval_m,
        reversal_split_mm,
        neighbor_radius_mm,
        resampling_interval_mm,
        skip_alpha,
    ) = args

    t0 = time.perf_counter()
    elev_map = np.memmap(mmap_path, dtype=np.dtype(dtype_str), mode="r", shape=shape)

    rc = build_separating_line_for_corridor(
        elev_map,
        corridor,
        img_h=img_h,
        img_w=img_w,
        bbox_w_m=bbox_w_m,
        bbox_h_m=bbox_h_m,
        eff_width_mm=eff_width_mm,
        eff_height_mm=eff_height_mm,
        interval_m=interval_m,
        fine_threshold_m=fine_threshold_m,
        fine_interval_m=fine_interval_m,
        reversal_split_mm=reversal_split_mm,
        neighbor_radius_mm=neighbor_radius_mm,
        resampling_interval_mm=resampling_interval_mm,
        skip_alpha=skip_alpha,
    )
    if rc is None:
        return None

    rows_full, cols_full = rc
    xs_m = _px_to_m_x(cols_full, img_w, bbox_w_m)
    ys_m = _px_to_m_y(rows_full, img_h, bbox_h_m)
    return {
        "corridor": corridor,
        "rows_full": rows_full.astype(np.float32, copy=False),
        "cols_full": cols_full.astype(np.float32, copy=False),
        "xs_m": xs_m.astype(np.float32, copy=False),
        "ys_m": ys_m.astype(np.float32, copy=False),
        "t_sec": float(time.perf_counter() - t0),
    }


def _neighbor_radius_m_from_context(ctx, neighbor_radius_mm: float) -> float:
    meters_per_mm_w = float(ctx.bbox_w_m) / float(max(1e-9, ctx.eff_width_mm))
    meters_per_mm_h = float(ctx.bbox_h_m) / float(max(1e-9, ctx.eff_height_mm))
    meters_per_mm = 0.5 * (meters_per_mm_w + meters_per_mm_h)
    return float(neighbor_radius_mm) * meters_per_mm


def _corridor_key(corridor: dict) -> tuple[str, int]:
    return (str(corridor["kind"]), int(corridor["i"]))


def _effective_print_scale(args) -> float:
    return float(args.print_scale)


def _effective_nozzle_diameter_mm(args) -> float:
    return float(args.nozzle_diameter_mm) / _effective_print_scale(args)


def _effective_fitting_clearance_mm(args) -> float:
    return float(args.fitting_clearance) / _effective_print_scale(args)


def load_raster_context(args, log: logging.Logger) -> RasterContext:
    """
    Inputs:
    - Parsed CLI arguments and a logger.

    Outputs:
    - Returns the loaded DEM plus all physical sizing metadata needed by the
      corridor and mesh-export pipeline.
    """
    local_crs = choose_local_crs(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    _, _, _, _, bbox_w_m, bbox_h_m = project_bbox_to_local_m(
        args.lat_min, args.lat_max, args.lon_min, args.lon_max, local_crs
    )
    aspect_m = bbox_w_m / float(bbox_h_m) if bbox_h_m else 1.0
    eff_width_mm, eff_height_mm = fit_mm_bbox_preserve_aspect_ratio(
        args.final_width_mm, args.final_height_mm, aspect_m
    )
    premerge_shape = _premerge_output_shape_for_nozzle(
        eff_width_mm=float(eff_width_mm),
        eff_height_mm=float(eff_height_mm),
        nozzle_diameter_mm=_effective_nozzle_diameter_mm(args),
        oversample_factor=2.0,
    )

    with Timer("load elevation_map", log):
        elev_map, _elev_transform = build_elevation_map(
            args.lat_min,
            args.lat_max,
            args.lon_min,
            args.lon_max,
            cache_array=True,
            output_shape=premerge_shape,
        )
        if elev_map is None or elev_map.size == 0:
            raise RuntimeError("build_elevation_map returned an empty array")
        raw_void_count = int(np.sum(np.isfinite(elev_map) & (elev_map <= -32000)))
        raw_negative_count = int(np.sum(np.isfinite(elev_map) & (elev_map < 0)))
        elev_map = _replace_hgt_voids_with_nan(elev_map)
        img_h, img_w = elev_map.shape

    log.info(
        "premerge raster target: %dx%d -> loaded %dx%d",
        int(premerge_shape[0]),
        int(premerge_shape[1]),
        int(img_h),
        int(img_w),
    )
    log.info(
        "merged raster negatives: total=%d, hgt_voids=%d",
        raw_negative_count,
        raw_void_count,
    )

    with Timer("downsample elevation_map", log):
        elev_map, downsample_steps = downsample_elevation_map_for_nozzle(
            elev_map,
            eff_width_mm=float(eff_width_mm),
            eff_height_mm=float(eff_height_mm),
            nozzle_diameter_mm=_effective_nozzle_diameter_mm(args),
        )
    elev_map = _clamp_nonnegative(elev_map)
    img_h, img_w = elev_map.shape
    log.info(
        "downsampled raster: %dx%d (steps=%d, nozzle=%.3fmm, mm/px=%.4f x %.4f)",
        img_h,
        img_w,
        downsample_steps,
        _effective_nozzle_diameter_mm(args),
        float(eff_width_mm) / float(max(1, img_w)),
        float(eff_height_mm) / float(max(1, img_h)),
    )

    return RasterContext(
        elev_map=elev_map,
        img_h=img_h,
        img_w=img_w,
        bbox_w_m=float(bbox_w_m),
        bbox_h_m=float(bbox_h_m),
        eff_width_mm=float(eff_width_mm),
        eff_height_mm=float(eff_height_mm),
    )


def plan_tiling(ctx: RasterContext, args) -> TilingPlan:
    col_starts_mm = plan_tile_starts_mm(ctx.eff_width_mm, args.bed_width_mm, args.overlap_width_mm)
    row_starts_mm = plan_tile_starts_mm(ctx.eff_height_mm, args.bed_height_mm, args.overlap_width_mm)
    # Keep centered tiling in mm for export/cropping, but build corridor windows
    # with the original pixel convention: rounded nominal starts plus a fixed
    # rounded overlap width. The contour solver was tuned against that setup.
    col_starts_px = [_mm_to_px_x(mm, ctx.eff_width_mm, ctx.img_w) for mm in col_starts_mm]
    row_starts_px = [_mm_to_px_y(mm, ctx.eff_height_mm, ctx.img_h) for mm in row_starts_mm]
    overlap_px_x = _mm_to_px_x(args.overlap_width_mm, ctx.eff_width_mm, ctx.img_w)
    overlap_px_y = _mm_to_px_y(args.overlap_width_mm, ctx.eff_height_mm, ctx.img_h)

    corridors = []
    for i in range(1, len(col_starts_px)):
        x0 = int(col_starts_px[i])
        x1 = int(min(ctx.img_w, x0 + overlap_px_x))
        if x1 > x0:
            corridors.append({"kind": "col", "i": i, "x0": x0, "x1": x1, "y0": 0, "y1": ctx.img_h})
    for i in range(1, len(row_starts_px)):
        y0 = int(row_starts_px[i])
        y1 = int(min(ctx.img_h, y0 + overlap_px_y))
        if y1 > y0:
            corridors.append({"kind": "row", "i": i, "x0": 0, "x1": ctx.img_w, "y0": y0, "y1": y1})

    return TilingPlan(
        corridors=corridors,
        col_starts_mm=col_starts_mm,
        row_starts_mm=row_starts_mm,
        col_starts_px=col_starts_px,
        row_starts_px=row_starts_px,
        overlap_px_x=overlap_px_x,
        overlap_px_y=overlap_px_y,
    )


def resolve_worker_count(requested_workers: int) -> int:
    if int(requested_workers) > 0:
        return int(requested_workers)
    return min((os.cpu_count() or 1), 4)


def compute_corridor_lines_parallel(
    ctx: RasterContext,
    plan: TilingPlan,
    args,
    workers: int,
    log: logging.Logger,
) -> list[dict]:
    """
    Inputs:
    - Raster context, tiling plan, parsed args, worker count, and logger.

    Outputs:
    - Returns one solved separating polyline per overlap corridor.
    """
    if not plan.corridors:
        raise RuntimeError("No corridors computed (check bed/overlap/final size settings).")

    interval = float(args.elevation_interval_m)
    if interval <= 0:
        raise ValueError("--elevation-interval-m must be > 0")
    if float(args.elevation_fine_interval_m) <= 0:
        raise ValueError("--elevation-fine-interval-m must be > 0")
    if float(args.resampling_interval_mm) <= 0:
        raise ValueError("--resampling-interval-mm must be > 0")

    tmp_dir = Path(tempfile.gettempdir())
    mmap_path = str(tmp_dir / f"depth_map_elev_{os.getpid()}.mmap")

    with Timer("write memmap", log):
        mm = np.memmap(mmap_path, dtype=ctx.elev_map.dtype, mode="w+", shape=ctx.elev_map.shape)
        mm[:] = ctx.elev_map
        mm.flush()
        del mm

    try:
        job_args = [
            (
                corridor,
                mmap_path,
                ctx.elev_map.shape,
                str(ctx.elev_map.dtype),
                ctx.img_h,
                ctx.img_w,
                float(ctx.bbox_w_m),
                float(ctx.bbox_h_m),
                float(ctx.eff_width_mm),
                float(ctx.eff_height_mm),
                float(interval),
                float(args.elevation_fine_threshold_m),
                float(args.elevation_fine_interval_m),
                float(args.reversal_split_mm),
                float(args.neighbor_radius_mm),
                float(args.resampling_interval_mm),
                float(args.skip_alpha),
            )
            for corridor in plan.corridors
        ]

        results = []
        with Timer("compute corridor lines (multiprocessing)", log):
            ctx_mp = get_context("spawn")
            with ctx_mp.Pool(processes=workers) as pool:
                it = pool.imap_unordered(_worker_build_corridor_line, job_args, chunksize=1)
                for result in tqdm(it, total=len(job_args), desc="Corridors", unit="corridor"):
                    if result is not None:
                        results.append(result)
    finally:
        try:
            os.remove(mmap_path)
        except Exception:
            pass

    if not results:
        raise RuntimeError("No corridor lines were produced.")

    log.info("corridor lines produced: %d / %d", len(results), len(plan.corridors))
    log.info("avg corridor time: %.3fs", float(np.mean([r["t_sec"] for r in results])) if results else 0.0)
    return results


def _line_rc_from_result(result: dict) -> np.ndarray:
    return np.column_stack((result["rows_full"], result["cols_full"])).astype(float, copy=False)


def _result_from_line_rc(result: dict, line_rc: np.ndarray, ctx: RasterContext) -> dict:
    rows_full = np.asarray(line_rc[:, 0], dtype=np.float32)
    cols_full = np.asarray(line_rc[:, 1], dtype=np.float32)
    return {
        "corridor": result["corridor"],
        "rows_full": rows_full,
        "cols_full": cols_full,
        "xs_m": _px_to_m_x(cols_full, ctx.img_w, ctx.bbox_w_m).astype(np.float32, copy=False),
        "ys_m": _px_to_m_y(rows_full, ctx.img_h, ctx.bbox_h_m).astype(np.float32, copy=False),
        "t_sec": result.get("t_sec", 0.0),
    }


def _find_inside_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if cur == prev + 1:
            prev = cur
            continue
        runs.append((start, prev))
        start = cur
        prev = cur
    runs.append((start, prev))
    return runs


def extract_line_section_in_square(
    line_rc: np.ndarray,
    *,
    center_mm: np.ndarray,
    half_width_mm: float,
    mm_per_px_x: float,
    mm_per_px_y: float,
    axis_idx: int,
) -> Optional[dict]:
    if line_rc.shape[0] < 2:
        return None

    pts_mm = np.column_stack((line_rc[:, 1] * mm_per_px_x, line_rc[:, 0] * mm_per_px_y))
    delta_mm = np.abs(pts_mm - center_mm[None, :])
    inside = (delta_mm[:, 0] <= float(half_width_mm)) & (delta_mm[:, 1] <= float(half_width_mm))
    runs = _find_inside_runs(inside)
    if not runs:
        return None

    best = None
    best_score = (-1.0, -1)
    for start, end in runs:
        ext_start = max(0, start - 1)
        ext_end = min(line_rc.shape[0] - 1, end + 1)
        section = line_rc[ext_start : ext_end + 1]
        axis_extent = float(np.max(section[:, axis_idx]) - np.min(section[:, axis_idx]))
        score = (axis_extent, section.shape[0])
        if score > best_score:
            best_score = score
            best = {
                "start_idx": ext_start,
                "end_idx": ext_end,
                "section_rc": np.asarray(section, dtype=float),
            }
    return best


def _orient_replacement_path(section_rc: np.ndarray, replacement_rc: np.ndarray) -> np.ndarray:
    if replacement_rc.shape[0] < 2:
        return replacement_rc
    d_forward = float(np.linalg.norm(replacement_rc[0] - section_rc[0])) + float(
        np.linalg.norm(replacement_rc[-1] - section_rc[-1])
    )
    d_reverse = float(np.linalg.norm(replacement_rc[-1] - section_rc[0])) + float(
        np.linalg.norm(replacement_rc[0] - section_rc[-1])
    )
    return replacement_rc if d_forward <= d_reverse else replacement_rc[::-1].copy()


def _node_id_for_point(key_to_id: dict[tuple[int, int], int], point_rc: np.ndarray, quantize_q: float = 2.0) -> Optional[int]:
    return key_to_id.get(_quantize_rc(point_rc, quantize_q))


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))


def _path_edge_keys(path_ids: list[int]) -> set[tuple[int, int]]:
    return {_edge_key(path_ids[i - 1], path_ids[i]) for i in range(1, len(path_ids)) if path_ids[i - 1] != path_ids[i]}


def _polyline_node_ids(
    key_to_id: dict[tuple[int, int], int],
    polyline_rc: np.ndarray,
    quantize_q: float = 2.0,
) -> list[int]:
    out: list[int] = []
    for point_rc in np.asarray(polyline_rc, dtype=float):
        nid = _node_id_for_point(key_to_id, point_rc, quantize_q=quantize_q)
        if nid is None:
            continue
        if not out or out[-1] != int(nid):
            out.append(int(nid))
    return out


def _edge_length_map(adj: dict[int, list[tuple[int, float]]]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for u, neighbors in adj.items():
        for v, w in neighbors:
            key = _edge_key(u, v)
            if key not in out:
                out[key] = float(w)
    return out


def _cycle_edge_keys(adj: dict[int, list[tuple[int, float]]]) -> set[tuple[int, int]]:
    nbrs: dict[int, set[int]] = {int(u): {int(v) for v, _w in neighbors} for u, neighbors in adj.items()}
    deg = {u: len(vs) for u, vs in nbrs.items()}
    queue = deque(u for u, d in deg.items() if d < 2)

    while queue:
        u = int(queue.popleft())
        if deg.get(u, 0) >= 2:
            continue
        for v in list(nbrs.get(u, set())):
            nbrs[v].discard(u)
            deg[v] = len(nbrs[v])
            if deg[v] == 1:
                queue.append(v)
        nbrs[u].clear()
        deg[u] = 0

    cycle_edges: set[tuple[int, int]] = set()
    for u, vs in nbrs.items():
        for v in vs:
            if u < v:
                cycle_edges.add((u, v))
    return cycle_edges


def _node_set_from_edge_keys(edge_keys: set[tuple[int, int]]) -> set[int]:
    nodes: set[int] = set()
    for u, v in edge_keys:
        nodes.add(int(u))
        nodes.add(int(v))
    return nodes


def _build_adj_from_edge_keys(
    edge_keys: set[tuple[int, int]],
    edge_lengths: dict[tuple[int, int], float],
) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = {}
    for u, v in edge_keys:
        w = float(edge_lengths[_edge_key(u, v)])
        adj.setdefault(int(u), []).append((int(v), w))
        adj.setdefault(int(v), []).append((int(u), w))
    return adj


def _cycle_components_from_edge_keys(
    edge_keys: set[tuple[int, int]],
    edge_lengths: dict[tuple[int, int], float],
) -> list[tuple[set[int], set[tuple[int, int]]]]:
    if not edge_keys:
        return []
    adj = _build_adj_from_edge_keys(edge_keys, edge_lengths)
    cycle_edges = _cycle_edge_keys(adj)
    if not cycle_edges:
        return []
    cycle_adj = _build_adj_from_edge_keys(cycle_edges, edge_lengths)
    comps: list[tuple[set[int], set[tuple[int, int]]]] = []
    for comp_nodes_list in connected_components(cycle_adj):
        comp_nodes = {int(u) for u in comp_nodes_list}
        comp_edges = {edge for edge in cycle_edges if int(edge[0]) in comp_nodes and int(edge[1]) in comp_nodes}
        if comp_edges:
            comps.append((comp_nodes, comp_edges))
    return comps


def _build_quadrant_replacement_paths_from_graph(
    coords: list[np.ndarray],
    adj: dict[int, list[tuple[int, float]]],
    key_to_id: dict[tuple[int, int], int],
    vertical_section: np.ndarray,
    horizontal_section: np.ndarray,
    *,
    ctx,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    top_pt = vertical_section[int(np.argmin(vertical_section[:, 0]))]
    bottom_pt = vertical_section[int(np.argmax(vertical_section[:, 0]))]
    left_pt = horizontal_section[int(np.argmin(horizontal_section[:, 1]))]
    right_pt = horizontal_section[int(np.argmax(horizontal_section[:, 1]))]

    top_id = _node_id_for_point(key_to_id, top_pt)
    bottom_id = _node_id_for_point(key_to_id, bottom_pt)
    left_id = _node_id_for_point(key_to_id, left_pt)
    right_id = _node_id_for_point(key_to_id, right_pt)
    if None in (top_id, bottom_id, left_id, right_id):
        return None

    quadrant_specs = [
        ("tr", int(top_id), int(right_id)),
        ("rb", int(right_id), int(bottom_id)),
        ("bl", int(bottom_id), int(left_id)),
        ("lt", int(left_id), int(top_id)),
    ]
    quadrant_polylines: dict[str, np.ndarray] = {}
    for label, start_id, end_id in quadrant_specs:
        path_ids = dijkstra_path(adj, start_id, end_id)
        if not path_ids:
            return None
        quadrant_polylines[label] = np.array([coords[n] for n in path_ids], dtype=float)

    border_coords, border_adj, border_key_to_id = build_graph_from_polylines(
        [quadrant_polylines[label] for label, _start_id, _end_id in quadrant_specs],
        mppx=ctx.mppx,
        mppy=ctx.mppy,
        quantize_q=2.0,
        resample_step_m=0.0,
    )
    if not border_coords:
        return None

    top_border_id = _node_id_for_point(border_key_to_id, top_pt)
    bottom_border_id = _node_id_for_point(border_key_to_id, bottom_pt)
    left_border_id = _node_id_for_point(border_key_to_id, left_pt)
    right_border_id = _node_id_for_point(border_key_to_id, right_pt)
    if None in (top_border_id, bottom_border_id, left_border_id, right_border_id):
        return None

    quadrant_edge_keys: dict[str, set[tuple[int, int]]] = {}
    for label, polyline_rc in quadrant_polylines.items():
        path_ids = _polyline_node_ids(border_key_to_id, polyline_rc, quantize_q=2.0)
        if len(path_ids) < 2:
            return None
        quadrant_edge_keys[label] = _path_edge_keys(path_ids)

    edge_lengths = _edge_length_map(border_adj)
    protected_nodes = {int(top_border_id), int(bottom_border_id), int(left_border_id), int(right_border_id)}

    while True:
        combined_edge_keys: set[tuple[int, int]] = set()
        for edge_keys in quadrant_edge_keys.values():
            combined_edge_keys.update(edge_keys)

        loop_components = [
            (loop_nodes, loop_edges)
            for (loop_nodes, loop_edges) in _cycle_components_from_edge_keys(combined_edge_keys, edge_lengths)
            if not (loop_nodes & protected_nodes)
        ]
        if not loop_components:
            break

        progress = False
        for loop_nodes, loop_edges in loop_components:
            best_label = None
            best_shared_vertices = -1
            best_shared_len = -1.0
            for label, path_edges in quadrant_edge_keys.items():
                shared_edges = loop_edges & path_edges
                shared_vertices = len(loop_nodes & _node_set_from_edge_keys(path_edges))
                shared_len = float(sum(edge_lengths.get(edge, 0.0) for edge in shared_edges))
                if shared_vertices > best_shared_vertices or (
                    shared_vertices == best_shared_vertices and shared_len > best_shared_len
                ):
                    best_label = label
                    best_shared_vertices = shared_vertices
                    best_shared_len = shared_len

            if best_label is None or best_shared_vertices <= 0:
                return None

            quadrant_edge_keys[best_label] = quadrant_edge_keys[best_label] ^ loop_edges
            progress = True

        if not progress:
            return None

    final_edge_keys: set[tuple[int, int]] = set()
    for edge_keys in quadrant_edge_keys.values():
        final_edge_keys.update(edge_keys)
    final_adj = _build_adj_from_edge_keys(final_edge_keys, edge_lengths)
    if any(node_id not in final_adj for node_id in protected_nodes):
        return None

    vertical_path_ids = dijkstra_path(final_adj, int(top_border_id), int(bottom_border_id))
    horizontal_path_ids = dijkstra_path(final_adj, int(left_border_id), int(right_border_id))
    if not vertical_path_ids or not horizontal_path_ids:
        return None

    vertical_path = np.array([border_coords[n] for n in vertical_path_ids], dtype=float)
    horizontal_path = np.array([border_coords[n] for n in horizontal_path_ids], dtype=float)
    return vertical_path, horizontal_path


def _build_local_replacement_paths(
    vertical_section: np.ndarray,
    horizontal_section: np.ndarray,
    *,
    ctx,
    neighbor_radius_mm: float,
    resampling_interval_mm: float,
    skip_alpha: float,
    graph_obj_path: Optional[str] = None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    resample_step_m = _neighbor_radius_m_from_context(ctx, resampling_interval_mm)
    coords, adj, key_to_id = build_graph_from_polylines(
        [vertical_section, horizontal_section],
        mppx=ctx.mppx,
        mppy=ctx.mppy,
        quantize_q=2.0,
        resample_step_m=resample_step_m,
    )
    if not coords:
        return None

    neighbor_radius_m = _neighbor_radius_m_from_context(ctx, neighbor_radius_mm)
    add_proximity_edges_within_component_radius(
        coords,
        adj,
        node_ids=list(adj.keys()),
        mppx=ctx.mppx,
        mppy=ctx.mppy,
        radius_m=neighbor_radius_m,
        alpha=float(skip_alpha),
        resample_step_m=resample_step_m,
    )

    connect_components_by_nearest_mst(
        coords,
        adj,
        mppx=ctx.mppx,
        mppy=ctx.mppy,
        resample_step_m=resample_step_m,
    )

    if graph_obj_path:
        save_graph_obj(
            graph_obj_path,
            coords,
            adj,
            mm_per_px_x=ctx.mm_per_px_x,
            mm_per_px_y=ctx.mm_per_px_y,
        )

    quadrant_paths = _build_quadrant_replacement_paths_from_graph(
        coords,
        adj,
        key_to_id,
        vertical_section,
        horizontal_section,
        ctx=ctx,
    )
    return quadrant_paths


def _intersection_replacement_task(task: dict) -> Optional[dict]:
    col_result = task["col_result"]
    row_result = task["row_result"]
    center_mm = np.asarray(task["center_mm"], dtype=float)
    half_width_mm = float(task["half_width_mm"])
    ctx = task["ctx"]

    col_line_rc = _line_rc_from_result(col_result)
    row_line_rc = _line_rc_from_result(row_result)

    col_section = extract_line_section_in_square(
        col_line_rc,
        center_mm=center_mm,
        half_width_mm=half_width_mm,
        mm_per_px_x=ctx.mm_per_px_x,
        mm_per_px_y=ctx.mm_per_px_y,
        axis_idx=0,
    )
    row_section = extract_line_section_in_square(
        row_line_rc,
        center_mm=center_mm,
        half_width_mm=half_width_mm,
        mm_per_px_x=ctx.mm_per_px_x,
        mm_per_px_y=ctx.mm_per_px_y,
        axis_idx=1,
    )
    if col_section is None or row_section is None:
        return None

    local_paths = _build_local_replacement_paths(
        col_section["section_rc"],
        row_section["section_rc"],
        ctx=ctx,
        neighbor_radius_mm=float(task["neighbor_radius_mm"]),
        resampling_interval_mm=float(task["resampling_interval_mm"]),
        skip_alpha=float(task["skip_alpha"]),
        graph_obj_path=task.get("graph_obj_path"),
    )
    if local_paths is None:
        return None

    col_replacement = _orient_replacement_path(col_section["section_rc"], local_paths[0])
    row_replacement = _orient_replacement_path(row_section["section_rc"], local_paths[1])

    return {
        "col_key": task["col_key"],
        "row_key": task["row_key"],
        "col_replacement": {
            "start_idx": int(col_section["start_idx"]),
            "end_idx": int(col_section["end_idx"]),
            "replacement_rc": col_replacement,
        },
        "row_replacement": {
            "start_idx": int(row_section["start_idx"]),
            "end_idx": int(row_section["end_idx"]),
            "replacement_rc": row_replacement,
        },
    }


def _concat_polyline_parts(parts: list[np.ndarray]) -> np.ndarray:
    clean_parts = [np.asarray(part, dtype=float) for part in parts if part is not None and part.size > 0]
    if not clean_parts:
        return np.zeros((0, 2), dtype=float)
    out = [clean_parts[0]]
    for part in clean_parts[1:]:
        cur = part
        if np.allclose(out[-1][-1], cur[0]):
            cur = cur[1:]
        if cur.size > 0:
            out.append(cur)
    return np.vstack(out)


def apply_replacements_to_line(line_rc: np.ndarray, replacements: list[dict]) -> np.ndarray:
    if not replacements:
        return np.asarray(line_rc, dtype=float)

    current = np.asarray(line_rc, dtype=float)
    offset = 0
    for repl in sorted(replacements, key=lambda item: item["start_idx"]):
        start_idx = int(repl["start_idx"]) + offset
        end_idx = int(repl["end_idx"]) + offset
        if start_idx < 0 or end_idx >= current.shape[0] or start_idx > end_idx:
            continue
        replacement_rc = _orient_replacement_path(current[start_idx : end_idx + 1], np.asarray(repl["replacement_rc"], dtype=float))
        current = _concat_polyline_parts(
            [
                current[:start_idx],
                replacement_rc,
                current[end_idx + 1 :],
            ]
        )
        offset += replacement_rc.shape[0] - (end_idx - start_idx + 1)
    return current


def refine_intersections_parallel(
    results: list[dict],
    ctx: RasterContext,
    args,
    workers: int,
    log: logging.Logger,
) -> list[dict]:
    col_results = [r for r in results if r["corridor"]["kind"] == "col"]
    row_results = [r for r in results if r["corridor"]["kind"] == "row"]
    if not col_results or not row_results:
        return results

    graph_out_dir = (Path(args.mesh_out_dir).expanduser() if str(args.mesh_out_dir).strip() else Path.cwd()) / "intersection_graphs"
    graph_out_dir.mkdir(parents=True, exist_ok=True)

    metric_ctx = IntersectionMetrics(
        img_h=ctx.img_h,
        img_w=ctx.img_w,
        bbox_w_m=ctx.bbox_w_m,
        bbox_h_m=ctx.bbox_h_m,
        eff_width_mm=ctx.eff_width_mm,
        eff_height_mm=ctx.eff_height_mm,
        mm_per_px_x=ctx.mm_per_px_x,
        mm_per_px_y=ctx.mm_per_px_y,
        mppx=ctx.mppx,
        mppy=ctx.mppy,
    )

    tasks = []
    for col_result in col_results:
        col_corridor = col_result["corridor"]
        center_x_mm = 0.5 * (float(col_corridor["x0"]) + float(col_corridor["x1"])) * ctx.mm_per_px_x
        for row_result in row_results:
            row_corridor = row_result["corridor"]
            center_y_mm = 0.5 * (float(row_corridor["y0"]) + float(row_corridor["y1"])) * ctx.mm_per_px_y
            graph_obj_path = graph_out_dir / f"crossing_col_{int(col_corridor['i']):02d}_row_{int(row_corridor['i']):02d}.obj"
            tasks.append(
                {
                    "col_key": _corridor_key(col_corridor),
                    "row_key": _corridor_key(row_corridor),
                    "col_result": col_result,
                    "row_result": row_result,
                    "center_mm": np.array([center_x_mm, center_y_mm], dtype=float),
                    "half_width_mm": 0.5 * float(args.overlap_width_mm),
                    "neighbor_radius_mm": float(args.neighbor_radius_mm),
                    "resampling_interval_mm": float(args.resampling_interval_mm),
                    "skip_alpha": float(args.skip_alpha),
                    "graph_obj_path": str(graph_obj_path),
                    "ctx": metric_ctx,
                }
            )

    if not tasks:
        return results

    processes = max(1, min(len(tasks), max(1, workers)))
    replacements_by_key: dict[tuple[str, int], list[dict]] = {}
    applied_pairs = 0

    with Timer("refine intersections (multiprocessing)", log):
        ctx_mp = get_context("spawn")
        with ctx_mp.Pool(processes=processes) as pool:
            it = pool.imap_unordered(_intersection_replacement_task, tasks, chunksize=1)
            for replacement in tqdm(it, total=len(tasks), desc="Intersections", unit="pair"):
                if replacement is None:
                    continue
                replacements_by_key.setdefault(replacement["col_key"], []).append(replacement["col_replacement"])
                replacements_by_key.setdefault(replacement["row_key"], []).append(replacement["row_replacement"])
                applied_pairs += 1

    log.info("intersection replacements produced: %d / %d", applied_pairs, len(tasks))
    log.info("intersection search graphs written to: %s", str(graph_out_dir))

    refined_results = []
    for result in results:
        line_rc = _line_rc_from_result(result)
        key = _corridor_key(result["corridor"])
        replacements = replacements_by_key.get(key, [])
        refined_line = apply_replacements_to_line(line_rc, replacements)
        refined_results.append(_result_from_line_rc(result, refined_line, ctx))

    return refined_results


def plot_corridor_lines(results: list[dict], ctx: RasterContext, args):
    fig, axp = plt.subplots(figsize=(12, 6))
    axp.imshow(ctx.elev_map, cmap="terrain", origin="upper", extent=ctx.extent_m)
    axp.set_aspect("equal", adjustable="box")
    for result in results:
        axp.plot(result["xs_m"], result["ys_m"], color="black", linewidth=1.0)
    axp.set_title(
        f"All corridors | interval={float(args.elevation_interval_m):.0f} m | neighbor_r={float(args.neighbor_radius_mm):.1f} mm | "
        f"resample={float(args.resampling_interval_mm):.1f} mm | alpha={float(args.skip_alpha):.2f} | lines={len(results)}"
    )
    axp.set_axis_off()
    plt.tight_layout()


def _tile_bounds_mm(
    ctx: RasterContext,
    args,
    col_starts_mm: list[float],
    row_starts_mm: list[float],
    tx: int,
    ty: int,
) -> tuple[float, float, float, float]:
    x0_mm, x1_mm = _clip_tile_interval_mm(float(col_starts_mm[tx]), float(args.bed_width_mm), float(ctx.eff_width_mm))
    y0_mm, y1_mm = _clip_tile_interval_mm(float(row_starts_mm[ty]), float(args.bed_height_mm), float(ctx.eff_height_mm))
    return x0_mm, x1_mm, y0_mm, y1_mm


def _tile_crop_bounds_px(
    ctx: RasterContext,
    args,
    tile_bounds_mm: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x0_mm, x1_mm, y0_mm, y1_mm = tile_bounds_mm
    pad_mm = float(args.overlap_width_mm) + max(
        float(args.resampling_interval_mm),
        _effective_fitting_clearance_mm(args),
        1.0,
    )
    crop_x0_mm = max(0.0, x0_mm - pad_mm)
    crop_x1_mm = min(float(ctx.eff_width_mm), x1_mm + pad_mm)
    crop_y0_mm = max(0.0, y0_mm - pad_mm)
    crop_y1_mm = min(float(ctx.eff_height_mm), y1_mm + pad_mm)

    x0_px = max(0, int(np.floor(crop_x0_mm / float(ctx.mm_per_px_x))))
    x1_px = min(int(ctx.img_w), int(np.ceil(crop_x1_mm / float(ctx.mm_per_px_x))) + 1)
    y0_px = max(0, int(np.floor(crop_y0_mm / float(ctx.mm_per_px_y))))
    y1_px = min(int(ctx.img_h), int(np.ceil(crop_y1_mm / float(ctx.mm_per_px_y))) + 1)
    return x0_px, x1_px, y0_px, y1_px


def _parse_export_tile_selection(export_meshes: Optional[list[str]], nx: int, ny: int) -> Optional[set[tuple[int, int]]]:
    if export_meshes is None or len(export_meshes) == 0:
        return None

    selected: set[tuple[int, int]] = set()
    for raw in export_meshes:
        token = str(raw).strip()
        if token.startswith("tile_"):
            token = token[5:]
        parts = token.split("_")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Invalid tile selector '{raw}'. Use TY_TX, for example 02_03.")
        ty = int(parts[0])
        tx = int(parts[1])
        if ty < 0 or ty >= int(ny) or tx < 0 or tx >= int(nx):
            raise ValueError(
                f"Tile selector '{raw}' is out of range. Valid rows: 00..{int(ny) - 1:02d}, cols: 00..{int(nx) - 1:02d}."
            )
        selected.add((tx, ty))
    return selected


def _worker_export_tile_mesh(args):
    """
    Inputs:
    - Serialized tile export task arguments, including one DEM crop and the
      neighboring cutline geometry needed to isolate a single tile.

    Outputs:
    - Returns a small status dict describing a successful export, empty result,
      or failure for one tile.
    """
    (
        tx,
        ty,
        out_path,
        mmap_path,
        shape,
        dtype_str,
        crop_x0_px,
        crop_x1_px,
        crop_y0_px,
        crop_y1_px,
        mm_per_px_x,
        mm_per_px_y,
        bottom_thickness_mm,
        desired_height_mm,
        height_exponent,
        norm_h_min,
        norm_h_max,
        total_height_mm,
        print_scale,
        z_min,
        z_max,
        top_line,
        bot_line,
        left_line,
        right_line,
    ) = args

    t0 = time.perf_counter()
    elev_map = np.memmap(mmap_path, dtype=np.dtype(dtype_str), mode="r", shape=shape)
    crop = elev_map[crop_y0_px:crop_y1_px, crop_x0_px:crop_x1_px]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return {"status": "error", "tx": tx, "ty": ty, "error": f"crop too small: {crop.shape}"}

    try:
        local_mesh = generate_full_mesh(
            crop,
            mm_per_px_x=float(mm_per_px_x),
            mm_per_px_y=float(mm_per_px_y),
            bottom_thickness_mm=float(bottom_thickness_mm),
            desired_height_mm=float(desired_height_mm),
            height_exponent=float(height_exponent),
            norm_h_min=float(norm_h_min),
            norm_h_max=float(norm_h_max),
            origin_x_mm=float(crop_x0_px) * float(mm_per_px_x),
            origin_y_mm=float(crop_y0_px) * float(mm_per_px_y),
        )
        # Cut sequentially: first isolate the requested horizontal strip, then
        # isolate the matching column strip inside that slice.
        row_tool = create_extruded_tool(np.vstack([top_line, bot_line[::-1]]), float(z_min), float(z_max))
        row_slice = _mesh_intersection(local_mesh, row_tool, preferred_engine="manifold")
        if row_slice.is_empty:
            return {"status": "empty", "tx": tx, "ty": ty}

        col_tool = create_extruded_tool(np.vstack([left_line, right_line[::-1]]), float(z_min), float(z_max))
        tile_mesh = _mesh_intersection(row_slice, col_tool, preferred_engine="manifold")
        if tile_mesh.is_empty:
            return {"status": "empty", "tx": tx, "ty": ty}

        tile_mesh = _flip_mesh_across_x_axis(tile_mesh, float(total_height_mm))
        if abs(float(print_scale) - 1.0) > 1e-9:
            tile_mesh.apply_scale([float(print_scale), float(print_scale), float(print_scale)])
        tile_mesh.export(str(out_path))
        return {
            "status": "ok",
            "tx": tx,
            "ty": ty,
            "path": str(out_path),
            "t_sec": float(time.perf_counter() - t0),
        }
    except Exception as exc:
        return {"status": "error", "tx": tx, "ty": ty, "error": str(exc)}
    finally:
        del crop
        del elev_map


def export_tile_meshes(
    results: list[dict],
    ctx: RasterContext,
    plan: TilingPlan,
    args,
    log: logging.Logger,
):
    """
    Inputs:
    - Solved corridor lines, raster context, tiling plan, parsed args, and logger.

    Outputs:
    - Writes the selected tile meshes to disk and logs progress.
    """
    out_dir = Path(args.mesh_out_dir).expanduser() if str(args.mesh_out_dir).strip() else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    lines_mm = []
    for result in results:
        pts_mm = np.column_stack((result["cols_full"] * ctx.mm_per_px_x, result["rows_full"] * ctx.mm_per_px_y))
        lines_mm.append(_flip_points_across_x_axis(pts_mm, float(ctx.eff_height_mm)))
    z_lines = args.bottom_thickness_mm + args.desired_height_mm + 0.5
    save_lines_obj(str(out_dir / "corridor_lines.obj"), lines_mm, z=z_lines)
    log.info("Exported corridor_lines.obj")

    col_lines: dict[int, np.ndarray] = {}
    row_lines: dict[int, np.ndarray] = {}
    for result in results:
        corridor = result["corridor"]
        pts_mm = np.column_stack((result["cols_full"] * ctx.mm_per_px_x, result["rows_full"] * ctx.mm_per_px_y))
        if corridor["kind"] == "col":
            col_lines[int(corridor["i"])] = pts_mm
        else:
            row_lines[int(corridor["i"])] = pts_mm

    z_min = -1.0
    z_max = args.bottom_thickness_mm + args.desired_height_mm + 1.0
    nx = len(plan.col_starts_px)
    ny = len(plan.row_starts_px)
    global_h_min = float(np.nanmin(ctx.elev_map))
    global_h_max = float(np.nanmax(ctx.elev_map))
    if not np.isfinite(global_h_min) or not np.isfinite(global_h_max):
        raise RuntimeError("Cannot export meshes because the global elevation range is invalid")
    col_starts_mm = plan.col_starts_mm
    row_starts_mm = plan.row_starts_mm
    half_clearance = _effective_fitting_clearance_mm(args) / 2.0
    selected_tiles = _parse_export_tile_selection(args.export_meshes, nx, ny)

    missing_cols = [idx for idx in range(1, nx) if idx not in col_lines]
    missing_rows = [idx for idx in range(1, ny) if idx not in row_lines]
    if missing_cols or missing_rows:
        raise RuntimeError(
            f"Cannot export meshes because corridor lines are missing: cols={missing_cols}, rows={missing_rows}"
        )

    tasks = []
    for ty in range(ny):
        top_line = (
            np.array([[0.0, 0.0], [float(ctx.eff_width_mm), 0.0]], dtype=float)
            if ty == 0
            else offset_polyline(row_lines[ty], -half_clearance, constrain_axis=0)
        )
        bot_line = (
            np.array([[0.0, float(ctx.eff_height_mm)], [float(ctx.eff_width_mm), float(ctx.eff_height_mm)]], dtype=float)
            if ty == ny - 1
            else offset_polyline(row_lines[ty + 1], half_clearance, constrain_axis=0)
        )
        for tx in range(nx):
            if selected_tiles is not None and (tx, ty) not in selected_tiles:
                continue
            left_line = (
                np.array([[0.0, 0.0], [0.0, float(ctx.eff_height_mm)]], dtype=float)
                if tx == 0
                else offset_polyline(col_lines[tx], half_clearance, constrain_axis=1)
            )
            right_line = (
                np.array([[float(ctx.eff_width_mm), 0.0], [float(ctx.eff_width_mm), float(ctx.eff_height_mm)]], dtype=float)
                if tx == nx - 1
                else offset_polyline(col_lines[tx + 1], -half_clearance, constrain_axis=1)
            )
            tile_bounds_mm = _tile_bounds_mm(ctx, args, col_starts_mm, row_starts_mm, tx, ty)
            crop_x0_px, crop_x1_px, crop_y0_px, crop_y1_px = _tile_crop_bounds_px(ctx, args, tile_bounds_mm)
            out_path = out_dir / f"tile_{ty:02d}_{tx:02d}.stl"
            tasks.append(
                (
                    tx,
                    ty,
                    str(out_path),
                    None,
                    None,
                    str(ctx.elev_map.dtype),
                    crop_x0_px,
                    crop_x1_px,
                    crop_y0_px,
                    crop_y1_px,
                    float(ctx.mm_per_px_x),
                    float(ctx.mm_per_px_y),
                    float(args.bottom_thickness_mm),
                    float(args.desired_height_mm),
                    float(args.height_exponent),
                    float(global_h_min),
                    float(global_h_max),
                    float(ctx.eff_height_mm),
                    _effective_print_scale(args),
                    float(z_min),
                    float(z_max),
                    np.asarray(top_line, dtype=np.float32),
                    np.asarray(bot_line, dtype=np.float32),
                    np.asarray(left_line, dtype=np.float32),
                    np.asarray(right_line, dtype=np.float32),
                )
            )

    export_workers = max(1, min(int(args.mesh_export_workers), len(tasks)))
    if selected_tiles is None:
        log.info("exporting all tiles")
    else:
        chosen = ", ".join(f"{ty:02d}_{tx:02d}" for tx, ty in sorted(selected_tiles, key=lambda item: (item[1], item[0])))
        log.info("exporting selected tiles: %s", chosen)
    log.info("export tiles via local crops: %d tasks, workers=%d", len(tasks), export_workers)
    if not tasks:
        log.warning("No tile export tasks were generated")
        return

    tmp_dir = Path(tempfile.gettempdir())
    mmap_path = str(tmp_dir / f"depth_map_mesh_{os.getpid()}.mmap")
    with Timer("write export memmap", log):
        mm = np.memmap(mmap_path, dtype=ctx.elev_map.dtype, mode="w+", shape=ctx.elev_map.shape)
        mm[:] = ctx.elev_map
        mm.flush()
        del mm

    try:
        job_args = [
            (
                tx,
                ty,
                out_path,
                mmap_path,
                ctx.elev_map.shape,
                str(ctx.elev_map.dtype),
                crop_x0_px,
                crop_x1_px,
                crop_y0_px,
                crop_y1_px,
                mm_per_px_x,
                mm_per_px_y,
                bottom_thickness_mm,
                desired_height_mm,
                height_exponent,
                norm_h_min,
                norm_h_max,
                total_height_mm,
                print_scale,
                z_min,
                z_max,
                top_line,
                bot_line,
                left_line,
                right_line,
            )
            for (
                tx,
                ty,
                out_path,
                _mmap_path,
                _shape,
                _dtype_str,
                crop_x0_px,
                crop_x1_px,
                crop_y0_px,
                crop_y1_px,
                mm_per_px_x,
                mm_per_px_y,
                bottom_thickness_mm,
                desired_height_mm,
                height_exponent,
                norm_h_min,
                norm_h_max,
                total_height_mm,
                print_scale,
                z_min,
                z_max,
                top_line,
                bot_line,
                left_line,
                right_line,
            ) in tasks
        ]

        with Timer("process tiles (boolean, multiprocessing)", log):
            ctx_mp = get_context("spawn")
            with ctx_mp.Pool(processes=export_workers) as pool:
                it = pool.imap_unordered(_worker_export_tile_mesh, job_args, chunksize=1)
                for result in tqdm(it, total=len(job_args), desc="Export tiles", unit="tile"):
                    status = str(result.get("status", "error"))
                    if status == "ok":
                        log.info(
                            "Exported %s (tile %02d_%02d, %.2fs)",
                            result["path"],
                            int(result["ty"]),
                            int(result["tx"]),
                            float(result["t_sec"]),
                        )
                    elif status == "empty":
                        log.warning("Tile %02d_%02d produced an empty mesh", int(result["ty"]), int(result["tx"]))
                    else:
                        log.error(
                            "Failed to export tile %02d_%02d: %s",
                            int(result["ty"]),
                            int(result["tx"]),
                            result.get("error", "unknown error"),
                        )
    finally:
        try:
            os.remove(mmap_path)
        except OSError:
            pass

    log.info("Meshes written to: %s", str(out_dir))


def main():
    """Run the full DEM -> corridor -> optional tile-mesh export pipeline."""
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("depth_map")
    args = parse_args()
    if float(args.resampling_interval_mm) <= 0:
        raise ValueError("--resampling-interval-mm must be > 0")
    if float(args.print_scale) <= 0:
        raise ValueError("--print-scale must be > 0")
    if float(args.height_exponent) <= 0:
        raise ValueError("--height-exponent must be > 0")
    if int(args.mesh_export_workers) <= 0:
        raise ValueError("--mesh-export-workers must be > 0")
    ctx = load_raster_context(args, log)
    plan = plan_tiling(ctx, args)
    workers = resolve_worker_count(args.workers)
    results = compute_corridor_lines_parallel(ctx, plan, args, workers, log)
    results = refine_intersections_parallel(results, ctx, args, workers, log)
    plot_corridor_lines(results, ctx, args)

    if args.export_meshes is not None:
        export_tile_meshes(results, ctx, plan, args, log)
        plt.close("all")
        return

    plt.show()


if __name__ == "__main__":
    main()
