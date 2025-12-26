from gpx2pdf import build_elevation_map
import argparse
import numpy as np
import heapq
import os
import tempfile
from pathlib import Path
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
    start_0 = 0
    start_i = start_{i-1} + bed_mm - overlap_mm
    Stop once the last tile's end >= total_mm.
    """
    if bed_mm <= 0:
        raise ValueError("bed dimension must be > 0")
    if overlap_mm < 0:
        raise ValueError("overlap_width_mm must be >= 0")
    if overlap_mm >= bed_mm:
        raise ValueError("overlap_width_mm must be < bed dimension")

    starts = [0.0]
    step = bed_mm - overlap_mm
    # Guard against infinite loops due to floating error
    for _ in range(1, 10_000):
        last_start = starts[-1]
        if last_start + bed_mm >= total_mm:
            break
        starts.append(last_start + step)
    return starts


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
    ap.add_argument("--lat-min", default=46.0, type=float)
    ap.add_argument("--lat-max", default=46.5, type=float)
    ap.add_argument("--lon-min", default=8.0, type=float)
    ap.add_argument("--lon-max", default=8.5, type=float)

    # all physical parameters in mm
    ap.add_argument("--final-width-mm",   default=1500, type=float, help="Assembled final print width (mm)")
    ap.add_argument("--final-height-mm",  default=1500,  type=float, help="Assembled final print height (mm)")
    ap.add_argument("--bed-width-mm",     default=300,  type=float, help="Printer bed width (mm)")
    ap.add_argument("--bed-height-mm",    default=300,  type=float, help="Printer bed height (mm)")
    ap.add_argument("--overlap-width-mm", default=50,   type=float, help="Overlap width (mm)")
    ap.add_argument("--elevation-interval-m", default=200.0, type=float, help="Contour interval in meters")
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
        default=8,
        type=int,
        help="Number of worker processes (0 => conservative default). One corridor is processed per worker.",
    )
    ap.add_argument(
        "--export-meshes",
        action="store_true",
        help="If set, export one STL mesh per tile region.",
    )
    ap.add_argument(
        "--mesh-out-dir",
        default="",
        type=str,
        help="Output directory for STL meshes (default: current working directory).",
    )
    ap.add_argument(
        "--bottom-thickness-mm",
        default=1.5,
        type=float,
        help="Extra solid thickness added below the minimum height (mm).",
    )
    ap.add_argument(
        "--desired-height-mm",
        default=20.0,
        type=float,
        help="Relief height above the bottom thickness (mm). Heights are normalized into [bottom_thickness_mm, bottom_thickness_mm + desired_height_mm] after the nonlinear mapping.",
    )
    ap.add_argument(
        "--fitting-clearance",
        default=0.2,
        type=float,
        help="Total clearance gap between tiles (mm). Edges are offset by half this amount.",
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


def _px_to_m_x(x_px: np.ndarray, img_w: int, bbox_w_m: float) -> np.ndarray:
    denom = float(max(1, img_w - 1))
    return (np.asarray(x_px, dtype=float) / denom) * float(bbox_w_m)


def _px_to_m_y(y_px: np.ndarray, img_h: int, bbox_h_m: float) -> np.ndarray:
    denom = float(max(1, img_h - 1))
    return (np.asarray(y_px, dtype=float) / denom) * float(bbox_h_m)


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


def build_graph_from_polylines(polylines_rc: list[np.ndarray], mppx: float, mppy: float, quantize_q: float = 2.0):
    """
    Build an undirected weighted graph from polylines in (row,col).
    Edge weights are metric lengths (meters), not pixel lengths.
    """
    key_to_id: dict[tuple[int, int], int] = {}
    coords: list[np.ndarray] = []
    adj: dict[int, list[tuple[int, float]]] = {}

    def get_node_id(p_rc: np.ndarray) -> int:
        key = _quantize_rc(p_rc, quantize_q)
        nid = key_to_id.get(key)
        if nid is None:
            nid = len(coords)
            key_to_id[key] = nid
            coords.append(np.asarray(p_rc, dtype=float))
            adj[nid] = []
        return nid

    def add_edge(u: int, v: int, w_m: float):
        if u == v:
            return
        adj[u].append((v, float(w_m)))
        adj[v].append((u, float(w_m)))

    for pl in polylines_rc:
        pts = np.asarray(pl, dtype=float)
        if pts.shape[0] < 2:
            continue
        prev = get_node_id(pts[0])
        for k in range(1, pts.shape[0]):
            cur = get_node_id(pts[k])
            w_m = _metric_len_rc(coords[prev], coords[cur], mppx=mppx, mppy=mppy)
            add_edge(prev, cur, w_m)
            prev = cur

    return coords, adj


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
        dx = float(xy_all[u, 0] - xy_all[v, 0])
        dy = float(xy_all[u, 1] - xy_all[v, 1])
        w_m = float(np.hypot(dx, dy))
        adj[u].append((v, w_m))
        adj[v].append((u, w_m))

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
        w_m = _metric_len_rc(coords[from_id], new_p, mppx=mppx, mppy=mppy)
        adj[from_id].append((nid, float(w_m)))
        adj[nid].append((from_id, float(w_m)))
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
):
    """
    Within ONE connected component, add an undirected edge between candidate node pairs within radius_m,
    but skip "path-neighborhood" edges:

      Add (u,v) only if d_graph(u,v) > alpha * d_direct(u,v),

    where d_graph is shortest-path distance in the CURRENT subgraph (snapshot before adding new edges)
    and d_direct is 2D Euclidean metric distance.

    Does NOT add edges to other components.
    """
    if radius_m is None or float(radius_m) <= 0.0:
        return
    if not node_ids or len(node_ids) < 2:
        return
    alpha = float(alpha)
    if alpha <= 0:
        return

    comp_set = set(int(i) for i in node_ids)

    # Snapshot adjacency (only within this component) so decisions don't change while we add edges.
    adj0: dict[int, list[tuple[int, float]]] = {}
    for u in comp_set:
        adj0[u] = [(v, float(w)) for (v, w) in adj.get(u, []) if v in comp_set]

    ids = np.asarray(list(comp_set), dtype=int)
    # stable order for local indexing
    ids.sort()
    xy = _coords_metric_xy([coords_rc[i] for i in ids], mppx=mppx, mppy=mppy)  # (n,2) meters
    r = float(radius_m)
    eps = 1e-9

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

            # prepare targets (only lj>li to avoid duplicates)
            targets: list[tuple[int, float]] = []
            max_direct = 0.0
            for lj in neigh:
                lj = int(lj)
                if lj <= li:
                    continue
                dx = float(xy[li, 0] - xy[lj, 0])
                dy = float(xy[li, 1] - xy[lj, 1])
                d_direct = float(np.hypot(dx, dy))
                if d_direct <= 0.0 or d_direct > r + eps:
                    continue
                targets.append((lj, d_direct))
                if d_direct > max_direct:
                    max_direct = d_direct

            if not targets:
                continue

            u = int(ids[li])
            cutoff = alpha * max_direct
            dist_u = dijkstra_limited(u, cutoff=cutoff)

            for lj, d_direct in targets:
                v = int(ids[lj])
                d_graph = dist_u.get(v, float("inf"))
                if d_graph > alpha * d_direct + eps:
                    add_edge(u, v, d_direct)
    else:
        # brute-force fallback (component-local)
        n = xy.shape[0]
        for li in range(n):
            targets: list[tuple[int, float]] = []
            max_direct = 0.0
            for lj in range(li + 1, n):
                dx = float(xy[li, 0] - xy[lj, 0])
                dy = float(xy[li, 1] - xy[lj, 1])
                d_direct = float(np.hypot(dx, dy))
                if d_direct <= r + eps:
                    targets.append((lj, d_direct))
                    if d_direct > max_direct:
                        max_direct = d_direct

            if not targets:
                continue

            u = int(ids[li])
            cutoff = alpha * max_direct
            dist_u = dijkstra_limited(u, cutoff=cutoff)

            for lj, d_direct in targets:
                v = int(ids[lj])
                d_graph = dist_u.get(v, float("inf"))
                if d_graph > alpha * d_direct + eps:
                    add_edge(u, v, d_direct)


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
    reversal_split_mm: float,
    neighbor_radius_mm: float,
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

    levels = np.arange(
        np.floor(crop_min / interval_m) * interval_m,
        np.ceil(crop_max / interval_m) * interval_m + interval_m,
        interval_m,
        dtype=float,
    )

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

    coords, adj = build_graph_from_polylines(
        selected_segments_crop_rc,
        mppx=mppx,
        mppy=mppy,
        quantize_q=2.0,
    )
    if not coords:
        return None

    # neighbor radius specified in mm on output map -> meters in-world
    meters_per_mm_w = float(bbox_w_m) / float(max(1e-9, eff_width_mm))
    meters_per_mm_h = float(bbox_h_m) / float(max(1e-9, eff_height_mm))
    meters_per_mm = 0.5 * (meters_per_mm_w + meters_per_mm_h)
    neighbor_radius_m = float(neighbor_radius_mm) * meters_per_mm

    comps_pre = connected_components(adj)
    for comp in comps_pre:
        add_proximity_edges_within_component_radius(
            coords,
            adj,
            node_ids=comp,
            mppx=mppx,
            mppy=mppy,
            radius_m=neighbor_radius_m,
            alpha=float(skip_alpha),
        )

    connect_components_by_nearest_mst(coords, adj, mppx=mppx, mppy=mppy)

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
) -> trimesh.Trimesh:
    """
    Generate the full solid mesh for the elevation map in mm coordinates.
    """
    h, w = elev_map.shape
    
    # Normalize heights
    valid_mask = np.isfinite(elev_map)
    if not np.any(valid_mask):
        raise ValueError("Elevation map has no valid data")
        
    h_min = float(np.nanmin(elev_map[valid_mask]))
    h_max = float(np.nanmax(elev_map[valid_mask]))
    denom = max(1e-12, h_max - h_min)
    
    # Normalize to [0, 1]
    t = (elev_map - h_min) / denom
    t[~valid_mask] = 0.0 # Handle NaNs by setting to min height
    
    # Map to mm Z
    z_values = bottom_thickness_mm + t * desired_height_mm
    
    # Create grid of X, Y in mm
    x_idx = np.arange(w)
    y_idx = np.arange(h)
    xv, yv = np.meshgrid(x_idx, y_idx)
    
    x_mm = xv * mm_per_px_x
    y_mm = yv * mm_per_px_y
    
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
    Create a prism tool from a 2D boundary polygon (N, 2).
    """
    # Ensure closed polygon
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0]])
        
    poly = Polygon(boundary_points)
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
        reversal_split_mm,
        neighbor_radius_mm,
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
        reversal_split_mm=reversal_split_mm,
        neighbor_radius_mm=neighbor_radius_mm,
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


def main():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("depth_map")
    args = parse_args()

    with Timer("load elevation_map", log):
        elev_map, _elev_transform = build_elevation_map(
            args.lat_min, args.lat_max, args.lon_min, args.lon_max, cache_array=True
        )
        if elev_map is None or elev_map.size == 0:
            raise RuntimeError("build_elevation_map returned an empty array")
        elev_map = np.asarray(elev_map, dtype=np.float32)
        img_h, img_w = elev_map.shape

    # local CRS only for correct aspect (meters)
    local_crs = choose_local_crs(args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    _, _, _, _, bbox_w_m, bbox_h_m = project_bbox_to_local_m(
        args.lat_min, args.lat_max, args.lon_min, args.lon_max, local_crs
    )
    aspect_m = bbox_w_m / float(bbox_h_m) if bbox_h_m else (img_w / float(img_h))

    eff_width_mm, eff_height_mm = fit_mm_bbox_preserve_aspect_ratio(
        args.final_width_mm, args.final_height_mm, aspect_m
    )

    # tile planning
    col_starts_mm = plan_tile_starts_mm(eff_width_mm, args.bed_width_mm, args.overlap_width_mm)
    row_starts_mm = plan_tile_starts_mm(eff_height_mm, args.bed_height_mm, args.overlap_width_mm)
    col_starts_px = [_mm_to_px_x(mm, eff_width_mm, img_w) for mm in col_starts_mm]
    row_starts_px = [_mm_to_px_y(mm, eff_height_mm, img_h) for mm in row_starts_mm]
    overlap_px_x = _mm_to_px_x(args.overlap_width_mm, eff_width_mm, img_w)
    overlap_px_y = _mm_to_px_y(args.overlap_width_mm, eff_height_mm, img_h)

    corridors = []
    for i in range(1, len(col_starts_px)):
        x0 = int(col_starts_px[i])
        x1 = int(min(img_w, x0 + overlap_px_x))
        if x1 > x0:
            corridors.append({"kind": "col", "i": i, "x0": x0, "x1": x1, "y0": 0, "y1": img_h})
    for i in range(1, len(row_starts_px)):
        y0 = int(row_starts_px[i])
        y1 = int(min(img_h, y0 + overlap_px_y))
        if y1 > y0:
            corridors.append({"kind": "row", "i": i, "x0": 0, "x1": img_w, "y0": y0, "y1": y1})

    if not corridors:
        raise RuntimeError("No corridors computed (check bed/overlap/final size settings).")

    extent_m = (0.0, float(bbox_w_m), float(bbox_h_m), 0.0)  # origin upper => y down

    interval = float(args.elevation_interval_m)
    if interval <= 0:
        raise ValueError("--elevation-interval-m must be > 0")

    # --- multiprocessing over corridors ---
    # Conservative default to avoid Windows pagefile/RAM blowups during spawn.
    if int(args.workers) > 0:
        workers = int(args.workers)
    else:
        workers = min((os.cpu_count() or 1), 4)

    # Disk-backed memmap instead of shared_memory (Windows-friendly for big arrays)
    tmp_dir = Path(tempfile.gettempdir())
    mmap_path = str(tmp_dir / f"depth_map_elev_{os.getpid()}.mmap")

    with Timer("write memmap", log):
        mm = np.memmap(mmap_path, dtype=elev_map.dtype, mode="w+", shape=elev_map.shape)
        mm[:] = elev_map
        mm.flush()
        del mm

    try:
        job_args = [
            (
                c,
                mmap_path,
                elev_map.shape,
                str(elev_map.dtype),
                img_h,
                img_w,
                float(bbox_w_m),
                float(bbox_h_m),
                float(eff_width_mm),
                float(eff_height_mm),
                float(interval),
                float(args.reversal_split_mm),
                float(args.neighbor_radius_mm),
                float(args.skip_alpha),
            )
            for c in corridors
        ]

        results = []
        with Timer("compute corridor lines (multiprocessing)", log):
            ctx = get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                it = pool.imap_unordered(_worker_build_corridor_line, job_args, chunksize=1)
                for r in tqdm(it, total=len(job_args), desc="Corridors", unit="corridor"):
                    if r is not None:
                        results.append(r)

    finally:
        try:
            os.remove(mmap_path)
        except Exception:
            pass

    if not results:
        raise RuntimeError("No corridor lines were produced.")

    log.info("corridor lines produced: %d / %d", len(results), len(corridors))
    log.info("avg corridor time: %.3fs", float(np.mean([r["t_sec"] for r in results])) if results else 0.0)

    # --- plot full map + all corridor lines (existing behavior) ---
    fig, axp = plt.subplots(figsize=(12, 6))
    axp.imshow(elev_map, cmap="terrain", origin="upper", extent=extent_m)
    axp.set_aspect("equal", adjustable="box")

    for r in results:
        axp.plot(r["xs_m"], r["ys_m"], color="black", linewidth=1.0)

    axp.set_title(
        f"All corridors | interval={interval:.0f} m | neighbor_r={float(args.neighbor_radius_mm):.1f} mm | "
        f"alpha={float(args.skip_alpha):.2f} | workers={workers} | lines={len(results)}"
    )
    axp.set_axis_off()
    plt.tight_layout()

    if not args.export_meshes:
        plt.show()
        return

    # --- Generate Full Mesh ---
    mm_per_px_x = float(eff_width_mm) / float(img_w)
    mm_per_px_y = float(eff_height_mm) / float(img_h)
    
    out_dir = Path(args.mesh_out_dir).expanduser() if str(args.mesh_out_dir).strip() else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export corridor lines as OBJ
    lines_mm = []
    for r in results:
        pts_mm = np.column_stack((
            r["cols_full"] * mm_per_px_x,
            r["rows_full"] * mm_per_px_y
        ))
        lines_mm.append(pts_mm)
    
    z_lines = args.bottom_thickness_mm + args.desired_height_mm + 0.5
    save_lines_obj(str(out_dir / "corridor_lines.obj"), lines_mm, z=z_lines)
    log.info("Exported corridor_lines.obj")

    with Timer("generate full mesh", log):
        full_mesh = generate_full_mesh(
            elev_map,
            mm_per_px_x=mm_per_px_x,
            mm_per_px_y=mm_per_px_y,
            bottom_thickness_mm=args.bottom_thickness_mm,
            desired_height_mm=args.desired_height_mm
        )
        if not full_mesh.is_volume:
            log.warning("Full mesh is not a watertight volume! Boolean operations may fail.")
        
    # --- Prepare Tools ---
    # Organize lines by index
    col_lines = {}
    row_lines = {}
    for r in results:
        c = r["corridor"]
        # Convert px lines to mm
        pts_mm = np.column_stack((
            r["cols_full"] * mm_per_px_x,
            r["rows_full"] * mm_per_px_y
        ))
        if c["kind"] == "col":
            col_lines[c["i"]] = pts_mm
        else:
            row_lines[c["i"]] = pts_mm

    z_min = -1.0
    z_max = args.bottom_thickness_mm + args.desired_height_mm + 1.0
    
    nx = len(col_starts_px)
    ny = len(row_starts_px)
    
    # out_dir created above

    # Try to use manifold engine
    boolean_engine = None
    try:
        # Simple check if we can import it or if trimesh detects it
        # trimesh.boolean.intersection([], engine='manifold')
        boolean_engine = 'manifold'
    except Exception:
        pass

    half_clearance = args.fitting_clearance / 2.0

    with Timer("process tiles (boolean)", log):
        for ty in range(ny):
            # Build Row Tool
            # Top boundary
            if ty == 0:
                top_line = np.array([[0, 0], [eff_width_mm, 0]])
            else:
                # Offset Right (down/inwards)
                top_line = offset_polyline(row_lines[ty], -half_clearance, constrain_axis=0)
                
            # Bottom boundary
            if ty == ny - 1:
                bot_line = np.array([[0, eff_height_mm], [eff_width_mm, eff_height_mm]])
            else:
                # Offset Left (up/inwards)
                bot_line = offset_polyline(row_lines[ty + 1], half_clearance, constrain_axis=0)
            
            # Construct Polygon: Top (L->R) -> Right Edge -> Bottom (R->L) -> Left Edge
            # Right edge connects Top[-1] to Bot[-1]
            # Left edge connects Bot[0] to Top[0]
            
            poly_pts_row = np.vstack([
                top_line,
                bot_line[::-1]
            ])
            
            row_tool = create_extruded_tool(poly_pts_row, z_min, z_max)
            if not row_tool.is_volume:
                log.warning(f"Row tool {ty} is not a volume.")
            
            # Intersect full mesh with row tool
            try:
                row_slice = full_mesh.intersection(row_tool, engine=boolean_engine)
            except Exception as e:
                log.error(f"Failed to cut row {ty}: {e}")
                continue
                
            if row_slice.is_empty:
                continue

            for tx in range(nx):
                # Build Column Tool                # Left boundary
                if tx == 0:
                    left_line = np.array([[0, 0], [0, eff_height_mm]])
                else:
                    # Offset Right (right/inwards)
                    left_line = offset_polyline(col_lines[tx], half_clearance, constrain_axis=1)
                    
                # Right boundary
                if tx == nx - 1:
                    right_line = np.array([[eff_width_mm, 0], [eff_width_mm, eff_height_mm]])
                else:
                    # Offset Left (left/inwards)
                    right_line = offset_polyline(col_lines[tx + 1], -half_clearance, constrain_axis=1)
                    
                # Construct Polygon: Left (T->B) -> Bottom Edge -> Right (B->T) -> Top Edge
                poly_pts_col = np.vstack([
                    left_line,
                    right_line[::-1]
                ])
                
                col_tool = create_extruded_tool(poly_pts_col, z_min, z_max)
                
                try:
                    tile_mesh = row_slice.intersection(col_tool, engine=boolean_engine)
                except Exception as e:
                    log.error(f"Failed to cut tile {ty}_{tx}: {e}")
                    continue
                    
                if tile_mesh.is_empty:
                    continue
                    
                out_path = out_dir / f"tile_{ty:02d}_{tx:02d}.stl"
                tile_mesh.export(str(out_path))
                log.info(f"Exported {out_path}")

    log.info("Meshes written to: %s", str(out_dir))

    plt.show()


if __name__ == "__main__":
    main()