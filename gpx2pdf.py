#!/usr/bin/env python3
"""
gpx2pdf.py (updated)

Fixes:
 - Draw white halo as one grouped path behind the colored stroke
 - Draw colored stroke runs grouped by identical colormap index (reduces PDF primitives)
 - Fix tile alignment and cropping: map covers the entire page (including margins)
 - Ensure tiles are cached and stitched correctly

Usage:
    python gpx2pdf.py track.gpx out.pdf
    See --help for CLI args.
"""
import os
import math
import argparse
import logging
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt

import requests
import numpy as np
from PIL import Image
import gpxpy
import cv2
from pyproj import Transformer
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling


# ---------------- CONFIG ----------------
CONFIG = {
    "width_cm": 10.0,
    "height_cm": 10.0,
    "padding_cm": 1.0,
    "dpi": 800,
    "colormap": "JET",
    "elev_min": None,
    "elev_max": None,
    "colored_track_pt": 1.0,
    "white_halo_pt": 2.0,
    "tile_url_template": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg",
    "tile_cache_dir": "./tiles",
    "tile_size": 256,
    "max_zoom": 20,
    "min_zoom": 0,
    "http_headers": {"User-Agent": "gpx2pdf/1.0 (+https://example.org/)"},
    "save_background_png": True,
    "smoothing_window": 50,  # window size for elevation profile smoothing in samples
}
_CV2_COLORMAPS = {
    "AUTUMN": cv2.COLORMAP_AUTUMN,
    "BONE": cv2.COLORMAP_BONE,
    "JET": cv2.COLORMAP_JET,
    "WINTER": cv2.COLORMAP_WINTER,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "OCEAN": cv2.COLORMAP_OCEAN,
    "SUMMER": cv2.COLORMAP_SUMMER,
    "SPRING": cv2.COLORMAP_SPRING,
    "COOL": cv2.COLORMAP_COOL,
    "HSV": cv2.COLORMAP_HSV,
    "PINK": cv2.COLORMAP_PINK,
    "HOT": cv2.COLORMAP_HOT,
}

# ---------- helpers -------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_gpx(gpx_path):
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    pts = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                if p.latitude is None or p.longitude is None:
                    continue
                pts.append((p.longitude, p.latitude, p.elevation if p.elevation is not None else 0.0))
    if not pts:
        raise ValueError("No trackpoints found in GPX.")
    return np.array(pts, dtype=float)

_transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def lonlat_to_mercator(lon, lat):
    x, y = _transformer_to_3857.transform(lon, lat)
    return x, y

_transformer_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
def mercator_to_lonlat(x, y):
    lon, lat = _transformer_to_4326.transform(x, y)
    return lon, lat

EARTH_RADIUS = 6378137.0
HALF_WORLD = math.pi * EARTH_RADIUS
INITIAL_RESOLUTION = 2 * HALF_WORLD / 256.0

def meters_per_pixel_for_zoom(z, tile_size=256):
    return INITIAL_RESOLUTION / (2 ** z) * (256.0 / tile_size)

def choose_zoom_for_mpp(target_mpp, min_z=0, max_z=20, tile_size=256):
    if target_mpp <= 0:
        return max_z
    for z in range(min_z, max_z + 1):
        res = meters_per_pixel_for_zoom(z, tile_size=tile_size)
        if res <= target_mpp:
            return z
    return max_z

def mercator_to_global_pixel(x, y, z, tile_size=256):
    res = meters_per_pixel_for_zoom(z, tile_size=tile_size)
    pixel_x = (x + HALF_WORLD) / res
    pixel_y = (HALF_WORLD - y) / res
    return pixel_x, pixel_y

def global_pixel_to_mercator(pixel_x, pixel_y, z, tile_size=256):
    res = meters_per_pixel_for_zoom(z, tile_size=tile_size)
    x = pixel_x * res - HALF_WORLD
    y = HALF_WORLD - pixel_y * res
    return x, y

def global_pixel_to_tile(pixel_x, pixel_y, tile_size=256):
    tx = int(math.floor(pixel_x / tile_size))
    ty = int(math.floor(pixel_y / tile_size))
    return tx, ty

def global_pixel_to_lonlat(pixel_x, pixel_y, z, tile_size=256):
    x, y = global_pixel_to_mercator(pixel_x, pixel_y, z, tile_size)
    lon, lat = mercator_to_lonlat(x, y)
    return lon, lat

def lonlat_to_global_pixel(lon, lat, z, tile_size=256):
    x, y = lonlat_to_mercator(lon, lat)
    px, py = mercator_to_global_pixel(x, y, z, tile_size)
    return px, py

def fetch_tile(z, x, y, config):
    cache_dir = Path(config["tile_cache_dir"])
    ensure_dir(cache_dir)
    layer_name = "eox_s2cloudless"
    ext = "jpg"
    cache_name = cache_dir / f"{layer_name}_{z}_{y}_{x}.{ext}"
    if cache_name.exists():
        # print(f"Using cached tile {z}/{x}/{y}")
        try:
            img = Image.open(cache_name)
            img.load()
            return img
        except Exception:
            cache_name.unlink(missing_ok=True)
            
    # print(f"Fetching tile {z}/{x}/{y}")
    url = config["tile_url_template"].format(z=z, x=x, y=y)
    headers = config.get("http_headers", {})
    resp = requests.get(url, headers=headers, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch tile {z}/{y}/{x} -> HTTP {resp.status_code}: {url}")
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    try:
        img.save(cache_name, optimize=True)
    except Exception:
        pass
    return img

def stitch_tiles(z, tx_min, tx_max, ty_min, ty_max, config):
    tile_size = config["tile_size"]
    cols = tx_max - tx_min + 1
    rows = ty_max - ty_min + 1
    big_w = cols * tile_size
    big_h = rows * tile_size
    big = Image.new("RGB", (big_w, big_h))
    for ix, tx in enumerate(range(tx_min, tx_max + 1)):
        for iy, ty in enumerate(range(ty_min, ty_max + 1)):
            try:
                t = fetch_tile(z, tx, ty, config)
            except Exception as e:
                logging.warning(f"Failed to fetch tile {z}/{ty}/{tx}: {e}")
                t = Image.new("RGB", (tile_size, tile_size), (200, 200, 200))
            big.paste(t, (ix * tile_size, iy * tile_size))
    return big, (tx_min, ty_min)

def build_elevation_map(lat_min, lat_max, lon_min, lon_max, zoom=12, cache_dir="./elevation_tiles"):
    """
    Download and assemble SRTM (NASADEM) elevation tiles for the bounding box.
    Tiles are cached in `cache_dir`. The assembled result is returned as a
    monolithic (height x width) numpy array in meters.

    Uses NASA SRTM1 (1 arcsecond ≈ 30 m) data from AWS Open Data via
    https://elevation-tiles-prod.s3.amazonaws.com
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Approximate tile coverage in 1x1 degree steps
    lat_range = range(int(math.floor(lat_min)), int(math.ceil(lat_max)))
    lon_range = range(int(math.floor(lon_min)), int(math.ceil(lon_max)))

    src_files_to_mosaic = []
    for lat in lat_range:
        for lon in lon_range:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            filename = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt.gz"
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{abs(lat):02d}/{filename}"
            local_path = os.path.join(cache_dir, filename)

            # Download tile if missing
            if not os.path.exists(local_path):
                r = requests.get(url)
                if r.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                else:
                    print(f"⚠️ Missing elevation tile {filename}")
                    continue

            # Open with rasterio directly (gzip-compressed)
            src = rasterio.open(f"gzip://{local_path}")
            src_files_to_mosaic.append(src)

    # Merge multiple tiles into one elevation array
    mosaic, out_trans = merge(src_files_to_mosaic)
    for src in src_files_to_mosaic:
        src.close()

    return mosaic[0], out_trans

def sample_elevation_from_map(elevation_map, transform, lats, lons):
    """
    Given an assembled elevation map (numpy array) and its affine transform,
    returns interpolated elevation values for each (lat, lon) pair.
    """
    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    rows = np.clip(rows, 0, elevation_map.shape[0] - 1)
    cols = np.clip(cols, 0, elevation_map.shape[1] - 1)
    return elevation_map[rows, cols]

def rasterize_elevation_map(elevation_map, transform, min_x_px, max_x_px, min_y_px, max_y_px, z, tile_size=256):
    """
    Rasterize the elevation map to a pixel array covering the given global pixel bbox
    at zoom level z. Returns a 2D numpy array of shape (height, width) with elevation values.
    """
    height = max_y_px - min_y_px + 1
    width = max_x_px - min_x_px + 1
    elevation_array = np.zeros((height, width), dtype=np.float32)
    
    # Create meshgrid of all pixel coordinates
    y_coords, x_coords = np.meshgrid(
        np.arange(min_y_px, max_y_px + 1),
        np.arange(min_x_px, max_x_px + 1),
        indexing='ij'
    )
    lons, lats = global_pixel_to_lonlat(x_coords.flatten(), y_coords.flatten(), z, tile_size)
    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    valid_mask = (
        (rows >= 0) & (rows < elevation_map.shape[0]) &
        (cols >= 0) & (cols < elevation_map.shape[1])
    )
    valid_elevations = elevation_map[rows[valid_mask], cols[valid_mask]]
    
    output_y_indices = y_coords.flatten()[valid_mask] - min_y_px
    output_x_indices = x_coords.flatten()[valid_mask] - min_x_px
    elevation_array[output_y_indices, output_x_indices] = valid_elevations
    
    return elevation_array

def apply_colormap_to_values(vals, vmin, vmax, cmap_name="JET"):
    if cmap_name not in _CV2_COLORMAPS:
        raise ValueError(f"Unknown colormap '{cmap_name}'")
    lower = vmin if vmin is not None else float(np.nanmin(vals))
    upper = vmax if vmax is not None else float(np.nanmax(vals))
    if upper == lower:
        upper = lower + 1.0
    norm = np.clip((np.asarray(vals, dtype=float) - lower) / (upper - lower), 0.0, 1.0)
    u8 = np.round(norm * 255).astype(np.uint8).ravel()
    cmap = _CV2_COLORMAPS[cmap_name]
    # Map unique u8 values to RGB once to reduce repeated work
    unique_vals = np.unique(u8)
    rgb_map = {}
    for v in unique_vals:
        bgr = cv2.applyColorMap(np.array([[v]], dtype=np.uint8), cmap)[0, 0]
        rgb_map[int(v)] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    colors = np.array([rgb_map[int(v)] for v in u8], dtype=np.uint8)
    return colors.reshape((-1, 3)), u8.reshape((-1,))

def create_pdf_with_track(
    out_pdf_path,
    bg_image_pil,
    merc_xs,
    merc_ys,
    elevs,
    config,
    bbox_minx,
    bbox_miny,
    bbox_maxx,
    bbox_maxy,
    padding_cm,
    zoom_center_x_m=None,
    zoom_center_y_m=None,
    zoom_factor=None,
):
    w_cm = config["width_cm"]
    h_cm = config["height_cm"]
    dpi = config["dpi"]
    page_w_pt = w_cm * cm
    page_h_pt = h_cm * cm
    pad_pt = padding_cm * cm
    avail_w_pt = page_w_pt - 2 * pad_pt
    avail_h_pt = page_h_pt - 2 * pad_pt

    page_w_px = int(round((w_cm * CM_TO_INCH) * dpi))
    page_h_px = int(round((h_cm * CM_TO_INCH) * dpi))

    tmp_png = Path(out_pdf_path).with_suffix(".bg.png")
    bg_image_pil.save(tmp_png, dpi=(dpi, dpi))

    c = canvas.Canvas(out_pdf_path, pagesize=(page_w_pt, page_h_pt))
    c.drawImage(str(tmp_png), 0, 0, width=page_w_pt, height=page_h_pt, preserveAspectRatio=False, mask='auto')

    # Mapping mercator meters -> PDF points:
    width_m = bbox_maxx - bbox_minx
    height_m = bbox_maxy - bbox_miny
    meters_per_pt = max(width_m / page_w_pt, height_m / page_h_pt)
    if meters_per_pt == 0:
        meters_per_pt = 1.0
    center_x = 0.5 * (bbox_minx + bbox_maxx)
    center_y = 0.5 * (bbox_miny + bbox_maxy)

    def merc_to_point(xs, ys):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        x_pts = (xs - center_x) / meters_per_pt + page_w_pt / 2.0
        y_pts = (ys - center_y) / meters_per_pt + page_h_pt / 2.0
        if zoom_center_x_m is not None and zoom_center_y_m is not None and zoom_factor is not None:
            x_pts = zoom_factor * (xs - zoom_center_x_m) / meters_per_pt + page_w_pt / 2.0
            y_pts = zoom_factor * (ys - zoom_center_y_m) / meters_per_pt + page_h_pt / 2.0
        return x_pts, y_pts

    elev_min = config["elev_min"]
    elev_max = config["elev_max"]
    if elev_min is None:
        elev_min = float(np.nanmin(elevs))
    if elev_max is None:
        elev_max = float(np.nanmax(elevs))

    # segment color mapping: use segment-averaged elevation
    seg_vals = 0.5 * (elevs[:-1] + elevs[1:])
    seg_colors_rgb, seg_u8 = apply_colormap_to_values(seg_vals, elev_min, elev_max, cmap_name=config["colormap"])

    x_pts, y_pts = merc_to_point(merc_xs, merc_ys)

    c.setLineJoin(1)
    c.setLineCap(1)

    halo_width_pt = float(config["white_halo_pt"])
    color_width_pt = float(config["colored_track_pt"])

    # 1) Draw the white halo as
    path = c.beginPath()
    path.moveTo(float(x_pts[0]), float(y_pts[0]))
    for xi, yi in zip(x_pts[1:], y_pts[1:]):
        path.lineTo(float(xi), float(yi))
    c.setStrokeColorRGB(1.0, 1.0, 1.0)
    c.setLineWidth(halo_width_pt)
    c.drawPath(path, stroke=1, fill=0)

    # 2) Draw colored strokes on top, grouped by runs of identical colormap index
    # seg_u8 is the color index (0..255) for each segment
    nseg = len(seg_u8)
    if nseg <= 0:
        # fallback: single point
        pass
    else:
        i = 0
        while i < nseg:
            j = i + 1
            while j < nseg and seg_u8[j] == seg_u8[i]:
                j += 1
            # create a path from point i to point j (note segments cover points i..j)
            p = c.beginPath()
            p.moveTo(float(x_pts[i]), float(y_pts[i]))
            # segments from i..j -> points i+1 .. j
            for k in range(i + 1, j + 1):
                p.lineTo(float(x_pts[k]), float(y_pts[k]))
            r, g, b = seg_colors_rgb[i] / 255.0
            c.setStrokeColorRGB(float(r), float(g), float(b))
            c.setLineWidth(color_width_pt)
            c.drawPath(p, stroke=1, fill=0)
            i = j

    # start/end markers (optional)
    start_r_pt = max(0.5, color_width_pt * 1.5)
    end_r_pt = start_r_pt
    c.setFillColorRGB(0.0, 0.7, 0.0)
    c.circle(float(x_pts[0]), float(y_pts[0]), start_r_pt, stroke=0, fill=1)
    c.setFillColorRGB(0.9, 0.0, 0.0)
    c.circle(float(x_pts[-1]), float(y_pts[-1]), end_r_pt, stroke=0, fill=1)

    c.showPage()
    c.save()
    try:
        tmp_png.unlink()
    except Exception:
        pass


CM_TO_INCH = 1.0 / 2.54
# -------- main workflow ----------
def process_gpx_to_pdf(gpx_file, out_pdf, user_config):
    cfg = CONFIG.copy()
    cfg.update(user_config or {})
    ensure_dir(cfg["tile_cache_dir"])

    pts = read_gpx(gpx_file)
    lons = pts[:, 0]
    lats = pts[:, 1]

    track_points_x_m, track_points_y_m = lonlat_to_mercator(lons, lats)
    track_min_x_m, track_min_y_m = float(track_points_x_m.min()), float(track_points_y_m.min())
    track_max_x_m, track_max_y_m = float(track_points_x_m.max()), float(track_points_y_m.max())
    track_width_m = track_max_x_m - track_min_x_m
    track_height_m = track_max_y_m - track_min_y_m

    # Canvas pixel sizes
    canvas_width_px = int(round(cfg["width_cm"] * CM_TO_INCH * cfg["dpi"]))
    canvas_height_px = int(round(cfg["height_cm"] * CM_TO_INCH * cfg["dpi"]))
    pad_px = int(round(cfg["padding_cm"] * CM_TO_INCH * cfg["dpi"]))
    avail_width_px = canvas_width_px - 2 * pad_px
    avail_height_px = canvas_height_px - 2 * pad_px
    if avail_width_px <= 0 or avail_height_px <= 0:
        raise ValueError("Padding too large for the chosen page size.")

    # Determine target meters-per-pixel so track fits into the inner area (avail)
    mpp_x = track_width_m / float(avail_width_px)
    mpp_y = track_height_m / float(avail_height_px)
    target_mpp = max(mpp_x, mpp_y)

    tile_size = cfg["tile_size"]
    map_z = choose_zoom_for_mpp(target_mpp, min_z=cfg["min_zoom"], max_z=cfg["max_zoom"], tile_size=tile_size)
    map_mpp = meters_per_pixel_for_zoom(map_z, tile_size=tile_size)
    logging.info(f"Using zoom level z={map_z}; resolution {map_mpp:.6f} m/px; target {target_mpp:.6f} m/px")

    # convert bbox corners to global pixel space. Remember that y axis is flipped wrt mercator
    track_min_x_px, track_max_y_px = mercator_to_global_pixel(track_min_x_m, track_min_y_m, map_z, tile_size)
    track_max_x_px, track_min_y_px = mercator_to_global_pixel(track_max_x_m, track_max_y_m, map_z, tile_size)
    track_width_px = track_max_x_px - track_min_x_px
    track_height_px = track_max_y_px - track_min_y_px
    print(f"Target padding in px: {pad_px}, actual padding: {(canvas_width_px - track_width_px) // 2}, {(canvas_height_px - track_height_px) // 2}")
    zoom_factor_x = avail_width_px / float(track_width_px)
    zoom_factor_y = avail_height_px / float(track_height_px)
    zoom_factor = min(zoom_factor_x, zoom_factor_y)
    zoom_center_x_m = 0.5 * (track_min_x_m + track_max_x_m)
    zoom_center_y_m = 0.5 * (track_min_y_m + track_max_y_m)
    if zoom_factor > 1.0:
        logging.warning(f"Warning: Interpolation lost {100 * ():.2f}% image detail.")
    print(zoom_factor, zoom_factor_x, zoom_factor_y)
    
    # center pixel for the track
    center_px_x = 0.5 * (track_min_x_px + track_max_x_px)
    center_px_y = 0.5 * (track_min_y_px + track_max_y_px)

    # crop a full-page region (w_px x h_px) centered at center_px
    crop_min_x_px = int(math.floor(center_px_x - canvas_width_px / zoom_factor / 2.0))
    crop_min_y_px = int(math.floor(center_px_y - canvas_height_px / zoom_factor / 2.0))
    crop_max_x_px = crop_min_x_px + canvas_width_px / zoom_factor
    crop_max_y_px = crop_min_y_px + canvas_height_px / zoom_factor
    
    # Elevation Processing
    crop_min_lon, crop_min_lat = global_pixel_to_lonlat(crop_min_x_px, crop_max_y_px, map_z, tile_size)
    crop_max_lon, crop_max_lat = global_pixel_to_lonlat(crop_max_x_px, crop_min_y_px, map_z, tile_size)
    elev_map, elev_transform = build_elevation_map(crop_min_lat, crop_max_lat, crop_min_lon, crop_max_lon)
    elevs = sample_elevation_from_map(elev_map, elev_transform, lats, lons)
    # Smooth the elevation profile using a simple moving average
    window_size = min(user_config.get("smoothing_window", 10), len(elevs) // 10 + 1)  # Adaptive window size, max 50
    if window_size >= 3 and window_size % 2 == 0:
        window_size += 1  # Ensure odd window size for symmetry
    if len(elevs) >= window_size:
        elevs = np.convolve(elevs, np.ones(window_size) / window_size, mode='same')
    
    # # Generate and save height map for fun
    # height_array = rasterize_elevation_map(elev_map, elev_transform, crop_min_x_px, crop_max_x_px, crop_min_y_px, crop_max_y_px, map_z, tile_size)
    # # normalize between height_array min/max
    # ha_min = float(np.nanmin(height_array[height_array != 0]))
    # ha_max = float(np.nanmax(height_array)) 
    # height_array = (height_array - ha_min) / (ha_max - ha_min) * 255.0
    # # save as image
    # height_img = Image.fromarray(np.uint8(height_array), mode='L')
    # height_img_path = Path(out_pdf).with_suffix(".elevation_map.png")
    # height_img.save(height_img_path, dpi=(cfg["dpi"], cfg["dpi"]))
    # logging.info(f"Saved elevation map to {height_img_path}")

    tx_min, ty_min = global_pixel_to_tile(crop_min_x_px, crop_min_y_px, tile_size)
    tx_max, ty_max = global_pixel_to_tile(crop_max_x_px - 1, crop_max_y_px - 1, tile_size)

    # add small margin to avoid edge artifacts
    tx_min -= 1
    ty_min -= 1
    tx_max += 1
    ty_max += 1

    logging.info(f"Fetching tiles z={map_z}, x={tx_min}..{tx_max}, y={ty_min}..{ty_max}")

    big_img, (origin_tx, origin_ty) = stitch_tiles(map_z, tx_min, tx_max, ty_min, ty_max, cfg)
    global_origin_x_px = origin_tx * tile_size
    global_origin_y_px = origin_ty * tile_size
    # scale the big_img by zoom_factor
    if zoom_factor != 1.0:
        new_w = int(round(big_img.width * zoom_factor))
        new_h = int(round(big_img.height * zoom_factor))
        big_img = big_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        global_origin_x_px = int(round(center_px_x - (center_px_x - global_origin_x_px) * zoom_factor))
        global_origin_y_px = int(round(center_px_y - (center_px_y - global_origin_y_px) * zoom_factor))
        crop_min_x_px = int(round(center_px_x - (center_px_x - crop_min_x_px) * zoom_factor))
        crop_min_y_px = int(round(center_px_y - (center_px_y - crop_min_y_px) * zoom_factor))
        crop_max_x_px = crop_min_x_px + canvas_width_px / zoom_factor
        crop_max_y_px = crop_min_y_px + canvas_height_px / zoom_factor

    crop_left_in_big = crop_min_x_px - global_origin_x_px
    crop_top_in_big = crop_min_y_px - global_origin_y_px
    crop_box = (int(crop_left_in_big), int(crop_top_in_big), int(crop_left_in_big + canvas_width_px), int(crop_top_in_big + canvas_height_px))
    # clamp crop box into big_img range
    big_w, big_h = big_img.size
    crop_box = (
        max(0, crop_box[0]),
        max(0, crop_box[1]),
        min(big_w, crop_box[2]),
        min(big_h, crop_box[3]),
    )
    content_img = big_img.crop(crop_box)

    # If crop was clipped (rare), paste into full sized white canvas
    if content_img.size != (canvas_width_px, canvas_height_px):
        full = Image.new("RGB", (canvas_width_px, canvas_height_px), (255, 255, 255))
        full.paste(content_img, (max(0, - (crop_min_x_px - global_origin_x_px)), max(0, - (crop_min_y_px - global_origin_y_px))))
        content_img = full

    if cfg["save_background_png"]:
        out_png = Path(out_pdf).with_suffix(".background.png")
        content_img.save(out_png, dpi=(cfg["dpi"], cfg["dpi"]))
        logging.info(f"Saved assembled background to {out_png}")

    # compute bbox in mercator for the cropped full-page pixels
    res = meters_per_pixel_for_zoom(map_z, tile_size=cfg["tile_size"])
    final_px_left = crop_min_x_px
    final_px_top = crop_min_y_px
    tl_x_m = final_px_left * res - HALF_WORLD
    tl_y_m = HALF_WORLD - final_px_top * res
    br_x_m = (final_px_left + canvas_width_px) * res - HALF_WORLD
    br_y_m = HALF_WORLD - (final_px_top + canvas_height_px) * res

    bbox_minx = min(tl_x_m, br_x_m)
    bbox_maxx = max(tl_x_m, br_x_m)
    bbox_miny = min(br_y_m, tl_y_m)
    bbox_maxy = max(br_y_m, tl_y_m)

    # create the PDF with the assembled full-page background and vector track on top
    create_pdf_with_track(
        out_pdf,
        content_img,
        track_points_x_m, track_points_y_m, elevs,
        cfg,
        bbox_minx, bbox_miny, bbox_maxx, bbox_maxy,
        cfg["padding_cm"],
        zoom_center_x_m, zoom_center_y_m,
        zoom_factor
    )

    print("Done. PDF written to:", out_pdf)
    print("If you used EOX tiles, please include attribution: 'Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH'.")


def parse_args():
    ap = argparse.ArgumentParser(description="Render GPX to a printable PDF with satellite backdrop.")
    ap.add_argument("gpx", help="Input GPX file")
    ap.add_argument("out_pdf", help="Output PDF file")
    ap.add_argument("--width-cm", type=float, default=CONFIG["width_cm"])
    ap.add_argument("--height-cm", type=float, default=CONFIG["height_cm"])
    ap.add_argument("--padding-cm", type=float, default=CONFIG["padding_cm"])
    ap.add_argument("--dpi", type=int, default=CONFIG["dpi"])
    ap.add_argument("--cmap", type=str, default=CONFIG["colormap"])
    ap.add_argument("--elev-min", type=float, default=None)
    ap.add_argument("--elev-max", type=float, default=None)
    ap.add_argument("--tile-cache", type=str, default=CONFIG["tile_cache_dir"])
    ap.add_argument("--tile-url", type=str, default=CONFIG["tile_url_template"], help="Tile URL template with {z}/{x}/{y}")
    ap.add_argument("--no-save-bg", dest="save_bg", action="store_false")
    ap.add_argument("--smoothing-window", type=int, default=CONFIG["smoothing_window"], help="Smoothing window size for elevation profile")
    ap.set_defaults(save_bg=True)
    return ap.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    user_cfg = {
        "width_cm": args.width_cm,
        "height_cm": args.height_cm,
        "padding_cm": args.padding_cm,
        "dpi": args.dpi,
        "colormap": args.cmap,
        "elev_min": args.elev_min,
        "elev_max": args.elev_max,
        "tile_cache_dir": args.tile_cache,
        "tile_url_template": args.tile_url,
        "save_background_png": args.save_bg,
    }
    process_gpx_to_pdf(args.gpx, args.out_pdf, user_cfg)

if __name__ == "__main__":
    main()
