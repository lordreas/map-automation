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
import io
import os
import math
import argparse
import logging
from io import BytesIO
from pathlib import Path
import pickle
import zipfile
import pandas as pd
from tqdm.auto import tqdm

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
import geopandas as gpd
from shapely.geometry import Point
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.graphics.shapes import Group, Shape
import xml.etree.ElementTree as ET
from io import StringIO
import re


# Configuration defaults are now handled by argparse
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
    times = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                if p.latitude is None or p.longitude is None:
                    continue
                pts.append((p.longitude, p.latitude, p.elevation if p.elevation is not None else 0.0))
                times.append(p.time)
    if not pts:
        raise ValueError("No trackpoints found in GPX.")
    return np.array(pts, dtype=float), times

_transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
def lonlat_to_mercator(lon, lat):
    x, y = _transformer_to_3857.transform(lon, lat)
    return x, y

_transformer_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
def mercator_to_lonlat(x, y):
    lon, lat = _transformer_to_4326.transform(x, y)
    return lon, lat

def calculate_speeds_from_track(pts, times, unit="km/h"):
    """
    Calculate speeds between consecutive trackpoints.
    
    Args:
        pts: numpy array of shape (n, 3) with [lon, lat, elev]
        times: list of datetime objects for each point
        unit: output unit - "m/s", "km/h", "kt", or "min/km"
    
    Returns:
        numpy array of speeds in specified unit for each segment (length n-1)
    """
    if len(pts) < 2 or len(times) != len(pts):
        return np.array([])
    
    speeds = []
    for i in range(len(pts) - 1):
        if times[i] is None or times[i+1] is None:
            speeds.append(0.0)
            continue
            
        # Calculate distance between consecutive points
        lon1, lat1 = pts[i, 0], pts[i, 1]
        lon2, lat2 = pts[i+1, 0], pts[i+1, 1]
        
        # Convert to mercator and calculate distance
        x1, y1 = lonlat_to_mercator(lon1, lat1)
        x2, y2 = lonlat_to_mercator(lon2, lat2)
        distance_m = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate time difference
        time_diff_s = (times[i+1] - times[i]).total_seconds()
        
        if time_diff_s > 0:
            speed_ms = distance_m / time_diff_s
        else:
            speed_ms = 0.0
        
        # Convert to requested unit
        if unit == "m/s":
            final_speed = speed_ms
        elif unit == "km/h":
            final_speed = speed_ms * 3.6
        elif unit == "kt":
            final_speed = speed_ms * 1.943844  # m/s to knots
        elif unit == "min/km":
            # For pace: minutes per kilometer
            if speed_ms > 0:
                final_speed = 1000.0 / speed_ms / 60.0  # minutes per km
            else:
                final_speed = 0.0
        else:
            raise ValueError(f"Unknown unit: {unit}")
            
        speeds.append(final_speed)
    
    return np.array(speeds, dtype=float)

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

def fetch_tile(z, x, y, args):
    cache_dir = Path(args.tile_cache)
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
    url = args.tile_url.format(z=z, x=x, y=y)
    resp = requests.get(url, headers={}, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch tile {z}/{y}/{x} -> HTTP {resp.status_code}: {url}")
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    try:
        img.save(cache_name, optimize=True)
    except Exception:
        pass
    return img

def stitch_tiles(z, tx_min, tx_max, ty_min, ty_max, args):
    tile_size = args.tile_size
    cols = tx_max - tx_min + 1
    rows = ty_max - ty_min + 1
    big_w = cols * tile_size
    big_h = rows * tile_size
    big = Image.new("RGB", (big_w, big_h))
    total = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
    with tqdm(total=total, desc=f"Fetching tiles z={z}", unit="tile") as p:
        for ix, tx in enumerate(range(tx_min, tx_max + 1)):
            for iy, ty in enumerate(range(ty_min, ty_max + 1)):
                try:
                    t = fetch_tile(z, tx, ty, args)
                except Exception as e:
                    logging.warning(f"Failed to fetch tile {z}/{ty}/{tx}: {e}")
                    t = Image.new("RGB", (tile_size, tile_size), (200, 200, 200))
                big.paste(t, (ix * tile_size, iy * tile_size))
                p.update(1)
    return big, (tx_min, ty_min)

def build_elevation_map(
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    zoom=12,
    cache_dir="./elevation_tiles",
    cache_array=False,
    output_shape=None,
    merge_resampling=Resampling.average,
):
    """
    Download and assemble SRTM (NASADEM) elevation tiles for the bounding box.
    Tiles are cached in `cache_dir`. The assembled result is returned as a
    monolithic (height x width) numpy array in meters.

    Uses NASA SRTM1 (1 arcsecond ≈ 30 m) data from AWS Open Data via
    https://elevation-tiles-prod.s3.amazonaws.com
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_version = "bboxmerge_v2"

    cache_suffix = ""
    if output_shape is not None:
        out_h = max(1, int(output_shape[0]))
        out_w = max(1, int(output_shape[1]))
        resamp_name = getattr(merge_resampling, "name", str(merge_resampling)).lower()
        cache_suffix = f"_shape_{out_h}x{out_w}_resamp_{resamp_name}_{cache_version}"
    else:
        cache_suffix = f"_{cache_version}"

    array_cache_path = os.path.join(cache_dir, f"elevation_{lat_min}_{lat_max}_{lon_min}_{lon_max}{cache_suffix}.npy")
    transform_cache_path = os.path.join(cache_dir, f"elevation_{lat_min}_{lat_max}_{lon_min}_{lon_max}{cache_suffix}_transform.pkl")
    if cache_array and os.path.exists(array_cache_path):
        print(f"Loading cached elevation array from {array_cache_path} ...")
        elev_array = np.load(array_cache_path, mmap_mode="r")
        with open(transform_cache_path, "rb") as f:
            out_trans = pickle.load(f)
        return elev_array, out_trans
    
    # Approximate tile coverage in 1x1 degree steps
    lat_range = range(int(math.floor(lat_min)), int(math.ceil(lat_max)))
    lon_range = range(int(math.floor(lon_min)), int(math.ceil(lon_max)))

    src_files_to_mosaic = []
    total = len(lat_range) * len(lon_range)
    with tqdm(total=total, desc="Fetching elevation tiles", unit="tile") as p:
        for lat in lat_range:
            for lon in lon_range:
                ns = "N" if lat >= 0 else "S"
                ew = "E" if lon >= 0 else "W"
                filename = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt.gz"
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{abs(lat):02d}/{filename}"
                local_path = os.path.join(cache_dir, filename)

                # Download tile if missing
                if not os.path.exists(local_path):
                    r = requests.get(url, timeout=20)
                    if r.status_code == 200:
                        with open(local_path, "wb") as f:
                            f.write(r.content)
                    else:
                        print(f"⚠️ Missing elevation tile {filename}")
                        p.update(1)
                        continue

                # Open with rasterio directly (gzip-compressed)
                src = rasterio.open(f"gzip://{local_path}")
                src_files_to_mosaic.append(src)
                p.update(1)

    # Merge multiple tiles into one elevation array. When output_shape is provided,
    # cap the merged raster resolution before materializing it in memory.
    merge_kwargs = {}
    # Crop the merge to the requested bbox instead of materializing the full
    # integer-degree tile union. Otherwise the returned raster extent is larger
    # than the requested region and degree-tile seams land at regular positions
    # in the downstream downsampled DEM.
    merge_kwargs["bounds"] = (
        float(lon_min),
        float(lat_min),
        float(lon_max),
        float(lat_max),
    )
    if output_shape is not None and src_files_to_mosaic:
        out_h = max(1, int(output_shape[0]))
        out_w = max(1, int(output_shape[1]))
        lon_span = max(1e-12, float(lon_max - lon_min))
        lat_span = max(1e-12, float(lat_max - lat_min))
        native_res_x, native_res_y = src_files_to_mosaic[0].res
        native_res_x = abs(float(native_res_x))
        native_res_y = abs(float(native_res_y))
        requested_res_x = lon_span / float(out_w)
        requested_res_y = lat_span / float(out_h)
        merge_kwargs["res"] = (
            max(native_res_x, requested_res_x),
            max(native_res_y, requested_res_y),
        )
        merge_kwargs["resampling"] = merge_resampling

    mosaic, out_trans = merge(src_files_to_mosaic, **merge_kwargs)
    for src in src_files_to_mosaic:
        src.close()
    
    if cache_array:
        np.save(array_cache_path, mosaic[0])
        with open(transform_cache_path, "wb") as f:
            pickle.dump(out_trans, f)
        del mosaic
        elev_array = np.load(array_cache_path, mmap_mode="r")
        return elev_array, out_trans

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

def load_settlements(min_lon, max_lon, min_lat, max_lat, min_population=10_000):
    """
    Loads populated places from Natural Earth, filters them by population.

    Args:
        min_population (int): Minimum population for inclusion.
    Returns:
        GeoDataFrame: Filtered settlements with geometry in WGS84 (lat/lon).
    """
    cache_dir = Path("./allCountries_cache")
    ensure_dir(cache_dir)
    URL = "https://download.geonames.org/export/dump/allCountries.zip"
    zip_path = cache_dir / "allCountries.zip"
    txt_path = cache_dir / "allCountries.txt"

    # Download and extract allCountries.txt if needed
    if not txt_path.exists():
        if not zip_path.exists():
            print(f"Downloading {URL} ...")
            r = requests.get(URL, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Extracting allCountries.txt ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("allCountries.txt", path=cache_dir)

    # Read the relevant columns directly
    cols = ["geonameid","name","asciiname","alternatenames","latitude","longitude",
            "feature_class","feature_code","country_code","cc2","admin1","admin2",
            "admin3","admin4","population","elevation","dem","timezone","moddate"]
    df = pd.read_csv(txt_path, sep="\t", header=None, names=cols,
                     usecols=["name","latitude","longitude","population"], dtype={"population": int})

    # filter population > 100
    df = df[df["population"] > min_population][["name","latitude","longitude","population"]]
        
    result = {
        line["name"]: {
            "lon": line["longitude"], 
            "lat": line["latitude"], 
            "population": line["population"]
        } for _, line in df.iterrows() if line["name"] and line["longitude"] >= min_lon and line["longitude"] <= max_lon and line["latitude"] >= min_lat and line["latitude"] <= max_lat
    }
    return result

def load_and_scale_svg_marker(svg_filename, desired_size, args):
    """
    Load an SVG marker file, apply annotation color, and scale it to the desired size.
    
    Args:
        svg_filename (str): Name of the SVG file in the markers directory
        desired_size (float): Desired size in points for the marker
        args: Arguments object containing annotation_color
    
    Returns:
        tuple: (drawing, scaled_width, scaled_height) or (None, 0, 0) if failed
    """
    try:
        svg_path = Path(__file__).parent / "markers" / svg_filename
        if not svg_path.exists():
            return None, 0, 0
            
        svg_string = svg_path.read_text(encoding="utf-8")
        
        def svg_all_white(svg_text):
            """Convert SVG colors to annotation_color"""
            # Convert annotation color to CSS format
            r, g, b = args.annotation_color
            color_str = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            
            # Register the default SVG namespace
            if 'xmlns=' in svg_text:
                ns_match = re.search(r'xmlns="([^"]+)"', svg_text)
                if ns_match:
                    ns = ns_match.group(1)
                    ET.register_namespace('', ns)

            root = ET.fromstring(svg_text)
            
            # Find and modify <style> tags
            for style_tag in root.findall('.//{http://www.w3.org/2000/svg}style'):
                if style_tag.text:
                    def replace_color(match):
                        prop = match.group(1)
                        color = match.group(2).strip()
                        if color.lower() in ('none', 'transparent'):
                            return match.group(0)
                        return f'{prop}: {color_str}'

                    css_text = re.sub(r'(fill|stroke)\s*:\s*([^;}]+)', replace_color, style_tag.text)
                    style_tag.text = css_text

            # Modify inline fill/stroke attributes
            for elem in root.iter():
                for attr in ("fill", "stroke"):
                    val = elem.attrib.get(attr)
                    if val and val.lower() not in ("none", "transparent"):
                        elem.set(attr, color_str)
            
            out = StringIO()
            ET.ElementTree(root).write(out, encoding='unicode')
            return out.getvalue()
        
        drawing = svg2rlg(StringIO(svg_all_white(svg_string)))
        
        # Get drawing dimensions
        dw = getattr(drawing, "width", None)
        dh = getattr(drawing, "height", None)
        if not dw or not dh:
            try:
                bbox = drawing.getBounds()
                dw = bbox[2] - bbox[0]
                dh = bbox[3] - bbox[1]
            except Exception:
                dw = dh = 1.0
        
        dw = float(dw)
        dh = float(dh)
        
        # Scale to desired size
        scale = desired_size / max(dw, dh)
        drawing.scale(scale, scale)
        
        return drawing, dw * scale, dh * scale
        
    except Exception:
        return None, 0, 0

def create_pdf_with_track_and_settlements(
    out_pdf_path,
    bg_image_pil,
    merc_xs,
    merc_ys,
    colorize_values,
    args,
    bbox_minx,
    bbox_miny,
    bbox_maxx,
    bbox_maxy,
    padding_cm,
    zoom_config=None,
    settlements=None
):
    w_cm = args.width_cm
    h_cm = args.height_cm
    dpi = args.dpi
    page_w_pt = w_cm * cm
    page_h_pt = h_cm * cm
    pad_pt = padding_cm * cm

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
        if zoom_config is not None:
            x_pts = zoom_config["factor"] * (xs - zoom_config["center_x_m"]) / meters_per_pt + page_w_pt / 2.0
            y_pts = zoom_config["factor"] * (ys - zoom_config["center_y_m"]) / meters_per_pt + page_h_pt / 2.0
        return x_pts, y_pts

    elev_min = args.colorize_min_val
    elev_max = args.colorize_max_val
    if elev_min is None:
        elev_min = float(np.nanmin(colorize_values))
    if elev_max is None:
        elev_max = float(np.nanmax(colorize_values))

    # segment color mapping: use segment-averaged values
    seg_vals = 0.5 * (colorize_values[:-1] + colorize_values[1:])
    seg_colors_rgb, seg_u8 = apply_colormap_to_values(seg_vals, elev_min, elev_max, cmap_name=args.cmap)

    x_pts, y_pts = merc_to_point(merc_xs, merc_ys)

    c.setLineJoin(1)
    c.setLineCap(1)

    halo_width_pt = float(args.white_halo_pt)
    color_width_pt = float(args.colored_track_pt)

    # 1) Draw the white halo as
    path = c.beginPath()
    path.moveTo(float(x_pts[0]), float(y_pts[0]))
    for xi, yi in zip(x_pts[1:], y_pts[1:]):
        path.lineTo(float(xi), float(yi))
    c.setStrokeColorRGB(*args.annotation_color)
    c.setLineWidth(halo_width_pt)
    c.drawPath(path, stroke=1, fill=0)

    # 2) Draw colored strokes on top, grouped by runs of identical colormap index
    # seg_u8 is the color index (0..255) for each segment
    nseg = len(seg_u8)
    if nseg <= 0:
        # fallback: single point
        pass
    else:
        # Draw all colored runs into a single Form XObject so Illustrator imports it as one element
        form_name = "colored_path"
        c.beginForm(form_name, 0, 0, page_w_pt, page_h_pt)
        c.setLineJoin(1)
        c.setLineCap(1)
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
        c.endForm()
        c.doForm(form_name)

    # start/end markers with SVG
    marker_size = max(args.start_goal_marker_width_pt, color_width_pt * 2.0)

    # Start marker
    drawing, scaled_w, scaled_h = load_and_scale_svg_marker("start.svg", marker_size, args)
    if drawing is not None:
        tx = float(x_pts[0])
        ty = float(y_pts[0])
        renderPDF.draw(drawing, c, tx, ty)
    
    # End marker
    drawing, scaled_w, scaled_h = load_and_scale_svg_marker("goal.svg", marker_size, args)
    if drawing is not None:
        tx = float(x_pts[-1])
        ty = float(y_pts[-1])
        renderPDF.draw(drawing, c, tx, ty)
    
    for s_name, s_info in settlements.items():
        x_m, y_m = lonlat_to_mercator(s_info["lon"], s_info["lat"])
        x_px, y_px = merc_to_point(x_m, y_m)
        
        # Determine if this is a large or small settlement
        is_large_settlement = s_info["population"] >= args.large_settlement_population
        svg_filename = "big_settlement.svg" if is_large_settlement else "small_settlement.svg"
        desired_size = float(args.city_marker_width_pt)
        
        # Draw custom SVG marker using the reusable function
        drawing, scaled_w, scaled_h = load_and_scale_svg_marker(svg_filename, desired_size, args)
        if drawing is not None:
            tx = float(x_px) - scaled_w / 2.0
            ty = float(y_px) - scaled_h / 2.0
            renderPDF.draw(drawing, c, tx, ty)
        else:
            # fallback circle if SVG missing or failed to load
            c.setFillColorRGB(*args.annotation_color)
            c.circle(float(x_px), float(y_px), 4, stroke=0, fill=1)
        
        # Draw city name at top right of marker
        c.setFont("Helvetica", 10)
        text_x = float(x_px) + 6
        text_y = float(y_px) + 6
        c.setFillColorRGB(*args.annotation_color)
        c.drawString(text_x, text_y, s_name)

    # Draw legend in bottom left
    legend_square_size = 4.0 * (72.0 / 25.4)  # 4mm in points
    legend_padding = 1.0 * (72.0 / 25.4)      # 1mm in points
    legend_start_x = 10.0 * (72.0 / 25.4)     # 10mm from left edge
    legend_start_y = 10.0 * (72.0 / 25.4)     # 10mm from bottom edge
    
    # Calculate legend dimensions
    legend_squares = 5
    legend_height = legend_squares * legend_square_size + (legend_squares - 1) * legend_padding
    
    # Prepare text samples for width calculation
    c.setFont("Helvetica", 8)
    
    unit_str = args.unit
    sample_texts = []
    for i in range(legend_squares):
        val = elev_min + (elev_max - elev_min) * i / (legend_squares - 1)
        
        if args.colorize_by == "elevation":
            text = f"{val:.0f} {unit_str}"
        elif args.unit == "min/km":
            # For pace, format as MM:SS per km
            minutes = int(val)
            seconds = int((val - minutes) * 60)
            text = f"{minutes:02d}:{seconds:02d} {unit_str}"
        else:
            text = f"{val:.1f} {unit_str}"
        sample_texts.append(text)
    
    # Calculate maximum text width
    max_text_width = max(c.stringWidth(text, "Helvetica", 8) for text in sample_texts)
    legend_width = legend_square_size + legend_padding + max_text_width
    
    # Draw background rectangle with 50% gray at 50% opacity
    bg_margin = 1.0 * (72.0 / 25.4)  # 1mm margin around elements
    bg_x = legend_start_x - bg_margin
    bg_y = legend_start_y - bg_margin
    bg_width = legend_width + 2 * bg_margin
    bg_height = legend_height + 2 * bg_margin
    
    c.saveState()
    c.setFillColorRGB(0.5, 0.5, 0.5)  # 50% gray
    c.setFillAlpha(0.5)               # 50% opacity
    c.rect(bg_x, bg_y, bg_width, bg_height, stroke=0, fill=1)
    c.restoreState()
    
    # Draw legend squares and text
    for i in range(legend_squares):
        # Calculate value for this square (from min to max)
        val = elev_min + (elev_max - elev_min) * i / (legend_squares - 1)
        
        # Get color for this value
        colors_rgb, _ = apply_colormap_to_values([val], elev_min, elev_max, cmap_name=args.cmap)
        r, g, b = colors_rgb[0] / 255.0
        
        # Square position (from bottom to top)
        square_y = legend_start_y + i * (legend_square_size + legend_padding)
        
        # Draw colored square
        c.setFillColorRGB(float(r), float(g), float(b))
        c.rect(legend_start_x, square_y, legend_square_size, legend_square_size, stroke=0, fill=1)
        
        # Draw value text
        text_x = legend_start_x + legend_square_size + legend_padding
        text_y = square_y + legend_square_size / 2.0 - 3.0  # Center vertically
        
        if args.colorize_by == "elevation":
            text = f"{val:.0f} {unit_str}"
        elif args.unit == "min/km":
            # For pace, format as MM:SS per km
            minutes = int(val)
            seconds = int((val - minutes) * 60)
            text = f"{minutes:02d}:{seconds:02d} {unit_str}"
        else:
            text = f"{val:.1f} {unit_str}"
            
        c.setFillColorRGB(*args.annotation_color)
        c.setFont("Helvetica", 8)
        c.drawString(text_x, text_y, text)

    # Draw scale bar in bottom right
    scale_margin = 10.0 * (72.0 / 25.4)  # 10mm from edges
    scale_start_x = page_w_pt - scale_margin
    scale_start_y = 10.0 * (72.0 / 25.4)  # 10mm from bottom
    max_scale_width_pt = args.max_scale_width_mm * (72.0 / 25.4)
    
    # Calculate scale based on meters per point
    possible_scales = []  # in meters
    for exp in range(1, 10):
        for base in [1, 2.5, 5]:
            possible_scales.append(int(base * (10 ** exp)))
    scale_distance = None
    scale_width_pt = None
    
    for scale_m in possible_scales:
        width_pt = scale_m / meters_per_pt
        if width_pt <= max_scale_width_pt:
            scale_distance = scale_m
            scale_width_pt = width_pt
        else:
            break
    
    if scale_distance is not None:
        # Draw scale line
        scale_line_x1 = scale_start_x - scale_width_pt
        scale_line_x2 = scale_start_x
        scale_line_y = scale_start_y
        
        c.setStrokeColorRGB(*args.annotation_color)
        c.setLineWidth(1.0)
        
        # Main horizontal line
        c.line(scale_line_x1, scale_line_y, scale_line_x2, scale_line_y)
        
        # Left serif
        serif_length = 5.0  # 5 points
        c.line(scale_line_x1, scale_line_y - serif_length/2, scale_line_x1, scale_line_y + serif_length/2)
        
        # Right serif
        c.line(scale_line_x2, scale_line_y - serif_length/2, scale_line_x2, scale_line_y + serif_length/2)
        
        # Scale text
        if scale_distance >= 1000:
            scale_text = f"{scale_distance // 1000} km"
        else:
            scale_text = f"{scale_distance} m"
        
        c.setFont("Helvetica", 8)
        text_width = c.stringWidth(scale_text, "Helvetica", 8)
        text_x = scale_line_x1 + (scale_width_pt - text_width) / 2.0  # Center text above scale
        text_y = scale_line_y + 8  # 8 points above scale line
        
        c.setFillColorRGB(*args.annotation_color)
        c.drawString(text_x, text_y, scale_text)

    c.showPage()
    c.save()
    try:
        tmp_png.unlink()
    except Exception:
        pass


CM_TO_INCH = 1.0 / 2.54
# -------- main workflow ----------
def process_gpx_to_pdf(gpx_file, out_pdf, args):
    ensure_dir(args.tile_cache)

    pts, times = read_gpx(gpx_file)
    lons = pts[:, 0]
    lats = pts[:, 1]

    track_points_x_m, track_points_y_m = lonlat_to_mercator(lons, lats)
    track_min_x_m, track_min_y_m = float(track_points_x_m.min()), float(track_points_y_m.min())
    track_max_x_m, track_max_y_m = float(track_points_x_m.max()), float(track_points_y_m.max())
    track_width_m = track_max_x_m - track_min_x_m
    track_height_m = track_max_y_m - track_min_y_m

    # Canvas pixel sizes
    canvas_width_px = int(round(args.width_cm * CM_TO_INCH * args.dpi))
    canvas_height_px = int(round(args.height_cm * CM_TO_INCH * args.dpi))
    pad_px = int(round(args.padding_cm * CM_TO_INCH * args.dpi))
    avail_width_px = canvas_width_px - 2 * pad_px
    avail_height_px = canvas_height_px - 2 * pad_px
    if avail_width_px <= 0 or avail_height_px <= 0:
        raise ValueError("Padding too large for the chosen page size.")

    # Determine target meters-per-pixel so track fits into the inner area (avail)
    mpp_x = track_width_m / float(avail_width_px)
    mpp_y = track_height_m / float(avail_height_px)
    target_mpp = max(mpp_x, mpp_y)

    tile_size = args.tile_size
    map_z = choose_zoom_for_mpp(target_mpp, min_z=args.min_zoom, max_z=args.max_zoom, tile_size=tile_size)
    map_mpp = meters_per_pixel_for_zoom(map_z, tile_size=tile_size)
    logging.info(f"Using zoom level z={map_z}; resolution {map_mpp:.6f} m/px; target {target_mpp:.6f} m/px")

    # convert bbox corners to global pixel space. Remember that y axis is flipped wrt mercator
    track_min_x_px, track_max_y_px = mercator_to_global_pixel(track_min_x_m, track_min_y_m, map_z, tile_size)
    track_max_x_px, track_min_y_px = mercator_to_global_pixel(track_max_x_m, track_max_y_m, map_z, tile_size)
    track_width_px = track_max_x_px - track_min_x_px
    track_height_px = track_max_y_px - track_min_y_px
    zoom_factor_x = avail_width_px / float(track_width_px)
    zoom_factor_y = avail_height_px / float(track_height_px)
    zoom_factor = min(zoom_factor_x, zoom_factor_y)
    zoom_center_x_m = 0.5 * (track_min_x_m + track_max_x_m)
    zoom_center_y_m = 0.5 * (track_min_y_m + track_max_y_m)
    if zoom_factor > 1.0:
        # zoom_factor > 1 => we're upscaling (interpolating) tiles; report "native detail kept"
        native_detail_kept_pct = 100.0 / float(zoom_factor)
        logging.warning(
            f"Warning: upscaling by {zoom_factor:.2f}x; native detail kept ~{native_detail_kept_pct:.1f}%."
        )

    # center pixel for the track
    center_px_x = 0.5 * (track_min_x_px + track_max_x_px)
    center_px_y = 0.5 * (track_min_y_px + track_max_y_px)

    # crop a full-page region (w_px x h_px) centered at center_px
    crop_min_x_px = int(math.floor(center_px_x - canvas_width_px / zoom_factor / 2.0))
    crop_min_y_px = int(math.floor(center_px_y - canvas_height_px / zoom_factor / 2.0))
    crop_max_x_px = crop_min_x_px + canvas_width_px / zoom_factor
    crop_max_y_px = crop_min_y_px + canvas_height_px / zoom_factor
    
    # Elevation/Speed Processing
    crop_min_lon, crop_min_lat = global_pixel_to_lonlat(crop_min_x_px, crop_max_y_px, map_z, tile_size)
    crop_max_lon, crop_max_lat = global_pixel_to_lonlat(crop_max_x_px - 1, crop_max_y_px - 1, map_z, tile_size)
    
    if args.colorize_by == "elevation":
        elev_map, elev_transform = build_elevation_map(crop_min_lat, crop_max_lat, crop_min_lon, crop_max_lon)
        colorize_values = sample_elevation_from_map(elev_map, elev_transform, lats, lons)
        # Smooth the elevation profile using a simple moving average
        window_size = min(args.smoothing_window, len(colorize_values) // 10 + 1)  # Adaptive window size, max 50
        if window_size >= 3 and window_size % 2 == 0:
            window_size += 1  # Ensure odd window size for symmetry
        if len(colorize_values) >= window_size:
            colorize_values = np.convolve(colorize_values, np.ones(window_size) / window_size, mode='same')
    elif args.colorize_by == "speed":
        speeds = calculate_speeds_from_track(pts, times, unit=args.unit)
        # For speed, we need one value per point, but speeds are per segment
        # Extend speeds array to match points by duplicating the last speed
        if len(speeds) > 0:
            colorize_values = np.append(speeds, speeds[-1])
        else:
            colorize_values = np.zeros(len(pts))
        # Smooth the speed profile using a simple moving average
        window_size = min(args.smoothing_window, len(colorize_values) // 10 + 1)
        if window_size >= 3 and window_size % 2 == 0:
            window_size += 1
        if len(colorize_values) >= window_size:
            colorize_values = np.convolve(colorize_values, np.ones(window_size) / window_size, mode='same')
    else:
        raise ValueError(f"Unknown colorize_by value: {args.colorize_by}")
    
    # # Generate and save height map for fun (only for elevation mode)
    # if args.colorize_by == "elevation":
    #     height_array = rasterize_elevation_map(elev_map, elev_transform, crop_min_x_px, crop_max_x_px, crop_min_y_px, crop_max_y_px, map_z, tile_size)
    #     # normalize between height_array min/max
    #     ha_min = float(np.nanmin(height_array[height_array != 0]))
    #     ha_max = float(np.nanmax(height_array)) 
    #     height_array = (height_array - ha_min) / (ha_max - ha_min) * 255.0
    #     # save as image
    #     height_img = Image.fromarray(np.uint8(height_array), mode='L')
    #     height_img_path = Path(out_pdf).with_suffix(".elevation_map.png")
    #     height_img.save(height_img_path, dpi=(cfg["dpi"], cfg["dpi"]))
    #     logging.info(f"Saved elevation map to {height_img_path}")

    tx_min, ty_min = global_pixel_to_tile(crop_min_x_px, crop_min_y_px, tile_size)
    tx_max, ty_max = global_pixel_to_tile(crop_max_x_px - 1, crop_max_y_px - 1, tile_size)

    # add small margin to avoid edge artifacts
    tx_min -= 1
    ty_min -= 1
    tx_max += 1
    ty_max += 1

    logging.info(f"Fetching tiles z={map_z}, x={tx_min}..{tx_max}, y={ty_min}..{ty_max}")

    big_img, (origin_tx, origin_ty) = stitch_tiles(map_z, tx_min, tx_max, ty_min, ty_max, args)
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

    if args.save_bg:
        out_png = Path(out_pdf).with_suffix(".background.png")
        content_img.save(out_png, dpi=(args.dpi, args.dpi))
        logging.info(f"Saved assembled background to {out_png}")

    # compute bbox in mercator for the cropped full-page pixels
    res = meters_per_pixel_for_zoom(map_z, tile_size=args.tile_size)
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
    
    settlements = load_settlements(
        crop_min_lon,
        crop_max_lon,
        crop_min_lat,
        crop_max_lat,
        min_population=args.small_settlement_population
    )

    # create the PDF with the assembled full-page background and vector track on top
    create_pdf_with_track_and_settlements(
        out_pdf,
        content_img,
        track_points_x_m, track_points_y_m, colorize_values,
        args,
        bbox_minx, bbox_miny, bbox_maxx, bbox_maxy,
        args.padding_cm,
        zoom_config={"center_x_m": zoom_center_x_m, "center_y_m": zoom_center_y_m, "factor": zoom_factor},
        settlements=settlements
    )

    print("Done. PDF written to:", out_pdf)
    print("If you used EOX tiles, please include attribution: 'Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH'.")


def parse_args():
    ap = argparse.ArgumentParser(description="Render GPX to a printable PDF with satellite backdrop.")
    ap.add_argument("gpx", help="Input GPX file")
    ap.add_argument("out_pdf", help="Output PDF file")
    ap.add_argument("--width-cm",                    default=12.7,      type=float, help="Width of the output PDF in centimeters")
    ap.add_argument("--height-cm",                   default=12.7,      type=float, help="Height of the output PDF in centimeters")
    ap.add_argument("--padding-cm",                  default=1.0,       type=float, help="Padding around the map in centimeters")
    ap.add_argument("--dpi",                         default=450,       type=int,   help="Resolution of the output PDF in dots per inch")
    ap.add_argument("--colorize-by",                 default="speed", choices=["elevation", "speed"], help="Colorize track by elevation or speed")
    ap.add_argument("--unit",                        default="kt",      type=str,   help="Unit for colorization values: 'm' for elevation, 'm/s', 'km/h', 'kt', 'min/km' for speed")
    ap.add_argument("--colorize-min-val",            default=None,      type=float, help="Minimum value for colormap normalization")
    ap.add_argument("--colorize-max-val",            default=None,      type=float, help="Maximum value for colormap normalization")
    ap.add_argument("--cmap",                        default="JET",     type=str,   help="Colormap for elevation profile (e.g., JET, HOT, RAINBOW)")
    ap.add_argument("--elev-min",                    default=None,      type=float, help="Minimum elevation for colormap normalization")
    ap.add_argument("--elev-max",                    default=None,      type=float, help="Maximum elevation for colormap normalization")
    ap.add_argument("--tile-cache",                  default="./tiles", type=str,   help="Directory to cache downloaded map tiles")
    ap.add_argument("--smoothing-window",            default=50,        type=int,   help="Smoothing window size for elevation profile")
    ap.add_argument("--small-settlement-population", default=10_000,    type=int,   help="Minimum population count for a small settlement to appear on the map")
    ap.add_argument("--large-settlement-population", default=500_000,   type=int,   help="Minimum population count for a large settlement marker")
    ap.add_argument("--colored-track-pt",            default=1.0,       type=float, help="Width of colored track in points")
    ap.add_argument("--white-halo-pt",               default=2.0,       type=float, help="Width of white halo around track in points")
    ap.add_argument("--tile-size",                   default=256,       type=int,   help="Tile size in pixels")
    ap.add_argument("--max-zoom",                    default=20,        type=int,   help="Maximum zoom level")
    ap.add_argument("--min-zoom",                    default=0,         type=int,   help="Minimum zoom level")
    ap.add_argument("--city-marker-width-pt",        default=10.0,      type=float, help="Width of city markers in points")
    ap.add_argument("--start-goal-marker-width-pt",  default=15.0,      type=float, help="Width of start/goal markers in points")
    ap.add_argument("--max-scale-width-mm",          default=30.0,      type=float, help="Maximum width of scale bar in millimeters")
    ap.add_argument("--tile-url",                    default="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg", type=str, help="Tile URL template with {z}/{x}/{y}")
    ap.add_argument("--no-save-bg",    dest="save_bg",    action="store_false")
    ap.set_defaults(save_bg=True)
    return ap.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.annotation_color = (1.0, 1.0, 1.0)  # white (R, G, B values 0-1)
    
    # Set default units if not specified
    if args.unit is None:
        if args.colorize_by == "elevation":
            args.unit = "m"
        else:  # speed
            args.unit = "km/h"
    
    process_gpx_to_pdf(args.gpx, args.out_pdf, args)

if __name__ == "__main__":
    main()
