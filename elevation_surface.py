"""Shared DEM coastline policy for mesh generation and projection lighting.

Keep the positive terrain, mask disagreements, floor and patch rules identical.
GIS dependencies are loaded only when constructing a missing land mask.
"""
from pathlib import Path
from typing import Optional
import logging
import time
import numpy as np

DEFAULT_MIN_LAND_ELEVATION_M=-20.0
DEFAULT_MIN_BELOW_SEA_PATCH_PX=10

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

COASTLINE_URL = "https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip"

COASTLINE_ARCHIVE_NAME = "land-polygons-complete-4326.zip"

COASTLINE_SHAPEFILE = "land-polygons-complete-4326/land_polygons.shp"

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

def land_mask_cache_path(coastline_dir: Path, args, shape: tuple[int, int]) -> Path:
    """Name the cached mask after the bbox and grid it was rasterised for."""
    return Path(coastline_dir) / (
        f"land_mask_{args.lat_min}_{args.lat_max}_{args.lon_min}_{args.lon_max}"
        f"_{int(shape[0])}x{int(shape[1])}.npy"
    )

def ensure_coastline_archive(coastline_dir: Path, log: logging.Logger) -> Path:
    """Fetch the land polygon archive unless it is already on disk."""
    archive = Path(coastline_dir) / COASTLINE_ARCHIVE_NAME
    if archive.exists() and archive.stat().st_size > 0:
        return archive
    import urllib.request

    archive.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading land polygons (about 900 MB, once): %s", COASTLINE_URL)
    partial = archive.with_suffix(".part")
    with Timer("download land polygons", log):
        urllib.request.urlretrieve(COASTLINE_URL, partial)
    partial.replace(archive)
    return archive

def build_land_mask(args, shape: tuple[int, int], log: logging.Logger) -> Optional[np.ndarray]:
    """
    Inputs:
    - Parsed CLI arguments, the raster grid shape, and a logger.

    Outputs:
    - Returns a boolean grid, True where the OSM coastline says land, or `None`
      when the coastline was switched off.

    The result is cached per bounding box and grid, since rasterising the polygons
    takes far longer than reading them back.
    """
    if not str(args.coastline_dir).strip():
        return None
    coastline_dir = Path(str(args.coastline_dir)).expanduser()
    cache = land_mask_cache_path(coastline_dir, args, shape)
    if cache.exists():
        log.info("land mask read from cache: %s", cache)
        return np.load(cache)

    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    archive = ensure_coastline_archive(coastline_dir, log)
    with Timer("read land polygons for the bbox", log):
        land = gpd.read_file(
            f"zip://{archive}!{COASTLINE_SHAPEFILE}",
            bbox=(args.lon_min, args.lat_min, args.lon_max, args.lat_max),
        )
    log.info("land polygons covering the bbox: %d", len(land))
    transform = from_bounds(args.lon_min, args.lat_min, args.lon_max, args.lat_max, shape[1], shape[0])
    with Timer("rasterise land mask", log):
        # all_touched keeps rocks and islets smaller than a pixel, which would
        # otherwise be dropped and then flattened to sea level.
        mask = rasterize(
            ((geom, 1) for geom in land.geometry),
            out_shape=(int(shape[0]), int(shape[1])),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, mask)
    log.info("land covers %.2f%% of the bounding box; mask cached at %s", 100.0 * mask.mean(), cache)
    return mask

def apply_sea_level_model(
    elev_map: np.ndarray,
    land_mask: Optional[np.ndarray],
    min_land_elevation_m: float,
    min_patch_px: int = 1,
) -> np.ndarray:
    """
    Inputs:
    - `elev_map`: the raster as loaded, with the DEM's own negative values.
    - `land_mask`: True where the coastline says land, or `None` to flatten
      everything below sea level as before.
    - `min_land_elevation_m`: floor applied to land that lies below sea level.
    - `min_patch_px`: smallest connected run of below-sea-level land that is kept.

    Outputs:
    - Returns the raster the printed surface is built from: sea flat at zero,
      ground below sea level kept so it still shows relief.

    Only samples the DEM already reads as negative are touched. Positive ground is
    left exactly as it is, so a coastline that disagrees with the DEM by a few
    pixels - which happens along every steep coast - can never flatten real relief.

    That same disagreement speckles steep shorelines with isolated samples that dip
    just below zero, so patches smaller than `min_patch_px` are put back at sea
    level. Neighbours are counted edgewise, which treats a diagonal string of single
    pixels - the shape a misplaced coastline actually makes - as the separate specks
    it is rather than as one patch.
    """
    out = np.array(elev_map, dtype=np.float32, copy=True)
    below = np.isfinite(out) & (out < 0.0)
    if land_mask is None:
        out[below] = 0.0
        return out
    land_below = below & land_mask
    if int(min_patch_px) > 1 and land_below.any():
        from scipy import ndimage

        labels, count = ndimage.label(land_below)
        if count:
            sizes = np.bincount(labels.ravel())
            big_enough = sizes >= int(min_patch_px)
            big_enough[0] = False
            land_below = big_enough[labels]
    # sea, plus any speck of land below sea level too small to be real
    out[below & ~land_below] = 0.0
    np.maximum(out, np.float32(min_land_elevation_m), out=out, where=land_below)
    return out
