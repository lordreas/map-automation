"""Prepare the shared land-aware lighting surface once, on the laptop.

Runtime reads two rows of this NPY at a time. No GIS/connected-component work
is performed during animation or on the memory-constrained projector.
"""
import argparse
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from elevation_surface import (apply_sea_level_model,build_land_mask,_replace_hgt_voids_with_nan,
                               DEFAULT_MIN_LAND_ELEVATION_M,DEFAULT_MIN_BELOW_SEA_PATCH_PX)
from .config import DEFAULT,ROOT

SURFACE_VERSION='land-surface-v1'


def paths(dem):
    dem=Path(dem)
    surface=dem.with_name(dem.stem+'-land-surface.npy')
    return surface,surface.with_suffix('.json')


def source_signature(dem):
    p=Path(dem);stat=p.stat()
    return list(_source_signature(str(p.resolve()),stat.st_size,stat.st_mtime_ns))


@lru_cache(maxsize=16)
def _source_signature(path,size,mtime_ns):
    # This Radxa's rsync preserves seconds but drops fractional timestamps.
    # A cached header/tail fingerprint stays portable and detects same-second
    # changes to small fixtures without rereading the full 468 MB every frame.
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        digest.update(source.read(65536));source.seek(max(0,size-65536));digest.update(source.read(65536))
    return Path(path).name,size,mtime_ns//1_000_000_000,digest.hexdigest()


def prepare(dem,geo,coastline_dir=None,land_mask=None,min_land_elevation_m=DEFAULT_MIN_LAND_ELEVATION_M,min_patch_px=DEFAULT_MIN_BELOW_SEA_PATCH_PX):
    dem=Path(dem);surface,metadata=paths(dem)
    source=np.load(dem,mmap_mode='r',allow_pickle=False)
    if source.ndim!=2:raise ValueError('DEM must be a 2D north-up raster.')
    if land_mask is None:
        args=SimpleNamespace(**{k:geo[k] for k in ['lat_min','lat_max','lon_min','lon_max']},
                             coastline_dir=str(coastline_dir or ROOT.parent/'coastline'))
        land_mask=build_land_mask(args,source.shape,logging.getLogger(__name__))
    if land_mask is None or land_mask.shape!=source.shape:
        raise ValueError('A land mask matching the source DEM grid is required.')
    # Match terrain generation: voids remain invalid; positive elevations are
    # never cut by the coastline; small four-connected negative land patches
    # disappear; retained negative ground is limited by the same land floor.
    clean=_replace_hgt_voids_with_nan(source)
    result=apply_sea_level_model(clean,land_mask,min_land_elevation_m,min_patch_px)
    result[~np.isfinite(result)]=0
    # int16 input, integer land floor and no resampling make this lossless while
    # halving SD-card storage. Floating DEM sources remain float32.
    if source.dtype==np.int16 and float(min_land_elevation_m).is_integer():result=result.astype(np.int16)
    temp=surface.with_suffix('.tmp.npy');np.save(temp,result);temp.replace(surface)
    mask_hash=hashlib.sha256(np.ascontiguousarray(land_mask).view(np.uint8)).hexdigest()
    note={'version':SURFACE_VERSION,'source':source_signature(dem),'shape':list(result.shape),
          'geo':{k:geo[k] for k in ['lat_min','lat_max','lon_min','lon_max']},
          'min_land_elevation_m':min_land_elevation_m,'min_patch_px':min_patch_px,
          'mask_sha256':mask_hash,'negative_land_cells':int(np.count_nonzero(result<0))}
    temp=metadata.with_suffix('.tmp');temp.write_text(json.dumps(note,indent=2)+'\n');temp.replace(metadata)
    print(f'Prepared {surface}: {note["negative_land_cells"]:,} below-sea-level land cells retained',flush=True)
    return surface


def prepared(dem,geo):
    surface,metadata=paths(dem)
    if not surface.is_file() or not metadata.is_file():
        raise ValueError('Land-aware DEM surface missing. On the laptop run: python -m map_projection.prepare_terrain --dem '+str(dem))
    note=json.loads(metadata.read_text())
    expected={k:geo[k] for k in ['lat_min','lat_max','lon_min','lon_max']}
    if note.get('version')!=SURFACE_VERSION or note.get('source')!=source_signature(dem) or note.get('geo')!=expected:
        raise ValueError('Prepared land surface is stale or has a different geographic extent. Re-run map_projection.prepare_terrain.')
    return surface,note


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dem',type=Path,default=ROOT.parent/'elevation_tiles/elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy')
    parser.add_argument('--coastline-dir',type=Path,default=ROOT.parent/'coastline')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--min-land-elevation-m',type=float,default=DEFAULT_MIN_LAND_ELEVATION_M)
    parser.add_argument('--min-below-sea-patch-px',type=int,default=DEFAULT_MIN_BELOW_SEA_PATCH_PX)
    parser.add_argument('--if-needed',action='store_true',help='Reuse a matching prepared surface.')
    args=parser.parse_args();geo=DEFAULT['geo']
    if args.config:
        from .config import load
        geo=load(args.config)['geo']
    logging.basicConfig(level=logging.INFO)
    if args.if_needed:
        try:
            surface,note=prepared(args.dem,geo)
            if note['min_land_elevation_m']==args.min_land_elevation_m and note['min_patch_px']==args.min_below_sea_patch_px:
                print(f'Reusing {surface}');return
        except ValueError:pass
    prepare(args.dem,geo,args.coastline_dir,min_land_elevation_m=args.min_land_elevation_m,min_patch_px=args.min_below_sea_patch_px)


if __name__=='__main__':main()
