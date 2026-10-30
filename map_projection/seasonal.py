"""Real monthly NASA Blue Marble observations; never synthesize seasonal snow."""
import calendar
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from .prepare_layers import fetch

SOURCE={'credit':'NASA Blue Marble Next Generation / Reto Stöckli, monthly composites 2004 (500 m)','url':'https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/','year':2004,'resolution_m':500}


def prepare_month(month,geo,shape,folder):
 import fcntl
 with (folder/f'satellite-{month:02d}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _prepare_month(month,geo,shape,folder)


def _prepare_month(month,geo,shape,folder):
 import rasterio
 from rasterio.windows import from_bounds
 path=folder/f'satellite-{month:02d}.npy'
 if path.exists():return path
 h,w=shape;name=calendar.month_name[month].lower()
 url=f'https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-base/{name}/world.2004{month:02d}.3x21600x21600.C1_geo.tif'
 if not (0<=geo['lon_min']<geo['lon_max']<=90 and 0<=geo['lat_min']<geo['lat_max']<=90):raise ValueError('The current monthly NASA download tile covers northeastern Earth; extend tile selection for other regions.')
 with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_TIMEOUT='90',GDAL_CACHEMAX=64):
  with rasterio.open(url) as source:
   window=from_bounds(geo['lon_min'],geo['lat_min'],geo['lon_max'],geo['lat_max'],source.transform)
   data=source.read(indexes=[1,2,3],window=window,out_shape=(3,h,w),resampling=rasterio.enums.Resampling.bilinear)
 if data.dtype!=np.uint8 or data.max()==0:raise ValueError('Invalid NASA monthly image')
 temp=path.with_suffix('.tmp.npy');np.save(temp,np.ascontiguousarray(np.moveaxis(data,0,-1)));temp.replace(path)
 print(f'Satellite: {calendar.month_name[month]} 2004 ready',flush=True);return path


def prepare(geo,shape,folder,workers=3):
 with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(lambda month:prepare_month(month,geo,shape,folder),range(1,13)))
 (folder/'seasonal.json').write_text(json.dumps(dict(SOURCE,months=list(range(1,13))),indent=2))
 return SOURCE

if __name__=='__main__':
 from .config import DEFAULT
 from .prepare_layers import directory
 folder=directory(DEFAULT['geo']);folder.mkdir(parents=True,exist_ok=True)
 prepare(DEFAULT['geo'],(2396,4096),folder)
