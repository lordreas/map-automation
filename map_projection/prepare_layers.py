"""Download and prepare installation-sized geographic layers on the laptop.

Only the clipped 4096-wide products are transferred to the projector. Runtime
rendering is offline; this command can also be run on the board if desired.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from . import config

SCHEMA=1
SOURCES={
 'satellite':('https://tiles.maps.eox.at/wms','s2cloudless', 'EOxCloudless by EOX IT Services GmbH / modified Copernicus Sentinel data 2016; CC BY 4.0', 'https://maps.eox.at/'),
 'semantic':('https://mapproxy.terrascope.be/mapproxy/service','esa-worldcover-map-10m-2021-v2_map','© ESA WorldCover project 2021 / modified Copernicus Sentinel data 2021; ESA WorldCover consortium', 'https://esa-worldcover.org/en/data-access'),
}
RIVERS='https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_eu_shp.zip'


def key(geo,width=4096):
 return hashlib.sha256(json.dumps([SCHEMA,geo,width],sort_keys=True).encode()).hexdigest()[:20]


def directory(geo,width=4096,root=None):
 return Path(root or config.ROOT/'.cache/layers')/key(geo,width)


def fetch(url,data=None,timeout=180):
 request=urllib.request.Request(url,data=data,headers={'User-Agent':'AlpineTerrainProjector/1.0 (personal offline relief map)'})
 with urllib.request.urlopen(request,timeout=timeout) as response:return response.read()


def image_layer(name,geo,width,height,path):
 if path.exists():
  with Image.open(path) as im:
   if im.size==(width,height):return
 url,layer,_,_=SOURCES[name]
 # Four ordinary WMS image requests avoid service-specific 2048-pixel limits.
 result=Image.new('RGB',(width,height))
 for y0,y1 in ((0,height//2),(height//2,height)):
  for x0,x1 in ((0,width//2),(width//2,width)):
   west=geo['lon_min']+(geo['lon_max']-geo['lon_min'])*x0/width
   east=geo['lon_min']+(geo['lon_max']-geo['lon_min'])*x1/width
   north=geo['lat_max']-(geo['lat_max']-geo['lat_min'])*y0/height
   south=geo['lat_max']-(geo['lat_max']-geo['lat_min'])*y1/height
   params=dict(service='WMS',version='1.1.1',request='GetMap',srs='EPSG:4326',bbox=f'{west},{south},{east},{north}',width=x1-x0,height=y1-y0,format='image/png',styles='',layers=layer)
   raw=fetch(url+'?'+urllib.parse.urlencode(params))
   try:
    with Image.open(io.BytesIO(raw)) as image:
     image.load()
     if image.size!=(x1-x0,y1-y0):raise ValueError('WMS returned incorrect image dimensions')
     result.paste(image.convert('RGB'),(x0,y0))
   except Exception as exc:raise RuntimeError(f'{name}: WMS image failed: {raw[:200]!r}') from exc
 temp=path.with_suffix('.tmp');result.save(temp,format='PNG');temp.replace(path)
 print(f'{name}: {width} × {height} ready',flush=True)


def raster_elevation(source,geo,size,path):
 if path.exists():return
 from .prepare_terrain import prepared
 prepared_path,_=prepared(Path(source),geo)
 z=np.load(prepared_path,mmap_mode='r',allow_pickle=False)
 # PIL's float raster resampling remains bounded; preserve the sea-mask cleanup.
 output=np.asarray(Image.fromarray(np.asarray(z)).resize(size,Image.Resampling.BILINEAR),dtype=np.float32)
 if not np.all(np.isfinite(output)):raise ValueError('Elevation layer contains nonfinite values')
 np.save(path,output)


def raster_rivers(geo,size,folder,downloads):
 path=folder/'rivers.npy';metadata=folder/'rivers.json'
 if path.exists() and metadata.exists():
  meta=json.loads(metadata.read_text())
  if meta.get('schema')==2:return meta
 from scipy.ndimage import distance_transform_edt
 from .hydrology import network_phases,width_weights
 import geopandas as gpd
 archive=downloads/'HydroRIVERS_v10_eu_shp.zip'
 if not archive.exists():
  print('Downloading HydroRIVERS Europe (~68 MB)…',flush=True)
  temp=archive.with_suffix('.tmp');temp.write_bytes(fetch(RIVERS,timeout=300))
  with zipfile.ZipFile(temp) as z:
   if z.testzip():raise ValueError('HydroRIVERS archive checksum failure')
  temp.replace(archive)
 with zipfile.ZipFile(archive) as z:member=next(n for n in z.namelist() if n.lower().endswith('.shp'))
 frame=gpd.read_file('/vsizip/'+str(archive.resolve())+'/'+member,bbox=(geo['lon_min'],geo['lat_min'],geo['lon_max'],geo['lat_max']))
 timing=network_phases(frame.HYRIV_ID,frame.NEXT_DOWN,frame.LENGTH_KM)
 weights=width_weights(np.asarray(frame.DIS_AV_CMS,np.float64))
 mask=Image.new('L',size);phases=Image.new('F',size);sizes=Image.new('F',size)
 ink=ImageDraw.Draw(mask);clock=ImageDraw.Draw(phases);weight_ink=ImageDraw.Draw(sizes);w,h=size;count=0
 # HydroRIVERS stores reach vertices from upstream to downstream. Drawing
 # downstream reaches last gives identical clock values at confluences.
 order=np.argsort(timing[:,0])
 for index in order:
  geom=frame.geometry.iloc[index]
  if geom is None or geom.is_empty:continue
  for line in (geom.geoms if geom.geom_type=='MultiLineString' else [geom]):
   points=np.asarray(line.coords)[:,:2]
   x=(points[:,0]-geo['lon_min'])/(geo['lon_max']-geo['lon_min'])*w-.5
   y=(geo['lat_max']-points[:,1])/(geo['lat_max']-geo['lat_min'])*h-.5
   lon,lat=np.radians(points[:,0]),np.radians(points[:,1]);dlon=np.diff(lon);dlat=np.diff(lat)
   length=np.r_[0,np.cumsum(np.sqrt((dlon*np.cos((lat[:-1]+lat[1:])*.5))**2+dlat**2))]
   phase=timing[index,0]+(timing[index,1]-timing[index,0])*length/max(length[-1],1e-12)
   for j in range(len(points)-1):
    steps=max(1,min(4096,int(np.ceil(max(abs(x[j+1]-x[j]),abs(y[j+1]-y[j]))/1.5))))
    for k in range(steps):
     a,b=k/steps,(k+1)/steps;edge=(x[j]*(1-a)+x[j+1]*a,y[j]*(1-a)+y[j+1]*a,x[j]*(1-b)+x[j+1]*b,y[j]*(1-b)+y[j+1]*b)
     ink.line(edge,fill=255,width=1);clock.line(edge,fill=float(phase[j]*(1-b)+phase[j+1]*b),width=1);weight_ink.line(edge,fill=float(weights[index]),width=1)
   count+=1
 distance,nearest=distance_transform_edt(np.asarray(mask)==0,return_indices=True)
 field=np.empty((h,w,3),np.float16);field[:,:,0]=distance
 field[:,:,1]=np.asarray(sizes)[nearest[0],nearest[1]];field[:,:,2]=np.asarray(phases)[nearest[0],nearest[1]]
 np.save(path,field)
 meta={'schema':2,'reaches':count,'source':'HydroRIVERS v1 Europe / HydroSHEDS, Lehner & Grill (2013)','url':'https://www.hydrosheds.org/products/hydrorivers','coverage':'Rivers with catchment ≥10 km² or mean discharge ≥0.1 m³/s; not every small stream.','animation':'Downstream topology with exact confluence times; discharge-weighted widths.'}
 metadata.write_text(json.dumps(meta,indent=2));print(f'Rivers: {count:,} connected reaches',flush=True);return meta


def read_elevation(text):
 match=re.match(r'^\s*([+-]?\d+(?:[.,]\d+)?)\s*(m|metres|meters|ft)?\s*$',str(text),re.I)
 if not match:return None
 digits=match[1]
 if ',' in digits and len(digits.rsplit(',',1)[1])==3 and len(digits.split(',',1)[0].lstrip('+-'))<=3:
  digits=digits.replace(',','')
 value=float(digits.replace(',','.'))
 return value*.3048 if (match[2] or '').lower()=='ft' else value


def raster_divides(geo,size,folder,downloads):
 path=folder/'divides.npy';metadata=folder/'divides.json'
 if path.exists():return json.loads(metadata.read_text())
 from scipy.ndimage import distance_transform_edt
 import geopandas as gpd
 from shapely import union_all
 archive=downloads/'hybas_eu_lev08_v1c.zip'
 if not archive.exists():
  print('Downloading HydroBASINS Europe, level 8…',flush=True)
  temp=archive.with_suffix('.tmp');temp.write_bytes(fetch('https://data.hydrosheds.org/file/hydrobasins/standard/hybas_eu_lev08_v1c.zip',timeout=300))
  with zipfile.ZipFile(temp) as z:
   if z.testzip():raise ValueError('Invalid watershed archive')
  temp.replace(archive)
 frame=gpd.read_file('/vsizip/'+str(archive.resolve())+'/hybas_eu_lev08_v1c.shp',bbox=(geo['lon_min'],geo['lat_min'],geo['lon_max'],geo['lat_max']))
 polygons=[p for p in frame.geometry if p is not None and not p.is_empty]
 # Shared catchment edges are divides. Remove the union's exterior (including
 # coastlines), rather than drawing coastal boundaries as mountain ridges.
 edges=union_all([p.boundary for p in polygons]).difference(union_all(polygons).boundary)
 w,h=size;mask=Image.new('L',size);draw=ImageDraw.Draw(mask)
 pending=[edges]
 while pending:
  line=pending.pop()
  if line.is_empty:continue
  if hasattr(line,'geoms'):pending.extend(line.geoms);continue
  points=[((lon-geo['lon_min'])/(geo['lon_max']-geo['lon_min'])*w-.5,(geo['lat_max']-lat)/(geo['lat_max']-geo['lat_min'])*h-.5) for lon,lat,*_ in line.coords]
  if len(points)>1:draw.line(points,fill=255,width=1)
 np.save(path,distance_transform_edt(np.asarray(mask)==0).astype(np.float16))
 meta={'basins':len(frame),'level':8,'source':'HydroBASINS v1, level 8 / HydroSHEDS','url':'https://www.hydrosheds.org/products/hydrobasins','coverage':'Shared drainage-basin boundaries; coastlines omitted.'}
 metadata.write_text(json.dumps(meta,indent=2));print(f'Drainage divides: {len(frame):,} basins',flush=True);return meta


def geonames_peaks(geo,downloads):
 """Named mountains in every country intersecting this installation's bbox."""
 if not (4<=geo['lon_min']<geo['lon_max']<=17 and 43<=geo['lat_min']<geo['lat_max']<=49):
  raise ValueError('Automatic summit country list covers this Alps installation; extend it for other regions.')
 countries=['FR','CH','IT','AT','DE','LI','SI','HR','HU','SK']
 def country(code):
  path=downloads/(code+'-geonames.zip')
  if not path.exists():
   temp=path.with_suffix('.tmp');temp.write_bytes(fetch('https://download.geonames.org/export/dump/'+code+'.zip'))
   with zipfile.ZipFile(temp) as z:
    if z.testzip():raise ValueError('Invalid GeoNames archive: '+code)
   temp.replace(path)
  nodes=[]
  with zipfile.ZipFile(path) as z, z.open(code+'.txt') as data:
   for raw in data:
    row=raw.decode('utf-8').rstrip('\n').split('\t')
    if len(row)<19 or row[6]!='T' or row[7] not in ('PK','PKS','MT'):continue
    lat,lon=float(row[4]),float(row[5])
    if geo['lat_min']<=lat<=geo['lat_max'] and geo['lon_min']<=lon<=geo['lon_max']:
     nodes.append({'type':'node','id':row[0],'lat':lat,'lon':lon,'tags':{'name':row[1],'ele':row[15]}})
  return nodes
 with ThreadPoolExecutor(max_workers=4) as pool:elements=[node for batch in pool.map(country,countries) for node in batch]
 return {'elements':elements,'provider':'GeoNames'}


def peaks(geo,size,folder,downloads):
 path=folder/'peaks.npz';raw_path=folder/'peaks-osm.json'
 if path.exists():return json.loads((folder/'peaks.json').read_text())
 if not raw_path.exists():
  response=geonames_peaks(geo,downloads)
  raw_path.write_text(json.dumps(response))
 else:response=json.loads(raw_path.read_text())
 z=np.load(folder/'elevation.npy',mmap_mode='r');w,h=size
 points=[];fallback=0
 for node in response.get('elements',[]):
  if node.get('type')!='node' or 'lat' not in node:continue
  x=(node['lon']-geo['lon_min'])/(geo['lon_max']-geo['lon_min'])*(w-1)
  y=(geo['lat_max']-node['lat'])/(geo['lat_max']-geo['lat_min'])*(h-1)
  if not (0<=x<w and 0<=y<h):continue
  elevation=read_elevation(node.get('tags',{}).get('ele',''))
  if elevation is None:elevation=float(z[min(h-1,round(y)),min(w-1,round(x))]);fallback+=1
  points.append((x,y,elevation))
 if not points:raise ValueError('Peak source returned no peaks')
 np.savez_compressed(path,points=np.array(points,np.float32))
 provider=response.get('provider','OpenStreetMap')
 meta={'count':len(points),'dem_elevations':fallback,'source':'GeoNames geographical database (CC BY 4.0)' if provider=='GeoNames' else '© OpenStreetMap contributors (ODbL)','url':'https://www.geonames.org/' if provider=='GeoNames' else 'https://www.openstreetmap.org/copyright','coverage':'Catalogued mountain summits; missing elevation tags sampled from the cleaned DEM.'}
 (folder/'peaks.json').write_text(json.dumps(meta,indent=2));print(f'Peaks: {len(points):,} mapped summits',flush=True);return meta


def prepare(geo,source,width=4096,root=None):
 folder=directory(geo,width,root);folder.mkdir(parents=True,exist_ok=True)
 downloads=folder.parent/'downloads';downloads.mkdir(exist_ok=True)
 height=round(width*geo['height_mm']/geo['width_mm']);size=(width,height)
 raster_elevation(source,geo,size,folder/'elevation.npy')
 with ThreadPoolExecutor(max_workers=3) as pool:
  images=[pool.submit(image_layer,n,geo,width,height,folder/(n+'.png')) for n in SOURCES]
  rivers=raster_rivers(geo,size,folder,downloads)
  from .road_data import prepare as prepare_roads
  roads=prepare_roads(geo,size,folder,downloads)
  divides=raster_divides(geo,size,folder,downloads)
  summit=peaks(geo,size,folder,downloads)
  for task in images:task.result()
 # Store deployment-ready RGB arrays: no PNG decoding on a layer toggle.
 for name in ('satellite','semantic'):
  path=folder/(name+'.npy');png=folder/(name+'.png')
  if not path.exists() or path.stat().st_mtime_ns<png.stat().st_mtime_ns:
   with Image.open(png) as image:np.save(path,np.asarray(image.convert('RGB')))
 from .seasonal import prepare as prepare_seasonal
 seasonal=prepare_seasonal(geo,(height,width),folder)
 meta={'schema':SCHEMA,'geo':geo,'width':width,'height':height,'created':datetime.now(timezone.utc).isoformat(),'sources':dict({n:{'credit':s[2],'url':s[3]} for n,s in SOURCES.items()},streets=roads),'seasonal':seasonal,'rivers':rivers,'peaks':summit,'divides':divides,'labels':False}
 tmp=folder/'manifest.tmp';tmp.write_text(json.dumps(meta,indent=2));tmp.replace(folder/'manifest.json')
 print('Layers ready: '+str(folder),flush=True);return folder


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=config.ROOT/'settings.json');p.add_argument('--dem',type=Path,required=True);p.add_argument('--width',type=int,default=4096);p.add_argument('--output',type=Path)
 args=p.parse_args();prepare(config.load(args.config)['geo'],args.dem,args.width,args.output)

if __name__=='__main__':main()
