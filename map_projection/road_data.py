"""Label-free, accurate major roads from OpenFreeMap's OSM vector tiles."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import threading
import numpy as np
from PIL import Image,ImageDraw


def prepare(geo,size,folder,downloads):
 from scipy.ndimage import distance_transform_edt
 import mapbox_vector_tile
 from .prepare_layers import fetch
 metadata=folder/'streets.json';path=folder/'streets.npy'
 if path.exists() and metadata.exists():
  meta=json.loads(metadata.read_text())
  if meta.get('schema')==2:return meta
 tilejson=json.loads(fetch('https://tiles.openfreemap.org/planet'));template=tilejson['tiles'][0]
 zoom=8;n=2**zoom
 def tile_y(lat):return (1-math.asinh(math.tan(math.radians(lat)))/math.pi)*n/2
 west=math.floor((geo['lon_min']+180)/360*n);east=math.floor((geo['lon_max']+180)/360*n)
 north=math.floor(tile_y(geo['lat_max']));south=math.floor(tile_y(geo['lat_min']))
 version=template.split('/planet/',1)[1].split('/',1)[0];cache=downloads/('roads-'+version);cache.mkdir(exist_ok=True)
 budget=[0];lock=threading.Lock()
 def load(tile):
  x,y=tile;cache_path=cache/f'{zoom}-{x}-{y}.pbf'
  if not cache_path.exists():
   raw=fetch(template.format(z=zoom,x=x,y=y))
   with lock:
    budget[0]+=len(raw)
    if budget[0]>50_000_000:raise ValueError('OpenFreeMap offline download exceeded 50 MB; use fewer tiles.')
   cache_path.write_bytes(raw)
  layer=mapbox_vector_tile.decode(cache_path.read_bytes()).get('transportation',{});extent=layer.get('extent',4096);lines=[]
  for feature in layer.get('features',[]):
   props=feature['properties'];kind=props.get('class')
   if kind not in ('motorway','trunk') or props.get('ramp') or props.get('brunnel')=='tunnel':continue
   geom=feature['geometry'];parts=geom['coordinates'] if geom['type']=='MultiLineString' else [geom['coordinates']] if geom['type']=='LineString' else []
   for part in parts:
    q=np.asarray(part,dtype=np.float64)
    if len(q)<2:continue
    lon=(x+q[:,0]/extent)/n*360-180
    lat=np.degrees(np.arctan(np.sinh(math.pi*(1-2*(y+1-q[:,1]/extent)/n))))
    px=(lon-geo['lon_min'])/(geo['lon_max']-geo['lon_min'])*size[0]-.5;py=(geo['lat_max']-lat)/(geo['lat_max']-geo['lat_min'])*size[1]-.5
    lines.append((1. if kind=='motorway' else .6,list(zip(px,py))))
  return lines
 tiles=[(x,y) for y in range(north,south+1) for x in range(west,east+1)]
 with ThreadPoolExecutor(max_workers=4) as pool:lines=[line for batch in pool.map(load,tiles) for line in batch]
 mask=Image.new('L',size);weight=Image.new('F',size);draw=ImageDraw.Draw(mask);weights=ImageDraw.Draw(weight)
 for value,points in sorted(lines,key=lambda a:a[0]):draw.line(points,fill=255,width=1);weights.line(points,fill=value,width=1)
 distance,nearest=distance_transform_edt(np.asarray(mask)==0,return_indices=True)
 field=np.empty((size[1],size[0],2),np.float16);field[:,:,0]=distance;field[:,:,1]=np.asarray(weight)[nearest[0],nearest[1]]
 temp=path.with_suffix('.tmp.npy');np.save(temp,field);temp.replace(path)
 meta={'schema':2,'segments':len(lines),'source':'© OpenStreetMap contributors / OpenMapTiles / OpenFreeMap','credit':'© OpenStreetMap contributors / OpenMapTiles / OpenFreeMap','url':'https://openfreemap.org/','coverage':'Motorways and major trunk highways only; ramps, minor roads, ferries and tunnels excluded. Classes estimate road size; lane counts are not available in these tiles.'}
 metadata.write_text(json.dumps(meta,indent=2));print(f'Roads: {len(lines):,} OSM motorway/trunk segments',flush=True);return meta

if __name__=='__main__':
 from .config import DEFAULT
 from .prepare_layers import directory
 folder=directory(DEFAULT['geo']);prepare(DEFAULT['geo'],(4096,2396),folder,folder.parent/'downloads')
