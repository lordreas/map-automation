"""Offline geographic layer state. Projection follows first hits on the STL."""
import ctypes as C
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
from PIL import Image
from . import config
from .prepare_layers import directory


_library=None

def native():
 global _library
 if _library is not None:return _library
 source=config.ROOT/'native/layers.c';tag=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
 target=config.ROOT/'.cache'/('layers-'+tag+'.so');target.parent.mkdir(exist_ok=True)
 if not target.exists():
  compiler=shutil.which('cc') or shutil.which('gcc')
  if not compiler:raise RuntimeError('A C compiler is required for geographic layers')
  tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
  try:subprocess.run([compiler,'-O3','-std=c99','-shared','-fPIC',str(source),'-o',str(tmp),'-lm'],check=True);tmp.replace(target)
  finally:tmp.unlink(missing_ok=True)
 lib=C.CDLL(str(target));f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');u16=np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS')
 lib.layer_uv.argtypes=[C.c_int,C.c_int,f32,f64,f64,C.c_double,C.c_double,C.c_double,C.c_double,u16]
 lib.layer_heights.argtypes=[C.c_int,C.c_int,u16,C.c_int,C.c_int,f32,f32]
 lib.layer_rivers.argtypes=[C.c_int,C.c_int,u16,C.c_int,C.c_int,u16,u16]
 lib.peak_distance.argtypes=[C.c_int,C.c_int,C.c_int,f32,C.c_float,C.c_float,f32];lib.peak_distance.restype=C.c_int
 _library=lib;return lib


def project_uv(depth,inverse,origin,bounds,geo):
 h,w=depth.shape;out=np.empty((h,w,2),np.uint16)
 native().layer_uv(w,h,depth,np.ascontiguousarray(inverse,np.float64),np.ascontiguousarray(origin,np.float64),(bounds[0]+bounds[3])/2,(bounds[1]+bounds[4])/2,geo['width_mm'],geo['height_mm'],out)
 return out


def peak_field(points,shape,low,high):
 h,w=shape;out=np.empty(shape,np.float32);count=native().peak_distance(w,h,len(points),np.ascontiguousarray(points,np.float32),low,high,out)
 if count<0:raise MemoryError('Cannot allocate peak distance field')
 return out.astype(np.float16),count


class RevealClock:
 def __init__(self,duration=6.):self.duration=duration;self.started=None;self.generation=0
 def sample(self):return min(1.,max(0.,(time.monotonic()-self.started)/self.duration)) if self.started is not None else 0.
 def start(self):
  if self.started is None:self.started=time.monotonic()
 def reset(self):self.started=None;self.generation+=1


class LayerStack:
 def __init__(self,root=None):
  self.root=root;self.lock=threading.RLock();self.uv=None;self.uv_key=None;self.folder=None;self.meta={};self.arrays={};self.error=None;self.selected_peaks=0;self.peak_key=None;self.peak_texture=None;self.generation=0;self.screen_elevation=None;self.height_key=None
  self.river_enabled=False;self.river_clock=RevealClock()
  self.resident=OrderedDict();self.screen_rivers=None;self.river_projection_key=None
  self.river_events=None;self.river_events_key=None

 def snapshot(self):
  with self.lock:return {'ready':bool(self.meta),'error':self.error,'sources':self.meta.get('sources',{}),'seasonal':self.meta.get('seasonal',{}),'satellite_asset':getattr(self,'satellite_asset','satellite'),'rivers':dict(self.meta.get('rivers',{}),animating=self.river_enabled and self.river_clock.sample()<1),'divides':self.meta.get('divides',{}),'peaks':dict(self.meta.get('peaks',{}),selected=self.selected_peaks)}

 def prepare(self,c,geometry_key,depth,inverse,origin,bounds):
  layers=c['layers'];folder=directory(c['geo'],root=self.root)
  if layers['rivers']['enabled'] and not self.river_enabled:self.river_clock.reset()
  self.river_enabled=layers['rivers']['enabled']
  needs_data=any(v['enabled'] for n,v in layers.items() if n!='white')
  if folder!=self.folder:
   self.folder=folder;self.arrays={};self.resident.clear();self.meta={};self.error=None;self.peak_key=None;self.peak_texture=None
   if (folder/'manifest.json').exists():
    self.meta=json.loads((folder/'manifest.json').read_text())
    if self.meta['geo']!=c['geo']:raise ValueError('Layer extent does not match geographic registration; prepare layers again.')
  if needs_data and not self.meta:
   self.error='Geographic data are not installed for this map extent. Run deploy.sh --prepare-layers.'
  else:self.error=None
  uv_key=(geometry_key,tuple(sorted(c['geo'].items())))
  if uv_key!=self.uv_key:
   self.uv=project_uv(depth,inverse,origin,bounds,c['geo']);self.uv_key=uv_key;self.generation+=1
  wanted=[]
  if layers['contours']['enabled']:wanted.append('elevation')
  if layers['rivers']['enabled']:wanted.append('rivers')
  if layers['divides']['enabled']:wanted.append('divides')
  for name in ('satellite','streets','semantic'):
   if layers[name]['enabled']:wanted.append(name)
  # Release disabled CPU products; the GPU mirrors this bounded set.
  self.arrays={k:v for k,v in self.arrays.items() if k in wanted}
  if self.meta:
   month=layers['satellite']['month']
   if month==-1:month=int(c['sun']['time'][5:7])
   satellite_asset=f'satellite-{month:02d}' if month>0 else 'satellite'
   if satellite_asset!=getattr(self,'satellite_asset',None):self.arrays.pop('satellite',None)
   for name in wanted:
    if name not in self.arrays:
     month=layers['satellite']['month']
     if month==-1:month=int(c['sun']['time'][5:7])
     asset=f'satellite-{month:02d}' if name=='satellite' and month>0 else name
     if name=='satellite' and asset!=getattr(self,'satellite_asset',None):self.arrays.pop(name,None)
     if asset in self.resident:
      self.arrays[name]=self.resident.pop(asset);self.resident[asset]=self.arrays[name]
      if name=='satellite':self.satellite_asset=asset
      continue
     path=folder/(asset+'.npy')
     if not path.exists():path=folder/(asset+'.png')
     if path.suffix=='.png':
      with Image.open(path) as image:self.arrays[name]=np.asarray(image.convert('RGB'))
     else:self.arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
     if name in ('satellite','semantic'):
      self.resident[asset]=self.arrays[name]
      while len(self.resident)>3:self.resident.popitem(last=False)
     if name=='satellite':self.satellite_asset=asset
   if 'rivers' in self.arrays:
    signature=(self.uv_key,str(folder), (folder/'rivers.npy').stat().st_mtime_ns)
    if self.river_projection_key!=signature:
     h,w=depth.shape;source=self.arrays['rivers'];gh,gw=source.shape[:2];self.screen_rivers=np.empty((h,w,3),np.float16)
     native().layer_rivers(w,h,self.uv,gw,gh,source.view(np.uint16),self.screen_rivers.view(np.uint16));self.river_projection_key=signature
    self.arrays['screen_rivers']=self.screen_rivers
    compensation=c['optics']['chromatic_pixels'] if not c['track']['enabled'] else 0.
    signature=(self.river_projection_key,layers['rivers']['river_thickness'],compensation)
    if self.river_events_key!=signature:
     from .river_events import compile_events
     self.river_events=compile_events(self.screen_rivers,layers['rivers']['river_thickness'],compensation);self.river_events_key=signature
   if 'elevation' in self.arrays:
    signature=(self.uv_key,str(folder))
    if self.height_key!=signature:
     z=self.arrays['elevation'];gh,gw=z.shape;h,w=depth.shape;heights=np.empty((h,w),np.float32)
     native().layer_heights(w,h,self.uv,gw,gh,z,heights)
     self.screen_elevation=heights.astype(np.float16);self.height_key=signature
    self.arrays['screen_elevation']=self.screen_elevation
   p=layers['peaks']
   if p['enabled']:
    signature=(str(folder),p['min_elevation'],p['max_elevation'])
    if signature!=self.peak_key:
     with np.load(folder/'peaks.npz',allow_pickle=False) as data:points=data['points']
     self.peak_texture,self.selected_peaks=peak_field(points,(self.meta['height'],self.meta['width']),p['min_elevation'],p['max_elevation']);self.peak_key=signature
    self.arrays['peaks']=self.peak_texture
   else:self.peak_texture=None;self.peak_key=None
  # The per-frame config is immutable; arrays change only on geometry/data edits.
  return {'key':self.uv_key,'uv':self.uv,'config':layers,'arrays':dict(self.arrays),'shape':depth.shape,'river_clock':self.river_clock if self.river_enabled and 'rivers' in self.arrays else None,'river_events':self.river_events if self.river_enabled else None}


def compose_cpu(frame,state,shading=True):
 """CPU fallback for installations without GLES; physical GPU is the fast path."""
 uv=state['uv'];valid=uv[:,:,0]!=65535;cfg=state['config'];h,w=uv.shape[:2]
 out=np.zeros((h,w,3),np.float32)
 if cfg['white']['enabled']:out[:]=255*cfg['white']['opacity']
 arrays=state['arrays']
 def sample(a):
  ah,aw=a.shape[:2];x=np.minimum((uv[:,:,0].astype(np.float32)/65534*(aw-1)).astype(int),aw-1);y=np.minimum((uv[:,:,1].astype(np.float32)/65534*(ah-1)).astype(int),ah-1)
  return a[y,x]
 for name in ('satellite','semantic'):
  if cfg[name]['enabled'] and name in arrays:
   opacity=cfg[name]['opacity'];image=sample(arrays[name])
   if name=='satellite':image=np.clip(image.astype(np.float32)*cfg[name]['brightness'],0,255)
   out=out*(1-opacity)+image*opacity
 if shading:out*=frame[:,:,:1].astype(np.float32)/255
 def ink(name,alpha):
  nonlocal out
  color=np.array([int(cfg[name]['color'][i:i+2],16) for i in (1,3,5)],np.float32)
  a=np.clip(alpha,0,1)[:,:,None];out=out*(1-a)+color*a
 if 'elevation' in arrays:
  z=sample(arrays['elevation']);p=cfg['contours'];grad=np.maximum(np.abs(np.gradient(z,axis=0))+np.abs(np.gradient(z,axis=1)),.1)
  d=abs(z-p['level']) if p['single'] else abs((z+p['interval']/2)%p['interval']-p['interval']/2)
  alpha=np.clip(p['width']/2+.5-d/grad,0,1)*(abs(z)>.05)
  if p['single'] and p['fill']!='line':
   alpha=np.clip((z-p['level'])/grad+.5,0,1)
   if p['fill']=='below':alpha=1-alpha
  ink('contours',alpha)
 for name in ('streets','rivers','peaks','divides'):
  if name in arrays:
   a=arrays[name];ah,aw=a.shape[:2];du=np.gradient(uv[:,:,0].astype(np.float32)/65534*(aw-1),axis=1);dv=np.gradient(uv[:,:,1].astype(np.float32)/65534*(ah-1),axis=0);scale=np.maximum(np.hypot(du,dv)/1.414,.05)
   sampled=sample(a)
   if name=='rivers':
    width=1+(cfg[name]['river_thickness']-1)*sampled[:,:,1];alpha=np.clip(width/2+.5-sampled[:,:,0]/scale,0,1)
    clock=state.get('river_clock');progress=clock.sample() if clock else 1
    alpha*=np.clip((progress*1.005-sampled[:,:,2])/.005,0,1);ink(name,alpha)
   elif name=='streets':ink(name,((1+(cfg[name]['width']-1)*sampled[:,:,1])/2+.5-sampled[:,:,0]/scale)*cfg[name]['opacity'])
   else:ink(name,(cfg[name]['width']/2+.5-sampled/scale)*(cfg[name].get('opacity',1)))
 out[~valid]=0
 return np.rint(np.clip(out,0,255)).astype(np.uint8)
