"""Synthetic two-event-per-cell workload: measure replay, not cache accuracy."""
import argparse,ctypes as C,json,tempfile,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from map_projection.experiments.gpu_context import context
from map_projection.frames import SunScene,SunFrame
from map_projection.horizon_cache import Events

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--width',type=int,default=4096);p.add_argument('--step',type=float,default=1);p.add_argument('--frames',type=int,default=120);p.add_argument('--output',type=Path,required=True);a=p.parse_args();context()
 from map_projection.gpu import Renderer,texture,framebuffer
 from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,glFinish
 d=np.load(a.input);source=d['z'];h=round(a.width*source.shape[0]/source.shape[1]);w=a.width;z=np.zeros((h,w),np.float32);n=h*w
 oldh,oldw=source.shape;lookup=d['lookup'];safe=np.maximum(lookup,0);uv=np.stack((safe%oldw*w//oldw,safe//oldw*h//oldh),axis=-1).astype(np.uint16);uv[lookup<0]=65535
 normals=np.zeros((h,w,3),np.float32);normals[:,:,2]=1
 with tempfile.TemporaryDirectory() as folder:
  path=Path(folder);np.save(path/'initial.npy',np.zeros(n,np.uint8));np.save(path/'offsets.npy',np.arange(n+1,dtype=np.uint32)*2)
  times=np.empty((n,2),np.uint32);times[:,0]=8*3600*1000+np.arange(n,dtype=np.uint32)%3600000;times[:,1]=16*3600*1000+np.arange(n,dtype=np.uint32)%3600000;np.save(path/'times.npy',times.reshape(-1));order=np.argsort(times.reshape(-1));np.save(path/'sorted-times.npy',times.reshape(-1)[order]);np.save(path/'sorted-cells.npy',(order//2).astype(np.uint32));del times,order
  (path/'manifest.json').write_text(json.dumps({'complete':True,'shape':[h,w],'start':'2026-10-25T00:00:00+00:00'}));events=Events(path)
  scene=SunScene('synthetic-event-replay',z,normals,uv,tuple(d['terrain_size']),events);r=Renderer();oh,ow=lookup.shape;tex=texture(13,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,ow,oh);fb=framebuffer(tex);timings=[]
  for i in range(a.frames+3):
   stamp=(datetime(2026,10,25,16,tzinfo=timezone.utc)+timedelta(seconds=i*a.step)).isoformat();t=time.monotonic();r.draw(SunFrame(scene,np.array([0.,0.,1.]),0,0,stamp,True,'events'),fb,False);glFinish()
   if i>=3:timings.append((time.monotonic()-t)*1000)
  result={'synthetic':True,'grid':[w,h],'output':[ow,oh],'frames':a.frames,'median_ms':float(np.median(timings)),'p95_ms':float(np.percentile(timings,95)),'cache_bytes':sum(p.stat().st_size for p in path.glob('*.npy'))};a.output.write_text(json.dumps(result,indent=2));print(result);r.close()
if __name__=='__main__':main()
