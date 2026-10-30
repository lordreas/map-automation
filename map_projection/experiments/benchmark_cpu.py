"""Isolated CPU experiment; does not change the production renderer or settings."""
import copy,ctypes as ct,json,statistics,subprocess,time,os
from pathlib import Path
import numpy as np
from map_projection import config
from map_projection.engine import Engine
from map_projection.geometry import projector
from map_projection.solar import position
os.sched_setaffinity(0,{6,7})
root=Path(__file__).resolve().parent
subprocess.run(['gcc','-O3','-mcpu=native','-fopenmp','-fPIC','-shared',str(root/'fast_dem.c'),'-lm','-o',str(root/'fast_dem.so')],check=True)
f=ct.CDLL(str(root/'fast_dem.so'))
f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
f.transpose_f32.argtypes=[f32,ct.c_int,ct.c_int,f32]
f.horizon_rows_fast.argtypes=[f32,ct.c_int,ct.c_int,ct.c_double,ct.c_double,f64,f32]
f.prepare_normals.argtypes=[ct.c_int,ct.c_int,f32,ct.c_double,ct.c_double,f32]
f.prepare_lookup.argtypes=[ct.c_int,ct.c_int,f32,f64,f64,ct.c_double,ct.c_double,ct.c_int,ct.c_int,i32]
f.shade_cells.argtypes=[ct.c_int,f32,f32,f32,f64,ct.c_double,ct.c_double,u8]
f.gather.argtypes=[ct.c_int,i32,u8,u8]
c=config.load(config.ROOT/'settings.json');c['pattern']='calibration';c['sun'].update(playing=False,terrain='dem')
e=Engine(config.ROOT.parent/'stl_files/full_map_0.4mm.stl',c,dem_path=config.ROOT.parent/'elevation_tiles/elevation_43.5_48.4_4.9_16.4_shape_11697x20000_resamp_average_bboxmerge_v2.npy')
while e.processed!=e.revision:time.sleep(.05)
m,inv,o,key=e.geometry(c);height,width=e.depth.shape
print('Projection ready',flush=True)
results=[]
for grid in [512,1024,2048]:
 c['sun']['grid_width']=grid
 z,ms,ts=e.terrain.get(c,print);gh,gw=z.shape
 bias=c['sun']['bias_mm']*ts[0]/ms[0]
 horizon=np.empty_like(z);normals=np.empty((*z.shape,3),np.float32)
 lookup=np.empty((height,width),np.int32);light=np.empty(z.shape,np.uint8);rgb=np.empty((height,width,3),np.uint8)
 t=time.perf_counter();f.prepare_normals(gw,gh,z,*ts,normals);f.prepare_lookup(width,height,e.depth,inv,o,*ms,gw,gh,lookup)
 zt=np.empty((gw,gh),np.float32);ht=np.empty_like(zt);f.transpose_f32(z,gw,gh,zt)
 prep=time.perf_counter()-t
 suns=[position(f'2026-10-24T{h:02d}:00:00+02:00',45.95,10.65)[0] for h in [9,12,16]]
 baseline=[]
 for sun in suns:
  t=time.perf_counter();reference=e.native.live(e.depth,e.normals,inv,o,z,ms,ts,sun,c['sun']['ambient'],bias,True);baseline.append(time.perf_counter()-t)
 for workers in [1,2]:
  f.threads(workers);samples=[];diff=0
  for sun in suns:
   t=time.perf_counter()
   if abs(sun[0]/(ts[0]/gw))>=abs(sun[1]/(ts[1]/gh)):
    st=np.array([-sun[1],-sun[0],sun[2]],np.float64)
    f.horizon_rows_fast(zt,gh,gw,ts[1]/gh,ts[0]/gw,st,ht)
    f.transpose_f32(ht,gh,gw,horizon)
   else:f.horizon_rows_fast(z,gw,gh,ts[0]/gw,ts[1]/gh,sun,horizon)
   a=time.perf_counter()
   f.shade_cells(gw*gh,z,horizon,normals,sun,c['sun']['ambient'],bias,light);b=time.perf_counter()
   f.gather(width*height,lookup,light,rgb);end=time.perf_counter()
   samples.append([a-t,b-a,end-b,end-t])
   ref=e.native.live(e.depth,e.normals,inv,o,z,ms,ts,sun,c['sun']['ambient'],bias,True)
   diff=max(diff,int(np.abs(ref.astype(np.int16)-rgb.astype(np.int16)).max()))
  r={'grid':grid,'threads':workers,'prepare_seconds':prep,'baseline_ms':round(statistics.median(baseline)*1000,2),
     'stages_ms':dict(zip(['horizon','lighting','gather','total'],[round(statistics.median(s[i] for s in samples)*1000,2) for i in range(4)])),
     'max_ms':round(max(row[3] for row in samples)*1000,2),'cpu_affinity':[6,7],'max_grey_difference':diff,'rendered_sun_times':['09:00','12:00','16:00']}
  results.append(r);print(json.dumps(r),flush=True)
np.savez(root/'gpu-input.npz',z=z,horizon=horizon,normals=normals,lookup=lookup,light=light,sun=suns[-1],bias=bias,ambient=c['sun']['ambient'],terrain_size=np.array(ts))
(root/'cpu-tiled-results.json').write_text(json.dumps(results,indent=2)+'\n')
e.close()
