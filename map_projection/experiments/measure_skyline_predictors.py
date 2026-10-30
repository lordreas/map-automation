import numpy as np,json,time
from map_projection.skyline_store import Zstd
from pathlib import Path
source=np.load('map_projection/.cache/horizons4096/horizons.npy',mmap_mode='r');h,w=source.shape[1:];codec=Zstd();codes=np.arange(262144,dtype=np.uint32);lo=np.zeros(len(codes),np.float32);hi=np.full(len(codes),89.999,np.float32)
for bit in range(17,-1,-1):
 mid=(lo+hi)*np.float32(.5);take=(codes&(1<<bit))!=0;lo=np.where(take,mid,lo);hi=np.where(take,hi,mid)
table=(lo+hi)*np.float32(.5);totals={};errors=0;values=0;times={}
for yi in range(4):
 y=min(h-16,round((yi+.5)*h/4))
 for xi in range(4):
  x=min(w-256,round((xi+.5)*w/4));a=np.ascontiguousarray(source[220:1221,y:y+16,x:x+256]);idx=np.rint(a.astype(np.float64)/float(np.float32(89.999))*262144-.5).astype(np.uint32)
  errors+=int(np.count_nonzero(a.view(np.uint32)!=table[idx].view(np.uint32)));values+=a.size
  for name,base in [('floatbits',a.view(np.uint32)),('codes',idx)]:
   for axes in [(),(0,),(2,),(1,2),(0,2),(0,1,2)]:
    d=base.copy()
    for axis in axes:
     after=[slice(None)]*3;before=after.copy();after[axis]=slice(1,None);before[axis]=slice(None,-1);d[tuple(after)]=d[tuple(after)]-d[tuple(before)]
    if axes:d=(d<<1)^((d.view(np.int32)>>31).view(np.uint32))
    shuffled=d.view(np.uint8).reshape(-1,4).T.copy().tobytes();t=time.monotonic();packed=codec.compress(shuffled,6);k=name+str(axes);totals[k]=totals.get(k,0)+len(packed);times[k]=times.get(k,0)+time.monotonic()-t
 print(yi,'row complete',flush=True)
rows={k:{'projected_GB':v/(values*4)*h*w*1001*4/1e9,'fraction':v/(values*4),'sample_compress_seconds':times[k]} for k,v in totals.items()}
result={'values_tested':values,'lookup_reconstruction_errors':errors,'predictors':rows};print(json.dumps(result,indent=2));Path('map_projection/experiments/results/2026-10-25/skyline-predictors.json').write_text(json.dumps(result,indent=2))
