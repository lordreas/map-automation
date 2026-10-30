import ctypes as C,json,time
from pathlib import Path
import numpy as np
from map_projection.skyline_store import Zstd,elevation_levels
bank=np.load('map_projection/.cache/horizons4096/horizons.npy',mmap_mode='r');h,w=bank.shape[1:];lib=C.CDLL(str(Path('map_projection/experiments/results/2026-10-25/skyline_compression.so').resolve()));fn=lib.simplify;fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_double,C.c_void_p];codec=Zstd();results={}
for tol in [.00035,.001,.005,.01,.05,.1]:
 size=0;total=0
 for yi in range(4):
  y=min(h-16,round((yi+.5)*h/4))
  for xi in range(4):
   x=min(w-256,round((xi+.5)*w/4));profiles=np.ascontiguousarray(bank[220:1221,y:y+16:2,x:x+256:2].reshape(1001,-1).T);n,width=profiles.shape;keep=np.zeros(profiles.shape,np.uint8);fn(profiles.ctypes.data,n,width,tol,keep.ctypes.data)
   rows,cols=np.nonzero(keep);counts=np.bincount(rows,minlength=n).astype(np.uint16);az=cols.astype(np.uint16);az[1:]=np.where(rows[1:]==rows[:-1],cols[1:]-cols[:-1],cols[1:])
   values=profiles[rows,cols];code=np.rint(values.astype(np.float64)/float(np.float32(89.999))*262144-.5).astype(np.uint32);delta=code.copy();delta[1:]=np.where(rows[1:]==rows[:-1],code[1:]-code[:-1],code[1:]);zz=(delta<<1)^((delta.view(np.int32)>>31).view(np.uint32))
   payload=counts.tobytes()+az.view(np.uint8).reshape(-1,2).T.copy().tobytes()+zz.view(np.uint8).reshape(-1,4).T.copy().tobytes();size+=len(codec.compress(payload,6));total+=n
 results[str(tol)]={'projected_GB':size/total*h*w/1e9,'sample_bytes':size,'sample_profiles':total};print(tol,results[str(tol)],flush=True)
Path('map_projection/experiments/results/2026-10-25/skyline-adaptive-compressed.json').write_text(json.dumps(results,indent=2))
