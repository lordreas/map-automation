"""Compare lossless tiled codecs and bounded-error skyline simplification."""
import argparse,ctypes as C,json,time,subprocess
from pathlib import Path
import numpy as np
from map_projection.skyline_store import Zstd,encode,decode

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 bank=np.load(a.source/'horizons.npy',mmap_mode='r');terrain=np.load(a.terrain,mmap_mode='r');h,w=terrain.shape;first,last=220,1220;angles=last-first+1;codec=Zstd();tiles=[];profiles=[];sea=[]
 for yi in range(4):
  y=min(h-16,round((yi+.5)*h/4))
  for xi in range(4):
   x=min(w-256,round((xi+.5)*w/4))
   data=np.ascontiguousarray(bank[first:last+1,y:y+16,x:x+256]);tiles.append(data)
   profiles.append(np.ascontiguousarray(data[:,::2,::2].reshape(angles,-1).T));sea.append(terrain[y:y+16:2,x:x+256:2].reshape(-1)==0)
 print('Sample loaded',flush=True);raw=sum(t.nbytes for t in tiles);methods={}
 for method in ['raw','byte_shuffle','azimuth_xor_shuffle']:
  for level in [3,6,12]:
   sizes=[];seconds=0.;decode_seconds=0.
   for tile in tiles:
    if method=='raw':data=tile.tobytes()
    elif method=='byte_shuffle':data=tile.view(np.uint8).reshape(-1,4).T.copy().tobytes()
    else:data=encode(tile)
    t=time.monotonic();packed=codec.compress(data,level);seconds+=time.monotonic()-t;sizes.append(len(packed));t=time.monotonic();restored=codec.decompress(packed,len(data));decode_seconds+=time.monotonic()-t;assert restored==data
   row={'fraction':sum(sizes)/raw,'projected_sector_GB':sum(sizes)/raw*angles*h*w*4/1e9,'sample_compress_seconds':seconds,'sample_decompress_seconds':decode_seconds,'tile_fractions':[s/t.nbytes for s,t in zip(sizes,tiles)]};methods[f'{method}_{level}']=row;print(method,level,json.dumps({k:v for k,v in row.items() if k!='tile_fractions'}),flush=True)
 profiles=np.concatenate(profiles);sea=np.concatenate(sea);n,width=profiles.shape
 source=Path(__file__).with_suffix('.cpp');libpath=a.output.parent/'skyline_compression.so';libpath.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(libpath)],check=True)
 lib=C.CDLL(str(libpath.resolve()));fn=lib.simplify;fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_double,C.c_void_p]
 simplified=[]
 for tol in [0.,.00035,.001,.005,.01,.05,.1]:
  keep=np.zeros(profiles.shape,np.uint8);t=time.monotonic();fn(profiles.ctypes.data,n,width,tol,keep.ctypes.data);counts=keep.sum(axis=1);elapsed=time.monotonic()-t
  errors=[]
  for i in range(n):
   ix=np.flatnonzero(keep[i]);estimate=np.interp(np.arange(width),ix,profiles[i,ix]);errors.append(float(np.max(np.abs(estimate-profiles[i]))))
  # uint16 azimuth index + float32 elevation, and uint32 profile offsets.
  mean=float(counts.mean());row={'tolerance_degrees':tol,'mean_knots':mean,'median_knots':float(np.median(counts)),'p95_knots':float(np.percentile(counts,95)),'mean_sea_level_knots':float(counts[sea].mean()) if sea.any() else None,'mean_other_knots':float(counts[~sea].mean()),'projected_uncompressed_GB':h*w*(mean*6+4)/1e9,'verified_max_sample_error_degrees':max(errors),'seconds':elapsed};simplified.append(row);print('adaptive',json.dumps(row),flush=True)
 result={'sample_tiles':len(tiles),'tile_shape':[16,256],'sample_profiles':n,'sea_level_fraction':float(sea.mean()),'sector':[55,305],'sample_raw_bytes':raw,'lossless':methods,'simplification':simplified,'limitation':'Projected sizes use 16 geographic tiles. Error bounds concern existing stored azimuth samples, not the unsampled true skyline.'};a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
