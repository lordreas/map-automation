"""Cold tile decode and warm angular-query measurements, including the LRU."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from map_projection.skyline_store import CompressedHorizons

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--iterations',type=int,default=1000);a=p.parse_args()
 h=CompressedHorizons(a.input,budget=256*1024**2);ix=np.arange(min(np.prod(h.shape),h.tile_h*h.shape[1]),dtype=np.int64);az=h.meta['azimuth_start']+20
 t=time.monotonic();h.at(az,ix);cold=time.monotonic()-t;timings=[]
 for i in range(a.iterations):
  t=time.monotonic();h.at(az+i*.00001,ix);timings.append(time.monotonic()-t)
 result={'indices':len(ix),'cold_query_ms':cold*1000,'warm_median_ms':float(np.median(timings))*1000,'warm_p95_ms':float(np.percentile(timings,95))*1000,'decoded_tile_bytes':h.bytes,'compressed_bytes':sum(p.stat().st_size for p in a.input.iterdir()),'iterations':a.iterations,'peak_rss_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));h.close()
if __name__=='__main__':main()
