"""Time the optional batched compiler and compare against existing event data."""
import argparse,json,resource,time
from pathlib import Path
import numpy as np
from map_projection.horizon_cache import open_horizons,build_day,Events
def main():
 p=argparse.ArgumentParser();p.add_argument('--horizons',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bounded',action='store_true');a=p.parse_args()
 h=open_horizons(a.horizons);t=time.monotonic();e=build_day(h,a.destination,'2026-10-25',45.95,10.65,vectorized=True,bounded=a.bounded);seconds=time.monotonic()-t;r=Events(a.reference)
 np.testing.assert_array_equal(e.initial,r.initial);np.testing.assert_array_equal(e.offsets,r.offsets)
 errors=np.abs(e.times.astype(np.int64)-r.times.astype(np.int64));np.testing.assert_array_less(errors,51)
 result={'bounded':a.bounded,'seconds':seconds,'cells':len(e.initial),'events':e.meta['events'],'initial_and_offsets_exact':True,'max_timestamp_difference_ms':int(errors.max(initial=0)),'differing_timestamps':int(np.count_nonzero(errors)),'peak_rss_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));h.close()
if __name__=='__main__':main()
