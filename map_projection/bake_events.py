"""Compile a UTC day's events from a reusable geographic horizon cache."""
import argparse
from pathlib import Path
from .horizon_cache import open_horizons,build_day
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--horizons',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--date',required=True);p.add_argument('--latitude',type=float,default=45.95);p.add_argument('--longitude',type=float,default=10.65);p.add_argument('--step',type=float,default=1);p.add_argument('--precision',type=float,default=.05)
 p.add_argument('--vectorized',action='store_true',help='Batch solar and skyline bisection queries; experimental optimization.')
 p.add_argument('--bounded',action='store_true',help='Also exclude receivers with constant visibility bounds; experimental.')
 p.add_argument('--compiler',choices=['native','python'],help='Daily compiler (native C by default; legacy vector flags select Python).')
 p.add_argument('--threads',type=int,default=min(4,__import__('os').cpu_count() or 1),help='Native worker threads, default up to four.')
 a=p.parse_args()
 if a.step<=0 or a.precision<=0:raise ValueError('Step and precision must be positive')
 compiler=a.compiler or ('python' if a.vectorized or a.bounded else 'native')
 if compiler=='native':
  if a.vectorized or a.bounded:raise ValueError('Python optimization flags require --compiler python')
  from .native_events import build_day as native_day
  native_day(a.horizons,a.output/a.date,a.date,a.latitude,a.longitude,a.step,a.precision,a.threads)
 else:build_day(open_horizons(a.horizons),a.output/a.date,a.date,a.latitude,a.longitude,a.step,a.precision,vectorized=a.vectorized or a.bounded,bounded=a.bounded)
