"""Wait for the resumable horizon bake, then compile one UTC day."""
import argparse,json,time
from pathlib import Path
from .native_events import build_day
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--horizons',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--date',required=True);p.add_argument('--latitude',type=float,default=45.95);p.add_argument('--longitude',type=float,default=10.65);a=p.parse_args()
 while True:
  try:ready=json.loads((a.horizons/'manifest.json').read_text()).get('complete')
  except (FileNotFoundError,json.JSONDecodeError):ready=False
  if ready:break
  time.sleep(20)
 build_day(a.horizons,a.output/a.date,a.date,a.latitude,a.longitude)
