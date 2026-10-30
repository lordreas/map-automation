"""Time a complete on-board day bake, including flushing it to storage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--horizons',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--date',required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--wait-seconds',type=float,default=0)
    a=p.parse_args()
    deadline=time.monotonic()+a.wait_seconds
    while True:
        try:
            meta=json.loads((a.horizons/'manifest.json').read_text())
            if meta.get('complete') and (a.horizons/'tiles.zst').is_file():break
        except (FileNotFoundError,json.JSONDecodeError):pass
        if time.monotonic()>=deadline:raise RuntimeError('Complete horizon cache not available')
        time.sleep(5)
    start=time.monotonic()
    subprocess.run([sys.executable,'-m','map_projection.bake_events','--horizons',str(a.horizons),'--output',str(a.output),'--date',a.date,'--threads',str(a.threads)],check=True)
    compile_seconds=time.monotonic()-start
    os.sync()
    wall_seconds=time.monotonic()-start
    day=a.output/a.date
    result={'date':a.date,'threads':a.threads,'command_seconds':compile_seconds,'seconds_including_storage_flush':wall_seconds,'native':json.loads((day/'native-stats.json').read_text()),'peak_child_rss_KiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'bytes':sum(f.stat().st_size for f in day.iterdir() if f.is_file()),'sha256':{}}
    for name in ['initial.npy','offsets.npy','times.npy','sorted-times.npy','sorted-cells.npy']:
        digest=hashlib.sha256()
        with (day/name).open('rb') as source:
            for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
        result['sha256'][name]=digest.hexdigest()
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
