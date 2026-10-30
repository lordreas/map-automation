"""Shared on-demand/scheduled native day compilation and bounded disk retention."""
import argparse
from datetime import date,datetime,time as daytime,timedelta,timezone
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid
from zoneinfo import ZoneInfo

BUDGET=10_000_000_000
DAY=re.compile(r'^\d{4}-\d{2}-\d{2}$')
PROGRESS=re.compile(r'native events tiles (\d+)/(\d+)')


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)


@lru_cache(maxsize=128)
def utc_days(local_day,zone,latitude=None,longitude=None):
    d=date.fromisoformat(local_day);z=ZoneInfo(zone)
    first=datetime.combine(d,daytime.min,z).astimezone(timezone.utc).date()
    last=(datetime.combine(d+timedelta(days=1),daytime.min,z)-timedelta(microseconds=1)).astimezone(timezone.utc).date()
    days=[(first+timedelta(days=n)).isoformat() for n in range((last-first).days+1)]
    if latitude is None or longitude is None:return tuple(days)
    # Night requires no visibility events. Sample at <=15-minute intervals
    # with a 2-degree conservative angular-motion margin to retain every
    # potentially sunlit UTC date, without baking yesterday's midnight alone.
    from .solar import position
    start=datetime.combine(d,daytime.min,z).astimezone(timezone.utc)
    end=datetime.combine(d+timedelta(days=1),daytime.min,z).astimezone(timezone.utc)
    needed={datetime.combine(d,daytime(12),z).astimezone(timezone.utc).date().isoformat()}
    tick=start
    while tick<=end:
        if position(tick.isoformat(),latitude,longitude)[1]['altitude']>-2:needed.add(tick.date().isoformat())
        if tick==end:break
        tick=min(end,tick+timedelta(minutes=15))
    return tuple(day for day in days if day in needed)


def complete(path):
    try:
        m=json.loads((Path(path)/'manifest.json').read_text())
        return bool(m.get('complete')) and all((Path(path)/name).is_file() for name in ('initial.npy','offsets.npy','times.npy','sorted-times.npy','sorted-cells.npy'))
    except (OSError,ValueError):return False


def size(path):return sum(p.stat().st_size for p in path.iterdir() if p.is_file() and not p.is_symlink())


def prune(root,max_bytes=BUDGET,pinned=()):
    """Only complete ISO-date event directories; never touch the horizon bank."""
    root=Path(root)
    with (root/'retention.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _prune_locked(root,max_bytes,pinned)


def _prune_locked(root,max_bytes,pinned):
    root=Path(root);entries=[]
    pinned=set(pinned)
    try:pinned.add(json.loads((root/'active.json').read_text())['day'])
    except (OSError,ValueError,KeyError):pass
    for p in root.iterdir():
        if DAY.fullmatch(p.name) and p.is_dir() and not p.is_symlink() and complete(p):
            entries.append((p.stat().st_mtime,p,size(p)))
    total=sum(n for _,_,n in entries);removed=[]
    for _,p,n in sorted(entries):
        if total<=max_bytes:break
        if p.name in pinned:continue
        shutil.rmtree(p);total-=n;removed.append(p.name)
    return {'bytes':total,'limit_bytes':max_bytes,'removed':removed}


class DayCompiler:
    def __init__(self,horizons,root,latitude=45.95,longitude=10.65,max_bytes=BUDGET):
        self.horizons=Path(horizons);self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.latitude,self.longitude=latitude,longitude;self.max_bytes=max_bytes
        self.status_path=self.root/'generation.json';self.pin_path=self.root/'active.json'

    def status(self):
        try:return json.loads(self.status_path.read_text())
        except (OSError,ValueError):return {'state':'idle','progress':0}

    def pin(self,day):
        atomic_json(self.pin_path,{'day':day})
        if (self.root/day).is_dir():os.utime(self.root/day,None)

    def load(self,path):
        from .horizon_cache import Events
        with (self.root/'retention.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            self.pin(Path(path).name)
            return Events(path)

    def ensure(self,day,extra_pins=(),stopped=None):
        date.fromisoformat(day)
        with (self.root/'compile.lock').open('a') as lock:
            # A single compiler across the app and the nightly systemd job.
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if stopped and stopped.wait(.25):return False
                    if not stopped:time.sleep(.25)
            for stale in self.root.glob('.building-*'):
                if stale.is_dir() and not stale.is_symlink():shutil.rmtree(stale)
            if complete(self.root/day):return True
            from .native_events import build
            build()  # Build the content-addressed executable before progress starts.
            temp=self.root/('.building-'+day+'-'+uuid.uuid4().hex)
            status={'state':'generating','date':day,'progress':0.,'stage':'Preparing skylines','started':datetime.now(timezone.utc).isoformat()}
            atomic_json(self.status_path,status)
            command=[os.sys.executable,'-m','map_projection.bake_events','--horizons',str(self.horizons),'--output',str(temp),'--date',day,'--latitude',str(self.latitude),'--longitude',str(self.longitude),'--threads','4']
            start=time.monotonic();process=None
            try:
                environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=environment)
                try:os.setpriority(os.PRIO_PROCESS,process.pid,10)
                except ProcessLookupError:pass
                for line in process.stdout:
                    print(line,end='',flush=True)
                    if stopped and stopped.is_set():process.terminate();break
                    match=PROGRESS.search(line)
                    if match:
                        done,total=map(int,match.groups());status.update(progress=.95*done/total,completed=done,total=total,stage='Computing shadow transitions' if done<total else 'Writing event tables')
                        atomic_json(self.status_path,status)
                if process.wait()!=0:raise RuntimeError('Native day compiler failed; see the system service journal')
                built=temp/day
                if not complete(built):raise RuntimeError('Compiler did not publish a complete day')
                destination=self.root/day
                if destination.exists():
                    if destination.is_symlink():raise RuntimeError('Refusing a symlink in the cache directory')
                    shutil.rmtree(destination)  # Only an incomplete cache, under the compiler lock.
                built.replace(destination)
                pins=set(extra_pins)|{day}
                try:pins.add(json.loads(self.pin_path.read_text())['day'])
                except (OSError,ValueError,KeyError):pass
                retention=prune(self.root,self.max_bytes,pins)
                os.sync()
                status.update(state='ready',progress=1.,stage='Ready',seconds=time.monotonic()-start,retention=retention)
                atomic_json(self.status_path,status)
                return True
            except Exception as exc:
                status.update(state='error',error=str(exc),stage='Generation failed');atomic_json(self.status_path,status)
                raise
            finally:
                if process and process.poll() is None:process.terminate();process.wait(timeout=10)
                shutil.rmtree(temp,ignore_errors=True)


class CacheManager:
    """One latest requested day set, serviced without blocking controls/rendering."""
    def __init__(self,compiler,wake):
        self.compiler=compiler;self.wake=wake;self.lock=threading.Lock();self.changed=threading.Event();self.stopped=threading.Event()
        self.requested=();self.failed=None;self.failed_days=();self.retry_at=0
        self.thread=threading.Thread(target=self.run,name='daily-cache-manager',daemon=True);self.thread.start()

    def request(self,days):
        days=tuple(dict.fromkeys(days))
        with self.lock:
            if days==self.requested or (self.failed and days==self.failed_days):return
            self.requested=days;self.failed=None;self.retry_at=time.monotonic()+.5
        self.changed.set()

    def reset_failure(self):
        with self.lock:self.failed=None;self.failed_days=()

    def run(self):
        while not self.stopped.is_set():
            self.changed.wait(.5);self.changed.clear()
            with self.lock:days=self.requested;retry_at=self.retry_at
            if not days or time.monotonic()<retry_at:continue
            try:
                for day in days:
                    with self.lock:
                        if days!=self.requested:break
                    if not self.compiler.ensure(day,days,self.stopped):break
                else:
                    with self.lock:
                        if self.requested==days:self.requested=()
                    self.wake()
            except Exception as exc:
                with self.lock:self.failed=str(exc);self.failed_days=days;self.requested=()
                self.wake()

    def close(self):self.stopped.set();self.changed.set()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--horizons',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('--date');p.add_argument('--next-day',action='store_true');p.add_argument('--max-bytes',type=int,default=BUDGET)
    a=p.parse_args();c=json.loads(a.config.read_text());sun=c['sun'];zone=sun['timezone']
    selected=a.date or (datetime.now(ZoneInfo(zone)).date()+timedelta(days=1 if a.next_day else 0)).isoformat()
    days=utc_days(selected,zone,sun['latitude'],sun['longitude']);compiler=DayCompiler(a.horizons,a.events,sun['latitude'],sun['longitude'],a.max_bytes)
    for day in days:compiler.ensure(day,days)
    pins=set(days)
    try:pins.add(json.loads(compiler.pin_path.read_text())['day'])
    except (OSError,ValueError,KeyError):pass
    with (compiler.root/'compile.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);print(json.dumps(prune(compiler.root,a.max_bytes,pins)))


if __name__=='__main__':main()
