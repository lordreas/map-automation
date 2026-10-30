"""Build/invoke the standalone C daily compiler; metadata orchestration only."""
from datetime import date,datetime,timezone
import hashlib,json,math,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def build():
    source=ROOT/'native/event_compiler.c'
    signature=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target=ROOT/'.cache'/('event-compiler-'+signature)
    if target.is_file():return target
    cc=shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if not cc:raise RuntimeError('A C compiler is required to build the native daily compiler')
    target.parent.mkdir(exist_ok=True)
    temporary=target.with_name(target.name+'.tmp-'+str(os.getpid()))
    try:
        subprocess.run([cc,'-O3','-std=c11','-Wall','-Wextra','-ffp-contract=off','-fopenmp',str(source),'-o',str(temporary),'-lm','-ldl'],check=True)
        temporary.replace(target)
    finally:
        if temporary.exists():temporary.unlink()
    return target

def build_day(source,destination,day,latitude,longitude,step=1.,precision=.05,threads=4):
    """All numerical work, decompression, sorting and NPY writing are native.

    Bounds exclude only constant states at the same discrete sample times as
    the Python reference. This retains its sampling limitations, recorded in
    the manifest; native speed does not certify unsampled shadow episodes.
    """
    from .horizon_cache import Events,SCHEMA
    source=Path(source).resolve();out=Path(destination).resolve();selected=date.fromisoformat(day)
    if sys.byteorder!='little':raise RuntimeError('The native cache format requires a little-endian host')
    if not all(math.isfinite(v) for v in (latitude,longitude,step,precision)):raise ValueError('Nonfinite compiler parameter')
    if not .01<=step<=86400 or not 0<precision<=step:raise ValueError('Require 0.01 <= step <= 86400 and 0 < precision <= step')
    if not -90<=latitude<=90 or not -180<=longitude<=180:raise ValueError('Invalid solar location')
    if not isinstance(threads,int) or not 1<=threads<=64:raise ValueError('Threads must be an integer in [1,64]')
    meta=json.loads((source/'manifest.json').read_text())
    if not meta.get('complete') or meta.get('schema')!=SCHEMA:raise ValueError('Incomplete or unsupported skyline bank')
    storage=meta.get('storage');compressed=storage=='zstd-predict-v2'
    if storage and not compressed:raise ValueError('Native compiler supports raw horizons or zstd-predict-v2')
    if compressed:
        h,w=map(int,meta['shape']);angles=int(meta['angles']);th,tw=meta['tile_shape'];az_start=meta['azimuth_start'];az_step=meta['angular_step_degrees']
    else:
        raw=np.load(source/'horizons.npy',mmap_mode='r',allow_pickle=False)
        if raw.ndim!=3 or raw.dtype!=np.float32:raise ValueError('Native compiler requires an azimuth-major float32 bank')
        angles,h,w=raw.shape;th,tw=16,256;az_start=0.;az_step=360/angles;del raw
    if h*w>=2**32-1:raise ValueError('Terrain exceeds uint32 event format')
    out.mkdir(parents=True,exist_ok=True);manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('complete'):raise ValueError('Completed event cache already exists; use a new output directory')
    executable=build()
    manifest.write_text(json.dumps({'schema':SCHEMA,'complete':False,'compiler':'native-c-v1'}))
    args=[source,out,h,w,angles,th,tw,az_start,az_step,selected.year,selected.timetuple().tm_yday,latitude,longitude,step,precision,threads,int(compressed)]
    subprocess.run([str(executable),*map(str,args)],check=True)
    stats=json.loads((out/'native-stats.json').read_text())
    result={'schema':SCHEMA,'complete':True,'horizon_key':meta['key'],'shape':[h,w],'date':day,'start':datetime.combine(selected,datetime.min.time(),tzinfo=timezone.utc).isoformat(),'latitude':latitude,'longitude':longitude,'step_seconds':step,'timestamp_precision_seconds':precision,'angular_step_degrees':az_step,'interval_certified':False,'events':stats['events'],'solar_model':'noaa-fractional-year-subsecond-v2','compiler':'native-c-v1','native_stats':stats}
    temporary=out/'manifest.tmp';temporary.write_text(json.dumps(result,indent=2));temporary.replace(manifest)
    return Events(out)
