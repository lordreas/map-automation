"""Calibration-independent terrain skylines and compact daily visibility events.

Horizon files carry their angular resolution explicitly: interpolation is an
approximation, not a claim of subsecond geometric accuracy. Event bisection
controls timestamp precision separately from that angular approximation.
"""
import hashlib,json,os
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
from .solar import position,positions_for_day

SCHEMA=1

def identity(z,physical,bias):
    h=hashlib.sha256(np.ascontiguousarray(z,np.float32).tobytes())
    h.update(json.dumps([SCHEMA,list(map(float,physical)),float(bias),'bilinear-point-sun']).encode())
    return h.hexdigest()

class Horizons:
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text())
        if self.meta['schema']!=SCHEMA:raise ValueError('Unsupported horizon cache')
        if not self.meta.get('complete'):raise ValueError('Horizon cache is incomplete')
        self.data=np.load(self.path/'horizons.npy',mmap_mode='r');self.angles=self.data.shape[0]
    def at(self,azimuth,indices=None):
        q=(azimuth%360)*self.angles/360;i=int(np.floor(q));f=q-i
        a=self.data[i].reshape(-1);b=self.data[(i+1)%self.angles].reshape(-1)
        if indices is not None:a=a[indices];b=b[indices]
        return np.asarray(a,np.float32)*(1-f)+np.asarray(b,np.float32)*f
    def at_many(self,azimuths,indices):
        ix=np.asarray(indices,np.int64);q=(np.asarray(azimuths,np.float64)%360)*self.angles/360;i=np.floor(q).astype(np.int64);fraction=q-i;f=fraction.astype(np.float32);left=(1-fraction).astype(np.float32)
        y,x=np.divmod(ix,self.data.shape[2])
        return self.data[i,y,x]*left+self.data[(i+1)%self.angles,y,x]*f

def open_horizons(path,budget=256*1024**2):
    meta=json.loads((Path(path)/'manifest.json').read_text())
    if meta.get('storage') in ('zstd-xor-shuffle-v1','zstd-predict-v2'):
        from .skyline_store import CompressedHorizons
        return CompressedHorizons(path,budget)
    return Horizons(path)

class Events:
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text())
        if not self.meta.get('complete'):raise ValueError('Daily event cache is incomplete')
        self.initial=np.load(self.path/'initial.npy',mmap_mode='r')
        self.offsets=np.load(self.path/'offsets.npy',mmap_mode='r')
        self.times=np.load(self.path/'times.npy',mmap_mode='r')
    def visibility(self,seconds):
        # CPU diagnostic/preview. Physical playback uses the GPU implementation.
        result=self.initial.astype(bool).copy()
        for i in range(len(result)):
            n=np.searchsorted(self.times[self.offsets[i]:self.offsets[i+1]],seconds*1000,side='right')
            result[i]^=bool(n%2)
        return result.reshape(self.meta['shape'])

def build_day(horizons,destination,day,latitude,longitude,step=1.,precision=.05,chunk=65536,vectorized=False,bounded=False):
    """Stream per-cell event lists; memory bounded independently of day length.

All step-spaced states are evaluated. Transitions within each bracket are
bisected. Episodes entirely between brackets can be missed: step and precision
are both recorded and must not be confused with certified interval bounds.
"""
    start=datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    if start.hour or start.minute:raise ValueError('Use a date at UTC midnight')
    if not 0<precision<=step or step>86400:raise ValueError('Require 0 < precision <= step <= 86400')
    if bounded and not vectorized:raise ValueError('Bounded sweep requires vectorized queries')
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('complete'):
        raise ValueError('Completed event cache already exists; use a new output directory')
    manifest.write_text(json.dumps({'schema':SCHEMA,'complete':False}))
    shape=horizons.shape if hasattr(horizons,'shape') else horizons.data.shape[1:];n=int(np.prod(shape));grid=np.arange(0,86400+step,step);grid=np.minimum(grid,86400)
    if vectorized:solar=np.stack(positions_for_day(day,grid,latitude,longitude),axis=-1)
    else:
        solar=[]
        for t in grid:
            _,s=position((start+timedelta(seconds=float(t))).isoformat(),latitude,longitude);solar.append((s['azimuth'],s['altitude']))
        solar=np.array(solar)
    initial=np.lib.format.open_memmap(out/'initial.npy',mode='w+',dtype=np.uint8,shape=(n,))
    offsets=np.lib.format.open_memmap(out/'offsets.npy',mode='w+',dtype=np.uint32,shape=(n+1,));total=0
    raw=out/'times.raw'
    with raw.open('wb') as stream:
        for base in range(0,n,chunk):
            ix=np.arange(base,min(base+chunk,n));previous=(solar[0,1]>horizons.at(solar[0,0],ix)) if solar[0,1]>0 else np.zeros(len(ix),bool)
            initial[ix]=previous;lists=[[] for _ in ix]
            block_end=0;active=np.arange(len(ix))
            for k in range(1,len(grid)):
                if bounded and k>=block_end:
                    block_end=min(len(grid),k+max(1,int(60/step)))
                    window=solar[k-1:block_end];daylight=window[:,1]>0
                    if not daylight.any():active=np.empty(0,np.int64)
                    else:
                        az_min=float(window[daylight,0].min());az_max=float(window[daylight,0].max())
                        angular_step=horizons.meta.get('angular_step_degrees',360/horizons.angles)
                        first=(np.floor(az_min/angular_step)+1)*angular_step
                        angles=[az_min,az_max,*np.arange(first,az_max,angular_step)]
                        low=np.full(len(ix),np.inf,np.float32);high=np.full(len(ix),-np.inf,np.float32)
                        for angle in angles:
                            value=horizons.at(float(angle),ix);np.minimum(low,value,out=low);np.maximum(high,value,out=high)
                        # Float32 interpolation can overshoot an endpoint by a
                        # few ULPs; expand bounds before excluding any receiver.
                        for _ in range(4):low=np.nextafter(low,-np.inf,dtype=np.float32);high=np.nextafter(high,np.inf,dtype=np.float32)
                        all_lit=high<np.float32(window[:,1].min())
                        all_dark=low>=np.float32(window[:,1].max())
                        active=np.flatnonzero(~(all_lit|all_dark))
                if bounded:
                    if not len(active):continue
                    az,el=solar[k];current=previous.copy()
                    current[active]=(el>horizons.at(az,ix[active])) if el>0 else False
                else:
                    az,el=solar[k];current=(el>horizons.at(az,ix)) if el>0 else np.zeros(len(ix),bool)
                changed=np.flatnonzero(current!=previous)
                if len(changed):
                    lo=np.full(len(changed),grid[k-1]);hi=np.full(len(changed),grid[k]);before=previous[changed]
                    while np.max(hi-lo)>precision:
                        mid=(hi+lo)/2;states=[]
                        # Different crossing times require different solar queries.
                        states=np.zeros(len(mid),bool)
                        if vectorized:
                            azimuth,altitude=positions_for_day(day,mid,latitude,longitude)
                            above=altitude>0
                            # Match the scalar float-vs-float32-array comparison
                            # used by the reference compiler, including rounding.
                            states[above]=altitude[above].astype(np.float32)>horizons.at_many(azimuth[above],ix[changed[above]])
                        else:
                            for t in np.unique(mid):
                                group=np.flatnonzero(mid==t)
                                _,s=position((start+timedelta(seconds=float(t))).isoformat(),latitude,longitude)
                                states[group]=(s['altitude']>horizons.at(s['azimuth'],ix[changed[group]])) & (s['altitude']>0)
                        same=np.array(states)==before;lo=np.where(same,mid,lo);hi=np.where(same,hi,mid)
                    for j,t in zip(changed,(lo+hi)/2):lists[j].append(round(t*1000))
                previous=current
            if vectorized:
                counts=np.fromiter((len(values) for values in lists),dtype=np.uint32,count=len(lists))
                offsets[base:base+len(ix)]=total+np.cumsum(counts,dtype=np.uint32)-counts
                np.fromiter((t for values in lists for t in values),dtype=np.uint32).tofile(stream);total+=int(counts.sum())
            else:
                for j,values in enumerate(lists):
                    offsets[base+j]=total;a=np.asarray(values,np.uint32);a.tofile(stream);total+=len(a)
            print('daily events',base+len(ix),'/',n,flush=True)
    offsets[n]=total;initial.flush();offsets.flush()
    times=np.lib.format.open_memmap(out/'times.npy',mode='w+',dtype=np.uint32,shape=(total,))
    if total:times[:]=np.memmap(raw,dtype=np.uint32,mode='r',shape=(total,))
    times.flush();raw.unlink()
    order=np.argsort(times,kind='stable')
    counts=np.diff(offsets.astype(np.int64));cells=np.repeat(np.arange(n,dtype=np.uint32),counts)
    np.save(out/'sorted-times.npy',np.asarray(times)[order]);np.save(out/'sorted-cells.npy',cells[order])
    del order,cells
    meta={'schema':SCHEMA,'complete':True,'horizon_key':horizons.meta['key'],'shape':list(shape),'date':day,'start':start.isoformat(),'latitude':latitude,'longitude':longitude,'step_seconds':step,'timestamp_precision_seconds':precision,'angular_step_degrees':horizons.meta.get('angular_step_degrees',360/horizons.angles),'interval_certified':False,'events':total,'solar_model':'noaa-fractional-year-subsecond-v2','compiler':'batched-bounded-v1' if bounded else 'batched-v1' if vectorized else 'scalar-v1'}
    temp=out/'manifest.tmp';temp.write_text(json.dumps(meta,indent=2));temp.replace(out/'manifest.json');return Events(out)
