"""Lossless, independently compressed geographic skyline tiles.

Recover exact binary-search elevation codes where possible, predict across
azimuth and geographic neighbours, byte-shuffle, then Zstandard. Other float
patterns use a reversible raw-bit fallback. Every source float32 bit is retained.
Decode is bounded to one tile; an LRU bounds RAM.
Uses the system libzstd, with no Python-package dependency.
"""
import ctypes as C
import ctypes.util
from collections import OrderedDict
import json,time
from pathlib import Path
import numpy as np

class Zstd:
    def __init__(self):
        name=C.util.find_library('zstd')
        if not name:raise RuntimeError('Install the system libzstd runtime')
        self.lib=C.CDLL(name)
        for name,args,ret in [('ZSTD_compressBound',[C.c_size_t],C.c_size_t),('ZSTD_compress',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int],C.c_size_t),('ZSTD_decompress',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],C.c_size_t),('ZSTD_isError',[C.c_size_t],C.c_uint)]:
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=ret
    def compress(self,data,level=6):
        output=C.create_string_buffer(self.lib.ZSTD_compressBound(len(data)))
        n=self.lib.ZSTD_compress(output,len(output),data,len(data),level)
        if self.lib.ZSTD_isError(n):raise RuntimeError('Zstandard compression failed')
        return output.raw[:n]
    def decompress(self,data,size):
        output=C.create_string_buffer(size);n=self.lib.ZSTD_decompress(output,size,data,len(data))
        if self.lib.ZSTD_isError(n) or n!=size:raise ValueError('Corrupt skyline tile')
        return output.raw[:n]

def encode(array):
    bits=np.ascontiguousarray(array,dtype='<f4').view('<u4')
    delta=bits.copy();delta[1:]=bits[1:]^bits[:-1]
    return delta.view(np.uint8).reshape(-1,4).T.copy().tobytes()

def decode(data,shape):
    bits=np.frombuffer(data,np.uint8).reshape(4,-1).T.copy().view('<u4').reshape(shape)
    np.bitwise_xor.accumulate(bits,axis=0,out=bits)
    return bits.view('<f4')

_levels=None
def elevation_levels():
    global _levels
    if _levels is None:
        codes=np.arange(262144,dtype=np.uint32);lo=np.zeros(len(codes),np.float32);hi=np.full(len(codes),89.999,np.float32)
        for bit in range(17,-1,-1):
            mid=(lo+hi)*np.float32(.5);take=(codes&(1<<bit))!=0;lo=np.where(take,mid,lo);hi=np.where(take,hi,mid)
        _levels=(lo+hi)*np.float32(.5)
    return _levels

def encode_predicted(array):
    a=np.ascontiguousarray(array,dtype='<f4');base=a.view('<u4');flag=0
    if np.isfinite(a).all() and a.min()>=0 and a.max()<89.999:
        codes=np.rint(a.astype(np.float64)/float(np.float32(89.999))*262144-.5).astype(np.uint32)
        if codes.max()<262144 and np.array_equal(elevation_levels()[codes].view(np.uint32),base):base=codes;flag=1
    d=base.copy()
    for axis in (0,2):
        after=[slice(None)]*3;before=after.copy();after[axis]=slice(1,None);before[axis]=slice(None,-1)
        d[tuple(after)]=d[tuple(after)]-d[tuple(before)]
    zigzag=(d<<1)^((d.view(np.int32)>>31).view(np.uint32))
    return zigzag.view(np.uint8).reshape(-1,4).T.copy().tobytes(),flag

def decode_predicted(data,shape,flag):
    u=np.frombuffer(data,np.uint8).reshape(4,-1).T.copy().view('<u4').reshape(shape)
    u=(u>>1)^(np.uint32(0)-(u&1))
    for axis in (2,0):np.cumsum(u,axis=axis,dtype=np.uint32,out=u)
    if flag==1:
        if u.max()>=262144:raise ValueError('Corrupt skyline elevation code')
        return elevation_levels()[u]
    if flag!=0:raise ValueError('Unknown skyline predictor')
    return u.view('<f4')

class CompressedHorizons:
    def __init__(self,path,budget=256*1024**2):
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text())
        if self.meta.get('storage') not in ('zstd-xor-shuffle-v1','zstd-predict-v2') or not self.meta.get('complete'):raise ValueError('Incomplete or unsupported skyline store')
        self.codec=Zstd();self.shape=tuple(self.meta['shape']);self.angles=self.meta['angles'];self.cache=OrderedDict();self.bytes=0;self.budget=budget
        self.query_indices=None;self.query_groups=None;self.query_planes={}
        self.index=np.load(self.path/'index.npy',mmap_mode='r');self.stream=(self.path/'tiles.zst').open('rb')
        self.tile_h,self.tile_w=self.meta['tile_shape'];self.columns=(self.shape[1]+self.tile_w-1)//self.tile_w
    def tile(self,i):
        if i in self.cache:self.cache.move_to_end(i);return self.cache[i]
        row,col=divmod(i,self.columns);h=min(self.tile_h,self.shape[0]-row*self.tile_h);w=min(self.tile_w,self.shape[1]-col*self.tile_w)
        offset,size=map(int,self.index[i,:2]);self.stream.seek(offset);packed=self.stream.read(size)
        raw=self.codec.decompress(packed,self.angles*h*w*4)
        data=decode_predicted(raw,(self.angles,h,w),int(self.index[i,2])) if self.meta['storage']=='zstd-predict-v2' else decode(raw,(self.angles,h,w))
        while self.cache and self.bytes+data.nbytes>self.budget:
            _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
        if data.nbytes<=self.budget:self.cache[i]=data;self.bytes+=data.nbytes
        return data
    def at(self,azimuth,indices=None):
        q=((float(azimuth)%360)-self.meta['azimuth_start'])/self.meta['angular_step_degrees']
        if not 0<=q<=self.angles-1:raise ValueError('Azimuth is outside the stored skyline sector')
        lo=int(np.floor(q));hi=min(lo+1,self.angles-1);f=q-lo
        ix=np.arange(np.prod(self.shape),dtype=np.int64) if indices is None else np.asarray(indices,dtype=np.int64)
        same=self.query_indices is not None and np.array_equal(ix,self.query_indices)
        if not same:
            y,x=np.divmod(ix,self.shape[1]);ids=y//self.tile_h*self.columns+x//self.tile_w
            self.query_groups=[]
            for tile_id in np.unique(ids):
                selected=np.flatnonzero(ids==tile_id)
                self.query_groups.append((int(tile_id),selected,y[selected]%self.tile_h,x[selected]%self.tile_w))
            self.query_indices=ix.copy();self.query_planes={}
        for angle in (lo,hi):
            if angle not in self.query_planes:
                plane=np.empty(ix.shape,np.float32)
                for tile_id,selected,yy,xx in self.query_groups:plane[selected]=self.tile(tile_id)[angle,yy,xx]
                self.query_planes[angle]=plane
        result=self.query_planes[lo]*(1-f)+self.query_planes[hi]*f
        self.query_planes={k:v for k,v in self.query_planes.items() if k in (lo,hi)}
        return result
    def at_many(self,azimuths,indices):
        """Different azimuth per cell, without discarding the main sweep cache."""
        ix=np.asarray(indices,np.int64);q=((np.asarray(azimuths,np.float64)%360)-self.meta['azimuth_start'])/self.meta['angular_step_degrees']
        if q.shape!=ix.shape or np.any((q<0)|(q>self.angles-1)):raise ValueError('Invalid skyline batch')
        lo=np.floor(q).astype(np.int64);hi=np.minimum(lo+1,self.angles-1);fraction=q-lo;f=fraction.astype(np.float32);left=(1-fraction).astype(np.float32)
        y,x=np.divmod(ix,self.shape[1]);ids=y//self.tile_h*self.columns+x//self.tile_w;result=np.empty(ix.shape,np.float32)
        for tile_id in np.unique(ids):
            selected=np.flatnonzero(ids==tile_id);tile=self.tile(int(tile_id));yy=y[selected]%self.tile_h;xx=x[selected]%self.tile_w
            result[selected]=tile[lo[selected],yy,xx]*left[selected]+tile[hi[selected],yy,xx]*f[selected]
        return result
    def close(self):self.stream.close();self.cache.clear();self.query_planes.clear();self.query_indices=None;self.query_groups=None;self.bytes=0


def pack(source,destination,start=55.,end=305.,tile_shape=(16,256),level=6):
    source=Path(source);out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Choose a new destination directory')
    meta=json.loads((source/'manifest.json').read_text())
    if not meta.get('complete'):raise ValueError('Source skyline bank is incomplete')
    a=np.load(source/'horizons.npy',mmap_mode='r');step=360/a.shape[0];first=round(start/step);last=round(end/step)
    if start!=first*step or end!=last*step or not 0<=first<last<a.shape[0]:raise ValueError('Sector endpoints must be source azimuth samples')
    h,w=a.shape[1:];th,tw=tile_shape;codec=Zstd();index=[];offset=0;t=time.monotonic()
    manifest={**meta,'complete':False,'storage':'zstd-predict-v2','angles':last-first+1,'azimuth_start':start,'azimuth_end':end,'angular_step_degrees':step,'shape':[h,w],'tile_shape':list(tile_shape),'level':level,'lossless_relative_to':'source float32 samples'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (out/'tiles.zst').open('wb') as stream:
        for y in range(0,h,th):
            for x in range(0,w,tw):
                data=np.ascontiguousarray(a[first:last+1,y:y+th,x:x+tw]);encoded,flag=encode_predicted(data);packed=codec.compress(encoded,level)
                # Verify every encoded tile, not just a sample of the archive.
                restored=decode_predicted(codec.decompress(packed,len(encoded)),data.shape,flag)
                if not np.array_equal(restored.view(np.uint32),data.view(np.uint32)):raise RuntimeError('Lossless tile verification failed')
                stream.write(packed);index.append((offset,len(packed),flag));offset+=len(packed)
            print(f'skyline compression rows {min(y+th,h)}/{h}, {offset/1e9:.3f} GB, {time.monotonic()-t:.1f}s',flush=True)
    np.save(out/'index.npy',np.asarray(index,np.uint64));manifest.update(complete=True,bytes=offset,generation_seconds=time.monotonic()-t)
    tmp=out/'manifest.tmp';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(out/'manifest.json');return manifest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=55);p.add_argument('--end',type=float,default=305);p.add_argument('--level',type=int,default=6)
    args=p.parse_args();print(json.dumps(pack(args.source,args.output,args.start,args.end,level=args.level),indent=2))
