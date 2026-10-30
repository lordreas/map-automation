"""Independent GPX overlay, registered through precise projector/STL hits."""
import colorsys
import ctypes as C
import hashlib
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'Genf__Salzburg.gpx'


def random_color():
    return '#'+''.join(f'{round(c*255):02x}' for c in colorsys.hsv_to_rgb(secrets.randbelow(360)/360,.85,1.))


def read_gpx(path):
    root=ET.parse(path).getroot();segments=[]
    groups=root.findall('.//{*}trkseg')+root.findall('.//{*}rte')
    for group in groups:
        points=[]
        for item in group:
            if item.tag.rsplit('}',1)[-1] not in ('trkpt','rtept'):continue
            try:
                lat,lon=float(item.attrib['lat']),float(item.attrib['lon'])
                if not (-90<=lat<=90 and -180<=lon<=180):raise ValueError
                points.append((lat,lon))
            except (KeyError,ValueError):
                if len(points)>1:segments.append(np.array(points,np.float64))
                points=[]
        if len(points)>1:segments.append(np.array(points,np.float64))
    if not segments:raise ValueError('GPX contains no usable track segments')
    return segments


def raster_track(segments,geo,width=4096,line_width=4.):
    height=round(width*geo['height_mm']/geo['width_mm'])
    lengths=[]
    for points in segments:
        p=np.radians(points);delta=np.diff(p,axis=0)
        a=np.sin(delta[:,0]/2)**2+np.cos(p[:-1,0])*np.cos(p[1:,0])*np.sin(delta[:,1]/2)**2
        lengths.append(np.r_[0,np.cumsum(2*6371008.8*np.arcsin(np.sqrt(np.clip(a,0,1))))])
    total=sum(p[-1] for p in lengths);total=max(total,1.)
    mask=Image.new('L',(width,height));phase=Image.new('F',(width,height))
    ink=ImageDraw.Draw(mask);clock=ImageDraw.Draw(phase);offset=0;lines=[]
    for points,distance in zip(segments,lengths):
        x=(points[:,1]-geo['lon_min'])/(geo['lon_max']-geo['lon_min'])*width-.5
        y=(geo['lat_max']-points[:,0])/(geo['lat_max']-geo['lat_min'])*height-.5
        progress=(offset+distance)/total;offset+=distance[-1]
        for i in range(len(points)-1):
            # Subdivide long GPS spans so the reveal advances by distance,
            # independent of the track recorder's sample rate.
            n=max(1,min(4096,math.ceil(max(abs(x[i+1]-x[i]),abs(y[i+1]-y[i]))/2)))
            for j in range(n):
                a,b=j/n,(j+1)/n
                line=(x[i]*(1-a)+x[i+1]*a,y[i]*(1-a)+y[i+1]*a,x[i]*(1-b)+x[i+1]*b,y[i]*(1-b)+y[i+1]*b)
                lines.append((line,float(progress[i]*(1-b)+progress[i+1]*b)))
    # Earliest arrival wins at self-crossings.
    stroke=max(1,round(line_width))
    for line,t in reversed(lines):ink.line(line,fill=255,width=stroke);clock.line(line,fill=t,width=stroke)
    alpha=np.asarray(mask,dtype=np.float32)/255
    weighted=np.asarray(phase)*alpha
    def blur(a,axis):
        out=a*.75
        if axis==0:out[:-1]+=.125*a[1:];out[1:]+=.125*a[:-1]
        else:out[:,:-1]+=.125*a[:,1:];out[:,1:]+=.125*a[:,:-1]
        return out
    for axis in (0,1):alpha=blur(alpha,axis);weighted=blur(weighted,axis)
    phase=np.divide(weighted,alpha,out=np.zeros_like(weighted),where=alpha>1e-6)
    return np.rint(alpha*255).astype(np.uint8),phase,total


def project_track(depth,inverse,origin,bounds,geo,mask,phase):
    source=ROOT/'native/track.c';key=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target=ROOT/'.cache'/('track-'+key+'.so');target.parent.mkdir(exist_ok=True)
    if not target.exists():
        cc=shutil.which('cc') or shutil.which('gcc')
        if not cc:raise RuntimeError('A C compiler is required for GPX projection')
        tmp=target.with_name(target.name+'.'+str(os.getpid())+'.tmp')
        try:subprocess.run([cc,'-O3','-std=c99','-shared','-fPIC',str(source),'-o',str(tmp),'-lm'],check=True);tmp.replace(target)
        finally:tmp.unlink(missing_ok=True)
    lib=C.CDLL(str(target));f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.project_track.argtypes=[C.c_int,C.c_int,f32,f64,f64,C.c_double,C.c_double,C.c_double,C.c_double,C.c_int,C.c_int,u8,f32,f32];lib.project_track.restype=C.c_int
    h,w=depth.shape;th,tw=mask.shape;buffer=np.empty((h*w,4),np.float32)
    n=lib.project_track(w,h,depth,np.ascontiguousarray(inverse),origin,(bounds[0]+bounds[3])/2,(bounds[1]+bounds[4])/2,geo['width_mm'],geo['height_mm'],tw,th,mask,phase,buffer)
    return buffer[:n].copy()


class TrackLayer:
    def __init__(self,path=SOURCE):
        self.path=Path(path);self.lock=threading.RLock();self.enabled=False;self.setting=None;self.color=random_color();self.started=None
        self.vertices=None;self.key=None;self.raster=None;self.raster_key=None;self.distance=0.;self.points=0;self.error=None;self.generation=0

    def configure(self,c):
        with self.lock:
            enabled=c['enabled'];setting=c['color']
            if enabled and (not self.enabled or setting!=self.setting):
                self.color=setting or random_color();self.started=None;self.generation+=1
            self.enabled=enabled;self.setting=setting

    def prepare(self,c,key,depth,inverse,origin,bounds):
        if not c['track']['enabled']:return
        signature=(key,tuple(sorted(c['geo'].items())),self.path.stat().st_mtime_ns,c['track'].get('width',4.))
        if signature==self.key:return
        raster_key=signature[1:]
        if raster_key!=self.raster_key:
            segments=read_gpx(self.path);mask,phase,distance=raster_track(segments,c['geo'],line_width=c['track'].get('width',4.))
            self.raster=(mask,phase);self.raster_key=raster_key;self.distance=distance;self.points=sum(len(s) for s in segments)
        vertices=project_track(depth,inverse,origin,bounds,c['geo'],*self.raster)
        with self.lock:self.vertices=vertices;self.key=signature;self.started=None;self.generation+=1;self.error=None

    def replay(self):
        with self.lock:
            if self.setting is None:self.color=random_color()
            self.started=None;self.generation+=1

    def state(self):
        with self.lock:
            if not self.enabled or self.vertices is None:return None
            progress=min(1.,(time.monotonic()-self.started)/2.0) if self.started is not None else 0.
            return {'key':self.key,'vertices':self.vertices,'color':self.color,'progress':progress,'generation':self.generation,'animating':progress<1}

    def presented(self):
        with self.lock:
            if self.enabled and self.vertices is not None and self.started is None:self.started=time.monotonic()

    def snapshot(self):
        with self.lock:
            return {'enabled':self.enabled,'color':self.color,'name':self.path.stem,'points':self.points,'distance_km':self.distance/1000,'ready':self.vertices is not None,'error':self.error,'animating':bool(self.enabled and self.vertices is not None and (self.started is None or time.monotonic()-self.started<2))}


def compose_track(frame,state):
    if state is None:return frame
    vertices=state['vertices'];reveal=state['progress']*1.008
    alpha=vertices[:,3]*np.clip((reveal-vertices[:,2])/.008,0,1)
    pick=alpha>0
    if not np.any(pick):return frame
    h,w=frame.shape[:2];v=vertices[pick];a=alpha[pick,None]
    x=np.clip(np.rint((v[:,0]+1)*w/2-.5).astype(int),0,w-1)
    y=np.clip(np.rint((1-v[:,1])*h/2-.5).astype(int),0,h-1)
    ink=np.array([int(state['color'][i:i+2],16) for i in (1,3,5)])
    out=frame.copy();out[y,x]=np.rint(out[y,x]*(1-a)+ink*a).astype(np.uint8)
    return out
