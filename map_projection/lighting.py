"""Stable daily lighting exposure. Never normalize an individual animation frame."""
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import numpy as np
from .solar import position


def daily_peak(normals,sun):
 start=datetime.fromisoformat(sun['time']).astimezone(ZoneInfo(sun['timezone'])).replace(hour=0,minute=0,second=0,microsecond=0)
 directions=[]
 for minute in range(0,1440,5):
  vector,_=position((start+timedelta(minutes=minute)).isoformat(),sun['latitude'],sun['longitude'])
  if vector[2]>0:directions.append(vector)
 if not directions:return 1.
 # A deterministic normal sample bounds the working set on the 2 GB board.
 # Exposure is fixed for the whole local day, independent of playback speed.
 flat=np.asarray(normals).reshape(-1,3);step=max(1,(len(flat)+65535)//65536)
 sample=np.ascontiguousarray(flat[::step],np.float32);directions=np.ascontiguousarray(directions,np.float32).T
 peak=0.
 for begin in range(0,len(sample),2048):peak=max(peak,float((sample[begin:begin+2048]@directions).max(initial=0)))
 return max(.1,min(1.,peak))


def options(sun,peak):
 return (1/peak if sun['normalize'] else 1.,sun['shadow_floor_amount'] if sun['shadow_floor'] else 0.,1/2.2)


def apply(frame,valid,gain,floor,gamma):
 v=np.clip(frame.astype(np.float32)/255*gain,0,1)
 v=floor+(1-floor)*np.power(v,gamma)
 out=np.rint(v*255).astype(np.uint8);out[~valid]=0;return out


GLSL='''
uniform float light_gain;uniform float light_floor;uniform float light_gamma;
float exposure(float value){return light_floor+(1.-light_floor)*pow(clamp(value*light_gain,0.,1.),light_gamma);}
'''
