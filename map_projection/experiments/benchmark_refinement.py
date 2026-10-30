"""Validate isolated-highlight refinement and time fresh 4K frames offscreen."""
import time,json,ctypes as C,argparse
from pathlib import Path
import numpy as np
from map_projection.experiments.gpu_context import context
from map_projection.frames import SunScene,SunFrame
from map_projection.solar import position
from map_projection.experiments.benchmark_gpu_shadows import dda
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--grid',type=int,default=2048);parser.add_argument('--trials',type=int,default=5)
args=parser.parse_args()
ctx=context()
from map_projection.gpu import Renderer,texture,framebuffer
from OpenGL.GLES3 import *
data=np.load(args.input);source=data['z'];ts=data['terrain_size'];original=data['lookup'];oldh,oldw=source.shape
width=args.grid;height=round(width*oldh/oldw)
yi=np.minimum(((np.arange(height)+.5)*oldh/height).astype(int),oldh-1)
xi=np.minimum(((np.arange(width)+.5)*oldw/width).astype(int),oldw-1)
z=np.ascontiguousarray(source[np.ix_(yi,xi)]);gh,gw=z.shape;oh,ow=original.shape
lookup=np.where(original<0,-1,(np.maximum(original,0)//oldw*height//oldh)*width+np.maximum(original,0)%oldw*width//oldw).astype(np.int32)
ny,nx=np.gradient(z,ts[1]/gh,ts[0]/gw);normals=np.stack((-nx,ny,np.ones_like(z)),axis=-1);normals=np.ascontiguousarray(normals/np.linalg.norm(normals,axis=-1,keepdims=True),dtype=np.float32)
uv=np.stack((np.maximum(lookup,0)%gw,np.maximum(lookup,0)//gw),axis=-1).astype(np.uint16);uv[lookup<0]=65535
scene=SunScene('firefly-check',z,normals,uv,tuple(ts));renderer=Renderer();bias=float(data['bias']);ambient=float(data['ambient'])
tex=texture(9,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,ow,oh);fb=framebuffer(tex)
results=[];rng=np.random.default_rng(47)
for minute in [16*60,16*60+20,16*60+40,17*60,17*60+10]:
    stamp=f'2026-10-25T{minute//60:02}:{minute%60:02}:00+01:00';sun,angles=position(stamp,45.95,10.65);frame=SunFrame(scene,sun,ambient,bias,stamp)
    timings=[]
    for trial in range(args.trials+1):
        t=time.monotonic();renderer.draw(frame,fb,False);glFinish()
        if trial:timings.append((time.monotonic()-t)*1000)
    glBindFramebuffer(GL_FRAMEBUFFER,renderer.light_fb);before=np.empty((gh,gw),np.uint8);glReadPixels(0,0,gw,gh,GL_RED,GL_UNSIGNED_BYTE,before.ctypes.data_as(C.c_void_p))
    glBindFramebuffer(GL_FRAMEBUFFER,renderer.filtered_fb);rgba=np.empty((gh,gw,4),np.uint8);glReadPixels(0,0,gw,gh,GL_RGBA,GL_UNSIGNED_BYTE,rgba.ctypes.data_as(C.c_void_p));after=rgba[:,:,0]
    changed=np.argwhere(before!=after)
    direction=np.array([sun[0]/(ts[0]/gw),-sun[1]/(ts[1]/gh)]);dominant=max(abs(direction));rise=sun[2]/dominant;direction/=dominant
    coords=changed[rng.choice(len(changed),min(128,len(changed)),replace=False)] if len(changed) else []
    for y,x in coords:assert dda(z,int(x),int(y),direction,rise,bias),(stamp,int(x),int(y))
    row={'time':stamp,'altitude':angles['altitude'],'median_ms':round(float(np.median(timings)),3),'removed_false_highlights':len(changed),'exact_cpu_ray_checks':len(coords)}
    results.append(row);print(json.dumps(row),flush=True)
# Static sun repeated after different suns must be byte-stable.
renderer.draw(frame,fb,False);first=renderer.capture(True)
for _ in range(4):renderer.draw(frame,fb,False)
last=renderer.capture(True);assert np.array_equal(first,last)
out={'method':'fragment_compaction_and_hierarchical_compute_rays','renderer':renderer.renderer,'output':[ow,oh],'grid':[gw,gh],'results':results,'repeated_frame_identical':True}
args.output.write_text(json.dumps(out,indent=2)+'\n')
