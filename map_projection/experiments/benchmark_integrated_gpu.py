"""Exercise the actual application GPU renderer without taking over HDMI."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np
from map_projection.experiments.gpu_context import context
from map_projection.frames import SunScene,SunFrame
from map_projection.solar import position


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--trials',type=int,default=25)
    args=parser.parse_args();ctx=context()
    from map_projection.gpu import Renderer,texture,framebuffer
    from OpenGL.GLES3 import glFinish,glBindFramebuffer,glReadPixels,glDeleteFramebuffers,glDeleteTextures,GL_FRAMEBUFFER,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE
    data=np.load(args.input);z=np.ascontiguousarray(data['z']);ts=data['terrain_size'];lookup=data['lookup'];gh,gw=z.shape;oh,ow=lookup.shape
    dy,dx=np.gradient(z,ts[1]/gh,ts[0]/gw);normals=np.stack((-dx,dy,np.ones_like(z)),axis=-1)
    normals=np.ascontiguousarray(normals/np.linalg.norm(normals,axis=-1,keepdims=True),dtype=np.float32)
    uv=np.stack((np.maximum(lookup,0)%gw,np.maximum(lookup,0)//gw),axis=-1).astype(np.uint16);uv[lookup<0]=65535
    scene=SunScene('land-aware-integrated',z,normals,uv,tuple(ts));renderer=Renderer()
    tex=texture(8,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,ow,oh);fb=framebuffer(tex)
    frames=[];per_sun=[]
    for hour in ['09','12','16','18']:
        sun,_=position(f'2026-10-24T{hour}:00:00+02:00',45.95,10.65)
        frame=SunFrame(scene,sun,float(data['ambient']),float(data['bias']),hour);samples=[]
        for trial in range(args.trials+1):
            t=time.perf_counter();renderer.draw(frame,fb,physical=False);glFinish();elapsed=(time.perf_counter()-t)*1000
            if trial:samples.append(elapsed);frames.append(elapsed)
        per_sun.append({'hour':hour,'median_ms':round(float(np.median(samples)),3),'max_ms':round(max(samples),3)})
    rgba=np.empty((oh,ow,4),np.uint8);glBindFramebuffer(GL_FRAMEBUFFER,fb)
    glReadPixels(0,0,ow,oh,GL_RGBA,GL_UNSIGNED_BYTE,rgba.ctypes.data_as(C.c_void_p))
    full=renderer.capture(True);difference=int(np.max(np.abs(full.astype(np.int16)-rgba[:,:,0].astype(np.int16))))
    night=SunFrame(scene,np.array([1.,0.,-1.]),.12,float(data['bias']),'night')
    renderer.draw(night,fb,False);assert not np.any(renderer.capture(True))
    renderer.draw(frame,fb,False);assert np.any(renderer.capture(True))
    result={'renderer':renderer.renderer,'version':renderer.version,'grid':[gw,gh],'output':[ow,oh],
            'frames':len(frames),'median_ms':round(float(np.median(frames)),3),'p95_ms':round(float(np.percentile(frames,95)),3),
            'full_capture_byte_difference':difference,'day_night_recovery':True,'negative_land_cells':int(np.count_nonzero(z<0)),'per_sun':per_sun}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    glBindFramebuffer(GL_FRAMEBUFFER,0);renderer.close();glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex])


if __name__=='__main__':main()
