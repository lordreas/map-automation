"""Resumable laptop GPU skyline generation, independent of projector calibration."""
import argparse,ctypes as C,json,math,time
from pathlib import Path
import numpy as np
from .horizon_cache import identity,SCHEMA

def bake(z,physical,bias,destination,angles=1440):
    from .experiments.gpu_context import context
    context()
    from .gpu import Renderer,compute,texture,framebuffer
    from .gpu_reference import RAYS,Reference
    from .frames import SunScene
    from OpenGL.GLES3 import GL_R32F,GL_RED,GL_FLOAT,GL_FRAMEBUFFER,GL_WRITE_ONLY,glBindFramebuffer,glReadPixels,glUseProgram,glUniform2f,glUniform2i,glFinish
    from OpenGL.GLES3.VERSION.GLES3_3_1 import glDispatchCompute,glMemoryBarrier,glBindImageTexture
    h,w=z.shape;r=Renderer();scene=SunScene('horizon-bake',z,np.zeros((h,w,3),np.float32),np.zeros((1,1,2),np.uint16),tuple(physical));r.scene(scene,'reference');ref=Reference(r);ref.prepare(scene)
    source=RAYS[:RAYS.index('void main(){')].replace('layout(rgba8,binding=1)','layout(r32f,binding=1)').replace('uniform float rise;','float rise;')
    source+='''
uniform float horizontal_scale;
void main(){ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(pixel,size)))return;
 float lo=0.,hi=89.999;
 for(int i=0;i<18;i++){float mid=(lo+hi)*.5;rise=tan(radians(mid))*horizontal_scale;int hit=blocked(pixel);if(hit<0)atomicAdd(failures,1u);if(hit!=0)lo=mid;else hi=mid;}
 imageStore(result,pixel,vec4((lo+hi)*.5));}
'''
    program=compute(source);r.locations[program]={};out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    key=identity(z,physical,bias);manifest=out/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['key']!=key or old['angles']!=angles:raise ValueError('Existing horizon cache has different geometry or resolution')
        completed=old.get('completed',0)
    else:completed=0
    import shutil
    required=angles*h*w*4
    if not (out/'horizons.npy').exists() and required>shutil.disk_usage(out).free:raise ValueError(f'Horizon cache needs {required/1e9:.1f} GB; insufficient disk space')
    mode='r+' if (out/'horizons.npy').exists() else 'w+'
    data=np.lib.format.open_memmap(out/'horizons.npy',mode=mode,dtype=np.float32,shape=(angles,h,w))
    meta={'schema':SCHEMA,'key':key,'angles':angles,'shape':[h,w],'physical':list(map(float,physical)),'bias':float(bias),'elevation_precision_degrees':90/2**18,'angular_step_degrees':360/angles,'complete':False,'completed':completed,'interval_certified':False}
    tex=texture(13,None,GL_R32F,GL_RED,GL_FLOAT,w,h,immutable=True);fb=framebuffer(tex)
    glUseProgram(program)
    for name,value in [('terrain',0),('hierarchy',9),('levels',ref.levels)]:r.integer(program,name,value)
    r.real(program,'bias',bias);glUniform2i(r.location(program,'size'),w,h)
    # Reference preparation binds its hierarchy at unit nine.
    from OpenGL.GLES3 import glBindBufferBase,glBindBuffer,glBufferSubData
    glBindBufferBase(0x90D2,2,ref.diagnostics);glBindBuffer(0x90D2,ref.diagnostics);zero=np.zeros(1,np.uint32);glBufferSubData(0x90D2,0,4,zero.ctypes.data_as(C.c_void_p))
    glBindImageTexture(1,tex,0,False,0,GL_WRITE_ONLY,GL_R32F)
    try:
        for a in range(completed,angles):
            stamp=time.monotonic();az=math.radians(a*360/angles);sx=math.sin(az)/(physical[0]/w);sy=-math.cos(az)/(physical[1]/h);dominant=max(abs(sx),abs(sy))
            glUniform2f(r.location(program,'direction'),sx/dominant,sy/dominant);r.real(program,'horizontal_scale',1/dominant)
            glDispatchCompute((w+7)//8,(h+7)//8,1);glMemoryBarrier(0x428);glFinish()
            glBindFramebuffer(GL_FRAMEBUFFER,fb);image=np.empty((h,w),np.float32);glReadPixels(0,0,w,h,GL_RED,GL_FLOAT,image.ctypes.data_as(C.c_void_p))
            glBindBuffer(0x90D2,ref.diagnostics)
            from OpenGL.GLES3 import glMapBufferRange,glUnmapBuffer,GL_MAP_READ_BIT
            pointer=glMapBufferRange(0x90D2,0,4,GL_MAP_READ_BIT)
            if not pointer:raise RuntimeError('Horizon diagnostics unavailable')
            try:failures=int.from_bytes(C.string_at(pointer,4),'little')
            finally:glUnmapBuffer(0x90D2)
            if failures:raise RuntimeError(f'Horizon traversal failed: {failures}')
            data[a]=image;data.flush();meta.update(completed=a+1,complete=a+1==angles)
            temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(meta,indent=2));temp.replace(manifest)
            print(f'horizon {a+1}/{angles}: {time.monotonic()-stamp:.3f}s',flush=True)
    finally:ref.close();r.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--terrain',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--angles',type=int,default=1440)
    args=p.parse_args();d=np.load(args.input);z=np.ascontiguousarray(np.load(args.terrain) if args.terrain else d['z'],np.float32)
    if args.angles<4:raise ValueError('At least four azimuth samples required')
    bake(z,d['terrain_size'],float(d['bias']),args.output,args.angles)
if __name__=='__main__':main()
