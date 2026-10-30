"""One-shot 4K precise benchmark, fixed memory usage on the 2 GB board."""
import argparse,ctypes as C,json,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from map_projection.experiments.gpu_context import context
from map_projection.frames import SunFrame,SunScene
from map_projection.solar import position

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=6);args=p.parse_args()
    context()
    from map_projection.gpu import Renderer,texture,framebuffer
    from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,GL_MAP_READ_BIT,glFinish,glBindFramebuffer,glReadPixels,glBindBuffer,glMapBufferRange,glUnmapBuffer,glDeleteTextures,glDeleteFramebuffers
    data=np.load(args.input);z=np.ascontiguousarray(np.load(args.terrain),np.float32);h,w=z.shape;ts=data['terrain_size'];oldh,oldw=data['z'].shape
    original=data['lookup'];safe=np.maximum(original,0);uv=np.stack((safe%oldw*w//oldw,safe//oldw*h//oldh),axis=-1).astype(np.uint16);uv[original<0]=65535
    dy,dx=np.gradient(z,ts[1]/h,ts[0]/w);normals=np.stack((-dx,dy,np.ones_like(z)),axis=-1);normals=np.ascontiguousarray(normals/np.linalg.norm(normals,axis=-1,keepdims=True),np.float32)
    del original,safe,dy,dx;scene=SunScene('precise-benchmark',z,normals,uv,tuple(ts));oh,ow=uv.shape[:2]
    args.output.parent.mkdir(parents=True,exist_ok=True);raw_dir=args.output.parent/(args.output.stem+'-frames');raw_dir.mkdir(exist_ok=True)
    stamps=[(datetime.fromisoformat('2026-10-25T16:20:00+01:00')+timedelta(seconds=i*20)).isoformat() for i in range(args.frames)]
    results={};errors=0
    for method in ('reference','precise'):
        renderer=Renderer();tex=texture(13,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,ow,oh);fb=framebuffer(tex);rows=[]
        for i,stamp in enumerate(stamps):
            sun,_=position(stamp,45.95,10.65);frame=SunFrame(scene,sun,0,float(data['bias']),stamp,True,method)
            # Static upload/compilation/warmup are outside timings. This is not
            # a cached sun frame: every timed call shades the new sun again.
            if i==0:renderer.draw(frame,fb,False);glFinish()
            start=time.perf_counter();renderer.draw(frame,fb,False);glFinish();elapsed=(time.perf_counter()-start)*1000
            glBindFramebuffer(GL_FRAMEBUFFER,renderer.filtered_fb);rgba=np.empty((h,w,4),np.uint8)
            glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,rgba.ctypes.data_as(C.c_void_p));light=np.ascontiguousarray(rgba[:,:,0]);del rgba
            row={'time':stamp,'milliseconds':round(elapsed,3)}
            if method=='reference':np.save(raw_dir/f'{i}.npy',light)
            else:
                expected=np.load(raw_dir/f'{i}.npy',mmap_mode='r');diff=np.abs(light.astype(np.int16)-expected.astype(np.int16));bad=int((diff>1).sum());errors+=bad
                row.update(different_pixels=bad,max_byte_error=int(diff.max()))
                glBindBuffer(0x90D2,renderer.candidates);ptr=glMapBufferRange(0x90D2,0,4,GL_MAP_READ_BIT)
                row['traced_rays']=int.from_bytes(C.string_at(ptr,4),'little');glUnmapBuffer(0x90D2);renderer.precise.check(wait=True)
            rows.append(row);print(method,json.dumps(row),flush=True)
        results[method]=rows;renderer.close();glDeleteTextures(1,[tex]);glDeleteFramebuffers(1,[fb])
    result={'grid':[w,h],'output':[ow,oh],'results':results,'different_pixels':errors,'budget_ms':1000/30,
            'median_precise_ms':float(np.median([r['milliseconds'] for r in results['precise'][1:]]))}
    result['meets_30fps_offscreen']=result['median_precise_ms']<=result['budget_ms']
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if errors:raise SystemExit('Precise output disagrees with the reference')
if __name__=='__main__':main()
