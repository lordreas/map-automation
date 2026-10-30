"""Compare a moving-sun sequence against all-cell continuous DEM ray casts.

A reversal is a lit/shadow transition followed by another within the sequence;
we report it in both methods rather than assuming real terrain is convex.
"""
import argparse,ctypes as C,hashlib,json,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from map_projection.experiments.gpu_context import context
from map_projection.frames import SunScene,SunFrame
from map_projection.solar import position
from map_projection.reference import bilinear_blocked

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grid',type=int,default=512);p.add_argument('--frames',type=int,default=121);p.add_argument('--step',type=int,default=20)
    p.add_argument('--terrain',type=Path,help='Use an actual cached terrain NPY instead of resizing the input grid')
    p.add_argument('--start',default='2026-10-25T16:20:00+01:00');args=p.parse_args()
    context()
    from map_projection.gpu import Renderer,texture,framebuffer
    from OpenGL.GLES3 import GL_RED,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,glBindFramebuffer,glReadPixels,glFinish
    d=np.load(args.input);source=d['z'];oh,ow=source.shape;w=args.grid;h=round(w*oh/ow)
    z=np.ascontiguousarray(source[np.ix_(np.minimum(((np.arange(h)+.5)*oh/h).astype(int),oh-1),np.minimum(((np.arange(w)+.5)*ow/w).astype(int),ow-1))])
    if args.terrain:z=np.ascontiguousarray(np.load(args.terrain),np.float32);h,w=z.shape
    y,x=np.mgrid[:h,:w];uv=np.stack((x,y),axis=-1).astype(np.uint16);normals=np.zeros((h,w,3),np.float32);normals[:,:,2]=1
    scene=SunScene('temporal-reference',z,normals,uv,tuple(d['terrain_size']));r=Renderer()
    tex=texture(10,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);fb=framebuffer(tex)
    sequences={'fast':[],'reference':[],'raw_scan':[]};previous={};transitions={m:np.zeros((h,w),np.uint16) for m in sequences};different=0;times={'fast':[],'reference':[]};mismatches=checks=0;rng=np.random.default_rng(55)
    start=datetime.fromisoformat(args.start)
    for i in range(args.frames):
        stamp=(start+timedelta(seconds=i*args.step)).isoformat();sun,_=position(stamp,45.95,10.65)
        if sun[2]<=0:break
        for method in ('fast','reference'):
            t=time.monotonic();r.draw(SunFrame(scene,sun,0,float(d['bias']),stamp,True,method),fb,False);glFinish();times[method].append((time.monotonic()-t)*1000)
            glBindFramebuffer(GL_FRAMEBUFFER,fb);image=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,image.ctypes.data_as(C.c_void_p))
            mask=image[:,:,0]>0
            if method in previous:transitions[method]+=(mask!=previous[method])
            previous[method]=mask
            sequences[method].append(np.packbits(mask,axis=1))
            if method=='fast':
                glBindFramebuffer(GL_FRAMEBUFFER,r.light_fb);raw_mask=np.empty((h,w),np.uint8)
                glReadPixels(0,0,w,h,GL_RED,GL_UNSIGNED_BYTE,raw_mask.ctypes.data_as(C.c_void_p));raw_mask=raw_mask>0
                if 'raw_scan' in previous:transitions['raw_scan']+=(raw_mask!=previous['raw_scan'])
                previous['raw_scan']=raw_mask;sequences['raw_scan'].append(np.packbits(raw_mask,axis=1))
            if method=='reference' and i%20==0:
                raw=np.array([sun[0]/(scene.terrain_size[0]/w),-sun[1]/(scene.terrain_size[1]/h)]);dominant=max(abs(raw));direction=raw/dominant;rise=sun[2]/dominant
                for yy,xx in rng.integers([0,0],[h,w],(64,2)):
                    expected=bilinear_blocked(z,int(xx),int(yy),direction,rise,float(d['bias']))
                    mismatches+=expected==mask[yy,xx];checks+=1
        different+=int(np.count_nonzero(previous['fast']!=previous['reference']))
        if i%20==0:print(i,stamp,flush=True)
    frames=len(sequences['fast']);result={'renderer':r.renderer,'grid':[w,h],'frames':frames,'step_seconds':args.step,'start':args.start,'reference_surface':'continuous bilinear DEM / point sun','bias_metres':float(d['bias']),'terrain_sha256':hashlib.sha256(z.tobytes()).hexdigest(),'timed_output':[w,h],
      'different_classifications':different,'classification_error_fraction':different/(frames*w*h),'oracle_checks':checks,'oracle_mismatches':int(mismatches)}
    for method in sequences:
        counts=transitions[method]
        result[method]={'median_ms':float(np.median(times[method][1:])) if times.get(method) else None,'shadow_transitions':int(counts.sum()),'pixels_with_multiple_transitions':int((counts>=2).sum()),'pixels_with_four_or_more_transitions':int((counts>=4).sum()),'max_transitions':int(counts.max())}
    result['fast_reversals_where_reference_constant']=int(((transitions['fast']>=2)&(transitions['reference']==0)).sum())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),fast=np.array(sequences['fast']),reference=np.array(sequences['reference']),raw_scan=np.array(sequences['raw_scan']),frames=frames,width=w,z=z)
    print(json.dumps(result,indent=2),flush=True);r.close()
if __name__=='__main__':main()
