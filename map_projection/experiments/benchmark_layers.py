"""Measure the real 4K map compositing pass, independently of HDMI swap/preview."""
import argparse
import copy
import json
import time
import urllib.request
import numpy as np


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=int,default=90);p.add_argument('--url',default='http://127.0.0.1:8080');p.add_argument('--mode',choices=['contours','satellite','all','rivers','rivers-map']);p.add_argument('--all-contours',action='store_true');p.add_argument('--compensation',type=float,default=0.);p.add_argument('--preview',action='store_true',help='Include 15 Hz downsampling, readback and JPEG encoding for river replay');args=p.parse_args()
 from .gpu_context import context
 context()
 from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,GL_COLOR_BUFFER_BIT,glBindFramebuffer,glClearColor,glClear,glFinish,glDeleteFramebuffers,glDeleteTextures,glReadPixels,glBlitFramebuffer,GL_READ_FRAMEBUFFER,GL_DRAW_FRAMEBUFFER,GL_LINEAR
 from map_projection.gpu import texture,framebuffer
 from map_projection.gpu_layers import LayerRenderer
 from map_projection.layers import LayerStack
 from map_projection.geometry import projector
 from map_projection.native import Native
 from map_projection.engine import digest
 from map_projection import config
 with urllib.request.urlopen(args.url+'/api/state') as response:state=json.load(response)
 c=config.merge(config.DEFAULT,state['config']);bounds=np.array(state['mesh']['bounds'],np.float64)
 mesh=(config.ROOT.parent/'stl_files/full_map_0.4mm.stl').resolve();stat=mesh.stat();mesh_key=digest([str(mesh),stat.st_size,stat.st_mtime_ns])
 matrix,inverse,origin,_=projector(c,bounds);w,h=c['output']['width'],c['output']['height']
 key=digest([Native().signature,mesh_key,matrix.tolist(),w,h]);depth=np.load(config.ROOT/'.cache'/(key+'-depth.npy'),mmap_mode='r')
 stack=LayerStack();r=LayerRenderer();target_tex=texture(10,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);target=framebuffer(target_tex)
 r.target(w,h);glBindFramebuffer(GL_FRAMEBUFFER,r.light_fb);glClearColor(1,1,1,1);glClear(GL_COLOR_BUFFER_BIT)
 reports=[]
 preview_tex=texture(9,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,960,540);preview_fb=framebuffer(preview_tex)
 half_tex=texture(12,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w//2,h//2);half_fb=framebuffer(half_tex)
 try:
  for mode,name,enabled in [('contours','contours',['contours']),('satellite','satellite + contours',['satellite','contours']),('all','all geographic layers',['satellite','streets','semantic','contours','rivers','peaks','divides']),('rivers','rivers',['rivers']),('rivers-map','satellite + rivers',['satellite','rivers'])]:
   if args.mode and args.mode!=mode:continue
   for value in c['layers'].values():value['enabled']=False
   for layer in enabled:c['layers'][layer]['enabled']=True
   started=time.monotonic();layers=stack.prepare(c,key,depth,inverse,origin,bounds);prepare=time.monotonic()-started
   class Clock:
    progress=0.
    def sample(self):return self.progress
   clock=Clock();layers['river_clock']=clock
   for _ in range(3):r.draw(layers,target,compensation=args.compensation)
   glFinish();start=time.monotonic();next_preview=start;preview_count=0
   for i in range(args.frames):
    clock.progress=(i+1)/args.frames
    layers['config']['contours']['level']=float(i*20);layers['config']['contours']['single']=not args.all_contours
    r.draw(layers,target,compensation=args.compensation)
    if args.preview and r.river_renderer is not None and time.monotonic()>=next_preview:
     import io
     from PIL import Image
     glBindFramebuffer(GL_READ_FRAMEBUFFER,target);glBindFramebuffer(GL_DRAW_FRAMEBUFFER,half_fb)
     glBlitFramebuffer(0,h,w,0,0,0,w//2,h//2,GL_COLOR_BUFFER_BIT,GL_LINEAR)
     glBindFramebuffer(GL_READ_FRAMEBUFFER,half_fb);glBindFramebuffer(GL_DRAW_FRAMEBUFFER,preview_fb)
     glBlitFramebuffer(0,0,w//2,h//2,0,0,960,540,GL_COLOR_BUFFER_BIT,GL_LINEAR);glBindFramebuffer(GL_FRAMEBUFFER,preview_fb)
     pixels=np.empty((540,960,4),np.uint8);glReadPixels(0,0,960,540,GL_RGBA,GL_UNSIGNED_BYTE,pixels)
     Image.fromarray(pixels).convert('RGB').save(io.BytesIO(),format='JPEG',quality=85,subsampling=2)
     preview_count+=1;next_preview=max(next_preview+1/15,time.monotonic())
    # Force completion for every frame: tile-based GPUs may otherwise elide
    # opaque draws overwritten before the final finish.
    glFinish()
   glFinish();seconds=time.monotonic()-start
   report=dict(mode=name,chromatic_pixels=args.compensation,preview_frames=preview_count,preview_enabled=args.preview,single_contour=not args.all_contours,width=w,height=h,frames=args.frames,seconds=seconds,fps=args.frames/seconds,prepare_seconds=prepare);reports.append(report);
   if r.river_renderer is not None:report.update(r.river_renderer.timings)
   print(json.dumps(report),flush=True)
 finally:r.close();glDeleteFramebuffers(1,[target]);glDeleteTextures(1,[target_tex]);glDeleteFramebuffers(1,[preview_fb]);glDeleteTextures(1,[preview_tex]);glDeleteFramebuffers(1,[half_fb]);glDeleteTextures(1,[half_tex])
 return reports

if __name__=='__main__':main()
