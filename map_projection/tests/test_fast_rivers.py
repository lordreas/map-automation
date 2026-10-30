import copy,os,unittest
import numpy as np
from map_projection.layers import native


class RiverProjectionTests(unittest.TestCase):
 def test_native_field_projection_keeps_phases_sizes_and_first_hit_mask(self):
  h=w=16;y,x=np.mgrid[:h,:w];uv=np.stack(((x+.5)/w*65534,(y+.5)/h*65534),axis=2).astype(np.uint16)
  source=np.stack((abs(y-8),x/15,np.full_like(x,.5,dtype=float)),axis=2).astype(np.float16);out=np.empty_like(source)
  uv[0,0]=65535;native().layer_rivers(w,h,uv,w,h,source.view(np.uint16),out.view(np.uint16))
  self.assertGreater(out[0,0,0],50000);np.testing.assert_allclose(out[2:-2,2:-2,2],.5,atol=.001);np.testing.assert_allclose(out[2:-2,2:-2,1],source[2:-2,2:-2,1],atol=.002)
  np.testing.assert_allclose(out[4:12,2:-2,0],source[4:12,2:-2,0],atol=.02)


@unittest.skipUnless(os.environ.get('MAP_PROJECTION_GPU_TESTS')=='1','Enable GPU tests')
class FastRiverGpuTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from map_projection.experiments.gpu_context import context
  cls.context=context()

 def test_sparse_reveal_with_cached_color_background_and_elevation_fills(self):
  from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,glBindFramebuffer,glReadPixels,glGenVertexArrays,glBindVertexArray,glDeleteVertexArrays,glDeleteFramebuffers,glDeleteTextures
  from map_projection.gpu import texture,framebuffer
  from map_projection.gpu_layers import LayerRenderer
  from map_projection.config import DEFAULT
  class Clock:
   progress=0.
   def sample(self):return self.progress
  clock=Clock();r=LayerRenderer();w=h=64;vao=glGenVertexArrays(1);glBindVertexArray(vao);tex=texture(10,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);fb=framebuffer(tex);r.target(w,h)
  cfg=copy.deepcopy(DEFAULT['layers'])
  for v in cfg.values():v['enabled']=False
  cfg['satellite'].update(enabled=True,brightness=1.);cfg['rivers'].update(enabled=True,color='#ffffff')
  y,x=np.mgrid[:h,:w];uv=np.stack(((x+.5)/w*65534,(y+.5)/h*65534),axis=2).astype(np.uint16)
  field=np.stack((abs(y-32),np.ones_like(x),x/63),axis=2).astype(np.float16)
  state={'key':'fast-river','uv':uv,'config':cfg,'arrays':{'satellite':np.full((h,w,3),[30,50,80],np.uint8),'screen_rivers':field},'river_clock':clock}
  def draw():
   glBindVertexArray(vao);r.draw(state,fb,shading=False);glBindFramebuffer(GL_FRAMEBUFFER,fb);a=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,a);return a[::-1,:,:3].copy()
  try:
   a=draw();key=r.river_renderer.base_key;np.testing.assert_array_equal(a[10,10],[30,50,80])
   previous=a
   for phase in (.25,.5,.75,1):
    clock.progress=phase;current=draw();self.assertTrue(np.all(current>=previous));self.assertEqual(r.river_renderer.base_key,key);previous=current
   self.assertGreater(current[32,40,0],200)
   cfg['satellite']['enabled']=False;cfg['rivers']['enabled']=False;cfg['contours'].update(enabled=True,single=True,level=300,fill='above');state['arrays']={'screen_elevation':(y*10).astype(np.float16)}
   a=draw();self.assertGreater(a[40,40,0],250);self.assertEqual(a[20,40,0],0)
   cfg['contours']['fill']='below';a=draw();self.assertGreater(a[20,40,0],250);self.assertEqual(a[40,40,0],0)
  finally:r.close();glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex]);glDeleteVertexArrays(1,[vao])

 def test_compute_matches_reference_through_replay_and_chromatic_correction(self):
  from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,glBindFramebuffer,glReadPixels,glGenVertexArrays,glBindVertexArray,glDeleteVertexArrays,glDeleteFramebuffers,glDeleteTextures
  from map_projection.gpu import texture,framebuffer
  from map_projection.gpu_layers import LayerRenderer
  from map_projection.gpu_rivers_reference import Rivers as Reference
  from map_projection.config import DEFAULT
  fast=LayerRenderer();reference=LayerRenderer();reference.river_renderer=Reference(reference)
  w=h=64;vao=glGenVertexArrays(1);glBindVertexArray(vao)
  targets=[]
  for unit in (10,11):
   tex=texture(unit,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);targets.append((tex,framebuffer(tex)))
  fast.target(w,h);reference.target(w,h)
  cfg=copy.deepcopy(DEFAULT['layers'])
  for v in cfg.values():v['enabled']=False
  cfg['rivers'].update(enabled=True,color='#48baff');cfg['satellite'].update(enabled=True,brightness=1.)
  y,x=np.mgrid[:h,:w];uv=np.stack(((x+.5)/w*65534,(y+.5)/h*65534),axis=2).astype(np.uint16)
  field=np.stack((np.minimum(abs(y-28.3),abs(x-45.2)),x/63,(x+y)/126),axis=2).astype(np.float16)
  background=np.stack((x+30,y+40,np.full_like(x,80)),axis=2).astype(np.uint8)
  class Clock:
   progress=0.
   def sample(self):return self.progress
  clock=Clock();state={'key':'reference-comparison','uv':uv,'config':cfg,'arrays':{'satellite':background,'screen_rivers':field},'river_clock':clock}
  def read(r,target,strength):
   glBindVertexArray(vao);r.draw(state,target,shading=False,compensation=strength);glBindFramebuffer(GL_FRAMEBUFFER,target)
   out=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,out);return out[::-1].copy()
  try:
   for width,strength in ((1,0),(5,.5),(12,-8)):
    cfg['rivers']['river_thickness']=width
    for progress in (0,.25,.499,.503,.75,1,.2,0,1):
     clock.progress=progress;a=read(fast,targets[0][1],strength);b=read(reference,targets[1][1],strength)
     error=abs(a.astype(int)-b.astype(int));self.assertLessEqual(error.max(),6 if progress<1 else 2);self.assertLess(error.mean(),.15)
  finally:
   fast.close();reference.close()
   for tex,fb in targets:glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex])
   glDeleteVertexArrays(1,[vao])

 def test_compute_channels_are_race_free_and_preview_reads_current_replay(self):
  from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,glBindFramebuffer,glReadPixels,glGenVertexArrays,glBindVertexArray,glDeleteVertexArrays,glDeleteFramebuffers,glDeleteTextures
  from map_projection.gpu import texture,framebuffer
  from map_projection.gpu_layers import LayerRenderer
  from map_projection.config import DEFAULT
  r=LayerRenderer();w=h=64;vao=glGenVertexArrays(1);glBindVertexArray(vao)
  tex=texture(10,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);fb=framebuffer(tex)
  smalltex=texture(11,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,16,16);smallfb=framebuffer(smalltex);r.target(w,h)
  cfg=copy.deepcopy(DEFAULT['layers'])
  for v in cfg.values():v['enabled']=False
  cfg['rivers'].update(enabled=True,color='#ffffff')
  y,x=np.mgrid[:h,:w];field=np.stack((abs(y-30),np.ones_like(x),np.full_like(x,.5,dtype=float)),axis=2).astype(np.float16)
  class Clock:
   progress=0.
   def sample(self):return self.progress
  clock=Clock();state={'key':'channel-sync','config':cfg,'arrays':{'screen_rivers':field},'river_clock':clock}
  try:
   for progress in (.499,.501,.502,.504,1,0,.502,1):
    clock.progress=progress;glBindVertexArray(vao);r.draw(state,fb,shading=False);glBindFramebuffer(GL_FRAMEBUFFER,fb)
    a=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,a);a=a[::-1].copy()
    np.testing.assert_array_equal(a[:,:,0],a[:,:,1]);np.testing.assert_array_equal(a[:,:,0],a[:,:,2])
    if progress==0:self.assertEqual(a[:,:,:3].max(),0)
    r.river_renderer.preview(smallfb,16,16);glBindFramebuffer(GL_FRAMEBUFFER,smallfb)
    b=np.empty((16,16,4),np.uint8);glReadPixels(0,0,16,16,GL_RGBA,GL_UNSIGNED_BYTE,b)
    expected=a.reshape(16,4,16,4,4).mean(axis=(1,3));np.testing.assert_allclose(b,expected,atol=1)
  finally:r.close();glDeleteFramebuffers(2,[fb,smallfb]);glDeleteTextures(2,[tex,smalltex]);glDeleteVertexArrays(1,[vao])
