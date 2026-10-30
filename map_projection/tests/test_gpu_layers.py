import copy,os,unittest
import numpy as np


@unittest.skipUnless(os.environ.get('MAP_PROJECTION_GPU_TESTS')=='1','Enable GPU tests')
class GpuLayersTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from map_projection.experiments.gpu_context import context
  cls.context=context()

 def test_black_background_satellite_shading_contours_and_river_width(self):
  from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_TEXTURE_2D,GL_RED,GL_FRAMEBUFFER,GL_ARRAY_BUFFER,GL_TRIANGLES,GL_COLOR_BUFFER_BIT,GL_TEXTURE0,glActiveTexture,glBindTexture,glTexSubImage2D,glBindFramebuffer,glReadPixels,glGenVertexArrays,glBindVertexArray,glDeleteVertexArrays,glDeleteFramebuffers,glDeleteTextures
  from map_projection.gpu import texture,framebuffer
  from map_projection.gpu_layers import LayerRenderer
  from map_projection.config import DEFAULT
  r=LayerRenderer();w=h=64;vao=glGenVertexArrays(1);glBindVertexArray(vao)
  tex=texture(10,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);fb=framebuffer(tex)
  r.target(w,h);glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,r.light_tex)
  light=np.full((h,w),128,np.uint8);glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RED,GL_UNSIGNED_BYTE,light)
  cfg=copy.deepcopy(DEFAULT['layers'])
  for value in cfg.values():value['enabled']=False
  y,x=np.mgrid[:h,:w];uv=np.stack(((x+.5)/w*65534,(y+.5)/h*65534),axis=2).astype(np.uint16)
  state={'key':'test','uv':uv,'config':cfg,'arrays':{}}
  def draw():
   glBindVertexArray(vao);r.draw(state,fb);glBindFramebuffer(GL_FRAMEBUFFER,fb)
   out=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,out)
   return out[::-1,:,:3].copy()
  try:
   self.assertFalse(np.any(draw()))
   cfg['satellite'].update(enabled=True,brightness=1.);state['arrays']['satellite']=np.full((h,w,3),[200,100,50],np.uint8)
   np.testing.assert_allclose(draw()[30,30],[100,50,25],atol=1)
   cfg['satellite']['enabled']=False;cfg['contours'].update(enabled=True,single=True,level=315,width=2)
   state['arrays']={'elevation':(x*10).astype(np.float32)}
   a=draw();self.assertGreater(a[:,31:33].mean(),100);self.assertEqual(a[:,:20].max(),0)
   cfg['contours']['enabled']=False;cfg['rivers']['enabled']=True
   field=np.stack((np.abs(x-32),np.ones_like(x),np.full_like(x,.5,dtype=float)),axis=2).astype(np.float16)
   state['arrays']={'rivers':field};cfg['rivers']['river_thickness']=1
   small=np.count_nonzero(draw().max(axis=2));cfg['rivers']['river_thickness']=8
   self.assertGreater(np.count_nonzero(draw().max(axis=2)),small*2)
  finally:
   r.close();glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex]);glDeleteVertexArrays(1,[vao])

 def test_river_reveal_is_monotonic_and_single_contour_uses_projected_heights(self):
  from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,glBindFramebuffer,glReadPixels,glGenVertexArrays,glBindVertexArray,glDeleteVertexArrays,glDeleteFramebuffers,glDeleteTextures
  from map_projection.gpu import texture,framebuffer
  from map_projection.gpu_layers import LayerRenderer
  from map_projection.config import DEFAULT
  class Clock:
   progress=0.
   def sample(self):return self.progress
  clock=Clock();r=LayerRenderer();w=h=64;vao=glGenVertexArrays(1);glBindVertexArray(vao);tex=texture(10,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);fb=framebuffer(tex);r.target(w,h)
  cfg=copy.deepcopy(DEFAULT['layers'])
  for value in cfg.values():value['enabled']=False
  y,x=np.mgrid[:h,:w];uv=np.stack(((x+.5)/w*65534,(y+.5)/h*65534),axis=2).astype(np.uint16)
  fields=np.stack((abs(y-32),x/63,np.tile(np.linspace(0,1,w),(h,1))),axis=2).astype(np.float16)
  state={'key':'river-reveal','uv':uv,'config':cfg,'arrays':{'rivers':fields},'river_clock':clock}
  def draw():
   glBindVertexArray(vao);r.draw(state,fb,shading=False);glBindFramebuffer(GL_FRAMEBUFFER,fb)
   out=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,out);return out[::-1,:,:3].copy()
  try:
   cfg['rivers']['enabled']=True;previous=np.zeros((h,w,3),np.uint8)
   for phase in (0,.25,.5,.75,1):
    clock.progress=phase;current=draw();self.assertTrue(np.all(current>=previous));previous=current
   self.assertGreater(current[:,50:].sum(),0)
   cfg['rivers']['enabled']=False;cfg['contours'].update(enabled=True,single=True,level=350,width=2)
   heights=np.asarray(y*10,np.float16);state['arrays']={'elevation':np.zeros((2,2),np.float32),'screen_elevation':heights}
   a=draw();self.assertGreater(a[34:37].sum(),0);self.assertFalse(np.any(a[:20]))
   cfg['contours']['enabled']=False;cfg['divides']['enabled']=True;state['arrays']={'divides':abs(x-30).astype(np.float16)}
   self.assertGreater(draw()[:,29:32].sum(),0)
  finally:r.close();glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex]);glDeleteVertexArrays(1,[vao])

 def test_inline_contour_compensation_moves_only_green(self):
  from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,glBindFramebuffer,glReadPixels,glGenVertexArrays,glBindVertexArray,glDeleteVertexArrays,glDeleteFramebuffers,glDeleteTextures
  from map_projection.gpu import texture,framebuffer
  from map_projection.gpu_layers import LayerRenderer
  from map_projection.config import DEFAULT
  r=LayerRenderer();w=h=64;vao=glGenVertexArrays(1);glBindVertexArray(vao);tex=texture(10,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);fb=framebuffer(tex);r.target(w,h)
  cfg=copy.deepcopy(DEFAULT['layers'])
  for value in cfg.values():value['enabled']=False
  cfg['contours'].update(enabled=True,single=True,level=500,width=2)
  y,x=np.mgrid[:h,:w];state={'key':'inline-ca','config':cfg,'arrays':{'screen_elevation':np.asarray(x*10,np.float16)}}
  try:
   samples={}
   for strength in (0,8,-8):
    glBindVertexArray(vao);r.draw(state,fb,shading=False,compensation=strength);glBindFramebuffer(GL_FRAMEBUFFER,fb)
    out=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,out);samples[strength]=out[:,:,:3].copy()
   for strength in (8,-8):
    np.testing.assert_array_equal(samples[strength][:,:,0],samples[0][:,:,0]);np.testing.assert_array_equal(samples[strength][:,:,2],samples[0][:,:,2])
   def center(a):
    row=a[32,:,1].astype(float);return (row*np.arange(w)).sum()/row.sum()
   self.assertLess(center(samples[8]),center(samples[0])-1);self.assertGreater(center(samples[-8]),center(samples[0])+1)
  finally:r.close();glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex]);glDeleteVertexArrays(1,[vao])
