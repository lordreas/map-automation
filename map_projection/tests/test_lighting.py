import copy,unittest
import numpy as np
from map_projection.config import DEFAULT
from map_projection.lighting import daily_peak,options,apply


class LightingTests(unittest.TestCase):
 def test_one_daily_reference_makes_peak_one_without_per_frame_rescaling(self):
  sun=copy.deepcopy(DEFAULT['sun']);sun.update(time='2026-10-29T12:00:00+01:00',latitude=45.95,longitude=10.65)
  normals=np.zeros((3,3,3),np.float32);normals[:,:,2]=1
  peak=daily_peak(normals,sun);self.assertTrue(.2<peak<1)
  gain,floor,gamma=options(sun,peak);self.assertAlmostEqual(gain*peak,1)
  sun['time']='2026-10-29T17:00:00+01:00';self.assertEqual(options(sun,peak),(gain,floor,gamma))

 def test_floor_is_ten_percent_and_preserves_silhouette_with_real_black_when_disabled(self):
  frame=np.zeros((2,2,3),np.uint8);valid=np.array([[True,False],[True,True]])
  lit=apply(frame,valid,1,.1,1/2.2);self.assertTrue(25<=lit[0,0,0]<=26);self.assertFalse(lit[0,1].any())
  self.assertFalse(apply(frame,valid,1,0,1/2.2).any())


@unittest.skipUnless(__import__('os').environ.get('MAP_PROJECTION_GPU_TESTS')=='1','Enable GPU tests')
class LightingGpuTests(unittest.TestCase):
 def test_gpu_night_floor_and_day_peak_do_not_illuminate_invalid_pixels(self):
  from map_projection.experiments.gpu_context import context
  context()
  from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,GL_FRAMEBUFFER,glBindFramebuffer,glReadPixels,glDeleteFramebuffers,glDeleteTextures
  from map_projection.gpu import Renderer,texture,framebuffer
  from map_projection.frames import SunFrame,SunScene
  r=Renderer();w=h=32;uv=np.zeros((h,w,2),np.uint16);uv[0,0]=65535
  normals=np.zeros((2,2,3),np.float32);normals[:,:,2]=1
  scene=SunScene('lighting-test',np.zeros((2,2),np.float32),normals,uv,(2,2))
  tex=texture(13,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);fb=framebuffer(tex)
  def draw(sun,gain,floor):
   f=SunFrame(scene,np.array(sun,np.float32),0,0,'2026-10-29T12:00:00+01:00',light_gain=gain,light_floor=floor,light_gamma=1/2.2)
   r.draw(f,fb);glBindFramebuffer(GL_FRAMEBUFFER,fb);out=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,out);return out[::-1]
  try:
   night=draw([0,0,-1],1,.1);self.assertTrue(25<=night[10,10,0]<=26);self.assertEqual(night[0,0,0],0)
   self.assertEqual(draw([0,0,-1],1,0)[:,:,:3].max(),0)
   self.assertGreaterEqual(draw([.6,0,.8],1.25,.1)[10,10,0],254)
  finally:r.close();glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex])
