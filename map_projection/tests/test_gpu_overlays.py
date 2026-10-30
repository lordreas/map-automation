import ctypes as C
import os,unittest
import numpy as np


@unittest.skipUnless(os.environ.get('MAP_PROJECTION_GPU_TESTS')=='1','Enable GPU tests')
class OverlayGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from map_projection.experiments.gpu_context import context
        cls.context=context()

    def target(self,w,h):
        from map_projection.gpu import texture,framebuffer
        from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE
        tex=texture(8,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);return tex,framebuffer(tex)

    def release(self,tex,fb):
        from OpenGL.GLES3 import glDeleteFramebuffers,glDeleteTextures,glBindFramebuffer,GL_FRAMEBUFFER
        glBindFramebuffer(GL_FRAMEBUFFER,0);glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex])

    def test_chromatic_green_shift_reverses_without_moving_red_blue(self):
        from map_projection.gpu import Renderer
        r=Renderer();w=h=64;image=np.zeros((h,w,3),np.uint8);image[:,47:51]=255
        tex,fb=self.target(w,h)
        try:
            r.color_capture=True;source=r.chromatic_target(w,h);r.draw_rgb(image,source)
            samples={}
            for strength in (0,8,-8):
                r.compensate(strength,fb);samples[strength]=r.capture(full=True,output_framebuffer=fb)[:,:,:3]
                np.testing.assert_array_equal(samples[strength][:,:,0],image[:,:,0]);np.testing.assert_array_equal(samples[strength][:,:,2],image[:,:,2])
            np.testing.assert_array_equal(samples[0],image)
            def center(a):
                row=a[32,:,1].astype(float);return np.dot(np.arange(w),row)/row.sum()
            self.assertLess(center(samples[8]),center(samples[0])-1)
            self.assertGreater(center(samples[-8]),center(samples[0])+1)
        finally:self.release(tex,fb);r.close()

    def test_colored_track_reveal_and_paused_preview_preserve_rgb(self):
        from map_projection.gpu import Renderer
        r=Renderer();w=h=64;image=np.full((h,w,3),80,np.uint8);tex,fb=self.target(w,h)
        points=np.array([[2*(10.5)/w-1,1-2*(12.5)/h,.2,1],[2*40.5/w-1,1-2*35.5/h,.8,1]],np.float32)
        state={'vertices':points,'color':'#ff0000','progress':.5,'generation':1}
        try:
            r.color_capture=True;r.draw_rgb(image,fb);r.draw_track(state,fb)
            actual=r.capture(full=True,output_framebuffer=fb)
            np.testing.assert_array_equal(actual[12,10,:3],[255,0,0]);np.testing.assert_array_equal(actual[35,40,:3],[80,80,80])
            state['progress']=1;r.draw_rgb(image,fb);r.draw_track(state,fb)
            r.snapshot_output(fb,w,h)
            preview=r.capture(size=(32,32))
            self.assertEqual(preview.shape,(32,32,4));self.assertTrue(np.any(preview[:,:,0]>preview[:,:,1]))
            self.assertTrue(np.all(preview[:,:,3]==255))
        finally:self.release(tex,fb);r.close()
