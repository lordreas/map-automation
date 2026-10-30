"""Opt-in numerical checks against a real GLES 3.1 context (any GPU vendor).
Run: MAP_PROJECTION_GPU_TESTS=1 python -m unittest map_projection.tests.test_gpu
"""
import os
import unittest
import ctypes as C
import numpy as np


@unittest.skipUnless(os.environ.get('MAP_PROJECTION_GPU_TESTS')=='1','Set MAP_PROJECTION_GPU_TESTS=1 for GLES tests')
class GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from map_projection.experiments.gpu_context import context
        cls.context=context()
        from map_projection.gpu import Renderer
        cls.renderer=Renderer()

    @classmethod
    def tearDownClass(cls):cls.renderer.close()

    def read(self,w,h):
        from OpenGL.GLES3 import glReadPixels,GL_RGBA,GL_UNSIGNED_BYTE
        rgba=np.empty((h,w,4),np.uint8)
        glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,rgba.ctypes.data_as(C.c_void_p))
        return rgba[:,:,:3]

    def target(self,w,h):
        from map_projection.gpu import texture,framebuffer
        from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE
        tex=texture(8,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h)
        return tex,framebuffer(tex)

    def release(self,tex,fb):
        from OpenGL.GLES3 import glBindFramebuffer,GL_FRAMEBUFFER,glDeleteFramebuffers,glDeleteTextures
        glBindFramebuffer(GL_FRAMEBUFFER,0);glDeleteFramebuffers(1,[fb]);glDeleteTextures(1,[tex])

    def test_exact_native_pixels_and_physical_y_orientation(self):
        # Odd rows also catch GL's default four-byte RGB unpack alignment.
        h,w=65,67;image=np.zeros((h,w,3),np.uint8)
        image[0,0]=[255,0,0];image[-1,-1]=[0,255,0];image[31,17]=[0,0,255]
        tex,fb=self.target(w,h)
        try:
            self.renderer.draw_rgb(image,fb,physical=False)
            np.testing.assert_array_equal(self.read(w,h),image)
            self.renderer.draw_rgb(image,fb,physical=True)
            np.testing.assert_array_equal(self.read(w,h),image[::-1])
        finally:self.release(tex,fb)

    def test_gpu_shadows_match_double_precision_sheared_surface(self):
        from map_projection.frames import SunScene,SunFrame
        from map_projection.experiments.benchmark_gpu_scan import cpu_shear
        y,x=np.mgrid[:20,:32];z=(6*np.exp(-((x-13)**2+(y-9)**2)/12)).astype(np.float32)
        z[:4,:4]=-2
        ts=(64.,40.);dy,dx=np.gradient(z,2.,2.)
        normals=np.stack((-dx,dy,np.ones_like(z)),axis=-1);normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
        normals=np.ascontiguousarray(normals,np.float32)
        v,u=np.mgrid[:40,:64];uv=np.stack((u//2,v//2),axis=-1).astype(np.uint16)
        scene=SunScene('hills',z,normals,uv,ts);tex,fb=self.target(64,40)
        removed=retained=0
        try:
            for vector in ([1,0,.4],[-1,0,.4],[0,1,.4],[0,-1,.4],[1,1,.3],[-1,1,.3],[1,-1,.3],[-1,-1,.3],[0,0,1],[1,0,-.1]):
                sun=np.array(vector,dtype=float);sun/=np.linalg.norm(sun);bias=.07
                self.renderer.draw(SunFrame(scene,sun,.12,bias,'test'),fb,physical=False)
                actual=self.read(64,40)[:,:,0]
                sx,sy=sun[0]/2,-sun[1]/2;horizontal=abs(sx)>=abs(sy);dominant=sx if horizontal else sy
                if sun[2]<=0:light=np.zeros_like(z,dtype=np.uint8)
                else:
                    blocked=np.zeros_like(z,bool)
                    if abs(dominant)>1e-12:
                        count=32 if horizontal else 20;across=20 if horizontal else 32
                        offset=(sy if horizontal else sx)/abs(dominant);rise=sun[2]/abs(dominant)
                        qmin=np.floor(min(0,offset*(count-1)));lines=int(np.ceil(across+abs(offset)*(count-1)))+2
                        h=cpu_shear(z,horizontal,count-1 if dominant>0 else 0,-1 if dominant>0 else 1,offset,rise,qmin,lines)
                        blocked=h>z+bias
                    diffuse=np.maximum(0,np.einsum('ijk,k->ij',normals,sun));diffuse[blocked]=0
                    light=np.floor(255*(.12+.88*diffuse)+.5).astype(np.uint8)
                    if abs(dominant)>1e-12:
                        # Independent cell-by-cell rays validate the sparse
                        # hierarchical refinement, including real lit peaks.
                        from map_projection.experiments.benchmark_gpu_shadows import dda
                        bright=light/255>(.12+1.5/255);padded=np.pad(bright.astype(int),1)
                        neighbors=sum(padded[y:y+20,x:x+32] for y in range(3) for x in range(3))
                        direction=np.array([sun[0]/2,-sun[1]/2])/abs(dominant)
                        for y,x in np.argwhere(bright & (neighbors<=4)):
                            if dda(z,int(x),int(y),direction,rise,bias):
                                light[y,x]=31;removed+=1
                            else:retained+=1
                expected=light[uv[:,:,1],uv[:,:,0]]
                self.assertLessEqual(np.abs(actual.astype(int)-expected.astype(int)).max(),1,vector)
                full=self.renderer.capture(True)
                np.testing.assert_array_equal(full,actual)
            # Night and then daylight must recover without stale lighting.
            self.assertGreater(removed,0)
            self.assertGreater(retained,0)
        finally:self.release(tex,fb)

    def test_overlay_off_keeps_the_registered_map_visible_even_at_night(self):
        from map_projection.frames import SunScene,SunFrame
        z=np.zeros((4,4),np.float32);normals=np.zeros((4,4,3),np.float32);normals[:,:,2]=1
        y,x=np.mgrid[:4,:4];uv=np.stack((x,y),axis=-1).astype(np.uint16);uv[0,0]=65535
        scene=SunScene('plain-map',z,normals,uv,(4.,4.));tex,fb=self.target(4,4)
        try:
            self.renderer.draw(SunFrame(scene,np.array([1.,0.,-1.]),0,0,'night',False),fb,False)
            expected=np.full((4,4,3),255,np.uint8);expected[0,0]=0
            np.testing.assert_array_equal(self.read(4,4),expected)
        finally:self.release(tex,fb)

    def test_async_preview_keeps_the_requested_frame_after_scene_changes(self):
        from map_projection.frames import SunScene,SunFrame
        from OpenGL.GLES3 import glFinish
        z=np.zeros((8,8),np.float32);normals=np.zeros((8,8,3),np.float32);normals[:,:,2]=1
        y,x=np.mgrid[:8,:8];uv=np.stack((x,y),axis=-1).astype(np.uint16)
        scene=SunScene('async-capture',z,normals,uv,(8.,8.));tex,fb=self.target(8,8)
        try:
            frame=SunFrame(scene,np.array([0.,0.,1.]),0,0,'day')
            self.renderer.draw(frame,fb,False)
            expected=self.renderer.capture(True)
            pending=self.renderer.capture(True,asynchronous=True)
            self.renderer.draw(SunFrame(scene,np.array([0.,0.,-1.]),0,0,'night'),fb,False)
            glFinish();self.renderer.poll_captures()
            self.assertTrue(pending.ready);self.assertIsNone(pending.error)
            np.testing.assert_array_equal(pending.image,expected)
        finally:self.release(tex,fb)

    def test_preview_filters_checkerboard(self):
        from map_projection.frames import SunScene,SunFrame
        z=np.zeros((64,64),np.float32);v,u=np.mgrid[:64,:64]
        normals=np.zeros((64,64,3),np.float32);normals[:,:,2]=np.where((u+v)%2,1,-1)
        scene=SunScene('preview-checker',z,normals,np.stack((u,v),axis=-1).astype(np.uint16),(64.,64.))
        tex,fb=self.target(64,64)
        try:
            self.renderer.draw(SunFrame(scene,np.array([0.,0.,1.]),0,0,'test'),fb,physical=False)
            preview=self.renderer.capture(False,(16,16))
            self.assertTrue(np.all((preview>=126)&(preview<=129)))
        finally:self.release(tex,fb)


if __name__=='__main__':unittest.main()
