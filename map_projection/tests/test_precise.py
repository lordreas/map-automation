"""Focused equivalence check for conservative bounds, witnesses and ray fallback."""
import os,unittest,ctypes as C
import numpy as np

@unittest.skipUnless(os.environ.get('MAP_PROJECTION_GPU_TESTS')=='1','Requires real GLES')
class PreciseTests(unittest.TestCase):
    def test_matches_reference_across_directions_and_cached_witnesses(self):
        from map_projection.experiments.gpu_context import context
        context()
        from map_projection.gpu import Renderer,texture,framebuffer
        from map_projection.frames import SunScene,SunFrame
        from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,glReadPixels,glDeleteTextures,glDeleteFramebuffers,glFinish
        y,x=np.mgrid[:47,:79];rng=np.random.default_rng(157)
        z=(50*np.sin(x/6)*np.cos(y/8)+rng.uniform(-10,10,x.shape)).astype(np.float32)
        z[:8,:]=0;z[:,60:]=0
        normals=np.zeros((*z.shape,3),np.float32);normals[:,:,2]=1
        scene=SunScene('certified-partial-blocks',z,normals,np.stack((x,y),axis=-1).astype(np.uint16),(158.,94.))
        reference=Renderer();fast=Renderer();tex=texture(13,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,79,47);fb=framebuffer(tex)
        try:
            vectors=[(1,.1,.05),(1,.11,.049),(1,.12,.048),(1,.11,.049),(-1,.4,.1),(0,1,.1),(.01,-1,.02),(-1,-1,.1),(1,1,.1),(1,-1,.1),(1,0,.3),(1,0,10)]
            for vector in vectors:
                sun=np.array(vector,float);sun/=np.linalg.norm(sun)
                images=[]
                for renderer,method in [(reference,'reference'),(fast,'precise')]:
                    renderer.draw(SunFrame(scene,sun,.12,.08,'test',True,method),fb,False)
                    a=np.empty((47,79,4),np.uint8);glReadPixels(0,0,79,47,GL_RGBA,GL_UNSIGNED_BYTE,a.ctypes.data_as(C.c_void_p));images.append(a.copy())
                self.assertLessEqual(np.abs(images[0].astype(int)-images[1].astype(int)).max(),1,vector)
            glFinish();fast.precise.check(wait=True)
        finally:
            reference.close();fast.close();glDeleteTextures(1,[tex]);glDeleteFramebuffers(1,[fb])
