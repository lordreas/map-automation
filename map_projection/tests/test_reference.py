"""Analytic terrain intersections and independent GPU/float64 comparisons."""
import os,unittest
import numpy as np
from map_projection.reference import bilinear_blocked

class ReferenceTests(unittest.TestCase):
    def test_flat_surface_and_ramp_have_no_cell_wall_self_shadows(self):
        y,x=np.mgrid[:12,:18]
        for z in (np.zeros_like(x),3*x+2*y):
            self.assertFalse(bilinear_blocked(z,3,4,[1,.2],4,0.001))
        self.assertTrue(bilinear_blocked(3*x+2*y,3,4,[1,.2],2,0.001))

    def test_interior_of_bilinear_patch_can_occlude_even_when_endpoints_do_not(self):
        # z(u,v)=4u(1-v); along u=v=s, z=4s(1-s).
        z=np.array([[0,4],[0,0]],np.float32)
        self.assertTrue(bilinear_blocked(z,0,0,[1,1],0,.9))
        self.assertFalse(bilinear_blocked(z,0,0,[1,1],0,1.1))

    def test_boundary_parallel_ray_still_intersects_the_edge_surface(self):
        z=np.zeros((8,8));z[4,:]=10
        self.assertTrue(bilinear_blocked(z,7,1,[0,1],.1,.01))
        self.assertTrue(bilinear_blocked(z,0,6,[0,-1],.1,.01))

    def test_reference_configuration_requires_the_geographic_live_surface(self):
        import copy
        from map_projection.config import DEFAULT,validate
        c=copy.deepcopy(DEFAULT);c['sun'].update(backend='reference',terrain='dem',renderer='live')
        validate(c)
        for changes in ({'terrain':'stl'},{'renderer':'mesh'}):
            invalid=copy.deepcopy(c);invalid['sun'].update(changes)
            with self.assertRaises(ValueError):validate(invalid)

    def test_ridge_shadow_crosses_only_once_when_altitude_decreases(self):
        z=np.zeros((12,24));z[:,12]=10
        blocked=[bilinear_blocked(z,3,5,[1,.1],rise,.01) for rise in np.linspace(2,0,100)]
        self.assertEqual(sum(a!=b for a,b in zip(blocked,blocked[1:])),1)

@unittest.skipUnless(os.environ.get('MAP_PROJECTION_GPU_TESTS')=='1','Set MAP_PROJECTION_GPU_TESTS=1 for GLES tests')
class ReferenceGpuTests(unittest.TestCase):
    def test_all_pixels_match_independent_float64_patch_traversal(self):
        from map_projection.experiments.gpu_context import context
        context()
        from map_projection.gpu import Renderer,texture,framebuffer
        from map_projection.frames import SunScene,SunFrame
        from OpenGL.GLES3 import glBindFramebuffer,glReadPixels,GL_FRAMEBUFFER,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,glDeleteTextures,glDeleteFramebuffers
        import ctypes as C
        rng=np.random.default_rng(56);h,w=19,27;z=rng.uniform(-2,15,(h,w)).astype(np.float32)
        y,x=np.mgrid[:h,:w];uv=np.stack((x,y),axis=-1).astype(np.uint16)
        normals=np.zeros((h,w,3),np.float32);normals[:,:,2]=1
        scene=SunScene('oracle-terrain',z,normals,uv,(w,h));r=Renderer()
        tex=texture(10,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,w,h);fb=framebuffer(tex)
        try:
            for dx,dy in [(1,.37),(-1,.99),(.001,1),(0,-1),(1,1),(-.7,-1)]:
                sun=np.array([dx,-dy,.43]);sun/=np.linalg.norm(sun)
                r.draw(SunFrame(scene,sun,0,.08,'test',True,'reference'),fb,False)
                glBindFramebuffer(GL_FRAMEBUFFER,r.filtered_fb)
                actual=np.empty((h,w,4),np.uint8);glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,actual.ctypes.data_as(C.c_void_p))
                expected=np.array([[bilinear_blocked(z,x,y,[dx,dy],.43,.08) for x in range(w)] for y in range(h)])
                np.testing.assert_array_equal(actual[:,:,0]==0,expected,err_msg=str((dx,dy)))
        finally:
            r.close();glDeleteTextures(1,[tex]);glDeleteFramebuffers(1,[fb])
