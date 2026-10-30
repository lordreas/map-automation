import copy
import io
from datetime import datetime
import json
from pathlib import Path
import struct
import tempfile
import time
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from map_projection.config import DEFAULT, ROOT, validate
from map_projection.engine import Engine
from map_projection.geometry import apply, geo_to_xy, homography, projector, sun_camera, unproject, xy_to_geo
from map_projection.native import Native
from map_projection.server import start
from map_projection.solar import position
from map_projection.terrain import Terrain, prepare_top
from map_projection.prepare_terrain import prepare


def fixture(path):
    # Ground and raised square, with vertical occluding walls and a duplicate
    # downward bottom. Geographic north is positive STL Y after export.
    vertices=[]
    def quad(a,b,c,d):vertices.extend(((a,b,c),(a,c,d)))
    quad((0,0,0),(10,0,0),(10,10,0),(0,10,0))
    quad((4,4,2),(6,4,2),(6,6,2),(4,6,2))
    quad((4,4,0),(6,4,0),(6,4,2),(4,4,2))
    quad((6,4,0),(6,6,0),(6,6,2),(6,4,2))
    quad((6,6,0),(4,6,0),(4,6,2),(6,6,2))
    quad((4,6,0),(4,4,0),(4,4,2),(4,6,2))
    quad((0,0,0),(0,10,0),(10,10,0),(10,0,0))
    with path.open("wb") as f:
        f.write(bytes(80));f.write(struct.pack("<I",len(vertices)))
        for triangle in vertices:
            f.write(struct.pack("<12fH",0,0,0,*np.array(triangle).ravel(),0))
    return np.array(vertices,dtype=float)-[5,5,0]


def reference_ray(triangles,origin,direction):
    """Independent NumPy Möller–Trumbore oracle, no rasterizer reuse."""
    a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
    p=np.cross(direction,e2);det=(p*e1).sum(axis=1)
    valid=np.abs(det)>1e-12
    inverse=np.zeros_like(det);inverse[valid]=1/det[valid]
    tvec=origin-a
    u=(tvec*p).sum(axis=1)*inverse
    q=np.cross(tvec,e1)
    v=(q*direction).sum(axis=1)*inverse
    t=(e2*q).sum(axis=1)*inverse
    valid&=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>1e-8)
    return origin+direction*t[valid].min() if valid.any() else None


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/".cache").mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/".cache")
        cls.path=Path(cls.temp.name)/"fixture.stl"
        cls.triangles=fixture(cls.path)
        cls.native=Native()
        cls.bounds=cls.native.scan(cls.path)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def config(self):
        c=copy.deepcopy(DEFAULT)
        c["output"].update(width=128,height=128)
        c["pixel"]={"x":64,"y":64}
        c["projector"].update(x=3.,y=5.,z=20.,throw_ratio=1.2)
        return c

    def test_nearest_intersections_and_inverse_for_off_axis_and_keystone(self):
        c=self.config()
        _,_,_,corners=projector(c,self.bounds)
        c["corners"]=(corners+[[3,2],[-2,1],[1,-4],[-3,0]]).tolist()
        matrix,inverse,origin,_=projector(c,self.bounds)
        depth,_=self.native.raster(self.path,self.bounds,matrix,128,128)
        rng=np.random.default_rng(123)
        for u,v in rng.integers(0,128,(180,2)):
            direction=inverse @ [u+.5,v+.5,1]
            ref=reference_ray(self.triangles,origin,direction)
            if ref is None:self.assertFalse(np.isfinite(depth[v,u]))
            else:
                actual=unproject((u,v),depth[v,u],inverse,origin)
                np.testing.assert_allclose(actual,ref,atol=3e-6)
                np.testing.assert_allclose(apply(matrix,actual)[0],[u+.5,v+.5],atol=1e-7)

    def test_direct_ray_matches_oracle_and_top_visibility(self):
        origin=np.array([0.,0.,20.]);direction=np.array([0.,0.,-1.])
        hit=self.native.ray(self.path,self.bounds,origin,direction)
        np.testing.assert_allclose(hit,[0,0,2])
        self.assertIsNone(self.native.ray(self.path,self.bounds,[20,20,20],[0,0,-1]))
        for target in ([0,0,0],[-3,0,0],[1,0,1]):
            ray=np.array(target)-origin
            np.testing.assert_allclose(self.native.ray(self.path,self.bounds,origin,ray),reference_ray(self.triangles,origin,ray))

    def test_mirror_changes_rays_and_inverse_consistently(self):
        c=self.config();m,inv,o,_=projector(c,self.bounds)
        c["output"]["mirror"]=True
        mirrored,mi,mo,_=projector(c,self.bounds)
        point=np.array([2.,1.,.4])
        a=apply(m,point)[0];b=apply(mirrored,point)[0]
        np.testing.assert_allclose(b,[128-a[0],a[1]])
        d,_=self.native.raster(self.path,self.bounds,m,128,128)
        md,_=self.native.raster(self.path,self.bounds,mirrored,128,128)
        np.testing.assert_allclose(d[:,::-1],md,rtol=1e-6)

    def test_cast_shadow_blocks_the_light(self):
        c=self.config();c["projector"].update(x=0,y=0)
        m,inv,origin,_=projector(c,self.bounds)
        depth,normals=self.native.raster(self.path,self.bounds,m,128,128)
        # A later coincident, downward-facing bottom must not replace the
        # upward surface because of sub-float depth rounding (ARM FMA).
        visible_normals=normals[np.isfinite(depth)]
        np.testing.assert_array_equal(visible_normals[:,2],127)
        sun=np.array([1.,0.,1.])/np.sqrt(2)
        sm,_=sun_camera(sun,self.bounds,512)
        shadow,_=self.native.raster(self.path,self.bounds,sm,512,512,False)
        frame=self.native.shade(depth,normals,inv,origin,sm,shadow,sun,.1,.001)
        def intensity(point):
            u,v=np.floor(apply(m,point)[0]).astype(int)
            return int(frame[v,u,0])
        self.assertLess(intensity([-2,0,0]),30)
        self.assertGreater(intensity([3,0,0]),180)

    def test_live_horizon_cardinal_and_diagonal(self):
        terrain=np.zeros((9,9),np.float32);terrain[4,4]=4
        sun=np.array([1.,0.,1.])/np.sqrt(2)
        result=np.empty_like(terrain)
        self.native.lib.horizon(terrain,9,9,1.,1.,sun,result)
        np.testing.assert_allclose(result[4],[0,1,2,3,4,0,0,0,0],atol=1e-5)
        np.testing.assert_allclose(result[3],0)
        sun=np.array([1.,1.,1.])/np.sqrt(3)
        self.native.lib.horizon(terrain,9,9,1.,1.,sun,result)
        self.assertAlmostEqual(result[5,3],3,places=5)
        self.assertAlmostEqual(result[6,2],2,places=5)

    def test_dem_ocean_is_flat_before_coastal_interpolation(self):
        # Deep sea, shallow sea, a void and coastal land must not create either
        # the old -1000 m cliff or a negative/NaN interpolation halo.
        with tempfile.TemporaryDirectory(dir=ROOT/".cache") as directory:
            directory=Path(directory);source=directory/"dem.npy"
            np.save(source,np.array([[-2000,-500,np.nan,100]]*2,np.float32))
            c=self.config();c["sun"].update(terrain="dem",grid_width=8)
            c["geo"].update(width_mm=4.,height_mm=1.)
            prepare(source,c['geo'],land_mask=np.array([[False,False,False,True]]*2),min_patch_px=1)
            terrain=Terrain(self.native,self.path,np.array([0,0,0,4,1,2.]),directory,"coast-test",source)
            z,_,_=terrain.get(c,lambda _:None)
            np.testing.assert_allclose(z[0],[0,0,0,0,0,25,75,100])
            np.testing.assert_allclose(z[1],z[0])
            # Reload the cached result, too.
            terrain.key=None
            np.testing.assert_array_equal(terrain.get(c,lambda _:None)[0],z)
            terrain.data=None  # Release the cache mmap before temporary cleanup.

    def test_live_renderer_casts_shadow_on_a_flat_receiver(self):
        c=self.config();c["projector"].update(x=0,y=0)
        matrix,inverse,origin,_=projector(c,self.bounds)
        depth,normals=self.native.raster(self.path,self.bounds,matrix,128,128)
        terrain=np.zeros((9,9),np.float32);terrain[4,4]=4
        sun=np.array([1.,0.,1.])/np.sqrt(2)
        frame=self.native.live(depth,normals,inverse,origin,terrain,(10.,10.),(10.,10.),sun,.1,.001,True)
        def intensity(point):
            u,v=np.floor(apply(matrix,point)[0]).astype(int)
            return int(frame[v,u,0])
        # Both receivers have a flat local DEM normal. Only the one behind the
        # peak is dark, demonstrating occlusion separately from normal shading.
        self.assertLess(intensity([-3,0,0]),30)
        self.assertGreater(intensity([3,0,0]),180)

    def test_fast_cpu_dem_matches_reference_in_all_sun_quadrants(self):
        c=self.config();matrix,inverse,origin,_=projector(c,self.bounds)
        depth,normals=self.native.raster(self.path,self.bounds,matrix,128,128)
        z=np.zeros((9,9),np.float32);z[4,4]=4;z[1:3,1:3]=-2
        terrain_normals,lookup=self.native.lighting_scene(depth,inverse,origin,z,(10.,10.),(10.,10.))
        work={}
        for vector in ([1,0,1],[-1,0,1],[0,1,1],[0,-1,1],[1,1,1],[-1,1,1],[1,-1,1],[-1,-1,1],[0,0,1],[1,0,-1]):
            sun=np.array(vector,dtype=float);sun/=np.linalg.norm(sun)
            reference=self.native.live(depth,normals,inverse,origin,z,(10.,10.),(10.,10.),sun,.1,.001,True)
            fast=self.native.fast_live(z,terrain_normals,lookup,(10.,10.),sun,.1,.001,work)
            self.assertLessEqual(np.abs(reference.astype(int)-fast.astype(int)).max(),1,vector)

    def test_sun_day_night_and_offset_equivalence(self):
        a,_=position("2026-06-21T12:00:00+02:00",46,10)
        b,_=position("2026-06-21T10:00:00Z",46,10)
        np.testing.assert_allclose(a,b)
        _,noon=position("2026-03-20T12:00:00Z",0,0)
        self.assertGreater(noon["altitude"],87)
        night,_=position("2026-06-21T00:00:00+02:00",46,10)
        self.assertLess(night[2],0)

    def test_swiss_dst_and_nonexistent_time(self):
        c=self.config();c["sun"]["time"]="2026-07-01T12:00:00"
        self.assertTrue(validate(c)["sun"]["time"].endswith("+02:00"))
        c["sun"]["time"]="2026-12-01T12:00:00"
        self.assertTrue(validate(c)["sun"]["time"].endswith("+01:00"))
        c["sun"]["time"]="2026-03-29T02:30:00"
        with self.assertRaises(ValueError):validate(c)

    def test_geo_north_and_roundtrip(self):
        geo=DEFAULT["geo"]
        bounds=np.array([0,0,0,1500,geo["height_mm"],45.])
        x,y=geo_to_xy(46.5,7.2,bounds,geo)
        self.assertGreater(y,0)
        actual=xy_to_geo(x,y,bounds,geo)
        self.assertAlmostEqual(actual["latitude"],46.5)
        self.assertAlmostEqual(actual["longitude"],7.2)

    def test_top_filter_preserves_original_vertices(self):
        top=Path(self.temp.name)/"top.stl"
        self.assertEqual(prepare_top(self.path,top),4)
        hit=self.native.ray(top,self.bounds,[0,0,20],[0,0,-1])
        np.testing.assert_allclose(hit,[0,0,2])

    def test_single_pixel_and_checkerboard_have_no_intermediate_values(self):
        e=object.__new__(Engine);c=self.config();c["pattern"]="pixel"
        frame=e.render(c)
        self.assertEqual(np.count_nonzero(frame[:,:,0]),1)
        self.assertEqual(frame[64,64].tolist(),[255,255,255])
        c["pattern"]="checkerboard";frame=e.render(c)
        self.assertEqual(set(np.unique(frame)),{0,255})
        self.assertTrue(np.all(frame[:,1:,0]!=frame[:,:-1,0]))

    def test_preview_antialiases_without_changing_projector_pixels(self):
        e=object.__new__(Engine);e.lock=threading.RLock()
        c=self.config();c["pattern"]="checkerboard";e.frame=e.render(c)
        preview=np.asarray(Image.open(io.BytesIO(e.preview(size=(32,32)))))
        self.assertEqual(preview.shape,(32,32,3))
        self.assertTrue(np.all((preview>100)&(preview<155)))
        physical=np.asarray(Image.open(io.BytesIO(e.preview(full=True))))
        np.testing.assert_array_equal(physical,e.frame)
        self.assertEqual(set(np.unique(physical)),{0,255})

    def test_repeated_sun_request_and_late_job_do_not_replace_calibration(self):
        c=self.config();c["pattern"]="calibration"
        e=Engine(self.path,c,cache=Path(self.temp.name))
        started=threading.Event();release=threading.Event();finished=threading.Event()
        original=e.render
        def slow_render(config):
            if config["pattern"]=="sun":
                started.set();release.wait(3)
                e.set_status("Old job finishing")
                finished.set()
                return np.full((128,128,3),255,np.uint8)
            return original(config)
        try:
            deadline=time.monotonic()+3
            while e.processed!=e.revision and time.monotonic()<deadline:time.sleep(.01)
            e.render=slow_render
            e.update({"pattern":"sun"});self.assertTrue(started.wait(2))
            revision=e.revision
            for _ in range(5):e.update({"pattern":"sun"})
            self.assertEqual(e.revision,revision)
            self.assertTrue(e.snapshot()["pending"])
            self.assertEqual(e.snapshot()["displayed_pattern"],"calibration")
            e.nudge(0,1,0,5)
            calibration=e.frame.copy()
            release.set();self.assertTrue(finished.wait(2));time.sleep(.05)
            self.assertEqual(e.snapshot()["status"],"Ready")
            self.assertFalse(e.snapshot()["pending"])
            np.testing.assert_array_equal(e.frame,calibration)
            e.update({"pattern":"sun"})
            deadline=time.monotonic()+2
            while e.processed!=e.revision and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(e.frame_pattern,"sun")
            self.assertFalse(e.snapshot()["pending"])
        finally:
            release.set();e.close();e.worker.join(timeout=5)

    def test_degenerate_calibration_and_nonfinite_config_rejected(self):
        with self.assertRaises(ValueError):homography([[0,0],[1,0],[1,1],[0,1]],[[0,0],[0,0],[2,2],[0,2]])
        c=self.config();c["projector"]["x"]=float("nan")
        with self.assertRaises(ValueError):validate(c)
        c=self.config();c["output"]["width"]=128.0
        with self.assertRaises(ValueError):validate(c)

    def test_http_validation_pixel_nudge_and_persistence(self):
        c=self.config();c["pattern"]="calibration"
        settings=Path(self.temp.name)/"settings.json"
        engine=Engine(self.path,c,settings,cache=Path(self.temp.name))
        server=start(engine,"127.0.0.1",0)
        url=f"http://127.0.0.1:{server.server_port}"
        def post(path,data,origin=None):
            headers={"Content-Type":"application/json"}
            if origin:headers["Origin"]=origin
            request=Request(url+path,json.dumps(data).encode(),headers)
            with urlopen(request) as response:return json.load(response)
        try:
            before=engine.snapshot()["corners"][0]
            after=post("/api/nudge",{"corner":0,"dx":1,"dy":0})
            np.testing.assert_allclose(after["corners"][0],[before[0]+1,before[1]])
            self.assertEqual(json.loads(settings.read_text())["corners"],after["config"]["corners"])
            after=post("/api/nudge",{"corner":0,"dx":0,"dy":1,"step":5})
            np.testing.assert_allclose(after["corners"][0],[before[0]+1,before[1]+5])
            after=post("/api/nudge",{"corner":0,"dx":0,"dy":-1,"step":10})
            np.testing.assert_allclose(after["corners"][0],[before[0]+1,before[1]-5])
            with self.assertRaises(HTTPError):post("/api/nudge",{"corner":0,"dx":1,"dy":0,"step":7})
            with self.assertRaises(HTTPError) as error:post("/api/config",{"projector":{"z":0}})
            self.assertEqual(error.exception.code,400)
            with self.assertRaises(HTTPError) as error:post("/api/config",{},"https://unrelated.example")
            self.assertEqual(error.exception.code,403)
            with self.assertRaises(HTTPError) as error:post("/api/config",{"output":{"width":float("nan")}})
            self.assertEqual(error.exception.code,400)
        finally:
            engine.close();engine.worker.join(timeout=5);engine.depth=engine.normals=None
            server.shutdown();server.server_close()


if __name__=="__main__":unittest.main()
