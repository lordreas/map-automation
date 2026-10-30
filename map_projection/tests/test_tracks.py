import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from map_projection.config import DEFAULT,validate
from map_projection.geometry import projector,unproject,xy_to_geo,apply
from map_projection.native import Native
from map_projection.tracks import read_gpx,raster_track,project_track,TrackLayer,random_color
from map_projection.tests.test_projection import fixture


class TrackTests(unittest.TestCase):
    def test_segments_stay_separate_and_reveal_uses_distance(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'track.gpx';p.write_text('<gpx xmlns="urn:test"><trk><trkseg><trkpt lat=".5" lon=".1"/><trkpt lat=".5" lon=".2"/></trkseg><trkseg><trkpt lat=".5" lon=".8"/><trkpt lat=".5" lon=".9"/></trkseg></trk></gpx>')
            segments=read_gpx(p);self.assertEqual(len(segments),2)
            geo=dict(lat_min=0,lat_max=1,lon_min=0,lon_max=1,width_mm=10,height_mm=10)
            mask,phase,length=raster_track(segments,geo,128)
            self.assertEqual(mask[64,64],0);self.assertGreater(mask[64,20],0);self.assertGreater(mask[64,110],0)
            self.assertLess(phase[64,20],phase[64,110]);self.assertTrue(21000<length<23000)
            self.assertTrue(np.all((phase>=0)&(phase<=1)))

    def test_registered_pixels_use_raised_stl_surface_and_actual_first_hit(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=Path(folder)/'mesh.stl';fixture(mesh);native=Native();bounds=native.scan(mesh)
            c=copy.deepcopy(DEFAULT);c['output'].update(width=96,height=96);c['projector'].update(x=0,y=12,z=30,throw_ratio=2)
            c['geo'].update(lat_min=0,lat_max=1,lon_min=0,lon_max=1,width_mm=10,height_mm=10)
            matrix,inverse,origin,_=projector(c,bounds);depth,_=native.raster(mesh,bounds,matrix,96,96)
            segments=[np.array([[.5,.05],[.5,.95]])];mask,phase,_=raster_track(segments,c['geo'],256)
            points=project_track(depth,inverse,origin,bounds,c['geo'],mask,phase)
            self.assertGreater(len(points),20)
            x=np.rint((points[:,0]+1)*48-.5).astype(int);y=np.rint((1-points[:,1])*48-.5).astype(int)
            hit=np.array([unproject((u,v),depth[v,u],inverse,origin) for u,v in zip(x,y)])
            latitude=np.array([xy_to_geo(*q[:2],bounds,c['geo'])['latitude'] for q in hit])
            self.assertLess(np.max(np.abs(latitude-.5)),.025)
            self.assertGreater(np.max(hit[:,2]),1.9)
            top=apply(matrix,[[0,0,2]])[0]
            self.assertLess(np.min(np.linalg.norm(np.column_stack((x+.5,y+.5))-top,axis=1)),2)
            self.assertGreater(np.linalg.norm(top-apply(matrix,[[0,0,0]])[0]),1)

    def test_color_validation_and_reveal_waits_for_first_presentation(self):
        c=copy.deepcopy(DEFAULT);c['track']['color']='#nocolr'
        with self.assertRaises(ValueError):validate(c)
        for _ in range(8):self.assertRegex(random_color(),r'^#[0-9a-f]{6}$')
        layer=TrackLayer();layer.configure({'enabled':True,'color':'#123456'});layer.vertices=np.zeros((1,4),np.float32);layer.key=('a',)
        self.assertEqual(layer.state()['progress'],0)
        layer.presented();self.assertIsNotNone(layer.started)
        layer.replay();self.assertIsNone(layer.started);self.assertEqual(layer.color,'#123456')
