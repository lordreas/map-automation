import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from map_projection.config import DEFAULT,validate
from map_projection.geometry import projector,unproject,xy_to_geo
from map_projection.layers import project_uv,peak_field,compose_cpu
from map_projection.native import Native
from map_projection.prepare_layers import read_elevation
from map_projection.tests.test_projection import fixture


class LayerTests(unittest.TestCase):
 def test_neutral_layer_stack_preserves_existing_sunlight_frame_without_extra_pass(self):
  from types import SimpleNamespace
  from map_projection.engine import Engine
  from map_projection.frames import SunFrame,SunScene
  c=copy.deepcopy(DEFAULT);c['output'].update(width=64,height=64)
  scene=SunScene('test',np.zeros((2,2),np.float32),np.ones((2,2,3),np.float32),np.zeros((64,64,2),np.uint16),(1,1))
  frame=SunFrame(scene,np.array([0,0,1]),0,0,c['sun']['time'])
  state={'test':'layer state'};e=Engine.__new__(Engine);e.sun_frame=lambda c:frame;e.geometry=lambda c:(None,None,None,'geometry');e.depth=None;e.bounds=None;e.layers=SimpleNamespace(prepare=lambda *args:state)
  rendered=e.render(c);self.assertIs(rendered.scene,frame.scene);self.assertIsNone(rendered.layers)
  c['layers']['white']['enabled']=False
  self.assertIs(e.render(c).layers,state)

 def test_uv_uses_first_stl_hit_including_raised_surface_and_invalid_pixels(self):
  with tempfile.TemporaryDirectory() as folder:
   mesh=Path(folder)/'mesh.stl';fixture(mesh);n=Native();bounds=n.scan(mesh)
   c=copy.deepcopy(DEFAULT);c['output'].update(width=96,height=96);c['projector'].update(y=12,z=30,throw_ratio=2)
   c['geo'].update(lat_min=0,lat_max=1,lon_min=0,lon_max=1,width_mm=10,height_mm=10)
   _,inv,origin,_=projector(c,bounds);depth,_=n.raster(mesh,bounds,projector(c,bounds)[0],96,96)
   uv=project_uv(depth,inv,origin,bounds,c['geo']);self.assertTrue(np.all(uv[~np.isfinite(depth)]==65535))
   yy,xx=np.where(uv[:,:,0]!=65535);self.assertGreater(len(xx),100)
   heights=[]
   for x,y in zip(xx[::35],yy[::35]):
    hit=unproject((x,y),depth[y,x],inv,origin);geo=xy_to_geo(*hit[:2],bounds,c['geo']);expected=[geo['longitude'],1-geo['latitude']]
    np.testing.assert_allclose(uv[y,x]/65534,expected,atol=1/65534);heights.append(hit[2])
   self.assertGreater(max(heights),1.9)

 def test_peak_filter_distance_is_exact_and_handles_no_selected_summits(self):
  points=np.array([[2,2,1000],[6,5,4000],[4,3,2000]],np.float32)
  for low,high in [(0,5000),(1500,2500),(3500,5000),(5000,6000)]:
   a,count=peak_field(points,(8,9),low,high);chosen=points[(points[:,2]>=low)&(points[:,2]<=high)]
   self.assertEqual(count,len(chosen));y,x=np.mgrid[:8,:9]
   if count:
    expected=np.sqrt(np.min([(x-p[0])**2+(y-p[1])**2 for p in chosen],axis=0));np.testing.assert_allclose(a,expected,atol=.004)
   else:self.assertTrue(np.all(a>50000))

 def test_black_when_all_layers_disabled_and_shading_multiplies_color(self):
  cfg=copy.deepcopy(DEFAULT['layers'])
  for value in cfg.values():value['enabled']=False
  uv=np.zeros((8,8,2),np.uint16);state={'uv':uv,'config':cfg,'arrays':{}}
  frame=np.full((8,8,3),128,np.uint8)
  self.assertFalse(np.any(compose_cpu(frame,state)))
  cfg['satellite'].update(enabled=True,brightness=1.);state['arrays']['satellite']=np.full((2,2,3),[200,100,50],np.uint8)
  out=compose_cpu(frame,state);np.testing.assert_array_equal(out[0,0],[100,50,25])
  state['uv'][0,0]=65535;self.assertFalse(np.any(compose_cpu(frame,state)[0,0]))

 def test_config_limits_elevation_units_and_width(self):
  self.assertEqual(read_elevation('4,810 m'),4810)
  self.assertEqual(read_elevation('4810,5 m'),4810.5)
  self.assertAlmostEqual(read_elevation('100 ft'),30.48)
  self.assertIsNone(read_elevation('unknown'))
  for patch in [dict(width=0),dict(enabled='true')]:
   c=copy.deepcopy(DEFAULT);c['layers']['rivers'].update(patch)
   with self.assertRaises(ValueError):validate(c)
  c=copy.deepcopy(DEFAULT);c['layers']['peaks'].update(min_elevation=4000,max_elevation=1000)
  with self.assertRaises(ValueError):validate(c)
