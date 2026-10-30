import json,tempfile,unittest,os
from pathlib import Path
import numpy as np
from map_projection.horizon_cache import Horizons,Events,identity
class CacheTests(unittest.TestCase):
 def test_angular_wrap_and_geometry_identity(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);np.save(p/'horizons.npy',np.arange(4,dtype=np.float32).reshape(4,1,1));(p/'manifest.json').write_text(json.dumps({'schema':1,'complete':True}))
   h=Horizons(p);self.assertAlmostEqual(float(h.at(315)[0]),1.5)
  z=np.zeros((4,4),np.float32);self.assertNotEqual(identity(z,[4,4],.1),identity(z,[4,4],.2))
 def test_event_parity_and_arbitrary_time_seeks(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);np.save(p/'initial.npy',np.array([0,1],np.uint8));np.save(p/'offsets.npy',np.array([0,2,3],np.uint32));np.save(p/'times.npy',np.array([1000,2000,3000],np.uint32));(p/'manifest.json').write_text(json.dumps({'complete':True,'shape':[1,2]}))
   e=Events(p)
   for t,expected in [(4,[0,0]),(1,[1,1]),(0,[0,1]),(2,[0,1])]:np.testing.assert_array_equal(e.visibility(t)[0],expected)
 def test_fractional_seconds_affect_solar_direction(self):
  from map_projection.solar import position
  a,_=position('2026-10-25T12:00:00+00:00',46,10);b,_=position('2026-10-25T12:00:00.5+00:00',46,10)
  self.assertGreater(np.linalg.norm(a-b),0)
 def test_vector_sun_matches_scalar_through_year_and_leap_boundaries(self):
  from map_projection.solar import position,positions_for_day
  from datetime import datetime,timedelta,timezone
  seconds=np.array([0,.03125,21600.5,43200.125,86399.96875,86400])
  for day in ['2026-10-25','2024-02-29','2024-12-31','2026-12-31']:
   az,el=positions_for_day(day,seconds,45.95,10.65)
   for i,t in enumerate(seconds):
    _,s=position((datetime.fromisoformat(day).replace(tzinfo=timezone.utc)+timedelta(seconds=float(t))).isoformat(),45.95,10.65)
    self.assertAlmostEqual(az[i],s['azimuth'],places=10);self.assertAlmostEqual(el[i],s['altitude'],places=10)

@unittest.skipUnless(os.environ.get('MAP_PROJECTION_GPU_TESTS')=='1','Requires GLES')
class EventGpuTests(unittest.TestCase):
 def test_incremental_replay_handles_forward_reverse_and_multiple_events(self):
  import ctypes as C
  from map_projection.experiments.gpu_context import context
  context()
  from map_projection.gpu import Renderer,texture,framebuffer
  from map_projection.frames import SunScene,SunFrame
  from OpenGL.GLES3 import GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,glReadPixels
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder);initial=np.array([0,1,0,1],np.uint8);offsets=np.array([0,4,5,5,6],np.uint32);times=np.array([1000,2000,2500,3000,3000,1500],np.uint32);order=np.argsort(times);cells=np.repeat(np.arange(4,dtype=np.uint32),np.diff(offsets))
   for name,a in [('initial',initial),('offsets',offsets),('times',times),('sorted-times',times[order]),('sorted-cells',cells[order])]:np.save(p/(name+'.npy'),a)
   (p/'manifest.json').write_text(json.dumps({'complete':True,'shape':[2,2],'start':'2026-10-25T00:00:00+00:00'}));e=Events(p);z=np.zeros((2,2),np.float32);normal=np.zeros((2,2,3),np.float32);normal[:,:,2]=1;y,x=np.mgrid[:2,:2];scene=SunScene('event-test',z,normal,np.stack((x,y),-1).astype(np.uint16),(2,2),e);r=Renderer();tex=texture(13,None,GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,2,2);fb=framebuffer(tex)
   try:
    for t in [0,1,4,2,2.75,1.5,0,4]:
     from datetime import datetime,timedelta,timezone
     stamp=(datetime(2026,10,25,tzinfo=timezone.utc)+timedelta(seconds=t)).isoformat();r.draw(SunFrame(scene,np.array([0.,0.,1.]),0,0,stamp,True,'events'),fb,False);image=np.empty((2,2,4),np.uint8);glReadPixels(0,0,2,2,GL_RGBA,GL_UNSIGNED_BYTE,image.ctypes.data_as(C.c_void_p));np.testing.assert_array_equal(image[:,:,0]>0,e.visibility(t))
     np.testing.assert_array_equal(r.capture(True)>0,e.visibility(t))
     # Downsampling shades the tiny destination directly and averages all four
     # source pixels, rather than creating an intermediate terrain light grid.
     expected=round(float(e.visibility(t).mean())*255)
     self.assertLessEqual(abs(int(r.capture(False,(1,1))[0,0])-expected),1)
     r.draw(SunFrame(scene,np.array([0.,0.,1.]),0,0,stamp,True,'events'),fb,True)
     np.testing.assert_array_equal(r.capture(True,output_framebuffer=fb)>0,e.visibility(t))
     self.assertLessEqual(abs(int(r.capture(False,(1,1),output_framebuffer=fb)[0,0])-expected),1)
     r.snapshot_output(output_framebuffer=fb,width=2,height=2)
     from unittest.mock import patch
     with patch.object(r.event_renderer,'preview',side_effect=AssertionError('Paused thumbnail must reuse its snapshot')):
      self.assertLessEqual(abs(int(r.capture(False,(1,1))[0,0])-expected),1)
    r.draw(SunFrame(scene,np.array([0.,0.,-1.]),0,0,stamp,False,'events'),fb,False)
    np.testing.assert_array_equal(r.capture(True),np.full((2,2),255,np.uint8))
   finally:r.close()
