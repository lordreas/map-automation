import copy
from datetime import datetime,timedelta
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from elevation_surface import apply_sea_level_model,_replace_hgt_voids_with_nan
from map_projection.config import DEFAULT,load,validate
from map_projection.playback import PlaybackClock,TimeTransition
from map_projection.prepare_terrain import prepare,prepared
from map_projection.tests.test_projection import fixture
from map_projection.engine import Engine


class SurfaceTests(unittest.TestCase):
    def test_shared_mask_disagreement_floor_and_connected_patch_rules(self):
        z=np.full((8,12),50,np.float32);land=np.ones(z.shape,bool)
        z[1:3,1:4]=-5                 # six connected land cells: retained
        z[1,1]=-500                  # disagreement with bathymetry: floor
        z[4,2]=-3;z[5,3]=-3           # diagonal specks: removed
        z[6,7:10]=-300;land[6,7:10]=False  # ocean: zero
        z[0,10]=40;land[0,10]=False   # positive land never cut by mask
        z[7,0]=-32768                # void cannot become a false land patch
        clean=_replace_hgt_voids_with_nan(z)
        out=apply_sea_level_model(clean,land,-20,min_patch_px=4)
        self.assertEqual(out[1,1],-20);self.assertEqual(out[2,3],-5)
        self.assertEqual(out[4,2],0);self.assertEqual(out[5,3],0)
        np.testing.assert_array_equal(out[6,7:10],0)
        self.assertEqual(out[0,10],40);self.assertTrue(np.isnan(out[7,0]))
        np.testing.assert_array_equal(z[6,7:10],-300)

    def test_prepared_surface_matches_shared_policy_and_rejects_stale_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'dem.npy'
            z=np.array([[-100,-7,-6,20],[-50,-5,-4,30]],np.int16);np.save(path,z)
            mask=np.array([[False,True,True,False],[False,True,True,True]])
            p=prepare(path,DEFAULT['geo'],land_mask=mask,min_patch_px=3)
            result=np.load(p)
            np.testing.assert_array_equal(result,[[0,-7,-6,20],[0,-5,-4,30]])
            self.assertEqual(result.dtype,np.int16)
            self.assertEqual(prepared(path,DEFAULT['geo'])[0],p)
            np.save(path,z+1)
            with self.assertRaisesRegex(ValueError,'stale'):prepared(path,DEFAULT['geo'])


class PlaybackTests(unittest.TestCase):
    def sun(self,**kw):
        s=copy.deepcopy(DEFAULT['sun']);s.update(time='2026-10-24T17:00:00+02:00',minutes_per_second=15)
        s.update(kw);return s

    def test_speed_is_independent_of_render_cadence(self):
        clock=PlaybackClock(self.sun(),100)
        initial=datetime.fromisoformat(self.sun()['time'])
        for cadence in [1/15,1/60,.31]:
            count=int(2/cadence)
            for n in range(count):clock.sample(100+n*cadence)
            self.assertEqual(clock.sample(102)[0],initial+timedelta(minutes=30))

    def test_full_day_loop_and_dst_day_lengths(self):
        s=self.sun(time='2026-10-24T23:30:00+02:00',minutes_per_second=60)
        clock=PlaybackClock(s,10)
        self.assertEqual(clock.sample(11)[0].isoformat(),'2026-10-24T00:30:00+02:00')
        for day,hours in [('2026-03-29',23),('2026-10-25',25)]:
            c=PlaybackClock(self.sun(time=day+'T00:00:00+01:00'),0)
            self.assertEqual(c.end-c.begin,hours*3600)

    def test_now_and_return_ease_in_out_and_hit_exact_endpoints(self):
        from zoneinfo import ZoneInfo
        zone=ZoneInfo('Asia/Kathmandu');begin=datetime(2026,10,25,9,tzinfo=zone)
        target=begin+timedelta(hours=4)
        motion=TimeTransition(begin,'Asia/Kathmandu',target)
        self.assertEqual(motion.sample(100)[0],begin)
        motion.start(100)
        self.assertEqual(motion.sample(100.5)[0],begin+timedelta(hours=2))
        self.assertLess((motion.sample(100.1)[0]-begin).total_seconds(),4*3600*.1)
        self.assertEqual(motion.sample(101),(target,True))
        now=TimeTransition(begin,'Asia/Kathmandu');now.start(100,wall=target)
        self.assertEqual(now.sample(101)[0],target+timedelta(seconds=1))

    def test_validation_and_saved_dates(self):
        for patch in [{'minutes_per_second':0},{'minutes_per_second':float('inf')},{'minutes_per_second':61},{'enabled':1}]:
            c=copy.deepcopy(DEFAULT);c['sun'].update(patch)
            with self.assertRaises(ValueError):validate(c)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'settings.json';c=copy.deepcopy(DEFAULT)
            c['sun'].update(date_mode='fixed',time='2026-01-03T14:00:00+01:00');path.write_text(json.dumps(c))
            self.assertTrue(load(path)['sun']['time'].startswith('2026-01-03'))
            c['sun']['date_mode']='today';path.write_text(json.dumps(c))
            from zoneinfo import ZoneInfo
            self.assertTrue(load(path)['sun']['time'].startswith(datetime.now(ZoneInfo(c['sun']['timezone'])).date().isoformat()))

    def test_migrate_retired_controls_and_installation_timezone(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json';c=copy.deepcopy(DEFAULT)
            c['sun'].update(frame_rate=15,range_start=600,range_end=900,loop=False,date_mode='fixed')
            path.write_text(json.dumps(c))
            loaded=load(path,timezone_override='Asia/Kathmandu')
            self.assertEqual(loaded['sun']['timezone'],'Asia/Kathmandu')
            self.assertTrue(loaded['sun']['time'].endswith('+05:45'))
            for key in ('frame_rate','range_start','range_end','loop'):self.assertNotIn(key,loaded['sun'])

    def test_clock_events_report_the_presented_frame(self):
        from map_projection.frames import SunScene,SunFrame
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mesh.stl';fixture(path);c=copy.deepcopy(DEFAULT)
            c['output'].update(width=128,height=128);c['pattern']='calibration'
            c['pixel'].update(x=64,y=64);c['projector'].update(x=3.,y=5.,z=20.)
            engine=Engine(path,c,cache=Path(folder))
            try:
                engine.close();engine.worker.join(3)
                scene=SunScene('clock',np.zeros((1,1),np.float32),np.ones((1,1,3),np.float32),np.zeros((128,128,2),np.uint16),(1.,1.))
                frame=SunFrame(scene,np.array([0.,0.,1.]),0,0,'2026-10-25T16:21:40+01:00')
                engine.presented(frame)
                self.assertEqual(engine.time_event()['time'],frame.time)
                engine.frame_time='2026-10-25T16:22:00+01:00'
                self.assertEqual(engine.time_event()['time'],frame.time)
            finally:engine.close();engine.worker.join(3)

    def test_worker_live_speed_scrub_and_stop_returns_to_playback_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'mesh.stl';fixture(path);c=copy.deepcopy(DEFAULT)
            c['output'].update(width=128,height=128);c['pattern']='calibration'
            c['pixel'].update(x=64,y=64)
            c['projector'].update(x=3.,y=5.,z=20.,throw_ratio=1.2)
            c['sun'].update(time='2026-10-24T12:00:00+02:00',minutes_per_second=1)
            engine=Engine(path,c,cache=Path(folder))
            def ready():
                deadline=time.monotonic()+3
                while engine.processed!=engine.revision and time.monotonic()<deadline:time.sleep(.005)
                self.assertIsNone(engine.error)
            try:
                ready();engine.render=lambda _:np.zeros((128,128,3),np.uint8)
                engine.update({'pattern':'sun','sun':{'playing':True}});ready();time.sleep(.08)
                engine.update({'sun':{'playing':False}});ready()
                paused=engine.snapshot()['config']['sun']['time'];time.sleep(.09)
                self.assertEqual(engine.snapshot()['config']['sun']['time'],paused)
                self.assertGreater(datetime.fromisoformat(paused),datetime.fromisoformat(c['sun']['time']))
                engine.update({'sun':{'playing':True}});ready();time.sleep(.07)
                self.assertGreater(datetime.fromisoformat(engine.config['sun']['time']),datetime.fromisoformat(paused))
                anchor=engine.playback_origin
                engine.update({'sun':{'minutes_per_second':60}});ready();time.sleep(.03)
                self.assertEqual(engine.playback_origin,anchor)
                engine.stop_animation();ready();deadline=time.monotonic()+2
                while engine.transition is not None and time.monotonic()<deadline:time.sleep(.01)
                self.assertFalse(engine.config['sun']['playing'])
                self.assertEqual(engine.config['sun']['time'],anchor[0])
                engine.play();ready();time.sleep(.02)
                engine.stop_animation();ready()
                engine.update({'sun':{'time':'2026-10-24T15:15:00+02:00','playing':False}});ready()
                self.assertIsNone(engine.transition)
                self.assertTrue(engine.config['sun']['time'].startswith('2026-10-24T15:15:00'))
            finally:engine.close();engine.worker.join(3)


if __name__=='__main__':unittest.main()
