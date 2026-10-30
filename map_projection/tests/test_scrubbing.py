import copy,tempfile,time,unittest
from pathlib import Path
import numpy as np
from map_projection.config import DEFAULT
from map_projection.engine import Engine
from map_projection.frames import SunFrame,SunScene
from map_projection.tests.test_projection import fixture


class ScrubTests(unittest.TestCase):
    def test_completed_time_frames_survive_newer_input_but_not_pattern_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);fixture(p/'mesh.stl');c=copy.deepcopy(DEFAULT)
            c['pixel'].update(x=32,y=32);c['output'].update(width=64,height=64);c['pattern']='black';c['projector'].update(y=12,z=30)
            engine=Engine(p/'mesh.stl',c,cache=p/'cache',gpu=True)
            original=engine.render
            scene=SunScene('test',np.zeros((2,2),np.float32),np.ones((2,2,3),np.float32),np.zeros((64,64,2),np.uint16),(10.,10.))
            def slow(c):
                if c['pattern']!='sun':return original(c)
                time.sleep(.045)
                return SunFrame(scene,np.array([0.,0.,1.]),0,0,c['sun']['time'])
            engine.render=slow
            try:
                start=time.monotonic();initial=engine.frame_id
                while time.monotonic()-start<.4:
                    second=int((time.monotonic()-start)*100)
                    engine.update({'pattern':'sun','sun':{'time':f'2026-10-27T16:20:{second:02d}+01:00'}},persist=False)
                    time.sleep(.007)
                self.assertGreaterEqual(engine.frame_id-initial,4)
                engine.update({'pattern':'black'},persist=False);time.sleep(.1)
                self.assertNotIsInstance(engine.frame,SunFrame);self.assertFalse(np.any(engine.frame))
            finally:engine.close();engine.worker.join(2)
