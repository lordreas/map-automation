import io,json
from pathlib import Path
import tempfile,threading,time
import unittest
from unittest.mock import patch
from map_projection.day_cache import utc_days,prune,complete,DayCompiler

FILES=('initial.npy','offsets.npy','times.npy','sorted-times.npy','sorted-cells.npy')
def day(path,n=100):
    path.mkdir(parents=True)
    (path/'manifest.json').write_text(json.dumps({'complete':True}))
    for name in FILES:(path/name).write_bytes(b'x'*n)


class DayCacheTests(unittest.TestCase):
    def test_dst_and_night_only_utc_boundary(self):
        self.assertEqual(utc_days('2026-10-27','Europe/Zurich'),('2026-10-26','2026-10-27'))
        self.assertEqual(utc_days('2026-10-27','Europe/Zurich',45.95,10.65),('2026-10-27',))
        self.assertEqual(utc_days('2026-03-29','Europe/Zurich',45.95,10.65),('2026-03-29',))
        self.assertEqual(utc_days('2026-10-27','Asia/Tokyo',45.95,10.65),('2026-10-26','2026-10-27'))

    def test_retention_keeps_active_day_and_does_not_touch_horizons(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'events';root.mkdir();sky=Path(folder)/'horizons';sky.mkdir();(sky/'tiles.zst').write_bytes(b'safe')
            for i,d in enumerate(['2026-01-01','2026-01-02','2026-01-03']):
                p=root/d;day(p);import os;os.utime(p,(i+1,i+1))
            (root/'active.json').write_text(json.dumps({'day':'2026-01-01'}))
            result=prune(root,1100)
            self.assertEqual(result['removed'],['2026-01-02']);self.assertLessEqual(result['bytes'],1100)
            self.assertTrue(complete(root/'2026-01-01'));self.assertTrue(complete(root/'2026-01-03'))
            self.assertEqual((sky/'tiles.zst').read_bytes(),b'safe')

    def test_single_compiler_and_atomic_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            compiler=DayCompiler(Path(folder)/'horizon',Path(folder)/'events');calls=[]
            class Process:
                pid=0
                def __init__(self,command,**kwargs):
                    calls.append(command);out=Path(command[command.index('--output')+1])/'2026-01-04'
                    self.stdout=io.StringIO('native events tiles 1/2\nnative events tiles 2/2\n');self.out=out
                def wait(self,**kwargs):
                    if not self.out.exists():time.sleep(.03);day(self.out)
                    return 0
                def poll(self):return 0
            errors=[]
            def run():
                try:compiler.ensure('2026-01-04')
                except Exception as e:errors.append(e)
            with patch('map_projection.native_events.build'),patch('map_projection.day_cache.subprocess.Popen',Process),patch('map_projection.day_cache.os.sync'),patch('map_projection.day_cache.os.setpriority'):
                threads=[threading.Thread(target=run) for _ in range(2)]
                for t in threads:t.start()
                for t in threads:t.join(3)
            self.assertEqual(errors,[]);self.assertEqual(len(calls),1);self.assertTrue(complete(compiler.root/'2026-01-04'))
            self.assertEqual(compiler.status()['state'],'ready');self.assertEqual(compiler.status()['progress'],1)
            self.assertFalse(list(compiler.root.glob('.building-*')))

    def test_failed_compilation_is_never_published(self):
        with tempfile.TemporaryDirectory() as folder:
            compiler=DayCompiler(Path(folder)/'horizon',Path(folder)/'events')
            class Failed:
                pid=0
                stdout=io.StringIO('failure\n')
                def wait(self,**kwargs):return 1
                def poll(self):return 1
            with patch('map_projection.native_events.build'),patch('map_projection.day_cache.subprocess.Popen',return_value=Failed()),patch('map_projection.day_cache.os.setpriority'):
                with self.assertRaises(RuntimeError):compiler.ensure('2026-01-04')
            self.assertFalse(complete(compiler.root/'2026-01-04'));self.assertEqual(compiler.status()['state'],'error')

    def test_requested_date_keeps_previous_day_until_ready(self):
        import copy
        from types import SimpleNamespace
        from map_projection.config import DEFAULT
        from map_projection.engine import Engine
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);day(root/'2026-10-27')
            e=Engine.__new__(Engine);requests=[]
            e.cache_compiler=SimpleNamespace(root=root)
            e.cache_manager=SimpleNamespace(request=lambda dates:requests.append(dates))
            e.daily_events=SimpleNamespace(path=root/'2026-10-27',meta={'start':'2026-10-27T00:00:00+00:00'})
            e.cache_pending=None
            c=copy.deepcopy(DEFAULT);c['sun'].update(backend='events',renderer='live',terrain='dem',time='2026-10-29T16:20:00+01:00')
            rendered=e.cached_time(c)
            self.assertEqual(c['sun']['time'],'2026-10-29T16:20:00+01:00')
            self.assertEqual(rendered['sun']['time'],'2026-10-27T16:20:00+01:00')
            self.assertEqual(e.cache_pending,'2026-10-29');self.assertEqual(requests,[('2026-10-29',)])
            day(root/'2026-10-29');rendered=e.cached_time(c)
            self.assertIs(rendered,c);self.assertIsNone(e.cache_pending)
