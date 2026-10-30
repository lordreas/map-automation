import json,shutil,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
from map_projection.horizon_cache import Horizons,Events,build_day
from map_projection.native_events import build_day as native_day
from map_projection.skyline_store import pack,elevation_levels

@unittest.skipUnless(shutil.which('cc') or shutil.which('gcc'),'Requires C compiler')
class NativeCompilerTests(unittest.TestCase):
 def source(self,root,values):
  p=Path(root)/'source';p.mkdir();np.save(p/'horizons.npy',values)
  (p/'manifest.json').write_text(json.dumps({'schema':1,'key':'native-test','complete':True,'angles':len(values),'shape':list(values.shape[1:])}))
  return p
 def compare(self,reference,actual,tolerance=50):
  np.testing.assert_array_equal(reference.initial,actual.initial)
  np.testing.assert_array_equal(reference.offsets,actual.offsets)
  np.testing.assert_allclose(reference.times,actual.times,rtol=0,atol=tolerance)
  times=np.load(actual.path/'sorted-times.npy');cells=np.load(actual.path/'sorted-cells.npy')
  np.testing.assert_array_equal(times,np.sort(actual.times))
  expected_cells=np.repeat(np.arange(len(actual.initial)),np.diff(actual.offsets))
  order=np.lexsort((actual.times,expected_cells));stored_order=np.lexsort((times,cells))
  np.testing.assert_array_equal(cells[stored_order],expected_cells[order])
  np.testing.assert_array_equal(times[stored_order],actual.times[order])
 def test_raw_and_compressed_match_scalar_and_each_other(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);values=elevation_levels()[np.random.default_rng(24).integers(1,70000,(24,3,7))]
   src=self.source(p,values);packed=p/'packed';pack(src,packed,start=60,end=300,tile_shape=(2,3))
   reference=build_day(Horizons(src),p/'reference','2026-10-25',45.95,10.65,step=600)
   raw=native_day(src,p/'raw','2026-10-25',45.95,10.65,step=600,threads=1)
   compressed=native_day(packed,p/'compressed','2026-10-25',45.95,10.65,step=600,threads=3)
   self.compare(reference,raw);self.compare(raw,compressed,tolerance=0)
   self.assertEqual(compressed.meta['compiler'],'native-c-v1')
   with self.assertRaises(ValueError):native_day(packed,p/'compressed','2026-10-25',45.95,10.65)
 def test_general_step_year_boundary_and_southern_longitude(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);src=self.source(p,np.random.default_rng(22).uniform(1,35,(24,2,3)).astype(np.float32))
   reference=build_day(Horizons(src),p/'reference','2024-12-31',-35,-150,step=61.25)
   result=native_day(src,p/'native','2024-12-31',-35,-150,step=61.25,threads=2)
   self.compare(reference,result)
 def test_empty_events_have_valid_arrays(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);src=self.source(p,np.full((24,2,3),89,np.float32))
   result=native_day(src,p/'out','2026-10-25',45.95,10.65,step=600,threads=2)
   self.assertEqual(result.meta['events'],0);self.assertEqual(len(result.times),0)
   np.testing.assert_array_equal(result.visibility(40000),np.zeros((2,3),bool))
 def test_compressed_raw_bits_fallback_and_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);src=self.source(p,np.random.default_rng(99).uniform(1,25,(24,2,3)).astype(np.float32));packed=p/'packed'
   pack(src,packed,start=60,end=300,tile_shape=(2,3))
   reference=build_day(Horizons(src),p/'reference','2026-10-25',45.95,10.65,step=600)
   result=native_day(packed,p/'out','2026-10-25',45.95,10.65,step=600,threads=1)
   self.compare(reference,result)
   archive=packed/'tiles.zst';archive.write_bytes(archive.read_bytes()[:-5])
   with self.assertRaises(subprocess.CalledProcessError):native_day(packed,p/'broken','2026-10-25',45.95,10.65,step=600,threads=1)
   self.assertFalse(json.loads((p/'broken/manifest.json').read_text())['complete'])
