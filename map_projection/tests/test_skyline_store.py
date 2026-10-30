import json,tempfile,unittest
from pathlib import Path
import numpy as np
from map_projection.skyline_store import Zstd,encode,decode,CompressedHorizons,pack,elevation_levels,encode_predicted,decode_predicted
from map_projection.horizon_cache import Horizons,open_horizons,build_day
class SkylineStorageTests(unittest.TestCase):
 def test_predictor_recovers_every_elevation_code_exactly(self):
  a=elevation_levels().reshape(256,16,64)
  encoded,flag=encode_predicted(a)
  self.assertEqual(flag,1)
  restored=decode_predicted(encoded,a.shape,flag)
  np.testing.assert_array_equal(a.view(np.uint32),restored.view(np.uint32))
 def test_predictor_raw_fallback_preserves_arbitrary_float_bits(self):
  a=np.random.default_rng(11).integers(0,2**32,(31,7,13),np.uint32).view(np.float32)
  encoded,flag=encode_predicted(a)
  self.assertEqual(flag,0)
  restored=decode_predicted(encoded,a.shape,flag)
  np.testing.assert_array_equal(a.view(np.uint32),restored.view(np.uint32))
 def test_lossless_float_patterns_and_constants(self):
  codec=Zstd();rng=np.random.default_rng(51)
  for a in [np.zeros((10,4,9),np.float32),rng.normal(size=(10,4,9)).astype(np.float32),rng.integers(0,2**32,(10,4,9),np.uint32).view(np.float32)]:
   data=encode(a);restored=decode(codec.decompress(codec.compress(data),len(data)),a.shape)
   np.testing.assert_array_equal(a.view(np.uint32),restored.view(np.uint32))
 def test_sector_lookup_edge_tiles_and_bounded_lru(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/'source';source.mkdir();target=Path(d)/'packed'
   a=np.random.default_rng(18).uniform(0,70,(24,7,11)).astype(np.float32);np.save(source/'horizons.npy',a)
   (source/'manifest.json').write_text(json.dumps({'complete':True,'schema':1,'key':'geometry','shape':[7,11]}))
   pack(source,target,start=60,end=300,tile_shape=(3,4));original=Horizons(source);compressed=open_horizons(target,budget=1000)
   try:
    for angle in [60,300,67.125,119.55,230]:
     for indices in [None,np.array([76,2,9,30,0,76])]:
      np.testing.assert_allclose(compressed.at(angle,indices),original.at(angle,indices),atol=1e-5,rtol=0)
      self.assertLessEqual(compressed.bytes,1000)
    indices=np.array([76,2,9,30,0,76]);angles=np.array([60,300,67.125,119.55,230,90])
    compressed.at(100,indices);saved=compressed.query_indices.copy()
    expected=np.array([original.at(angle,np.array([ix]))[0] for angle,ix in zip(angles,indices)])
    np.testing.assert_allclose(compressed.at_many(angles,indices),expected,atol=1e-5,rtol=0)
    np.testing.assert_array_equal(compressed.query_indices,saved)
    with self.assertRaises(ValueError):compressed.at(30)
    self.assertEqual(compressed.meta['key'],'geometry')
   finally:compressed.close()
 def test_truncated_tile_is_rejected(self):
  codec=Zstd();encoded=codec.compress(b'abcd'*100)
  with self.assertRaises(ValueError):codec.decompress(encoded[:-5],400)
 def test_day_compiler_accepts_compressed_bank(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'source';source.mkdir();packed=root/'packed'
   a=np.random.default_rng(82).uniform(1,25,(24,2,3)).astype(np.float32)
   np.save(source/'horizons.npy',a)
   (source/'manifest.json').write_text(json.dumps({'schema':1,'complete':True,'key':'geometry','shape':[2,3]}))
   pack(source,packed,start=60,end=300,tile_shape=(1,2))
   raw=Horizons(source);compressed=open_horizons(packed)
   try:
    x=build_day(raw,root/'raw-events','2026-10-25',45.95,10.65,step=600,precision=.05)
    y=build_day(compressed,root/'packed-events','2026-10-25',45.95,10.65,step=600,precision=.05)
    v=build_day(compressed,root/'vector-events','2026-10-25',45.95,10.65,step=600,precision=.05,vectorized=True)
    b=build_day(compressed,root/'bounded-events','2026-10-25',45.95,10.65,step=600,precision=.05,vectorized=True,bounded=True)
    np.testing.assert_array_equal(x.initial,y.initial)
    np.testing.assert_array_equal(x.offsets,y.offsets)
    np.testing.assert_allclose(x.times,y.times,rtol=0,atol=50)
    self.assertEqual(y.meta['angular_step_degrees'],15)
    np.testing.assert_array_equal(y.offsets,v.offsets)
    np.testing.assert_allclose(y.times,v.times,rtol=0,atol=50)
    np.testing.assert_array_equal(v.offsets,b.offsets)
    np.testing.assert_array_equal(v.times,b.times)
   finally:compressed.close()
