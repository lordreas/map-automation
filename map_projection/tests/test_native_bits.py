import shutil,unittest
import numpy as np
from map_projection.native_bits import Bits
from map_projection.gpu_bitset import timeline

@unittest.skipUnless(shutil.which('cc') or shutil.which('gcc'),'Requires C compiler')
class NativeBitTests(unittest.TestCase):
 def test_parallel_and_serial_preserve_duplicate_parity(self):
  rng=np.random.default_rng(71);nwords=4096;cells=rng.integers(0,nwords*32,180000,dtype=np.uint32)
  initial=rng.integers(0,2**32,nwords,dtype=np.uint32);expected=initial.copy()
  np.bitwise_xor.at(expected,cells>>5,np.uint32(1)<<(cells&31))
  for threads in (1,4):
   words=initial.copy();native=Bits(nwords,threads);native.flip(words,cells)
   np.testing.assert_array_equal(words,expected)
   native.flip(words,cells);np.testing.assert_array_equal(words,initial)
 def test_timeline_empty_and_duplicate_chunk_boundaries(self):
  times,ends=timeline(np.empty(0,np.uint32));self.assertEqual(len(times),0);self.assertEqual(len(ends),0)
  data=np.repeat(np.array([10,20,30],np.uint32),[(1<<20)-1,5,2])
  times,ends=timeline(data);np.testing.assert_array_equal(times,[10,20,30]);np.testing.assert_array_equal(ends,[(1<<20)-1,(1<<20)+4,(1<<20)+6])
  with self.assertRaises(ValueError):timeline(np.array([10,9],np.uint32))
