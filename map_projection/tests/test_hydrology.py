import unittest
import numpy as np
from map_projection.hydrology import network_phases,width_weights


class HydrologyTests(unittest.TestCase):
 def test_every_tributary_meets_the_receiving_front_at_the_same_instant(self):
  ids=[10,11,12,20,21,30];down=[20,20,21,30,30,0];length=[8,2,1,3,1,2]
  phase=network_phases(ids,down,length)
  for i,parent in enumerate(down):
   if parent in ids:self.assertAlmostEqual(float(phase[i,1]),float(phase[ids.index(parent),0]),places=6)
  self.assertEqual(phase[0,0],0);self.assertEqual(phase[-1,1],1)
  self.assertGreater(phase[1,0],phase[0,0]);self.assertTrue(np.all(phase[:,1]>phase[:,0]))

 def test_disconnected_networks_each_fill_the_animation_and_cycles_are_rejected(self):
  p=network_phases([1,2,3],[2,0,0],[1,2,8]);self.assertEqual(p[0,0],0);self.assertEqual(p[1,1],1);self.assertEqual(p[2,0],0);self.assertEqual(p[2,1],1)
  with self.assertRaises(ValueError):network_phases([1,2],[2,1],[1,1])

 def test_discharge_weights_are_finite_monotonic_and_bounded(self):
  w=width_weights(np.array([0,.1,1,10,100,1000,np.nan]))
  self.assertTrue(np.all(np.isfinite(w)));self.assertTrue(np.all(np.diff(w[:6])>=0));self.assertTrue(np.all((w>=0)&(w<=1)))
