"""Input and continuous-weight contract checks; acceptance remains Isaac-only."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jvs_mock.wbc_trajectory import CameraTrajectory
from jvs_mock.shot_predictive_qp import ShotPredictiveQP


class ShotContract(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1]/'trajectories/shot_combo1.json'
        self.traj = CameraTrajectory(self.path,np.zeros(3),np.eye(3),np.eye(3))

    def test_junction_continuity(self):
        for t in [shot['start_s'] for shot in self.traj.shots[1:]]:
            left,right = self.traj.shot_weights(t-1e-6),self.traj.shot_weights(t+1e-6)
            self.assertLess(max(abs(left[k]-right[k]) for k in left),2e-6)
            for x in (t-1,t,t+1):
                w=self.traj.shot_weights(x)
                self.assertAlmostEqual(sum(w.values()),1)
                self.assertGreaterEqual(min(w.values()),0)
            before,after=self.traj.sample(t-1e-6),self.traj.sample(t+1e-6)
            np.testing.assert_allclose(before[0],after[0],atol=1e-6)
            np.testing.assert_allclose(before[1],after[1],atol=1e-6)
            np.testing.assert_allclose(before[2],after[2],atol=1e-6)

    def test_context_and_end(self):
        pan=self.traj.shots[1]
        c=self.traj.shot_context((pan['start_s']+pan['end_s'])/2)
        self.assertEqual((c['shot_type'],c['shot_phase'],c['next_shot']),('pan',.5,'moveout'))
        self.assertEqual(self.traj.shot_context(100.)['shot_phase'],1.)
        np.testing.assert_array_equal(self.traj.sample(100.)[2],np.zeros(6))

    def test_live_validation(self):
        c=ShotPredictiveQP();c.set_trajectory(self.traj)
        for v in ({},dict(shot_type='unknown',shot_phase=.5,next_shot='pan',remaining_s=1.),
                  dict(shot_type='pan',shot_phase=float('nan'),next_shot='pan',remaining_s=1.)):
            with self.assertRaises(ValueError):c.set_shot_input(v)
        c.set_shot_input(dict(shot_type='pan',shot_phase=.8,next_shot='moveout',remaining_s=2.,received_s=10.))
        self.assertEqual(c.context(11.)['shot_type'],'pan')
        self.assertEqual(c.context(13.)['shot_type'],'moveout')
        self.assertAlmostEqual(c.weights(12.)['moveout'],.5)


if __name__=='__main__':unittest.main()
