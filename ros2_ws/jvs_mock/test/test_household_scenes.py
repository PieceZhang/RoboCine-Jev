"""Contracts for the graded, floor-standing household obstacle scenes."""
import json
from pathlib import Path
import unittest

import numpy as np

from jvs_mock.collision_route import CollisionRoute
from jvs_mock.jvs_kinematics import forward_kinematics
from jvs_mock.kc_svsdf import SweptCollision, obstacle_spheres
from jvs_mock.wbc_trajectory import CameraTrajectory


ROOT = Path(__file__).resolve().parents[1]


class HouseholdSceneContract(unittest.TestCase):
    def test_shapes_stand_on_ground_and_difficulty_increases(self):
        expected_counts = {'easy': 3, 'medium': 5, 'hard': 8}
        proxy_counts = []
        for level, count in expected_counts.items():
            path = ROOT/'scenes'/f'household_{level}.json'
            scene = json.loads(path.read_text())
            collision = SweptCollision(path)
            self.assertEqual(scene['difficulty'], level)
            self.assertEqual(len(scene['obstacles']), count)
            self.assertEqual({o['type'] for o in scene['obstacles']},
                             {'cylinder', 'cube', 'triangular_pyramid'})
            for obstacle in scene['obstacles']:
                height = (obstacle['height'] if obstacle['type'] == 'cylinder'
                          else obstacle['size'] if obstacle['type'] == 'cube'
                          else obstacle['size'][2])
                self.assertAlmostEqual(obstacle['center'][2]-height/2, 0.)
            proxy_counts.append(len(collision.centers))
        self.assertEqual(proxy_counts, sorted(proxy_counts))

    def test_proxy_spheres_contain_primitive_sample_points(self):
        rng = np.random.default_rng(7)
        for level in ('easy', 'medium', 'hard'):
            scene = json.loads((ROOT/'scenes'/f'household_{level}.json').read_text())
            for obstacle in scene['obstacles']:
                center = np.asarray(obstacle['center'])
                kind = obstacle['type']
                if kind == 'cylinder':
                    angle = rng.uniform(0, 2*np.pi, 200)
                    radius = obstacle['radius']*np.sqrt(rng.uniform(0, 1, 200))
                    points = center+np.c_[radius*np.cos(angle), radius*np.sin(angle),
                                         rng.uniform(-obstacle['height']/2,
                                                     obstacle['height']/2, 200)]
                elif kind == 'cube':
                    points = center+rng.uniform(-obstacle['size']/2,
                                                obstacle['size']/2, (200, 3))
                else:
                    sx,sy,sz = obstacle['size']
                    points = center+np.array([[-sx/2,-sy/2,-sz/2],
                                              [sx/2,-sy/2,-sz/2],
                                              [0,sy/2,-sz/2], [0,0,sz/2]])
                proxies = obstacle_spheres(obstacle)
                covered = np.any(np.column_stack([
                    np.linalg.norm(points-proxy_center, axis=1) <= proxy_radius+1e-12
                    for proxy_center,proxy_radius in proxies]), axis=1)
                self.assertTrue(np.all(covered), obstacle['name'])

    def test_moveout_guide_remains_open(self):
        camera_p,camera_r = forward_kinematics(np.zeros(6))
        trajectory = CameraTrajectory(ROOT/'trajectories'/'shot_moveout.json',
                                      camera_p, camera_r, np.eye(3))
        lengths = []
        for level in ('easy', 'medium', 'hard'):
            collision = SweptCollision(ROOT/'scenes'/f'household_{level}.json')
            route = CollisionRoute(collision, trajectory, np.zeros(9))
            points = np.array([route.sample(t)
                               for t in np.linspace(0, trajectory.duration, 2001)])
            centers = np.array([center for center,_ in route.circles])
            radii = np.array([radius for _,radius in route.circles])
            gap = np.min(np.linalg.norm(points[:,None]-centers[None,:], axis=2)-radii)
            self.assertGreaterEqual(gap, -1e-3, level)
            lengths.append(float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        self.assertLess(lengths[0], lengths[1])
        self.assertLess(lengths[1], lengths[2])


if __name__ == '__main__':
    unittest.main()
