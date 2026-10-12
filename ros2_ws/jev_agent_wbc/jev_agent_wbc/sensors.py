"""Runtime sensing. Geometry returns are tracked without obstacle IDs or future paths."""
import numpy as np
from scipy.spatial.transform import Rotation

from .dynamic_lidar_map import DynamicLidarMap


def directions():
    # Dual sparse scanners: sufficient near-field angular coverage without RTX rendering.
    azimuth = np.linspace(-np.pi, np.pi, 240, endpoint=False)
    elevation = np.r_[np.linspace(-.45, .6, 15), np.linspace(.7, 1.48, 11)]
    a, e = np.meshgrid(azimuth, elevation)
    return np.column_stack([(np.cos(e)*np.cos(a)).ravel(), (np.cos(e)*np.sin(a)).ravel(), np.sin(e).ravel()])


RAYS = directions()


def sphere_raycast(origin, rays, spheres, maximum=4.5):
    distances = np.full(len(rays), maximum)
    hit = np.zeros(len(rays), bool)
    for sphere in spheres:
        relative = np.asarray(origin)-sphere['center']
        along = rays@relative
        disc = along**2-(relative@relative-sphere['radius']**2)
        valid = disc >= 0
        distance = -along-np.sqrt(np.maximum(disc, 0.))
        valid &= (distance > .05) & (distance < distances)
        distances[valid], hit[valid] = distance[valid], True
    return origin+rays[hit]*distances[hit, None]


class RuntimeLidar:
    def __init__(self):
        self.mapping = DynamicLidarMap(voxel_size=.035, map_radius=4.5, retention_s=.5, minimum_clearance=.025,
                                       minimum_height=.03, maximum_height=1.6,
                                       velocity_window_s=.8, fusion_window_s=.20)
        self.revision = 0
        self.last = -float('inf')
        self.subject_observation = None

    def integrate(self, state, base_z, t, world_points_by_sensor, origins):
        scans = [dict(points=points-origin, rotation=np.eye(3), translation=origin,
                      state=np.asarray(state).copy(), base_z=base_z,
                      base_rotation=Rotation.from_euler('z', state[2]).as_matrix(), stamp=t+1., sensor=name)
                 for name, points, origin in zip(('rear', 'front'), world_points_by_sensor, origins)]
        self.mapping.integrate_scans(scans)
        snapshot = self.mapping.snapshot()
        # Mapping requires positive initial timestamps; convert to the shared shot clock.
        snapshot['observation_stamp'] -= 1.
        self.last, self.revision = t, self.revision+1
        return snapshot

    @staticmethod
    def origins(state, base_z):
        rotation = Rotation.from_euler('z', state[2]).as_matrix()
        base = np.r_[state[:2], base_z]
        return [base+rotation@np.array(offset) for offset in [(-.085, 0., .235), (.16, 0., .34)]]

    def synthetic_scan(self, scene, state, base_z, t):
        """CPU harness sensor simulation; only ray hit points cross into the tracker."""
        origins = self.origins(state, base_z)
        spheres = scene.obstacles(t)
        points = [sphere_raycast(origin, RAYS, spheres) for origin in origins]
        # Sparse distant wall supports the map when the mover is temporarily unseen.
        wall = np.array([[3.5, y, z] for y in np.linspace(-1.5, 1.5, 41) for z in np.arange(.1, .8, .075)])
        points = [np.vstack([p, wall]) for p in points]
        return self.integrate(state, base_z, t, points, origins)


def estimate_subject_from_points(points, initial_center, previous=None, stamp=0., symmetric=False):
    """Runtime 3D centroid/PCA orientation; not a simulator truth angle substitute.

    An initial subject association is provided by the runtime tracker. Symmetric
    objects cannot yield reliable yaw from a single scan and are explicitly flagged.
    """
    points = np.asarray(points, float)
    selected = points[np.linalg.norm(points-np.asarray(initial_center), axis=1) < .45]
    if len(selected) < 8:
        return dict(valid=False, reason='insufficient_subject_lidar_returns', stamp_s=stamp)
    center = .5*(selected.max(axis=0)+selected.min(axis=0))
    _, _, axes = np.linalg.svd(selected-center, full_matrices=False)
    forward = axes[0].copy()
    forward[2] = 0.
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([1., 0., 0.])
    forward /= np.linalg.norm(forward)
    if previous and previous.get('valid') and forward@np.asarray(previous['rotation'])[:, 0] < 0:
        forward *= -1
    left = np.cross([0., 0., 1.], forward)
    rotation = np.column_stack([forward, left, [0., 0., 1.]])
    return dict(valid=True, position=center.tolist(), rotation=rotation.tolist(), stamp_s=stamp,
                orientation_valid=not symmetric, source='lidar_centroid_pca', returns=len(selected))
