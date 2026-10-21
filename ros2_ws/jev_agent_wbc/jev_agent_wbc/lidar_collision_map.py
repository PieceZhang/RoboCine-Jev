"""Measured LiDAR surface voxels in odom; no scene geometry is consulted.

The map retains occluded static surfaces for a bounded lifetime. Missing returns
are not evidence of free space. It is not a complete reconstruction or SLAM.
"""
import numpy as np
from .kc_svsdf import CollisionGeometry


_worker_map = None


def integrate_batch(settings, scans):
    """Persistent process worker: dense scans never block the control callback."""
    global _worker_map
    if _worker_map is None:
        _worker_map = LidarCollisionMap(**settings)
    import time
    started = time.monotonic()
    for scan in scans:
        _worker_map.integrate(**scan)
    return dict(cells=_worker_map.cells, scans=_worker_map.scans, bad_scan=_worker_map.bad_scan,
                revision=_worker_map.revision, update_ms=(time.monotonic() - started) * 1000.)


def cloud_xyz(message):
    """Decode XYZ while respecting PointCloud2 row padding and byte order."""
    fields = {field.name: field for field in message.fields}
    names = ('x', 'y', 'z')
    if any(name not in fields or fields[name].datatype not in (7, 8)
           or fields[name].count != 1 for name in names):
        raise ValueError('PointCloud2 requires scalar float32/float64 XYZ')
    endian = '>' if message.is_bigendian else '<'
    dtype = np.dtype(dict(names=names,
        formats=[endian + ('f4' if fields[name].datatype == 7 else 'f8') for name in names],
        offsets=[fields[name].offset for name in names], itemsize=message.point_step))
    cloud = np.ndarray((message.height, message.width), dtype=dtype,
        buffer=message.data, strides=(message.row_step, message.point_step))
    points = np.column_stack([cloud[name].ravel() for name in names])
    return points[np.isfinite(points).all(axis=1)]


def voxel_keys(points, size):
    """Integer voxel IDs and one measured representative, without row sorting."""
    keys = np.floor(points / size).astype(np.int64)
    if not len(keys):
        return keys, np.empty(0, dtype=int)
    low = keys.min(axis=0)
    shape = keys.max(axis=0) - low + 1
    offsets = keys - low
    encoded = (offsets[:, 0] * shape[1] + offsets[:, 1]) * shape[2] + offsets[:, 2]
    _, indices = np.unique(encoded, return_index=True)
    return keys[indices], indices


class LidarCollisionMap:
    def __init__(self, voxel_size=.08, map_radius=3., minimum_height=.03,
                 maximum_height=1.60, minimum_clearance=.025, retention_s=30., confirmations=2,
                 geometry=None):
        settings = np.array([voxel_size, map_radius, minimum_height, maximum_height,
                             minimum_clearance, retention_s])
        if (not np.isfinite(settings).all() or min(voxel_size, map_radius,
                minimum_clearance, retention_s) <= 0 or minimum_height >= maximum_height):
            raise ValueError('invalid LiDAR map parameters')
        self.voxel_size, self.map_radius = voxel_size, map_radius
        self.minimum_height, self.maximum_height = minimum_height, maximum_height
        self.minimum_clearance, self.retention_s = minimum_clearance, retention_s
        self.cells = {}
        self.observations = {}
        self.confirmations = int(confirmations)
        if self.confirmations < 1:
            raise ValueError('confirmations must be positive')
        self.geometry = CollisionGeometry() if geometry is None else geometry
        self.sensor_housings = (([-.085, 0., .235], .065), ([.190, 0., .274], .065)) if geometry is None else ()
        self.revision = 0
        self.scans = {}
        self.bad_scan = None

    def integrate(self, points, rotation, translation, state, base_z, base_rotation, stamp, sensor,
                  minimum_range=.1, maximum_range=30., self_filter_margin_m=.06):
        if not np.isfinite(self_filter_margin_m) or self_filter_margin_m<0:
            raise ValueError('self filter margin must be finite and nonnegative')
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        # Zero/range-invalid returns must never become occupied voxels at the sensor.
        distance = np.linalg.norm(points, axis=1)
        keep = np.isfinite(points).all(axis=1) & (distance >= minimum_range) & (distance <= maximum_range)
        world = points[keep] @ np.asarray(rotation).T + translation
        keep = ((world[:, 2] >= self.minimum_height) & (world[:, 2] <= self.maximum_height)
                & (np.linalg.norm(world[:, :2] - state[:2], axis=1) <= self.map_radius))
        world = world[keep]
        # Reduce dense scans before the articulated self mask, keeping real returns.
        _, indices = voxel_keys(world, self.voxel_size / 3)
        world = world[indices]
        if len(world):
            distances = self.geometry.distances(state, world, base_z, base_rotation)
            # The rear profile specifies 2 cm range accuracy. Keep a 6 cm
            # self-return band before voxelization; otherwise rare noisy body
            # returns produce inflated occupied cells intersecting the robot.
            keep = np.min(distances, axis=1) > self_filter_margin_m
            # Sensor housings are visuals absent from CollisionGeometry.
            local = (world - np.r_[state[:2], base_z]) @ base_rotation
            for center, radius in self.sensor_housings:
                keep &= np.linalg.norm(local - center, axis=1) > radius
            world = world[keep]
        # Preserve filtered measured coordinates for dynamic shape/velocity
        # estimation. Voxel centers are collision envelopes, not measurements.
        self.filtered_points = world.copy()
        keys, _ = voxel_keys(world, self.voxel_size)
        if len(world):
            centers = (keys + .5) * self.voxel_size
            ds = self.geometry.distances(state, centers, base_z, base_rotation).min(axis=1)
            bad = keys[ds < np.sqrt(3) * self.voxel_size / 2 + self.minimum_clearance]
            if len(bad):
                self.bad_scan = dict(points=world, state=state, base_z=base_z,
                    base_rotation=base_rotation, translation=translation, rotation=rotation,
                    stamp=stamp, sensor=sensor, bad_keys=bad)
        for key in map(tuple, keys):
            count, previous = self.observations.get(key, (0, -np.inf))
            if stamp - previous >= .03:
                count = min(self.confirmations, count + 1) if stamp - previous < 1. else 1
                self.observations[key] = (count, stamp)
            if count >= self.confirmations or key in self.cells:
                self.cells[key] = max(stamp, self.cells.get(key, -np.inf))
        self.observations = {key: observation for key, observation in self.observations.items()
                             if stamp - observation[1] < 1.}
        if self.cells:
            cells = np.array(list(self.cells), dtype=np.int32)
            seen = np.fromiter(self.cells.values(), dtype=float)
            keep = ((stamp - seen <= self.retention_s) &
                (np.linalg.norm((cells[:, :2] + .5) * self.voxel_size - state[:2], axis=1)
                 <= self.map_radius))
            self.cells = dict(zip(map(tuple, cells[keep]), seen[keep]))
        self.scans[sensor] = self.scans.get(sensor, 0) + 1
        self.revision += 1
        return len(world)

    def snapshot(self):
        centers = (np.array(sorted(self.cells), dtype=float).reshape(-1, 3) + .5) * self.voxel_size
        return dict(format='jvs_lidar_surfels/v1', surfel_centers=centers,
                    surfel_radius=np.sqrt(3.) * self.voxel_size / 2,
                    minimum_clearance_m=self.minimum_clearance, revision=self.revision)
