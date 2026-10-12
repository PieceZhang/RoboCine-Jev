"""ROS adapter for timestamped dual LiDAR collision observations."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import time

import numpy as np
from rclpy.duration import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener, TransformException

from .lidar_collision_map import LidarCollisionMap, cloud_xyz, integrate_batch
from .shot_preview_worker import initialize_worker


def transform_parts(transform):
    t, q = transform.transform.translation, transform.transform.rotation
    return np.array([t.x, t.y, t.z]), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()


class LidarCollisionInput:
    def __init__(self, node, value):
        self.node = node
        self.settings = {name: float(value('lidar_' + name)) for name in
            ('voxel_size', 'map_radius', 'minimum_height', 'maximum_height',
             'minimum_clearance', 'retention_s')}
        self.map = LidarCollisionMap(**self.settings)
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=initialize_worker)
        self.future = None
        self.batch_stamps = {}
        self.timeout = float(value('lidar_timeout_s'))
        self.startup_timeout = float(value('lidar_startup_timeout_s'))
        if min(self.timeout, self.startup_timeout) <= 0:
            raise ValueError('LiDAR timeouts must be positive')
        self.topics = list(value('lidar_topics'))
        if not self.topics or len(set(self.topics)) != len(self.topics):
            raise ValueError('lidar_topics must be nonempty and unique')
        self.buffer = Buffer(cache_time=Duration(seconds=10.))
        self.listener = TransformListener(self.buffer, node)
        self.pending = {topic: deque(maxlen=12) for topic in self.topics}
        self.stamps = {}
        self.first_scan = {}
        self.walls = {}
        self.joints = deque(maxlen=500)
        self.started_wall = time.monotonic()
        self.error = ''
        self.failed = False
        self.rejected = 0
        self.update_ms = 0.
        self.last_scans = {}
        self.received = {}
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.subscriptions = [node.create_subscription(PointCloud2, topic,
            lambda message, topic=topic: self.on_cloud(message, topic),
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=self.callback_group) for topic in self.topics]
        self.timer = node.create_timer(.05, self.update, callback_group=self.callback_group)

    def on_cloud(self, message, topic):
        self.received[topic] = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        self.pending[topic].append(message)

    def update(self):
        if not self.joints or self.failed:
            return
        if self.future is not None:
            if not self.future.done():
                return
            try:
                result = self.future.result()
            except Exception as error:
                self.error = f'LiDAR worker failed: {error}'
                self.failed = True
                self.future = None
                return
            self.future = None
            updated = LidarCollisionMap(**self.settings)
            for name in ('cells', 'scans', 'revision', 'bad_scan'):
                setattr(updated, name, result[name])
            self.update_ms = result['update_ms']
            self.stamps.update(self.batch_stamps)
            self.walls.update({topic: time.monotonic() for topic in self.batch_stamps})
            self.map = updated
        scans = []
        self.batch_stamps = {}
        for topic, queue in self.pending.items():
            for _ in range(len(queue)):
                if not queue:
                    continue
                # Keep a future scan queued until its TF arrives; do not use latest TF.
                message = queue[0]
                stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
                first = self.first_scan.setdefault(topic, stamp)
                # Discard full-scan buffers which can straddle robot reset/render
                # warmup. All accepted observations belong to the settled pose.
                if stamp - first < 1.:
                    queue.popleft()
                    continue
                if stamp <= self.stamps.get(topic, -np.inf):
                    queue.popleft()
                    continue
                try:
                    when = Time.from_msg(message.header.stamp)
                    sensor = self.buffer.lookup_transform('odom', message.header.frame_id, when)
                    base = self.buffer.lookup_transform('odom', 'base_link', when)
                except TransformException as error:
                    self.error = str(error)
                    break
                joint_stamp, joints = min(tuple(self.joints), key=lambda item: abs(item[0] - stamp))
                if abs(joint_stamp - stamp) > .1:
                    queue.popleft()
                    self.rejected += 1
                    continue
                queue.popleft()
                try:
                    translation, rotation = transform_parts(sensor)
                    position, base_rotation = transform_parts(base)
                    yaw = np.arctan2(base_rotation[1, 0], base_rotation[0, 0])
                    state = np.r_[position[:2], yaw, joints]
                    points = cloud_xyz(message)
                    if not len(points):
                        self.rejected += 1
                        continue
                    scan = dict(points=points, rotation=rotation, translation=translation,
                        state=state, base_z=position[2], base_rotation=base_rotation, stamp=stamp,
                        sensor=topic, minimum_range=1. if message.header.frame_id == 'front_lidar_link' else .16)
                    scans.append(scan)
                    self.last_scans[topic] = scan
                    self.batch_stamps[topic] = stamp
                    self.error = ''
                except (ValueError, TypeError) as error:
                    self.error = str(error)
                    self.rejected += 1
        if scans:
            self.future = self.pool.submit(integrate_batch, self.settings, scans)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def ready(self, now):
        return (not self.failed and len(self.map.cells) >= 100 and all(self.map.scans.get(topic, 0) >= 2
            and now - self.stamps.get(topic, -np.inf) <= self.timeout for topic in self.topics))

    def stale(self, now):
        return self.failed or any(now - self.stamps.get(topic, -np.inf) > self.timeout
                   or time.monotonic() - self.walls.get(topic, -np.inf) > 5.
                   for topic in self.topics)

    def save(self, folder):
        snapshot = self.map.snapshot()
        if self.map.bad_scan is not None:
            np.savez_compressed(folder / 'lidar_near_scan.npz', **self.map.bad_scan)
        for index, topic in enumerate(self.topics):
            if topic in self.last_scans:
                np.savez_compressed(folder / f'lidar_last_scan_{index}.npz', **self.last_scans[topic])
        np.savez_compressed(folder / 'lidar_map.npz', **snapshot)
        (folder / 'lidar_map.json').write_text(json.dumps(dict(
            scans=self.map.scans, stamps=self.stamps, voxels=len(self.map.cells),
            received=self.received, worker_inflight=self.future is not None,
            revision=self.map.revision, rejected=self.rejected, error=self.error,
            frame='odom', source='live ROS PointCloud2 only'), indent=2) + '\n')
