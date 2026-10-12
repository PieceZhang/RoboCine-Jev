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

from .lidar_collision_map import cloud_xyz
from .lidar_collision_input import LidarCollisionInput
from .dynamic_lidar_map import DynamicLidarMap, integrate_dynamic_batch
from .shot_preview_worker import initialize_worker


def transform_parts(transform):
    t, q = transform.transform.translation, transform.transform.rotation
    return np.array([t.x, t.y, t.z]), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()


class DynamicLidarCollisionInput(LidarCollisionInput):
    def __init__(self, node, value):
        super().__init__(node, value)
        self.settings.update({name: float(value('dynamic_' + name)) for name in
            ('fusion_window_s', 'track_timeout_s', 'cluster_distance', 'velocity_window_s', 'maximum_speed')})
        self.map = DynamicLidarMap(**self.settings)
        self.track_history = []
        self.live_stamp = -np.inf

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
            updated = result['map']
            self.update_ms = result['update_ms']
            self.stamps.update(self.batch_stamps)
            self.walls.update({topic: time.monotonic() for topic in self.batch_stamps})
            self.map = updated
            self.track_history.append(dict(stamp=updated.stamp, tracks=updated.track_diagnostics()))
            if updated.stamp - self.live_stamp >= .5:
                self.live_stamp = updated.stamp
                target = self.node.metrics.folder / 'dynamic_live.json'
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(self.track_history[-1]) + '\n')
                temporary.replace(target)
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
                        sensor=topic, minimum_range=.16)
                    scans.append(scan)
                    self.last_scans[topic] = scan
                    self.batch_stamps[topic] = stamp
                    self.error = ''
                except (ValueError, TypeError) as error:
                    self.error = str(error)
                    self.rejected += 1
        if scans:
            self.future = self.pool.submit(integrate_dynamic_batch, self.settings, scans)

    def save(self, folder):
        super().save(folder)
        (folder / 'track_history.json').write_text(json.dumps(self.track_history) + '\n')
        (folder / 'dynamic_tracks.json').write_text(json.dumps(dict(
            tracks=self.map.snapshot()['tracks'], moving_observations=self.map.moving_observations,
            expired_tracks=self.map.expired_tracks, observation_stamp=self.map.stamp,
            source='timestamped LiDAR only'), indent=2) + '\n')
