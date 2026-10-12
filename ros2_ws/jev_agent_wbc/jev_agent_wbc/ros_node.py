"""ROS 2 adapter: task file + runtime state/subject/LiDAR; one command publisher."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.clock import Clock, ClockType
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from .dynamic_lidar_input import DynamicLidarCollisionInput
from .reference import OriginalReference
from .engine import CinematicEngine
from .contracts import native, validate_task
from .isaac_bridge import ARM_JOINT_NAMES, WHEEL_JOINT_NAMES


class CinematicNode(Node):
    def __init__(self):
        super().__init__('cinematic_wbc')
        defaults = dict(task_file='', output_directory='/tmp/jev_cinematic_ros',
            api_url='http://127.0.0.1:8000/v1/systemone', mode='jev', control_rate_hz=50., allocation_preview=False,
            subject_pose_topic='/cinematic/subject_pose', state_timeout_s=.2,
            lidar_topics=['/lidar/points', '/lidar/front/points'], lidar_voxel_size=.04,
            lidar_map_radius=4.5, lidar_minimum_height=.03, lidar_maximum_height=1.6,
            lidar_minimum_clearance=.025, lidar_retention_s=.5, lidar_timeout_s=.6,
            lidar_startup_timeout_s=60., dynamic_fusion_window_s=.2,
            dynamic_track_timeout_s=2.5, dynamic_cluster_distance=.16,
            dynamic_velocity_window_s=.8, dynamic_maximum_speed=1.)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        value = lambda key: self.get_parameter(key).value
        self.task = validate_task(json.loads(Path(value('task_file')).read_text()))
        self.output = Path(value('output_directory'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.metrics = SimpleNamespace(folder=self.output)  # LiDAR audit destination
        self.period = 1./float(value('control_rate_hz'))
        if not .005 <= self.period <= .05:
            raise ValueError('control rate must be in [20, 200] Hz')
        self.timeout = float(value('state_timeout_s'))
        self.api_url, self.mode = value('api_url'), value('mode')
        self.allocation_preview = value('allocation_preview')
        self.lidar = DynamicLidarCollisionInput(self, value)
        self.subject = self.state = self.q_command = self.engine = None
        self.base_z, self.base_rotation = 0., np.eye(3)
        self.joints = self.odom = None
        self.joint_wall = self.odom_wall = time.monotonic()
        self.start_s = self.previous_s = None
        self.completed, self.fault = False, ''
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.feedback_group = MutuallyExclusiveCallbackGroup()
        self.publisher = self.create_publisher(JointState, '/isaac_joint_commands', 1)
        self.status = self.create_publisher(String, '/cinematic/diagnostics', 10)
        self.scene_start = self.create_publisher(String, '/jvs/dynamic_scene/start', 1)
        self.create_subscription(JointState, '/joint_states', self.on_joints, qos_profile_sensor_data, callback_group=self.feedback_group)
        self.create_subscription(Odometry, '/base_controller/odom', self.on_odom, qos_profile_sensor_data, callback_group=self.feedback_group)
        self.create_subscription(PoseStamped, value('subject_pose_topic'), self.on_subject, 1, callback_group=self.feedback_group)
        self.create_timer(self.period, self.tick, callback_group=self.control_group)
        self.create_timer(.05, self.watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME), callback_group=self.control_group)
        self.telemetry = (self.output/'ros_telemetry.jsonl').open('w')

    def on_joints(self, message):
        names = {name: i for i, name in enumerate(message.name)}
        if not set(ARM_JOINT_NAMES).issubset(names):
            return
        q = np.array([message.position[names[n]] for n in ARM_JOINT_NAMES])
        if not np.isfinite(q).all():
            return
        stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        self.joints = q
        self.joint_wall = time.monotonic()
        self.lidar.joints.append((stamp, q.copy()))

    def on_odom(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        if not np.isfinite(values).all() or np.linalg.norm(values[3:]) < 1e-8:
            return
        self.odom = (np.array(values[:3]), Rotation.from_quat(values[3:]).as_matrix())
        self.odom_wall = time.monotonic()

    def on_subject(self, message):
        if message.header.frame_id not in ('odom', 'world', ''):
            self.get_logger().warning('Subject pose must be in the trajectory world/odom frame')
            return
        p = message.pose.position
        point = np.array([p.x, p.y, p.z])
        if np.isfinite(point).all():
            self.subject = point

    def publish(self, command, dt):
        self.q_command += dt*command[2:]
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = [*WHEEL_JOINT_NAMES, *ARM_JOINT_NAMES]
        message.position = [0., 0., *self.q_command.tolist()]
        message.velocity = [*(self.engine.limits.wheels@command).tolist(), *command[2:].tolist()]
        self.publisher.publish(message)

    def watchdog(self):
        if self.engine is None or self.completed:
            return
        if max(time.monotonic()-self.joint_wall, time.monotonic()-self.odom_wall) > self.timeout:
            self.fault = 'runtime robot feedback stale; constrained braking requested'

    def tick(self):
        now = self.get_clock().now().nanoseconds*1e-9
        if self.joints is None or self.odom is None or self.subject is None:
            return
        bp, br = self.odom
        state = np.r_[bp[:2], np.arctan2(br[1, 0], br[0, 0]), self.joints]
        if self.engine is None:
            if not self.lidar.ready(now):
                return
            reference = OriginalReference(self.task, self.subject)
            self.start_s = self.previous_s = now
            self.q_command = state[3:].copy()
            snapshot = self.lidar.map.snapshot()
            snapshot['observation_stamp'] -= self.start_s
            self.engine = CinematicEngine(reference, snapshot, self.output, mode=self.mode, api_url=self.api_url,
                                          preview=self.allocation_preview)
            start = String()
            start.data = json.dumps(dict(epoch_s=now))
            self.scene_start.publish(start)
            self.get_logger().info('Cinematic WBC started with a fixed original-time budget')
            return
        dt = now-self.previous_s
        if dt <= 0.:
            return
        self.previous_s = now
        if dt > .1:
            self.fault = 'control callback gap exceeded 0.1 s; braking'
            dt = .1
        t = now-self.start_s
        snapshot = self.lidar.map.snapshot()
        snapshot['observation_stamp'] -= self.start_s
        try:
            if self.fault or self.completed or self.lidar.stale(now):
                self.engine.wbc.update_collision_map(snapshot)
                self.engine.wbc.set_time(t)
                p, r, _ = self.engine.active.sample(t)
                u, info = self.engine.wbc.solve(state, p, r, np.zeros(6), dt,
                    base_z=bp[2], base_rotation=br, commanded_q=self.q_command, trajectory_time=t, brake=True)
            else:
                u, info = self.engine.step(state, snapshot, t, dt, bp[2], br, self.q_command)
            self.publish(u, dt)
            info.update(t=t, fault=self.fault)
            self.telemetry.write(json.dumps(native(info), allow_nan=False)+'\n')
            message = String()
            message.data = json.dumps(native(info), allow_nan=False)
            self.status.publish(message)
            if t >= self.engine.original.duration and not self.completed:
                self.completed = True
                (self.output/'ros_summary.json').write_text(json.dumps(native(dict(self.engine.summary(),
                    completed=True, time_overrun_s=max(0., t-self.engine.original.duration),
                    failure=self.fault, safety_verification='online model constraints only; no physical truth audit')), indent=2)+'\n')
        except Exception as error:
            self.fault = f'no certified control: {error}'
            self.get_logger().error(self.fault)
            # No uncertified command is published. Hardware integration must provide
            # an independent actuator watchdog. The lockstep evaluation runner
            # stops advancing physics on failure; that is not a hardware stop.

    def close(self):
        if self.engine:
            self.engine.close()
        self.lidar.close()
        self.lidar.save(self.output)
        self.telemetry.close()


def main(args=None):
    rclpy.init(args=args)
    node = CinematicNode()
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
