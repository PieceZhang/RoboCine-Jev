"""ROS 2 whole-body camera trajectory tracking with a single actuator publisher."""
import json
import signal
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.signals import SignalHandlerOptions
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .isaac_bridge import ARM_JOINT_NAMES, WHEEL_JOINT_NAMES
from .whole_body_qp import Limits, WholeBodyQP, whole_body_kinematics
from .wbc_metrics import Metrics
from .wbc_trajectory import CameraTrajectory
from .shot_predictive_qp import ShotPredictiveQP
from .shot_preview_worker import AsyncShotPredictiveQP, AsyncObstaclePredictiveQP
from .shot_execution import FixedMotionShotClock
from .whole_body_controller import WholeBodyController
from .dynamic_predictive_qp import AsyncDynamicPredictiveQP
from .dynamic_lidar_input import DynamicLidarCollisionInput


class DynamicWholeBodyController(WholeBodyController):
    def __init__(self):
        Node.__init__(self, 'whole_body_controller')
        defaults = dict(trajectory_file='', output_directory='/tmp/jvs_wbc',
                        controller_mode='kc', obstacle_file='', mesh_source='lidar',
                        lidar_topics=['/lidar/points', '/lidar/front/points'],
                        lidar_voxel_size=.04, lidar_map_radius=4.5, lidar_minimum_height=.03,
                        lidar_maximum_height=1.60, lidar_minimum_clearance=.025,
                        lidar_retention_s=1.5, lidar_timeout_s=.8, lidar_startup_timeout_s=60.,
                        preview_horizon_steps=8, preview_dt_s=.4,
                        preview_replan_s=.2, shot_blend_s=2.,
                        control_rate_hz=50., state_timeout_s=.15, settle_s=2., arm_prepare_s=0.,
                        adaptive_shot_timing=True, subject_lock=True,
                        position_rms_tolerance=.05, orientation_rms_tolerance=.1,
                        position_max_tolerance=0., tracking_weight=10.,
                        wheel_radius=.07, wheel_separation=.21, wheel_speed=15.,
                        wheel_acceleration=20., position_margin=.01,
                        velocity_limits=[.5, 1.2, 1., 1., 1., 1.2, 1.2, 1.5],
                        acceleration_limits=[.5, 1.5, 2., 2., 2., 2., 2., 2.],
                        dynamic_fusion_window_s=.30, dynamic_track_timeout_s=2.5,
                        dynamic_cluster_distance=.16, dynamic_velocity_window_s=1.8,
                        dynamic_maximum_speed=1.)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        value = lambda key: self.get_parameter(key).value
        self.period = 1./float(value('control_rate_hz'))
        if not .005 <= self.period <= .05:
            raise ValueError('control_rate_hz must be in [20, 200]')
        self.timeout = float(value('state_timeout_s'))
        self.settle = float(value('settle_s'))
        self.arm_prepare = float(value('arm_prepare_s'))
        if not np.isfinite(self.arm_prepare) or self.arm_prepare < 0:
            raise ValueError('arm_prepare_s must be finite and nonnegative')
        if not np.isfinite(self.timeout) or self.timeout <= 0 or self.settle < 0:
            raise ValueError('invalid timeout/settling parameters')
        self.trajectory_file = str(value('trajectory_file'))
        # Validate input before publishing any commands.
        CameraTrajectory(self.trajectory_file, np.zeros(3), np.eye(3), np.eye(3))
        limits = Limits(**{key: float(value(key)) for key in ('wheel_radius', 'wheel_separation',
                         'wheel_speed', 'wheel_acceleration', 'position_margin')},
                        velocity=np.array(value('velocity_limits')),
                        acceleration=np.array(value('acceleration_limits')))
        # Runtime-generated connectors share actuator/preview scales; no
        # trajectory-specific timing rules or additional execution states.
        self.transition_options = dict(
            minimum=max(2.*float(value('shot_blend_s')),
                        int(value('preview_horizon_steps'))*float(value('preview_dt_s'))),
            speed=min(.2, .5*limits.velocity[0]), acceleration=.5*limits.acceleration[0],
            angular_speed=.5*limits.velocity[1], angular_acceleration=.5*limits.acceleration[1])
        if value('controller_mode') != 'kc' or value('mesh_source') != 'lidar':
            raise ValueError('dynamic controller requires kc and live lidar')
        self.lidar = DynamicLidarCollisionInput(self, value)
        self.map_revision = -1
        self.controller = AsyncDynamicPredictiveQP(limits, scene=self.lidar.map.snapshot(),
            horizon_steps=int(value('preview_horizon_steps')), horizon_dt=float(value('preview_dt_s')),
            replan_s=float(value('preview_replan_s')), blend_s=float(value('shot_blend_s')),
            tracking_weight=float(value('tracking_weight')))
        self.metrics = Metrics(str(value('output_directory')), limits)
        self.metrics.position_tolerance = float(value('position_rms_tolerance'))
        self.metrics.orientation_tolerance = float(value('orientation_rms_tolerance'))
        if any(not np.isfinite(v) or v <= 0. for v in
               (self.metrics.position_tolerance, self.metrics.orientation_tolerance)):
            raise ValueError('accuracy tolerances must be positive and finite')
        maximum = float(value('position_max_tolerance'))
        if not np.isfinite(maximum) or maximum < 0.:
            raise ValueError('position_max_tolerance must be finite and nonnegative (0 disables the peak gate)')
        self.metrics.position_max_tolerance = maximum if maximum > 0. else np.inf
        self.adaptive_timing = bool(value('adaptive_shot_timing'))
        self.subject_lock = bool(value('subject_lock'))
        self.shot_clock = FixedMotionShotClock()
        self.scene_started = False
        self.scene_start = self.create_publisher(String, "/jvs/dynamic_scene/start", 1)
        self.publisher = self.create_publisher(JointState, '/isaac_joint_commands', 1)
        self.diagnostics = self.create_publisher(String, '/wbc/diagnostics', 10)
        self.feedback_group = MutuallyExclusiveCallbackGroup()
        feedback_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, '/joint_states', self.on_joints, feedback_qos,
                                 callback_group=self.feedback_group)
        self.create_subscription(Odometry, '/base_controller/odom', self.on_odom, feedback_qos,
                                 callback_group=self.feedback_group)
        self.create_subscription(String, '/wbc/shot_context', self.on_shot_context, 10)
        self.joints = self.odom = None
        self.joint_wall = self.odom_wall = time.monotonic()
        self.trajectory = None
        self.first_time = self.start_time = self.last_time = None
        self.q_command = None
        self.failure = ''
        self.failure_reported = False
        self.emergency = False
        self.completed = False
        self.shutdown_requested = False
        self.slow_ticks = []
        self.create_timer(self.period, self.tick)
        self.create_timer(.1, self.watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info('Waiting for synchronized Isaac odometry and joint state')

    def tick(self):
        wall_started = time.monotonic()
        self.controller.set_time(self.get_clock().now().nanoseconds * 1e-9)
        super().tick()
        elapsed = time.monotonic()-wall_started
        if elapsed > .1:
            self.slow_ticks.append(dict(physical_time=self.controller.physical_time, wall_duration_s=elapsed,
                surfels=len(self.controller.collision.centers), failure=self.failure))
            if elapsed > .3 and not self.failure:
                self.get_logger().warning(f'Dynamic control callback took {elapsed:.3f}s wall time')
        if self.failure and not (self.metrics.folder / 'failure_scene.npz').exists():
            np.savez_compressed(self.metrics.folder / 'failure_scene.npz',
                **{key: value for key, value in self.controller.collision.scene.items() if key != 'tracks'})
            (self.metrics.folder / 'failure_context.json').write_text(json.dumps(dict(
                physical_time=self.controller.physical_time,
                tracks=self.controller.collision.scene.get('tracks', []),
                safety=self.controller.safety_info), indent=2) + '\n')
        if self.trajectory is not None and not self.scene_started:
            message = String()
            message.data = json.dumps(dict(epoch_s=self.start_time))
            (self.metrics.folder / 'dynamic_epoch.json').write_text(message.data + '\n')
            self.scene_start.publish(message)
            self.scene_started = True


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DynamicWholeBodyController()
    executor = MultiThreadedExecutor(num_threads=3) if node.lidar else SingleThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        # launch may forward SIGINT after the terminal/process group already
        # delivered it. Finish bounded braking and file/pool cleanup once.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.shutdown_requested = True
        # Keep ROS alive while the same constrained QP ramps down all actuators.
        node.failure = node.failure or ('' if node.completed else 'interrupted')
        if node.trajectory is not None:
            node.start_time = -1e9  # select the constrained braking phase
            deadline = time.monotonic()+2.5
            while rclpy.ok() and time.monotonic() < deadline and np.max(np.abs(node.controller.previous)) > 1e-5:
                rclpy.spin_once(node, executor=executor, timeout_sec=.02)
        if rclpy.ok() and node.q_command is not None:
            node.publish(np.zeros(8))
    finally:
        executor.shutdown()
        if node.lidar is not None:
            node.lidar.close()
            node.lidar.save(node.metrics.folder)
        node.metrics.finish(node.completed and not node.failure, node.failure or ('' if node.completed else 'interrupted'))
        node.metrics.close()
        (node.metrics.folder/'slow_control_ticks.json').write_text(json.dumps(node.slow_ticks, indent=2)+'\n')
        if isinstance(node.controller, AsyncShotPredictiveQP):
            node.controller.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
