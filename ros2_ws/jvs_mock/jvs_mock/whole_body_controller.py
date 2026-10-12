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
from .shot_execution import ShotClock


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class WholeBodyController(Node):
    def __init__(self):
        super().__init__('whole_body_controller')
        defaults = dict(trajectory_file='', output_directory='/tmp/jvs_wbc',
                        controller_mode='stage1', obstacle_file='', mesh_source='groundtruth',
                        lidar_topics=['/lidar/points', '/lidar/front/points'],
                        lidar_voxel_size=.08, lidar_map_radius=3., lidar_minimum_height=.03,
                        lidar_maximum_height=1.60, lidar_minimum_clearance=.025,
                        lidar_retention_s=30., lidar_timeout_s=2., lidar_startup_timeout_s=60.,
                        preview_horizon_steps=8, preview_dt_s=.4,
                        preview_replan_s=.2, shot_blend_s=2.,
                        control_rate_hz=50., state_timeout_s=.15, settle_s=2., arm_prepare_s=0.,
                        adaptive_shot_timing=False, subject_lock=False,
                        position_rms_tolerance=.02, orientation_rms_tolerance=.035,
                        position_max_tolerance=0., tracking_weight=10.,
                        wheel_radius=.07, wheel_separation=.21, wheel_speed=15.,
                        wheel_acceleration=20., position_margin=.01,
                        velocity_limits=[.5, 1.2, 1., 1., 1., 1.2, 1.2, 1.5],
                        acceleration_limits=[.5, 1.5, 2., 2., 2., 2., 2., 2.])
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
        mode = value('controller_mode')
        if mode not in ('stage1', 'stage2', 'no_preview', 'none', 'per_link', 'kc'):
            raise ValueError('unknown controller_mode')
        predictive = AsyncShotPredictiveQP if mode == 'stage2' else ShotPredictiveQP
        extras = {}
        self.lidar = None
        if value('mesh_source') not in ('groundtruth', 'lidar'):
            raise ValueError('mesh_source must be groundtruth or lidar')
        if value('mesh_source') == 'lidar':
            if mode not in ('none', 'per_link', 'kc'):
                raise ValueError('LiDAR requires a collision controller mode')
            from .lidar_collision_input import LidarCollisionInput
            self.lidar = LidarCollisionInput(self, value)
        self.map_revision = -1
        if mode in ('none','per_link','kc'):
            predictive = AsyncObstaclePredictiveQP
            extras = dict(scene=self.lidar.map.snapshot() if self.lidar else str(value('obstacle_file')),
                          collision_mode=mode)
        self.controller = WholeBodyQP(limits) if mode == 'stage1' else predictive(limits,
            **extras, preview=mode != 'no_preview', horizon_steps=int(value('preview_horizon_steps')),
            horizon_dt=float(value('preview_dt_s')), replan_s=float(value('preview_replan_s')),
            blend_s=float(value('shot_blend_s')), tracking_weight=float(value('tracking_weight')))
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
        self.shot_clock = ShotClock()
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
        self.create_timer(self.period, self.tick)
        self.create_timer(.1, self.watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info('Waiting for synchronized Isaac odometry and joint state')

    def on_shot_context(self, message):
        if not isinstance(self.controller, ShotPredictiveQP) or self.start_time is None:
            return
        try:
            context = json.loads(message.data)
            context['received_s'] = self.get_clock().now().nanoseconds*1e-9-self.start_time
            self.controller.set_shot_input(context)
        except (ValueError, TypeError, KeyError) as error:
            self.get_logger().error(f'Rejected shot input: {error}')

    def on_joints(self, msg):
        names = list(WHEEL_JOINT_NAMES + ARM_JOINT_NAMES)
        if not all(n in msg.name for n in names):
            return
        indices = [msg.name.index(n) for n in names]
        if len(msg.position) <= max(indices) or len(msg.velocity) <= max(indices):
            return
        q, dq = np.array(msg.position)[indices], np.array(msg.velocity)[indices]
        if np.all(np.isfinite(q)) and np.all(np.isfinite(dq)):
            self.joints = stamp_seconds(msg.header.stamp), q, dq
            self.joint_wall = time.monotonic()
            if self.lidar is not None:
                self.lidar.joints.append((self.joints[0], q[2:].copy()))

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                  msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
        if np.all(np.isfinite(values)) and np.linalg.norm(values[3:7]) > .5:
            self.odom = msg
            self.odom_wall = time.monotonic()

    def publish(self, command):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(WHEEL_JOINT_NAMES + ARM_JOINT_NAMES)
        msg.position = [0., 0., *self.q_command.tolist()]
        msg.velocity = [*(self.controller.limits.wheels @ command).tolist(), *command[2:].tolist()]
        self.publisher.publish(msg)

    def watchdog(self):
        if self.shutdown_requested:
            return
        if (self.lidar is not None and self.trajectory is None and not self.emergency
                and time.monotonic() - self.lidar.started_wall > self.lidar.startup_timeout):
            self.failure = 'LiDAR startup timeout: ' + (self.lidar.error or 'both scans/map required')
            self.emergency = True
            if self.q_command is not None:
                self.publish(np.zeros(8))
            self.lidar.save(self.metrics.folder)
            self.metrics.finish(False, self.failure)
        # Isaac can pause during initial shader/physics setup after its first
        # state publication. Arm the watchdog only once tracking actually starts.
        if self.q_command is None or self.trajectory is None:
            return
        if time.monotonic()-min(self.joint_wall, self.odom_wall) > 1.:
            self.failure = 'feedback absent for 1s wall time; emergency stop (constraints not guaranteed)'
            self.emergency = True
            self.publish(np.zeros(8))
            if not self.completed:
                self.metrics.finish(False, self.failure)
        if self.lidar is not None and self.lidar.stale(self.get_clock().now().nanoseconds * 1e-9):
            self.failure = 'LiDAR feedback stale; emergency stop'
            self.emergency = True
            self.publish(np.zeros(8))
            if not self.completed:
                self.metrics.finish(False, self.failure)

    def tick(self):
        if self.joints is None or self.odom is None:
            return
        odom = self.odom
        if self.emergency:
            self.publish(np.zeros(8))
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_time is not None and now <= self.last_time:
            if now < self.last_time:
                self.failure = 'simulation clock reset; restart controller after simulator reset'
                self.emergency = True
                if self.q_command is not None:
                    self.publish(np.zeros(8))
                self.metrics.finish(False, self.failure)
            return
        dt = self.period if self.last_time is None else now-self.last_time
        if dt > .1:
            if self.trajectory is not None:
                self.failure = 'control deadline missed (>100 ms simulation time)'
            dt = self.period
        self.last_time = now
        stamp, q, dq = self.joints
        odom_stamp = stamp_seconds(odom.header.stamp)
        if self.trajectory is not None and max(now-stamp, now-odom_stamp, abs(stamp-odom_stamp)) > self.timeout:
            self.failure = 'stale or unsynchronized state feedback'
        if self.first_time is None:
            self.first_time = now
        bp, bq = odom.pose.pose.position, odom.pose.pose.orientation
        br = Rotation.from_quat([bq.x, bq.y, bq.z, bq.w]).as_matrix()
        yaw = np.arctan2(br[1, 0], br[0, 0])
        state = np.r_[bp.x, bp.y, yaw, q[2:]]
        p, r, _ = whole_body_kinematics(state, bp.z, br)
        if self.q_command is None:
            self.q_command = q[2:].copy()
        if self.lidar is not None:
            if self.trajectory is None and not self.lidar.ready(now):
                self.publish(np.zeros(8))
                return
            if self.trajectory is not None and (self.lidar.stale(now) or not self.lidar.map.cells):
                self.failure = 'LiDAR feedback stale; emergency stop'
                self.emergency = True
                self.publish(np.zeros(8))
                self.metrics.finish(False, self.failure)
                return
            if self.map_revision != self.lidar.map.revision:
                snapshot = self.lidar.map.snapshot()
                self.controller.update_collision_map(snapshot)
                self.map_revision = snapshot['revision']
        if now-self.first_time < self.settle:
            self.publish(np.zeros(8))
            return
        if self.trajectory is None:
            self.trajectory = CameraTrajectory(self.trajectory_file, p, r, br,
                                               transition_options=self.transition_options)
            if hasattr(self.shot_clock, 'set_trajectory'):
                self.shot_clock.set_trajectory(self.trajectory)
            self.metrics.trajectory = self.trajectory
            (self.metrics.folder/'reference.json').write_text(json.dumps(dict(
                initial_camera_position=p.tolist(), initial_camera_rotation=r.tolist(),
                initial_base_rotation=br.tolist(),
                trajectory_format=self.trajectory.document.get('format'),
                transition_options=self.transition_options,
                execution_shots=self.trajectory.shots,
                automatic_transitions=self.trajectory.automatic_transitions,
                motion_timing_mode=getattr(self.shot_clock, 'motion_timing_mode', 'adaptive'),
                execution_duration_s=self.trajectory.duration,
                subject_odom_m=None if self.trajectory.subject is None else self.trajectory.subject.tolist()), indent=2)+'\n')
            if isinstance(self.controller, ShotPredictiveQP):
                self.controller.set_trajectory(self.trajectory)
            preparation = self.arm_prepare if isinstance(self.controller, ShotPredictiveQP) else 0.
            self.start_time = now+preparation
            if preparation:
                self.get_logger().info(f'Preparing arm for {preparation:g}s while holding the initial camera target')
            self.get_logger().info(f'Starting {self.trajectory.duration:.1f}s whole-body trajectory')
        elapsed = now-self.start_time
        wall_elapsed = elapsed
        rate = 1.
        if self.adaptive_timing and elapsed >= 0. and not self.shutdown_requested:
            ref_p, ref_r, _ = self.trajectory.sample(self.shot_clock.time)
            from .arm_comfort import arm_metrics
            reach = arm_metrics(state[3:])['arm_horizontal_reach_m']
            heading_error = 0.
            if getattr(self.controller, 'route', None) is not None:
                heading = self.controller.heading(self.shot_clock.time, state[2], state=state)[0]
                heading_error = self.controller.heading_error(heading, state)
            elapsed, rate = self.shot_clock.advance(dt, np.linalg.norm(ref_p-p),
                np.linalg.norm(Rotation.from_matrix(ref_r@r.T).as_rotvec()),
                self.trajectory.transition(self.shot_clock.time) is not None, heading_error, reach,
                minimum_rate=.5 if getattr(self.controller, 'projected_exit', lambda _: False)(self.shot_clock.time) else .15,
                position_scale=.5*min(self.metrics.position_tolerance, self.metrics.position_max_tolerance),
                orientation_scale=.5*self.metrics.orientation_tolerance,
                preparing=getattr(self.controller, 'transition_preparing', lambda _: False)(self.shot_clock.time))
        target_p, target_r, twist = self.trajectory.sample(elapsed)
        reference_r = target_r.copy()
        acquire_end = self.trajectory.shots[0].get('focus_acquire_end_s', 2.)
        if self.subject_lock and self.trajectory.subject is not None and elapsed >= acquire_end:
            forward = self.trajectory.subject-p
            forward /= np.linalg.norm(forward)
            left = np.cross([0., 0., 1.], forward)
            left /= np.linalg.norm(left)
            target_r = np.column_stack([forward, left, np.cross(forward, left)])
        twist *= rate
        if isinstance(self.controller, ShotPredictiveQP):
            self.controller.path_rate = rate
        brake = bool(self.failure) or elapsed > self.trajectory.duration+2.
        try:
            command, info = self.controller.solve(state, target_p, target_r, twist, dt,
                base_z=bp.z, base_rotation=br, commanded_q=self.q_command, brake=brake,
                **({'trajectory_time': max(0., elapsed), 'preparing': elapsed < 0.}
                   if isinstance(self.controller, ShotPredictiveQP) else {}))
        except (RuntimeError, ValueError) as error:
            self.failure = str(error)
            if hasattr(self.controller, 'failure_problem'):
                (self.metrics.folder/'failure_qp.json').write_text(json.dumps(self.controller.failure_problem,indent=2)+'\n')
            self.get_logger().error('WBC fault: ' + self.failure)
            self.metrics.finish(False, self.failure)
            # A second constrained QP minimizes speed, preserving hard bounds.
            # If physical disturbances make even braking infeasible, explicitly
            # mark emergency stop; never describe it as constraint-compliant.
            try:
                command, info = self.controller.solve(state, p, r, np.zeros(6), dt,
                    base_z=bp.z, base_rotation=br, commanded_q=self.q_command, brake=True)
            except (RuntimeError, ValueError):
                self.emergency = True
                self.publish(np.zeros(8))
                return
        self.q_command += command[2:]*dt
        self.publish(command)
        if self.completed:
            return
        measured = np.r_[odom.twist.twist.linear.x, odom.twist.twist.angular.z, dq[2:]]
        info.update(wall_elapsed_s=wall_elapsed, path_rate=rate,
                    transition_active=int(self.trajectory.transition(elapsed) is not None))
        info['base_z'] = float(bp.z)
        info.update({f'base_rotvec_{axis}': float(v) for axis, v in
                     zip('xyz', Rotation.from_matrix(br).as_rotvec())})
        if self.lidar is not None:
            info.update(lidar_voxels=len(self.lidar.map.cells), lidar_revision=self.map_revision,
                        lidar_update_ms=self.lidar.update_ms,
                        lidar_age_s=max(now - stamp for stamp in self.lidar.stamps.values()))
        # Report the original planned orientation, even when subject locking
        # deliberately adjusts composition for a displaced camera position.
        info['orientation_error'] = float(np.linalg.norm(Rotation.from_matrix(reference_r@r.T).as_rotvec()))
        info.update({f'camera_{axis}': float(v) for axis, v in zip('xyz', p)})
        info.update({f'camera_forward_{axis}': float(v) for axis, v in zip('xyz', r[:, 0])})
        row = self.metrics.record(elapsed, dt, state, command, measured, dq[:2],
            odom.twist.twist.linear.y, info,
            'preparing' if elapsed < 0. else 'tracking' if elapsed <= self.trajectory.duration else 'stopping')
        message = String()
        message.data = json.dumps(dict(**row, fault=self.failure))
        self.diagnostics.publish(message)
        if self.failure and not self.failure_reported:
            self.failure_reported = True
            self.metrics.finish(False, self.failure)
            self.get_logger().error('WBC fault: ' + self.failure)
        if not self.completed and elapsed > self.trajectory.duration+3.:
            self.completed = True
            summary = self.metrics.finish(not self.failure, self.failure)
            if self.lidar is not None:
                self.lidar.save(self.metrics.folder)
            self.get_logger().info('WBC_RESULT ' + json.dumps(summary))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = WholeBodyController()
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
        if isinstance(node.controller, AsyncShotPredictiveQP):
            node.controller.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
