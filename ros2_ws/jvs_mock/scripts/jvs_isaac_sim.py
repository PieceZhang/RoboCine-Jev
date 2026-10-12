#!/usr/bin/env python3
"""Run the JVS mock as a ROS-connected Isaac Sim articulation.

Only Python's standard library is imported before ``SimulationApp`` starts.
Omniverse, USD and Isaac modules are deliberately imported afterwards, as
required by Isaac Sim's standalone-Python lifecycle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")
WHEEL_LINKS = ("left_wheel_link", "right_wheel_link")
CASTER_LINKS = (
    "front_caster_wheel_link",
    "rear_left_caster_wheel_link",
    "rear_right_caster_wheel_link",
)
CASTER_JOINTS = (
    "front_caster_swivel_joint",
    "front_caster_wheel_joint",
    "rear_left_caster_swivel_joint",
    "rear_left_caster_wheel_joint",
    "rear_right_caster_swivel_joint",
    "rear_right_caster_wheel_joint",
)

DEFAULT_LIDAR_CONFIG = "JVS_Nearfield"
LIDAR_PROFILE_SETTING = "/app/sensors/nv/lidar/profileBaseFolder"


def add_front_lidar_arguments(parser):
    parser.add_argument("--lidar-mode", choices=("front", "both"), default="both",
                        help="front: front LiDAR only; both: front and rear LiDARs (default)")
    parser.add_argument("--enable-front-lidar", type=boolean, default=True,
                        help="legacy rear-only switch when false; enable-lidar is the master switch")
    parser.add_argument("--front-lidar-config", default="Example_Rotary")
    parser.add_argument("--front-lidar-topic", default="/lidar/front/points")


def enabled_lidars(args):
    """Resolve the master switch, two public modes, and legacy rear-only switch."""
    if not args.enable_lidar:
        return ()
    if args.lidar_mode == "front":
        if not args.enable_front_lidar:
            raise ValueError("lidar-mode=front conflicts with enable-front-lidar=false")
        return ("front",)
    if args.lidar_mode != "both":
        raise ValueError(f"unsupported LiDAR mode: {args.lidar_mode}")
    return ("rear", "front") if args.enable_front_lidar else ("rear",)


def validate_front_lidar_arguments(args):
    enabled = enabled_lidars(args)
    if "rear" in enabled:
        if not args.lidar_config.strip() or not args.lidar_topic.strip() or not args.lidar_frame_id.strip():
            raise ValueError("rear LiDAR config, topic and frame must not be empty")
    if "front" in enabled:
        if not args.front_lidar_config.strip() or not args.front_lidar_topic.strip():
            raise ValueError("front LiDAR config and topic must not be empty")
    if len(enabled) == 2:
        if args.front_lidar_topic.strip('/') == args.lidar_topic.strip('/'):
            raise ValueError("front and rear LiDAR topics must be distinct")
        if args.lidar_frame_id == "front_lidar_link":
            raise ValueError("front and rear LiDAR frames must be distinct")


class RobotResetService:
    """Reset one live Isaac articulation to its declared startup state."""

    def __init__(self, robot_path, native_pose, scene_pose, service_name):
        import numpy as np
        import rclpy
        from isaacsim.core.prims import SingleArticulation
        from std_srvs.srv import Trigger

        self._np = np
        self._rclpy = rclpy
        self.robot_path = robot_path
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self.node = rclpy.create_node("jvs_isaac_reset_server")
        self.command_filter = None
        self.physics_feedback = None
        self.articulation = SingleArticulation(
            robot_path,
            name="jvs_reset_articulation",
            reset_xform_properties=False,
        )
        self.articulation.initialize()
        self.position = np.asarray(native_pose[:3], dtype=float)
        yaw = float(native_pose[3])
        self.orientation = np.asarray(
            [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)],
            dtype=float,
        )
        self.scene_pose = dict(scene_pose)
        self.joint_names = list(self.articulation.dof_names)
        self.joint_positions = np.zeros(len(self.joint_names), dtype=float)
        self.service = self.node.create_service(
            Trigger, service_name, self._reset
        )
        self.node.get_logger().info(
            f"Robot reset service ready: {service_name}"
        )

    def hold_startup_pose(self):
        """Start at rest; PhysX wakes the robot when a drive command arrives.

        This is a one-time initialization, not a per-frame pose override.
        Contacts and subsequent commands retain normal dynamic behavior.
        """
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, Sdf

        zeros3 = self._np.zeros(3, dtype=float)
        self.articulation.set_world_pose(self.position, self.orientation)
        self.articulation.set_linear_velocity(zeros3)
        self.articulation.set_angular_velocity(zeros3)
        self.articulation.set_joint_positions(self.joint_positions)
        self.articulation.set_joint_velocities(self._np.zeros(len(self.joint_names)))
        physics = get_physx_simulation_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        body_id = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(self.robot_path))
        physics.put_to_sleep(stage_id, body_id)
        if not physics.is_sleeping(stage_id, body_id):
            raise RuntimeError("could not initialize the robot in its idle state")
        self.node.get_logger().info("Robot initialized at rest, waiting for joint commands")

    def _reset(self, request, response):
        del request
        np = self._np
        try:
            zeros3 = np.zeros(3, dtype=float)
            zeros_joint = np.zeros(len(self.joint_names), dtype=float)
            self.articulation.set_world_pose(self.position, self.orientation)
            self.articulation.set_linear_velocity(zeros3)
            self.articulation.set_angular_velocity(zeros3)
            self.articulation.set_joint_positions(self.joint_positions)
            self.articulation.set_joint_velocities(zeros_joint)

            actual_position, actual_orientation = self.articulation.get_world_pose()
            actual_joints = self.articulation.get_joint_positions()
            actual_linear_velocity = self.articulation.get_linear_velocity()
            actual_angular_velocity = self.articulation.get_angular_velocity()
            actual_joint_velocities = self.articulation.get_joint_velocities()
            position_ok = np.allclose(actual_position, self.position, atol=1e-5)
            orientation_ok = (
                abs(float(np.dot(actual_orientation, self.orientation)))
                >= 1.0 - 1e-5
            )
            joints_ok = np.allclose(actual_joints, self.joint_positions, atol=1e-5)
            velocities_ok = (
                np.allclose(actual_linear_velocity, zeros3, atol=1e-5)
                and np.allclose(actual_angular_velocity, zeros3, atol=1e-5)
                and np.allclose(actual_joint_velocities, zeros_joint, atol=1e-5)
            )
            if not (position_ok and orientation_ok and joints_ok and velocities_ok):
                raise RuntimeError(
                    "Isaac articulation did not accept the complete reset state"
                )
            if self.command_filter is not None:
                self.command_filter.reset()

            response.success = True
            response.message = json.dumps({
                "scene_y_up": self.scene_pose,
                "isaac_z_up": {
                    "position_m": self.position.tolist(),
                    "orientation_wxyz": self.orientation.tolist(),
                },
                "joint_names": self.joint_names,
                "joint_positions_rad": self.joint_positions.tolist(),
                "root_linear_velocity_mps": [0.0, 0.0, 0.0],
                "root_angular_velocity_radps": [0.0, 0.0, 0.0],
                "simulation_clock_reset": False,
            }, separators=(",", ":"))
            self.node.get_logger().info("Robot articulation reset completed")
        except Exception as error:
            response.success = False
            response.message = f"{type(error).__name__}: {error}"
            self.node.get_logger().error(
                f"Robot articulation reset failed: {response.message}"
            )
        return response

    def spin_once(self):
        for _ in range(3):
            self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        if self.physics_feedback is not None:
            self.physics_feedback.close()
        if self.command_filter is not None:
            self.command_filter.close()
        self.node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


class PhysicsFeedback:
    """Publish one full-precision pose/joint snapshot and its clock together."""

    def __init__(self, reset_server, frequency, initial_time=0.):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from omni.physx import get_physx_interface
        from sensor_msgs.msg import JointState
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        self.np, self.rotation = np, Rotation
        self.joint_type, self.odom_type, self.clock_type = JointState, Odometry, Clock
        self.transform_type, self.tf_type = TransformStamped, TFMessage
        self.robot, self.names = reset_server.articulation, reset_server.joint_names
        self.origin = reset_server.position.copy()
        w,x,y,z=reset_server.orientation
        self.initial_inverse = Rotation.from_quat([x,y,z,w]).inv()
        self.time, self.next_time, self.period = initial_time, initial_time, 1./frequency
        self.samples, self.error = 0, ''
        node=reset_server.node
        self.joints=node.create_publisher(JointState, '/joint_states', 1)
        self.odom=node.create_publisher(Odometry, '/base_controller/odom', 1)
        self.clock=node.create_publisher(Clock, '/clock', 1)
        self.tf=node.create_publisher(TFMessage, '/tf', 1)
        self.subscription=get_physx_interface().subscribe_physics_step_events(self.on_step)
        reset_server.physics_feedback=self

    def on_step(self, dt):
        self.time += float(dt)
        if self.error or self.time < self.next_time-1e-9:
            return
        self.next_time += self.period
        try:
            clock=self.clock_type()
            clock.clock.sec=int(self.time)
            clock.clock.nanosec=int(round((self.time-int(self.time))*1e9))
            if clock.clock.nanosec >= 1000000000:
                clock.clock.sec += 1
                clock.clock.nanosec -= 1000000000
            js=self.joint_type();js.header.stamp=clock.clock;js.name=self.names
            js.position=self.robot.get_joint_positions().astype(float).tolist()
            js.velocity=self.robot.get_joint_velocities().astype(float).tolist()
            p,q=self.robot.get_world_pose();w,x,y,z=q
            native=self.rotation.from_quat([x,y,z,w])
            relative=self.initial_inverse*native
            position=self.initial_inverse.apply(p-self.origin)
            quaternion=relative.as_quat()
            linear=native.inv().apply(self.robot.get_linear_velocity())
            angular=native.inv().apply(self.robot.get_angular_velocity())
            odom=self.odom_type();odom.header.stamp=clock.clock
            odom.header.frame_id='odom';odom.child_frame_id='base_link'
            for axis,value in zip('xyz',position):setattr(odom.pose.pose.position,axis,float(value))
            for axis,value in zip('xyzw',quaternion):setattr(odom.pose.pose.orientation,axis,float(value))
            for axis,value in zip('xyz',linear):setattr(odom.twist.twist.linear,axis,float(value))
            for axis,value in zip('xyz',angular):setattr(odom.twist.twist.angular,axis,float(value))
            tf=self.transform_type();tf.header=odom.header;tf.child_frame_id=odom.child_frame_id
            for axis in 'xyz':setattr(tf.transform.translation,axis,getattr(odom.pose.pose.position,axis))
            tf.transform.rotation=odom.pose.pose.orientation
            self.joints.publish(js);self.odom.publish(odom);self.tf.publish(self.tf_type(transforms=[tf]))
            self.clock.publish(clock)
            self.samples += 1
        except Exception as error:
            self.error=str(error);print('JVS physics feedback failed: '+self.error,flush=True)

    def close(self):
        self.subscription=None
        print('JVS physics feedback audit: '+json.dumps(dict(samples=self.samples,error=self.error,
            physics_time_s=self.time, period_s=self.period, full_precision=True)),flush=True)
        if self.error:raise RuntimeError(self.error)


class PhysicsCommandFilter:
    """Execute latest drive targets per PhysX step; zero frequency is unfiltered."""

    def __init__(self, reset_server, urdf_path, frequency, initial_time=0., audit_file=''):
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        from omni.physx import get_physx_interface
        from sensor_msgs.msg import JointState
        package = str(Path(__file__).resolve().parents[1])
        if package not in sys.path:
            sys.path.insert(0, package)
        from jvs_mock.joint_command_filter import JointCommandFilter
        self.robot = reset_server.articulation
        names = list(self.robot.dof_names)
        self.indices = np.asarray([names.index(n) for n in WHEEL_JOINTS+ARM_JOINTS])
        limits = {j.get('name'): j.find('limit') for j in ET.parse(urdf_path).getroot().findall('joint')}
        self.filter = JointCommandFilter(self.robot.get_joint_positions(self.indices[2:]),
            [float(limits[n].get('lower')) for n in ARM_JOINTS],
            [float(limits[n].get('upper')) for n in ARM_JOINTS],
            [float(limits[n].get('velocity')) for n in ARM_JOINTS], frequency)
        self.time, self.steps, self.error = initial_time, 0, ''
        self.frequency, self.audit_file = frequency, audit_file
        self.reset_server = reset_server
        self.maximum_command_latency = 0.
        self.np, self.action = np, ArticulationAction
        self.commands = reset_server.node.create_subscription(JointState, '/isaac_joint_commands', self.on_command, 1)
        self.subscription = get_physx_interface().subscribe_physics_step_events(self.on_step)
        reset_server.command_filter = self

    def reset(self):
        self.filter.reset(self.robot.get_joint_positions(self.indices[2:]))

    def on_command(self, message):
        try:
            indices = [message.name.index(n) for n in WHEEL_JOINTS+ARM_JOINTS]
            stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
            if self.filter.push(stamp, [message.position[i] for i in indices[2:]],
                    [message.velocity[i] for i in indices[2:]], [message.velocity[i] for i in indices[:2]], self.time):
                self.maximum_command_latency = max(self.maximum_command_latency, self.time-stamp)
        except (ValueError, IndexError):
            self.filter.rejected += 1

    def on_step(self, dt):
        self.time += float(dt)
        # Receive pending commands during substeps too. Rendering may be slow;
        # the depth-one subscription discards stale intermediate targets.
        self.reset_server.spin_once()
        if self.error or self.filter.command is None:
            return
        try:
            q, v, w = self.filter.step(self.time, float(dt))
            self.robot.apply_action(self.action(joint_positions=self.np.r_[0., 0., q],
                joint_velocities=self.np.r_[w, v], joint_indices=self.indices))
            self.steps += 1
        except Exception as error:
            self.error = str(error)
            print('JVS physics command filter failed: '+self.error, flush=True)

    def close(self):
        self.subscription = None
        report = dict(format='jvs_physics_command_filter/v1', frequency_hz=self.frequency,
            command_mode='unfiltered' if self.frequency == 0 else 'critically_damped_filter',
            execution='physical_steps', subscriber_depth=1,
            maximum_command_latency_s=self.maximum_command_latency,
            physics_steps=self.steps, accepted_commands=self.filter.accepted,
            rejected_commands=self.filter.rejected, unsynchronized_commands=self.filter.unsynchronized,
            bounded_steps=self.filter.bounded_steps, maximum_command_age_s=self.filter.maximum_age,
            error=self.error, joint_state_override=False)
        if self.audit_file:
            Path(self.audit_file).write_text(json.dumps(report, indent=2)+'\n')
        print('JVS command filter audit: '+json.dumps(report), flush=True)
        if self.error:
            raise RuntimeError(self.error)


class CameraMotionAudit:
    """Observe the live PhysX bodies at every step without altering their state."""

    def __init__(self, path, robot, camera_path, *, initial_time=0., scene=None):
        import csv
        from isaacsim.core.prims import SingleRigidPrim
        from omni.physx import get_physx_interface
        self.robot = robot
        self.camera = SingleRigidPrim(camera_path, name='camera_motion_audit',
                                      reset_xform_properties=False)
        self.camera.initialize()
        names = list(robot.dof_names)
        self.indices = [names.index(name) for name in WHEEL_JOINTS + ARM_JOINTS]
        self.time = float(initial_time)
        self.scene = scene
        self.samples = 0
        self.error = ''
        kp, kd = robot.get_articulation_controller().get_gains()
        self.drive_gains_rad = {name: dict(stiffness=float(kp[i]), damping=float(kd[i]))
                               for name, i in zip(WHEEL_JOINTS + ARM_JOINTS, self.indices)}
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('x', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['simulation_time_s', 'scene_time_s',
            *['camera_' + a for a in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')],
            *['camera_linear_' + a for a in 'xyz'], *['camera_angular_' + a for a in 'xyz'],
            *['base_' + a for a in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')],
            *['base_linear_' + a for a in 'xyz'], *['base_angular_' + a for a in 'xyz'],
            *['position_' + n for n in WHEEL_JOINTS + ARM_JOINTS],
            *['velocity_' + n for n in WHEEL_JOINTS + ARM_JOINTS]])
        self.subscription = get_physx_interface().subscribe_physics_step_events(self.on_step)

    def on_step(self, dt):
        self.time += float(dt)
        if self.error:
            return
        try:
            cp, cq = self.camera.get_world_pose()
            bp, bq = self.robot.get_world_pose()
            epoch = self.scene.epoch if self.scene is not None else 0.
            scene_time = self.time - epoch if epoch is not None else float('nan')
            self.writer.writerow([self.time, scene_time, *cp, *cq,
                *self.camera.get_linear_velocity(), *self.camera.get_angular_velocity(),
                *bp, *bq, *self.robot.get_linear_velocity(), *self.robot.get_angular_velocity(),
                *self.robot.get_joint_positions(self.indices),
                *self.robot.get_joint_velocities(self.indices)])
            self.samples += 1
            if self.samples % 1000 == 0:
                self.file.flush()
        except Exception as error:
            self.error = str(error)
            print('JVS camera physics audit failed: ' + self.error, flush=True)

    def close(self):
        self.subscription = None
        self.file.close()
        self.path.with_suffix('.json').write_text(json.dumps(dict(
            format='jvs_camera_physics_audit/v1', samples=self.samples,
            error=self.error, unfiltered=True, sampling='every PhysX step',
            drive_gains_rad=self.drive_gains_rad,
            quaternion_order='wxyz', coordinate_frame='Isaac world Z-up')) + '\n')
        if self.error:
            raise RuntimeError('camera physics audit failed: ' + self.error)


def boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def angular_drive_gain_to_usd(gain: float) -> float:
    """Convert a per-radian PhysX gain to USD's per-degree angular gain.

    Isaac's articulation API uses radians; authoring the same numeric gain
    directly on a USD angular drive multiplies its physical value by 180/pi.
    """
    return float(gain)*math.pi/180.


def add_physics_arguments(parser: argparse.ArgumentParser, *, solver_type: str = 'PGS') -> None:
    """Share explicit PhysX/servo settings across the standalone and scene runners."""
    parser.add_argument("--solver-velocity-iterations", type=int, default=16)
    parser.add_argument("--arm-drive-stiffness", type=float, default=200., help='PhysX gain per radian')
    parser.add_argument("--arm-drive-damping", type=float, default=1000., help='PhysX gain per radian/second')
    parser.add_argument('--arm-drive-type', choices=['force', 'acceleration'], default='force',
                        help='force gains in Nm/rad; acceleration gains normalize the articulation inertia')
    parser.add_argument("--actuator-armature", type=float, default=0.)
    parser.add_argument("--solver-type", choices=['TGS', 'PGS'], default=solver_type)
    parser.add_argument("--wheel-drive-damping", type=float, default=20., help='Nm per radian/second')
    parser.add_argument('--command-filter-hz', type=float, default=0., help='critical drive-target filter; zero disables it')
    parser.add_argument('--physics-feedback-hz', type=float, default=50., help='synchronized physical snapshots; zero uses the legacy render graph')



def validate_physics_arguments(args: argparse.Namespace) -> None:
    if not 1 <= getattr(args, 'solver_velocity_iterations', 16) <= 255:
        raise ValueError('solver-velocity-iterations must be in [1, 255]')
    if any(not math.isfinite(getattr(args, key, default)) or getattr(args, key, default) <= 0
           for key, default in [('arm_drive_stiffness', 200.), ('arm_drive_damping', 1000.), ('wheel_drive_damping', 20.)]):
        raise ValueError('drive gains must be positive and finite')
    if not math.isfinite(getattr(args, 'actuator_armature', 0.)) or getattr(args, 'actuator_armature', 0.) < 0:
        raise ValueError('actuator-armature must be finite and nonnegative')
    if not math.isfinite(args.physics_hz) or args.physics_hz <= 0:
        raise ValueError('physics-hz must be positive and finite')
    frequency = getattr(args, 'command_filter_hz', 0.)
    if not math.isfinite(frequency) or not 0 <= frequency <= args.physics_hz/10:
        raise ValueError('command-filter-hz must be finite and between zero and physics-hz/10')
    feedback = getattr(args, 'physics_feedback_hz', 0.)
    if not math.isfinite(feedback) or not 0 <= feedback <= args.physics_hz:
        raise ValueError('physics-feedback-hz must be finite and between zero and physics-hz')



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xacro", required=True, help="JVS robot Xacro file")
    parser.add_argument(
        "--xacro-executable", default="xacro", help="xacro executable or absolute path"
    )
    parser.add_argument("--stage", default="", help="optional environment USD to open")
    parser.add_argument("--obstacles", default="", help="static jvs_obstacles/v1 JSON in world frame")
    parser.add_argument("--save-stage", default="", help="optional composed USD output")
    parser.add_argument("--headless", type=boolean, default=False)
    parser.add_argument("--add-ground-plane", type=boolean, default=True)
    parser.add_argument("--physics-hz", type=float, default=1000.0)
    add_physics_arguments(parser)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after N app updates; 0 runs until Isaac Sim exits",
    )
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.002)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument(
        "--enable-lidar",
        type=boolean,
        default=True,
        help="create the body-mounted RTX LiDAR and publish its ROS 2 point cloud",
    )
    parser.add_argument(
        "--lidar-config",
        default=DEFAULT_LIDAR_CONFIG,
        help="Isaac Sim RTX LiDAR configuration name",
    )
    parser.add_argument(
        "--lidar-variant",
        default="",
        help="optional RTX LiDAR model variant",
    )
    parser.add_argument("--lidar-topic", default="/lidar/points")
    parser.add_argument("--lidar-frame-id", default="lidar_link")
    parser.add_argument(
        "--lidar-full-scan",
        type=boolean,
        default=True,
        help="publish one accumulated scan instead of partial data each render frame",
    )
    parser.add_argument(
        "--lidar-motion-bvh",
        type=boolean,
        default=False,
        help="enable RTX Motion BVH for intra-scan motion effects (uses much more VRAM)",
    )
    parser.add_argument("--lidar-debug-view", type=boolean, default=False)
    add_front_lidar_arguments(parser)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    validate_front_lidar_arguments(args)
    validate_physics_arguments(args)
    xacro = Path(args.xacro).expanduser().resolve()
    if not xacro.is_file():
        raise FileNotFoundError(f"Xacro file does not exist: {xacro}")
    args.xacro = str(xacro)
    if os.path.sep in args.xacro_executable:
        executable = Path(args.xacro_executable).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"xacro executable does not exist: {executable}")
        args.xacro_executable = str(executable)
    elif shutil.which(args.xacro_executable) is None:
        raise FileNotFoundError(f"cannot find xacro executable: {args.xacro_executable}")
    if args.stage:
        stage = Path(args.stage).expanduser().resolve()
        if not stage.is_file():
            raise FileNotFoundError(f"environment USD does not exist: {stage}")
        args.stage = str(stage)
    if args.physics_hz <= 0.0:
        raise ValueError("physics-hz must be positive")
    if args.max_steps < 0:
        raise ValueError("max-steps must not be negative")


def expand_xacro(args: argparse.Namespace, output: Path) -> None:
    command = [
        args.xacro_executable,
        args.xacro,
        "sim_backend:=isaac",
        "-o",
        str(output),
    ]
    environment = os.environ.copy()
    xacro_pythonpath = environment.pop("JVS_XACRO_PYTHONPATH", "")
    if xacro_pythonpath:
        environment["PYTHONPATH"] = xacro_pythonpath
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"xacro expansion failed: {detail}")


def enable_extensions(simulation_app) -> None:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in (
        "isaacsim.asset.importer.urdf",
        "isaacsim.ros2.bridge",
        "isaacsim.sensors.rtx",
    ):
        manager.set_extension_enabled_immediate(extension, True)
    for _ in range(3):
        simulation_app.update()


def prepare_stage(args: argparse.Namespace, simulation_app):
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux

    context = omni.usd.get_context()
    if args.stage:
        if not context.open_stage(args.stage):
            raise RuntimeError(f"Isaac Sim could not open environment stage: {args.stage}")
    else:
        context.new_stage()
    simulation_app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not create a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if not stage.GetPrimAtPath("/World"):
        UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    if args.add_ground_plane and not stage.GetPrimAtPath("/World/groundPlane"):
        PhysicsSchemaTools.addGroundPlane(
            stage,
            "/World/groundPlane",
            "Z",
            50.0,
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(0.35, 0.35, 0.35),
        )
    if not stage.GetPrimAtPath("/World/DistantLight"):
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
        light.CreateIntensityAttr(500.0)
    if args.obstacles:
        from pxr import UsdPhysics, Vt
        import json
        from jvs_mock.kc_svsdf import validate_scene
        scene = validate_scene(json.loads(Path(args.obstacles).read_text()))
        colors = {
            'sphere': Gf.Vec3f(.80, .20, .15),
            'cylinder': Gf.Vec3f(.18, .48, .76),
            'cube': Gf.Vec3f(.85, .55, .12),
            'triangular_pyramid': Gf.Vec3f(.42, .68, .28),
        }
        for index, obstacle in enumerate(scene['obstacles']):
            path = f'/World/WBCObstacle{index}'
            kind = obstacle['type']
            center = obstacle['center']
            if kind == 'sphere':
                shape = UsdGeom.Sphere.Define(stage, path)
                shape.CreateRadiusAttr(float(obstacle['radius']))
                shape.AddTranslateOp().Set(Gf.Vec3d(*center))
            elif kind == 'cylinder':
                shape = UsdGeom.Cylinder.Define(stage, path)
                shape.CreateAxisAttr(UsdGeom.Tokens.z)
                shape.CreateRadiusAttr(float(obstacle['radius']))
                shape.CreateHeightAttr(float(obstacle['height']))
                shape.AddTranslateOp().Set(Gf.Vec3d(*center))
            elif kind == 'cube':
                shape = UsdGeom.Cube.Define(stage, path)
                shape.CreateSizeAttr(float(obstacle['size']))
                shape.AddTranslateOp().Set(Gf.Vec3d(*center))
            else:
                sx, sy, sz = map(float, obstacle['size'])
                yaw = math.radians(float(obstacle.get('yaw_deg', 0.)))
                cosine, sine = math.cos(yaw), math.sin(yaw)
                local = [(-sx/2, -sy/2, -sz/2), (sx/2, -sy/2, -sz/2),
                         (0., sy/2, -sz/2), (0., 0., sz/2)]
                points = [Gf.Vec3f(center[0]+cosine*x-sine*y,
                                   center[1]+sine*x+cosine*y, center[2]+z)
                          for x,y,z in local]
                shape = UsdGeom.Mesh.Define(stage, path)
                shape.CreatePointsAttr(Vt.Vec3fArray(points))
                shape.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3, 3, 3]))
                shape.CreateFaceVertexIndicesAttr(Vt.IntArray(
                    [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]))
                shape.CreateDoubleSidedAttr(True)
            shape.CreateDisplayColorAttr([colors[kind]])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            if kind == 'triangular_pyramid':
                UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr().Set(
                    'convexHull')
    return stage


def import_robot(urdf_path: Path) -> str:
    import omni.kit.commands
    from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

    ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not ok:
        raise RuntimeError("Isaac Sim could not create an URDF import configuration")
    config.merge_fixed_joints = False
    config.fix_base = False
    config.make_default_prim = False
    config.create_physics_scene = True
    config.import_inertia_tensor = True
    config.distance_scale = 1.0
    # The Isaac 5.x binding requires its enum rather than the legacy integer.
    # Wheel drives are changed to velocity control after the import.
    config.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_POSITION
    config.default_position_drive_damping = 100.0
    ok, robot_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=config,
        get_articulation_root=True,
    )
    if not ok or not robot_path:
        raise RuntimeError("Isaac Sim failed to import the JVS URDF")
    return str(robot_path)


def _joint_prims(stage) -> dict[str, object]:
    wanted = set(WHEEL_JOINTS + ARM_JOINTS + CASTER_JOINTS)
    found = {prim.GetName(): prim for prim in stage.Traverse() if prim.GetName() in wanted}
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"imported articulation is missing joints: {sorted(missing)}")
    return found


def configure_robot_spawn(stage, robot_path: str, args: argparse.Namespace) -> None:
    """Place the complete imported robot before PhysX builds its constraints.

    The URDF importer puts the link bodies alongside the articulation root
    (base_link), not beneath it. Moving only that body tears the initial joint
    frames apart; the first physics steps then try to repair the assembly.
    Author the spawn on their common Xform so every link moves together.
    """
    from pxr import Gf, UsdGeom

    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim:
        raise RuntimeError(f"invalid imported articulation root: {robot_path}")
    robot_scope = robot_prim.GetParent()
    if robot_scope.IsPseudoRoot() or not robot_scope.IsA(UsdGeom.Xform):
        raise RuntimeError(f"imported robot has no enclosing Xform: {robot_path}")

    def set_pose(prim, position, orientation):
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        for property_name in tuple(prim.GetPropertyNames()):
            if property_name.startswith("xformOp:"):
                prim.RemoveProperty(property_name)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(position)
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(orientation)
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1.0))

    # Retain float ops on the mobile root: PhysX writes its live pose as floats.
    # Its local URDF rest pose is identity; the enclosing Xform owns the spawn.
    set_pose(robot_prim, Gf.Vec3f(0.0), Gf.Quatf(1.0))
    set_pose(
        robot_scope,
        Gf.Vec3f(args.spawn_x, args.spawn_y, args.spawn_z),
        Gf.Quatf(
            math.cos(args.spawn_yaw * 0.5),
            Gf.Vec3f(0.0, 0.0, math.sin(args.spawn_yaw * 0.5)),
        ),
    )


def configure_articulation(stage, robot_path: str, args: argparse.Namespace, urdf_path: Path) -> None:
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    configure_robot_spawn(stage, robot_path, args)
    robot_prim = stage.GetPrimAtPath(robot_path)

    articulation = PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
    articulation.CreateSolverPositionIterationCountAttr(32)
    articulation.CreateSolverVelocityIterationCountAttr(getattr(args, 'solver_velocity_iterations', 16))

    # The fixed spherical supports must slide freely.  Without an explicit
    # material PhysX gives them its default friction, which locks a differential
    # base even though the wheel drives are active.
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/JvsPhysicsMaterials"))

    def physics_material(name: str, friction: float):
        material = UsdShade.Material.Define(
            stage, Sdf.Path(f"/World/JvsPhysicsMaterials/{name}")
        )
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr().Set(friction)
        material_api.CreateDynamicFrictionAttr().Set(friction)
        material_api.CreateRestitutionAttr().Set(0.0)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr().Set("min")
        return material

    wheel_material = physics_material("Wheels", 1.0)
    caster_material = physics_material("Casters", 0.02)
    ground_prim = stage.GetPrimAtPath("/World/groundPlane")
    if args.add_ground_plane and ground_prim:
        UsdShade.MaterialBindingAPI.Apply(ground_prim).Bind(
            wheel_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
    robot_root = Sdf.Path(robot_path).GetParentPath()
    for link_names, material in (
        (WHEEL_LINKS, wheel_material),
        (CASTER_LINKS, caster_material),
    ):
        for link_name in link_names:
            collision_prim = stage.GetPrimAtPath(
                robot_root.AppendChild(link_name).AppendChild("collisions")
            )
            if not collision_prim:
                raise RuntimeError(f"missing collision geometry for {link_name}")
            UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )

    joints = _joint_prims(stage)
    limits = {
        joint.get("name"): float(joint.find("limit").get("effort"))
        for joint in ET.parse(urdf_path).getroot().findall("joint")
        if joint.find("limit") is not None
    }
    for name, prim in joints.items():
        if name not in CASTER_JOINTS and getattr(args, 'actuator_armature', 0.) > 0:
            # Optional reflected actuator inertia (kg*m^2); never modify passive casters.
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr().Set(args.actuator_armature)
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        if name in CASTER_JOINTS:
            # Isaac's URDF importer gives every movable joint a position drive.
            # A zero-force drive leaves caster swivel/rolling joints passive.
            drive.CreateMaxForceAttr().Set(0.0)
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)
            drive.CreateTargetVelocityAttr().Set(0.0)
        elif name in WHEEL_JOINTS:
            drive.CreateMaxForceAttr().Set(limits[name])
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(angular_drive_gain_to_usd(getattr(args, 'wheel_drive_damping', 20.)))
            drive.CreateTargetVelocityAttr().Set(0.0)
        else:
            # Preserve the URDF torque limits. Servo gains are expressed in
            # SI radians, then converted for USD's degree-based angular drive.
            drive.CreateMaxForceAttr().Set(limits[name])
            drive.CreateTypeAttr().Set(getattr(args, 'arm_drive_type', 'force'))
            drive.CreateStiffnessAttr(angular_drive_gain_to_usd(getattr(args, 'arm_drive_stiffness', 200.)))
            drive.CreateDampingAttr(angular_drive_gain_to_usd(getattr(args, 'arm_drive_damping', 1000.)))
            drive.CreateTargetPositionAttr().Set(0.0)
            drive.CreateTargetVelocityAttr().Set(0.0)


def configure_physics_rate(stage, physics_hz: float, solver_type: str = 'PGS') -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise RuntimeError("URDF import did not create a physics scene")
    scene = UsdPhysics.Scene(scenes[0])
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(scenes[0]).CreateTimeStepsPerSecondAttr(
        int(round(physics_hz))
    )
    PhysxSchema.PhysxSceneAPI.Apply(scenes[0]).CreateSolverTypeAttr().Set(solver_type)


def register_lidar_profile(config_name: str) -> bool:
    """Register the packaged JVS profile and report whether it is custom."""
    if config_name != DEFAULT_LIDAR_CONFIG:
        return False

    import carb.settings

    profile = Path(__file__).resolve().parent.parent / "config" / f"{config_name}.json"
    if not profile.is_file():
        raise FileNotFoundError(f"packaged RTX LiDAR profile does not exist: {profile}")

    settings = carb.settings.get_settings()
    profile_dirs = list(settings.get(LIDAR_PROFILE_SETTING) or [])
    profile_dir = str(profile.parent)
    if profile_dir not in profile_dirs:
        settings.set_string_array(LIDAR_PROFILE_SETTING, [profile_dir, *profile_dirs])
    return True


def create_rtx_lidar(stage, robot_path: str, args: argparse.Namespace,
                     link_name: str = "lidar_link") -> str:
    """Mount one RTX LiDAR at its URDF frame, without additional rotation."""
    import omni.kit.commands
    from pxr import Sdf

    robot_scope = Sdf.Path(robot_path).GetParentPath()
    lidar_link_path = robot_scope.AppendChild(link_name)
    lidar_link = stage.GetPrimAtPath(lidar_link_path)
    if not lidar_link or not lidar_link.IsValid():
        raise RuntimeError(f"imported robot is missing {link_name}: {lidar_link_path}")

    custom_profile = register_lidar_profile(args.lidar_config)
    command_args = {
        "path": "/JvsRtxLidar",
        "parent": str(lidar_link_path),
        "config": args.lidar_config,
        "visibility": False,
    }
    if custom_profile:
        # Custom JSON profiles use Isaac's camera-backed RTX LiDAR path. The
        # profile search directory above makes this independent of the Isaac
        # installation and avoids copying files into it.
        command_args["force_camera_prim"] = True
    if args.lidar_variant:
        command_args["variant"] = args.lidar_variant
    ok, sensor = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar", **command_args
    )
    if not ok or sensor is None or not sensor.IsValid():
        raise RuntimeError(
            f"Isaac Sim failed to create RTX LiDAR config {args.lidar_config!r}"
        )
    expected_type = "Camera" if custom_profile else "OmniLidar"
    if sensor.GetTypeName() != expected_type:
        raise RuntimeError(
            f"RTX LiDAR command returned {sensor.GetTypeName()!r}, "
            f"expected {expected_type!r}"
        )
    return str(sensor.GetPath())


def create_front_rtx_lidar(stage, robot_path, args):
    if "front" not in enabled_lidars(args):
        return ""
    front_args = argparse.Namespace(**vars(args))
    front_args.lidar_config = args.front_lidar_config
    front_args.lidar_variant = ""
    return create_rtx_lidar(stage, robot_path, front_args, "front_lidar_link")


def create_ros_graph(
    robot_path: str,
    lidar_path: str = "",
    lidar_topic: str = "/lidar/points",
    lidar_frame_id: str = "lidar_link",
    lidar_full_scan: bool = True,
    lidar_debug_view: bool = False,
    front_lidar_path: str = "",
    front_lidar_topic: str = "/lidar/front/points",
    filtered_commands: bool = False,
    physics_feedback: bool = False,
) -> None:
    import omni.graph.core as og
    import usdrt
    from pxr import Sdf

    target = [usdrt.Sdf.Path(robot_path)]
    keys = og.Controller.Keys
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
        ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ("PublishOdomTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
        ("SubscribeCommand", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("Controller", "isaacsim.core.nodes.IsaacArticulationController"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ComputeOdom.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishOdom.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishOdomTF.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "SubscribeCommand.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "Controller.inputs:execIn"),
        ("Context.outputs:context", "PublishClock.inputs:context"),
        ("Context.outputs:context", "PublishJointState.inputs:context"),
        ("Context.outputs:context", "PublishTF.inputs:context"),
        ("Context.outputs:context", "PublishOdom.inputs:context"),
        ("Context.outputs:context", "PublishOdomTF.inputs:context"),
        ("Context.outputs:context", "SubscribeCommand.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishOdomTF.inputs:timeStamp"),
        ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
        ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
        ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
        ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
        ("ComputeOdom.outputs:position", "PublishOdomTF.inputs:translation"),
        ("ComputeOdom.outputs:orientation", "PublishOdomTF.inputs:rotation"),
        ("SubscribeCommand.outputs:jointNames", "Controller.inputs:jointNames"),
        ("SubscribeCommand.outputs:positionCommand", "Controller.inputs:positionCommand"),
        ("SubscribeCommand.outputs:velocityCommand", "Controller.inputs:velocityCommand"),
    ]
    values = [
        ("PublishJointState.inputs:topicName", "joint_states"),
        ("PublishJointState.inputs:targetPrim", target),
        ("PublishTF.inputs:topicName", "tf"),
        ("PublishTF.inputs:targetPrims", target),
        # Odom owns the base_link parent. Publishing world -> base_link here
        # as well produces a nondeterministically disconnected TF tree.
        ("PublishTF.inputs:parentPrim", target),
        ("ComputeOdom.inputs:chassisPrim", target),
        ("PublishOdom.inputs:topicName", "base_controller/odom"),
        ("PublishOdom.inputs:odomFrameId", "odom"),
        ("PublishOdom.inputs:chassisFrameId", "base_link"),
        ("PublishOdomTF.inputs:topicName", "tf"),
        ("PublishOdomTF.inputs:parentFrameId", "odom"),
        ("PublishOdomTF.inputs:childFrameId", "base_link"),
        ("SubscribeCommand.inputs:topicName", "isaac_joint_commands"),
        ("Controller.inputs:robotPath", robot_path),
    ]
    if filtered_commands:
        excluded = {'SubscribeCommand', 'Controller'}
        nodes = [(name, kind) for name, kind in nodes if name not in excluded]
        connections = [(a, b) for a, b in connections
                       if a.split('.')[0] not in excluded and b.split('.')[0] not in excluded]
        values = [(key, value) for key, value in values if key.split('.')[0] not in excluded]
    if physics_feedback:
        excluded = {'PublishClock', 'PublishJointState', 'ComputeOdom', 'PublishOdom', 'PublishOdomTF'}
        nodes = [(name, kind) for name, kind in nodes if name not in excluded]
        connections = [(a, b) for a, b in connections
                       if a.split('.')[0] not in excluded and b.split('.')[0] not in excluded]
        values = [(key, value) for key, value in values if key.split('.')[0] not in excluded]
    for index, (sensor_path, topic, frame) in enumerate([
        (lidar_path, lidar_topic, lidar_frame_id),
        (front_lidar_path, front_lidar_topic, "front_lidar_link"),
    ]):
        if not sensor_path:
            continue
        start_nodes, start_connections, start_values = len(nodes), len(connections), len(values)
        nodes.extend(
            [
                (
                    "RunOneSimulationFrame",
                    "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame",
                ),
                (
                    "CreateLidarRenderProduct",
                    "isaacsim.core.nodes.IsaacCreateRenderProduct",
                ),
                (
                    "PublishLidarPointCloud",
                    "isaacsim.ros2.bridge.ROS2RtxLidarHelper",
                ),
            ]
        )
        connections.extend(
            [
                (
                    "OnPlaybackTick.outputs:tick",
                    "RunOneSimulationFrame.inputs:execIn",
                ),
                (
                    "RunOneSimulationFrame.outputs:step",
                    "CreateLidarRenderProduct.inputs:execIn",
                ),
                (
                    "CreateLidarRenderProduct.outputs:execOut",
                    "PublishLidarPointCloud.inputs:execIn",
                ),
                (
                    "CreateLidarRenderProduct.outputs:renderProductPath",
                    "PublishLidarPointCloud.inputs:renderProductPath",
                ),
                (
                    "Context.outputs:context",
                    "PublishLidarPointCloud.inputs:context",
                ),
            ]
        )
        values.extend(
            [
                (
                    "CreateLidarRenderProduct.inputs:cameraPrim",
                    [Sdf.Path(sensor_path)],
                ),
                # RTX LiDAR output density comes from its sensor profile, not
                # image resolution.  A small render product saves substantial
                # VRAM in large environments such as Kujiale.
                ("CreateLidarRenderProduct.inputs:width", 128),
                ("CreateLidarRenderProduct.inputs:height", 128),
                ("PublishLidarPointCloud.inputs:topicName", topic),
                ("PublishLidarPointCloud.inputs:type", "point_cloud"),
                ("PublishLidarPointCloud.inputs:frameId", frame),
                ("PublishLidarPointCloud.inputs:fullScan", lidar_full_scan),
                (
                    "PublishLidarPointCloud.inputs:showDebugView",
                    lidar_debug_view,
                ),
                ("PublishLidarPointCloud.inputs:useSystemTime", False),
                (
                    "PublishLidarPointCloud.inputs:resetSimulationTimeOnStop",
                    True,
                ),
            ]
        )
        if index:
            # Each sensor needs its own render product/helper, but shares the
            # ROS context and simulation clock with the rest of the robot.
            names = {name for name, _ in nodes[start_nodes:]}
            def rename(value):
                if isinstance(value, str) and value.split('.')[0] in names:
                    return "Front" + value
                return value
            nodes[start_nodes:] = [(rename(n), t) for n, t in nodes[start_nodes:]]
            connections[start_connections:] = [(rename(a), rename(b))
                for a, b in connections[start_connections:]]
            values[start_values:] = [(rename(k), v) for k, v in values[start_values:]]
    og.Controller.edit(
        {"graph_path": "/World/JvsRosGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: nodes,
            keys.CONNECT: connections,
            keys.SET_VALUES: values,
        },
    )


def run(args: argparse.Namespace) -> int:
    # The bridge selects its ROS ABI while Kit starts, so these must exist first.
    os.environ.setdefault("ROS_DISTRO", "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "enable_motion_bvh": args.enable_lidar and args.lidar_motion_bvh,
        }
    )
    reset_server = None
    try:
        print("JVS Isaac Sim: enabling extensions", flush=True)
        enable_extensions(simulation_app)
        print("JVS Isaac Sim: extensions ready", flush=True)
        with tempfile.TemporaryDirectory(prefix="jvs_isaac_") as folder:
            urdf_path = Path(folder) / "jvs_mock.urdf"
            expand_xacro(args, urdf_path)
            print(f"JVS Isaac Sim: expanded {urdf_path.name}", flush=True)
            stage = prepare_stage(args, simulation_app)
            print("JVS Isaac Sim: USD stage ready", flush=True)
            robot_path = import_robot(urdf_path)
            simulation_app.update()
            print(f"JVS Isaac Sim: imported articulation {robot_path}", flush=True)
            configure_articulation(stage, robot_path, args, urdf_path)
            configure_physics_rate(stage, args.physics_hz, args.solver_type)
            lidar_path = ""
            front_lidar_path = ""
            if args.enable_lidar:
                if "rear" in enabled_lidars(args):
                    lidar_path = create_rtx_lidar(stage, robot_path, args)
                front_lidar_path = create_front_rtx_lidar(stage, robot_path, args)
                simulation_app.update()
                print(
                    "JVS Isaac Sim: RTX LiDAR created at "
                    f"rear={lidar_path or 'disabled'} (config={args.lidar_config}); "
                    f"front={front_lidar_path or 'disabled'} (config={args.front_lidar_config})",
                    flush=True,
                )
            create_ros_graph(
                robot_path,
                lidar_path=lidar_path,
                lidar_topic=args.lidar_topic,
                lidar_frame_id=args.lidar_frame_id,
                lidar_full_scan=args.lidar_full_scan,
                lidar_debug_view=args.lidar_debug_view,
                front_lidar_path=front_lidar_path,
                front_lidar_topic=args.front_lidar_topic,
                filtered_commands=args.command_filter_hz > 0 or args.physics_feedback_hz > 0,
                physics_feedback=args.physics_feedback_hz > 0,
            )
            print("JVS Isaac Sim: ROS graph ready", flush=True)
            if args.save_stage:
                output = Path(args.save_stage).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                stage.GetRootLayer().Export(str(output))

            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            if args.command_filter_hz > 0 or args.physics_feedback_hz > 0:
                from isaacsim.core.simulation_manager import SimulationManager
                SimulationManager.initialize_physics()
                reset_server = RobotResetService(robot_path,
                    (args.spawn_x, args.spawn_y, args.spawn_z, args.spawn_yaw),
                    dict(x=args.spawn_x, y=args.spawn_y, z=args.spawn_z, yaw=args.spawn_yaw), '/jvs/reset_robot')
                reset_server.hold_startup_pose()
                if args.physics_feedback_hz > 0:
                    PhysicsFeedback(reset_server, args.physics_feedback_hz)
                if args.command_filter_hz > 0 or args.physics_feedback_hz > 0:
                    PhysicsCommandFilter(reset_server, urdf_path, args.command_filter_hz)
            print(
                "JVS Isaac Sim ready: "
                f"articulation={robot_path}, command_topic=/isaac_joint_commands, "
                f"lidar_topic={args.lidar_topic if lidar_path else 'disabled'}, "
                f"front_lidar_topic={args.front_lidar_topic if front_lidar_path else 'disabled'}",
                flush=True,
            )
            steps = 0
            while simulation_app.is_running() and (
                args.max_steps == 0 or steps < args.max_steps
            ):
                simulation_app.update()
                if reset_server is not None:
                    reset_server.spin_once()
                steps += 1
            timeline.stop()
        return 0
    except Exception as error:
        print(
            f"JVS Isaac Sim failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if reset_server is not None:
            reset_server.close()
        simulation_app.close()


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    # SimulationApp forwards unknown process arguments to Kit.  Keep runner
    # options such as --max-steps private to this script so Kit does not
    # interpret them as its own lifecycle flags and stop during URDF import.
    sys.argv[:] = [sys.argv[0]]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
