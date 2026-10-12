"""Reset Isaac's JVS robot and verify the startup state before control begins."""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from .isaac_bridge import ARM_JOINT_NAMES, WHEEL_JOINT_NAMES


START_ARM_POSITIONS = np.zeros(len(ARM_JOINT_NAMES), dtype=float)


def _yaw_from_quaternion(quaternion) -> float:
    x, y, z, w = (
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    )
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class IsaacResetClient(Node):
    """Own the reset/verification phase and its zero-command publisher."""

    def __init__(self) -> None:
        super().__init__("jvs_isaac_reset_client")
        self.declare_parameter("reset_service", "/jvs/reset_robot")
        self.declare_parameter("timeout_s", 30.0)
        self.declare_parameter("position_tolerance_rad", 5.0e-3)
        self.declare_parameter("joint_velocity_tolerance_radps", 5.0e-2)
        self.declare_parameter("base_velocity_tolerance", 5.0e-3)
        self.declare_parameter("stable_samples", 5)
        self.declare_parameter("expected_scene_pose", "")

        service_name = str(self.get_parameter("reset_service").value)
        self.timeout = float(self.get_parameter("timeout_s").value)
        self.position_tolerance = float(
            self.get_parameter("position_tolerance_rad").value
        )
        self.joint_velocity_tolerance = float(
            self.get_parameter("joint_velocity_tolerance_radps").value
        )
        self.base_velocity_tolerance = float(
            self.get_parameter("base_velocity_tolerance").value
        )
        self.stable_samples = int(self.get_parameter("stable_samples").value)
        tolerances = (
            self.position_tolerance,
            self.joint_velocity_tolerance,
            self.base_velocity_tolerance,
        )
        if not service_name or self.timeout <= 0.0 or min(tolerances) <= 0.0:
            raise ValueError("invalid reset service, timeout, or feedback tolerances")
        if self.stable_samples < 1:
            raise ValueError("stable_samples must be positive")

        self.client = self.create_client(Trigger, service_name)
        self.command_publisher = self.create_publisher(
            JointState, "/isaac_joint_commands", 1
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joints,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            "/base_controller/odom",
            self._on_odom,
            qos_profile_sensor_data,
        )
        self.joints = None
        self.odom = None
        self.joint_generation = 0
        self.odom_generation = 0

    def _on_joints(self, message: JointState) -> None:
        names = list(WHEEL_JOINT_NAMES + ARM_JOINT_NAMES)
        if not all(name in message.name for name in names):
            return
        indices = [message.name.index(name) for name in names]
        if len(message.position) <= max(indices) or len(message.velocity) <= max(indices):
            return
        position = np.asarray(message.position, dtype=float)[indices]
        velocity = np.asarray(message.velocity, dtype=float)[indices]
        if np.all(np.isfinite(position)) and np.all(np.isfinite(velocity)):
            self.joints = position, velocity
            self.joint_generation += 1

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
        values = np.asarray([
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ])
        if np.all(np.isfinite(values)):
            self.odom = message
            self.odom_generation += 1

    def _publish_hold(self) -> None:
        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = list(WHEEL_JOINT_NAMES + ARM_JOINT_NAMES)
        command.position = [0.0, 0.0, *START_ARM_POSITIONS.tolist()]
        command.velocity = [0.0] * len(command.name)
        self.command_publisher.publish(command)

    def _feedback_matches(self) -> bool:
        if self.joints is None or self.odom is None:
            return False
        joint_position, joint_velocity = self.joints
        arm_ok = (
            np.max(np.abs(joint_position[2:] - START_ARM_POSITIONS))
            <= self.position_tolerance
        )
        joint_velocity_ok = (
            np.max(np.abs(joint_velocity)) <= self.joint_velocity_tolerance
        )
        twist = self.odom.twist.twist
        base_velocity = np.asarray([
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ])
        return bool(
            arm_ok
            and joint_velocity_ok
            and np.max(np.abs(base_velocity)) <= self.base_velocity_tolerance
        )

    def reset_and_verify(self) -> None:
        deadline = time.monotonic() + self.timeout
        self.get_logger().info("Waiting for Isaac reset service")
        while rclpy.ok() and time.monotonic() < deadline:
            self._publish_hold()
            if self.client.wait_for_service(timeout_sec=0.1):
                break
            rclpy.spin_once(self, timeout_sec=0.0)
        else:
            raise TimeoutError("Isaac reset service was not available")

        self.get_logger().info("Requesting Isaac robot startup-state reset")
        future = self.client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            self._publish_hold()
            rclpy.spin_once(self, timeout_sec=0.02)
        if not future.done():
            raise TimeoutError("Isaac reset service did not respond")
        response = future.result()
        if response is None or not response.success:
            detail = "empty response" if response is None else response.message
            raise RuntimeError(f"Isaac reset failed: {detail}")
        try:
            reset_state = json.loads(response.message)
        except json.JSONDecodeError as error:
            raise RuntimeError("Isaac reset returned invalid state JSON") from error
        expected_json = self.get_parameter('expected_scene_pose').value
        if expected_json:
            expected = json.loads(expected_json)
            actual = reset_state.get('scene_y_up', {})
            names = dict(x='x_m', z='z_m', yaw='yaw_rad', height='height_m')
            if any(abs(float(actual.get(names[k], float('inf')))-v) > 1e-5
                   for k, v in expected.items()):
                raise RuntimeError('scene spawn differs from trajectory/mesh spawn; restart '
                    'jvs_kc_wbc_isaac.launch.py with the same trajectory_file (initial_base_pose)')

        stable = 0
        checked_generation = (-1, -1)
        while rclpy.ok() and time.monotonic() < deadline:
            self._publish_hold()
            rclpy.spin_once(self, timeout_sec=0.02)
            generation = self.joint_generation, self.odom_generation
            if generation == checked_generation:
                continue
            if (
                generation[0] == checked_generation[0]
                or generation[1] == checked_generation[1]
            ):
                continue
            checked_generation = generation
            stable = stable + 1 if self._feedback_matches() else 0
            if stable >= self.stable_samples:
                break
        else:
            raise TimeoutError("reset feedback did not converge to the startup state")

        joint_position, joint_velocity = self.joints
        pose = self.odom.pose.pose
        twist = self.odom.twist.twist
        observed = {
            "odom_pose": {
                "x_m": pose.position.x,
                "y_m": pose.position.y,
                "z_m": pose.position.z,
                "yaw_rad": _yaw_from_quaternion(pose.orientation),
            },
            "base_twist": {
                "linear_mps": [twist.linear.x, twist.linear.y, twist.linear.z],
                "angular_radps": [
                    twist.angular.x,
                    twist.angular.y,
                    twist.angular.z,
                ],
            },
            "wheel_positions_rad": joint_position[:2].tolist(),
            "arm_positions_rad": joint_position[2:].tolist(),
            "controlled_joint_velocities_radps": joint_velocity.tolist(),
        }
        self.get_logger().info(
            "ISAAC_RESET_COMPLETE declared="
            + json.dumps(reset_state, separators=(",", ":"))
        )
        self.get_logger().info(
            "ISAAC_START_STATE observed="
            + json.dumps(observed, separators=(",", ":"))
        )
        self.get_logger().info(
            "Reset feedback verified; trajectory controller may now start"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = IsaacResetClient()
        node.reset_and_verify()
    except Exception as error:
        if node is not None:
            node.get_logger().error(f"Isaac startup-state reset failed: {error}")
        else:
            print(f"Isaac startup-state reset failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
