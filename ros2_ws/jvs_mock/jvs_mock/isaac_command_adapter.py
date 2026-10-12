"""Adapt the public JVS command contract to Isaac Sim bridge messages."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .isaac_bridge import (
    ARM_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    acceleration_limited_twist,
    arm_positions,
    differential_wheel_velocities,
    limited_twist,
)


class IsaacCommandAdapter(Node):
    """Preserve Gazebo-facing topics while driving Isaac's OmniGraph."""

    def __init__(self) -> None:
        super().__init__("jvs_isaac_command_adapter")
        self.declare_parameter("arm_command_topic", "/arm_position_controller/commands")
        self.declare_parameter("isaac_command_topic", "/isaac_joint_commands")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("cmd_vel_timeout_s", 0.5)
        self.declare_parameter("max_linear_velocity", 1.0)
        self.declare_parameter("max_angular_velocity", 3.0)
        self.declare_parameter("max_linear_acceleration", 1.0)
        self.declare_parameter("max_angular_acceleration", 7.0)
        self.declare_parameter("wheel_radius", 0.07)
        self.declare_parameter("wheel_separation", 0.21)

        arm_topic = str(self.get_parameter("arm_command_topic").value)
        isaac_command_topic = str(self.get_parameter("isaac_command_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._timeout_s = float(self.get_parameter("cmd_vel_timeout_s").value)
        self._max_linear = float(self.get_parameter("max_linear_velocity").value)
        self._max_angular = float(self.get_parameter("max_angular_velocity").value)
        self._max_linear_acceleration = float(self.get_parameter("max_linear_acceleration").value)
        self._max_angular_acceleration = float(self.get_parameter("max_angular_acceleration").value)
        acceleration_limited_twist(
            (0.0, 0.0), (0.0, 0.0), 0.0,
            self._max_linear_acceleration, self._max_angular_acceleration,
        )
        self._wheel_radius = float(self.get_parameter("wheel_radius").value)
        self._wheel_separation = float(self.get_parameter("wheel_separation").value)
        limited_twist(0.0, 0.0, self._max_linear, self._max_angular)
        differential_wheel_velocities(
            0.0, 0.0, self._wheel_radius, self._wheel_separation
        )
        if rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self._timeout_s <= 0.0:
            raise ValueError("cmd_vel_timeout_s must be positive")

        self._command_publisher = self.create_publisher(
            JointState, isaac_command_topic, 10
        )
        self.create_subscription(Float64MultiArray, arm_topic, self._on_arm_command, 10)
        self.create_subscription(Twist, cmd_vel_topic, self._on_velocity_command, 10)

        self._linear_x = 0.0
        self._angular_z = 0.0
        self._arm_positions = [0.0] * len(ARM_JOINT_NAMES)
        self._last_velocity_ns: int | None = None
        self._last_publish_ns = self.get_clock().now().nanoseconds
        self._previous_twist = (0.0, 0.0)
        self.create_timer(1.0 / rate_hz, self._publish_command)
        self.get_logger().info(
            f"Isaac command adapter: {arm_topic} and {cmd_vel_topic} "
            f"-> {isaac_command_topic}"
        )

    def _on_arm_command(self, message: Float64MultiArray) -> None:
        try:
            positions = arm_positions(message.data)
        except ValueError as error:
            self.get_logger().error(f"Ignoring invalid arm command: {error}")
            return
        self._arm_positions = positions

    def _on_velocity_command(self, message: Twist) -> None:
        try:
            self._linear_x, self._angular_z = limited_twist(
                message.linear.x,
                message.angular.z,
                self._max_linear,
                self._max_angular,
            )
        except ValueError as error:
            self.get_logger().error(f"Replacing invalid velocity command with stop: {error}")
            self._linear_x = 0.0
            self._angular_z = 0.0
        self._last_velocity_ns = self.get_clock().now().nanoseconds

    def _command(self, linear_x: float, angular_z: float) -> JointState:
        wheel_velocities = differential_wheel_velocities(
            linear_x,
            angular_z,
            self._wheel_radius,
            self._wheel_separation,
        )
        output = JointState()
        output.header.stamp = self.get_clock().now().to_msg()
        output.name = list(WHEEL_JOINT_NAMES + ARM_JOINT_NAMES)
        # Wheel position targets are harmless because their drive stiffness is
        # zero; arm velocity targets are zero alongside the position servo.
        output.position = [0.0, 0.0, *self._arm_positions]
        output.velocity = [*wheel_velocities, *([0.0] * len(ARM_JOINT_NAMES))]
        return output

    def _publish_command(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        dt = (now_ns - self._last_publish_ns) * 1e-9
        if dt < 0.0:
            self._last_velocity_ns = None
            self._previous_twist = (0.0, 0.0)
            dt = 0.0
        self._last_publish_ns = now_ns
        stale = (
            self._last_velocity_ns is None
            or (now_ns - self._last_velocity_ns) * 1e-9 > self._timeout_s
        )
        linear_x = 0.0
        angular_z = 0.0
        if not stale:
            linear_x = self._linear_x
            angular_z = self._angular_z
        self._previous_twist = acceleration_limited_twist(
            (linear_x, angular_z), self._previous_twist, dt,
            self._max_linear_acceleration, self._max_angular_acceleration,
        )
        self._command_publisher.publish(self._command(*self._previous_twist))

    def publish_stop(self) -> None:
        self._command_publisher.publish(self._command(0.0, 0.0))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IsaacCommandAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
