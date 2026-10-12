"""Continuously publish a fixed velocity command to the mobile base."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class BaseVelocityCommander(Node):
    def __init__(self) -> None:
        super().__init__("base_velocity_commander")
        self.declare_parameter("command_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("linear_velocity", 0.10)
        self.declare_parameter("angular_velocity", 0.15)

        topic = str(self.get_parameter("command_topic").value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        self._linear_velocity = float(
            self.get_parameter("linear_velocity").value
        )
        self._angular_velocity = float(
            self.get_parameter("angular_velocity").value
        )
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")

        self._publisher = self.create_publisher(Twist, topic, 10)
        self.create_timer(1.0 / rate_hz, self._publish_command)
        self.get_logger().info(
            f"Publishing velocity commands to {topic} at {rate_hz:.1f} Hz "
            f"(linear.x={self._linear_velocity:.3f} m/s, "
            f"angular.z={self._angular_velocity:.3f} rad/s)"
        )

    def _publish_command(self) -> None:
        msg = Twist()
        msg.linear.x = self._linear_velocity
        msg.angular.z = self._angular_velocity
        self._publisher.publish(msg)

    def publish_stop(self) -> None:
        self._publisher.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BaseVelocityCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_stop()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # A launch service can forward a second SIGINT during cleanup.
            pass


if __name__ == "__main__":
    main()
