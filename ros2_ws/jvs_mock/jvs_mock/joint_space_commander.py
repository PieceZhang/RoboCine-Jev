"""Continuously publish a slow six-joint motion to the arm controller."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from .motion import JOINT_NAMES, sample_joint_positions


class JointSpaceCommander(Node):
    def __init__(self) -> None:
        super().__init__("joint_space_commander")
        self.declare_parameter("command_topic", "/arm_position_controller/commands")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("period_s", 10.0)
        self.declare_parameter("amplitude_scale", 1.0)

        topic = str(self.get_parameter("command_topic").value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        self._period_s = float(self.get_parameter("period_s").value)
        self._amplitude_scale = float(self.get_parameter("amplitude_scale").value)
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        # Validate user parameters before the timer begins.
        sample_joint_positions(0.0, self._period_s, self._amplitude_scale)

        self._publisher = self.create_publisher(Float64MultiArray, topic, 10)
        self._start_ns: int | None = None
        self._last_log_second = -1
        self.create_timer(1.0 / rate_hz, self._publish_command)
        self.get_logger().info(
            f"Publishing {JOINT_NAMES} to {topic} at {rate_hz:.1f} Hz "
            f"(period={self._period_s:.1f}s, amplitude_scale={self._amplitude_scale:.2f})"
        )

    def _publish_command(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self._start_ns is None:
            self._start_ns = now_ns
        elapsed_s = (now_ns - self._start_ns) * 1e-9
        command = sample_joint_positions(
            elapsed_s, self._period_s, self._amplitude_scale
        )
        msg = Float64MultiArray()
        msg.data = command
        self._publisher.publish(msg)

        whole_second = int(elapsed_s)
        if whole_second > 0 and whole_second % 5 == 0 and whole_second != self._last_log_second:
            self._last_log_second = whole_second
            values = ", ".join(f"{value:+.3f}" for value in command)
            self.get_logger().info(f"arm command [rad]: [{values}]")

    def publish_zero(self) -> None:
        msg = Float64MultiArray()
        msg.data = [0.0] * len(JOINT_NAMES)
        self._publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointSpaceCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_zero()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # A launch service can forward a second SIGINT during cleanup.
            pass


if __name__ == "__main__":
    main()
