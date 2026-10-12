"""Follow a relative 6-DoF task-space JSON trajectory with the JVS arm."""

from __future__ import annotations

from bisect import bisect_right
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .jvs_kinematics import (
    JOINT_LOWER,
    JOINT_NAMES,
    JOINT_UPPER,
    forward_kinematics,
    load_relative_trajectory,
    relative_pose_target,
    solve_ik,
)


class JvsFollowTraj(Node):
    """Capture the current pose, solve all waypoints, then play them once."""

    def __init__(self) -> None:
        super().__init__("jvs_follow_traj")
        self.declare_parameter("trajectory_file", "")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("command_topic", "/arm_position_controller/commands")
        self.declare_parameter("control_rate_hz", 0.0)
        self.declare_parameter("start_delay_s", 0.5)
        self.declare_parameter("log_commands", True)
        self.declare_parameter("ik_max_iterations", 200)
        self.declare_parameter("ik_position_tolerance", 2.0e-4)
        self.declare_parameter("ik_orientation_tolerance", 2.0e-3)
        self.declare_parameter("ik_damping", 1.0e-3)

        trajectory_file = str(self.get_parameter("trajectory_file").value)
        if not trajectory_file:
            raise RuntimeError("trajectory_file parameter is required")
        self._trajectory = load_relative_trajectory(trajectory_file)
        self._max_iterations = int(self.get_parameter("ik_max_iterations").value)
        self._position_tolerance = float(
            self.get_parameter("ik_position_tolerance").value
        )
        self._orientation_tolerance = float(
            self.get_parameter("ik_orientation_tolerance").value
        )
        self._damping = float(self.get_parameter("ik_damping").value)
        self._start_delay_s = float(self.get_parameter("start_delay_s").value)
        self._log_commands = bool(self.get_parameter("log_commands").value)
        if self._start_delay_s < 0.0:
            raise ValueError("start_delay_s cannot be negative")

        requested_rate = float(self.get_parameter("control_rate_hz").value)
        self._control_rate_hz = requested_rate or self._trajectory.sample_rate_hz
        if not math.isfinite(self._control_rate_hz) or self._control_rate_hz <= 0.0:
            raise ValueError(
                "control_rate_hz must be positive, or zero to use the JSON rate"
            )

        states_topic = str(self.get_parameter("joint_states_topic").value)
        command_topic = str(self.get_parameter("command_topic").value)
        self._publisher = self.create_publisher(Float64MultiArray, command_topic, 10)
        self._subscription = self.create_subscription(
            JointState, states_topic, self._on_joint_state, 10
        )
        self._latest_joint_positions: np.ndarray | None = None
        self._joint_commands: list[np.ndarray] | None = None
        self._sample_times = [
            sample.time_from_start for sample in self._trajectory.samples
        ]
        self._start_ns: int | None = None
        self._ready_ns: int | None = None
        self._last_wait_log_ns = 0
        self._published_command_count = 0
        self._finished = False
        self._timer = self.create_timer(1.0 / self._control_rate_hz, self._tick)

        self.get_logger().info(
            f"Loaded {len(self._trajectory.samples)} relative task-space samples "
            f"({self._trajectory.duration_s:.1f}s, {self._trajectory.sample_rate_hz:.1f} Hz) "
            f"from {self._trajectory.source}; waiting for {states_topic}"
        )

    def _on_joint_state(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in JOINT_NAMES):
            return
        candidate = np.array([positions[name] for name in JOINT_NAMES], dtype=float)
        if np.all(np.isfinite(candidate)):
            self._latest_joint_positions = candidate

    def _plan_joint_trajectory(self) -> bool:
        if self._latest_joint_positions is None:
            return False
        initial_q = self._latest_joint_positions.copy()
        outside_lower = np.any(initial_q < JOINT_LOWER - 1.0e-6)
        outside_upper = np.any(initial_q > JOINT_UPPER + 1.0e-6)
        if outside_lower or outside_upper:
            self.get_logger().error(
                "Current joint state is outside the configured URDF limits"
            )
            return False
        initial_position, initial_orientation = forward_kinematics(initial_q)
        commands: list[np.ndarray] = []
        seed = initial_q
        for index, sample in enumerate(self._trajectory.samples):
            target_position, target_orientation = relative_pose_target(
                initial_position,
                initial_orientation,
                sample.pose,
                self._trajectory.frame_id,
            )
            solution = solve_ik(
                target_position,
                target_orientation,
                seed,
                max_iterations=self._max_iterations,
                position_tolerance=self._position_tolerance,
                orientation_tolerance=self._orientation_tolerance,
                damping=self._damping,
            )
            if solution is None:
                self.get_logger().error(
                    f"IK failed at sample {index} (t={sample.time_from_start:.3f}s); "
                    "no controller commands were sent"
                )
                return False
            commands.append(solution)
            seed = solution
        self._joint_commands = commands
        self.get_logger().info(
            f"IK solved all {len(commands)} samples; playback starts after "
            f"{self._start_delay_s:.2f}s"
        )
        return True

    def _publish(self, positions: np.ndarray) -> None:
        message = Float64MultiArray()
        message.data = positions.tolist()
        self._publisher.publish(message)
        self._published_command_count += 1
        if self._log_commands:
            values = ", ".join(f"{value:.6f}" for value in message.data)
            self.get_logger().info(
                f"Sent joint command #{self._published_command_count} "
                f"(joint1..joint6, rad): [{values}]"
            )

    def _sample_joint_command(self, elapsed_s: float) -> np.ndarray:
        assert self._joint_commands is not None
        if elapsed_s <= 0.0:
            return self._joint_commands[0]
        if elapsed_s >= self._sample_times[-1]:
            return self._joint_commands[-1]
        upper = bisect_right(self._sample_times, elapsed_s)
        lower = upper - 1
        t0, t1 = self._sample_times[lower], self._sample_times[upper]
        ratio = (elapsed_s - t0) / (t1 - t0)
        return (
            (1.0 - ratio) * self._joint_commands[lower]
            + ratio * self._joint_commands[upper]
        )

    def _tick(self) -> None:
        if self._finished:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._joint_commands is None:
            if self._latest_joint_positions is None:
                if now_ns - self._last_wait_log_ns >= 5_000_000_000:
                    self._last_wait_log_ns = now_ns
                    self.get_logger().info(
                        "Still waiting for a complete six-joint JointState"
                    )
                return
            if not self._plan_joint_trajectory():
                self._finished = True
                self._timer.cancel()
                return
            self._ready_ns = now_ns

        assert self._ready_ns is not None
        if now_ns - self._ready_ns < int(self._start_delay_s * 1.0e9):
            return
        if self._start_ns is None:
            self._start_ns = now_ns
        elapsed_s = (now_ns - self._start_ns) * 1.0e-9
        self._publish(self._sample_joint_command(elapsed_s))
        if elapsed_s >= self._trajectory.duration_s:
            self._finished = True
            self._timer.cancel()
            self.get_logger().info(
                f"Trajectory complete at t={self._trajectory.duration_s:.3f}s; "
                "holding the final joint command"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JvsFollowTraj()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
