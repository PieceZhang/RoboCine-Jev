"""Wait for and report the controllers expected by the simulation bringup."""

from __future__ import annotations

import time

import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node


EXPECTED_CONTROLLERS = {
    "joint_state_broadcaster",
    "base_controller",
    "arm_position_controller",
}


class ControllerStatus(Node):
    def __init__(self) -> None:
        super().__init__("controller_status")
        self.declare_parameter("controller_manager", "/controller_manager")
        self.declare_parameter("timeout_s", 90.0)
        manager = str(self.get_parameter("controller_manager").value).rstrip("/")
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.client = self.create_client(ListControllers, f"{manager}/list_controllers")

    def wait_until_active(self) -> bool:
        deadline = time.monotonic() + self.timeout_s
        self.get_logger().info("Waiting for the Gazebo controller manager...")
        while rclpy.ok() and time.monotonic() < deadline:
            if not self.client.wait_for_service(timeout_sec=1.0):
                continue
            future = self.client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.done() or future.result() is None:
                continue
            states = {item.name: item.state for item in future.result().controller}
            missing = EXPECTED_CONTROLLERS - states.keys()
            inactive = {
                name: states[name]
                for name in EXPECTED_CONTROLLERS & states.keys()
                if states[name] != "active"
            }
            if not missing and not inactive:
                self.get_logger().info(
                    "Controllers active: " + ", ".join(sorted(EXPECTED_CONTROLLERS))
                )
                return True
            self.get_logger().info(
                f"Controller startup in progress; missing={sorted(missing)}, "
                f"not_active={inactive}"
            )
            time.sleep(1.0)
        self.get_logger().error("Timed out waiting for all controllers to become active")
        return False


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ControllerStatus()
    try:
        return 0 if node.wait_until_active() else 1
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

