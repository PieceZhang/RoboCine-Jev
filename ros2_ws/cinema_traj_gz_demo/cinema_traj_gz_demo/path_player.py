#!/usr/bin/env python3
"""Kinematic Gazebo Fortress (gz-sim6) playback of a cinema_traj trajectory.

Substitutes for a physics stack when all you need is to SEE the retimed
trajectory fly: loads the exported samples, then teleports the static
``camera_drone`` entity along them through the world's ``set_pose`` service
(``/world/<world>/set_pose``, exposed to ROS 2 as
``ros_gz_interfaces/srv/SetEntityPose`` by the ros_gz_bridge parameter_bridge
started from the launch file; provided by the gz-sim UserCommands system).
Static entity ⇒ no gravity between ticks. This is visualization-grade
playback matching our kinematic (executable-math) view of the trajectory; a
physics/offboard test needs e.g. PX4 SITL.

Pure interpolation helpers (``interp_pose`` / ``_nlerp``) are module-level so
they unit-test on macOS without rclpy / ros_gz_interfaces (imported only at
node entry, mirroring cinema_traj_bridge.traj_server).
"""

from __future__ import annotations

import math
from bisect import bisect_right


def _nlerp(qa, qb, u: float) -> tuple:
    """Normalized-lerp between (w, x, y, z) quaternions, shortest arc."""
    if sum(x * y for x, y in zip(qa, qb)) < 0.0:
        qb = [-b for b in qb]
    out = [a + u * (b - a) for a, b in zip(qa, qb)]
    n = math.sqrt(sum(o * o for o in out)) or 1.0
    return tuple(o / n for o in out)


def interp_pose(poses: list, t: float) -> dict:
    """Pose {stamp_s, position, orientation} at time t (clamped at both ends)."""
    if not poses:
        raise ValueError("no poses")
    if t <= poses[0]["stamp_s"]:
        return poses[0]
    if t >= poses[-1]["stamp_s"]:
        return poses[-1]
    ts = [p["stamp_s"] for p in poses]
    i = min(bisect_right(ts, t) - 1, len(poses) - 2)
    a, b = poses[i], poses[i + 1]
    span = max(b["stamp_s"] - a["stamp_s"], 1e-9)
    u = (t - a["stamp_s"]) / span
    pos = tuple(pa + u * (pb - pa) for pa, pb in zip(a["position"], b["position"]))
    return {"stamp_s": t, "position": pos,
            "orientation": _nlerp(a["orientation"], b["orientation"], u)}


class PathPlayer:                                   # rclpy-free core, bound in main
    def __init__(self, node, poses: list, entity: str, world: str,
                 time_scale: float, loop: bool, rate_hz: float):
        from ros_gz_interfaces.srv import SetEntityPose
        self._node, self._poses, self._entity = node, poses, entity
        self._scale, self._loop = max(time_scale, 1e-3), loop
        self._duration = max(poses[-1]["stamp_s"] - poses[0]["stamp_s"], 1e-6)
        self._cli = node.create_client(SetEntityPose, f"/world/{world}/set_pose")
        if not self._cli.wait_for_service(timeout_sec=60.0):
            raise RuntimeError(
                f"/world/{world}/set_pose not available — is the ros_gz_bridge "
                "parameter_bridge running and the Fortress world loaded with "
                "the UserCommands system?")
        self._t0 = None
        node.create_timer(1.0 / max(rate_hz, 1.0), self._tick)
        node.get_logger().info(
            f"playing {len(poses)} poses ({self._duration:.1f}s) on entity "
            f"'{entity}' via /world/{world}/set_pose, time_scale={time_scale}, "
            f"loop={loop}")

    def _tick(self) -> None:
        from geometry_msgs.msg import Point, Pose, Quaternion
        from ros_gz_interfaces.msg import Entity
        from ros_gz_interfaces.srv import SetEntityPose
        now = self._node.get_clock().now().nanoseconds * 1e-9
        if self._t0 is None:
            self._t0 = now
        t = (now - self._t0) * self._scale
        if self._loop:
            t = t % self._duration
        p = interp_pose(self._poses, t)
        req = SetEntityPose.Request()
        req.entity = Entity(name=self._entity, type=Entity.MODEL)   # id=0 → name
        req.pose = Pose(
            position=Point(x=p["position"][0], y=p["position"][1], z=p["position"][2]),
            orientation=Quaternion(
                x=p["orientation"][1], y=p["orientation"][2],
                z=p["orientation"][3], w=p["orientation"][0]))
        self._cli.call_async(req)


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node

    from cinema_traj_bridge import traj_loader

    rclpy.init(args=args)
    node = Node("path_player")
    node.declare_parameter("traj_file", "")
    node.declare_parameter("entity", "camera_drone")
    node.declare_parameter("world", "demo_room")
    node.declare_parameter("time_scale", 10.0)
    node.declare_parameter("loop", False)
    node.declare_parameter("rate_hz", 30.0)

    traj_file = str(node.get_parameter("traj_file").value)
    if not traj_file:
        raise RuntimeError("traj_file parameter is required")
    samples = traj_loader.load_samples(traj_file)
    poses = traj_loader.build_path_data(samples, "map", 0.0)["poses"]
    player = PathPlayer(
        node, poses,
        entity=str(node.get_parameter("entity").value),
        world=str(node.get_parameter("world").value),
        time_scale=float(node.get_parameter("time_scale").value),
        loop=bool(node.get_parameter("loop").value),
        rate_hz=float(node.get_parameter("rate_hz").value))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        _ = player


if __name__ == "__main__":
    main()
