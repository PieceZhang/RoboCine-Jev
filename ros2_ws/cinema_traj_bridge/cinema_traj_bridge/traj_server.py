#!/usr/bin/env python3
"""cinema_traj_bridge — serve an exported camera trajectory over ROS 2.

Main interface: latched ``nav_msgs/msg/Path`` on ``/cam_desired_traj`` —
late-joining subscribers receive the whole path (transient_local + reliable).
Also publishes an RViz ``visualization_msgs/msg/MarkerArray`` on
``~/path_markers`` (camera path + subject capsule, debugging only) and offers
``std_srvs/srv/SetBool`` on ``~/republish`` to re-publish on demand.

All math lives in ``traj_loader`` (rclpy-free, unit-tested on macOS); rclpy is
imported only here at node entry.

    ros2 run cinema_traj_bridge traj_server --ros-args -p traj_file:=/path/to/traj_samples_retimed.json
    ros2 topic echo /cam_desired_traj --once
    ros2 service call /cinema_traj_bridge_server/republish std_srvs/srv/SetBool '{data: true}'
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from std_srvs.srv import SetBool
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion

from . import traj_loader


def _stamp(node, sec: float):
    from builtin_interfaces.msg import Time
    t = Time()
    t.sec = int(sec)
    t.nanosec = int(round((sec - int(sec)) * 1e9))
    return t


class TrajServer(Node):

    def __init__(self):
        super().__init__("cinema_traj_bridge")
        self.declare_parameter("traj_file", "")
        self.declare_parameter("scene_file", "")       # optional: subject markers
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("start_offset_s", 0.0)  # added to now() at t=0
        self.declare_parameter("republish_period", 0.0)
        self.declare_parameter("path_step", 1)         # downsample long streams
        # planner=jarvis → /cam_desired_traj orientations in the robot's
        # optical frame (Z+ forward, X+ right, Y+ down — the URDF's
        # camera_optical_frame); default keeps the FLU drone convention
        self.declare_parameter("planner", "drone")

        traj_file = str(self.get_parameter("traj_file").value)
        if not traj_file:
            raise RuntimeError("traj_file parameter is required "
                               "(--ros-args -p traj_file:=...)")
        samples = traj_loader.load_samples(traj_file)

        import json
        subject = None
        scene_file = str(self.get_parameter("scene_file").value)
        if scene_file:
            subject = json.loads(open(scene_file).read()).get("subject_track")

        frame_id = str(self.get_parameter("frame_id").value)
        start = self.get_clock().now().nanoseconds * 1e-9 \
            + float(self.get_parameter("start_offset_s").value)
        step = max(1, int(self.get_parameter("path_step").value))

        planner = str(self.get_parameter("planner").value)
        if planner not in ("drone", "jarvis"):
            raise RuntimeError(f"planner must be 'drone' or 'jarvis', got {planner!r}")
        self._path_data = traj_loader.build_path_data(samples, frame_id, start,
                                                      step, planner=planner)
        self._markers_data = traj_loader.build_markers_data(self._path_data, subject)

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self._pub_path = self.create_publisher(Path, "/cam_desired_traj", latched)
        self._pub_markers = self.create_publisher(MarkerArray, "~/path_markers", latched)
        self._srv = self.create_service(SetBool, "~/republish", self._on_republish)

        period = float(self.get_parameter("republish_period").value)
        self._timer = None
        if period > 0.0:
            self._timer = self.create_timer(period, self.publish_all)

        self.publish_all()
        self.get_logger().info(
            f"loaded {len(samples)} samples from {traj_file}; "
            f"{len(self._path_data['poses'])} poses → /cam_desired_traj (latched); "
            f"planner={planner} ("
            + ("robot optical frame Z+ fwd / X+ right / Y+ down"
               if planner == "jarvis" else "FLU drone body") + ")")

    # ------------------------------------------------------------------ #
    def _build_path_msg(self) -> Path:
        msg = Path()
        msg.header.frame_id = self._path_data["frame_id"]
        msg.header.stamp = _stamp(self, self._path_data["start_epoch_s"])
        for p in self._path_data["poses"]:
            ps = PoseStamped()
            ps.header.frame_id = self._path_data["frame_id"]
            ps.header.stamp = _stamp(self, p["stamp_s"])
            ps.pose.position = Point(x=p["position"][0], y=p["position"][1],
                                     z=p["position"][2])
            w, x, y, z = p["orientation"]
            ps.pose.orientation = Quaternion(x=x, y=y, z=z, w=w)
            msg.poses.append(ps)
        return msg

    def _build_markers_msg(self) -> MarkerArray:
        msg = MarkerArray()
        for i, m in enumerate(self._markers_data):
            mk = Marker()
            mk.header.frame_id = m["frame_id"]
            mk.header.stamp = _stamp(self, self._path_data["start_epoch_s"])
            mk.ns, mk.id, mk.type, mk.action = m["ns"], m["id"], m["type"], m["action"]
            mk.pose.orientation.w = 1.0
            for pt in m["points"]:
                mk.points.append(Point(x=pt[0], y=pt[1], z=pt[2]))
            mk.scale.x = m["scale"]
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = m["color_rgba"]
            msg.markers.append(mk)
        return msg

    def publish_all(self) -> None:
        self._pub_path.publish(self._build_path_msg())
        self._pub_markers.publish(self._build_markers_msg())

    def _on_republish(self, request, response):
        if request.data:
            self.publish_all()
            response.success, response.message = True, "path re-published"
        elif self._timer is not None:
            self._timer.cancel()
            self._timer = None
            response.success, response.message = True, "periodic republish stopped"
        else:
            response.success, response.message = False, "nothing to stop"
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
