"""Pure trajectory loading + ROS data construction — NO rclpy dependency.

This module is importable and unit-testable on any machine (macOS included).
All ROS message construction is expressed as plain data (dicts / tuples);
``traj_server.py`` is the thin rclpy glue that turns this data into real
messages.

Frame conventions (pinned by tests/test_traj_loader.py):

  * input  — cinema_traj exports are Y-up world: (x, y_up, z), yaw 0 → +Z
             positive toward +X, gimbal pitch positive up, roll always 0.
  * output — ROS REP-103-style map frame: x = north = world +X,
             y = east = world +Z, z = up = world +Y. Camera body is
             x-forward / y-left / z-up; the quaternion maps body → world so
             that body +x points along the look direction (roll 0).
  * planner=jarvis — the 2-wheel + 6-axis robot's head camera is the URDF's
             camera_optical_frame: Z+ FORWARD, X+ RIGHT, Y+ DOWN. In that
             mode the quaternion maps THIS optical body → world (built from
             the export's robot_quat_wxyz, which shares the convention), and
             positions stay world-frame (consumer converts to base_link).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_samples(path: str) -> list[dict]:
    """Read traj_samples.json / traj_samples_retimed.json (preferred — has the
    frame doc) or traj_samples.csv into [{t, x, y, z, yaw_deg, pitch_deg,
    roll_deg, fov_deg?, segment?}...] sorted by t."""
    p = Path(path)
    if p.suffix == ".json":
        doc = json.loads(p.read_text(encoding="utf-8"))
        frame = doc.get("frame", {})
        if frame and frame.get("up") != "+Y":
            raise ValueError(f"{p}: expected '+Y' up frame, got {frame.get('up')!r}")
        samples = sorted(doc.get("samples", []), key=lambda s: s["t"])
    elif p.suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("t") not in (None, "")]
        samples = [{"t": float(r["t"]),
                    "x": float(r["x"]), "y": float(r["y"]), "z": float(r["z"]),
                    "yaw_deg": float(r["yaw_deg"]), "pitch_deg": float(r["pitch_deg"]),
                    "roll_deg": float(r.get("roll_deg") or 0.0),
                    "fov_deg": float(r["fov_deg"]) if r.get("fov_deg") else None,
                    "segment": r.get("segment") or ""}
                   for r in rows]
    else:
        raise ValueError(f"unsupported trajectory file: {p} (want .json / .csv)")
    if len(samples) < 2:
        raise ValueError(f"{p}: need >= 2 samples")
    return samples


# --------------------------------------------------------------------------- #
# Frame conversion (Y-up world → ROS x-east-style map frame, z-up)
# --------------------------------------------------------------------------- #
def world_to_ros_position(p) -> tuple[float, float, float]:
    """(x, y_up, z)_world → (x=north, y=east, z=up)_ros."""
    return (float(p[0]), float(p[2]), float(p[1]))


def quaternion_from_yaw_pitch(yaw_world_deg: float, pitch_deg: float) -> tuple:
    """(w, x, y, z) quaternion, body→world, camera x-forward y-left z-up,
    roll 0. Heading ψ about +z (from +x toward +y) equals the NED yaw
    90° − yaw_world; gimbal pitch positive up rotates body +x above horizon."""
    psi = math.radians(90.0 - yaw_world_deg)     # heading about ros +z
    rho = math.radians(pitch_deg)                # positive up
    cz, sz = math.cos(psi / 2.0), math.sin(psi / 2.0)
    cy, sy = math.cos(rho / 2.0), math.sin(rho / 2.0)
    # q = qz(ψ) ⊗ qy(−ρ): rotate −ρ about y (pitch up), then ψ about z
    return (cz * cy, sz * sy, -cz * sy, sz * cy)


def quaternion_robot_from_sample(sample: dict) -> tuple:
    """(w, x, y, z) body→world quaternion in the robot's optical frame
    (Z+ forward, X+ right, Y+ down), from an export sample's
    ``robot_quat_wxyz``. Raises when the export predates the robot frame —
    re-run ``run_traj.py --stage export``."""
    q = sample.get("robot_quat_wxyz")
    if not q:
        raise ValueError(
            "sample has no robot_quat_wxyz — re-run the export stage "
            "(planner=jarvis needs the robot camera frame)")
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def look_vector_ros(yaw_world_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    """Unit look vector in ROS map axes (dual of the quaternion)."""
    q = quaternion_from_yaw_pitch(yaw_world_deg, pitch_deg)
    # rotate body +x by q
    w, x, y, z = q
    return (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z),
            2.0 * (x * z - w * y))


# --------------------------------------------------------------------------- #
# nav_msgs/Path data
# --------------------------------------------------------------------------- #
def build_path_data(samples: list[dict], frame_id: str, start_epoch_s: float,
                     step: int = 1, planner: str = "drone") -> dict:
    """Construct the plain-data equivalent of nav_msgs/msg/Path.

    poses[i]: {stamp_s, position (x,y,z), orientation (w,x,y,z)} with
    stamp_s = start_epoch_s + t_i (absolute seconds). Use step > 1 to
    downsample very long retimed streams for RViz.

    planner="jarvis" swaps the orientation convention to the robot's
    optical frame (Z+ forward, X+ right, Y+ down) — positions remain
    world-frame; the robot node converts to its base_link.
    """
    robot_mode = planner == "jarvis"
    poses = []
    for s in samples[::step]:
        orientation = quaternion_robot_from_sample(s) if robot_mode else             quaternion_from_yaw_pitch(s["yaw_deg"], s["pitch_deg"])
        poses.append({
            "stamp_s": start_epoch_s + float(s["t"]),
            "position": world_to_ros_position((s["x"], s["y"], s["z"])),
            "orientation": orientation,
        })
    if poses and samples[::step][-1] is not samples[-1]:        # always keep the end
        last = samples[-1]
        orientation = quaternion_robot_from_sample(last) if robot_mode else             quaternion_from_yaw_pitch(last["yaw_deg"], last["pitch_deg"])
        poses.append({
            "stamp_s": start_epoch_s + float(last["t"]),
            "position": world_to_ros_position((last["x"], last["y"], last["z"])),
            "orientation": orientation,
        })
    return {"frame_id": frame_id, "start_epoch_s": start_epoch_s,
            "planner": planner, "poses": poses}


# --------------------------------------------------------------------------- #
# visualization_msgs/MarkerArray data (RViz debugging)
# --------------------------------------------------------------------------- #
def build_markers_data(path_data: dict, subject_track: dict | None = None,
                       subject_step_s: float = 0.5) -> list[dict]:
    """Markers as plain data: camera path LINE_STRIP + start/end spheres, plus
    subject feet/head polylines and a capsule at t0 when a scene subject_track
    is given. Marker type numbers follow visualization_msgs/Marker."""
    markers = [{
        "ns": "cinema_traj", "id": 0, "type": 4, "action": 0,   # LINE_STRIP
        "frame_id": path_data["frame_id"],
        "points": [p["position"] for p in path_data["poses"]],
        "scale": 0.05, "color_rgba": (0.1, 0.9, 0.9, 0.9),
    }]
    if path_data["poses"]:
        first, last = path_data["poses"][0], path_data["poses"][-1]
        markers.append({"ns": "cinema_traj", "id": 1, "type": 2, "action": 0,  # SPHERE
                        "frame_id": path_data["frame_id"], "points": [first["position"]],
                        "scale": 0.18, "color_rgba": (0.2, 0.9, 0.2, 1.0)})
        markers.append({"ns": "cinema_traj", "id": 2, "type": 2, "action": 0,
                        "frame_id": path_data["frame_id"], "points": [last["position"]],
                        "scale": 0.18, "color_rgba": (0.9, 0.2, 0.2, 1.0)})

    if subject_track:
        frames = sorted(subject_track.get("frames", []), key=lambda f: f["t"])
        if frames:
            t0, t1 = frames[0]["t"], frames[-1]["t"]
            n = max(2, int(math.ceil((t1 - t0) / max(subject_step_s, 1e-3))) + 1)
            height = float(subject_track.get("height_m", 1.75))
            feet, head = [], []
            for k in range(n):
                f = frames[min(int(k * (len(frames) - 1) / (n - 1)), len(frames) - 1)]
                p = f["pos"]
                feet.append(world_to_ros_position(p))
                head.append(world_to_ros_position((p[0], p[1] + height, p[2])))
            markers.append({"ns": "subject", "id": 10, "type": 4, "action": 0,
                            "frame_id": path_data["frame_id"], "points": feet,
                            "scale": 0.03, "color_rgba": (0.95, 0.6, 0.1, 0.9)})
            markers.append({"ns": "subject", "id": 11, "type": 4, "action": 0,
                            "frame_id": path_data["frame_id"], "points": head,
                            "scale": 0.03, "color_rgba": (0.95, 0.6, 0.1, 0.9)})
            p0 = frames[0]["pos"]
            markers.append({                                # capsule axis at t0
                "ns": "subject", "id": 12, "type": 4, "action": 0,
                "frame_id": path_data["frame_id"],
                "points": [world_to_ros_position(p0),
                           world_to_ros_position((p0[0], p0[1] + 0.9 * height, p0[2]))],
                "scale": 0.3, "color_rgba": (0.9, 0.2, 0.2, 0.45)})
    return markers
