"""cinema_traj_bridge traj_loader tests — pin the Y-up → ROS REP-103 mapping
and the Path/Marker data construction (pure logic, no rclpy)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "ros2_ws" / "cinema_traj_bridge"
sys.path.insert(0, str(BRIDGE))

from cinema_traj_bridge import traj_loader  # noqa: E402


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class TestFrameMapping:
    def test_position_yup_to_ros(self):
        assert traj_loader.world_to_ros_position((1.0, 2.0, 3.0)) == (1.0, 3.0, 2.0)

    def test_quaternion_looks_along_expected_direction(self):
        """Body +x after rotation must equal the world-frame look vector
        expressed in ROS axes — pinned against the authoritative se3 math."""
        from cinema_traj.common.se3 import look_at_R, rotation_to_yaw_pitch
        rng = np.random.default_rng(11)
        for _ in range(60):
            eye = rng.uniform(-2, 2, 3) + [0, 1.5, 0]
            target = rng.uniform(-2, 2, 3) + [0, 1.2, 0]
            yaw_w, pitch = rotation_to_yaw_pitch(look_at_R(eye, target))
            look_world = (target - eye) / np.linalg.norm(target - eye)
            look_ros_expected = np.array([look_world[0], look_world[2], look_world[1]])
            look_ros = np.array(traj_loader.look_vector_ros(yaw_w, pitch))
            np.testing.assert_allclose(look_ros, look_ros_expected, atol=1e-9)
            # and via the quaternion itself
            q = traj_loader.quaternion_from_yaw_pitch(yaw_w, pitch)
            np.testing.assert_allclose(_quat_to_R(q)[:, 0], look_ros_expected, atol=1e-9)

    def test_quaternion_keeps_camera_upright(self):
        """roll = 0 ⇒ body +z stays in the world up half-space."""
        for yaw_w in (0.0, 47.0, 180.0, -129.0):
            for pitch in (-85.0, -30.0, 0.0, 45.0):
                q = traj_loader.quaternion_from_yaw_pitch(yaw_w, pitch)
                body_z_world = _quat_to_R(q)[:, 2]
                assert body_z_world[2] > 0.0          # z-component (up) positive

    def test_identity_quaternion_at_yaw_90_pitch_0(self):
        """yaw_w=90° looks +X(world)=+x(ros) ⇒ identity orientation."""
        q = traj_loader.quaternion_from_yaw_pitch(90.0, 0.0)
        np.testing.assert_allclose(q, (1.0, 0.0, 0.0, 0.0), atol=1e-12)

    def test_yaw_0_pitch_0_points_east(self):
        """yaw_w=0 looks +Z(world)=east(+y ros): rotate +90° about z."""
        q = traj_loader.quaternion_from_yaw_pitch(0.0, 0.0)
        np.testing.assert_allclose(q, (math.cos(math.pi / 4), 0, 0,
                                       math.sin(math.pi / 4)), atol=1e-12)


def _samples(n=8):
    return [{"t": i * 0.5, "x": float(i), "y": 1.0 + 0.1 * i, "z": -0.5 * i,
             "yaw_deg": 20.0 * i, "pitch_deg": -10.0 + 2.0 * i, "roll_deg": 0.0,
             "fov_deg": 55.0, "segment": "segA"} for i in range(n)]


class TestLoadSamples:
    def test_json_roundtrip(self, tmp_path):
        p = tmp_path / "traj_samples.json"
        p.write_text(json.dumps({"frame": {"up": "+Y"}, "samples": _samples()}),
                     encoding="utf-8")
        out = traj_loader.load_samples(str(p))
        assert len(out) == 8
        assert out[0]["t"] == 0.0 and out[3]["x"] == 3.0

    def test_retimed_json_preferred_fields(self, tmp_path):
        p = tmp_path / "traj_samples_retimed.json"
        doc = {"frame": {"up": "+Y"}, "retime": {"scale": 17.8}, "samples": _samples()}
        p.write_text(json.dumps(doc), encoding="utf-8")
        assert len(traj_loader.load_samples(str(p))) == 8

    def test_rejects_non_yup_frame(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"frame": {"up": "-Z"}, "samples": _samples()}),
                     encoding="utf-8")
        with pytest.raises(ValueError):
            traj_loader.load_samples(str(p))

    def test_csv_roundtrip(self, tmp_path):
        import csv
        p = tmp_path / "traj_samples.csv"
        cols = ["t", "x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg",
                "fov_deg", "segment"]
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for s in _samples():
                w.writerow({k: s[k] for k in cols})
        out = traj_loader.load_samples(str(p))
        assert len(out) == 8
        assert out[2]["z"] == -1.0 and out[2]["segment"] == "segA"

    def test_needs_two_samples(self, tmp_path):
        p = tmp_path / "short.json"
        p.write_text(json.dumps({"frame": {"up": "+Y"}, "samples": _samples(1)}),
                     encoding="utf-8")
        with pytest.raises(ValueError):
            traj_loader.load_samples(str(p))


class TestPathData:
    def test_pose_fields_and_stamps(self):
        data = traj_loader.build_path_data(_samples(), "map", 1000.0)
        assert data["frame_id"] == "map"
        assert len(data["poses"]) == 8
        p0, p7 = data["poses"][0], data["poses"][-1]
        assert p0["stamp_s"] == pytest.approx(1000.0)
        assert p7["stamp_s"] == pytest.approx(1000.0 + 3.5)
        # position mapping y-up → z-up (last sample i=7: x=7, y=1.7, z=−3.5)
        assert p7["position"] == pytest.approx((7.0, -3.5, 1.7))
        # orientation maps body +x onto the look direction in ROS axes
        look = np.array(traj_loader.look_vector_ros(140.0, 4.0))
        np.testing.assert_allclose(_quat_to_R(p7["orientation"])[:, 0], look, atol=1e-9)

    def test_step_downsampling_keeps_endpoints(self):
        data = traj_loader.build_path_data(_samples(40), "map", 0.0, step=5)
        ts = [p["stamp_s"] for p in data["poses"]]
        assert ts[0] == 0.0 and ts[-1] == pytest.approx(19.5)
        assert len(data["poses"]) == 9            # 8 stepped poses + final pose
        assert all(b > a for a, b in zip(ts, ts[1:]))


class TestMarkerData:
    def test_camera_markers(self):
        data = traj_loader.build_path_data(_samples(), "map", 0.0)
        markers = traj_loader.build_markers_data(data)
        by_ns = {}
        for m in markers:
            by_ns.setdefault(m["ns"], []).append(m)
        assert len(by_ns["cinema_traj"]) == 3              # strip + 2 spheres
        strip = by_ns["cinema_traj"][0]
        assert strip["type"] == 4 and len(strip["points"]) == 8
        assert strip["points"][3] == pytest.approx((3.0, -1.5, 1.3))

    def test_subject_markers_from_scene_track(self):
        track = {"height_m": 1.75,
                 "frames": [{"t": i, "pos": [0.1 * i, 0.0, 0.2 * i], "heading": 0.0}
                            for i in range(10)]}
        data = traj_loader.build_path_data(_samples(), "map", 0.0)
        markers = traj_loader.build_markers_data(data, subject_track=track)
        subj = [m for m in markers if m["ns"] == "subject"]
        assert len(subj) == 3                              # feet + head + capsule
        capsule = subj[2]
        assert capsule["points"][0] == pytest.approx((0.0, 0.0, 0.0))
        assert capsule["points"][1] == pytest.approx((0.0, 0.0, 0.9 * 1.75))


class TestJarvisPlanner:
    """planner=jarvis — /cam_desired_traj orientations in the robot's optical
    frame (Z+ forward, X+ right, Y+ down), positions unchanged."""

    def _robot_samples(self, n=4):
        from cinema_traj.common.robot_frame import quaternion_robot
        out = _samples(n)
        for s in out:
            q = quaternion_robot([s["x"], s["y"], s["z"]],
                                 [s["x"] + 1.0, s["y"], s["z"]])
            s["robot_quat_wxyz"] = list(q)
        return out

    def test_orientation_uses_robot_quat(self):
        data = traj_loader.build_path_data(self._robot_samples(), "map", 0.0,
                                           planner="jarvis")
        assert data["planner"] == "jarvis"
        # forward body axis (optical +Z) rotated by the stored quaternion
        # must equal the robot frame's world forward — compare against the
        # rotation built from the same quaternion
        from cinema_traj.common.robot_frame import rotation_robot
        s0 = self._robot_samples()[0]
        R = rotation_robot([s0["x"], s0["y"], s0["z"]],
                           [s0["x"] + 1.0, s0["y"], s0["z"]])
        w, x, y, z = data["poses"][0]["orientation"]
        # rotate body +Z by q → R @ [0,0,1]
        qz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x),
                       1 - 2 * (x * x + y * y)])
        np.testing.assert_allclose(qz, R[:, 2], atol=1e-9)

    def test_positions_unchanged_between_planners(self):
        samples = self._robot_samples()
        drone = traj_loader.build_path_data(samples, "map", 0.0)
        jarvis = traj_loader.build_path_data(samples, "map", 0.0,
                                             planner="jarvis")
        for pd, pj in zip(drone["poses"], jarvis["poses"]):
            assert pd["position"] == pj["position"]

    def test_missing_robot_quat_raises(self):
        with pytest.raises(ValueError, match="robot_quat_wxyz"):
            traj_loader.build_path_data(_samples(3), "map", 0.0,
                                        planner="jarvis")
