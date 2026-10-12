"""Jarvis (jvs_mock, PR #11) trajectory conversion tests.

The robot's head camera is the URDF's ``camera_optical_frame``:
Z+ FORWARD, X+ RIGHT, Y+ DOWN. These tests pin that the converter's
relative poses, composed back with the robot's OWN math
(``jvs_kinematics.relative_pose_target``), reproduce the source
trajectory — i.e. the optical-frame semantics close the loop.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "ros2_ws" / "jvs_mock" / "jvs_mock"))

from export_jarvis_trajectory import (  # noqa: E402
    FORMAT, _rotation_from_quaternion, _rotation_to_rpy, convert,
)
from jvs_kinematics import rpy_matrix, relative_pose_target  # noqa: E402


def _fake_samples(n: int = 21):
    """A synthetic push-in: camera glides forward and pitches down a bit,
    with robot-frame quaternions stored per sample (like the export emits)."""
    from cinema_traj.common.robot_frame import quaternion_robot

    samples = []
    for k in range(n):
        t = k / (n - 1) * 2.0
        pos = [1.0 + 0.4 * t, 1.5 - 0.15 * t, -2.0 + 0.25 * t]
        look = [pos[0] + 1.0, pos[1] - 0.1 * t, pos[2] + 1.0 - 0.3 * t]
        q = quaternion_robot(pos, look)
        samples.append({"t": round(t, 3), "x": pos[0], "y": pos[1], "z": pos[2],
                        "robot_quat_wxyz": list(q)})
    return samples


class TestSchema:
    def test_header_fields(self):
        doc = convert(_fake_samples())
        assert doc["format"] == FORMAT
        assert doc["relative_to"] == "current_camera_link"
        assert doc["frame_id"] == "base_link"
        assert doc["dof_order"] == ["x", "y", "z", "roll", "pitch", "yaw"]
        assert doc["units"] == {"translation": "m", "rotation": "rad"}
        assert "Z+ FORWARD" in doc["camera_frame"]
        assert "X+ RIGHT" in doc["camera_frame"] and "Y+ DOWN" in doc["camera_frame"]

    def test_first_sample_is_identity(self):
        doc = convert(_fake_samples())
        assert doc["samples"][0]["time_from_start"] == 0.0
        assert doc["samples"][0]["pose"] == [0.0] * 6

    def test_rate_and_duration(self):
        doc = convert(_fake_samples(), rate_hz=10.0)
        assert doc["sample_rate_hz"] == 10.0
        assert doc["duration_s"] == pytest.approx(2.0)
        assert len(doc["samples"]) == 21               # 2 s × 10 Hz + 1


class TestRPY:
    def test_roundtrip_matrix_to_rpy(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            a, b, c = rng.uniform(-3.0, 3.0, 3)
            R = rpy_matrix(a, b, c)
            roll, pitch, yaw = _rotation_to_rpy(R)
            np.testing.assert_allclose(rpy_matrix(roll, pitch, yaw), R,
                                       atol=1e-9)

    def test_no_gimbal_lock_singularities(self):
        R = rpy_matrix(0.3, 1.0, -0.7)                 # away from ±90° pitch
        roll, pitch, yaw = _rotation_to_rpy(R)
        np.testing.assert_allclose(rpy_matrix(roll, pitch, yaw), R, atol=1e-9)


class TestOpticalFrameLoop:
    def test_jarvis_composition_reproduces_source(self):
        """The decisive check: feed our converted file back through the
        ROBOT'S OWN composition (relative_pose_target with the captured
        initial pose) — the absolute poses must equal the source export."""
        src = _fake_samples()
        doc = convert(src)
        p0 = np.array([src[0]["x"], src[0]["y"], src[0]["z"]])
        R0 = _rotation_from_quaternion(src[0]["robot_quat_wxyz"])

        max_dp = max_dR = 0.0
        for s, ref in zip(doc["samples"], src):
            target_pos, target_R = relative_pose_target(
                p0, R0, np.array(s["pose"]), "initial_camera_link")
            ref_pos = np.array([ref["x"], ref["y"], ref["z"]])
            ref_R = _rotation_from_quaternion(ref["robot_quat_wxyz"])
            max_dp = max(max_dp, float(np.linalg.norm(target_pos - ref_pos)))
            max_dR = max(max_dR, float(np.abs(target_R - ref_R).max()))
        assert max_dp < 5e-3, f"position loop broken: {max_dp*1000:.2f} mm"
        assert max_dR < 2e-3, f"rotation loop broken: {max_dR:.2e}"

    def test_base_link_frame_translates_in_base_axes(self):
        # frame_id=base_link composes Δp directly in base axes — the ROTATION
        # path differs but translation must be identity-composed
        src = _fake_samples()
        doc = convert(src, frame_id="base_link")
        p0 = np.array([src[0]["x"], src[0]["y"], src[0]["z"]])
        R0 = _rotation_from_quaternion(src[0]["robot_quat_wxyz"])
        s = doc["samples"][-1]
        target_pos, _ = relative_pose_target(p0, R0, np.array(s["pose"]),
                                             "base_link")
        # world Δp = R_ref·Δp_optical; with base_link, target = p0 + Δp_stored
        delta_stored = np.array(s["pose"][:3])
        np.testing.assert_allclose(target_pos, p0 + delta_stored, atol=1e-9)

    def test_quat_slerp_shortest_arc(self):
        # two quaternions with opposite signs encode the SAME rotation —
        # interpolation must not take the long way (output stays continuous)
        src = _fake_samples(21)
        src[10]["robot_quat_wxyz"] = [-v for v in src[10]["robot_quat_wxyz"]]
        doc = convert(src)
        poses = np.array([s["pose"] for s in doc["samples"]])
        jumps = np.abs(np.diff(poses[:, 3:], axis=0)).max()
        assert jumps < 0.3, f"orientation jumped {jumps:.2f} rad — slerp arc wrong"
