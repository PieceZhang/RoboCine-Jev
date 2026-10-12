"""Jarvis kinematics tests — FK against the hand-composed URDF chain, the
geometric Jacobian against central finite differences, IK round-trips on
FK-generated poses, the workspace envelope, and the direction-dependent
trackable speed (v_max). Robot definition: ros2_ws/jvs_mock @
yuelin/isaac-robot-controller (jvs_mock.urdf.xacro)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "ros2_ws" / "cinema_traj_bridge"
sys.path.insert(0, str(BRIDGE))

from cinema_traj.core.jarvis import (JARVIS_DEMO_CAPS, Q_HI, Q_LO, arm_ik,  # noqa: E402
                                     base_transform, base_yaw_rot,
                                     camera_from_look, check_trackable,
                                     fk_camera, jacobian, look_of,
                                     v_max, workspace_envelope, yaw_pitch_of)

TOL = 1e-9


# --------------------------------------------------------------------------- #
# FK — zero configuration vs the hand-composed URDF chain
# --------------------------------------------------------------------------- #
class TestFK:
    def test_zero_config_camera_pose(self):
        # base_link → neck (0.098, 0, 0.266), then the six joint origins,
        # then camera (0.038, 0, 0.1165) — sums to (0.122, −0.0015, 0.97879)
        # in the base frame; base yaw 0 maps base +x → world +Z, base +z → +Y
        T = fk_camera(0.0, 0.0, 0.0, [0.0] * 6)
        np.testing.assert_allclose(T[:3, 3], [-0.0015, 0.97879, 0.1220], atol=5e-5)
        np.testing.assert_allclose(look_of(T), [0.0, 0.0, 1.0], atol=TOL)

    def test_base_yaw_rotates_forward_axis(self):
        T = fk_camera(1.0, -2.0, math.pi / 2, [0.0] * 6)
        np.testing.assert_allclose(look_of(T), [1.0, 0.0, 0.0], atol=TOL)
        # base origin sits on the floor plane regardless of heading
        assert T[:3, 3][1] == pytest.approx(0.97879, abs=5e-5)

    def test_camera_look_matches_bridge_quaternion(self):
        # the binding cross-module contract is the LOOK VECTOR: the bridge's
        # body→world quaternion (ROS map axes: x=north=world X, y=east=world Z,
        # z=up=world Y) must rotate body +x onto the same world look vector
        # that camera_from_look builds from (yaw, pitch). The roll completion
        # about the look axis is unspecified by (yaw, pitch) and differs.
        from cinema_traj_bridge.traj_loader import quaternion_from_yaw_pitch
        M = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])   # ros→world comps
        for yaw_deg, pitch_deg in [(0, 0), (37, -12), (-95, 40), (180, 25)]:
            w, x, y, z = quaternion_from_yaw_pitch(yaw_deg, pitch_deg)
            Rq = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ])
            look_bridge = (M @ Rq)[:, 0]
            R = camera_from_look(math.radians(yaw_deg), math.radians(pitch_deg))
            np.testing.assert_allclose(R[:, 0], look_bridge, atol=1e-9)

    def test_yaw_pitch_roundtrip(self):
        T = fk_camera(0.0, 0.0, 0.7, [0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
        yaw, pitch = yaw_pitch_of(T)
        R = camera_from_look(yaw, pitch)
        np.testing.assert_allclose(R[:, 0], look_of(T), atol=1e-9)


# --------------------------------------------------------------------------- #
# Geometric Jacobian vs central finite differences
# --------------------------------------------------------------------------- #
class TestJacobian:
    def test_all_columns_match_finite_differences(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            q = rng.uniform(Q_LO, Q_HI)
            bx, bz, yaw = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi)
            J = jacobian(bx, bz, yaw, q)
            eps = 1e-6

            def pos(bx_, bz_, yaw_, q_):
                return fk_camera(bx_, bz_, yaw_, q_)[:3, 3]

            fwd = base_yaw_rot(yaw)[:, 0]
            # col 0 — forward drive: base translates along its heading
            fd = (pos(bx + fwd[0] * eps, bz + fwd[2] * eps, yaw, q)
                  - pos(bx - fwd[0] * eps, bz - fwd[2] * eps, yaw, q)) / (2 * eps)
            assert np.abs(fd - J[:3, 0]).max() < 1e-5
            # col 1 — base yaw about world +Y through the base origin
            fd = (pos(bx, bz, yaw + eps, q) - pos(bx, bz, yaw - eps, q)) / (2 * eps)
            assert np.abs(fd - J[:3, 1]).max() < 1e-5
            # cols 2..7 — joints
            for i in range(6):
                dq = np.zeros(6)
                dq[i] = eps
                fd = (pos(bx, bz, yaw, q + dq) - pos(bx, bz, yaw, q - dq)) / (2 * eps)
                assert np.abs(fd - J[:3, 2 + i]).max() < 1e-5


# --------------------------------------------------------------------------- #
# IK — round-trip on FK-generated poses (joint limits respected)
# --------------------------------------------------------------------------- #
class TestIK:
    def test_roundtrip_on_fk_poses(self):
        rng = np.random.default_rng(11)
        ok = 0
        n = 25
        for _ in range(n):
            q = rng.uniform(Q_LO * 0.8, Q_HI * 0.8)
            bx, bz, yaw = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi)
            T = fk_camera(bx, bz, yaw, q)
            sol, _err = arm_ik(base_transform(bx, bz, yaw), T)
            if sol is None:
                continue
            assert np.all(sol >= Q_LO - 1e-9) and np.all(sol <= Q_HI + 1e-9)
            T2 = fk_camera(bx, bz, yaw, sol)
            # acceptance = the solver's own tolerance contract (3 mm / 0.69°)
            if (np.linalg.norm(T2[:3, 3] - T[:3, 3]) < 4e-3
                    and np.linalg.norm(T2[:3, :3] - T[:3, :3]) < 1.5e-2):
                ok += 1
        assert ok >= n - 2, f"IK round-trip only {ok}/{n}"

    def test_unreachable_pose_reports_none(self):
        # 2 m above the base — far beyond the ~0.68 m arm envelope
        T_target = np.eye(4)
        T_target[:3, 3] = [0.3, 2.0, 0.0]
        sol, err = arm_ik(base_transform(0.0, 0.0, 0.0), T_target)
        assert sol is None
        assert np.isfinite(err)


# --------------------------------------------------------------------------- #
# Workspace envelope
# --------------------------------------------------------------------------- #
class TestEnvelope:
    def test_envelope_matches_analytic_bounds(self):
        env = workspace_envelope()
        # fully-stretched camera ≈ 1.0 m above the floor; folded ≈ 0.1 m
        assert 0.8 < env.z_max < 1.1
        assert 0.0 <= env.z_min < 0.35
        # horizontal reach well under a meter, above the tight fold
        assert 0.5 < env.reach_max < 0.8
        # the joint boxes nearly span vertical looks
        assert env.pitch_max - env.pitch_min > math.radians(150)


# --------------------------------------------------------------------------- #
# v_max — the direction-dependent trackable speed
# --------------------------------------------------------------------------- #
class TestVMax:
    def test_horizontal_fast_vertical_slow(self):
        # canonical resolution: the base drives any horizontal direction
        # (≤ ~0.85 m/s), vertical motion is arm-only (well under that)
        caps = JARVIS_DEMO_CAPS
        fwd = v_max((0.0, 0.0, 1.0), caps)
        side = v_max((1.0, 0.0, 0.0), caps)          # realigns → also base-driven
        up = v_max((0.0, 1.0, 0.0), caps)
        assert fwd == pytest.approx(side, rel=0.05)
        assert 0.6 < fwd <= 1.0
        assert up < 0.5 * fwd

    def test_v_max_finite_across_pitch_sweep(self):
        # every look-pitch mix admits a finite positive trackable speed; the
        # pinv redistributes across joints so blends are not componentwise
        # bounded — only finiteness, positivity and the horizontal ceiling
        caps = JARVIS_DEMO_CAPS
        fwd = v_max((0.0, 0.0, 1.0), caps)
        for deg in range(-80, 81, 20):
            a = math.radians(deg)
            d = np.array([0.0, math.sin(a), math.cos(a)])
            vm = v_max(d, caps)
            assert 0.0 < vm < np.inf
            assert vm <= fwd + 1e-6


# --------------------------------------------------------------------------- #
# check_trackable — exact configuration-stream verification
# --------------------------------------------------------------------------- #
def _straight_configs(v=0.5, omega=0.0, qdot=None, dt=0.1, n=10):
    qdot = np.zeros(6) if qdot is None else np.asarray(qdot)
    configs = []
    for k in range(n):
        q = np.clip(qdot * k * dt, Q_LO, Q_HI)
        configs.append({"t": k * dt,
                        "base": (0.0, k * dt * v, omega * k * dt),
                        "q": q})
    return configs


class TestCheckTrackable:
    def test_compliant_stream_is_clean(self):
        caps = JARVIS_DEMO_CAPS
        configs = _straight_configs(v=0.5, omega=1.0, qdot=[0.3] * 6)
        samples = [{"t": c["t"]} for c in configs]
        assert check_trackable(samples, configs, caps) == []

    def test_base_speed_violation_reported(self):
        caps = JARVIS_DEMO_CAPS
        configs = _straight_configs(v=caps.max_base_linear_vel + 0.2)
        samples = [{"t": c["t"]} for c in configs]
        warns = check_trackable(samples, configs, caps)
        assert warns and "base speed" in warns[0]

    def test_joint_rate_violation_reported(self):
        caps = JARVIS_DEMO_CAPS
        configs = _straight_configs(qdot=[caps.max_joint_vel[0] + 0.5] * 6)
        samples = [{"t": c["t"]} for c in configs]
        warns = check_trackable(samples, configs, caps)
        assert warns and "joint1" in warns[0]

    def test_joint_box_violation_reported(self):
        caps = JARVIS_DEMO_CAPS
        configs = _straight_configs(qdot=[2.0] * 6)
        for c in configs:
            c["q"] = np.full(6, Q_HI[0] + 0.1)      # outside the box
        samples = [{"t": c["t"]} for c in configs]
        warns = check_trackable(samples, configs, caps)
        assert warns and "outside" in warns[0]
