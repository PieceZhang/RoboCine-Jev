"""Robot camera frame tests — X+ right, Y+ down, Z+ forward (robot's
definition). Cross-checked against the OpenCV math already shipped in
tools/gsplat_render.viewmat_from_pos_look."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from cinema_traj.common.robot_frame import (  # noqa: E402
    pose_robot, quaternion_robot, rotation_robot, rotation_robot_torch,
)
from gsplat_render import viewmat_from_pos_look  # noqa: E402


def _axes_world(R):
    """Robot camera axes expressed in world space — the definition is
    body→world with AXES AS COLUMNS, so extract columns (numpy is row-major:
    R[:, 0] = x_right, R[:, 1] = y_down, R[:, 2] = z_forward)."""
    return R[:, 0], R[:, 1], R[:, 2]


class TestRotationRobot:
    def test_axes_named_correctly(self):
        # camera at origin looking toward +Z world, world up +Y:
        # forward = +Z world; right = f×up = (0,0,1)×(0,1,0) = (-1,0,0)… let
        # the assertions derive everything from the definition instead.
        R = rotation_robot([0, 0, 0], [0, 0, 1])
        x_right, y_down, z_fwd = _axes_world(R)
        np.testing.assert_allclose(z_fwd, [0, 0, 1])          # Z+ = look dir
        # right must be horizontal and perpendicular to forward
        assert abs(np.dot(x_right, z_fwd)) < 1e-12
        assert abs(x_right[1]) < 1e-12                        # no vertical tilt
        np.testing.assert_allclose(np.cross(x_right, y_down), z_fwd, atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12            # proper rotation

    def test_y_down_not_up(self):
        # level camera: the second column must point DOWN in world terms
        R = rotation_robot([0, 1.5, 0], [0, 1.5, 1])
        y_down = R[:, 1]
        assert y_down[1] < -0.99, "Y+ column must be image DOWN (world −Y)"

    def test_matches_gsplat_viewmat(self):
        # the view matrix is inv(c2w) with c2w columns [right, −up, fwd] —
        # its ROTATION part transposed must equal our body→world rotation
        rng = np.random.default_rng(0)
        for _ in range(20):
            pos = rng.uniform(-2, 2, 3); pos[1] = abs(pos[1]) + 0.4
            look = pos + rng.normal(size=3) * np.array([1, 0.3, 1])
            look[1] = pos[1] + rng.uniform(-0.8, 0.8)
            R_robot = rotation_robot(pos, look)
            view = viewmat_from_pos_look(pos, look)
            R_view = view[:3, :3]                     # world→camera
            np.testing.assert_allclose(R_robot, R_view.T, atol=1e-9)

    def test_degenerate_looking_straight_down(self):
        # look = −Y (straight down): the nudged-up fallback must still yield
        # a proper rotation whose Z+ column is exactly the look direction
        R = rotation_robot([0, 2.0, 0], [0, 0.0, 0])
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        np.testing.assert_allclose(R[:, 2], [0, -1, 0], atol=1e-9)
        x_right, y_down, z_fwd = R[:, 0], R[:, 1], R[:, 2]
        np.testing.assert_allclose(np.cross(x_right, y_down), z_fwd, atol=1e-9)

    def test_torch_matches_numpy_and_diffs(self):
        pos = torch.tensor([[0.0, 1.5, 0.0], [1.0, 1.0, -1.0]], dtype=torch.float64)
        look = torch.tensor([[0.0, 1.5, 2.0], [2.0, 1.2, -1.0]], dtype=torch.float64)
        R_t = rotation_robot_torch(pos, look)
        for i in range(len(pos)):
            np.testing.assert_allclose(R_t[i].numpy(),
                                       rotation_robot(pos[i].numpy(),
                                                      look[i].numpy()),
                                       atol=1e-9)
        pos.requires_grad_(True)
        rotation_robot_torch(pos, look).sum().backward()
        assert torch.isfinite(pos.grad).all()


class TestQuaternion:
    def test_identity_level_camera(self):
        # looking along world +Z from origin, level: yaw about −Y by 90°…
        # verify numerically: rotate body +Z (forward) by q → world look dir
        q = quaternion_robot([0, 0, 0], [0, 0, 1])

        def apply(q, v):
            w, x, y, z = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            return R @ np.asarray(v)

        np.testing.assert_allclose(apply(q, [0, 0, 1]), [0, 0, 1], atol=1e-12)
        # body +X (right) must map to world image-right
        R = rotation_robot([0, 0, 0], [0, 0, 1])
        np.testing.assert_allclose(apply(q, [1, 0, 0]), R[0], atol=1e-12)
        # body +Y (down) must map to world image-down
        np.testing.assert_allclose(apply(q, [0, 1, 0]), R[1], atol=1e-12)

    def test_random_positions_match_rotation(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            pos = rng.uniform(-2, 2, 3); pos[1] = abs(pos[1]) + 0.5
            look = pos + rng.normal(size=3) * [1, 0.3, 1]
            q = quaternion_robot(pos, look)
            R = rotation_robot(pos, look)

            def apply(q, v):
                w, x, y, z = q
                Rq = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
                return Rq @ np.asarray(v)

            for body_axis in np.eye(3):
                np.testing.assert_allclose(apply(q, body_axis),
                                           R @ body_axis, atol=1e-9)


class TestPoseRobot:
    def test_pose_documented_and_complete(self):
        pose = pose_robot([1.0, 1.5, -2.0], [1.0, 1.4, -1.0])
        assert set(pose) == {"position", "quaternion_wxyz",
                             "rotation_body_to_world", "frame"}
        assert "X+ right" in pose["frame"] and "Z+ forward" in pose["frame"]
        q = np.array(pose["quaternion_wxyz"])
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12
