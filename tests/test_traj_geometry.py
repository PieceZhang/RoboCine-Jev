"""Geometry unit tests: exact OBB SDF, SE3 helpers, scene SDF union, subject
capsule (numpy/torch parity + autograd sanity)."""

import math

import numpy as np
import pytest
import torch

from cinema_traj.common.obb import OBB, fit_obb_pca, ray_obb_intersect
from cinema_traj.common.scene import Scene
from cinema_traj.common.se3 import (angle_lerp, cart_to_sph, ease_in_out, look_at_R,
                                    look_at_R_torch, rotation_to_yaw_pitch, sph_offset)
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.common.subject import SubjectTrack


def _box(cx=0.0, cy=0.5, cz=0.0, ex=1.0, ey=1.0, ez=1.0):
    return OBB(np.array([cx, cy, cz]), np.eye(3), np.array([ex, ey, ez]))


# --------------------------------------------------------------------------- #
# OBB SDF
# --------------------------------------------------------------------------- #
class TestOBBSdf:
    def test_inside_negative(self):
        b = _box()
        assert b.sdf(np.array([0.0, 0.5, 0.0])) == pytest.approx(-0.5)   # at center
        assert b.sdf(np.zeros(3)) == pytest.approx(0.0)    # point ON the bottom face

    def test_face_outside(self):
        b = _box()
        assert b.sdf(np.array([1.0, 0.5, 0.0])) == pytest.approx(0.5)

    def test_corner_outside(self):
        b = _box()
        # nearest surface point is the box corner (0.5, 1.0, 0.5):
        # dist((1,1,1), (0.5,1,0.5)) = sqrt(0.5² + 0 + 0.5²)
        assert b.sdf(np.array([1.0, 1.0, 1.0])) == pytest.approx(math.sqrt(0.5))

    def test_torch_parity_and_grad(self):
        b = _box()
        p = torch.tensor([[1.2, 0.5, 0.0], [0.0, 0.5, 0.0]], dtype=torch.float64,
                         requires_grad=True)
        v = b.sdf_torch(p)
        np.testing.assert_allclose(v.detach().numpy(), [0.7, -0.5])
        v.sum().backward()
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0

    def test_rotated_box(self):
        # rotate 90° around Y: local x-axis becomes world z
        c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        b = OBB(np.zeros(3), R, np.array([2.0, 1.0, 1.0]))
        # world point 1.5 along +z is 0.5 outside the (now-z-extent=2) box face
        assert b.sdf(np.array([0.0, 0.0, 1.5])) == pytest.approx(0.5, abs=1e-9)

    def test_corners_count_and_pca_fit(self):
        rng = np.random.default_rng(0)
        base = _box(ex=2.0, ey=1.0, ez=0.6)
        assert base.corners().shape == (8, 3)
        # uniform samples inside the box (not just corners — PCA needs volume)
        pts = (rng.uniform(-0.5, 0.5, (3000, 3)) * base.extents) + base.center
        obb = fit_obb_pca(pts)
        # robust 5–95 percentile extents ≈ 90% of the true uniform range
        np.testing.assert_allclose(sorted(obb.extents),
                                   sorted(np.array([2.0, 1.0, 0.6]) * 0.9), rtol=0.08)


# --------------------------------------------------------------------------- #
# Ray ↔ OBB
# --------------------------------------------------------------------------- #
class TestRayOBB:
    def test_hit_and_miss(self):
        b = _box()
        t_e, t_x = ray_obb_intersect(np.array([[3.0, 0.5, 0.0]]),
                                     np.array([[-1.0, 0.0, 0.0]]), b)
        assert t_x[0] > t_e[0] and t_e[0] == pytest.approx(2.5)
        t_e, t_x = ray_obb_intersect(np.array([[3.0, 5.0, 0.0]]),
                                     np.array([[-1.0, 0.0, 0.0]]), b)
        assert t_x[0] < t_e[0]                          # miss

    def test_inside_origin(self):
        b = _box()
        t_e, t_x = ray_obb_intersect(np.array([[0.0, 0.5, 0.0]]),
                                     np.array([[1.0, 0.0, 0.0]]), b)
        assert t_e[0] < 0 < t_x[0]


# --------------------------------------------------------------------------- #
# SE3
# --------------------------------------------------------------------------- #
class TestSE3:
    def test_look_at_forward(self):
        R = look_at_R([0, 0, 0], [0, 0, 2])
        f = R[:, 2]
        np.testing.assert_allclose(f, [0, 0, 1], atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_look_at_degenerate_up(self):
        R = look_at_R([0, 0, 0], [0, 1, 0])             # looking straight up
        assert np.isfinite(R).all()
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)

    def test_yaw_pitch_roundtrip(self):
        R = look_at_R([1, 1, 1], [2, 0.5, 0.3])
        yaw, pitch = rotation_to_yaw_pitch(R)
        f = R[:, 2]
        np.testing.assert_allclose(
            [math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)),
             math.sin(math.radians(pitch)),
             math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))], f, atol=1e-6)

    def test_look_at_torch_matches_numpy(self):
        eye = np.array([0.2, 0.3, 0.4])
        tgt = np.array([1.0, 0.1, -0.5])
        Rn = look_at_R(eye, tgt)
        Rt = look_at_R_torch(torch.tensor(eye), torch.tensor(tgt)).numpy()
        np.testing.assert_allclose(Rn, Rt, atol=1e-9)

    def test_sph_roundtrip(self):
        az, el, d = 0.7, 0.3, 2.0
        v = sph_offset(az, el, d)
        a2, e2, d2 = cart_to_sph(v)
        assert (a2, e2, d2) == pytest.approx((az, el, d), abs=1e-12)

    def test_angle_lerp_shortest_arc(self):
        assert angle_lerp(3.0, -3.0, 0.5) == pytest.approx(math.pi)

    def test_ease_endpoints(self):
        assert float(ease_in_out(torch.tensor(0.0))) == 0.0
        assert float(ease_in_out(torch.tensor(1.0))) == 1.0
        assert float(ease_in_out(torch.tensor(0.5))) == 0.5


# --------------------------------------------------------------------------- #
# Scene SDF union
# --------------------------------------------------------------------------- #
class TestSceneSDF:
    def _sdf(self):
        return SceneSDF([_box(), _box(cx=4.0)], [(np.array([-5.0, -1.0, -5.0]),
                                                 np.array([5.0, 3.0, 5.0]))])

    def test_union_min(self):
        sdf = self._sdf()
        # box half-extent 0.5 → face at z=0.5; point at z=1.0 is 0.5 outside
        assert sdf.numpy_query(np.array([0.0, 0.5, 1.0])) == pytest.approx(0.5)
        assert sdf.numpy_query(np.array([4.0, 0.5, 1.0])) == pytest.approx(0.5)

    def test_torch_numpy_parity(self):
        sdf = self._sdf()
        pts = np.array([[0.0, 0.5, 3.0], [4.0, 0.5, 3.0], [2.0, 0.5, 0.0]])
        tp = torch.tensor(pts, dtype=torch.float64)
        np.testing.assert_allclose(sdf(tp).detach().numpy(), sdf.numpy_query(pts))

    def test_soft_min(self):
        sdf = SceneSDF([_box()], [(np.array([-5.0] * 3), np.array([5.0] * 3))], soft_min_k=8.0)
        # far from seams soft-min ≈ min (box face at z=0.5, point at z=5.0)
        assert sdf.numpy_query(np.array([0.0, 0.5, 5.0])) == pytest.approx(4.5, abs=1e-3)

    def test_room_bounds(self):
        sdf = self._sdf()
        lo, hi = sdf.room_bounds_of_point(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(lo, [-5, -1, -5])


# --------------------------------------------------------------------------- #
# Subject track
# --------------------------------------------------------------------------- #
def _track():
    return SubjectTrack.from_dict({
        "subject_id": "actor", "height_m": 1.8,
        "frames": [{"t": 0.0, "pos": [0.0, 0.0, 0.0], "heading": 0.0},
                   {"t": 2.0, "pos": [2.0, 0.0, 0.0], "heading": math.pi / 2}]})


class TestSubject:
    def test_interp_position_and_clamp(self):
        s = _track().state_at(1.0)
        np.testing.assert_allclose(s.pos, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(_track().state_at(9.0).pos, [2.0, 0.0, 0.0])

    def test_heading_shortest_arc(self):
        s = _track().state_at(1.0)
        assert s.heading == pytest.approx(math.pi / 4)

    def test_key_point_height(self):
        assert _track().key_point(0.0, "head")[1] == pytest.approx(0.92 * 1.8)

    def test_capsule_sdf_surface(self):
        tr = _track()
        # on the capsule surface (radius 0.3) at chest height
        v = tr.capsule_sdf(np.array([0.3, 0.9, 0.0]), t=0.0, radius=0.3)
        assert v == pytest.approx(0.0, abs=1e-9)
        assert tr.capsule_sdf(np.array([0.0, 0.9, 0.0]), 0.0, 0.3) < 0
        assert tr.capsule_sdf(np.array([2.0, 0.9, 0.0]), 0.0, 0.3) > 0

    def test_capsule_torch_grad(self):
        tr = _track()
        p = torch.tensor([[0.4, 0.9, 0.0]], dtype=torch.float64, requires_grad=True)
        v = tr.capsule_sdf_torch(p, t=0.0)
        v.backward()
        assert torch.isfinite(p.grad).all()

    def test_strictly_increasing_validation(self):
        scene = Scene.from_dict({
            "scene_id": "x", "rooms": [],
            "structures": [], "objects": [], "doors": [],
            "subject_track": {"frames": [{"t": 1.0, "pos": [0, 0, 0], "heading": 0},
                                         {"t": 1.0, "pos": [1, 0, 0], "heading": 0}]}})
        assert any("increasing" in e for e in scene.validate())
