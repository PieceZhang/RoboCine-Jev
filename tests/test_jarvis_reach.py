"""Jarvis reachability tests — camera_reachable against measured workspace
couplings, the build-stage geometry clamps, and the anchor band filter.

Measured facts pinned here (FK sampling + IK, jvs_mock arm):
  * level looks need camera z ≥ ~0.55 m, steep down-looks z ≤ ~0.85 m;
  * +45° up-looks need z ≥ ~0.7 m;
  * the base may park anywhere SDF-clear within the arm extension grid."""

import math

import numpy as np
import pytest

from cinema_traj.common.obb import OBB
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core import anchors as A
from cinema_traj.core.jarvis import (JARVIS_DEMO_CAPS, Q_HI, Q_LO,
                                     camera_reachable)
from cinema_traj.core.trajectories import (OrbitSegment, Param, StaticSegment,
                                           TrackFollowSegment, DollySegment)

EMPTY = SceneSDF([], [])
CAPS = JARVIS_DEMO_CAPS


# --------------------------------------------------------------------------- #
# camera_reachable
# --------------------------------------------------------------------------- #
class TestCameraReachable:
    def test_reachable_mid_band_level_look(self):
        r = camera_reachable((0.5, 0.75, 0.5), 0.0, 0.0, EMPTY,
                             tangent_yaw=0.0, caps=CAPS)
        assert r is not None
        bx, bz, theta = r["base"]
        # base parked behind the camera along the heading, inside the arm grid
        d = math.hypot(0.5 - bx, 0.5 - bz)
        assert min(CAPS.arm_extension) - 1e-9 <= d <= max(CAPS.arm_extension) + 1e-9
        q = np.asarray(r["q"])
        assert np.all(q >= Q_LO - 1e-9) and np.all(q <= Q_HI + 1e-9)

    def test_above_envelope_unreachable(self):
        assert camera_reachable((0.5, 1.5, 0.5), 0.0, 0.0, EMPTY,
                                tangent_yaw=0.0, caps=CAPS) is None

    def test_level_look_at_low_z_unreachable(self):
        # measured: the wrist cannot hold a LEVEL look below ~0.55 m
        assert camera_reachable((0.5, 0.25, 0.5), 0.0, 0.0, EMPTY,
                                tangent_yaw=0.0, caps=CAPS) is None

    def test_up_look_needs_height(self):
        assert camera_reachable((0.5, 0.75, 0.5), 0.0, math.radians(45), EMPTY,
                                tangent_yaw=0.0, caps=CAPS) is not None
        assert camera_reachable((0.5, 0.45, 0.5), 0.0, math.radians(45), EMPTY,
                                tangent_yaw=0.0, caps=CAPS) is None

    def test_base_blocked_by_wall(self):
        # wall slab covering every candidate base parking spot (heading 0 →
        # base at z = 0.5 − d, d ∈ [0.2, 0.6] → z ∈ [−0.1, 0.3]) while leaving
        # the camera itself (z = 0.75) clear
        wall = OBB(center=np.array([0.5, 0.5, 0.1]), rotation=np.eye(3),
                   extents=np.array([3.0, 2.4, 0.5]))
        sdf = SceneSDF([wall], [])
        r = camera_reachable((0.5, 0.75, 0.5), 0.0, 0.0, sdf,
                             tangent_yaw=0.0, caps=CAPS)
        assert r is None

    def test_prefer_branch_keeps_base_continuous(self):
        r1 = camera_reachable((0.5, 0.75, 0.5), 0.0, 0.0, EMPTY,
                              tangent_yaw=0.0, caps=CAPS)
        # a small step forward with the previous branch preferred → same d
        r2 = camera_reachable((0.5, 0.75, 0.55), 0.0, 0.0, EMPTY,
                              tangent_yaw=0.0, caps=CAPS,
                              prefer=(r1["base"][2], r1["d"]))
        assert r2 is not None
        assert r2["d"] == pytest.approx(r1["d"])


# --------------------------------------------------------------------------- #
# Build-stage clamps
# --------------------------------------------------------------------------- #
def _orbit(height_value=1.5, height_hi=2.5, radius=1.0):
    seg = OrbitSegment("orbit_test", 6.0, 0.0, "living_room", "object", "obj",
                       direction=1.0, sweep_deg=90.0,
                       fix={"center": [0.5, 0.4, 0.5], "phi_s": 0.0},
                       free={"radius": Param(radius, 0.4, 2.5 * radius),
                             "height": Param(height_value, 0.25, height_hi),
                             "end_angle": Param(math.radians(90), 0.0, math.radians(180))})
    return seg


class TestBuildClamps:
    def test_orbit_height_clamped_into_band(self):
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        seg = _orbit(height_value=1.5)
        _clamp_segment_to_jarvis_workspace(seg, EMPTY, CAPS)
        z_lo, z_hi = CAPS.cinematic_z_band()
        p = seg.free["height"]
        assert p.hi == pytest.approx(z_hi)
        assert p.value <= z_hi + 1e-9

    def test_orbit_height_band_conflict_raises(self):
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        seg = _orbit(height_value=1.5, height_hi=1.5)
        seg.free["height"] = Param(1.5, 1.4, 1.5)      # entirely above the band
        with pytest.raises(ValueError, match="高度带"):
            _clamp_segment_to_jarvis_workspace(seg, EMPTY, CAPS)

    def test_orbit_base_circle_shrinks_radius(self):
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        # room wall at z = 1.4: the camera circle (center z 0.5 + r) still
        # fits at r ≈ 0.6, but the base companion circle (√(r² + d²) ahead of
        # the camera along −tangent… i.e. the parked base ring) hits the
        # SCENE_BOUNDS_MARGIN earlier — the jarvis sweep must clamp below 1.0
        sdf = SceneSDF([], [(-2.0 * np.ones(3), np.array([2.0, 2.8, 1.4]))])
        seg = _orbit(height_value=0.7, radius=1.0)
        seg.free["height"] = Param(0.7, 0.25, 0.9)
        _clamp_segment_to_jarvis_workspace(seg, sdf, CAPS)
        assert seg.free["radius"].hi < 1.0             # pushed inside the wall

    def test_dolly_ride_height_pinned(self):
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        seg = DollySegment("move_in_test", 6.0, 0.0, "living_room", "object", "obj",
                           fix={"center": [0.9, 1.1, 0.9], "dir": [0.0, 0.0, 1.0]},
                           free={"r_s": Param(1.2, 0.5, 1.8),
                                 "r_e": Param(0.66, 0.55, 4.0)})
        _clamp_segment_to_jarvis_workspace(seg, EMPTY, CAPS)
        z_lo, z_hi = CAPS.cinematic_z_band()
        assert seg.fix["cam_height"] == pytest.approx(min(max(1.1, z_lo), z_hi))
        import torch
        th = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
        pos = seg.positions(th).numpy()
        np.testing.assert_allclose(pos[:, 1], seg.fix["cam_height"], atol=1e-9)

    def test_static_anchor_height_clamped(self):
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        seg = StaticSegment("static_test", 6.0, 0.0, "living_room", "object", "obj",
                            fix={"anchor": [0.0, 1.6, 1.0], "center": [0.0, 0.4, 0.0]})
        _clamp_segment_to_jarvis_workspace(seg, EMPTY, CAPS)
        z_lo, z_hi = CAPS.cinematic_z_band()
        assert seg.fix["anchor"][1] == pytest.approx(z_hi)

    def test_track_segment_tightened_around_init(self):
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        seg = TrackFollowSegment("follow_test", 6.0, 0.0, "living_room",
                                 "subject_head", "actor",
                                 fix={"key_point": "head"},
                                 free={"az_rel": Param(0.0, -math.pi, math.pi),
                                       "elev": Param(0.35, 0.05, 1.2),
                                       "dist": Param(1.6, 0.8, 4.0)})
        _clamp_segment_to_jarvis_workspace(seg, EMPTY, CAPS)
        d = seg.free["dist"]
        el = seg.free["elev"]
        assert d.hi <= 1.25 * 1.6 + 1e-9               # dist range narrowed
        assert el.hi <= 1.2 + 1e-9
        # the product stays inside the cinematic band at the bound corners
        z_lo, z_hi = CAPS.cinematic_z_band()
        assert d.lo * math.sin(el.hi) <= z_hi + 0.2    # approximate guard, audit is exact


# --------------------------------------------------------------------------- #
# Anchor band filter
# --------------------------------------------------------------------------- #
class TestAnchorBand:
    def test_object_anchors_filtered_into_band(self):
        from cinema_traj.common.scene import load_scene
        scene = load_scene("cinema_traj/demo/demo_apartment.json")
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        obj = scene.object_by_id("table_01")
        raw = A.select_anchor(obj, scene, sdf, "overhead", "far")
        assert raw, "drone path must still find anchors"
        filt = A.select_anchor(obj, scene, sdf, "overhead", "far",
                               jarvis_caps=CAPS)
        z_lo, z_hi = CAPS.cinematic_z_band()
        assert filt
        for a in filt:
            assert z_lo - 1e-9 <= float(a.pos[1]) <= z_hi + 1e-9


# --------------------------------------------------------------------------- #
# Dolly 可行区间密集扫描（非单调可行集 — kujiale move_in_sofa 的根因）
# --------------------------------------------------------------------------- #
class TestDollyFeasibleRun:
    def _segment(self):
        return DollySegment("move_in_hole", 6.0, 0.0, "r1", "object", "obj",
                            fix={"center": [2.0, 0.4, 2.0], "dir": [0.0, 0.0, 1.0]},
                            free={"r_s": Param(1.0, 0.3, 3.0),
                                  "r_e": Param(2.0, 0.3, 3.0)})

    def _hole_scene(self):
        # slab blocking the base companion band (r + arm-ext ∈ [1.45, 2.35]
        # from the center along +Z → base z ∈ [3.45, 4.35]) at y ≤ 0.5 —
        # the camera itself (y = 0.7) stays clear above it
        slab = OBB(center=np.array([2.0, 0.25, 3.9]), rotation=np.eye(3),
                   extents=np.array([3.0, 0.5, 0.9]))
        room = [np.array([-5.0, 0.0, -5.0]), np.array([5.0, 3.0, 9.0])]
        return SceneSDF([slab], [room])

    def test_interior_hole_excluded_from_bounds(self):
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        seg = self._segment()
        seg.fix["cam_height"] = 0.7
        _clamp_segment_to_jarvis_workspace(seg, EMPTY, CAPS,
                                           base_sdf=self._hole_scene())
        rs, re_ = seg.free["r_s"], seg.free["r_e"]
        # both bounds land inside ONE feasible run — never straddling the hole
        # (measured: short-arm run ends ~0.95, long-arm run starts ~1.95)
        below = rs.hi <= 1.0 and re_.hi <= 1.0
        above = rs.lo >= 1.9 and re_.lo >= 1.9
        assert below or above, f"bounds straddle the hole: {rs} {re_}"

    def test_clamped_segment_passes_dense_verify(self):
        from cinema_traj.core.jarvis import verify_reachable_trajectory
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        seg = self._segment()
        seg.fix["cam_height"] = 0.7
        _clamp_segment_to_jarvis_workspace(seg, EMPTY, CAPS,
                                           base_sdf=self._hole_scene())
        rep = verify_reachable_trajectory([seg], None, self._hole_scene(), CAPS)
        assert rep["status"] == "ok", rep["reasons"][:3]

    def test_fully_blocked_line_raises(self):
        from cinema_traj.stages.build import _clamp_segment_to_jarvis_workspace
        wall = OBB(center=np.array([2.0, 0.25, 2.0]), rotation=np.eye(3),
                   extents=np.array([3.0, 0.5, 9.0]))
        room = [np.array([-5.0, 0.0, -5.0]), np.array([5.0, 3.0, 9.0])]
        sdf = SceneSDF([wall], [room])
        seg = self._segment()
        seg.fix["cam_height"] = 0.7
        with pytest.raises(ValueError, match="无任何底盘"):
            _clamp_segment_to_jarvis_workspace(seg, EMPTY, CAPS, base_sdf=sdf)


# --------------------------------------------------------------------------- #
# 拍摄段硬门：anchor 候选重试
# --------------------------------------------------------------------------- #
class TestShootingGate:
    def _scene(self):
        """茶几（大矮盒）+ 目标小盒。坏 anchor 悬在茶几上方 — 底盘伴行点
        全部落进桌面 SDF → camera_reachable 毫秒级拒绝（不烧 IK）。"""
        from cinema_traj.common.scene import ObjectNode, Room, Scene
        room = Room("r1", "room", np.array([-5.0, 0.0, -5.0]),
                    np.array([5.0, 3.0, 9.0]))
        table = ObjectNode("table1", "table", "r1",
                           OBB(center=np.array([2.0, 0.25, 2.0]),
                               rotation=np.eye(3),
                               extents=np.array([2.0, 0.5, 2.0])))
        obj = ObjectNode("obj1", "box", "r1",
                         OBB(center=np.array([2.0, 0.55, 2.6]),
                             rotation=np.eye(3),
                             extents=np.array([0.3, 0.3, 0.3])))
        return Scene(scene_id="t", rooms=[room], objects=[table, obj])

    def test_first_unreachable_candidate_falls_to_next(self):
        from cinema_traj.core.jarvis import build_base_sdf
        from cinema_traj.stages.build import _build_shooting_segment_verified
        scene = self._scene()
        obj = scene.object_by_id("obj1")
        bad1 = A.Anchor(np.array([2.0, 0.75, 2.0]))    # over the table
        bad2 = A.Anchor(np.array([1.5, 0.80, 2.4]))    # over the table too
        good = A.Anchor(np.array([2.0, 0.75, 3.4]))    # north of it, base clear
        traj_step = {"kind": "object_traj", "type": "static", "target": "obj1",
                     "duration_s": 2.0}
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        seg, anchor = _build_shooting_segment_verified(
            traj_step, [bad1, bad2, good], obj, scene, sdf,
            build_base_sdf(scene), CAPS, 0.0)
        np.testing.assert_allclose(seg.fix["anchor"], good.pos, atol=1e-9)

    def test_all_candidates_unreachable_raises(self):
        from cinema_traj.core.jarvis import build_base_sdf
        from cinema_traj.stages.build import _build_shooting_segment_verified
        scene = self._scene()
        obj = scene.object_by_id("obj1")
        bad1 = A.Anchor(np.array([2.0, 0.75, 2.0]))
        bad2 = A.Anchor(np.array([1.5, 0.80, 2.4]))
        bad3 = A.Anchor(np.array([2.4, 0.80, 1.6]))
        traj_step = {"kind": "object_traj", "type": "static", "target": "obj1",
                     "duration_s": 2.0}
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        with pytest.raises(ValueError, match="不可达采样点"):
            _build_shooting_segment_verified(
                traj_step, [bad1, bad2, bad3], obj, scene, sdf,
                build_base_sdf(scene), CAPS, 0.0)
