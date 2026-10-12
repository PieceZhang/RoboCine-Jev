"""Trajectory library + anchors + plan schema tests (pure functions, no IO)."""

import math

import numpy as np
import pytest
import torch

from cinema_traj.common import constants as C
from cinema_traj.common.scene import load_scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core import anchors as A
from cinema_traj.core.plan_schema import normalize_plan, validate_plan
from cinema_traj.core.trajectories import (ArcTransition, DollySegment, OrbitSegment,
                                           Param, Segment, StaticSegment,
                                           TrackFollowSegment, ZoomSegment)

DEMO = "cinema_traj/demo/demo_apartment.json"


@pytest.fixture(scope="module")
def scene():
    return load_scene(DEMO)


@pytest.fixture(scope="module")
def sdf(scene):
    return SceneSDF(scene.collision_obbs(), scene.room_bounds())


# --------------------------------------------------------------------------- #
# Trajectory library
# --------------------------------------------------------------------------- #
class TestTrajectories:
    def test_static_and_registry_roundtrip(self):
        seg = StaticSegment("s", 2.0, 0.0, "living_room", "object", "table_01",
                            fix={"anchor": [1.0, 1.5, 2.0], "center": [0, 0, 0]})
        th = torch.linspace(0, 1, 5, dtype=torch.float64)
        p = seg.positions(th)
        assert p.shape == (5, 3)
        np.testing.assert_allclose(p.numpy(), np.tile([1.0, 1.5, 2.0], (5, 1)))
        d = seg.to_dict()
        seg2 = Segment.from_dict(d)
        np.testing.assert_allclose(seg2.positions(th).numpy(), p.numpy())

    def test_orbit_geometry(self):
        c = [0.0, 0.5, 0.0]
        seg = OrbitSegment("o", 4.0, 0.0, "r", "object", "t",
                           fix={"center": c, "phi_s": 0.0, "direction": 1.0,
                                "sweep_deg": 90.0},
                           free={"radius": Param(2.0, 1.0, 3.0),
                                 "height": Param(1.5, 0.5, 2.5),
                                 "end_angle": Param(math.pi / 2, 0.1, math.pi)})
        th = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        p = seg.positions(th).numpy()
        # start at phi=0: (r, h, 0); end at phi=pi/2: (0, h, r)
        np.testing.assert_allclose(p[0], [2.0, 1.5, 0.0], atol=1e-9)
        np.testing.assert_allclose(p[2], [0.0, 1.5, 2.0], atol=1e-9)
        assert seg.endpoints()[0][0] == pytest.approx(2.0)

    def test_dolly_eases_between_radii(self):
        seg = DollySegment("d", 2.0, 0.0, "r", "object", "t",
                           fix={"center": [0, 0, 0], "dir": [1.0, 0.0, 0.0]},
                           free={"r_s": Param(3.0, 1.0, 4.0),
                                 "r_e": Param(1.0, 0.5, 3.0)})
        p0, p1 = seg.endpoints()
        np.testing.assert_allclose(p0, [3.0, 0.0, 0.0])
        np.testing.assert_allclose(p1, [1.0, 0.0, 0.0])

    def test_arc_bulge_and_rebuild(self):
        seg = ArcTransition("a", 3.0, 0.0, "r", "object", "t",
                            fix={"p_s": [0.0, 1.0, 0.0], "p_e": [4.0, 1.0, 0.0],
                                 "chord": 4.0, "n_hat": [0.0, 0.0, 1.0]},
                            free={"alpha_deg": Param(45.0, -89.0, 89.0)})
        mid = seg.positions(torch.tensor([0.5], dtype=torch.float64)).numpy()[0]
        # bulge = (L/2)·tan(α)·sin(π·0.5) = 2·1·1 = 2 along n̂
        np.testing.assert_allclose(mid, [2.0, 1.0, 2.0], atol=1e-9)
        seg.rebuild(np.array([0.0, 1.0, 0.0]), np.array([2.0, 1.0, 0.0]))
        assert seg.free["alpha_deg"].value == 0.0        # reset for pass 2

    def test_zoom_fov_envelope(self):
        seg = ZoomSegment("z", 2.0, 0.0, "r", "object", "t",
                          fix={"anchor": [0, 1, 3], "center": [0, 1, 0],
                               "m_peak": 2.0, "fov0_deg": 55.0})
        th = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        fov = seg.fov_deg(th).numpy()
        assert fov[0] == pytest.approx(55.0)
        assert fov[1] < fov[0] < fov[2] or fov[1] < fov[2]   # peak zoom mid-shot
        assert fov[0] == pytest.approx(fov[2])               # returns to f0

    def test_track_follow_keeps_offset(self, scene):
        seg = TrackFollowSegment("f", 2.0, 0.0, "living_room", "subject_head", "actor_01",
                                 fix={},
                                 free={"az_rel": Param(0.0, -math.pi, math.pi),
                                       "elev": Param(0.2, 0.05, 1.2),
                                       "dist": Param(2.0, 0.8, 4.0)})
        th = torch.linspace(0, 1, 5, dtype=torch.float64)
        p = seg.positions(th, None, scene.subject).numpy()
        for i, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0]):
            s = scene.subject.state_at(t)
            # az_rel=0 → camera stays at fixed spherical offset in FRONT of subject
            fwd = np.array([math.cos(s.heading), 0.0, math.sin(s.heading)])
            rel = p[i] - s.pos
            horiz = np.linalg.norm(rel[[0, 2]])
            assert rel @ fwd == pytest.approx(horiz, abs=1e-9)
            assert horiz == pytest.approx(2.0 * math.cos(0.2), rel=1e-6)
            assert rel[1] == pytest.approx(2.0 * math.sin(0.2), rel=1e-6)

    def test_write_back_clamps(self):
        seg = StaticSegment("s", 1.0, 0.0, "r", "object", "t",
                            fix={"anchor": [0, 0, 0], "center": [0, 0, 0]},
                            free={"x": Param(0.0, -1.0, 1.0)})
        seg.write_back({"x": torch.tensor(5.0)})
        assert seg.free["x"].value == 1.0


# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #
class TestAnchors:
    def test_viewing_distance_clamped(self, scene):
        obj = scene.object_by_id("table_01")
        room = scene.room_of(obj)
        d = A.viewing_distance(obj, room, "medium")
        assert 0.3 <= d <= 3.0
        close = A.viewing_distance(obj, room, "close")
        far = A.viewing_distance(obj, room, "far")
        assert close < d < far

    def test_anchor_clear_and_in_room(self, scene, sdf):
        for oid in ("table_01", "sofa_01", "chandelier_01"):
            obj = scene.object_by_id(oid)
            top = A.select_anchor(obj, scene, sdf, "medium", "medium")
            assert top, f"no anchor for {oid}"
            a = top[0]
            assert sdf.numpy_query(a.pos) >= C.ANCHOR_COLLISION_MARGIN
            room = scene.room_of(obj)
            assert room.contains(a.pos), f"anchor for {oid} outside its room"

    def test_wall_object_anchor_from_interior(self, scene, sdf):
        obj = scene.object_by_id("sofa_01")            # against north wall (z≈-2)
        a = A.select_anchor(obj, scene, sdf, "medium", "medium")[0]
        assert a.pos[2] > obj.obb.center[2]            # from the room side, not the wall

    def test_nearest_neighbor_order(self, scene):
        objs = [scene.object_by_id(i) for i in ("table_01", "sofa_01", "plant_01")]
        start = np.array([-2.0, 1.0, 1.8])
        order = A.nearest_neighbor_order(start, objs)
        dists = {o.id: np.linalg.norm(o.obb.center - start) for o in objs}
        assert order[0] == min(dists, key=dists.get)   # greedy nearest first

    def test_subject_anchor(self, scene, sdf):
        assert scene.subject is not None
        a = A.select_subject_anchor(scene.subject, 0.0, scene, sdf)
        assert sdf.numpy_query(a.pos) >= C.ANCHOR_COLLISION_MARGIN

    def test_subject_anchor_azimuth_pinned(self, scene, sdf):
        """azimuth_deg pins the camera to one heading-relative side:
        0° = in front of the subject, 180° = behind (same frame as
        trajectories._heading_offset)."""
        import math
        subj = scene.subject
        assert subj is not None
        t = 0.0
        state = subj.state_at(t)
        head = subj.key_point(t, "head")
        front = A.select_subject_anchor(subj, t, scene, sdf, azimuth_deg=0.0)
        behind = A.select_subject_anchor(subj, t, scene, sdf, azimuth_deg=180.0)
        for a in (front, behind):
            assert sdf.numpy_query(a.pos) >= C.ANCHOR_COLLISION_MARGIN
        h = state.heading
        # heading direction in world (heading 0 → +X, toward +Z)
        hd = np.array([math.cos(h), 0.0, math.sin(h)])
        front_along = float(np.dot(np.asarray(front.pos) - head, hd))
        behind_along = float(np.dot(np.asarray(behind.pos) - head, hd))
        assert front_along > 0, f"azimuth 0 must be in front, got {front_along:.2f}"
        assert behind_along < 0, f"azimuth 180 must be behind, got {behind_along:.2f}"


# --------------------------------------------------------------------------- #
# Plan schema
# --------------------------------------------------------------------------- #
def _good_plan(scene):
    sofa = "sofa_01" if scene.object_by_id("sofa_01") else scene.objects[0].id
    return normalize_plan({
        "prompt": "demo", "object_sequence": [sofa],
        "steps": [
            {"kind": "anchor", "target": sofa, "elevation": "medium", "distance": "medium"},
            {"kind": "object_traj", "target": sofa, "type": "orbit_quarter", "duration_s": 5.0},
        ]}, scene)


class TestPlanSchema:
    def test_good_plan_valid(self, scene):
        assert validate_plan(_good_plan(scene), scene) == []

    def test_missing_transition_rejected(self, scene):
        sofa2 = "table_01"
        plan = _good_plan(scene)
        plan["steps"] += [{"kind": "anchor", "target": sofa2, "elevation": "low",
                           "distance": "far"},
                          {"kind": "object_traj", "target": sofa2, "type": "static",
                           "duration_s": 2.0}]
        errs = validate_plan(plan, scene)
        assert any("transition" in e for e in errs)

    def test_placement_constraint(self, scene):
        plan = _good_plan(scene)
        plan["steps"][1]["type"] = "orbit_full"        # sofa is wall-placed
        errs = validate_plan(plan, scene)
        assert any("forbidden" in e for e in errs)

    def test_unknown_target(self, scene):
        plan = _good_plan(scene)
        plan["steps"][0]["target"] = "unicorn_99"
        errs = validate_plan(plan, scene)
        assert any("unknown target" in e for e in errs)

    def test_subject_step_without_track(self, scene):
        plan = _good_plan(scene)
        plan["steps"][1]["kind"] = "subject_traj"
        plan["steps"][1]["type"] = "track_follow"
        static_scene = type(scene).from_dict({**scene.to_dict()} | {"subject_track": None})
        errs = validate_plan(plan, static_scene)
        assert any("no subject_track" in e for e in errs)

    def test_zh_alias_normalization(self, scene):
        plan = normalize_plan({"steps": [{"运镜": "锚点", "物体": "sofa_01",
                                          "高度": "低", "距离": "近"}]}, scene)
        assert plan["steps"][0]["kind"] == "anchor"
        assert plan["steps"][0]["elevation"] == "low"
        assert plan["steps"][0]["distance"] == "close"
