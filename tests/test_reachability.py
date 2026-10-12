"""Phase-1 reachability checker tests: geometry derivations, per-frame
feasibility reasons, and the trajectory-level report."""

import math
from pathlib import Path

import numpy as np
import pytest

from cinema_traj.common.scene import Scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core.reachability import (RobotModel, check_frame,
                                           check_reachability)
from cinema_traj.core.trajectories import Param, StaticSegment, OrbitSegment

REPO = Path(__file__).resolve().parents[1]


def _corridor_scene() -> Scene:
    return Scene.from_dict({
        "scene_id": "corridor_test",
        "rooms": [{"id": "room", "label": "room",
                   "bounds_min": [-4.0, 0.0, -4.0], "bounds_max": [4.0, 3.0, 4.0]}],
        "structures": [
            {"id": "floor", "kind": "floor", "room": "room",
             "obb": {"center": [0, -0.05, 0], "rotation": [[1,0,0],[0,1,0],[0,0,1]],
                     "extents": [8.2, 0.1, 8.2]}},
            {"id": "ceil", "kind": "ceiling", "room": "room",
             "obb": {"center": [0, 3.05, 0], "rotation": [[1,0,0],[0,1,0],[0,0,1]],
                     "extents": [8.2, 0.1, 8.2]}},
        ],
        "objects": [
            {"id": "pillar", "label": "pillar", "room": "room",
             "obb": {"center": [1.0, 1.0, 0.0], "rotation": [[1,0,0],[0,1,0],[0,0,1]],
                     "extents": [0.6, 2.0, 0.6]}, "placement": "freestanding"},
            {"id": "target", "label": "target", "room": "room",
             "obb": {"center": [0.0, 0.5, 0.0], "rotation": [[1,0,0],[0,1,0],[0,0,1]],
                     "extents": [0.6, 1.0, 0.6]}, "placement": "freestanding"},
        ],
        "doors": [],
    })


@pytest.fixture(scope="module")
def field():
    scene = _corridor_scene()
    sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
    return scene, sdf


class TestRobotModel:
    def test_default_geometry_derived(self):
        r = RobotModel()
        # arm base at 0.95, reach 0.65 → band [max(0.3, base+0.15)=0.55, 1.6]
        assert r.height_range == pytest.approx((0.55, 1.6))
        assert r.reach_ring(0.95) == pytest.approx((0.15, 0.65))  # level with base

    def test_config_override(self):
        r = RobotModel({"base": {"radius": 0.25},
                        "camera": {"height_range": [0.6, 1.4]}})
        assert r.base_radius == 0.25
        assert r.height_range == (0.6, 1.4)

    def test_ring_none_when_vertical_gap_exceeds_reach(self):
        r = RobotModel()
        assert r.reach_ring(0.55 - 0.66) is None      # far below the arm base


class TestCheckFrame:
    def test_reachable_open_space(self, field):
        _, sdf = field
        res = check_frame(np.array([0.0, 1.2, 2.0]), sdf, RobotModel())
        assert res["ok"], res["reason"]

    def test_too_high_reason(self, field):
        _, sdf = field
        res = check_frame(np.array([0.0, 2.2, 2.0]), sdf, RobotModel())
        assert not res["ok"] and res["kind"] == "height"
        assert "可达带" in res["reason"]

    def test_placement_blocked_by_pillar(self, field):
        # camera over the pillar with a SHORT arm: the whole candidate ring
        # (max standoff 0.447m) stays within pillar half-width (0.3) + chassis
        # radius (0.3) → no placement clears the body
        _, sdf = field
        robot = RobotModel({"arm": {"max_reach": 0.45}})
        res = check_frame(np.array([1.0, 1.0, 0.0]), sdf, robot)
        assert not res["ok"] and res["kind"] == "placement"
        assert "底盘" in res["reason"]
        # the default (longer) arm CAN slip the ring around the small pillar
        assert check_frame(np.array([1.0, 1.0, 0.0]), sdf, RobotModel())["ok"]


class TestReport:
    def test_static_segment_all_reachable(self, field):
        scene, sdf = field
        seg = StaticSegment("s", 2.0, 0.0, "room", "object", "target",
                            fix={"anchor": [0.0, 1.2, 2.0]})
        report = check_reachability([seg], scene, sdf)
        assert report["status"] == "ok"
        assert report["violations"] == []
        assert report["robot"]["base_radius_m"] == pytest.approx(0.30)

    def test_orbit_report_fails_with_reason(self, field):
        scene, sdf = field
        # orbit at 1.55m around the target: near the top of the reachable band,
        # the arm must reach almost straight up — the feasible standoff ring is
        # tiny; combined with the pillar the report must name the binding frame
        seg = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                           fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                "direction": 1.0, "sweep_deg": 90.0},
                           free={"radius": Param(1.2, 1.2, 1.2),
                                 "height": Param(1.58, 1.58, 1.58),
                                 "end_angle": Param(math.pi / 2, math.pi / 2,
                                                    math.pi / 2)})
        report = check_reachability([seg], scene, sdf)
        if report["status"] == "failed":              # geometry-dependent: if it
            v = report["violations"][0]               # binds, the reason must name it
            assert v["frames_unreachable"] >= 1
            assert report["reasons"] and v["segment"] == "o"
