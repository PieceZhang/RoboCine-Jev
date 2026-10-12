"""Cost terms + two-pass optimizer tests on synthetic scenes, plus the
end-to-end demo pipeline (MockPlanner, reduced iterations, CPU)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cinema_traj.common.scene import Scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core import costs
from cinema_traj.core.optimizer import OptCfg, door_route, optimize_segment, optimize_trajectory
from cinema_traj.core.trajectories import ArcTransition, OrbitSegment, Param
from cinema_traj.stages.build import build_trajectory
from cinema_traj.stages.plan import MockPlanner

DEMO = Path("cinema_traj/demo/demo_apartment.json")
REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Synthetic scene: one room with a box between the camera arc and the target
# --------------------------------------------------------------------------- #
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


class TestCosts:
    def test_sdf_cost_clear_vs_collision(self):
        sdf = SceneSDF([], [(np.array([-5.0, -1.0, -5.0]), np.array([5.0] * 3))])
        clear = torch.tensor([[0.0, 1.0, 3.0]] * 4, dtype=torch.float64)
        assert float(costs.cost_sdf(clear, sdf)) == 0.0
        from cinema_traj.common.obb import OBB
        sdf2 = SceneSDF([OBB(np.zeros(3), np.eye(3), np.ones(3))],
                        [(np.array([-5.0] * 3), np.array([5.0] * 3))])
        inside = torch.tensor([[0.0, 0.0, 0.0]] * 4, dtype=torch.float64)
        assert float(costs.cost_sdf(inside, sdf2)) > 0.0

    def test_occlusion_cost_detects_blocker(self):
        from cinema_traj.common.obb import OBB
        blocker = OBB(np.array([0.0, 0.0, 0.0]), np.eye(3), np.array([1.0, 1.0, 1.0]))
        sdf = SceneSDF([blocker], [(np.array([-5.0] * 3), np.array([5.0] * 3))])
        cam = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float64)
        tgt = torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float64)
        blocked = float(costs.cost_occl(cam, tgt, sdf))
        clear_cam = torch.tensor([[3.0, 2.0, 3.0]], dtype=torch.float64)
        clear = float(costs.cost_occl(clear_cam, tgt, sdf))
        assert blocked > clear

    def test_boundary_barrier(self):
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([4.0, 3.0, 4.0]))
        near = torch.tensor([[0.05, 1.5, 2.0]], dtype=torch.float64)
        far = torch.tensor([[2.0, 1.5, 2.0]], dtype=torch.float64)
        assert float(costs.cost_bnd(near, bounds)) > float(costs.cost_bnd(far, bounds))


class TestOptimizer:
    def test_orbit_pushed_clear_of_pillar(self):
        scene = _corridor_scene()
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        # orbit at radius 1.1 from phi=0 clips the pillar at (1, 1, 0)
        seg = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                           fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                "direction": 1.0, "sweep_deg": 90.0},
                           free={"radius": Param(1.1, 0.8, 3.0),
                                 "height": Param(1.0, 0.3, 2.6),
                                 "end_angle": Param(math.pi / 2, 0.3, math.pi)})
        th, _ = seg.sample_times()
        pre = sdf.numpy_query(seg.positions(th).numpy())
        assert pre.min() < 0.0, "test setup: initial arc should clip the pillar"
        res = optimize_segment(seg, scene, sdf, None, True,
                               OptCfg({"max_iters": 400}))
        post = sdf.numpy_query(seg.positions(th).numpy())
        assert post.min() >= -1e-6, f"still colliding ({res})"

    def test_orbit_full_and_half_spans_not_wrapped(self):
        # regression: wrap_angle folded ±360° to a zero span and flipped the
        # sign of exactly-180° sweeps — orbit_full must traverse ~2πr of arc
        def _arc_length(seg):
            th, _ = seg.sample_times()
            p = seg.positions(th).numpy()
            return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

        full = OrbitSegment("f", 8.0, 0.0, "room", "object", "target",
                            fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                 "direction": 1.0, "sweep_deg": 360.0},
                            free={"radius": Param(1.5, 0.8, 3.0),
                                  "height": Param(1.0, 0.3, 2.6),
                                  "end_angle": Param(2.0 * math.pi, 0.3, 1.6 * 2.0 * math.pi)})
        assert _arc_length(full) == pytest.approx(2.0 * math.pi * 1.5, rel=0.01)

        half = OrbitSegment("h", 4.0, 0.0, "room", "object", "target",
                            fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                 "direction": -1.0, "sweep_deg": 180.0},
                            free={"radius": Param(1.5, 0.8, 3.0),
                                  "height": Param(1.0, 0.3, 2.6),
                                  "end_angle": Param(-math.pi, -1.6 * math.pi, -0.3)})
        th, _ = half.sample_times()
        mid = half.positions(th).numpy()[len(th) // 2]
        # directed sweep −π from phi_s=0: halfway must be at φ=−π/2, i.e. −z side
        assert mid[2] < 0.0, "180° sweep direction was flipped by wrapping"
        assert _arc_length(half) == pytest.approx(math.pi * 1.5, rel=0.01)

    def test_orbit_end_angle_receives_gradient(self):
        # regression: float(phi_e) detached the graph → grad [.., .., 0.0] and
        # end_angle stayed frozen for the whole optimization
        seg = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                           fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                "direction": 1.0, "sweep_deg": 90.0},
                           free={"radius": Param(1.5, 0.8, 3.0),
                                 "height": Param(1.0, 0.3, 2.6),
                                 "end_angle": Param(math.pi / 2, 0.3, math.pi)})
        theta = torch.tensor([1.5, 1.0, math.pi / 2], dtype=torch.float64,
                             requires_grad=True)
        th, _ = seg.sample_times()
        params = {"radius": theta[0], "height": theta[1], "end_angle": theta[2]}
        seg.positions(th, params).sum().backward()
        assert float(theta.grad[2]) != 0.0, "end_angle gradient is cut off"

    def test_orbit_radius_clamped_to_room_clearance(self):
        # PR #2 known issue: paper radius bounds (×anchor distance) admit only
        # colliding circles around a subject near furniture/walls — the clamp
        # must size the bounds to the actual clearance instead
        from cinema_traj.stages.build import (_clamp_orbit_radius_to_clearance,
                                              _orbit_feasible_radius)
        scene = _corridor_scene()
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())

        # open center: a large circle can clear the small pillar entirely;
        # the room bound (±4 − 0.3 margin) is what limits r
        r = _orbit_feasible_radius(sdf, np.array([0.0, 1.0, 0.0]), 1.0, 3.0, 0.35)
        assert r == 3.0                                     # nothing binds below

        # near the +x wall: outside-wall space is NOT free (OBB union reads it
        # as positive SDF; the room bound is the real constraint)
        r = _orbit_feasible_radius(sdf, np.array([3.6, 1.0, 0.0]), 1.0, 3.0, 0.35)
        assert r is None

        # off-center: the pillar face + wall clamp a nominal far orbit
        seg = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                           fix={"center": [2.5, 0.5, 0.0], "phi_s": 0.0,
                                "direction": 1.0, "sweep_deg": 360.0},
                           free={"radius": Param(2.5, 1.5, 4.0),      # nominal far
                                 "height": Param(1.0, 0.3, 2.6),
                                 "end_angle": Param(2 * math.pi, 0.3, 1.6 * 2 * math.pi)})
        _clamp_orbit_radius_to_clearance(seg, sdf)
        p = seg.free["radius"]
        assert p.hi < 1.3, "radius bounds not clamped to local clearance"
        assert p.lo <= p.value <= p.hi

    def test_orbit_infeasible_position_raises(self):
        from cinema_traj.stages.build import _orbit_feasible_radius
        scene = _corridor_scene()
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        assert _orbit_feasible_radius(sdf, np.array([3.85, 1.0, 0.0]), 1.0,
                                      3.0, 0.35) is None

    def test_arc_bends_around_obstacle(self):
        scene = _corridor_scene()
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        # straight line from (0.9, 1.2, 2) to (0.9, 1.2, -2) passes through the
        # pillar at (1, 1, 0) (off-center, so the SDF gradient is non-zero) —
        # the arc must bend around it
        arc = ArcTransition("a", 2.0, 0.0, "room", "object", "target",
                            fix={"p_s": [0.9, 1.2, 2.0], "p_e": [0.9, 1.2, -2.0],
                                 "chord": 4.0, "n_hat": [1.0, 0.0, 0.0]},
                            free={"alpha_deg": Param(0.0, -89.0, 89.0)})
        th, _ = arc.sample_times()
        assert sdf.numpy_query(arc.positions(th).numpy()).min() < 0.0
        optimize_segment(arc, scene, sdf, None, False, OptCfg({"max_iters": 300}))
        assert sdf.numpy_query(arc.positions(th).numpy()).min() >= -1e-6

    def test_verification_ok_when_collision_free(self):
        from cinema_traj.stages.optimize import verify_collision_free
        scene = _corridor_scene()
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        # radius band that CAN clear the pillar (test above proves it)
        seg = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                           fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                "direction": 1.0, "sweep_deg": 90.0},
                           free={"radius": Param(1.1, 0.8, 3.0),
                                 "height": Param(1.0, 0.3, 2.6),
                                 "end_angle": Param(math.pi / 2, 0.3, math.pi)})
        optimize_segment(seg, scene, sdf, None, True, OptCfg({"max_iters": 400}))
        report = verify_collision_free([seg], scene, sdf)
        assert report["status"] == "ok"
        assert report["violations"] == []

    def test_verification_fails_with_reason_when_infeasible(self):
        from cinema_traj.stages.optimize import verify_collision_free
        scene = _corridor_scene()
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        # every allowed radius (0.9–1.2) clips the pillar (0.7–1.3 from the
        # center): no parameter setting is collision-free → hard failure
        seg = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                           fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                "direction": 1.0, "sweep_deg": 360.0},
                           free={"radius": Param(1.0, 0.9, 1.2),
                                 "height": Param(1.0, 0.9, 1.1),
                                 "end_angle": Param(2 * math.pi, 0.3, 1.6 * 2 * math.pi)})
        optimize_segment(seg, scene, sdf, None, True, OptCfg({"max_iters": 200}))
        report = verify_collision_free([seg], scene, sdf)
        assert report["status"] == "failed"
        v = report["violations"][0]
        assert v["segment"] == "o"
        assert v["worst_clearance_m"] < 0.0
        assert "pillar" in v["nearest_obstacle"]
        assert report["reasons"] and "o" in report["reasons"][0]

    def test_bypass_when_initially_clean(self):
        # SVSDF mid-end bypass: θ₀ collision-free (occl off → clean) → the
        # optimizer must not run a single iteration
        scene = _corridor_scene()
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        seg = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                           fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                "direction": 1.0, "sweep_deg": 90.0},
                           free={"radius": Param(2.5, 0.8, 3.0),
                                 "height": Param(1.5, 0.3, 2.6),
                                 "end_angle": Param(math.pi / 2, 0.3, math.pi)})
        r0 = optimize_segment(seg, scene, sdf, None, False,
                              OptCfg({"max_iters": 500}))
        assert r0["reason"] == "bypassed" and r0["iters"] == 0
        assert r0["collision_rate"] == 0.0
        # and with the bypass disabled the same segment still stops converged
        seg2 = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                            fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                 "direction": 1.0, "sweep_deg": 90.0},
                            free={"radius": Param(2.5, 0.8, 3.0),
                                  "height": Param(1.5, 0.3, 2.6),
                                  "end_angle": Param(math.pi / 2, 0.3, math.pi)})
        r1 = optimize_segment(seg2, scene, sdf, None, False,
                              OptCfg({"max_iters": 500, "bypass_if_clean": False}))
        assert r1["reason"] == "converged" and r1["iters"] < 500

    def test_occlusion_plateau_stops_early(self):
        # a blocker the free params cannot move away: occlusion stays > 0 while
        # collisions clear → must stop at occl_plateau, not burn max_iters
        from cinema_traj.common.obb import OBB
        scene = _corridor_scene()
        scene.objects.append(type(scene.objects[0])(
            "wall_behind", "wall", "room",
            OBB(np.array([-2.0, 1.0, 0.0]), np.eye(3), np.array([0.2, 2.0, 2.0])),
            "freestanding"))
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        # orbit on the FAR side of the target from the wall: the wall still
        # occludes the target for part of the arc regardless of radius/height
        seg = OrbitSegment("o", 4.0, 0.0, "room", "object", "target",
                           fix={"center": [0.0, 0.5, 0.0], "phi_s": 0.0,
                                "direction": 1.0, "sweep_deg": 90.0},
                           free={"radius": Param(2.5, 2.0, 2.8),
                                 "height": Param(1.5, 1.2, 1.8),
                                 "end_angle": Param(math.pi / 2, 0.3, math.pi)})
        r = optimize_segment(seg, scene, sdf, None, True,
                             OptCfg({"max_iters": 1000, "occl_plateau_iters": 50}))
        assert r["reason"] == "occl_plateau"
        assert r["iters"] < 1000
        assert r["collision_rate"] == 0.0

    def test_cross_room_door_routing(self):
        scene = Scene.from_dict({
            "scene_id": "two_rooms",
            "rooms": [{"id": "a", "label": "a", "bounds_min": [-4, 0, -2], "bounds_max": [0, 3, 2]},
                      {"id": "b", "label": "b", "bounds_min": [0, 0, -2], "bounds_max": [4, 3, 2]}],
            "structures": [],
            "objects": [],
            "doors": [{"id": "d1", "kind": "door", "rooms": ["a", "b"],
                       "obb": {"center": [0, 1.05, 0], "rotation": [[1,0,0],[0,1,0],[0,0,1]],
                               "extents": [0.12, 2.1, 0.9]}}],
        })
        wps = door_route(scene, np.array([-2.0, 1.0, 0.0]), np.array([2.0, 1.0, 0.0]), "a", "b")
        assert len(wps) == 1
        np.testing.assert_allclose(wps[0][:2], [0.0, 1.0], atol=0.35)   # near door center height


# --------------------------------------------------------------------------- #
# End-to-end on the demo scene (mock planner, fast iterations)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def demo_run():
    scene = Scene.from_dict(json.loads(DEMO.read_text(encoding="utf-8")))
    prompt = (REPO / "cinema_traj/demo/demo_prompt.txt").read_text(encoding="utf-8")
    plan = MockPlanner().plan(prompt, scene)
    segments, anchors = build_trajectory(scene, plan)
    sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
    cfg = OptCfg({"max_iters": 250})
    segments, log = optimize_trajectory(segments, scene, sdf, scene.subject, cfg)
    return scene, plan, segments, log, sdf


class TestEndToEnd:
    def test_mock_plan_validates(self, demo_run):
        scene, plan, *_ = demo_run
        assert validate_plan_safe(plan, scene) == []

    def test_plan_is_1_3_1(self, demo_run):
        _, plan, *_ = demo_run
        kinds = [s["kind"] for s in plan["steps"]]
        assert kinds[0] == "anchor"
        n = sum(1 for k in kinds if k == "anchor")
        assert len(kinds) == 3 * n - 1

    def test_subject_tracking_segments_present(self, demo_run):
        _, plan, segments, *_ = demo_run
        assert any(s.type.startswith("track_") for s in segments)

    def test_no_collision_after_optimization(self, demo_run):
        scene, _, segments, _, sdf = demo_run
        worst = 1e9
        for seg in segments:
            th, t_abs = seg.sample_times()
            p = seg.positions(th, None, scene.subject).numpy()
            worst = min(worst, float(sdf.numpy_query(p).min()))
        assert worst >= -1e-6, f"min clearance {worst}"

    def test_track_keeps_subject_in_front(self, demo_run):
        """During track segments the camera-to-subject distance stays sane and
        the camera never penetrates the subject capsule."""
        scene, _, segments, _, _ = demo_run
        for seg in segments:
            if not seg.type.startswith("track_"):
                continue
            th, t_abs = seg.sample_times()
            p = seg.positions(th, None, scene.subject).numpy()
            for i in range(len(t_abs)):
                t = float(t_abs[i])
                s = scene.subject.state_at(t)
                d = np.linalg.norm(p[i][:2] - s.pos[:2])
                assert 0.7 < d < 4.5
                cap = scene.subject.capsule_sdf(p[i][None], t)
                assert float(cap[0]) > -0.05

    def test_times_contiguous(self, demo_run):
        _, _, segments, *_ = demo_run
        for a, b in zip(segments, segments[1:]):
            assert b.t_start == pytest.approx(a.t_start + a.duration, abs=1e-9)


def validate_plan_safe(plan, scene):
    from cinema_traj.core.plan_schema import validate_plan
    return validate_plan(plan, scene)


# --------------------------------------------------------------------------- #
# arc 离散 α 兜底（Adam 单参数卡壳时的碰撞+可达双目标网格搜索）
# --------------------------------------------------------------------------- #
class TestArcAlphaGridSearch:
    def _arc(self):
        from cinema_traj.core.trajectories import ArcTransition
        seg = ArcTransition("arc_t", 3.0, 0.0, "r", "object", "o")
        seg.rebuild(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        return seg

    def test_prefers_collision_free_alpha(self):
        from cinema_traj.core.optimizer import _arc_alpha_grid_search
        seg = self._arc()
        seen = []

        def evaluate(s):
            a = float(s.free["alpha_deg"].value)
            seen.append(a)
            return {"collision": 0.0 if abs(a) >= 40 else 0.3, "reachable": 1.0}

        out = _arc_alpha_grid_search(seg, evaluate, current=10.0)
        assert out is not None
        assert out["collision"] == 0.0
        assert abs(out["alpha_deg"]) >= 40
        assert abs(float(seg.free["alpha_deg"].value)) >= 40
        assert 10.0 in seen                              # Adam's α is a candidate

    def test_all_bad_keeps_smallest_alpha(self):
        from cinema_traj.core.optimizer import _arc_alpha_grid_search
        seg = self._arc()

        def evaluate(s):
            return {"collision": 0.5, "reachable": 0.0}  # everything fails

        out = _arc_alpha_grid_search(seg, evaluate, current=30.0)
        assert out is not None and out["alpha_deg"] == 0.0   # |α| tie-break
        assert float(seg.free["alpha_deg"].value) == 0.0

    def test_empty_evaluation_returns_none(self):
        from cinema_traj.core.optimizer import _arc_alpha_grid_search
        seg = self._arc()
        assert _arc_alpha_grid_search(seg, lambda s: None, current=0.0) is None


class TestDollyRunClamp:
    """Drone-platform dolly feasibility (mirror of the orbit clamp): the
    camera flies the WHOLE [r_s, r_e] run, so a subject close-up floor of
    r_e ≥ 0.55 must not park it inside furniture the line grazes
    (frl_apartment_0 console regression: 19% collision before the clamp)."""

    def _scene_with_low_blocker(self):
        # room 8×3×4; a low wide block (top y=1.25) blocking the line at
        # r∈[0.4, 0.7] from the target at origin
        return Scene.from_dict({
            "scene_id": "dolly_clamp_test",
            "rooms": [{"id": "room", "label": "room",
                       "bounds_min": [-4.0, 0.0, -4.0],
                       "bounds_max": [4.0, 3.0, 4.0]}],
            "structures": [],
            "objects": [
                {"id": "console", "label": "console", "room": "room",
                 "obb": {"center": [1.0, 0.625, 0.0],
                         "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "extents": [0.6, 1.25, 2.0]},
                 "placement": "freestanding"},
            ],
            "doors": [],
        })

    def test_closeup_floor_shifted_out_of_blocker(self):
        from cinema_traj.stages.build import _clamp_dolly_run_to_clearance
        from cinema_traj.core.trajectories import DollySegment, Param

        scene = self._scene_with_low_blocker()
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        # camera rides the subject's chest height (y=0.63 inside the block's
        # y-span); dolly along +x: blocker covers r∈[0.55, 1.45]
        seg = DollySegment("move_in_target", 2.0, 0.0, "room", "subject_head",
                           "subject_01",
                           fix={"center": [0.0, 0.63, 0.0], "dir": [1.0, 0.0, 0.0]},
                           free={"r_s": Param(1.5, 0.6, 2.25),
                                 "r_e": Param(0.55, 0.55, 4.0)})
        _clamp_dolly_run_to_clearance(seg, sdf)
        # every radius in [r_s, r_e] must now be clear at the ride height
        for r in np.linspace(seg.free["r_s"].lo, seg.free["r_e"].hi, 25):
            p = np.array([r, 0.63, 0.0])
            assert float(sdf.numpy_query(p)) >= 0.15, f"r={r} still blocked"
        # the close-up end must have been pushed past the blocker
        assert seg.free["r_e"].lo >= 1.45

    def test_no_feasible_radius_raises(self):
        from cinema_traj.stages.build import _clamp_dolly_run_to_clearance
        from cinema_traj.core.trajectories import DollySegment, Param
        # wall directly in front of the target: every r on the line is inside
        scene = Scene.from_dict({
            "scene_id": "dolly_wall",
            "rooms": [{"id": "room", "label": "room",
                       "bounds_min": [-4.0, 0.0, -4.0],
                       "bounds_max": [4.0, 3.0, 4.0]}],
            "structures": [
                {"id": "wall", "kind": "wall", "room": "room",
                 "obb": {"center": [2.0, 1.5, 0.0],
                         "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "extents": [0.2, 3.0, 8.0]}},
            ],
            "objects": [], "doors": [],
        })
        sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
        seg = DollySegment("move_in_target", 2.0, 0.0, "room", "subject_head",
                           "subject_01",
                           fix={"center": [0.0, 0.0, 0.0], "dir": [1.0, 0.0, 0.0]},
                           free={"r_s": Param(1.0, 0.4, 1.5),
                                 "r_e": Param(0.55, 0.55, 4.0)})
        with pytest.raises(ValueError, match="推拉线"):
            _clamp_dolly_run_to_clearance(seg, sdf)
