"""Regressions for mesh guide connectivity and asynchronous preview failures."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from jvs_mock.collision_route import CollisionRoute, CollisionRouteError
from jvs_mock.kc_svsdf import SweptCollision
from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
from jvs_mock.shot_preview_worker import AsyncObstaclePredictiveQP
from jvs_mock.wbc_trajectory import CameraTrajectory
from jvs_mock.whole_body_qp import whole_body_kinematics


ROOT = Path(__file__).resolve().parents[1]


def line_trajectory(end):
    return SimpleNamespace(duration=4., sample=lambda t: (np.r_[np.asarray(end)*t/4., 0.],))


def circles_scene(circles):
    return SimpleNamespace(planar_circles=lambda points: circles)


def assert_free(route):
    times = np.unique(np.r_[route.curve.x, np.linspace(0, route.duration, 2001)])
    points = route.sample(times)
    for a, b in zip(points, points[1:]):
        assert route.safe_segment(a, b)


def test_overlapping_circle_union_projects_to_reachable_component():
    # The endpoint is enclosed by a ring of overlapping mesh-like circles.
    # Sequential radial pushes can leave it on the inaccessible inner wall.
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    circles = [(np.array([1., 0.]) + .45*np.array([np.cos(a), np.sin(a)]), .12)
               for a in angles]
    route = CollisionRoute(circles_scene(circles), line_trajectory([1., 0.]), np.zeros(9))
    assert route.projected_points > 0
    assert route.max_projection_m > .4
    assert not route.reachable(np.array([1., 0.]))
    assert_free(route)
    np.testing.assert_array_equal(route.sample(0), [0., 0.])


def test_detour_uses_whole_scene_and_interpolation_stays_free():
    # Detour extends farther than the old per-segment +/-0.7 m search box.
    circles = [(np.array([.5, y]), .12) for y in np.arange(-1., 1.01, .1)]
    route = CollisionRoute(circles_scene(circles), line_trajectory([1., 0.]), np.zeros(9))
    assert np.max(np.abs(route.sample(np.linspace(0, 4, 1001))[:, 1])) > 1.
    assert not route.safe_segment(np.zeros(2), np.array([1., 0.]))
    assert route.reachable(np.array([1., 0.]))
    assert_free(route)


def test_stationary_intervals_have_strict_times():
    route = CollisionRoute(circles_scene([]), line_trajectory([0., 0.]), np.zeros(9))
    assert np.all(np.diff(route.curve.x) > 0.)
    np.testing.assert_array_equal(route.sample(2.), [0., 0.])


def test_invalid_start_is_explicit():
    with pytest.raises(CollisionRouteError, match='initial base'):
        CollisionRoute(circles_scene([(np.zeros(2), .2)]), line_trajectory([1., 0.]), np.zeros(9))


def controller(cls=ObstaclePredictiveQP):
    scene = dict(format='jvs_obstacles/v1', minimum_clearance_m=.025,
                 obstacles=[dict(type='sphere', name='far', center=[3., 3., .2], radius=.1)])
    result = cls(scene=scene)
    p, r, _ = whole_body_kinematics(np.zeros(9))
    result.set_trajectory(CameraTrajectory(ROOT/'trajectories/shot_combo1.json', p, r, np.eye(3)))
    return result


def test_unavailable_soft_guide_preserves_constrained_tracking_and_braking():
    wbc = controller()
    state = np.zeros(9)
    p, r, _ = whole_body_kinematics(state)
    with patch('jvs_mock.collision_route.CollisionRoute', side_effect=CollisionRouteError('guide unavailable')):
        command, info = wbc.solve(state, p + [.01, 0., 0.], r, np.zeros(6), .02)
    assert wbc.plan_failures == 1
    assert info['preview_failure'] == 'guide unavailable'
    assert np.linalg.norm(command) > 0
    assert info['qp_violation'] <= 1e-6
    assert info['command_sweep_lower_m'] >= wbc.collision.margin
    stopped, info = wbc.solve(state, p, r, np.zeros(6), .02, brake=True)
    assert np.linalg.norm(stopped) <= np.linalg.norm(command) + 1e-8
    assert info['qp_violation'] <= 1e-6


def test_approaching_mesh_witness_is_active_before_nearest_four_switch():
    wbc = controller()
    wbc.collision.is_mesh = True
    wbc.previous[0] = .5
    centers, radii = np.zeros((5, 3)), np.zeros(5)
    # The fifth surface is farther away, but its normal opposes current travel.
    # Waiting for it to become nearest can leave no acceleration-feasible input.
    distances = np.array([[.2], [.21], [.22], [.23], [.7]])
    gradients = np.zeros((5, 1, 9))
    gradients[4, 0, 0] = -1.
    with patch.object(wbc.collision, 'active_obstacles', return_value=(centers, radii)) as active:
        with patch.object(wbc.collision.geometry, 'primitive_distances', return_value=(distances, gradients)):
            _, _, A, lo, _ = wbc.execution_collision(
                np.zeros(9), 0., None, .02, np.eye(8), np.zeros(8),
                np.empty((0, 8)), np.empty(0), np.empty(0))
    assert active.call_args.kwargs['per_body'] is None
    approaching = A[:, 0] == -1.
    assert np.any(approaching)
    assert np.all(lo[approaching] > -wbc.previous[0])


def test_fast_qp_refines_candidate_before_hard_residual_gate():
    import osqp
    wbc = controller()
    state = np.zeros(9)
    p, r, _ = whole_body_kinematics(state)
    solver = osqp.OSQP()
    original_solve = solver.solve
    calls = []

    def solve():
        result = original_solve()
        calls.append(result)
        if len(calls) == 1:
            # Simulate coarse numerical convergence with an invalid velocity.
            result.x[0] = 1.
        return result

    with patch('jvs_mock.shot_predictive_qp.osqp.OSQP', return_value=solver):
        with patch.object(solver, 'solve', side_effect=solve):
            command, info = wbc.solve(state, p, r, np.zeros(6), .02, brake=True)
    assert len(calls) == 2
    assert info['fast_retried'] == 1
    assert info['qp_violation'] <= 1e-6
    assert np.max(np.abs(command)) < 1e-8


def test_mesh_wall_approach_keeps_acceleration_feasible():
    wbc = controller()
    wbc.collision.is_mesh = True
    wbc.previous[0] = .5
    # Start outside the earlier braking band, with an initially feasible
    # velocity; witnesses are now queried well before this clearance.
    distance, dt = 1.1, .02
    gradients = np.zeros((1, 1, 9))
    gradients[0, 0, 0] = -1.
    with patch.object(wbc.collision, 'active_obstacles', return_value=(np.zeros((1, 3)), np.zeros(1))):
        for _ in range(250):
            with patch.object(wbc.collision.geometry, 'primitive_distances',
                              return_value=(np.array([[distance]]), gradients)):
                _, _, _, lo, _ = wbc.execution_collision(
                    np.zeros(9), 0., None, dt, np.eye(8), np.zeros(8),
                    np.empty((0, 8)), np.empty(0), np.empty(0))
            speed = min(.5, -lo[0])
            assert speed >= wbc.previous[0] - wbc.limits.acceleration[0]*dt - 1e-12
            distance -= speed*dt
            assert distance >= wbc.collision.margin
            wbc.previous[0] = speed


def test_fast_qp_iteration_limit_refines_same_constraints():
    import osqp
    wbc = controller()
    state = np.zeros(9)
    p, r, _ = whole_body_kinematics(state)
    stalled = SimpleNamespace(x=np.zeros(8), info=SimpleNamespace(
        status_val=osqp.constant('OSQP_MAX_ITER_REACHED'),
        status='maximum iterations reached', iter=12000))
    with patch.object(osqp.OSQP, 'solve', return_value=stalled):
        command, info = wbc.solve(state, p, r, np.zeros(6), .02, brake=True)
    assert info['fast_solver'] == 'slsqp'
    assert info['qp_violation'] <= 1e-6
    assert np.max(np.abs(command)) < 1e-8


@pytest.mark.parametrize('same_generation', [True, False])
def test_stale_preview_keeps_only_current_static_guide(same_generation):
    wbc = controller(AsyncObstaclePredictiveQP)
    try:
        route = SimpleNamespace(projected_points=2, max_projection_m=.1)
        wbc.pending = Future()
        wbc.pending.set_result(dict(route=route, plan=np.ones((wbc.n, 8)), reference=np.ones(8)))
        wbc.pending_time = 0.
        wbc.pending_execution_time = 0.
        wbc.execution_clock = 1.
        wbc.pending_generation = 0
        wbc.input_generation = 0 if same_generation else 1
        with patch.object(ObstaclePredictiveQP, 'solve', return_value=(np.zeros(8), {})):
            wbc.solve(trajectory_time=.1)  # slowed path clock; physical plan is already stale
        assert (wbc.route is route) == same_generation
        assert wbc.plan is None
        assert wbc.applied_time is None
        assert wbc.stale_plans == 1
    finally:
        wbc.close()


def test_expired_preview_does_not_repeat_terminal_velocity():
    wbc = controller(AsyncObstaclePredictiveQP)
    try:
        wbc.plan = np.ones((wbc.n, 8))
        wbc.reference = np.ones(8)
        wbc.applied_time = 0.
        wbc.applied_execution_time = 0.
        wbc.execution_clock = 4.
        with patch.object(ObstaclePredictiveQP, 'solve', return_value=(np.zeros(8), {})):
            _, info = wbc.solve(trajectory_time=.2)  # physical horizon expired during a hold
        assert wbc.plan is None
        assert wbc.applied_time is None
        np.testing.assert_array_equal(wbc.reference, np.zeros(8))
        assert info['preview_age_s'] == -1.
    finally:
        wbc.close()


def test_async_recovery_holds_without_search_then_follows_visible_worker_path():
    wbc = controller(AsyncObstaclePredictiveQP)
    try:
        wbc.route = SimpleNamespace(
            safe_segment=lambda a, b: np.linalg.norm(b-a) <= 1.1,
            search=lambda *_: pytest.fail('A* must not run in execution callback'))
        position, goal = np.array([0., 0.]), np.array([2., 0.])
        np.testing.assert_array_equal(wbc.recovery_waypoint(position, goal), position)
        wbc.recovery_path = np.array([[0., 0.], [1., 0.], [2., 0.]])
        wbc.recovery_goal = goal
        np.testing.assert_array_equal(wbc.recovery_waypoint(position, goal), [1., 0.])
        # A path to the previous shot's destination must not steer this shot.
        np.testing.assert_array_equal(wbc.recovery_waypoint(position, [0., 2.]), position)
    finally:
        wbc.close()


def test_projected_moveout_does_not_escape_during_stationary_transition():
    wbc = controller()
    wbc.collision.scene['format'] = 'jvs_lidar_surfels/v1'
    wbc.route = SimpleNamespace(sample=lambda t: np.array([0., t]),
        reachable=lambda _: pytest.fail('transition must not pursue a projected far-side goal'))
    with patch.object(wbc, '_base_reference', return_value=(.2, 0., -1)):
        with patch.object(wbc, 'projected_exit', return_value=True):
            speed, heading, direction = wbc.transition_reference(
                np.zeros(9), 35., dict(reference_s=38.1, end_s=38.))
    assert speed == 0.
    assert heading == pytest.approx(1.5 * np.pi)
    assert direction == -1


def test_kujiale_combo_mesh_guide():
    mesh = ROOT.parents[1]/'cinema_traj_out/kujiale_0021_kid_combo/simview/simview.kc_mesh.npz'
    if not mesh.is_file():
        pytest.skip('local Kujiale ground-truth mesh cache unavailable')
    collision = SweptCollision(mesh)
    p, r, _ = whole_body_kinematics(np.zeros(9))
    trajectory = CameraTrajectory(ROOT/'test/fixtures/combo_unreachable.json', p, r, np.eye(3))
    route = CollisionRoute(collision, trajectory, np.zeros(9))
    assert route.projected_points > 0
    assert_free(route)
    near_obstacle = np.zeros(9)
    near_obstacle[0] = .25
    terms, _ = collision.terms(near_obstacle, np.zeros(8), .4)
    assert terms
    assert all(sum(term[4] == body for term in terms) <= 2 for body in range(8))
    all_centers, _ = collision.active_obstacles(near_obstacle, per_body=None)
    assert len(all_centers) > len(terms)
