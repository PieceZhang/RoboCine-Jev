"""Motion allocation, posture reserve, and mesh anticipation regressions."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from jvs_mock.arm_comfort import COMFORT_POSE, arm_metrics, posture_velocity, preferred_posture
from jvs_mock.kc_svsdf import SweptCollision
from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
from jvs_mock.shot_predictive_qp import ShotPredictiveQP, wrap
from jvs_mock.jvs_kinematics import JOINT_UPPER
from jvs_mock.whole_body_qp import whole_body_kinematics
from jvs_mock.wbc_trajectory import CameraTrajectory


ROOT = Path(__file__).resolve().parents[1]


def pan_trajectory(moving):
    return SimpleNamespace(
        duration=10., shots=[dict(type='pan', start_s=0., end_s=10.)],
        sample=lambda t: (np.array([0., .1*t if moving else 0., 1.]), np.eye(3), np.zeros(6)),
        shot_weights=lambda t, blend: dict(neutral=0., orbit=0., pan=1., moveout=0.),
        shot_context=lambda t: dict(shot_type='pan', segment_index=0, start_s=0., end_s=10.),
        tangent=lambda t: np.pi/2 if moving else None)


def obstacle_controller():
    scene = dict(format='jvs_obstacles/v1', minimum_clearance_m=.025,
                 obstacles=[dict(type='sphere', name='far', center=[3., 3., .2], radius=.1)])
    wbc = ObstaclePredictiveQP(scene=scene)
    wbc.set_trajectory(pan_trajectory(True))
    wbc.route = SimpleNamespace(sample=lambda t: np.array([0., .1*max(t, 0.)]))
    return wbc


def test_translating_pan_allocates_base_motion_but_static_pan_keeps_semantics():
    wbc = ShotPredictiveQP()
    wbc.set_trajectory(pan_trajectory(True))
    assert wbc.motion_weights(5.)['pan'] == 0.
    assert wbc.motion_weights(5.)['moveout'] == 1.
    assert wbc.context(5.)['shot_type'] == 'pan'
    wbc.set_trajectory(pan_trajectory(False))
    assert wbc.motion_weights(5.)['pan'] == 1.


def test_reverse_following_avoids_turnaround_at_motion_reversal():
    wbc = obstacle_controller()
    state = np.r_[0., .4, -np.pi/2, np.zeros(6)]
    speed, heading, direction = wbc.base_reference(state, 4.)
    assert direction == -1
    assert speed < 0.
    assert abs(wrap(heading-state[2])) < 1e-8


def test_half_turn_unwinds_arm_instead_of_taking_shorter_blocked_turn():
    wbc = obstacle_controller()
    state = np.r_[0., .4, -1.34, COMFORT_POSE]
    state[[3, 8]] = 1.0
    assert wrap(1.82-state[2]) < 0.
    assert wbc.heading_error(1.82, state) > 0.


def test_along_path_overshoot_changes_speed_without_half_turn():
    wbc = obstacle_controller()
    state = np.r_[0., .4, np.pi/2, COMFORT_POSE]
    _, heading, _ = wbc.base_reference(state, 4.)
    state[1] = .6
    speed, overshoot_heading, _ = wbc.base_reference(state, 4.)
    assert speed < 0.
    assert abs(wrap(overshoot_heading-heading)) < 1e-8


def test_preferred_posture_preserves_camera_height_rotation_and_wrist_reserve():
    from scipy.spatial.transform import Rotation
    from jvs_mock.jvs_kinematics import forward_kinematics, JOINT_LOWER
    p, r, _ = whole_body_kinematics(np.zeros(9))
    q, offset = preferred_posture(0., r, p[2])
    assert np.linalg.norm(offset-p) < .002
    assert np.linalg.norm(Rotation.from_matrix(r@forward_kinematics(q)[1].T).as_rotvec()) < .005
    assert np.all(q >= JOINT_LOWER+.15)
    assert np.all(q <= JOINT_UPPER-.15)
    assert arm_metrics(q)['arm_sigma_min'] > .005


def test_stopped_reference_does_not_reuse_old_tangent():
    wbc = obstacle_controller()
    wbc.route = SimpleNamespace(sample=lambda t: np.zeros(2))
    state = np.r_[0., 0., .4, np.zeros(6)]
    speed, heading, _ = wbc.base_reference(state, 7.)
    assert speed == 0.
    assert heading == state[2]


def test_lagging_base_follows_corner_waypoint_instead_of_cutting_obstacle():
    from jvs_mock.collision_route import CollisionRoute
    wbc = obstacle_controller()
    wbc.collision = SweptCollision(dict(format='jvs_obstacles/v1', minimum_clearance_m=.025,
        obstacles=[dict(type='sphere', name='corner', center=[.6, .6, .2], radius=.2)]))
    trajectory = pan_trajectory(True)
    trajectory.sample = lambda t: (np.array([.12+.12*t, .12*t, 1.]), np.eye(3), np.zeros(6))
    wbc.set_trajectory(trajectory)
    state = np.r_[0., 0., 0., COMFORT_POSE]
    wbc.route = CollisionRoute(wbc.collision, trajectory, state)
    position = wbc.comfort_base_position(state, 9.)
    assert not wbc.route.safe_segment(state[:2], position)
    waypoint = wbc.route.search(state[:2], position)[1]
    _, heading, direction = wbc.base_reference(state, 9.)
    expected = np.arctan2(waypoint[1], waypoint[0])+(np.pi if direction < 0 else 0.)
    assert abs(wrap(heading-expected)) < 1e-8
    assert wbc.route.safe_segment(state[:2], waypoint)


def test_arm_reach_allowance_is_not_increased_by_base_lag():
    wbc = obstacle_controller()
    p = np.array([.12, .4, 1.])
    on_path = np.r_[0., .4, 0., COMFORT_POSE]
    lagging = on_path.copy()
    lagging[1] -= 1.
    assert wbc.comfort_reach_limit(on_path, p, 4.) == .25
    assert wbc.comfort_reach_limit(lagging, p, 4.) == .25
    # A projected base reference far from the camera can need extra reach.
    assert wbc.comfort_reach_limit(on_path, p+[.5, 0., 0.], 4.) > .25


def test_terminal_base_goal_does_not_orbit_with_measured_yaw():
    wbc = obstacle_controller()
    wbc.route.safe_segment = lambda a, b: True
    wbc.trajectory.shot_context = lambda t: dict(shot_type='pan', next_shot='neutral',
        segment_index=0, start_s=0., end_s=10.)
    state = np.r_[0., .4, 0., COMFORT_POSE]
    reference = wbc.comfort_base_position(state, 9.9)
    state[2] = 1.2
    assert np.allclose(reference, wbc.comfort_base_position(state, 9.9))


def test_terminal_approach_unwinds_yaw_and_aligns_before_translation():
    wbc = obstacle_controller()
    wbc.route.safe_segment = lambda a, b: True
    wbc.trajectory.shot_context = lambda t: dict(shot_type='pan', next_shot='neutral',
        segment_index=0, start_s=0., end_s=10.)
    state = np.r_[-.4, 1.2, -2., COMFORT_POSE]
    state[3] = 1.4
    speed, heading, direction = wbc.base_reference(state, 9.9)
    assert direction == 1
    assert wrap(heading-state[2]) > 0
    assert abs(speed) < .02


def test_straight_arm_has_finite_bounded_escape_preference():
    velocity, metrics = posture_velocity(np.zeros(6), np.ones(6))
    assert np.all(np.isfinite(velocity))
    assert np.max(np.abs(velocity)) <= .6
    assert np.linalg.norm(velocity) > .01
    assert metrics['arm_sigma_min'] < 1e-10
    assert arm_metrics(COMFORT_POSE)['arm_sigma_min'] > metrics['arm_sigma_min']


def test_camera_priority_reduces_conflicting_guide_error_with_same_hard_limits():
    state = np.r_[0., 0., 0., COMFORT_POSE]
    p, r, jacobian = whole_body_kinematics(state)
    trajectory = pan_trajectory(True)
    trajectory.sample = lambda t: (p+[.05*t, 0., 0.], r, np.array([.05, 0., 0., 0., 0., 0.]))
    trajectory.shot_context = lambda t: dict(shot_type='pan', shot_phase=t/10.,
        segment_index=0, start_s=0., end_s=10., next_shot='neutral')
    errors = []
    for weight in (1., 10.):
        controller = obstacle_controller()
        controller.tracking_weight = weight
        controller.set_trajectory(trajectory)
        controller.last_plan = 0.
        controller.route = SimpleNamespace(sample=lambda t: np.array([.1+.05*t, 0.]),
                                           projected_points=0, max_projection_m=0.)
        command, info = controller.solve(state, p, r, np.zeros(6), .1, trajectory_time=0.)
        errors.append(np.linalg.norm(jacobian@command))
        assert info['qp_violation'] <= 1e-6
        assert info['command_sweep_lower_m'] >= controller.collision.margin
        assert np.all(np.abs(command) <= controller.limits.acceleration*.1+1e-6)
    assert errors[1] < .5*errors[0]


def test_execution_priority_preserves_preview_allocation_tradeoff():
    state = np.r_[0., 0., 0., COMFORT_POSE]
    p, r, _ = whole_body_kinematics(state)
    trajectory = CameraTrajectory(ROOT/'trajectories/shot_combo1.json', p, r, np.eye(3))
    plans = []
    for weight in (1., 10.):
        controller = ShotPredictiveQP(horizon_steps=4, tracking_weight=weight)
        controller.set_trajectory(trajectory)
        controller.make_plan(state, 4., 0., state[3:])
        assert controller.plan_info['mpc_violation'] <= 1e-6
        plans.append(controller.plan)
    np.testing.assert_allclose(plans[0], plans[1], atol=1e-9)


def test_execution_mesh_query_sees_witness_before_cbf_activation(tmp_path):
    path = tmp_path/'mesh.npz'
    centers = np.array([[.4, 0., .13], [1.1, 0., .13]])
    np.savez(path, vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
             faces=np.array([[0,1,2]]), surfel_centers=centers,
             surfel_radius=.069, minimum_clearance=.025,
             metadata=np.asarray(json.dumps(dict(format='jvs_groundtruth_mesh/v1'))))
    collision = SweptCollision(path)
    nearby, _ = collision.active_obstacles(np.zeros(9))
    execution, _ = collision.active_obstacles(np.zeros(9), per_body=None)
    assert len(nearby) == 1
    assert len(execution) == 2


def test_nominal_trust_box_can_relax_without_relaxing_actuator_bounds():
    wbc = obstacle_controller()
    state = np.r_[0., 0., 0., COMFORT_POSE]
    state[4] = JOINT_UPPER[1]-.1
    p, r, _ = whole_body_kinematics(state)
    wbc.set_trajectory(CameraTrajectory(ROOT/'test/fixtures/combo_unreachable.json', p, r, np.eye(3)))
    wbc.route = None
    wbc.home = state[3:].copy()
    wbc.previous[3] = .4
    wbc.make_plan(state, 0., 0., state[3:])
    assert wbc.plan is not None
    assert wbc.plan_info['mpc_trust_relaxed'] == 1
    assert wbc.plan_info['mpc_violation'] <= 1e-6
