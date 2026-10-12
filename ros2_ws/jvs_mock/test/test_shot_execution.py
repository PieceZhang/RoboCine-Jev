"""Cinematic completion must measure motion, not merely an expired timer."""
import json
from pathlib import Path

import numpy as np
import pytest

from jvs_mock.shot_execution import ShotClock
from jvs_mock.trajectory_scene import trajectory_spawn
from jvs_mock.wbc_trajectory import CameraTrajectory
from jvs_mock.wbc_metrics import Metrics
from jvs_mock.whole_body_qp import Limits, whole_body_kinematics
from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_clock_slowdown_is_continuous_and_does_not_skip_path():
    clock = ShotClock()
    for i in range(400):
        previous_time, previous_rate = clock.time, clock.rate
        lag = .4 if i > 100 else 0.
        t, rate = clock.advance(.02, lag, 0.)
        assert 0. <= rate <= 1.
        assert abs(rate-previous_rate) <= .00800001
        assert t-previous_time == pytest.approx(.02*rate)
    assert rate == pytest.approx(.15)


def test_transition_includes_both_shots_stationary_windows():
    p, r, _ = whole_body_kinematics(np.zeros(9))
    trajectory = CameraTrajectory(ROOT/'trajectories/shot_combo1.json', p, r, np.eye(3))
    start = trajectory.shots[0]['motion_end_s']
    end = trajectory.shots[1]['motion_start_s']
    assert trajectory.transition(start-.1) is None
    for t in np.linspace(start, end, 4, endpoint=False):
        window = trajectory.transition(t)
        assert window['end_s'] == end
        np.testing.assert_allclose(trajectory.sample(t)[2], 0., atol=1e-5)
    assert trajectory.transition(end) is None


def test_spawn_and_target_contract_for_distinct_combos():
    spawns = []
    for index in (1, 2, 3, 4):
        path = ROOT/f'trajectories/shot_combo{index}.json'
        spawn = trajectory_spawn(path, {})
        spawns.append((spawn['x'], spawn['z'], spawn['yaw']))
        p, r, _ = whole_body_kinematics(np.zeros(9))
        trajectory = CameraTrajectory(path, p, r, np.eye(3))
        for t in np.linspace(2., trajectory.duration, 50):
            position, rotation, _ = trajectory.sample(t)
            direction = trajectory.subject-position
            direction /= np.linalg.norm(direction)
            np.testing.assert_allclose(rotation[:, 0], direction, atol=2e-5)
    assert len(set(spawns)) == 4


def test_spawn_rejects_nonfinite_and_wrong_frame(tmp_path):
    path = tmp_path/'trajectory.json'
    for pose in (dict(frame='odom', x=1., z=2., yaw=0., height=.002),
                 dict(frame='scene_y_up', x=float('nan'), z=2., yaw=0., height=.002)):
        path.write_text(json.dumps(dict(initial_base_pose=pose)))
        with pytest.raises(ValueError):
            trajectory_spawn(path, {})


def test_editing_json_spawn_retains_subject_and_shot_geometry(tmp_path):
    source = ROOT/'trajectories/shot_combo2.json'
    document = json.loads(source.read_text())
    document['initial_base_pose'].update(x=3.7, z=1.8, yaw=.55)
    path = tmp_path/'changed_start.json'
    path.write_text(json.dumps(document))
    assert trajectory_spawn(path, {})['x'] == 3.7
    p, r, _ = whole_body_kinematics(np.zeros(9))
    changed = CameraTrajectory(path, p, r, np.eye(3))
    original = CameraTrajectory(source, p, r, np.eye(3))
    for shot in changed.shots:
        start, end = shot['start_s'], shot['end_s']
        def movement(traj):
            a = traj.sample(start)[0]-traj.subject
            b = traj.sample(end)[0]-traj.subject
            return (np.arctan2(a[0]*b[1]-a[1]*b[0], a[:2]@b[:2])
                    if shot['type'] != 'moveout' else np.linalg.norm(b[:2])-np.linalg.norm(a[:2]))
        assert movement(changed) == pytest.approx(movement(original), abs=1e-6)
    for t in np.linspace(2., changed.duration, 25):
        position, rotation, _ = changed.sample(t)
        direction = changed.subject-position
        np.testing.assert_allclose(rotation[:, 0], direction/np.linalg.norm(direction), atol=2e-5)


def test_finished_clock_with_stationary_camera_is_not_success(tmp_path):
    p, r, _ = whole_body_kinematics(np.zeros(9))
    trajectory = CameraTrajectory(ROOT/'trajectories/shot_combo1.json', p, r, np.eye(3))
    metrics = Metrics(tmp_path, Limits())
    metrics.trajectory = trajectory
    rows = [dict(time_s=t, wall_elapsed_s=t, camera_x=p[0], camera_y=p[1], camera_z=p[2],
        camera_forward_x=r[0,0], camera_forward_y=r[1,0], camera_forward_z=r[2,0],
        position_error=0., orientation_error=0., command_sweep_lower_m=.1)
        for t in np.linspace(0., trajectory.duration, 100)]
    summary = dict(completed=True, failure='')
    metrics.shot_report(summary, rows)
    metrics.close()
    assert not summary['shot_motions_pass']
    assert not summary['approximate_success']
    assert all(abs(s['actual']) < 1e-9 for s in summary['shots'])


def test_low_rms_cannot_hide_a_position_peak_above_five_centimetres(tmp_path):
    metrics = Metrics(tmp_path, Limits())
    metrics.position_tolerance = metrics.position_max_tolerance = .05
    try:
        for i in range(20):
            metrics.record(i*.02, .02, np.zeros(9), np.zeros(8), np.zeros(8), np.zeros(2), 0.,
                           dict(position_error=.06 if i == 19 else .005, orientation_error=0.,
                                qp_violation=0., solve_s=0.))
        summary = metrics.finish(True)
        assert summary['position_error_rms'] < .05
        assert summary['position_error_max'] == pytest.approx(.06)
        assert summary['thresholds']['position_max_m'] == .05
        assert not summary['tracking_accuracy_pass']
    finally:
        metrics.close()


def test_reachable_combo_retains_spawn_subject_and_requested_shot_motion():
    path = ROOT/'trajectories/shot_combo1.json'
    original = ROOT/'test/fixtures/combo_unreachable.json'
    assert trajectory_spawn(path, dict(x=5., z=1., yaw=0., height=.002)) == trajectory_spawn(
        original, dict(x=5., z=1., yaw=0., height=.002))
    p, r, _ = whole_body_kinematics(np.zeros(9))
    trajectory = CameraTrajectory(path, p, r, np.eye(3))
    assert [s['type'] for s in trajectory.shots] == ['orbit', 'pan', 'moveout']
    assert [s.get('angle_deg') for s in trajectory.shots[:2]] == [20., -30.]
    assert trajectory.shots[-1]['distance'] == .7
    for t in np.linspace(2., trajectory.duration, 101):
        position, rotation, _ = trajectory.sample(t)
        direction = trajectory.subject-position
        np.testing.assert_allclose(rotation[:, 0], direction/np.linalg.norm(direction), atol=2e-5)


def test_only_projected_final_moveout_uses_a_fixed_escape_goal():
    p, r, _ = whole_body_kinematics(np.zeros(9))
    trajectory = CameraTrajectory(ROOT/'test/fixtures/combo_unreachable.json', p, r, np.eye(3))
    scene = dict(format='jvs_obstacles/v1', minimum_clearance_m=.025,
                 obstacles=[dict(type='sphere', name='far', center=[5., 5., 1.], radius=.1)])
    controller = ObstaclePredictiveQP(scene=scene)
    controller.set_trajectory(trajectory)
    nominal = lambda t: trajectory.sample(t)[0][:2]-p[:2]
    controller.route = SimpleNamespace(initial=np.zeros(2), sample=lambda t: nominal(t)+[.3, 0.])
    assert not controller.projected_exit(10.)
    assert not controller.projected_exit(30.)
    assert controller.projected_exit(40.)
    assert controller.terminal_approach(40.)
    assert controller.comfort_reach_limit(np.zeros(9), trajectory.sample(40.)[0], 40.) == .25
    controller.route = SimpleNamespace(initial=np.zeros(2), sample=nominal)
    assert not controller.projected_exit(40.)


def test_moveout_endpoint_alone_cannot_hide_repeated_reverse_motion(tmp_path):
    p, r, _ = whole_body_kinematics(np.zeros(9))
    trajectory = CameraTrajectory(ROOT/'test/fixtures/combo_unreachable.json', p, r, np.eye(3))
    rows = []
    for t in np.linspace(0., 54., 1081):
        position, rotation, _ = trajectory.sample(t)
        if 38. <= t <= 52.:
            outward = position-trajectory.subject
            outward[2] = 0.
            position += .2*np.sin(10*np.pi*(t-38.)/14.)*outward/np.linalg.norm(outward)
        forward = trajectory.subject-position
        forward /= np.linalg.norm(forward)
        rows.append(dict(time_s=t, wall_elapsed_s=t, **{f'camera_{a}':v for a,v in zip('xyz',position)},
            **{f'camera_forward_{a}':v for a,v in zip('xyz',forward)}, position_error=0., orientation_error=0.,
            command_sweep_lower_m=.1))
    metrics = Metrics(tmp_path, Limits())
    metrics.trajectory = trajectory
    summary = dict(completed=True, failure='')
    metrics.shot_report(summary, rows)
    metrics.close()
    shot = summary['shots'][-1]
    assert shot['completion_ratio'] == pytest.approx(1.)
    assert shot['motion_completed']
    assert not shot['motion_direction_pass']
    assert not summary['approximate_success']


def test_planar_recovery_continues_past_sideways_nearest_cell():
    scene = dict(format='jvs_obstacles/v1', minimum_clearance_m=.025,
                 obstacles=[dict(type='sphere', name='far', center=[5., 5., 1.], radius=.1)])
    controller = ObstaclePredictiveQP(scene=scene)
    nearest = np.array([0., -.05])
    corner = np.array([-.3, -.05])
    goal = np.array([-.3, 1.])
    controller.route = SimpleNamespace(safe_segment=lambda a,b: False,
        _nearest_node=lambda p: 0, _coord=lambda node: nearest,
        search=lambda a,b: np.array([nearest, corner, goal]))
    # Driving sideways into the nearby cell is not a viable diff-drive escape;
    # the next free-space corner permits backing out before the turn.
    np.testing.assert_array_equal(controller.recovery_waypoint(np.zeros(2), goal), corner)
