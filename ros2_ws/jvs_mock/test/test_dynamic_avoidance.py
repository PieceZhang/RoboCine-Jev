"""Relative-motion, map lifecycle and physical-time scenario contracts."""
import json
from pathlib import Path

import numpy as np
import pytest

from jvs_mock.dynamic_collision import DynamicCollision
from jvs_mock.dynamic_lidar_map import DynamicLidarMap
from jvs_mock.dynamic_scene import DynamicScene, combo_trajectories
from jvs_mock.kc_svsdf import advance
from jvs_mock.dynamic_predictive_qp import DynamicPredictiveQP
from jvs_mock.whole_body_qp import Limits


def scene(centers, velocities):
    return dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array(centers, float),
        surfel_radius=.03, surfel_velocities=np.array(velocities, float),
        surfel_track_ids=np.ones(len(centers), int), observation_stamp=4., minimum_clearance_m=.025)


def test_transverse_cart_qp_does_not_abort_on_false_infeasibility(monkeypatch):
    from scipy.optimize import linprog
    from jvs_mock.wbc_trajectory import CameraTrajectory
    from jvs_mock.whole_body_qp import whole_body_kinematics

    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'test/fixtures/dynamic_qp_false_infeasibility.json').read_text())
    H, g = np.array(data['hessian']), np.array(data['gradient'])
    A, lo, hi = np.array(data['matrix']), np.array(data['lower']), np.array(data['upper'])
    low, high = np.isfinite(lo), np.isfinite(hi)
    certificate = linprog(np.zeros(8), A_ub=np.vstack([-A[low], A[high]]),
                         b_ub=np.r_[-lo[low], hi[high]], bounds=[(None, None)]*8,
                         method='highs')
    assert certificate.success  # Independent evidence the recorded QP is feasible.
    state = np.array(data['state'])
    p, r, _ = whole_body_kinematics(state)
    controller = DynamicPredictiveQP(Limits(), scene=scene([[10., 10., .15]], [[0., 0., 0.]]), preview=False)
    controller.set_trajectory(CameraTrajectory(root/'trajectories/shot_combo3.json', p, r, np.eye(3)))
    controller.previous = np.array(data['previous'])
    monkeypatch.setattr(controller, 'execution_collision', lambda *args: (H, g, A, lo, hi))
    controller.fast_infeasibility_eps = 1e-4
    with pytest.raises(RuntimeError, match='primal infeasible'):
        controller.solve(state, p, r, np.zeros(6), data['dt'],
                         commanded_q=np.array(data['commanded_q']))
    controller.fast_infeasibility_eps = 1e-8
    command, info = controller.solve(state, p, r, np.zeros(6), data['dt'],
                                     commanded_q=np.array(data['commanded_q']))
    assert info['fast_retried'] == 1
    assert max(0., np.max(lo-A@command), np.max(A@command-hi)) <= 1e-6


def test_relative_sweep_detects_mid_interval_collision_with_stationary_robot():
    collision = DynamicCollision(scene([[.0, -.65, .13]], [[0., 1.3, 0.]]), uncertainty_rate=0.)
    state, command = np.zeros(9), np.zeros(8)
    assert collision.clearance(state).min() > .1
    assert collision.at_time(1.).clearance(state).min() > .1
    assert collision.sweep(state, command, 1.)['upper'].min() < 0


def test_moving_sweep_bound_encloses_dense_independent_samples():
    collision = DynamicCollision(scene([[.5, -.4, .4], [-.5, .5, .2]], [[-.2, .1, 0.], [.1, -.2, 0.]]))
    state = np.array([0., 0., .2, .5, -.2, .6, .1, -.5, .1])
    command = np.array([.1, .2, .1, -.1, .1, .1, -.1, .1])
    result = collision.sweep(state, command, .6, max_samples=257)
    dense = np.min([collision.at_time(t).clearance(advance(state, command, t))
                    for t in np.linspace(0, .6, 301)], axis=0)
    assert np.all(result['lower'] <= dense + 1e-8)
    assert np.all(result['upper'] >= dense - 1e-4)


def test_explicit_obstacle_distance_derivative_matches_independent_time_difference():
    collision = DynamicCollision(scene([[.5, .3, .3]], [[-.12, .07, 0.]]), physical_time=4.5)
    state = np.array([0., 0., .2, .5, -.2, .6, .1, -.5, .1])
    actual = collision.primitive_motion(state, collision.centers)
    epsilon = 1e-5
    forward, backward = collision.at_time(epsilon), collision.at_time(-epsilon)
    fd = ((forward.geometry.primitive_distances(state, forward.centers)[0] - forward.radii[:, None]) -
          (backward.geometry.primitive_distances(state, backward.centers)[0] - backward.radii[:, None])) / (2 * epsilon)
    np.testing.assert_allclose(actual, fd, atol=1e-7)


def test_dynamic_cbf_requires_more_separation_for_approaching_object():
    state = np.zeros(9)
    bounds = []
    for speed in (0., -.15):
        controller = DynamicPredictiveQP(Limits(), scene=scene([[.7, 0, .15]], [[speed, 0, 0]]))
        controller.preparing = True
        _, _, A, lo, _ = controller.execution_collision(state, 0., np.eye(3), .02,
            np.eye(8), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
        bounds.append(np.max(lo[A[:, 0] < -.9]))
    assert bounds[1] > bounds[0] + .1


def test_preview_obstacles_advance_in_physical_time_when_camera_path_is_paused(monkeypatch):
    from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
    controller = DynamicPredictiveQP(Limits(), scene=scene([[.7, .4, .2]], [[.1, 0., 0.]]))
    controller.path_rate = 0.
    seen = []
    monkeypatch.setattr(ObstaclePredictiveQP, 'preview_collision',
                        lambda self, *args: seen.append(self.collision.physical_time))
    for step in range(3):
        sensitivity = np.zeros((9, 64))
        sensitivity[3:, :step * 8] = .4
        controller.preview_collision(np.zeros(9), np.zeros(8), .4, 0., np.zeros(9), sensitivity,
                                     np.zeros(9), sensitivity, None, None, 0.)
    np.testing.assert_allclose(seen, [4., 4.4, 4.8])
    assert controller.collision.physical_time == 4.


def test_stopped_mover_remains_geometry_without_deadlocking_camera_clock():
    controller = DynamicPredictiveQP(Limits(), scene=scene([[.27, 0., .15]], [[0., 0., 0.]]))
    controller.preparing = True
    controller.execution_collision(np.zeros(9), 0., np.eye(3), .02,
        np.eye(8), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
    assert controller.collision.track_ids[0] == 1
    assert controller.safety_info['dynamic_constraint_count'] > 0
    assert np.isinf(controller.safety_info['dynamic_predicted_clearance_m'])
    assert controller.path_rate == 1.


def test_safe_lateral_crossing_has_feasible_acceleration_bounded_barrier():
    from scipy.optimize import linprog
    state = np.array([-.02044, -.19963, -1.36207, 1.1234, -.1374, .5464, -.1318, -.3007, .4428])
    observed = scene([[.29934, -.08287, .06]], [[-.0066, -.2287, 0.]])
    observed['surfel_radius'] = np.sqrt(3) * .04 / 2
    controller = DynamicPredictiveQP(Limits(), scene=observed)
    controller.preparing = True
    controller.previous = np.array([.06050, .000378, .49981, -.05129, -.00825, .13983, .08839, -.62841])
    A, lo, hi = controller.bounds(state[3:], .017, state[3:])
    _, _, A, lo, hi = controller.execution_collision(state, 0., np.eye(3), .017,
        np.eye(8), np.zeros(8), A, lo, hi)
    finite = np.isfinite(hi)
    problem = linprog(np.zeros(8), A_ub=np.vstack([-A, A[finite]]),
        b_ub=np.r_[-lo, hi[finite]], bounds=[(None, None)] * 8, method='highs')
    assert problem.success
    assert controller.collision.sweep(state, np.zeros(8), .6)['lower'].min() > .025


@pytest.mark.parametrize('fixture', ['dynamic_post_passage', 'dynamic_post_closer_passage'])
def test_near_post_passage_preserves_acceleration_bounds_and_certified_clearance(fixture):
    from scipy.optimize import linprog
    data = np.load(Path(__file__).parent/'fixtures'/f'{fixture}.npz')
    observed = scene(data['points'], data['velocities'])
    observed['surfel_radius'] = float(data['radius'])
    observed['observation_stamp'] = float(data['observation_stamp'])
    controller = DynamicPredictiveQP(Limits(), scene=observed)
    controller.set_time(float(data['physical_time']))
    state, controller.previous, dt = data['state'], data['previous'], float(data['dt'])
    z, rotation = float(data['base_z']), data['base_rotation']
    A, lo, hi = controller.bounds(state[3:], dt, data['commanded_q'])
    _, _, A, lo, hi = controller.execution_collision(state, z, rotation, dt,
        np.eye(8), np.zeros(8), A, lo, hi)
    finite = np.isfinite(hi)
    result = linprog(np.zeros(8), A_ub=np.vstack([-A, A[finite]]), b_ub=np.r_[-lo, hi[finite]],
                     bounds=[(None, None)]*8, method='highs')
    assert result.success
    assert controller.collision.sweep(state, np.zeros(8), dt, z, base_rotation=rotation)['lower'].min() > .025


def obstacle_points(center):
    angles = np.arange(16) * 2 * np.pi / 16
    return np.array([[center[0] + .1 * np.cos(a), center[1] + .1 * np.sin(a), z]
                     for z in np.arange(.10, .61, .06) for a in angles])


def scan(points, stamp):
    return dict(points=points, rotation=np.eye(3), translation=np.zeros(3), state=np.zeros(9),
                base_z=0., base_rotation=np.eye(3), stamp=stamp, sensor='rear')


def test_lidar_estimates_motion_without_truth_and_clears_departed_trail():
    mapping = DynamicLidarMap(voxel_size=.04, retention_s=.5)
    wall = np.array([[2., y, z] for y in np.arange(-1, 1, .1) for z in np.arange(.1, 1, .1)])
    for t in np.arange(.1, 2., .1):
        mapping.integrate_scans([scan(np.vstack([wall, obstacle_points([.7, -.4 + .2 * t])]), float(t))])
    snapshot = mapping.snapshot()
    assert snapshot['tracks']
    assert snapshot['tracks'][0]['velocity'][1] == pytest.approx(.2, abs=.06)
    static = snapshot['surfel_centers'][snapshot['surfel_track_ids'] == 0]
    assert np.all(static[:, 0] > 1.8)
    mapping.integrate_scans([scan(wall, 5.)])
    assert mapping.snapshot()['tracks'] == []
    assert mapping.expired_tracks >= 1


def test_moving_scans_align_to_one_physical_time_while_furniture_stays_fixed():
    mapping = DynamicLidarMap(voxel_size=.04)
    mapping.tracks[1] = dict(moving=True, center=np.array([1., .2, .35]),
                            velocity=np.array([0., .25, 0.]), stamp=2.,
                            model_extent=np.array([.12, .12, .3]))
    furniture = np.array([[2., 0., .3], [2., .1, .4]])
    chunks = [(t, np.vstack([obstacle_points([1., .2+.25*(t-2.)]), furniture])) for t in (1.8, 1.9, 2.)]
    aligned = mapping.align_measurements(chunks, 2.)
    expected = np.vstack([obstacle_points([1., .2]), furniture])
    for measured in np.split(aligned, 3):
        np.testing.assert_allclose(measured, expected, atol=1e-12)


def test_compact_worker_rpc_preserves_motion_tracking_across_pickled_batches():
    import pickle
    import jvs_mock.dynamic_lidar_map as module
    previous = module._worker
    module._worker = None
    try:
        for t in np.arange(.1, 2., .1):
            result = module.integrate_dynamic_batch(dict(voxel_size=.04), [scan(obstacle_points([1., .2*t]), float(t))])
            received = pickle.loads(pickle.dumps(result['map']))
        tracks = received.snapshot()['tracks']
        assert len(tracks) == 1
        np.testing.assert_allclose(tracks[0]['velocity'], [0., .2, 0.], atol=.06)
        np.testing.assert_allclose(tracks[0]['center'][:2], [1., .38], atol=.04)
    finally:
        module._worker = previous


def test_approaching_mover_requests_retreat_before_hard_barrier_requires_it():
    controller = DynamicPredictiveQP(Limits(), scene=scene([[.55, 0., .15]], [[-.25, 0., 0.]]))
    state = np.zeros(9)
    H, g, A, lo, hi = controller.execution_collision(state, 0., np.eye(3), .02,
        np.eye(8), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
    assert np.all(A@np.zeros(8) >= lo)
    assert np.linalg.solve(H, -g)[0] < 0
    assert np.linalg.eigvalsh(H).min() > 0


@pytest.mark.parametrize('speed', [0., -.10])
def test_mover_classification_does_not_release_acceleration_feasible_early_braking(speed):
    from scipy.optimize import linprog
    controller = DynamicPredictiveQP(Limits(), scene=scene([[.45, 0., .125]], [[speed, 0., 0.]]))
    controller.preparing = False
    controller.previous[0] = .18
    state, dt = np.zeros(9), .02
    A, lo, hi = controller.bounds(state[3:], dt, state[3:])
    _, _, A, lo, hi = controller.execution_collision(state, 0., np.eye(3), dt,
        np.eye(8), np.zeros(8), A, lo, hi)
    finite = np.isfinite(hi)
    result = linprog(-np.eye(8)[0], A_ub=np.vstack([-A, A[finite]]),
        b_ub=np.r_[-lo, hi[finite]], bounds=[(None, None)]*8, method='highs')
    assert result.success
    assert controller.safety_info['dynamic_braking_rows'] > 0
    assert result.x[0] <= controller.previous[0]-controller.limits.acceleration[0]*dt+2e-5
    assert np.max(lo-A@result.x) <= 1e-8


def test_dynamic_telemetry_schema_stays_fixed_when_anticipation_becomes_active():
    controller = DynamicPredictiveQP(Limits(), scene=scene([[.55, 0., .15]], [[0., 0., 0.]]))
    def evaluate():
        controller.execution_collision(np.zeros(9), 0., np.eye(3), .02,
            np.eye(8), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
        return set(controller.safety_info)
    first = evaluate()
    controller.update_collision_map(scene([[.55, 0., .15]], [[-.25, 0., 0.]]))
    assert evaluate() == first
    assert controller.safety_info['dynamic_anticipation_rows'] > 0


def test_approaching_mover_records_risk_without_slowing_camera_progress():
    controller = DynamicPredictiveQP(Limits(), scene=scene([[.8, 0., .15]], [[-.25, 0., 0.]]))
    controller.preparing = False
    _, _, A, lo, _ = controller.execution_collision(np.zeros(9), 0., np.eye(3), .02,
        np.eye(8), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
    assert controller.safety_info['dynamic_predicted_clearance_m'] < .08
    assert controller.path_rate == 1.
    # Early risk remains observable without imposing a camera-clock ceiling.
    assert np.all(A @ np.zeros(8) >= lo)
    controller.update_collision_map(scene([[.8, 0., .15]], [[.25, 0., 0.]]))
    controller.execution_collision(np.zeros(9), 0., np.eye(3), .02,
        np.eye(8), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
    assert controller.safety_info['dynamic_predicted_clearance_m'] > .33
    assert controller.path_rate == 1.


def test_moving_surface_does_not_reduce_camera_task_authority():
    initial = 10000.*np.eye(8)
    controller = DynamicPredictiveQP(Limits(), scene=scene([[.27, 0., .15]], [[-.2, 0., 0.]]))
    controller.preparing = False
    H, _, _, _, _ = controller.execution_collision(np.zeros(9), 0., np.eye(3), .02,
        initial.copy(), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
    assert np.linalg.eigvalsh(H-initial).min() >= -1e-6
    static = scene([[.27, 0., .15]], [[0., 0., 0.]])
    static['surfel_track_ids'][:] = 0
    controller.update_collision_map(static)
    H, _, _, _, _ = controller.execution_collision(np.zeros(9), 0., np.eye(3), .02,
        initial.copy(), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
    assert np.linalg.eigvalsh(H-initial).min() < -100.


def test_escape_heading_recovers_shoulder_reserve_for_a_lateral_approach():
    from jvs_mock.jvs_kinematics import JOINT_UPPER
    controller = DynamicPredictiveQP(Limits(), scene=scene([[.55, 0., .15]], [[-.15, 0., 0.]]))
    controller.preparing = False
    state = np.zeros(9)
    state[2], state[3] = -1.9, JOINT_UPPER[0]-.03
    controller.execution_collision(state, 0., np.eye(3), .02,
        np.eye(8), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
    # Positive base yaw permits negative shoulder yaw, recovering reserve. The
    # shorter clockwise alignment would push the shoulder into its upper stop.
    assert controller.safety_info['dynamic_escape_omega_target'] > .2


@pytest.mark.parametrize('point,yaw', [([.35, .4, .15], -1.4), ([.29, 0., .15], -1.9)])
def test_escape_turn_does_not_interrupt_direct_or_already_close_retreats(point, yaw):
    controller = DynamicPredictiveQP(Limits(), scene=scene([point], [[-.15, -.10, 0.]]))
    controller.preparing = False
    state = np.zeros(9)
    state[2] = yaw
    controller.execution_collision(state, 0., np.eye(3), .02,
        np.eye(8), np.zeros(8), np.empty((0, 8)), np.empty(0), np.empty(0))
    assert controller.safety_info['dynamic_escape_omega_target'] == 0.


def test_occluded_track_diagnostics_and_collision_snapshot_use_the_same_time():
    mapping = DynamicLidarMap(voxel_size=.04)
    for t in np.arange(.1, 3., .1):
        mapping.integrate_scans([scan(obstacle_points([1., -.4+.12*t]), float(t))])
    for t in np.arange(3., 3.8, .1):
        mapping.integrate_scans([scan(obstacle_points([2., .8]), float(t))])
    tracks = [t for t in mapping.track_diagnostics() if t['moving']]
    assert len(tracks) == 1 and tracks[0]['prediction_age_s'] > .5
    np.testing.assert_allclose(tracks[0]['center'], mapping.snapshot()['tracks'][0]['center'])
    np.testing.assert_allclose(tracks[0]['center'], np.asarray(tracks[0]['measured_center'])
        + np.asarray(tracks[0]['velocity'])*(mapping.stamp-tracks[0]['measurement_stamp']))


def test_sparse_moving_patch_does_not_create_an_independent_dynamic_model():
    mapping = DynamicLidarMap(voxel_size=.04)
    for t in np.arange(.1, 3., .1):
        points = np.array([[1.+.15*t, y, z] for y in (-.02, .02) for z in (.1, .22)])
        mapping.track(points, float(t))
    assert mapping.tracks
    assert not any(track['moving'] for track in mapping.tracks.values())


def test_robot_motion_does_not_create_obstacle_velocity_in_odom():
    mapping = DynamicLidarMap(voxel_size=.04)
    world = obstacle_points([1., .3])
    for t in np.arange(.1, 1.5, .1):
        observation = scan(world - [.1 * t, 0, 0], float(t))
        observation['translation'] = np.array([.1 * t, 0, 0])
        observation['state'][0] = .1 * t
        mapping.integrate_scans([observation])
    assert not mapping.snapshot()['tracks']


def test_delayed_robot_echo_is_filtered_with_recent_measured_robot_pose():
    mapping = DynamicLidarMap(voxel_size=.04, retention_s=1.5)
    wall = np.array([[2., y, z] for y in np.arange(-1., 1., .1)
                     for z in np.arange(.1, 1., .1)])
    mapping.integrate_scans([scan(wall, .1)])
    # Noisy returns from the old base posture, received after it moved.
    echo = np.array([[-.24, y, .14] for y in np.arange(-.06, .061, .02)])
    for t in (.3, .5, .7):
        observation = scan(np.vstack([wall, echo]), t)
        observation['state'][0] = .5
        mapping.integrate_scans([observation])
    centers = mapping.snapshot()['surfel_centers']
    assert len(centers) > 100
    assert np.all(centers[:, 0] > 1.8)


def test_low_distant_wall_packet_remains_static_without_a_false_mover():
    mapping = DynamicLidarMap(voxel_size=.04, map_radius=4.5)
    for t in np.arange(.1, 2.5, .1):
        points = np.array([[3., y + .2 * t, z] for y in np.arange(-.25, .26, .04)
                           for z in np.arange(.08, .5, .04)])
        mapping.integrate_scans([scan(points, float(t))])
    assert len(mapping.snapshot()['surfel_centers']) > 100
    assert not mapping.snapshot()['tracks']


def test_isolated_repeated_noise_does_not_become_a_static_obstacle():
    mapping = DynamicLidarMap(voxel_size=.04)
    real = obstacle_points([1., .3])
    noise = np.array([[-.5, -.5, .3], [.1, -.7, .8]])
    for t in (.1, .3, .5, .7):
        mapping.integrate_scans([scan(np.vstack([real, noise]), t)])
    centers = mapping.snapshot()['surfel_centers']
    assert len(centers) > 30
    assert np.all(centers[:, 0] > .8)


def test_self_shadow_rejects_returns_behind_an_opaque_robot_link():
    mapping = DynamicLidarMap(voxel_size=.04)
    origin = np.array([.190, 0., .274])
    real = obstacle_points([1., .6])
    ghosts = np.array([[-.18, y, z] for y in (-.02, 0., .02) for z in (.38, .4, .42)])
    assert mapping.geometry.distances(np.zeros(9), ghosts).min() > .1
    for t in (.1, .3, .5, .7):
        observation = scan(np.vstack([real, ghosts]) - origin, t)
        observation['translation'] = origin
        mapping.integrate_scans([observation])
    centers = mapping.snapshot()['surfel_centers']
    assert len(centers) > 30
    assert np.all(centers[:, 0] > .8)


def test_partial_occlusion_cannot_turn_known_static_surfaces_into_a_mover():
    mapping = DynamicLidarMap(voxel_size=.04)
    points = obstacle_points([1., .3])
    for t in np.arange(.1, 1., .1):
        mapping.integrate_scans([scan(points, float(t))])
    for t in np.arange(1., 1.7, .1):
        visible = points[points[:, 1] > .22 + .1 * (t - 1)]
        mapping.integrate_scans([scan(visible, float(t))])
    assert not mapping.snapshot()['tracks']


def test_scan_window_on_previously_seen_low_furniture_remains_static():
    mapping = DynamicLidarMap(voxel_size=.04)
    furniture = np.array([[1.5, y, z] for y in np.arange(-1., 1.01, .04)
                          for z in np.arange(.08, .59, .04)])
    for t in np.arange(.1, 2.1, .1):
        mapping.integrate_scans([scan(furniture, float(t))])
    for t in np.arange(2.1, 5., .1):
        center = -.5 + .25*(t-2.)
        packet = furniture[np.abs(furniture[:, 1]-center) < .22]
        mapping.integrate_scans([scan(packet, float(t))])
    assert not mapping.snapshot()['tracks']
    assert len(mapping.snapshot()['surfel_centers']) > 100


def test_moving_object_keeps_its_track_when_close_to_static_furniture():
    mapping = DynamicLidarMap(voxel_size=.04)
    furniture = np.array([[1.22, y, z] for y in np.arange(-.15, .66, .04)
                          for z in np.arange(.08, .79, .04)])
    identifiers = set()
    for t in np.arange(.1, 8., .1):
        center = [1., -.7+.15*t]
        mapping.integrate_scans([scan(np.vstack([furniture, obstacle_points(center)]), float(t))])
        if t > 3.:
            tracks = mapping.snapshot()['tracks']
            assert len(tracks) == 1
            np.testing.assert_allclose(tracks[0]['center'][:2], center, atol=.06)
            identifiers.add(tracks[0]['id'])
    assert len(identifiers) == 1


def test_moving_scan_window_along_a_wall_remains_static_collision_geometry():
    mapping = DynamicLidarMap(voxel_size=.04)
    for t in np.arange(.1, 2.5, .1):
        points = np.array([[2., y + .2 * t, z] for y in np.arange(-.25, .26, .04)
                           for z in np.arange(.08, 1.6, .08)])
        mapping.integrate_scans([scan(points, float(t))])
    snapshot = mapping.snapshot()
    assert len(snapshot['surfel_centers']) > 100
    assert not snapshot['tracks']


def test_dynamic_front_profile_covers_above_horizontal_at_the_existing_tilt():
    package = Path(__file__).resolve().parents[1]
    profile = json.loads((package / 'config/JVS_DynamicWide.json').read_text())['profile']
    elevations = np.array(profile['emitterStates'][0]['elevationDeg'])
    world_elevations = elevations - np.rad2deg(.898844565)
    assert world_elevations.max() > 20.
    assert np.min(abs(world_elevations)) < 1.
    assert profile['nearRangeM'] <= .16


def test_demo_paths_have_continuous_velocities_and_catalog_defaults():
    package = Path(__file__).resolve().parents[1]
    assert [path.name for path in combo_trajectories(package / 'trajectories')] == [
        f'shot_combo{i}{suffix}.json' for i in range(1, 5) for suffix in ('', '_actions')]
    for path in (package / 'scenes').glob('dynamic_*.json'):
        scenario = DynamicScene(path)
        for name, times in scenario.times.items():
            for t in times:
                assert np.linalg.norm(scenario.sample(name, float(t))[1]) < 1e-9
                np.testing.assert_allclose(scenario.sample(name, t - 1e-6)[0],
                                           scenario.sample(name, t + 1e-6)[0], atol=1e-7)
        assert scenario.frame == 'scene_y_up'
        # A furniture-safe route is anchored to the apartment, even when the
        # launch chooses another combo's initial heading and base position.
        for spawn in [dict(x=5., z=1., yaw=0., height=.002),
                      dict(x=3.5, z=2., yaw=.62, height=.002)]:
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_euler('z', spawn['yaw'] - np.pi/2)
            for obstacle in scenario.obstacles:
                point = scenario.sample(obstacle['name'], 12.)[0]
                native = rotation.apply(scenario.to_initial_base(point, spawn)) + [spawn['x'], -spawn['z'], spawn['height']]
                np.testing.assert_allclose(native, point[[0, 2, 1]] * [1, -1, 1], atol=1e-12)


def test_scenario_rejects_bad_models_before_simulation(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(dict(format='jvs_dynamic_scene/v1', obstacles=[dict(name='Bad',
        model=dict(type='box', size=[-1, 1, 1]), trajectory=[])])))
    with pytest.raises(ValueError, match='model'):
        DynamicScene(path)


@pytest.mark.parametrize('frame,vertical', [('initial_base', 2), ('scene_y_up', 1)])
def test_vertical_trajectories_are_rejected_by_the_horizontal_tracker_contract(tmp_path, frame, vertical):
    path = tmp_path / 'vertical.json'
    start, end = [1., .5, .3], [1., .5, .3]
    end[vertical] += .1
    path.write_text(json.dumps(dict(format='jvs_dynamic_scene/v1', frame=frame, obstacles=[dict(
        name='Post', model=dict(type='cylinder', radius=.07, height=.5),
        trajectory=[dict(time_s=0., position=start), dict(time_s=5., position=end)])])))
    with pytest.raises(ValueError, match='horizontal'):
        DynamicScene(path)
