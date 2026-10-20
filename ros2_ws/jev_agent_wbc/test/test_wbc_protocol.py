"""Clock, camera axes, geometric binding and capability boundaries before Agent ingress."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.representation import PARTS, Box, Observation, Pose, Request
from jev_agent_wbc.wbc_protocol import (
    AXES, CameraTrajectory, Capabilities, TimedTask, measured_feedback, negotiate,
    observation_from_measured_bounds, optical_camera, semantic_camera,
)


def fixture(*, geometry=False):
    mount = Pose((.3, -.1, .7), tuple(Rotation.from_euler('xyz', [.2, -.3, 1.2]).as_quat()))
    camera = mount.compose(Pose((.2, -.1, .4)))
    observation = Observation.from_fk(10., mount, camera,
        {PARTS[0]: (Box((0, 0, 0), (.3, .2, .1)),),
         PARTS[1]: (Box((.1, 0, .2), (.1, .2, .3)),)})
    trajectory = CameraTrajectory(10., (0., 1.), (camera, camera.compose(Pose((.1, 0, 0)))))
    request = None
    if geometry:
        request = Request(observation.id, trajectory.id, 10., 11., occupancy_preferences=(
            dict(part=PARTS[0], frame='world', relation='avoid_region',
                 region=Box((1, 0, 0), (.2, .2, .2)).to_dict(), margin_m=.05, weight=1.),)).to_dict()
    task = TimedTask('task-1', 10., observation.id, trajectory, .01, .02, request)
    return observation, task


def test_optical_conversion_matches_camera_forward_left_up_and_roundtrip():
    optical = Pose((.3, -.2, 1.), tuple(Rotation.from_euler('xyz', [.4, -.3, .9]).as_quat()))
    semantic = semantic_camera(optical)
    # Compare independent physical axis definitions, not just inverse roundtrip.
    np.testing.assert_allclose(semantic.rotation[:, 0], optical.rotation[:, 2], atol=1e-15)
    np.testing.assert_allclose(semantic.rotation[:, 1], -optical.rotation[:, 0], atol=1e-15)
    np.testing.assert_allclose(semantic.rotation[:, 2], -optical.rotation[:, 1], atol=1e-15)
    recovered = optical_camera(semantic)
    np.testing.assert_allclose(recovered.rotation, optical.rotation, atol=1e-15)
    assert recovered.translation_m == optical.translation_m


def test_absolute_clock_slerp_and_no_extrapolation():
    a = Pose((1., 2., .7), tuple(Rotation.from_euler('z', 170, degrees=True).as_quat()))
    b = Pose((3., 0., .9), tuple(Rotation.from_euler('z', -170, degrees=True).as_quat()))
    path = CameraTrajectory(50., (0., 2.), (a, b))
    sample = path.sample(51.)
    np.testing.assert_allclose(sample.translation_m, (2., 1., .8))
    assert abs(Rotation.from_matrix(sample.rotation).as_euler('xyz')[2]) == pytest.approx(np.pi)
    assert CameraTrajectory.from_dict(path.to_dict()).id == path.id
    for t in (49.999, 52.001, float('nan'), True):
        with pytest.raises(ValueError):
            path.sample(t)


def test_fractional_absolute_time_endpoint_roundoff():
    path = CameraTrajectory(.1, (0., .2), (Pose((0., 0., 0.)), Pose((.2, 0., 0.))))
    assert path.sample(path.end_s).translation_m[0] == pytest.approx(.2)


def test_feedforward_matches_independent_world_pose_derivative():
    a = Pose((.2, -.3, .7), tuple(Rotation.from_euler('xyz', [.4, -.3, 1.2]).as_quat()))
    b = Pose((.5, -.1, .8), tuple(Rotation.from_euler('xyz', [-.2, .1, 1.7]).as_quat()))
    path = CameraTrajectory(10., (0., 2.), (a, b))
    _, twist = path.sample_with_twist(11.)
    before, after = path.sample(11.-1e-5), path.sample(11.+1e-5)
    linear = (np.asarray(after.translation_m)-before.translation_m)/2e-5
    angular = Rotation.from_matrix(after.rotation@before.rotation.T).as_rotvec()/2e-5
    np.testing.assert_allclose(twist, np.r_[linear, angular], atol=5e-10)


@pytest.mark.parametrize('change', [
    dict(frame='map'), dict(axes='optical'), dict(interpolation='cubic'), dict(joint_positions=[]),
    dict(start_s=True), dict(start_s=-1.),
])
def test_implicit_frames_and_internal_state_rejected(change):
    _, task = fixture()
    data = task.trajectory.to_dict()
    data.update(change)
    with pytest.raises(ValueError):
        CameraTrajectory.from_dict(data)


@pytest.mark.parametrize('times', [(0., 0.), (1., 2.), (0., float('inf')), (0., True), (0.,)])
def test_malformed_reference_times_rejected(times):
    with pytest.raises(ValueError):
        CameraTrajectory(0., times, tuple(Pose((0, 0, 0)) for _ in times))


def test_optical_export_has_explicit_native_rig_axes():
    _, task = fixture()
    samples = task.trajectory.optical_samples()
    for raw, pose in zip(samples, task.trajectory.poses):
        reconstructed = semantic_camera(Pose(tuple(raw['position_m']), tuple(raw['quaternion_xyzw'])))
        np.testing.assert_allclose(reconstructed.rotation, pose.rotation, atol=1e-15)
    assert samples[0]['t_s'] == 0. and samples[-1]['t_s'] == 1.


def test_task_geometry_binding_and_copy():
    observation, task = fixture(geometry=True)
    data = task.to_dict()
    cloned = TimedTask.from_dict(data)
    data['geometry_request']['occupancy_preferences'][0]['weight'] = 5.
    assert cloned.geometry_request['occupancy_preferences'][0]['weight'] == 1.
    assert cloned.required_features == {'timed_camera_se3', 'platform_avoid_region'}
    for field in ('observation_id', 'camera_path_id'):
        bad = task.to_dict()
        bad['geometry_request'][field] = 'different'
        with pytest.raises(ValueError, match='binding'):
            TimedTask.from_dict(bad)
    assert observation.id == task.observation_id


def test_capability_negotiation_does_not_silently_drop_avoidance_preference():
    observation, task = fixture(geometry=True)
    capability = Capabilities('actual-native-backend', True, frozenset({'timed_camera_se3'}))
    result = negotiate(task, observation, capability, 10.1)
    assert not result['supported']
    assert result['unsupported_features'] == ['platform_avoid_region']
    assert result['feasible'] is None and not result['camera_reference_changed']
    complete = Capabilities('actual-native-backend', True, task.required_features)
    accepted = negotiate(task, observation, complete, 10.1)
    assert accepted['supported'] and accepted['feasible'] is None


def test_runtime_freshness_source_and_expiry_cannot_be_hidden_by_capabilities():
    observation, task = fixture()
    capability = Capabilities('planned-backend', False, frozenset({'timed_camera_se3'}))
    assert 'backend_unavailable' in negotiate(task, observation, capability, 10.1)['reasons']
    capability = Capabilities('runtime', True, frozenset({'timed_camera_se3'}))
    assert 'observation_stale_or_future' in negotiate(task, observation, capability, 10.26)['reasons']
    assert 'observation_stale_or_future' in negotiate(task, observation, capability, 9.9)['reasons']
    assert 'trajectory_expired' in negotiate(task, observation, capability, 11.1)['reasons']
    bad = observation.to_dict()
    bad['source'] = 'predicted_fk'
    assert 'current_physical_measurement_required' in negotiate(task, Observation.from_dict(bad), capability, 10.)['reasons']


def test_feedback_keeps_measured_geometry_and_feasibility_independent():
    observation, task = fixture()
    feedback = measured_feedback(task, observation, feasible=False, reason='self_collision')
    assert feedback['camera_within_tolerance'] and feedback['feasible'] is False
    assert feedback['position_error_m'] < 1e-15
    assert feedback['geometry']['world_from_mount'] == observation.world_from_mount.to_dict()
    assert feedback['remaining_adjustment'] == dict.fromkeys(PARTS)
    assert not feedback['safety_certificate']
    bad = observation.to_dict()
    bad['source'] = 'synthetic_fk'
    with pytest.raises(ValueError, match='measured'):
        measured_feedback(task, Observation.from_dict(bad))
    with pytest.raises(ValueError):
        measured_feedback(task, observation, remaining_adjustment=dict.fromkeys(PARTS, 1.1))
    with pytest.raises(ValueError, match='public failure code'):
        measured_feedback(task, observation, reason='ur5_shoulder_pan_joint at 6.28 rad')


def test_feedback_uses_timestamp_target_and_geodesic_error():
    observation, task = fixture()
    moved = Observation.from_fk(10.5, observation.world_from_mount,
        observation.world_from_camera.compose(Pose((0, 0, 0), tuple(Rotation.from_euler('z', .1).as_quat()))),
        observation.occupancy)
    feedback = measured_feedback(task, moved)
    assert feedback['position_error_m'] == pytest.approx(.05)
    assert feedback['orientation_error_rad'] == pytest.approx(.1)
    assert not feedback['camera_within_tolerance']
    assert feedback['feasible'] is None


def test_collider_bounds_remain_conservative_under_tilt_and_compression():
    mount = Pose((2., -.4, .8), tuple(Rotation.from_euler('xyz', [.3, -.2, .8]).as_quat()))
    optical = mount.compose(Pose((.1, .2, .3)))
    bounds = {PARTS[0]: [([0., 0., 0.], [1., 1., 1.]), ([1., 0., 0.], [2., 1., 1.])],
              PARTS[1]: [([2., -.2, 1.], [2.2, .3, 1.5])]}
    observation = observation_from_measured_bounds(1., mount, optical, bounds, box_budget=1)
    assert observation.source == 'measured_fk'
    for part in PARTS:
        for low, high in bounds[part]:
            raw = Box(tuple((np.array(low)+high)/2), tuple((np.array(high)-low)/2))
            vertices = mount.inverse().apply(raw.vertices)
            assert np.max(observation.occupancy[part][0].distance(vertices)) < 1e-14
    np.testing.assert_allclose(optical_camera(observation.world_from_camera).rotation, optical.rotation, atol=1e-15)
