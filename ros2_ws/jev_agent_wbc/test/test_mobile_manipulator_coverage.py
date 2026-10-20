"""Coverage fixtures must include torso travel and fail closed on partial audits."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_robot_sensor_coverage import (
    arguments, mobile_posture_targets, mobile_profile, sampled_acceptance, saved_acceptance_status,
    fixed_chain_allowed_pairs,
)


def test_torso_and_arm_use_full_physical_limits_without_wheel_sampling():
    profile = dict(joints=[dict(name='arm_joint', home=.2), dict(name='torso_joint', home=.15)])
    names = ['wheel_joint', 'torso_joint', 'arm_joint']
    limits = np.array([[-1e8, 1e8], [0., .35], [-7., 7.]])
    initial = np.array([.8, 0., 0.])
    original = limits.copy()
    home, indices, lower, upper, targets = mobile_posture_targets(profile, names, limits, initial, 64, 23)
    assert indices == [2, 1]
    np.testing.assert_array_equal(lower, [-7., 0.])
    np.testing.assert_array_equal(upper, [7., .35])
    np.testing.assert_array_equal(limits, original)
    assert len(targets) == 69
    by_name = dict(targets)
    assert by_name['torso_joint_low'][1] == 0.
    assert by_name['torso_joint_high'][1] == .35
    assert by_name['arm_joint_low'][2] == -7.
    assert by_name['arm_joint_high'][2] == 7.
    assert all(q[0] == .8 for _, q in targets)
    for _, q in targets:
        assert np.all(q[indices] >= lower) and np.all(q[indices] <= upper)
    assert np.array_equal(home, [.8, .15, .2])
    again = mobile_posture_targets(profile, names, limits, initial, 64, 23)[-1]
    np.testing.assert_array_equal([q for _, q in targets], [q for _, q in again])


def test_invalid_physical_bounds_or_home_are_rejected():
    profile = dict(joints=[dict(name='arm', home=0.)])
    with pytest.raises(ValueError, match='finite'):
        mobile_posture_targets(profile, ['arm'], [[-np.inf, np.inf]], [0.], 0, 1)
    with pytest.raises(ValueError, match='home'):
        mobile_posture_targets(profile, ['arm'], [[1., 2.]], [1.], 0, 1)


def test_same_robot_lidars_must_have_identical_acquisition_and_maximum_two(tmp_path):
    spec = dict(schema='robot_sensor_mounts/v1', camera={},
                lidar=dict(horizontal_samples=4, elevation_deg=[0.], min_range_m=.05, max_range_m=10.),
                robots=dict(test=dict(asset='unused.usd',
                                      camera=dict(parent='ee', translation_m=[0., 0., .1],
                                                  forward_parent=[0., 0., 1.], up_parent=[0., 1., 0.]),
                                      lidars=[dict(name='main', parent='base'), dict(name='aux', parent='base')])))
    (tmp_path/'robot.json').write_text(json.dumps(dict(robot_key='test', usd='robot.usd', sensors='sensors.json')))
    config = tmp_path/'sensors.json'
    config.write_text(json.dumps(spec))
    profile, _, config_path, asset, _ = mobile_profile(tmp_path)
    assert profile['robot_key'] == 'test'
    assert config_path == config and asset == tmp_path/'robot.usd'
    spec['robots']['test']['lidars'][1]['max_range_m'] = 9.
    config.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='must match'):
        mobile_profile(tmp_path)
    spec['robots']['test']['lidars'].append(dict(name='third', parent='base'))
    config.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='at most one'):
        mobile_profile(tmp_path)


def clear_summary():
    return dict(posture_count=79, diagnostic_prefix=False, lidar_ring_test_executed=True,
                original_revolute_joint_limits_unchanged=True, original_prismatic_joint_limits_unchanged=True,
                camera_gpu_render_verified=True, camera_all_sampled_views_clear=True,
                camera_all_sampled_housings_clear=True, lidar_origin_overlap_pose_count=0,
                lidar_all_test_targets_detected=True, posture_tracking_pass=True)


@pytest.mark.parametrize('field', ['diagnostic_prefix', 'lidar_ring_test_executed',
                                  'original_revolute_joint_limits_unchanged',
                                  'original_prismatic_joint_limits_unchanged', 'camera_gpu_render_verified',
                                  'camera_all_sampled_views_clear', 'camera_all_sampled_housings_clear',
                                  'lidar_all_test_targets_detected', 'posture_tracking_pass'])
def test_incomplete_or_failed_sampled_audit_cannot_pass(field):
    result = clear_summary()
    assert sampled_acceptance(result)
    result[field] = not result[field]
    assert not sampled_acceptance(result)


def test_scanner_origin_inside_robot_fails_acceptance():
    result = clear_summary()
    result['lidar_origin_overlap_pose_count'] = 1
    assert not sampled_acceptance(result)


def test_existing_robot_and_standalone_profile_are_mutually_exclusive():
    parsed = arguments(['--robot-dir', '/tmp/Robot_TIAGo', '--output', '/tmp/result'])
    assert parsed.robot is None and parsed.robot_dir == Path('/tmp/Robot_TIAGo')
    assert arguments(['--robot', 'ridgeback', '--output', '/tmp/result']).robot == 'ridgeback'
    with pytest.raises(SystemExit):
        arguments(['--robot', 'ridgeback', '--robot-dir', '/tmp/Robot_TIAGo', '--output', '/tmp/result'])


def test_successful_kit_exit_without_complete_passing_evidence_fails(tmp_path):
    assert saved_acceptance_status(tmp_path, 0) == 2
    summary = tmp_path/'summary.json'
    summary.write_text('{')
    assert saved_acceptance_status(tmp_path, 0) == 2
    summary.write_text(json.dumps(dict(complete=False, sampled_sensor_acceptance_pass=True)))
    assert saved_acceptance_status(tmp_path, 0) == 2
    summary.write_text(json.dumps(dict(complete=True, sampled_sensor_acceptance_pass=False)))
    assert saved_acceptance_status(tmp_path, 0) == 2
    summary.write_text(json.dumps(dict(complete=True, sampled_sensor_acceptance_pass=True)))
    assert saved_acceptance_status(tmp_path, 0) == 0
    assert saved_acceptance_status(tmp_path, 1) == 2
    (tmp_path/'error.txt').write_text('exception after writing summary')
    assert saved_acceptance_status(tmp_path, 0) == 2


def test_long_rigid_tool_chain_is_one_assembly_but_remote_arm_is_not():
    names = ['upper', 'elbow', 'wrist', 'adapter', 'sensor', 'gripper']
    all_joints = {name: {name} for name in names}
    fixed = {name: {name} for name in names}
    for a, b in zip(names, names[1:]):
        all_joints[a].add(b); all_joints[b].add(a)
    for a, b in zip(names[2:], names[3:]):
        fixed[a].add(b); fixed[b].add(a)
    allowed = fixed_chain_allowed_pairs(all_joints, fixed)
    assert 'gripper' in allowed['wrist'] and 'wrist' in allowed['gripper']
    assert 'elbow' in allowed['gripper']
    assert 'upper' not in allowed['gripper']
