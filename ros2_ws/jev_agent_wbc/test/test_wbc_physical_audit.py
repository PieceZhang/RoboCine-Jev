"""Corrupt stored evidence must not manufacture successful physical acceptance."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from jev_agent_wbc.representation import Pose, Observation, Box
from jev_agent_wbc.wbc_protocol import CameraTrajectory, TimedTask, optical_camera

SPEC = importlib.util.spec_from_file_location('physical_wbc_audit',
    Path(__file__).resolve().parents[1]/'scripts/audit_wbc_physical.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture(folder):
    sources = folder/'sources'
    sources.mkdir()
    (sources/'runtime.py').write_text('frozen diagnostic source\n')
    occupancy = {part:(Box((0., 0., 0.), (.1, .1, .1)),)
        for part in ('mobile_platform', 'camera_manipulator')}
    initial = Observation(0., Pose((0., 0., 0.)), Pose((0., 0., 0.)), occupancy)
    task = TimedTask('one', 0., initial.id,
        CameraTrajectory(0., (0., 1.), (Pose((0., 0., 0.)), Pose((0., 0., 0.)))), .03, .08)
    camera = AUDIT.pose(optical_camera(Pose((0., 0., 0.))).to_dict())
    rows = []
    for i in range(1, 21):
        stamp = i*.1
        feedback = dict(task_id='one', camera_path_id=task.trajectory.id, stamp_s=stamp,
            actual_camera=Pose((0., 0., 0.)).to_dict(), position_error_m=0., orientation_error_rad=0.,
            camera_reference_changed=False,
            constraints={k:dict(status='unknown', stamp_s=stamp, scope='not_available')
                for k in ('configuration', 'actuation', 'self_collision', 'environment', 'policy_domain')})
        observation = Observation(stamp, Pose((0., 0., 0.)), Pose((0., 0., 0.)), occupancy)
        feedback['geometry'] = observation.summary()
        if stamp > 1.:
            feedback.update(tracking_mode='explicit_terminal_hold', reference_stamp_s=1.)
        rows.append(dict(stamp_s=stamp, case='camera', phase='move' if stamp <= 1. else 'stop',
            T_world_base=np.eye(4).tolist(), T_world_camera_optical=camera.tolist(),
            T_world_reference_optical=camera.tolist(), joint_positions=[0., 0.],
            joint_names=['slide', 'turn'], joint_units=['m', 'rad'],
            measured_observation=observation.to_dict(),
            base_linear_velocity_world=[0., 0., 0.], base_angular_velocity_world=[0., 0., 0.],
            task_feedback=feedback))
    (folder/'timed_tasks.json').write_text(json.dumps([task.to_dict()]))
    (folder/'accepted_observations.json').write_text(json.dumps([initial.to_dict()]))
    (folder/'configuration.json').write_text(json.dumps(dict(joint_names=['slide', 'turn'])))
    (folder/'provenance.json').write_text(json.dumps(dict(measurement_hz=10., frozen_sha256={
        'runtime.py':hashlib.sha256((sources/'runtime.py').read_bytes()).hexdigest()})))
    (folder/'summary.json').write_text(json.dumps(dict(run_complete=True, sim_seconds=2., wall_seconds=2.)))
    save(folder, rows)
    return rows


def save(folder, rows):
    (folder/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_valid_physical_evidence_and_explicit_terminal_hold(tmp_path):
    fixture(tmp_path)
    result = AUDIT.audit(tmp_path)
    assert result['physical_acceptance_pass'], result
    assert result['tasks'][0]['strict_stop']['measured_window_s'] >= .5


@pytest.mark.parametrize('corruption', ['reference', 'timestamp', 'optical_camera', 'path_binding',
                                       'silent_hold', 'stale_constraint', 'source'])
def test_original_task_and_fresh_measurement_bindings_are_required(tmp_path, corruption):
    rows = fixture(tmp_path)
    row = rows[-1]
    if corruption == 'reference':
        row['T_world_reference_optical'][0][3] = .1
    elif corruption == 'timestamp':
        row['task_feedback']['stamp_s'] -= .1
    elif corruption == 'optical_camera':
        row['T_world_camera_optical'][0][3] = .1
    elif corruption == 'path_binding':
        row['task_feedback']['camera_path_id'] = 'other'
    elif corruption == 'silent_hold':
        row['task_feedback'].pop('tracking_mode')
    elif corruption == 'stale_constraint':
        row['task_feedback']['constraints']['configuration'] = dict(status='pass', stamp_s=0.)
    else:
        (tmp_path/'sources/runtime.py').write_text('changed\n')
    save(tmp_path, rows)
    result = AUDIT.audit(tmp_path)
    assert not result['audit_passed'] and not result['physical_acceptance_pass']


@pytest.mark.parametrize('bad', [True, float('nan'), float('inf')])
def test_invalid_stored_metrics_cannot_pass(tmp_path, bad):
    rows = fixture(tmp_path)
    rows[1]['task_feedback']['position_error_m'] = bad
    save(tmp_path, rows)
    assert not AUDIT.audit(tmp_path)['physical_acceptance_pass']


def test_all_joints_including_linear_axes_must_stop(tmp_path):
    rows = fixture(tmp_path)
    rows[-2]['joint_positions'][0] = .003
    save(tmp_path, rows)
    result = AUDIT.audit(tmp_path)
    assert result['audit_passed']
    assert not result['tasks'][0]['strict_stop']['passed']
    assert not result['physical_acceptance_pass']


def test_vertical_base_velocity_cannot_hide_behind_planar_stop(tmp_path):
    rows = fixture(tmp_path)
    rows[-1]['base_linear_velocity_world'][2] = .025
    save(tmp_path, rows)
    assert not AUDIT.audit(tmp_path)['physical_acceptance_pass']


def test_controller_pass_does_not_override_pace(tmp_path):
    fixture(tmp_path)
    (tmp_path/'summary.json').write_text(json.dumps(dict(run_complete=True, all_pass=True,
        timing_pass=True, sim_seconds=2., wall_seconds=3.)))
    result = AUDIT.audit(tmp_path)
    assert result['all_tasks_pass'] and not result['timing_pass']
    assert not result['physical_acceptance_pass']


def test_missing_measurement_interval_cannot_shorten_stop(tmp_path):
    rows = fixture(tmp_path)
    save(tmp_path, rows[:5]+rows[10:])
    result = AUDIT.audit(tmp_path)
    assert 'physical_measurement_gap' in result['audit_errors']
    assert not result['physical_acceptance_pass']


def test_uniform_decimation_cannot_hide_physical_samples(tmp_path):
    rows = fixture(tmp_path)
    save(tmp_path, rows[1::2])
    result = AUDIT.audit(tmp_path)
    assert 'physical_samples_missing_or_rate_changed' in result['audit_errors']
    assert not result['physical_acceptance_pass']


def test_feedback_cannot_skip_a_bad_physical_sample(tmp_path):
    rows = fixture(tmp_path)
    rows[5].pop('task_feedback')
    save(tmp_path, rows)
    result = AUDIT.audit(tmp_path)
    assert 'active_task_feedback_missing_or_wrong_task' in result['audit_errors']
    assert not result['physical_acceptance_pass']


def test_removing_a_joint_cannot_manufacture_strict_stop(tmp_path):
    rows = fixture(tmp_path)
    for row in rows:
        for key in ('joint_positions', 'joint_names', 'joint_units'):
            row[key] = row[key][:1]
    save(tmp_path, rows)
    result = AUDIT.audit(tmp_path)
    assert 'joint_measurements_do_not_cover_physical_articulation' in result['audit_errors']
    assert not result['physical_acceptance_pass']


def test_task_must_bind_real_admission_observation(tmp_path):
    fixture(tmp_path)
    path = tmp_path/'timed_tasks.json'
    tasks = json.loads(path.read_text())
    tasks[0]['observation_id'] = 'fabricated'
    path.write_text(json.dumps(tasks))
    result = AUDIT.audit(tmp_path)
    assert 'task_admission_not_bound_to_fresh_measured_observation' in result['audit_errors']
    assert not result['physical_acceptance_pass']


def test_settle_prefix_cannot_be_removed(tmp_path):
    rows = fixture(tmp_path)
    save(tmp_path, rows[1:])
    result = AUDIT.audit(tmp_path)
    assert 'physical_log_start_incomplete' in result['audit_errors']
    assert not result['physical_acceptance_pass']
