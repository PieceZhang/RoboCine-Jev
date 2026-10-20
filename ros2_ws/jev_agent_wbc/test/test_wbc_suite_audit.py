"""Checks for false passes in independent full-suite coverage validation."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('suite_auditor', SCRIPTS/'audit_wbc_suite.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(t=0., x=0., rot=0.):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec([0., 0., rot]).as_matrix()
    matrix[0, 3] = x
    return dict(stamp_s=t, T_world_base=matrix.tolist(), T_world_camera_optical=matrix.tolist(),
        T_world_reference_optical=matrix.tolist(), joint_positions=[0.], joint_units=['rad'],
        requested_body_velocity=[.1, 0., 0.], measured_body_velocity=[.1, 0., 0.],
        base_linear_velocity_world=[0., 0., 0.], base_angular_velocity_world=[0., 0., 0.], phase='move')


def test_backward_must_actually_move_backward():
    assert module.motion([row(x=0.), row(x=-.35)], 'backward', False)['passed']
    assert not module.motion([row(x=0.), row(x=.35)], 'backward', False)['passed']


def test_lateral_motion_checks_cross_track_drift():
    rows = [row(), row()]
    rows[-1]['T_world_base'][1][3] = .4
    assert module.motion(rows, 'left', False)['passed']
    rows[-1]['T_world_base'][0][3] = .09
    assert not module.motion(rows, 'left', False)['passed']


def test_yaw_direction_cannot_be_reversed():
    assert module.motion([row(), row(rot=.8)], 'yaw_ccw', False)['passed']
    assert not module.motion([row(), row(rot=.8)], 'yaw_cw', False)['passed']


def test_legged_velocity_gate_recomputes_measured_tail():
    rows = [row(t=i*.02) for i in range(50)]
    assert module.motion(rows, 'forward', True)['passed']
    for r in rows[25:]:
        r['measured_body_velocity'] = [.04, 0., 0.]
    assert not module.motion(rows, 'forward', True)['passed']


def test_legged_case_rejects_rewritten_preferred_velocity():
    rows = [row(t=i*.02) for i in range(50)]
    rows[-1]['requested_body_velocity'] = [.01, 0., 0.]
    with pytest.raises(ValueError, match='requested velocity changed'):
        module.motion(rows, 'forward', True)


def test_camera_hold_cannot_pass_nontrivial_shot():
    rows = [row(i*.1, x=.11*np.sin(i*.1), rot=.32*np.sin(i*.1)) for i in range(63)]
    assert module.excitation(rows)['passed']
    for r in rows:
        r['T_world_camera_optical'] = np.eye(4).tolist()
    assert not module.excitation(rows)['passed']


def test_small_reference_cannot_pass_by_being_inside_tolerance():
    rows = [row(i*.1, x=.01*np.sin(i*.1), rot=.02*np.sin(i*.1)) for i in range(63)]
    assert not module.excitation(rows)['passed']


def test_rotation_span_handles_quaternion_antipodes():
    matrices = [np.eye(4) for _ in range(2)]
    matrices[0][:3, :3] = Rotation.from_euler('z', np.pi-.1).as_matrix()
    matrices[1][:3, :3] = Rotation.from_euler('z', -np.pi+.1).as_matrix()
    assert module.rotation_span(matrices) == pytest.approx(.2)


def test_diagnostic_and_missing_phases_cannot_be_full_suite(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps(dict(all_pass=True, diagnostic_only=True, cases=[])))
    (tmp_path/'measurements.jsonl').write_text(json.dumps(row())+'\n')
    result = module.audit_suite(tmp_path, 'g1')
    assert not result['passed']
    assert 'diagnostic_is_not_full_suite' in result['audit_errors']


def test_camera_excitation_still_requires_all_joints_to_stop(tmp_path):
    rows = [row(2.+i*.1, x=.11*np.sin(i*.1), rot=.32*np.sin(i*.1)) for i in range(63)]
    for r in rows:
        r['case'] = 'camera_6d'
    for i in range(51):
        r = row(8.3+i*.02)
        r.update(case='camera_6d', phase='stop', joint_positions=[.02*i])
        rows.append(r)
    (tmp_path/'summary.json').write_text(json.dumps(dict(all_pass=True, cases=[dict(case='camera_6d', passed=True)])))
    (tmp_path/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = module.audit_suite(tmp_path, 'go2', camera_only=True)
    assert not result['passed']
    assert result['checks']['camera_6d']['movement']['passed']
    assert not result['checks']['camera_6d']['strict_stop']['passed']


def test_independent_arm_cannot_pass_when_one_axis_did_not_move():
    names = ['j'+str(i) for i in range(6)]
    initial = row()
    initial.update(joint_names=names, joint_positions=[0.]*6, phase='before')
    moved = row(.01)
    moved.update(joint_names=names, joint_positions=[.1]*6, phase='independent_arm')
    stopped = row(.02)
    stopped.update(joint_names=names, joint_positions=[-.1]*6, phase='independent_arm')
    configuration = dict(robot=dict(arm_joints=names), arm_model_limits=[[-1., 1.]]*6)
    assert module.independent_arm([initial, moved, stopped], configuration, 'husky')['passed']
    moved['joint_positions'][2] = stopped['joint_positions'][2] = 0.
    assert not module.independent_arm([initial, moved, stopped], configuration, 'husky')['passed']


def test_independent_arm_actual_base_drift_cannot_be_hidden_by_native_pass():
    names = ['j'+str(i) for i in range(6)]
    rows = [row(i*.01, x=.04 if i == 2 else 0.) for i in range(3)]
    for i, r in enumerate(rows):
        r.update(joint_names=names, joint_positions=[.1*i]*6,
            phase='independent_arm' if i else 'before')
    configuration = dict(robot=dict(arm_joints=names), arm_model_limits=[[-1., 1.]]*6)
    assert not module.independent_arm(rows, configuration, 'ridgeback')['passed']
