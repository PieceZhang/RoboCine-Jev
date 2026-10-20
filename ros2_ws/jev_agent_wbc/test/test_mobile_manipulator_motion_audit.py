"""Independent motion audit rejects rotational drift and misleading WBC results."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mobile_manipulator.audit_motion import audit, rotation_span_rad


@pytest.fixture
def evidence(tmp_path):
    sources = tmp_path/'sources'; sources.mkdir()
    source = sources/'0_control.py'; source.write_text('frozen controller source\n')
    config = dict(spec=dict(joints=[dict(name='arm', lower=-1., upper=1.)]),
                  dof_names=['arm'], controlled_joint_units=dict(arm='rad'))
    provenance = dict(sources={'/original/control.py': hashlib.sha256(source.read_bytes()).hexdigest()})
    phases = ['forward', 'backward', 'left', 'right', 'diagonal_ne', 'diagonal_sw',
              'independent_arm', 'whole_body_camera']+['final_stop']*51
    rows = []
    for i, phase in enumerate(phases):
        row = dict(t_s=(i+1)*.01, phase=phase, q=[0.], qd=[0.],
                   T_world_base=np.eye(4).tolist(), T_world_camera=np.eye(4).tolist())
        if phase == 'whole_body_camera':
            row['reference'] = np.eye(4).tolist()
        rows.append(row)
    acceptance = dict(complete=True, passed=True, diagnostic_only=False, simulated_seconds=len(rows)*.01,
                      phases=[dict(phase='whole_body_camera', position_rmse_m=0., orientation_rmse_rad=0.)])
    for name, data in [('acceptance.json', acceptance), ('configuration.json', config), ('provenance.json', provenance)]:
        (tmp_path/name).write_text(json.dumps(data))
    write_rows(tmp_path, rows)
    return tmp_path


def write_rows(folder, rows):
    (folder/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


def read_rows(folder):
    return [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]


def test_consistent_full_motion_evidence_passes_strict_audit(evidence):
    result = audit(evidence)
    assert result['motion_acceptance_verified']
    assert result['measured_stop_pass'] and result['recomputed_wbc_error_thresholds_pass']
    assert result['reported_wbc_errors_consistent']


@pytest.mark.parametrize('transform', ['T_world_base', 'T_world_camera'])
def test_rotation_in_place_cannot_pass_position_only_stop(evidence, transform):
    rows = read_rows(evidence)
    tail = [row for row in rows if row['phase'] == 'final_stop']
    for angle, row in zip(np.linspace(0., .025, len(tail)), tail):
        matrix = np.eye(4); matrix[:3, :3] = Rotation.from_euler('z', angle).as_matrix()
        row[transform] = matrix.tolist()
    write_rows(evidence, rows)
    result = audit(evidence)
    assert result['rigid_transforms_pass']
    assert not result['measured_stop_pass']
    assert not result['motion_acceptance_verified']


def test_rotation_span_catches_motion_on_both_sides_of_first_orientation():
    rotations = Rotation.from_euler('z', [0., -.009, .009]).as_matrix()
    assert rotation_span_rad(rotations) == pytest.approx(.018, abs=1e-12)


@pytest.mark.parametrize('position_error,orientation_error', [(.05, 0.), (0., .12)])
def test_claimed_pass_cannot_override_recomputed_wbc_threshold(evidence, position_error, orientation_error):
    rows = read_rows(evidence)
    for row in rows:
        if row['phase'] == 'whole_body_camera':
            reference = np.eye(4)
            reference[0, 3] = position_error
            reference[:3, :3] = Rotation.from_euler('z', orientation_error).as_matrix()
            row['reference'] = reference.tolist()
    write_rows(evidence, rows)
    acceptance = json.loads((evidence/'acceptance.json').read_text())
    acceptance['phases'][0].update(position_rmse_m=position_error, orientation_rmse_rad=orientation_error)
    (evidence/'acceptance.json').write_text(json.dumps(acceptance))
    result = audit(evidence)
    assert result['reported_wbc_errors_consistent']
    assert not result['recomputed_wbc_error_thresholds_pass']
    assert not result['motion_acceptance_verified']


def test_reported_error_disagreement_is_rejected_even_below_threshold(evidence):
    rows = read_rows(evidence)
    for row in rows:
        if row['phase'] == 'whole_body_camera':
            row['reference'][0][3] = .02
    write_rows(evidence, rows)
    result = audit(evidence)
    assert result['recomputed_wbc_error_thresholds_pass']
    assert not result['reported_wbc_errors_consistent']
    assert not result['motion_acceptance_verified']
