"""Stored nonfinite metrics must not turn physical probe failures into a pass."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/audit_legged_camera_probe.py'
SPEC = importlib.util.spec_from_file_location('legged_probe_audit', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(folder):
    rows = [dict(stamp_s=t, phase='pink_camera', T_world_camera_optical=np.eye(4).tolist(),
                 T_world_reference_optical=np.eye(4).tolist(), T_world_pelvis=np.eye(4).tolist(),
                 position_error_m=0., orientation_error_rad=0., joint_positions={'arm': 0.})
            for t in (.02, .04)]
    summary = dict(position_rmse_m=0., orientation_rmse_rad=0., joint_ownership=dict(pink=['arm']))
    return rows, summary


def save(folder, rows, summary):
    (folder/'measured.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (folder/'summary.json').write_text(json.dumps(summary))


def test_consistent_measured_probe_passes_audit(tmp_path):
    rows, summary = fixture(tmp_path)
    save(tmp_path, rows, summary)
    result = MODULE.audit(tmp_path)
    assert result['audit_passed'] and result['local_tracking_pass']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), True])
def test_nonfinite_or_boolean_stored_errors_never_pass(tmp_path, value):
    rows, summary = fixture(tmp_path)
    for row in rows:
        row['position_error_m'] = row['orientation_error_rad'] = value
    summary['position_rmse_m'] = summary['orientation_rmse_rad'] = value
    save(tmp_path, rows, summary)
    result = MODULE.audit(tmp_path)
    assert not result['audit_passed'] and not result['local_tracking_pass']
    assert 'stored_error_does_not_match_physical_pose' in result['errors']


def test_invalid_transform_is_recorded_as_failure(tmp_path):
    rows, summary = fixture(tmp_path)
    rows[0]['T_world_camera_optical'] = np.zeros((2, 2)).tolist()
    save(tmp_path, rows, summary)
    result = MODULE.audit(tmp_path)
    assert not result['audit_passed']
    assert 'invalid_physical_or_reference_transform' in result['errors']
