"""Independent rejection of raw joint-limit failures despite reported passes."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('limits_auditor', SCRIPTS/'audit_wbc_limits.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, *, robot='husky', q=0., qd=0., **changes):
    row = dict(stamp_s=.01, joint_names=['j'], joint_units=['rad'],
        q=[q], qd=[qd], measured_configuration_ok=True, maximum_actual_velocity_ratio=0.)
    row.update(changes)
    if robot == 'tiago':
        data = dict(raw_full_dof_guard=dict(dof_names=['j'], joint_units=['rad'], lower=[None],
            upper=[None], speed=[8.], position_numerical_tolerance=1e-5,
            velocity_numerical_tolerance=.0001))
        path = tmp_path/'configuration.json'
    elif robot in ('g1', 'go2'):
        data = dict(joint_names=['j'], position_limits=[[-1., 1.]], velocity_limits=[1.])
        row.update(measured_joint_velocity=[qd], measured_joint_velocity_peak_abs=[abs(qd)],
            minimum_substep_joint_margin_rad=1.-abs(q))
        row.update(changes)
        path = tmp_path/'actuator_audit.json'
    else:
        data = dict(joint_names=['j'], joint_units=['rad'], physical_limits=[[-1., 1.]], velocity_caps=[.65])
        path = tmp_path/'actuator_audit.json'
    path.write_text(json.dumps(data))
    (tmp_path/'measurements.jsonl').write_text(json.dumps(row)+'\n')
    return module.audit_limits(tmp_path, robot)


def test_small_command_and_reported_pass_do_not_excuse_actual_speed(tmp_path):
    result = fixture(tmp_path, qd=.6501615, command=[.04965])
    assert not result['passed']
    assert result['violations'][0]['kind'] == 'speed'


def test_tiago_checks_raw_instant_speed_not_position_average(tmp_path):
    result = fixture(tmp_path, robot='tiago', q=120., qd=8.7213, command=[-.8029])
    assert not result['passed']
    assert result['violation_count'] == 1  # explicitly continuous wheel


def test_legged_final_sample_cannot_hide_substep_peak(tmp_path):
    result = fixture(tmp_path, robot='go2', qd=.1, measured_joint_velocity_peak_abs=[1.1])
    assert not result['passed']


def test_legged_final_sample_cannot_hide_substep_position_excursion(tmp_path):
    result = fixture(tmp_path, robot='g1', minimum_substep_joint_margin_rad=-.001)
    assert not result['passed']


@pytest.mark.parametrize('changes', [dict(qd=[True]), dict(qd=[float('nan')]),
    dict(joint_names=['wrong']), dict(qd=[]), dict(joint_units=['m'])])
def test_missing_or_mislabeled_measurements_fail_closed(tmp_path, changes):
    assert not fixture(tmp_path, **changes)['passed']


def test_original_roundoff_tolerance_and_conservative_cap(tmp_path):
    assert fixture(tmp_path, qd=.6500001)['passed']


def test_bounded_position_violation_is_independent_of_speed(tmp_path):
    assert not fixture(tmp_path, q=1.0001)['passed']


def test_invalid_substep_peak_cannot_supply_false_proof(tmp_path):
    assert not fixture(tmp_path, robot='go2', qd=.5, measured_joint_velocity_peak_abs=[.1])['passed']
