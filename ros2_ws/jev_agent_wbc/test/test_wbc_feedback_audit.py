"""Public Agent feedback must retain actual measurements and task authority."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
SPEC = importlib.util.spec_from_file_location('feedback_audit_test',
    Path(__file__).resolve().parents[1]/'scripts/audit_wbc_feedback.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture(folder):
    task = dict(task_id='shot', position_tolerance_m=.05, orientation_tolerance_rad=.15,
                camera_trajectory=dict(id='original'), observation_id='physical')
    context = AUDIT.expected_context(task)
    feedback = dict(task_id='shot', stamp_s=1., feasible=None, reason='accepted',
        safety_certificate=False, evidence_kind='measured', position_error_m=.01,
        orientation_error_rad=.02, camera_within_tolerance=True,
        remaining_adjustment={part:None for part in AUDIT.PARTS})
    row = dict(stamp_s=1., task_feedback=feedback, current_solver_task_context=context,
        controller_status='fresh_command', pink_runtime_recovery=dict(seq=4, acquisition_stamp_s=.96,
            task_context_status='same_task', task_context=context), pink_upper_velocity={'arm':.01})
    (folder/'sources').mkdir();(folder/'sources/solver_context.py').write_text('frozen context marker\n')
    (folder/'summary.json').write_text(json.dumps(dict(robot='g1')))
    (folder/'timed_tasks.json').write_text(json.dumps([task]))
    return row


def result(folder, row):
    (folder/'measurements.jsonl').write_text(json.dumps(row)+'\n')
    return AUDIT.audit_feedback(folder)


def test_current_task_fresh_unknown_feasibility_and_tolerance(tmp_path):
    assert result(tmp_path, fixture(tmp_path))['passed']


@pytest.mark.parametrize('bad', ['old_context', 'wrong_echo', 'late', 'bool_sequence',
    'numeric_feasible', 'tolerance', 'certificate', 'reserve', 'nonfresh_velocity'])
def test_inconsistent_public_feedback_never_passes(tmp_path, bad):
    row = fixture(tmp_path);feedback = row['task_feedback'];packet = row['pink_runtime_recovery']
    if bad == 'old_context': row['current_solver_task_context'] = None
    elif bad == 'wrong_echo': packet['task_context'] = None
    elif bad == 'late': packet['acquisition_stamp_s'] = .9
    elif bad == 'bool_sequence': packet['seq'] = True
    elif bad == 'numeric_feasible': feedback['feasible'] = 1
    elif bad == 'tolerance': feedback['camera_within_tolerance'] = False
    elif bad == 'certificate': feedback['safety_certificate'] = True
    elif bad == 'reserve': feedback['remaining_adjustment']['mobile_platform'] = True
    else:
        row['controller_status'] = 'waiting_hold_upper_targets'
        feedback['reason'] = 'command_stale'
    assert not result(tmp_path, row)['passed']


def test_missing_fresh_is_command_stale_and_holds_upper_velocity(tmp_path):
    row = fixture(tmp_path);row['controller_status'] = 'superseded_task_hold_upper_targets'
    row['pink_upper_velocity'] = {};row['task_feedback']['reason'] = 'command_stale'
    assert result(tmp_path, row)['passed']
    row['task_feedback']['reason'] = 'accepted'
    assert not result(tmp_path, row)['passed']


def test_outside_tolerance_is_reported_even_when_run_rms_passes(tmp_path):
    row = fixture(tmp_path);row['task_feedback']['position_error_m'] = .06
    row['task_feedback']['camera_within_tolerance'] = False
    audit = result(tmp_path, row)
    assert audit['passed'] and audit['camera_tolerance_counts'] == {'outside':1}


def test_same_id_and_path_cannot_reuse_reply_after_new_tolerance(tmp_path):
    row = fixture(tmp_path)
    tasks = json.loads((tmp_path/'timed_tasks.json').read_text())
    tasks[0]['position_tolerance_m'] = .04
    (tmp_path/'timed_tasks.json').write_text(json.dumps(tasks))
    assert not result(tmp_path, row)['passed']


def lease():
    return dict(expired=False, checks=[dict(valid=True, sim_age_s=.02+i*.005,
        wall_age_s=.02+i*.001, execution_horizon_s=.005) for i in range(4)],
        execution_end_check=dict(valid=True), execution_end_wall_age_s=.025,
        execution_end_sim_age_s=.04)


def test_full_tick_and_each_substep_have_original_lease(tmp_path):
    row = fixture(tmp_path);row['upper_execution_lease'] = lease()
    assert result(tmp_path, row)['passed']


@pytest.mark.parametrize('bad', ['missing', 'malformed', 'short', 'wall', 'simulation', 'ending', 'flag'])
def test_expired_or_incomplete_tick_cannot_be_reported_fresh(tmp_path, bad):
    row = fixture(tmp_path);report = lease();row['upper_execution_lease'] = report
    if bad == 'missing': row['upper_execution_lease'] = None
    elif bad == 'malformed': report['checks'][0] = False
    elif bad == 'short': report['checks'].pop()
    elif bad in ('wall', 'simulation'):
        key = 'wall_age_s' if bad == 'wall' else 'sim_age_s'
        report['checks'][-1][key] = .058
    elif bad == 'ending': report['execution_end_wall_age_s'] = .061
    else: report['checks'][0]['valid'] = 1
    assert not result(tmp_path, row)['passed']


def test_substep_expiry_holds_remainder_and_reports_stale(tmp_path):
    row = fixture(tmp_path);report = lease()
    report.update(expired=True, first_expired_substep=2)
    report['checks'] = report['checks'][:2]+[dict(valid=False, reason='command_stale')]
    row['upper_execution_lease'] = report
    row['controller_status'] = 'command_expired_during_execution_hold_upper_targets'
    row['task_feedback']['reason'] = 'command_stale';row['pink_upper_velocity'] = {}
    assert result(tmp_path, row)['passed']
    report['checks'].append(lease()['checks'][-1])
    assert not result(tmp_path, row)['passed']
