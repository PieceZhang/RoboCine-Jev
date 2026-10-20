"""Recovered modules cannot hide physical failures or execute cancelled motion."""
import copy
import hashlib
import json

import pytest

from jev_agent_wbc.wbc_recovery_audit import UPPER, audit_recovery_rows


def fixture():
    task = dict(task_id='original', camera_trajectory=dict(id='path'),
                position_tolerance_m=.05, orientation_tolerance_rad=.15)
    def sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    context = dict(schema='g1_solver_task_context/v1', task_id='original',
        camera_path_id=sha(task['camera_trajectory']), task_sha256=sha(task))
    def fresh(stamp, seq, acquisition, sent):
        return dict(stamp_s=stamp, controller_status='fresh_command',
            task_feedback=dict(task_id='original', reason='accepted', feasible=None),
            current_solver_task_context=context,
            joint_names=list(UPPER), applied_joint_position_target=[0.]*15,
            applied_policy_velocity=[0., 0., .04], pink_upper_velocity={UPPER[0]:.01},
            pink_runtime_recovery=dict(seq=seq, acquisition_stamp_s=acquisition,
                command_sent_wall_s=sent, task_context_status='same_task', task_context=context),
            upper_execution_lease=dict(expired=False,
                checks=[dict(valid=True, sim_age_s=.02+i*.005, wall_age_s=.02+i*.001,
                             execution_horizon_s=.005) for i in range(4)],
                execution_end_check=dict(valid=True), execution_end_wall_age_s=.03,
                execution_end_sim_age_s=.04))
    rows = [fresh(1., 10, .96, 100.), fresh(1.02, 11, .98, 100.02),
            fresh(1.04, 12, 1., 100.04)]
    rows[1].update(controller_status='stale_or_failed_result_dropped',
        task_feedback=dict(task_id='original', reason='solver_failure', feasible=False),
        pink_runtime_recovery=dict(failure_kind='solver_failure', dropped_seq=11),
        pink_upper_velocity={}, upper_execution_lease=None,
        applied_policy_velocity=[0., 0., .024],
        policy_velocity_guard=dict(upper_reference_held=True, upper_velocity_target_zero=True,
                                  integral_frozen=True, collision_veto=False))
    return rows, [task]


def test_genuine_failure_remains_counted_after_bounded_fresh_recovery():
    rows, tasks = fixture(); result = audit_recovery_rows(rows, tasks)
    assert result['passed']
    assert result['reported_failure_samples'] == result['recovered_failure_samples'] == 1
    assert result['episodes'][0]['conservative_wall_duration_s'] == pytest.approx(.07)
    assert result['actual_physical_safety_verified'] is False


@pytest.mark.parametrize('bad', ['unrecovered', 'new_task', 'old_acquisition', 'old_sequence',
    'long_wall', 'nonfresh_velocity', 'reference_changed', 'no_brake', 'no_freeze',
    'collision', 'deadline', 'expired_substep', 'end_lease', 'changed_original', 'gap',
    'no_pre_fault_lease', 'boolean_sequence', 'masked_failure', 'hold_wrong_context',
    'masked_failure_and_reason', 'missing_active_feedback', 'sequence_regression'])
def test_unverified_or_unsafe_recovery_is_rejected(bad):
    rows, tasks = fixture()
    if bad == 'unrecovered': rows.pop()
    elif bad == 'new_task':
        tasks.append(dict(tasks[0], task_id='replacement'))
        rows[-1]['task_feedback']['task_id'] = 'replacement'
    elif bad == 'old_acquisition': rows[-1]['pink_runtime_recovery']['acquisition_stamp_s'] = .98
    elif bad == 'old_sequence': rows[-1]['pink_runtime_recovery']['seq'] = 11
    elif bad == 'long_wall': rows[-1]['pink_runtime_recovery']['command_sent_wall_s'] = 100.2
    elif bad == 'nonfresh_velocity': rows[1]['pink_upper_velocity'] = {UPPER[0]:.01}
    elif bad == 'reference_changed': rows[1]['applied_joint_position_target'][0] = .001
    elif bad == 'no_brake': rows[1]['applied_policy_velocity'][2] = .04
    elif bad == 'no_freeze': rows[1]['policy_velocity_guard']['integral_frozen'] = False
    elif bad == 'collision': rows[1]['task_feedback']['reason'] = 'self_collision'
    elif bad == 'deadline': rows[1]['task_feedback']['reason'] = 'deadline_exceeded'
    elif bad == 'expired_substep': rows[-1]['upper_execution_lease']['checks'][2]['valid'] = False
    elif bad == 'end_lease': rows[-1]['upper_execution_lease']['execution_end_wall_age_s'] = .061
    elif bad == 'changed_original': tasks[0]['position_tolerance_m'] = .1
    elif bad == 'gap': rows[-1]['stamp_s'] += .02
    elif bad == 'no_pre_fault_lease': rows[0]['upper_execution_lease'] = None
    elif bad == 'masked_failure': rows[1]['task_feedback']['feasible'] = None
    elif bad == 'hold_wrong_context': rows[1]['current_solver_task_context'] = None
    elif bad == 'masked_failure_and_reason':
        rows[1]['task_feedback'].update(feasible=None, reason='command_stale')
    elif bad == 'missing_active_feedback':
        hidden = copy.deepcopy(rows[1]); hidden['stamp_s'] = 1.04
        hidden.pop('task_feedback'); hidden['applied_joint_position_target'][0] = 1.
        hidden['applied_policy_velocity'] = [.5, .5, 1.]
        rows[-1]['stamp_s'] = 1.06
        rows.insert(2, hidden)
    elif bad == 'sequence_regression':
        rows[1]['pink_runtime_recovery']['dropped_seq'] = 1
        rows[-1]['pink_runtime_recovery']['seq'] = 2
    else: rows[-1]['pink_runtime_recovery']['seq'] = True
    assert not audit_recovery_rows(rows, tasks)['passed']


def test_original_simulation_budget_cannot_be_replaced_by_fast_wall_time():
    rows, tasks = fixture()
    held = rows[1]; rows = rows[:2]
    for index in range(2, 11):
        row = copy.deepcopy(held); row['stamp_s'] = 1.+index*.02
        rows.append(row)
    assert 'module_recovery_exceeds_original_simulation_budget' in audit_recovery_rows(rows, tasks)['audit_errors']


def test_expired_transport_remains_unknown_but_recovery_is_independently_checked():
    rows, tasks = fixture()
    rows[1]['pink_runtime_recovery']['failure_kind'] = 'stale_or_mismatched_transport'
    rows[1]['task_feedback'].update(reason='command_stale', feasible=None)
    result = audit_recovery_rows(rows, tasks)
    assert result['passed'] and result['reported_failure_samples'] == 0
    assert result['raw_module_result_samples'] == result['recovered_raw_module_result_samples'] == 1
    rows[-1]['pink_runtime_recovery']['acquisition_stamp_s'] = .98
    assert not audit_recovery_rows(rows, tasks)['passed']
