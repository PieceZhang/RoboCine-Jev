"""Independent, conservative evidence for bounded G1 module recovery.

This is an offline audit, not execution authority. A recovered solver failure
remains a real failure in public feedback. It cannot excuse a physical limit,
collision, task error, missed stopping gate, or an unrecovered deadline.
"""
import hashlib
import json
import math

import numpy as np


UPPER = ('waist_yaw_joint',) + tuple(
    f'{side}_{joint}_joint' for side in ('left', 'right') for joint in
    ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
     'wrist_roll', 'wrist_pitch', 'wrist_yaw'))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite numeric evidence required')
    return float(value)


def _context(task):
    def sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    return dict(schema='g1_solver_task_context/v1', task_id=task['task_id'],
                camera_path_id=sha(task['camera_trajectory']), task_sha256=sha(task))


def _fresh(row, context):
    packet = row['pink_runtime_recovery']
    stamp = _number(row['stamp_s'])
    acquisition = _number(packet['acquisition_stamp_s'])
    if (row.get('controller_status') != 'fresh_command'
            or packet.get('task_context_status') != 'same_task'
            or packet.get('task_context') != context
            or row.get('current_solver_task_context') != context
            or type(packet.get('seq')) is not int or packet['seq'] < 0
            or not -1e-8 <= stamp-acquisition <= .06+1e-8):
        raise ValueError('resumption requires original-context fresh motion')
    lease = row['upper_execution_lease']
    checks = lease['checks']
    if (lease.get('expired') is not False or len(checks) != 4
            or lease.get('execution_end_check', {}).get('valid') is not True):
        raise ValueError('resumption requires all four original motion leases')
    for check in checks:
        sim, wall = (_number(check[k]) for k in ('sim_age_s', 'wall_age_s'))
        if (check.get('valid') is not True or abs(_number(check['execution_horizon_s'])-.005) > 1e-8
                or min(sim, wall) < -1e-8 or max(sim, wall)+.005 > .06+1e-8):
            raise ValueError('resumption exceeds original motion lease')
    for key in ('execution_end_sim_age_s', 'execution_end_wall_age_s'):
        if not 0 <= _number(lease[key]) <= .06+1e-8:
            raise ValueError('resumption execution ended outside original lease')
    return packet


def audit_recovery_rows(rows, tasks):
    """Require fresh resumption within the existing 180 ms recovery budget.

    The last pre-fault fresh request's wall-send time gives a conservative
    earlier bound for fault onset. Resumed request send plus its recorded full
    execution age bounds completion. Both wall and simulation bounds must pass.
    Logs missing these facts remain unverified; timing averages cannot replace
    them. Mid-tick expiry needs explicit per-write evidence and is rejected here
    until that separate evidence is independently audited.
    """
    errors, episodes = [], []
    originals = {task['task_id']: task for task in tasks}
    active = None
    previous = None
    previous_fresh = None
    previous_context = None
    reported_failures = raw_module_results = 0

    def reject(value):
        if value not in errors:
            errors.append(value)

    try:
        for row in rows:
            if previous is not None and abs(_number(row['stamp_s'])-_number(previous['stamp_s'])-.02) > 1e-8:
                raise ValueError('recovery evidence requires contiguous original 50 Hz measurements')
            feedback = row.get('task_feedback')
            if feedback is None:
                if active is not None or row.get('controller_status') == 'stale_or_failed_result_dropped':
                    reject('module_recovery_feedback_missing')
                previous = row
                continue
            context = _context(originals[feedback['task_id']])
            stamp = _number(row['stamp_s'])
            fresh = row.get('controller_status') == 'fresh_command'
            failure = feedback.get('feasible') is False
            packet = row.get('pink_runtime_recovery', {})
            raw_result = (row.get('controller_status') == 'stale_or_failed_result_dropped'
                          and packet.get('failure_kind') in
                          ('solver_failure', 'transport_failure', 'stale_or_mismatched_transport'))
            if raw_result:
                raw_module_results += 1
            if (raw_result and packet.get('failure_kind') in ('solver_failure', 'transport_failure')
                    and (not failure or feedback.get('reason') not in ('solver_failure', 'solver_infeasible'))):
                reject('actual_module_failure_was_not_preserved_in_public_feedback')
            if (feedback.get('reason') in ('solver_failure', 'solver_infeasible', 'deadline_exceeded')
                    and not failure):
                reject('actual_module_failure_was_not_preserved_in_public_feedback')
            if failure:
                reported_failures += 1
                if (feedback.get('reason') not in ('solver_failure', 'solver_infeasible')
                        or packet.get('failure_kind') not in ('solver_failure', 'transport_failure')
                        or row.get('controller_status') != 'stale_or_failed_result_dropped'):
                    reject('failure_is_not_a_recoverable_module_result')
            if failure or raw_result:
                if active is None:
                    if (previous_fresh is None or previous_context != context or previous is None):
                        raise ValueError('no same-task pre-fault execution bound')
                    _fresh(previous_fresh, context)
                    active = dict(task_id=feedback['task_id'], context=context,
                        first_control_stamp_s=stamp-.02, failed_samples=0,
                        raw_module_result_samples=0,
                        pre_fault_seq=previous_fresh['pink_runtime_recovery']['seq'],
                        dropped_seq=-1, first_wall_bound_s=_number(
                            previous_fresh['pink_runtime_recovery']['command_sent_wall_s']))
                active['failed_samples'] += int(failure)
                active['raw_module_result_samples'] += int(raw_result)
                dropped = packet.get('dropped_seq')
                if type(dropped) is not int or dropped < 0:
                    reject('failed_module_sequence_missing')
                else:
                    if dropped <= active['pre_fault_seq']:
                        reject('dropped_result_sequence_does_not_follow_pre_fault_execution')
                    active['dropped_seq'] = max(active['dropped_seq'], dropped)
            if active is not None:
                if context != active['context']:
                    raise ValueError('new task cannot hide an unrecovered module failure')
                if row.get('current_solver_task_context') != context:
                    reject('module_recovery_hold_not_bound_to_original_task')
                if stamp-active['first_control_stamp_s'] > .18+1e-8:
                    reject('module_recovery_exceeds_original_simulation_budget')
                if fresh:
                    new = _fresh(row, context)
                    wall_end = (_number(new['command_sent_wall_s'])
                                + _number(row['upper_execution_lease']['execution_end_wall_age_s']))
                    wall_bound = wall_end-active['first_wall_bound_s']
                    if not 0 <= wall_bound <= .18+1e-8:
                        reject('module_recovery_wall_budget_unverified_from_conservative_bound')
                    if (new['seq'] <= max(active['dropped_seq'], active['pre_fault_seq'])
                            or _number(new['acquisition_stamp_s']) < active['first_control_stamp_s']-1e-8):
                        reject('recovery_resumed_a_cancelled_or_pre_fault_command')
                    episodes.append(dict(task_id=active['task_id'],
                        first_control_stamp_s=active['first_control_stamp_s'], resumed_stamp_s=stamp,
                        failed_samples=active['failed_samples'], resumed_seq=new['seq'],
                        raw_module_result_samples=active['raw_module_result_samples'],
                        conservative_wall_duration_s=wall_bound,
                        simulation_duration_s=stamp-active['first_control_stamp_s']))
                    active = None
                else:
                    if feedback.get('reason') == 'accepted' or row.get('pink_upper_velocity'):
                        reject('module_recovery_executed_nonfresh_motion')
                    guard = row.get('policy_velocity_guard') or {}
                    if (guard.get('upper_reference_held') is not True
                            or guard.get('upper_velocity_target_zero') is not True
                            or guard.get('integral_frozen') is not True
                            or guard.get('collision_veto') is not False):
                        reject('module_recovery_hold_or_brake_evidence_missing')
                    names = row['joint_names']
                    ids = [names.index(name) for name in UPPER]
                    if names != previous['joint_names']:
                        raise ValueError('joint order changed during module recovery')
                    actual_target = np.array([_number(x) for x in row['applied_joint_position_target']])
                    old_target = np.array([_number(x) for x in previous['applied_joint_position_target']])
                    if not np.array_equal(actual_target[ids], old_target[ids]):
                        reject('module_recovery_did_not_hold_committed_upper_reference')
                    velocity = np.array([_number(x) for x in row['applied_policy_velocity']])
                    old_velocity = np.array([_number(x) for x in previous['applied_policy_velocity']])
                    expected = np.clip(np.zeros(3), old_velocity-np.array([.5, .5, .8])*.02,
                                       old_velocity+np.array([.5, .5, .8])*.02)
                    if velocity.shape != (3,) or not np.allclose(velocity, expected, atol=1e-8, rtol=0):
                        reject('module_recovery_did_not_apply_original_slew_brake')
            if fresh:
                previous_fresh, previous_context = row, context
            previous = row
        if active is not None:
            reject('module_recovery_not_completed_with_fresh_command')
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        reject(str(error))
    return dict(schema='independent_g1_module_recovery_audit/v1',
        passed=not errors, audit_errors=errors, original_recovery_budget_s=.18,
        original_motion_lease_s=.06, reported_failure_samples=reported_failures,
        raw_module_result_samples=raw_module_results,
        recovered_failure_samples=sum(item['failed_samples'] for item in episodes),
        recovered_raw_module_result_samples=sum(item['raw_module_result_samples'] for item in episodes),
        episodes=episodes, actual_physical_safety_verified=False,
        scope='Module recovery only. Physical tracking, all-joint stopping, limits, '
              'collision, rendering and sensing require independent gates.')
