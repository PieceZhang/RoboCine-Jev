#!/usr/bin/env python3
"""Audit public feedback types, tolerance flags and G1 task-context binding.

Use alongside the physical audit, which checks measured errors, geometry,
constraints and task references. This audit cannot certify motion or avoidance.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from audit_wbc_physical import digest, finite


REASONS = frozenset(('not_evaluated', 'tracking_limit', 'self_collision',
    'environment_collision', 'contact_or_balance', 'out_of_policy_domain',
    'solver_infeasible', 'solver_failure', 'observation_stale', 'unsupported_request',
    'configuration_limit', 'actuation_limit', 'camera_view_blocked',
    'sensor_evidence_unknown', 'command_stale', 'deadline_exceeded', 'accepted'))
PARTS = frozenset(('mobile_platform', 'camera_manipulator'))


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def expected_context(task):
    return dict(schema='g1_solver_task_context/v1', task_id=task['task_id'],
                camera_path_id=fingerprint(task['camera_trajectory']),
                task_sha256=fingerprint(task))


def execution_lease_errors(report, fresh):
    """Independently check each executed 5ms interval and the recorded end."""
    errors = []
    if report is None:
        return ['missing_execution_lease_report'] if fresh else []
    if not isinstance(report, dict):
        return ['invalid_execution_lease_report']
    checks = report.get('checks', [])
    if not isinstance(checks, list) or not 1 <= len(checks) <= 4:
        return ['missing_physical_substep_lease_checks']
    invalid = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append('invalid_substep_lease_record')
            continue
        if type(check.get('valid')) is not bool:
            errors.append('invalid_substep_lease_flag')
        elif check['valid']:
            sim = float(finite(check.get('sim_age_s'), ()))
            wall = float(finite(check.get('wall_age_s'), ()))
            horizon = float(finite(check.get('execution_horizon_s'), ()))
            if (abs(horizon-.005) > 1e-8 or min(sim, wall) < -1e-8
                    or max(sim, wall)+horizon > .06+1e-8):
                errors.append('executed_substep_exceeds_original_lease')
        else:
            invalid.append(index)
    end = report.get('execution_end_check')
    end_valid = isinstance(end, dict) and end.get('valid') is True
    if invalid and (invalid != [len(checks)-1] or report.get('first_expired_substep') != invalid[0]):
        errors.append('execution_continued_after_expired_lease')
    if fresh and (len(checks) != 4 or invalid or not end_valid or report.get('expired') is not False):
        errors.append('fresh_command_without_complete_execution_lease')
    if fresh:
        for key in ('execution_end_wall_age_s', 'execution_end_sim_age_s'):
            age = float(finite(report.get(key), ()))
            if not 0 <= age <= .06+1e-8:
                errors.append('execution_ended_outside_original_lease')
    if invalid and report.get('expired') is not True:
        errors.append('expired_execution_not_reported')
    return errors


def audit_feedback(folder):
    folder = Path(folder).resolve()
    errors, reasons, tolerance, hashes = [], Counter(), Counter(), {}
    feedback_count = context_count = 0
    max_executed_command_age = 0.

    def read(name):
        hashes[name] = digest(folder/name)
        return json.loads((folder/name).read_text())

    def reject(error):
        if error not in errors:
            errors.append(error)

    try:
        summary = read('summary.json' if (folder/'summary.json').is_file() else 'acceptance.json')
        originals = read('timed_tasks.json' if (folder/'timed_tasks.json').is_file() else 'public_task.json')
        originals = originals if isinstance(originals, list) else [originals]
        tasks = {t['task_id']: t for t in originals}
        public = {}
        if (folder/'public_feedback.jsonl').is_file():
            hashes['public_feedback.jsonl'] = digest(folder/'public_feedback.jsonl')
            for line in (folder/'public_feedback.jsonl').read_text().splitlines():
                item = json.loads(line);item = item.get('feedback', item)
                public[round(float(item['stamp_s']), 8)] = item
        name = next(n for n in ('measurements.jsonl', 'samples.jsonl', 'measured.jsonl') if (folder/n).is_file())
        hashes[name] = digest(folder/name)
        strict_context = summary.get('robot') == 'g1' and (folder/'sources/solver_context.py').is_file()
        for line in (folder/name).read_text().splitlines():
            row = json.loads(line)
            stamp = float(finite(row.get('stamp_s', row.get('t_s')), ()))
            feedback = row.get('task_feedback', public.get(round(stamp, 8)))
            if feedback is None:
                continue
            feedback_count += 1
            task = tasks[feedback['task_id']]
            feasible = feedback.get('feasible')
            if feasible is not None and type(feasible) is not bool:
                reject('feasible_must_be_boolean_or_null')
            reason = feedback.get('reason')
            if reason not in REASONS:
                reject('invalid_public_reason')
            reasons[str(reason)] += 1
            if feedback.get('safety_certificate') is not False or feedback.get('evidence_kind') != 'measured':
                reject('public_feedback_claims_unverified_certificate')
            position = float(finite(feedback['position_error_m'], ()))
            angle = float(finite(feedback['orientation_error_rad'], ()))
            within = position <= task['position_tolerance_m'] and angle <= task['orientation_tolerance_rad']
            if position < 0 or angle < 0 or feedback.get('camera_within_tolerance') is not within:
                reject('public_tolerance_flag_inconsistent')
            tolerance['within' if within else 'outside'] += 1
            remaining = feedback.get('remaining_adjustment')
            if not isinstance(remaining, dict) or set(remaining) != PARTS:
                reject('missing_two_part_remaining_adjustment')
            else:
                for value in remaining.values():
                    if value is not None and not 0 <= float(finite(value, ())) <= 1:
                        reject('invalid_remaining_adjustment_fraction')
            if summary.get('robot') == 'g1':
                fresh = row.get('controller_status') == 'fresh_command'
                if not fresh and reason == 'accepted':
                    reject('missing_fresh_solver_command_reported_accepted')
                if strict_context:
                    context = expected_context(task)
                    if row.get('current_solver_task_context') != context:
                        reject('current_context_differs_from_original_task')
                    packet = row.get('pink_runtime_recovery', {})
                    if fresh:
                        context_count += 1
                        if packet.get('task_context_status') != 'same_task' or packet.get('task_context') != context:
                            reject('executed_reply_not_bound_to_current_task')
                        seq = packet.get('seq')
                        if type(seq) is not int or seq < 0:
                            reject('executed_reply_missing_sequence')
                        age = stamp-float(finite(packet.get('acquisition_stamp_s'), ()))
                        max_executed_command_age = max(max_executed_command_age, age)
                        if not 0 <= age <= .06+1e-8:
                            reject('executed_reply_outside_60ms_measured_lease')
                    elif row.get('pink_upper_velocity'):
                        reject('nonfresh_reply_executed_upper_velocity')
                    if 'upper_execution_lease' in row:
                        for error in execution_lease_errors(row['upper_execution_lease'], fresh):
                            reject(error)
        if not feedback_count:
            reject('no_physical_task_feedback')
    except (ValueError, TypeError, KeyError, OSError, StopIteration, AttributeError) as error:
        reject(str(error))
    return dict(schema='independent_wbc_feedback_audit/v1', folder=str(folder),
        passed=bool(feedback_count and not errors), audit_errors=errors,
        feedback_samples=feedback_count, reason_counts=dict(reasons),
        camera_tolerance_counts=dict(tolerance), fresh_context_bound_samples=context_count,
        maximum_executed_command_age_s=max_executed_command_age,
        sources_sha256=dict(hashes, auditor=digest(Path(__file__)),
            physical_auditor=digest(Path(__file__).with_name('audit_wbc_physical.py'))),
        scope='Public feedback consistency and actual G1 task-context provenance; run physical/suite/limits/render audits separately. No environment or sensor certification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_feedback(args.folder)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload, end='')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
