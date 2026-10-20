#!/usr/bin/env python3
"""Qualify basic legged control separately from the unchanged precision gates.

The user reduced G1/Go2 scope on 2026-10-20 to basic stable control and interface
adaptation. This report does not rewrite native summaries or precision audits.
It is not an environment, contact-force, sensing or avoidance certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_wbc_physical import audit as physical_audit, finite
from audit_wbc_limits import audit_limits
from audit_wbc_feedback import audit_feedback


CASES = frozenset(('forward', 'backward', 'left', 'right', 'diagonal_left',
                   'diagonal_right', 'yaw_left', 'yaw_right', 'camera_shot'))
STOP = dict(base_position_span_m=.01, base_rotation_span_rad=.03,
            camera_position_span_m=.02, camera_rotation_span_rad=.05,
            all_revolute_joint_span_rad=.05, final_linear_speed_mps=.03,
            final_angular_speed_radps=.05)


def qualify(physical, limits, feedback, summary, originals, rows):
    errors, cases = [], []
    def reject(error):
        if error not in errors:
            errors.append(error)
    try:
        if summary.get('robot') not in ('g1', 'go2'):
            reject('basic_scope_is_only_g1_and_go2')
        if (physical.get('audit_passed') is not True or physical.get('run_complete') is not True
                or physical.get('timing_pass') is not True or physical.get('native_reported_safety_failure')
                or physical.get('constraint_failure_samples')):
            reject('physical_provenance_completion_pace_or_safety_failed')
        if limits.get('passed') is not True:
            reject('original_all_joint_configuration_or_velocity_limits_failed')
        if feedback.get('passed') is not True:
            reject('public_feedback_or_fresh_task_binding_failed')
        if summary.get('fine_collision', {}).get('all_active_samples_fresh_pass') is not True:
            reject('measured_fine_collision_not_fresh_and_passing')
        physical_tasks = {item['task_id']:item for item in physical['tasks']}
        names = set()
        module_failures = 0
        for row in rows:
            item = row.get('task_feedback') or {}
            if item.get('feasible') is False:
                if item.get('reason') not in ('solver_failure', 'solver_infeasible'):
                    reject('non_module_request_failure_cannot_be_basic_qualified')
                module_failures += 1
        for task in originals:
            result = physical_tasks[task['task_id']]
            path = task['camera_trajectory']
            start = float(finite(path['start_s'], ()))
            end = start+float(finite(path['samples'][-1]['t_s'], ()))
            active = [r for r in rows if (r.get('task_feedback') or {}).get('task_id') == task['task_id']
                      and start < r['stamp_s'] <= end+1e-8]
            if not active:
                raise ValueError('no actual moving measurements')
            name = active[0]['case']; names.add(name)
            if name not in CASES or any(row['case'] != name for row in active):
                raise ValueError('unknown or changed case identity')
            latter = active[len(active)//2:]
            requested = finite(latter[0]['requested_body_velocity'], (3,))
            if not any(abs(requested) > 1e-9) or any(
                    not np.array_equal(finite(r['requested_body_velocity'], (3,)), requested) for r in latter):
                raise ValueError('basic directional request changed or empty')
            velocity = np.mean([finite(r['measured_body_velocity'], (3,)) for r in latter], axis=0)
            # Basic functionality requires useful progress in every requested
            # direction. It deliberately makes no precise velocity-error claim.
            directional = bool(np.all(np.sign(requested[abs(requested)>1e-9])
                                      * velocity[abs(requested)>1e-9] >= .01))
            stop = result['strict_stop']
            stop_values = {key:float(finite(stop[key], ())) for key in STOP if key in stop}
            joint_span = finite(stop['joint_span'])
            coarse = bool(float(finite(stop['measured_window_s'], ())) >= .5-1e-8
                and len(joint_span) == (29 if summary['robot'] == 'g1' else 18)
                and np.all(joint_span <= STOP['all_revolute_joint_span_rad'])
                and all(0 <= stop_values[key] <= cap for key,cap in STOP.items() if key in stop_values)
                and len(stop_values) == 6)
            camera = result.get('camera_tracking_pass') is True
            passed = bool(directional and coarse and camera)
            if not passed:
                reject(name+':basic_direction_camera_or_stop_failed')
            cases.append(dict(case=name, task_id=task['task_id'], requested_vx_vy_wz=requested.tolist(),
                measured_mean_vx_vy_wz=velocity.tolist(), velocity_error=(velocity-requested).tolist(),
                directional_response_pass=directional, original_camera_tracking_pass=camera,
                original_precision_stop_pass=stop.get('passed') is True,
                basic_stop_pass=coarse, maximum_joint_span_rad=float(max(joint_span)),
                measured_stop=stop, passed=passed))
        if names != CASES or len(cases) != len(CASES):
            reject('complete_nine_case_directional_coverage_missing')
    except (ValueError, KeyError, TypeError, IndexError) as error:
        reject(str(error))
        module_failures = locals().get('module_failures', 0)
    return dict(schema='independent_wbc_basic_control_audit/v1',
        scope='basic_stable_control_and_interface_adaptation', basic_control_pass=bool(cases and not errors),
        audit_errors=errors, cases=cases, basic_stop_limits=STOP,
        minimum_requested_axis_progress_mps_or_radps=.01,
        original_camera_tolerances_unchanged=True, original_physical_limits_unchanged=True,
        original_precision_physical_pass=physical.get('physical_acceptance_pass') is True,
        original_precision_stop_pass_count=sum(c['original_precision_stop_pass'] for c in cases),
        transient_module_failure_samples=module_failures,
        runtime_recovery_wall_bound_certified=False,
        environment_constraints_verified=False, sensors_avoidance_verified=False,
        precision_velocity_tracking_qualified=False,
        interpretation='Basic means measured directional motion, original camera RMS/final tolerances, '
            'original safety/feedback gates and a bounded coarse stand. Precision velocity/parking '
            'and universal trajectory support are not qualified. Public transient failures remain failures.')


def audit(folder):
    folder = Path(folder).resolve()
    physical = physical_audit(folder)
    summary = json.loads((folder/'summary.json').read_text())
    limits = audit_limits(folder, summary['robot'])
    feedback = audit_feedback(folder)
    originals = json.loads((folder/'timed_tasks.json').read_text())
    rows = [json.loads(line) for line in (folder/'measurements.jsonl').read_text().splitlines()]
    result = qualify(physical, limits, feedback, summary, originals, rows)
    result['folder'] = str(folder)
    result['independent_component_results'] = dict(
        physical_structure_pass=physical.get('audit_passed'), original_precision_pass=physical.get('physical_acceptance_pass'),
        limits_pass=limits['passed'], feedback_pass=feedback['passed'])
    result['independent_component_audits'] = dict(physical=physical, limits=limits, feedback=feedback)
    paths = [folder/name for name in ('summary.json', 'provenance.json', 'timed_tasks.json', 'measurements.jsonl')]
    paths.extend(Path(__file__).with_name(name) for name in
        ('audit_wbc_basic.py', 'audit_wbc_physical.py', 'audit_wbc_limits.py', 'audit_wbc_feedback.py'))
    result['sources_sha256'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(basic_control_pass=result['basic_control_pass'], audit_errors=result['audit_errors'])))
    raise SystemExit(0 if result['basic_control_pass'] else 2)
