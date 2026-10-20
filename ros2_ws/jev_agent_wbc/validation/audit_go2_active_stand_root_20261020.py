"""Independent measured-input audit for Go2's optional lateral stand brake.

This profile deliberately uses feedback during the original final three seconds.
The historical literal-zero-input auditor remains unchanged. Physical parking,
camera tracking, collision, policy stability and timing require separate audits.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

DT = .02
CAPS = np.array([.2, .2, .2])
SLEW = np.array([.5, .5, .8])


def same(a, b, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    return (a.shape == b.shape and a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu'
            and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=tol, rtol=0))


def audit_rows(rows, tasks, complete):
    errors, results = [], []
    if complete is not True:
        errors.append('physical_run_incomplete')
    if len(rows) < 2 or not same(np.diff([r['stamp_s'] for r in rows]),
                                 np.full(max(0, len(rows)-1), DT)):
        errors.append('original_50Hz_interval_missing')
    for task in tasks:
        path = task['camera_trajectory']
        end = path['start_s']+path['samples'][-1]['t_s']
        indices = [i for i, r in enumerate(rows) if r.get('phase') == 'stop'
                   and r.get('task_feedback', {}).get('task_id') == task['task_id']]
        failures, active = [], 0
        if (len(indices) != 250 or not indices or indices[0] == 0
                or not same(rows[indices[0]]['stamp_s']-DT, end)
                or not same(rows[indices[-1]]['stamp_s'], end+5.)
                or not same(np.diff(indices), np.ones(max(0, len(indices)-1)))
                or not same(np.diff([rows[i]['stamp_s'] for i in indices]),
                            np.full(max(0, len(indices)-1), DT))):
            failures.append('original_complete_five_second_stop_missing')
        prior_deadline_failure = False
        for i in indices:
            row, previous = rows[i], rows[i-1]
            now = row['stamp_s']-DT
            issued = np.asarray(row['applied_policy_velocity'], float)
            prior = np.asarray(previous['applied_policy_velocity'], float)
            report = row.get('terminal_allocation') or {}
            prior_deadline_failure |= report.get('deadline_failure_latched') is True
            if (issued.shape != (3,) or not np.isfinite(issued).all()
                    or np.any(abs(issued) > CAPS+1e-9)
                    or np.any(abs(issued-prior) > SLEW*DT+1e-9)):
                failures.append('original_input_caps_or_slew_changed')
            if not same(row['requested_body_velocity'], np.zeros(3)):
                failures.append('original_requested_stop_changed')
            if (report.get('original_world_task_modified') is not False
                    or report.get('all18_learned_actions_unchanged') is not True
                    or report.get('reference_mode') != 'explicit_original_endpoint_hold'
                    or not same(report.get('stamp_s'), now)
                    or not same(report.get('reference_stamp_s'), end)
                    or not same(report.get('final_body_input'), issued)):
                failures.append('terminal_context_or_actual_input_mismatch')
            if row.get('native_standing_mode') is not bool(np.all(issued == 0)):
                failures.append('literal_zero_label_does_not_match_actual_input')
            horizon = min(1.2, max(0., end+5.-now))
            expected = np.unique(np.round(np.r_[np.linspace(0., min(horizon, .8), 5),
                np.linspace(min(horizon, .8), horizon, 3)], 8))
            support = row.get('camera_support') or {}
            actual = np.unique(np.round(support.get('prediction_intervals_s', []), 8))
            if (support.get('planned_slew_brake_model') is not True
                    or not same(support.get('prediction_horizon_s'), horizon)
                    or not same(actual, expected, 1e-8)):
                failures.append('original_forecast_horizon_or_nodes_removed')
            if now < end+2.-1e-8:
                if not same(report.get('zero_input_deadline_s'), end+2.):
                    failures.append('original_initial_allocation_deadline_changed')
                continue
            active += 1
            measured = np.asarray(previous['measured_body_velocity'], float)
            raw = np.array([0., -.5*measured[1], 0.])
            target = np.clip(raw, [0., -.03, 0.], [0., .03, 0.])
            first = np.clip(target, np.maximum(-CAPS, prior-SLEW*DT),
                            np.minimum(CAPS, prior+SLEW*DT))
            if (report.get('profile') != 'vy_p05_v1'
                    or report.get('phase') != 'active_lateral_brake'
                    or not same(report.get('measurement_stamp_s'), now)
                    or not same(previous['stamp_s'], now)
                    or not same(report.get('measured_body_velocity'), measured)
                    or not same(report.get('previous_actual_input'), prior)
                    or not same(report.get('feedback_gain'), .5)
                    or not same(report.get('assist_vy_cap_mps'), .03)
                    or not same(report.get('requested_stop_velocity'), np.zeros(3))
                    or not same(report.get('raw_feedback_input'), raw)
                    or not same(report.get('bounded_feedback_target'), target)
                    or not same(report.get('cap_slew_admissible_first'), first)):
                failures.append('fresh_feedback_formula_or_source_mismatch')
            if (support.get('locked_input') is not True
                    or not report.get('future_input_assumption', '').startswith('Frozen current measured feedback target')
                    or report.get('prediction_model', {}).get('profile') != 'bounded_ramp_history_v2'):
                failures.append('feedback_input_not_used_by_original_reserved_forecast')
            vetoes = report.get('veto_reasons')
            if not isinstance(vetoes, list):
                failures.append('missing_explicit_final_veto_status')
                continue
            collision = row.get('fine_collision_evidence') or {}
            collision_stamp = collision.get('stamp_s')
            collision_fresh = (isinstance(collision_stamp, (int, float))
                and not isinstance(collision_stamp, bool) and np.isfinite(collision_stamp)
                and -1e-9 <= now-collision_stamp <= .06+1e-9)
            zi = row.get('actual_native_domain', {}).get('world_z_feedback') or {}
            required_veto = (support.get('feasible') is False
                or collision.get('status') != 'pass' or not collision_fresh
                or row.get('native_domain_recovery', {}).get('active') is True
                or prior_deadline_failure)
            if required_veto and not vetoes:
                failures.append('raw_safety_veto_missing_from_final_routing')
            expected_issued = (np.clip(np.zeros(3), prior-SLEW*DT, prior+SLEW*DT)
                               if vetoes or required_veto else first)
            if (not same(issued, expected_issued)
                    or report.get('final_feedback_only') is not (not bool(vetoes))):
                failures.append('actual_feedback_or_safe_brake_mismatch')
            if (vetoes or required_veto) and row['task_feedback'].get('feasible') is not False:
                failures.append('veto_not_reported_infeasible')
            if vetoes or required_veto:
                if (zi.get('integral_updated') is not False
                        or not same(zi.get('integral_after_m'), zi.get('integral_before_m'))):
                    failures.append('veto_did_not_freeze_or_restore_integral')
                if zi.get('measurement_fresh_and_clock_valid') is True and (
                        zi.get('externally_rolled_back') is not True
                        or zi.get('native_correction_accepted') is not False):
                    failures.append('fresh_same_tick_veto_missing_native_feedback_rollback')
            if abs(issued[0]) > 1e-12 or abs(issued[2]) > 1e-12 or abs(issued[1]) > .03+1e-9:
                failures.append('late_feedback_not_lateral_only_or_outside_assist_cap')
        if active != 150:
            failures.append('original_final_three_second_interval_missing')
        failures = sorted(set(failures))
        results.append(dict(task_id=task['task_id'], stop_samples=len(indices),
                            active_feedback_samples=active, errors=failures, passed=not failures))
        errors.extend(task['task_id']+':'+e for e in failures)
    return dict(schema='independent_go2_active_stand_input_audit/v1',
                passed=bool(results and not errors), errors=errors, tasks=results,
                scope='Fresh feedback formula and actual issued inputs within unchanged limits and original five-second stop; '
                      'physical all-joint parking, tracking, safety, timing and sensing require separate evidence.')


def audit(folder):
    folder = Path(folder).resolve()
    paths = [folder/name for name in ('timed_tasks.json', 'measurements.jsonl', 'summary.json', 'provenance.json')]
    try:
        tasks, rows, summary = (json.loads(paths[0].read_text()),
            [json.loads(line) for line in paths[1].read_text().splitlines()], json.loads(paths[2].read_text()))
        result = audit_rows(rows, tasks, summary.get('run_complete'))
        args = json.loads(paths[3].read_text())['arguments']
        if args.get('go2_active_stand_brake') != 'vy_p05_v1' or args.get('go2_terminal_native_hold') != 'none':
            result['errors'].append('candidate_profile_or_no_hold_configuration_mismatch')
            result['passed'] = False
    except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
        result = dict(passed=False, errors=[str(error)])
    result['sources_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    result['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=result['passed'], errors=result['errors'])))
    raise SystemExit(0 if result['passed'] else 2)
