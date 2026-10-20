"""Read-only check of issued Go2 terminal inputs against original Task clocks.

This complements physical/suite/limits audits. It does not predict stability or
certify the learned policy, continuous collision clearance, or sensing.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    folder = Path(folder).resolve()
    tasks = json.loads((folder/'timed_tasks.json').read_text())
    rows = [json.loads(line) for line in (folder/'measurements.jsonl').read_text().splitlines()]
    summary = json.loads((folder/'summary.json').read_text())
    errors, results = [], []
    dt = float(np.median(np.diff([r['stamp_s'] for r in rows])))
    if not np.isclose(dt, .02, atol=1e-9):
        errors.append('original_50Hz_input_interval_missing')
    for task in tasks:
        path = task['camera_trajectory']
        end = path['start_s']+path['samples'][-1]['t_s']
        indexes = [i for i, r in enumerate(rows) if r.get('phase') == 'stop'
            and r.get('task_feedback', {}).get('task_id') == task['task_id']]
        group = [rows[i] for i in indexes]
        commands = np.asarray([r['applied_policy_velocity'] for r in group])
        issue = np.asarray([r['stamp_s']-dt for r in group])
        zero = issue >= end+2.-1e-8
        task_errors = []
        if not group or len(group) < 250 or not np.isclose(group[-1]['stamp_s'], end+5., atol=1e-8):
            task_errors.append('original_five_second_stop_missing')
        if commands.shape != (len(group), 3) or not np.isfinite(commands).all() or np.any(abs(commands) > .2+1e-9):
            task_errors.append('actual_input_outside_original_caps')
        slew = np.asarray([(rows[i]['applied_policy_velocity'][axis]-rows[i-1]['applied_policy_velocity'][axis])/dt
            for i in indexes for axis in range(3)]).reshape(-1, 3)
        if np.any(abs(slew) > np.array([.5, .5, .8])+1e-7):
            task_errors.append('actual_input_outside_original_slew')
        if np.count_nonzero(zero) < 150 or np.any(commands[zero] != 0.):
            task_errors.append('original_last_three_seconds_not_literal_zero')
        endpoint_hashes = set()
        for r, stamp, command in zip(group, issue, commands):
            diagnostic = r.get('terminal_allocation') or {}
            if (diagnostic.get('error') or diagnostic.get('deadline_failure_latched')
                    or diagnostic.get('original_world_task_modified') is not False
                    or diagnostic.get('all18_learned_actions_unchanged') is not True
                    or diagnostic.get('reference_mode') != 'explicit_original_endpoint_hold'
                    or abs(diagnostic.get('stamp_s', -1)-stamp) > 1e-8
                    or abs(diagnostic.get('reference_stamp_s', -1)-end) > 1e-8
                    or abs(diagnostic.get('zero_input_deadline_s', -1)-(end+2.)) > 1e-8):
                task_errors.append('terminal_context_or_original_clock_changed')
            if not np.array_equal(diagnostic.get('final_body_input'), command):
                task_errors.append('terminal_report_differs_from_applied_input')
            if r.get('native_standing_mode') is not bool(np.all(command == 0.)):
                task_errors.append('standing_label_differs_from_applied_input')
            endpoint_hashes.add(diagnostic.get('original_endpoint_optical_sha256'))
            support = r.get('camera_support', {})
            horizon = min(1.2, max(0., end+5.-stamp))
            expected = np.unique(np.r_[np.linspace(0., min(horizon, .8), 5),
                np.linspace(min(horizon, .8), horizon, 3)])
            # Equivalent floating-point expressions of the absolute clock can
            # create duplicate nodes a few ulps apart at the .8s boundary.
            expected = np.unique(np.round(expected, 8))
            actual = np.unique(np.round(support.get('prediction_intervals_s', []), 8))
            if (support.get('planned_slew_brake_model') is not True
                    or not np.isclose(support.get('prediction_horizon_s', -1), horizon, atol=1e-8)
                    or actual.shape != expected.shape
                    or not np.allclose(actual, expected, atol=1e-8)):
                task_errors.append('original_reserved_forecast_removed')
        if len(endpoint_hashes) != 1 or None in endpoint_hashes:
            task_errors.append('terminal_endpoint_hash_changed')
        results.append(dict(task_id=task['task_id'], original_end_s=end, stop_samples=len(group),
            literal_zero_samples=int(np.count_nonzero(zero)), errors=sorted(set(task_errors)),
            maximum_actual_input_slew=np.max(abs(slew), axis=0).tolist(),
            endpoint_hashes=list(endpoint_hashes), passed=not task_errors))
        errors.extend(task['task_id']+':'+value for value in set(task_errors))
    if summary.get('run_complete') is not True:
        errors.append('physical_run_incomplete')
    return dict(schema='independent_go2_terminal_input_audit/v1', passed=bool(results and not errors),
        errors=errors, tasks=results, sources_sha256={name:sha(folder/name) for name in
            ('timed_tasks.json', 'measurements.jsonl', 'summary.json')}, auditor_sha256=sha(Path(__file__)),
        scope='Issued inputs, original absolute endpoint clock, retained forecast schedule; physical and sensing evidence separate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 2)
