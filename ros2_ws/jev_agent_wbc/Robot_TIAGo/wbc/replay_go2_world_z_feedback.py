#!/usr/bin/env python3
"""Pace1 immutable Go2 trace check of the new private native-input helper.

Errors are reused from history. This is not a prediction of physical response,
convergence, changed learned actions, sensor coverage or strict-stop success.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = args.result.resolve()
    package = result.parents[2]
    frozen = result/'sources'
    sys.path.insert(0, str(package))
    from jev_agent_wbc.wbc_protocol import TimedTask
    helper = package/'Robot_Go2/world_z_feedback.py'
    executed = args.output/'world_z_feedback_frozen.py'
    shutil.copy2(helper, executed)
    feedback = load('executed_world_z_feedback', executed)
    backend = load('frozen_go2_feedback_backend', frozen/'go2_backend.py')
    backend.UPSTREAM_MDP = frozen/'go2_upstream'
    source = load('frozen_go2_feedback_p', frozen/'camera_coordinator.py')
    sampler = load('frozen_go2_feedback_sampler', frozen/'optical_sampler.py').OpticalSampler
    tasks = [TimedTask.from_dict(x) for x in json.loads((result/'timed_tasks.json').read_text())]
    samplers = {t.task_id: sampler(t) for t in tasks}
    domain = backend.PolicyDomain()
    rows = [json.loads(x) for x in (result/'measurements.jsonl').open()]
    rows = [r for r in rows if 'task_feedback' in r]
    extrinsic = np.array(json.loads((result/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    enabled = feedback.WorldZFeedback(enabled=True, ki_per_s=.2)
    disabled = feedback.WorldZFeedback(enabled=False)
    reference_error = disabled_error = max_norm = max_i = 0.
    domain_reject = fallback_outside = cap_frozen = invalid_time = 0
    case_results = {}
    started = time.monotonic()
    dt = .02
    with (args.output/'trace.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            delay = started+index*dt-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            task_id = row['task_feedback']['task_id']
            reference = np.asarray(row['T_world_reference_optical'])
            reference_error = max(reference_error, float(np.max(np.abs(reference-
                samplers[task_id].sample(min(row['stamp_s'], samplers[task_id].end_s))))))
            measured = np.asarray(row['T_world_camera_optical'])
            link = np.asarray(row['T_world_base']).copy()
            link[:3, 3] += link[:3, :3]@np.array([0., 0., .07])
            native, check = enabled.feedback_native_command(backend, reference, measured, link, extrinsic,
                task_id=task_id, stamp_s=row['stamp_s'], now_s=row['stamp_s'], dt_s=dt,
                fresh=True, sample_seq=index, domain=domain)
            original, _ = source.feedback_native_command(backend, reference, measured, link, extrinsic, domain=domain)
            p_only, _ = disabled.feedback_native_command(backend, reference, measured, link, extrinsic, domain=domain)
            disabled_error = max(disabled_error, float(np.max(np.abs(original-p_only))))
            report = check['world_z_feedback']
            domain_reject += report['domain_correction_rejected']
            fallback_outside += check['status'] != 'inside_outer_domain_unverified'
            cap_frozen += report['cap_antiwindup_frozen']
            invalid_time += not report['measurement_fresh_and_clock_valid']
            max_norm = max(max_norm, report['proposed_position_correction_norm_m'])
            max_i = max(max_i, abs(enabled.integral_m))
            item = dict(case=row['case'], stamp_s=row['stamp_s'], task_id=task_id,
                actual_original_world_z_error_m=float(reference[2, 3]-measured[2, 3]),
                native_command=native.tolist(), domain_status=check['status'], world_z_feedback=report)
            case_results.setdefault(row['case'], []).append(item)
            stream.write(json.dumps(item, allow_nan=False)+'\n')
            if index % 500 == 0:
                print(f'fixed trace {index}/{len(rows)} {row["case"]}', flush=True)
    delay = started+len(rows)*dt-time.monotonic()
    if delay > 0:
        time.sleep(delay)
    elapsed = time.monotonic()-started
    original_probe = package/'validation/go2_z_integral_trace_husky_20261020.json'
    baseline = json.loads(original_probe.read_text())['results']['0.2']
    cases = {case: dict(rows=len(trace), final_I_m=trace[-1]['world_z_feedback']['integral_after_m'])
             for case, trace in case_results.items()}
    husky_match = bool(max_norm <= .025+1e-14 and max_i == baseline['maximum_abs_integral_m']
        and cap_frozen == baseline['cap_antiwindup_frozen_updates']
        and all(cases[k]['final_I_m'] == v['final_I_m'] for k, v in baseline['cases'].items()))
    inputs = [Path(__file__).resolve(), helper, executed, result/'measurements.jsonl',
              result/'timed_tasks.json', result/'sensor_mounts.json', result/'provenance.json',
              frozen/'go2_backend.py', frozen/'camera_coordinator.py', frozen/'optical_sampler.py',
              original_probe, package/'validation/go2_z_integral_trace_husky_20261020.py']
    summary = dict(schema='go2_world_z_helper_fixed_trace/v1', scope=__doc__,
        rows=len(rows), pace=1, replay_sample_dt_s=dt,
        replay_scheduled_seconds=len(rows)*dt, replay_wall_seconds=elapsed,
        timing_pass=elapsed <= len(rows)*dt*1.05,
        original_reference_max_error=reference_error,
        disabled_native_max_error_vs_original_p=disabled_error,
        maximum_total_position_correction_norm_m=max_norm, maximum_abs_integral_m=max_i,
        correction_domain_rejects=domain_reject, applied_native_outside_domain=fallback_outside,
        cap_antiwindup_frozen_updates=cap_frozen, invalid_measurement_clock_updates=invalid_time,
        matches_previous_independent_ki02_trace=husky_match,
        original_task_modified=False, physical_validation=False, error_convergence_verified=False,
        learned18actions_executed=False, strict_stop_verified=False,
        cases=cases, evidence_files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    summary['passed'] = bool(len(rows) == 4500 and summary['timing_pass'] and reference_error < 1e-12
        and disabled_error == 0. and max_norm <= .025+1e-14 and max_i <= .025
        and not(domain_reject or fallback_outside or invalid_time) and husky_match)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2); stream.write('\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
