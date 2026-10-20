#!/usr/bin/env python3
"""Read-only complete-suite comparison, including inherited vertical bias."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def inspect(name):
    folder=ROOT/'Robot_Go2/results'/name
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['run_complete']
    rows=[json.loads(line) for line in (folder/'measurements.jsonl').read_text().splitlines()]
    cases={}
    for result in summary['cases']:
        selected=[r for r in rows if r['case']==result['case']]
        last=selected[-1];actual=np.asarray(last['T_world_camera_optical']);reference=np.asarray(last['T_world_reference_optical'])
        tail=[r for r in selected if r['stamp_s']>=last['stamp_s']-.5-1e-9]
        cases[result['case']]=dict(velocity_pass=result['velocity_tracking_pass'],camera_pass=result['camera_tracking_pass'],
            stop_pass=result['strict_stop']['passed'],joint_stop_span_max_rad=max(result['strict_stop']['joint_span']),
            final_base_linear_speed_mps=result['strict_stop']['final_base_linear_speed_mps'],
            final_base_angular_speed_radps=result['strict_stop']['final_base_angular_speed_radps'],
            camera_position_rmse_m=result['camera_position_rmse_m'],camera_orientation_rmse_rad=result['camera_orientation_rmse_rad'],
            final_camera_position_error_m=last['position_error_m'],final_camera_orientation_error_rad=last['orientation_error_rad'],
            original_final_camera_z_error_m=float(actual[2,3]-reference[2,3]),
            original_reference_end_z_m=float(reference[2,3]),actual_camera_end_z_m=float(actual[2,3]),
            tail_native7_mean=np.mean([r['actual_native_command']for r in tail],axis=0).tolist(),
            tail_policy_input_max_abs=np.max(abs(np.asarray([r['applied_policy_velocity']for r in tail])),axis=0).tolist(),
            task_infeasible_count=sum(r.get('task_feedback',{}).get('feasible') is False for r in selected))
    return dict(summary={k:summary[k]for k in ['all_pass','sim_seconds','wall_seconds','timing_pass','task_infeasible_samples','known_constraint_failure_free']},
        cases=cases,accumulated_per_task_final_camera_z_bias_m=sum(c['original_final_camera_z_error_m'] for c in cases.values()))


def main():
    previous=inspect('adaptive_suite_softpose_01');candidate=inspect('adaptive_suite_prediction_01')
    files=[Path(__file__)]
    for run in ['adaptive_suite_softpose_01','adaptive_suite_prediction_01']:
        files.extend(ROOT/'Robot_Go2/results'/run/path for path in
            ['summary.json','measurements.jsonl','timed_tasks.json','provenance.json','sources/adaptive_runner.py','sources/camera_coordinator.py'])
    report=dict(schema='go2_prediction_full_readonly_comparison/v1',previous=previous,candidate=candidate,
        scope='Raw measurements retained unchanged. Each accepted WORLD Task remains original. A subsequent new Task inherits the actual measured initial pose by the original authoring rule; summed endpoint Z errors describe accumulation, not a controller reference correction.',
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    output=Path(__file__).with_suffix('.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(candidate,indent=2));print(output)


if __name__=='__main__':main()
