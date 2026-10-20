#!/usr/bin/env python3
"""Read-only post-run Z-input and strict-stop comparison; original logs untouched."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_world_z_01'


def main():
    script=ROOT/'validation/go2_prediction_full_compare_husky_20261020.py'
    spec=importlib.util.spec_from_file_location('full_comparator',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    previous=module.inspect('adaptive_suite_prediction_01');candidate=module.inspect('adaptive_suite_world_z_01')
    rows=[json.loads(line) for line in (RUN/'measurements.jsonl').read_text().splitlines()]
    active=[r for r in rows if r['case']!='settle']
    reports=[r['actual_native_domain']['world_z_feedback']for r in active]
    per_task={}
    for row,report in zip(active,reports):
        case=per_task.setdefault(row['case'],dict(initial_integral_before_m=report['integral_before_m'],
            max_integral_abs_m=0.,cap_antiwindup_freeze_count=0,outer_rollback_count=0,invalid_clock_count=0))
        case['final_integral_m']=report['integral_after_m']
        case['max_integral_abs_m']=max(case['max_integral_abs_m'],abs(report['integral_after_m']))
        case['cap_antiwindup_freeze_count']+=report['cap_antiwindup_frozen']
        case['outer_rollback_count']+=report['externally_rolled_back']
        case['invalid_clock_count']+=not report['measurement_fresh_and_clock_valid']
    files=[Path(__file__),script]
    for run in ['adaptive_suite_prediction_01','adaptive_suite_world_z_01','adaptive_camera6d_world_z_01']:
        folder=ROOT/'Robot_Go2/results'/run
        files.extend(folder/name for name in ['summary.json','measurements.jsonl','provenance.json','timed_tasks.json'])
    files.extend(RUN/'sources'/name for name in ['adaptive_runner.py','camera_coordinator.py','world_z_feedback.py'])
    result=dict(schema='go2_world_z_full_readonly/v1',scope='Post hoc measured comparison only. Physical pass remains original strict gates; correlation of cumulative reference height and stop is not a causality certificate.',
        previous=previous,candidate=candidate,world_z_feedback=dict(samples=len(reports),per_task=per_task,
            maximum_position_correction_m=max(r['proposed_position_correction_norm_m']for r in reports),
            maximum_orientation_correction_rad=max(np.linalg.norm(r['proposed_orientation_correction_rad'])for r in reports),
            domain_rejected_correction_count=sum(r['domain_correction_rejected']for r in reports),
            original_reference_changed=False),
        files_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()for path in files})
    output=Path(__file__).with_suffix('.json');assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(candidate,indent=2));print('I',json.dumps(per_task,indent=2));print(output)


if __name__=='__main__':main()
