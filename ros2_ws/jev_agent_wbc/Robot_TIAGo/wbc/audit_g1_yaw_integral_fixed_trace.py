#!/usr/bin/env python3
"""Moving-yaw input-layer counterfactual; recorded plant/safety schedule frozen.

No new physical response is predicted. Changed candidate inputs are retained in
the candidate's own slew/antiwindup state, never overwritten by old applied u.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();begin=time.perf_counter()
    a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'sources';frozen.mkdir()
    for source in [a.run/'sources/velocity_servo.py',a.run/'sources/adaptive_runner.py',
                   a.run/'velocity_servo_config.json',Path(__file__)]:shutil.copy2(source,frozen/source.name)
    spec=importlib.util.spec_from_file_location('original_frozen_velocity_servo',frozen/'velocity_servo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((frozen/'velocity_servo_config.json').read_text())
    groups={};previous=None
    with(a.run/'measurements.jsonl').open()as stream:
        for line in stream:
            row=json.loads(line)
            if row['case'].startswith('yaw_')and row['phase']=='move':
                groups.setdefault(row['case'],[]).append((row,previous))
            previous=row
    all_results={};baseline_error=0.;baseline_post_error=0.
    for case,samples in groups.items():
        variants={}
        for ki in [1.,2.,3.,4.]:
            servo=module.VelocityServo(limits=config['limits'],**config['parameters'])
            previous_applied=np.asarray(samples[0][1]['applied_policy_velocity'])
            trace=[]
            for index,(row,prior)in enumerate(samples):
                gains=row['pink_velocity_feedforward'];servo.feedforward_gain=np.asarray(gains['actual_positive_gain'])
                servo.negative_feedforward_gain=np.asarray(gains['actual_negative_gain'])
                servo.kp=np.asarray(gains['actual_kp']);servo.ki=np.asarray(gains['actual_ki']);servo.ki[2]=ki
                requested=servo.update(row['requested_body_velocity'],prior['measured_body_velocity'],.02)
                pre_integral=servo.integral.copy();applied=requested.copy();guard=row.get('policy_velocity_guard')
                if guard:
                    assert guard['collision_veto']is False,'Unmodelled collision override, not a PI comparison'
                    applied=np.clip(np.zeros(3),previous_applied-servo.slew*.02,previous_applied+servo.slew*.02)
                    servo.applied(applied,transient_hold=guard['integral_frozen'])
                if ki==1:
                    baseline_error=max(baseline_error,float(np.max(abs(requested-row['requested_policy_velocity']))))
                    baseline_post_error=max(baseline_post_error,float(np.max(abs(applied-row['applied_policy_velocity']))))
                trace.append(dict(stamp_s=row['stamp_s'],requested_yaw=float(requested[2]),
                    applied_yaw=float(applied[2]),integral_yaw=float(servo.integral[2]),
                    filtered_yaw=float(servo.filtered[2]),recorded_measured_yaw=float(row['measured_body_velocity'][2]),
                    held_integral=guard is not None,
                    saturated_I=bool(abs(servo.integral[2])>=servo.integral_limit[2]-1e-12),
                    applied_slew_margin=float(servo.slew[2]*.02-abs(applied[2]-previous_applied[2]))))
                previous_applied=applied.copy()
            values=np.asarray([r['applied_yaw']for r in trace]);I=np.asarray([r['integral_yaw']for r in trace])
            end_I=I[-1];end_u=values[-1]
            servo.reset()  # Exact original source: declared stop clears all PI state.
            variants[str(int(ki))]=dict(trace=trace,mean_applied_yaw=float(values.mean()),
                tail_half_mean_applied_yaw=float(values[125:].mean()),end_applied_yaw=float(end_u),
                end_integral_yaw=float(end_I),maximum_abs_integral=float(max(abs(I))),
                integral_cap_samples=sum(r['saturated_I']for r in trace),maximum_abs_applied_yaw=float(max(abs(values))),
                minimum_slew_margin=float(min(r['applied_slew_margin']for r in trace)),
                maximum_cap_violation=float(max(0.,max(abs(values))-1.)),
                minimum_literal_zero_brake_time_s=math.ceil((abs(end_u)-1e-12)/(.8*.02))*.02,
                declared_stop_integral_after_original_reset=float(servo.integral[2]),
                each_second_mean_applied_yaw=[float(values[s*50:(s+1)*50].mean())for s in range(5)])
        baseline=np.asarray([r['applied_yaw']for r in variants['1']['trace']])
        for key,value in variants.items():
            delta=np.asarray([r['applied_yaw']for r in value['trace']])-baseline
            value.update(maximum_abs_input_change=float(max(abs(delta))),
                signed_extra_input_integral_rad=float(delta.sum()*.02))
        all_results[case]=variants
    source=(frozen/'adaptive_runner.py').read_text()
    reset_order=(source.index('stop_anchor=bodies[base_name].copy();servo.reset()')
                 <source.index('policy_command=servo.update(desired_velocity,measured,dt,pose_error_body=anchor_error)'))
    report=dict(schema='g1_moving_yaw_ki_fixed_measurement_trace/v1',physical_execution=False,
        plant_response_predicted=False,original_task_modified=False,
        baseline_pre_guard_error=baseline_error,baseline_applied_error=baseline_post_error,
        baseline_exact=baseline_error<1e-12 and baseline_post_error<1e-12,
        cases=all_results,stop_reset_before_update_source_verified=reset_order,
        frozen_sha256={p.name:sha(p)for p in frozen.iterdir()if p.is_file()},
        input_sha256={'measurements.jsonl':sha(a.run/'measurements.jsonl')},
        scope='Only the moving-yaw PI input map on the identical saved measurements and module-hold schedule. Candidate commands keep their own previous input/I. No causal gain fitting, no predicted yaw tracking or stopping. Stop reset removes moving I; larger final u still needs physical slew braking.',
        audit_wall_seconds=time.perf_counter()-begin)
    assert report['baseline_exact']and reset_order
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='cases'},indent=2))
    for case,variants in all_results.items():
        print(case,json.dumps({k:{n:v[n]for n in ('tail_half_mean_applied_yaw','end_applied_yaw',
            'maximum_abs_integral','integral_cap_samples','minimum_slew_margin',
            'minimum_literal_zero_brake_time_s','signed_extra_input_integral_rad')}for k,v in variants.items()}))


if __name__=='__main__':main()
