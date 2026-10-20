#!/usr/bin/env python3
"""Reconstruct moving-yaw PI state from frozen source and prior measured rows.

This is an offline signal audit, not a replacement physical trajectory. The
reported state is useful only when every reconstructed pre-guard command agrees
with its logged value. It does not fit a plant or infer a causal transfer gain.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics

import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb')as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def audit(result):
    source=result/'sources/velocity_servo.py'
    spec=importlib.util.spec_from_file_location('actual_frozen_velocity_servo',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((result/'velocity_servo_config.json').read_text())
    groups={};controller=None;last=None;active_case=None;maximum_difference=0.
    selected_names=('waist_yaw_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint',
        'right_shoulder_yaw_joint','right_elbow_joint','right_wrist_pitch_joint')
    for line in (result/'measurements.jsonl').open():
        row=json.loads(line)
        if row['case'].startswith('yaw_')and row['phase']=='move':
            if row['case']!=active_case:
                controller=module.VelocityServo(limits=config['limits'],**config['parameters'])
                active_case=row['case'];groups[active_case]=dict(samples=[],initial={
                    'stamp_s':last['stamp_s'],'height_m':last['base_height_m'],
                    'upper_joints_rad':{n:last['joint_positions'][last['joint_names'].index(n)]for n in selected_names}})
            parameters=row['pink_velocity_feedforward']
            controller.feedforward_gain=np.array(parameters['actual_positive_gain'])
            controller.negative_feedforward_gain=np.array(parameters['actual_negative_gain'])
            controller.kp=np.array(parameters['actual_kp']);controller.ki=np.array(parameters['actual_ki'])
            predicted=controller.update(row['requested_body_velocity'],last['measured_body_velocity'],.02)
            discrepancy=float(np.max(abs(predicted-row['requested_policy_velocity'])))
            maximum_difference=max(maximum_difference,discrepancy)
            guard=row.get('policy_velocity_guard')
            if guard:
                controller.applied(row['applied_policy_velocity'],transient_hold=guard['integral_frozen'])
            groups[active_case]['samples'].append(dict(stamp_s=row['stamp_s'],
                applied_u=row['applied_policy_velocity'][2],measured_wz=row['measured_body_velocity'][2],
                reconstructed_integral=float(controller.integral[2]),reconstructed_filter=float(controller.filtered[2]),
                pre_guard_command_difference=discrepancy,fresh=row['controller_status']=='fresh_command',
                height_m=row['base_height_m'],upper_joints_rad={n:row['joint_positions'][row['joint_names'].index(n)]for n in selected_names}))
        last=row
    reports={}
    for case,group in groups.items():
        rows=group['samples'];tail=rows[round(len(rows)/2):]
        reports[case]=dict(initial=group['initial'],moving_samples=len(rows),
            tail_mean={k:statistics.mean(r[k]for r in tail)for k in (
                'applied_u','measured_wz','reconstructed_integral','reconstructed_filter','height_m')},
            end=rows[-1],
            each_second=[dict(second=second,
                **{k:statistics.mean(r[k]for r in rows[second*50:(second+1)*50])for k in (
                    'applied_u','measured_wz','reconstructed_integral','reconstructed_filter','height_m')},
                fresh_samples=sum(r['fresh']for r in rows[second*50:(second+1)*50]))for second in range(5)])
    return dict(replay_maximum_pre_guard_command_difference=maximum_difference,
        replay_consistent_with_logged_commands=maximum_difference<1e-10,cases=reports,
        inputs_sha256={str(p.relative_to(result)):sha(p)for p in (
            result/'measurements.jsonl',result/'summary.json',result/'velocity_servo_config.json',source)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='g1_frozen_moving_yaw_servo_replay/v1',physical_execution=False,
        interpretation='Reconstructed controller state, accepted only if pre-guard commands reproduce every logged moving sample; no fitted physical model or causal certificate.',
        results={str(p):audit(p)for p in args.result},source_sha256=sha(Path(__file__)))
    shutil.copy2(__file__,args.output/'analysis_source.py')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,data in report['results'].items():
        print(name,'replay_error',data['replay_maximum_pre_guard_command_difference'])
        for case,c in data['cases'].items():print(case,json.dumps(c['tail_mean']))


if __name__=='__main__':main()
