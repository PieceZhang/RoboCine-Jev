#!/usr/bin/env python3
"""Read-only input-servo reconstruction and measured parking decomposition."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np


def module(path):
    spec=importlib.util.spec_from_file_location('source_servo',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def audit(directory):
    rows=[json.loads(s) for s in (directory/'measurements.jsonl').open()]
    frozen=module(directory/'sources/velocity_servo.py')
    config=json.loads((directory/'velocity_servo_config.json').read_text())
    reports={}
    for case in ('yaw_left','yaw_right'):
        indices=[i for i,r in enumerate(rows) if r['case']==case and r['phase']=='move']
        old=frozen.VelocityServo(limits=config['limits'],**config['parameters'])
        candidate=frozen.VelocityServo(limits=config['limits'],**config['parameters'])
        trace=[];candidate_previous=np.zeros(3)
        for i in indices:
            row=rows[i];desired=np.asarray(row['requested_body_velocity']);measured=rows[i-1]['measured_body_velocity']
            recorded_schedule=row.get('pink_velocity_feedforward',{})
            if recorded_schedule:
                old.feedforward_gain=np.asarray(recorded_schedule['actual_positive_gain'])
                old.negative_feedforward_gain=np.asarray(recorded_schedule['actual_negative_gain'])
            candidate.feedforward_gain=np.array([1.,1.,1.])
            candidate.negative_feedforward_gain=np.array([1.,1.,1.4])
            prior=old.integral.copy();command=old.update(desired,measured,.02)
            proposed=candidate.update(desired,measured,.02)
            error=desired-old.filtered
            integrated=np.clip(prior+old.ki*error*.02,-old.integral_limit,old.integral_limit)
            raw=desired*np.where(desired<0,old.negative_feedforward_gain,old.feedforward_gain)+old.kp*error+integrated
            if row.get('policy_velocity_guard') is not None:
                actual=np.asarray(row['applied_policy_velocity'])
                old.applied(actual,transient_hold=not row['policy_velocity_guard'].get('collision_veto',False))
                proposed=np.clip(0.,candidate_previous-candidate.slew*.02,candidate_previous+candidate.slew*.02)
                candidate.applied(proposed,transient_hold=True)
            candidate_previous=proposed
            trace.append(dict(stamp_s=row['stamp_s'],native_actual=row['applied_policy_velocity'],
                measured_velocity=row['measured_body_velocity'],filtered_pre_step_velocity=old.filtered.tolist(),
                integral_input=old.integral.tolist(),filtered_tracking_error=error.tolist(),
                original_feedforward_term=(desired*np.where(desired<0,old.negative_feedforward_gain,old.feedforward_gain)).tolist(),
                original_PI_proposal=command.tolist(),original_unbounded_proposal=raw.tolist(),
                proposal_reconstruction_max_abs_error=float(np.max(np.abs(command-np.asarray(row['requested_policy_velocity'])))),
                integral_update_rejected=(np.abs(integrated-old.integral)>1e-12).tolist(),
                candidate_feedforward_1_1p4_recorded_measurement_only=proposed.tolist(),
                native_guard_active=row.get('policy_velocity_guard') is not None))
        windows=[]
        for start in range(0,len(trace),50):
            block=trace[start:start+50]
            windows.append(dict(offset_s=start*.02,n=len(block),
                actual_yaw_rad_s=float(np.mean([r['measured_velocity'][2] for r in block])),
                native_yaw_input_rad_s=float(np.mean([r['native_actual'][2] for r in block])),
                integral_mean_input_rad_s=float(np.mean([r['integral_input'][2] for r in block])),
                filtered_error_mean_rad_s=float(np.mean([r['filtered_tracking_error'][2] for r in block])),
                integral_update_rejected_ticks=sum(r['integral_update_rejected'][2] for r in block),
                missing_fresh_ticks=sum(r['native_guard_active'] for r in block)))
        reports[case]=dict(full_original_moving_interval_s=[rows[indices[0]]['stamp_s']-.02,rows[indices[-1]]['stamp_s']],
            samples=len(trace),one_second_windows=windows,
            original_proposal_reconstruction_max_abs_error=max(r['proposal_reconstruction_max_abs_error'] for r in trace),
            terminal_integral_input=old.integral.tolist(),trace=trace)
    parking={}
    for case in dict.fromkeys(r['case'] for r in rows if r['case']!='settle'):
        block=[r for r in rows if r['case']==case]
        tail=[r for r in block if r['stamp_s']>=block[-1]['stamp_s']-.5-1e-9]
        angular=np.asarray([r['base_angular_velocity_world'] for r in tail]);body=np.asarray([r['measured_body_velocity'] for r in tail])
        q=np.asarray([r['joint_positions'] for r in tail]);span=np.ptp(q,axis=0);rank=np.argsort(span)[::-1]
        transitions=[dict(stamp_s=r['stamp_s'],**r['parking_transition']) for r in block
            if r.get('parking_transition') and (r['parking_transition']['entered'] or r['parking_transition']['exited'])]
        parking[case]=dict(tail_interval_s=[tail[0]['stamp_s'],tail[-1]['stamp_s']],samples=len(tail),
            native_input_min=np.min([r['applied_policy_velocity'] for r in tail],axis=0).tolist(),
            native_input_max=np.max([r['applied_policy_velocity'] for r in tail],axis=0).tolist(),
            literal_zero_samples=int(sum(np.max(np.abs(r['applied_policy_velocity']))<=1e-12 for r in tail)),
            PARK_samples=sum(r['native_standing_mode'] for r in tail),
            angular_world_mean=angular.mean(axis=0).tolist(),angular_world_min=angular.min(axis=0).tolist(),
            angular_world_max=angular.max(axis=0).tolist(),final_angular_world=angular[-1].tolist(),
            final_angular_norm_rad_s=float(np.linalg.norm(angular[-1])),
            final_measured_body_vx_vy_wz=body[-1].tolist(),
            largest_joint_spans=[dict(joint=tail[0]['joint_names'][i],span_rad=float(span[i])) for i in rank[:6]],
            lower_damping_scales=sorted(set(r['native_lower_damping_scale'] for r in tail)),transitions=transitions)
    return dict(yaw=reports,parking=parking)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    output=dict(schema='g1_velocity_parking_source_reconstruction/v1',physical_execution=False,
        scope='Full original5s movement, no shifted scoring window. Source filter/integral/antiwindup reconstructed from preceding measurements.',
        counterfactual_limit='Candidate1/1.4 only replays recorded measurements and guard events; the closed-loop plant would change. It is not a predicted physical velocity or acceptance result.',
        candidate_input_caps_changed=False,shared_servo_source_changed=False,runs={},evidence_sha256={})
    for directory in args.runs:
        output['runs'][directory.name]=audit(directory)
        for p in (directory/'measurements.jsonl',directory/'sources/velocity_servo.py',directory/'velocity_servo_config.json'):
            output['evidence_sha256'][str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    output['evidence_sha256'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    shutil.copy2(Path(__file__),args.output/'executed_source.py')
    (args.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    for run,result in output['runs'].items():
        for case,report in result['yaw'].items():
            print(run,case,'PI reconstruction',report['original_proposal_reconstruction_max_abs_error'],
                'I_end',report['terminal_integral_input'][2])
            print(json.dumps(report['one_second_windows']))


if __name__=='__main__':main()
