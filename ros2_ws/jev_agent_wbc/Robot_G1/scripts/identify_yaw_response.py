#!/usr/bin/env python3
"""Offline input/response audit; no policy execution or plant modification."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def response(parameters,delay_steps,rows,indices):
    gain,tau,bias=parameters;alpha=np.exp(-.02/tau)
    state=rows[indices[0]-1]['measured_body_velocity'][2];predicted=[]
    for i in indices:
        u=rows[max(0,i-delay_steps)]['applied_policy_velocity'][2]
        state=alpha*state+(1-alpha)*(gain*u+bias)
        predicted.append(state)
    return np.asarray(predicted)


def fit(rows,indices):
    measured=np.array([rows[i]['measured_body_velocity'][2] for i in indices])
    candidates=[]
    for delay in range(0,61,2):
        result=least_squares(lambda p:response(p,delay,rows,indices)-measured,
            [.3,.3,0.],bounds=([0.,.02,-.15],[3.,3.,.15]),max_nfev=150)
        candidates.append(dict(delay_s=delay*.02,gain=result.x[0],tau_s=result.x[1],bias_rad_s=result.x[2],
            freerun_rmse_rad_s=float(np.sqrt(np.mean(result.fun**2))),parameters=result.x.tolist()))
    candidates.sort(key=lambda x:x['freerun_rmse_rad_s'])
    return candidates


def audit(directory,live_servo):
    rows=[json.loads(line) for line in (directory/'measurements.jsonl').open()]
    frozen=module('frozen_yaw_servo_'+directory.name,directory/'sources/velocity_servo.py')
    reports={};indices_by_case={}
    for name in ('yaw_left','yaw_right'):
        indices=[i for i,r in enumerate(rows) if r['case']==name and r['phase']=='move']
        if not indices:continue
        indices_by_case[name]=indices
        old=frozen.VelocityServo();counter=live_servo.VelocityServo()
        errors=[];losses=[];trace=[]
        for i in indices:
            row=rows[i];previous=rows[i-1]['measured_body_velocity']
            command=old.update(row['requested_body_velocity'],previous,.02)
            candidate=counter.update(row['requested_body_velocity'],previous,.02)
            actual=np.asarray(row['applied_policy_velocity'])
            unavailable=row['controller_status']!='fresh_command'
            if unavailable:
                prior=old.integral.copy();old.applied(actual)
                # Match the executor's brake against its PREVIOUS actual input,
                # rather than against the newly proposed PI command.
                previous_counter=np.asarray(trace[-1]['bounded_hold_counterfactual_input'] if trace
                    else rows[i-1]['applied_policy_velocity'])
                brake=np.clip(np.zeros(3),previous_counter-counter.slew*.02,previous_counter+counter.slew*.02)
                counter.applied(brake,transient_hold=True)
                losses.append(dict(stamp_s=row['stamp_s'],status=row['controller_status'],
                    proposed_yaw=float(command[2]),actual_yaw=float(actual[2]),
                    integral_before=prior.tolist(),integral_after=old.integral.tolist(),
                    yaw_integral_loss=float(prior[2]-old.integral[2]),
                    hard_input_slew_excess=float(max(0.,abs(actual[2]-rows[i-1]['applied_policy_velocity'][2])-.8*.02))))
            else:
                errors.append(float(np.max(np.abs(command-actual))))
                brake=candidate
            trace.append(dict(stamp_s=row['stamp_s'],actual_input=actual.tolist(),
                bounded_hold_counterfactual_input=brake.tolist(),measured_body_velocity=row['measured_body_velocity']))
        sec=[]
        for start in range(0,len(indices),50):
            group=[rows[i] for i in indices[start:start+50]]
            sec.append(dict(offset_s=start*.02,n=len(group),
                mean_input=np.mean([r['applied_policy_velocity'] for r in group],axis=0).tolist(),
                mean_actual=np.mean([r['measured_body_velocity'] for r in group],axis=0).tolist(),
                unavailable_count=sum(r['controller_status']!='fresh_command' for r in group)))
        fitted=fit(rows,indices)
        reports[name]=dict(original_interval_s=[rows[indices[0]]['stamp_s']-.02,rows[indices[-1]]['stamp_s']],
            samples=len(indices),fresh_input_reconstruction_max_abs_error=max(errors,default=0.),
            unavailable_events=losses,one_second_windows=sec,
            actual_mean_input=np.mean([rows[i]['applied_policy_velocity'] for i in indices],axis=0).tolist(),
            diagnostic_counterfactual_mean_input=np.mean([t['bounded_hold_counterfactual_input'] for t in trace],axis=0).tolist(),
            counterfactual_warning='Same recorded measurements and wait events only; not a rerun, a plant prediction, or a passing claim.',
            first_order_fit_top5=fitted[:5],trace=trace,
            measured_height_range_m=[min(rows[i]['base_height_m'] for i in indices),max(rows[i]['base_height_m'] for i in indices)],
            measured_tilt_range_rad=[min(rows[i]['base_tilt_rad'] for i in indices),max(rows[i]['base_tilt_rad'] for i in indices)])
    for name,indices in indices_by_case.items():
        other='yaw_right' if name=='yaw_left' else 'yaw_left'
        if other not in reports:continue
        model=reports[name]['first_order_fit_top5'][0]
        measured=np.array([rows[i]['measured_body_velocity'][2] for i in indices_by_case[other]])
        predicted=response(model['parameters'],round(model['delay_s']/.02),rows,indices_by_case[other])
        reports[name]['opposite_direction_cross_validation']=dict(case=other,
            freerun_rmse_rad_s=float(np.sqrt(np.mean((predicted-measured)**2))),
            predicted_mean_rad_s=float(np.mean(predicted)),actual_mean_rad_s=float(np.mean(measured)))
        reports[name]['model_identification_reliable']=False
        reports[name]['model_use']='Diagnostic only. Boundary solutions or failed cross-direction prediction do not justify a predictive servo or a constant feedforward gain.'
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    robot=Path(__file__).resolve().parents[1]
    live=module('candidate_velocity_servo',robot/'velocity_servo.py')
    inputs=[robot/'results/adaptive_waist_settle_01',robot/'results/adaptive_suite_posture_01']
    result=dict(schema='g1_yaw_input_diagnostic/v1',pace=1,physical_execution=False,
        fit_scope='Complete original five-second yaw intervals and recorded prehistory. Bounded gain/tau/bias, delay grid0–1.2s; each direction fitted separately, opposite direction held out.',
        limitations=['Closed-loop, short, gait-oscillatory measurements do not uniquely identify a stable plant.',
            'A fit or counterfactual input replay does not justify a new controller or certify tracking.',
            'Different upper-body profiles and serial case histories remain separate.'],
        runs={p.name:audit(p,live) for p in inputs},
        input_sha256={str(p.relative_to(robot)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),robot/'velocity_servo.py']+[f for d in inputs for f in [d/'measurements.jsonl',d/'sources/velocity_servo.py',d/'timed_tasks.json']]})
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for run,report in result['runs'].items():
        for case,values in report.items():
            print(run,case,'reconstruction_error',values['fresh_input_reconstruction_max_abs_error'],
                'mean_inputs',values['actual_mean_input'][2],values['diagnostic_counterfactual_mean_input'][2],
                'best_fit',values['first_order_fit_top5'][0],'crossval',values['opposite_direction_cross_validation'])


if __name__=='__main__':main()
