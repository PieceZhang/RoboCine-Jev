#!/usr/bin/env python3
"""Pace1 parallel simple-plant stress analysis, not physical acceptance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--moving-yaw-gains',default='2:4',help='Comma separated Kp:Ki diagnostic candidates')
    p.add_argument('--moving-yaw-feedforward',default='1:1',help='Comma separated positive:negative moving feedforward pairs')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]/'velocity_servo.py'
    spec=importlib.util.spec_from_file_location('unchanged_native_input_servo',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    profiles=[('baseline_ff1',.8,1.,1.,1.)]
    feedforward=[]
    for pair in args.moving_yaw_feedforward.split(','):
        positive,negative=map(float,pair.split(':'))
        if not np.isfinite([positive,negative]).all() or min(positive,negative)<=0:raise ValueError('positive finite feedforward pair required')
        feedforward.append((positive,negative))
    for pair in args.moving_yaw_gains.split(','):
        kp,ki=map(float,pair.split(':'))
        if not np.isfinite([kp,ki]).all() or min(kp,ki)<=0:raise ValueError('positive finite gain pair required')
        for positive,negative in feedforward:
            profiles.append((f'moving_yaw_PI{kp:g}_{ki:g}_ff{positive:g}_{negative:g}',kp,ki,positive,negative))
    cases=[]
    dt=.02;steps=1000;time_axis=np.arange(steps+1)*dt
    for profile,kp,ki,positive,negative in profiles:
        for gain in (.15,.3,.5,.8,1.2):
            for tau in (.1,.2,.4):
                for lag in (.02,.08,.15):
                    cases.append(dict(profile=profile,moving_kp=kp,moving_ki=ki,moving_ff_positive=positive,moving_ff_negative=negative,
                        gain=gain,tau_s=tau,lag_s=lag,controller=module.VelocityServo(),state=0.,
                        history=np.zeros(steps+1),commands=[],states=[],integrals=[]))
    start=time.monotonic();previous_phase=None
    for step in range(steps):
        stamp=step*dt;phase=step//250;moving=phase in (0,2)
        requested=.12 if phase==0 else (-.12 if phase==2 else 0.)
        for case in cases:
            servo=case['controller']
            if phase!=previous_phase:
                servo.reset()
                # The real parking supervisor slews from the actually issued
                # input after its PI reset. Preserve that continuity here.
                if case['commands']:servo.command[2]=case['commands'][-1]
            servo.kp[2]=case['moving_kp'] if moving else .8
            servo.ki[2]=case['moving_ki'] if moving else 1.
            servo.feedforward_gain[2]=case['moving_ff_positive'] if moving else 1.
            servo.negative_feedforward_gain[2]=case['moving_ff_negative'] if moving else 1.
            command=servo.update([0.,0.,requested],[0.,0.,case['state']],dt)[2]
            assert abs(command)<=1.+1e-12
            if step:assert abs(command-case['commands'][-1])<=.8*dt+1e-12
            case['commands'].append(float(command));case['integrals'].append(float(servo.integral[2]))
            case['history'][step+1]=float(command)
            # Piecewise-linear command delay, including the original zero
            # prehistory. No delay rounded down to a more favourable period.
            target_time=(step+1)*dt-case['lag_s']
            delayed=np.interp(target_time,time_axis[:step+2],case['history'][:step+2])
            alpha=np.exp(-dt/case['tau_s'])
            case['state']=float(alpha*case['state']+(1-alpha)*case['gain']*delayed)
            assert np.isfinite(case['state'])
            case['states'].append(case['state'])
        previous_phase=phase
        delay=start+(step+1)*dt-time.monotonic()
        if delay>0:time.sleep(delay)
    wall=time.monotonic()-start;reports=[]
    for case in cases:
        states=np.asarray(case['states']);cmd=np.asarray(case['commands']);I=np.asarray(case['integrals'])
        positive=float(np.mean(states[125:250]));negative=float(np.mean(states[625:750]))
        reports.append({k:case[k] for k in ('profile','moving_kp','moving_ki','moving_ff_positive','moving_ff_negative','gain','tau_s','lag_s')}|dict(
            positive_original_half_interval_mean_rad_s=positive,negative_original_half_interval_mean_rad_s=negative,
            original_velocity_threshold_rad_s=.03,moving_tracking_pass=bool(abs(positive-.12)<=.03 and abs(negative+.12)<=.03),
            peak_measured_rate_rad_s=float(np.max(np.abs(states))),peak_native_input_rad_s=float(np.max(np.abs(cmd))),
            integral_max_abs=float(np.max(np.abs(I))),finite_bounded=True,
            positive_final1s_peak_to_peak_rad_s=float(np.ptp(states[200:250])),
            negative_final1s_peak_to_peak_rad_s=float(np.ptp(states[700:750])),
            positive_max_accumulated_base_yaw_error_proxy_rad=float(np.max(np.abs(np.cumsum((states[:250]-.12)*dt)))),
            negative_max_accumulated_base_yaw_error_proxy_rad=float(np.max(np.abs(np.cumsum((states[500:750]+.12)*dt)))),
            stop_final_rates_rad_s=[float(states[499]),float(states[999])]))
    output=dict(schema='g1_simple_plant_yaw_servo_stress/v1',physical_execution=False,pace=1,
        simulated_seconds=20.,wall_seconds=wall,timing_pass=wall<=21.,cases=reports,
        unchanged=dict(policy_input_cap_rad_s=1.,input_slew_rad_s2=.8,filter_s=.1,integral_cap=.3,
            stop_kp=.8,stop_ki=1.,stop_feedforward_positive_negative=1.),
        candidate=dict(moving_yaw_profiles=profiles,other_axes_unchanged=True,stop_original_gains=True),
        limitations=['Independent scalar first-order delayed plants; no learned gait, 29joint dynamics, camera or collision geometry.',
            'Boundedness or passing these necessary examples does not certify the physical controller.',
            'For gain.15, original integralcap.3 and FF1 imply PI2 equilibrium rate .07615rad/s; it cannot meet the .09 minimum. This is a retained counterexample, not a reason to enlarge the cap.',
            'Accumulated base yaw error is only a20ms rectangle-rule input to camera-support analysis, not an actual optical-camera tracking error.'],
        evidence_sha256={str(source.resolve()):hashlib.sha256(source.read_bytes()).hexdigest(),
            str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    shutil.copy2(Path(__file__),args.output/'executed_source.py')
    (args.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    for profile,*_ in profiles:
        group=[r for r in reports if r['profile']==profile]
        print(profile,'tracking_pass',sum(r['moving_tracking_pass'] for r in group),'/',len(group),
            'max_rate',max(r['peak_measured_rate_rad_s'] for r in group),flush=True)
    print('pace',20./wall,flush=True)


if __name__=='__main__':main()
