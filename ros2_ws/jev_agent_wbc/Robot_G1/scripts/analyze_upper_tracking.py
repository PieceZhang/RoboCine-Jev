#!/usr/bin/env python3
"""Read-only explicit-PD tracking and damping sensitivity; no physical execution.

The local response sweep freezes original references and the unmodelled external
forcing. Its diagonal effective inertias are sensitivity parameters, not measured
link inertias. It cannot certify a collision clearance or a future physical run.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.integrate import solve_ivp


def source_gains(path):
    assignment=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='G1_29DOF' for t in n.targets))
    actuators=next(k.value for k in assignment.value.keywords if k.arg=='actuators')
    groups={ast.literal_eval(k):v for k,v in zip(actuators.keys,actuators.values)}
    return {group:{k.arg:ast.literal_eval(k.value) for k in groups[group].keywords
        if k.arg in ('stiffness','damping')} for group in ('arms','waist')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--independent-abort-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    robot=Path(__file__).resolve().parents[1]
    source=robot/'upstream/WBC-AGILE/agile/rl_env/assets/robots/unitree_g1.py'
    gains=source_gains(source)
    rows=[json.loads(line) for line in (args.result/'measurements.jsonl').open()]
    audit=json.loads((args.result/'actuator_audit.json').read_text())
    independent=json.loads(args.independent_abort_audit.read_text())
    selected=[r for r in rows if 26.50-1e-8<=r['stamp_s']<=27.18+1e-8]
    names=rows[0]['joint_names']
    owned=list(rows[-1]['pink_upper_velocity'])
    import re
    kp={};kd={}
    for name in owned:
        group='waist' if name.startswith('waist') else 'arms'
        kp[name]=next(float(v) for pattern,v in gains[group]['stiffness'].items() if re.fullmatch(pattern,name))
        kd[name]=next(float(v)*3 for pattern,v in gains[group]['damping'].items() if re.fullmatch(pattern,name))
    for group in ('arms','waist'):
        for name,actual in zip(audit['native_actuator_joint_names'][group],audit['native_pd_damping'][group][0]):
            if name in kd:assert abs(actual-kd[name])<2e-6
    indices=[names.index(n) for n in owned]
    cap=np.asarray(audit['effort_limits'])[indices]
    q=np.asarray([r['joint_positions'] for r in selected])[:,indices]
    qref=np.asarray([r['applied_joint_position_target'] for r in selected])[:,indices]
    qd=np.asarray([r['measured_joint_velocity'] for r in selected])[:,indices]
    vref=np.asarray([[r['pink_upper_velocity'].get(n,0.) for n in owned] for r in selected])
    gravity=np.asarray([[r['upper_gravity_feedforward'].get(n,0.) for n in owned] for r in selected])
    effort=np.asarray([r['applied_joint_effort_estimate'] for r in selected])[:,indices]
    peak=np.asarray([r['applied_joint_effort_peak_abs'] for r in selected])[:,indices]
    K=np.asarray([kp[n] for n in owned]);D=np.asarray([kd[n] for n in owned])
    uncapped3=K*(qref-q)+D*(vref-qd)+gravity
    additional=(2/3)*D*(vref-qd)
    uncapped5=uncapped3+additional
    critical=[n for n in owned if n.startswith('right_shoulder')]
    trace=[]
    source_trace={round(r['stamp_s'],8):r for r in independent['trace']}
    for i,r in enumerate(selected):
        proof=source_trace[round(r['stamp_s'],8)]
        item=dict(stamp_s=r['stamp_s'],independent_distance_m=proof['distance_m'],
            same_state_J_native_m_s=proof['J_dot_native_command_m_s'],
            same_state_J_actual_m_s=proof['J_dot_measured_qdot_m_s'],
            next20ms_measured_distance_rate_m_s=proof['next20ms_distance_finite_difference_m_s'],
            critical_joint={})
        for name in critical:
            j=owned.index(name)
            item['critical_joint'][name]=dict(position_rad=q[i,j],reference_rad=qref[i,j],
                reference_lead_rad=qref[i,j]-q[i,j],native_velocity_rad_s=vref[i,j],measured_velocity_rad_s=qd[i,j],
                issued_effort_last_substep_Nm=effort[i,j],substep_peak_effort_Nm=peak[i,j],
                original_effort_cap_Nm=cap[j],end_state_algebraic_PD_plus_g_3_Nm=uncapped3[i,j],
                end_state_additional_damping_5_Nm=additional[i,j],
                end_state_algebraic_PD_plus_g_5_capped_Nm=float(np.clip(uncapped5[i,j],-cap[j],cap[j])))
        trace.append(item)
    # Keep the measured external forcing in a delta model. A zero damping
    # change yields identically zero state correction. No Pin inertia or COM
    # principal-axis convention is guessed here.
    times=np.asarray([r['stamp_s'] for r in selected]);duration=times[-1]-times[0]
    sweep=[]
    for effective_inertia in (.03,.06,.12,.24,.48,.96):
        def rhs(t,x):
            delta=x[:len(owned)];delta_v=x[len(owned):]
            actual_v=np.array([np.interp(times[0]+t,times,qd[:,j]) for j in range(len(owned))])
            requested=np.array([np.interp(times[0]+t,times,vref[:,j]) for j in range(len(owned))])
            return np.r_[delta_v,(-K*delta-(5/3)*D*delta_v+(2/3)*D*(requested-actual_v))/effective_inertia]
        sol=solve_ivp(rhs,(0.,duration),np.zeros(2*len(owned)),t_eval=times-times[0],
            max_step=.001,rtol=1e-8,atol=1e-10)
        delta=sol.y[:len(owned)].T;delta_v=sol.y[len(owned):].T
        predicted_distance=[]
        for i,row in enumerate(selected):
            proof=source_trace[round(row['stamp_s'],8)]
            gradient=np.asarray([proof['critical_distance_gradient_m_per_rad'][n] for n in owned])
            predicted_distance.append(proof['distance_m']+float(gradient@delta[i]))
        sweep.append(dict(assumed_diagonal_effective_inertia_kg_m2=effective_inertia,
            local_linearized_minimum_distance_m=float(min(predicted_distance)),
            terminal_correction_rad={n:float(delta[-1,owned.index(n)]) for n in critical},
            terminal_rate_correction_rad_s={n:float(delta_v[-1,owned.index(n)]) for n in critical}))
    peak_ratio=np.max(peak/cap,axis=0)
    output=dict(schema='g1_upper_pd_tracking_analysis/v1',physical_execution=False,
        scope='Read-only measured torque and same-state explicit-PD algebra; frozen-reference diagonal sensitivity only.',
        source_result=str(args.result),sample_interval_s=.02,window_s=[float(times[0]),float(times[-1])],
        source_native_stiffness_Nm_rad=kp,actual_damping_3_Nm_s_rad=kd,
        no_measured_upper_effort_saturation=bool(np.max(peak_ratio)<1.-1e-6),
        all_upper_peak_effort_over_cap=float(np.max(peak_ratio)),
        joint_peak_effort_over_cap=dict(zip(owned,peak_ratio.tolist())),
        critical_peak_extra_damping_torque_Nm={n:float(np.max(np.abs(additional[:,owned.index(n)]))) for n in critical},
        same_state_damping5_algebraic_effort_over_cap=float(np.max(np.abs(uncapped5)/cap)),
        alignment_limit='Recorded issued effort is the last 5ms write evaluated before the final state; end-state PD algebra is a counterfactual, not a reconstruction of that write.',
        additional_relative_dissipation_integral_J=float(np.sum((2/3)*D*(qd-vref)**2)*.02),
        independent_clearance_gradient_association_verified=True,
        independent_gradient_report=str(args.independent_abort_audit),
        sensitivity_assumptions=['Original reference, velocity reference and external coupling residual frozen.',
            'Independent diagonal effective inertia parameters are uncalibrated; actual full-body mass matrix/coupling not used.',
            'First-order distance correction uses the independently verified gradient at each original state.',
            'This is not a clearance, timing, torque or stop acceptance certificate. New physical validation required.'],
        diagonal_closed_loop_sensitivity=sweep,
        physical_tracking_and_stop_verified=False,trace_file='trace.json',evidence_sha256={})
    sources=[Path(__file__),source,args.result/'measurements.jsonl',args.result/'actuator_audit.json',
        args.result/'constraint_abort.json',args.independent_abort_audit]
    for path in sources:output['evidence_sha256'][str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__),args.output/'executed_source.py')
    (args.output/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('no_measured_upper_effort_saturation','all_upper_peak_effort_over_cap',
        'critical_peak_extra_damping_torque_Nm','same_state_damping5_algebraic_effort_over_cap','diagonal_closed_loop_sensitivity')},indent=2))


if __name__=='__main__':main()
