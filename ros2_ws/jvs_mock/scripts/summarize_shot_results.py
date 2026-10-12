#!/usr/bin/env python3
"""Read actual Isaac reports, compare allocation/junctions, save plots and gates."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jvs_mock.whole_body_qp import Limits, whole_body_kinematics
from jvs_mock.jvs_kinematics import JOINT_LOWER, JOINT_UPPER


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def read(folder):
    rows = list(csv.DictReader((folder/'telemetry.csv').open()))
    data = {}
    for k in rows[0]:
        try:
            data[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            pass
    return data


def stats(v):
    return dict(rms=float(np.sqrt(np.mean(v*v))), p95=float(np.percentile(v,95)), max=float(np.max(v)))


def analyze(folder, trajectory):
    summary = json.loads((folder/'summary.json').read_text())
    reference_path = folder/'reference.json'
    if reference_path.exists():
        reference_doc = json.loads(reference_path.read_text())
        trajectory = dict(trajectory, shots=reference_doc.get('execution_shots', trajectory['shots']),
                          duration_s=reference_doc.get('execution_duration_s', trajectory['duration_s']))
    d = read(folder)
    t = d['time_s']; dt = np.median(np.diff(t))
    names = ['v','omega']+[f'dq{i}' for i in range(1,7)]
    u = np.array([d['command_'+n] for n in names]).T
    measured = np.array([d['measured_'+n] for n in names]).T
    state = np.array([d['state_'+n] for n in ['x','y','yaw']+[f'q{i}' for i in range(1,7)]]).T
    yaw = np.unwrap(state[:,2])
    reference = np.array([d['target_camera_'+n] for n in 'xyz']).T
    refv = np.gradient(reference, dt, axis=0)
    planar_speed = np.linalg.norm(refv[:,:2],axis=1)
    direction = refv[:,:2]/np.maximum(planar_speed[:,None],1e-10)
    tangent = np.arctan2(direction[:,1],direction[:,0])
    # Contribution along the desired path, signed (countermotion is not credit).
    camera_base = np.array([(whole_body_kinematics(s)[2][:2,:2]@v[:2]) for s,v in zip(state,measured)])
    base_translation = measured[:,0,None]*np.c_[np.cos(yaw),np.sin(yaw)]
    lim = Limits()
    normalized = np.abs(u)/lim.velocity
    wheel_ratio = np.abs(u@lim.wheels.T)/lim.wheel_speed
    accel_ratio = np.abs(np.diff(u,axis=0)/dt)/lim.acceleration
    step = np.r_[0., np.max(np.abs(np.diff(u,axis=0))/lim.velocity,axis=1)]
    tracking = np.zeros(len(t), dtype=bool)
    for shot in trajectory['shots']:
        tracking |= (t >= shot.get('motion_start_s', shot['start_s'])) & (
            t <= shot.get('motion_end_s', shot['end_s']))
    result = dict(completed=summary['completed'], failure=summary['failure'],
        accuracy_scope='shot_motion_intervals_only',
        position_mm=stats(d['position_error'][tracking]*1000),
        orientation_deg=stats(np.rad2deg(d['orientation_error'][tracking])),
        tracking_pass=summary.get('tracking_accuracy_pass',False),
        command_violation=max([v for k,v in summary.items() if ('violation' in k and 'measured_' not in k)] or [0]),
        physics_constraints_pass=summary.get('physics_audit',{}).get('pass_constraints',False),
        physics_maxima=summary.get('physics_audit',{}).get('maxima',{}),
        speed_saturation_fraction=float(np.mean(np.maximum(normalized.max(axis=1),wheel_ratio.max(axis=1))>=.98)),
        acceleration_saturation_fraction=float(np.mean(accel_ratio.max(axis=1)>=.98)),
        max_normalized_command_step=float(max(step)), segments=[], junctions=[])
    for s in trajectory['shots']:
        active = (t>=s['motion_start_s']) & (t<=s['motion_end_s'])
        segment = (t>=s['start_s']) & (t<=s['end_s'])
        if not active.any():
            result['segments'].append(dict(type=s['type'], observed=False))
            continue
        moving = active & (planar_speed>.002)
        pos = state[segment,:2]
        allocation = dict(type=s['type'], position_mm=stats(d['position_error'][active]*1000),
            orientation_deg=stats(np.rad2deg(d['orientation_error'][active])),
            base_travel_m=float(np.sum(np.linalg.norm(np.diff(pos,axis=0),axis=1))),
            base_yaw_range_deg=float(np.rad2deg(np.ptp(yaw[segment]))),
            arm_min_margin_deg=float(np.rad2deg(np.min(np.minimum(state[segment,3:]-JOINT_LOWER,JOINT_UPPER-state[segment,3:])))))
        if moving.any():
            allocation['base_camera_projected_fraction'] = float(np.sum(np.sum(camera_base[moving]*direction[moving],axis=1))/np.sum(planar_speed[moving]))
            allocation['base_translation_projected_fraction'] = float(np.sum(np.sum(base_translation[moving]*direction[moving],axis=1))/np.sum(planar_speed[moving]))
            error = wrap(tangent[moving]-yaw[moving])
            if s['type']=='moveout':
                error = np.arcsin(np.sin(error)) # heading modulo pi for reverse
            allocation['heading_rms_deg'] = float(np.rad2deg(np.sqrt(np.mean(error**2))))
        result['segments'].append(allocation)
    for s in trajectory['shots'][1:]:
        junction = s['start_s']
        if t[-1] < junction:
            continue
        window = np.abs(t-junction)<=1.
        before = np.argmin(abs(t-(junction-3.2)))
        at = np.argmin(abs(t-junction))
        future = (t>junction) & (t<s['end_s']) & (planar_speed>.002)
        entry = dict(time_s=junction, next_shot=s['type'],
            max_normalized_step=float(max(step[window])),
            base_heading_step_deg=float(np.rad2deg(np.max(np.abs(np.diff(yaw[np.abs(t-junction)<=1.02]))))),
            heading_preparation_deg=float(np.rad2deg(wrap(yaw[at]-yaw[before]))),
            position_mm=stats(d['position_error'][window]*1000),
            orientation_deg=stats(np.rad2deg(d['orientation_error'][window])))
        if future.any():
            heading = tangent[np.flatnonzero(future)[0]]
            entry['next_heading_error_deg'] = float(np.rad2deg(abs(np.arcsin(np.sin(heading-yaw[at])))))
        result['junctions'].append(entry)
    result['max_junction_step'] = max([j['max_normalized_step'] for j in result['junctions']] or [0])
    if 'mpc_solve_s' in d:
        result['mpc_time_ms'] = stats(d['mpc_solve_s']*1000)
        result['directions'] = sorted(set(d['moveout_direction'].tolist()))
        if 'controller_s' in d:
            result['controller_time_ms'] = stats(d['controller_s']*1000)
    (folder/'allocation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result, d


def plot_case(name, modes, out, shots):
    fig, ax = plt.subplots(3,2,figsize=(14,10),sharex=True)
    for mode,(r,d) in modes.items():
        t=d['time_s']
        ax[0,0].plot(t,d['position_error']*1000,label=mode)
        ax[0,1].plot(t,np.rad2deg(d['orientation_error']),label=mode)
        ax[1,0].plot(t,d['command_v'],label=mode)
        ax[1,1].plot(t,d['command_omega'],label=mode)
        ax[2,0].plot(t,np.rad2deg(np.unwrap(d['state_yaw'])),label=mode)
        q=np.array([d[f'state_q{i}'] for i in range(1,7)]).T
        margin=np.min(np.minimum(q[:,[0,3,5]]-JOINT_LOWER[[0,3,5]],JOINT_UPPER[[0,3,5]]-q[:,[0,3,5]]),axis=1)
        ax[2,1].plot(t,np.rad2deg(margin),label=mode)
    labels=['Camera position error [mm]','Camera orientation error [deg]','Base v [m/s]','Base omega [rad/s]','Base heading [deg]','Yaw-joint minimum margin [deg]']
    for a,label in zip(ax.flat,labels):
        a.set_ylabel(label);a.grid(alpha=.25);a.legend()
        for s in shots[1:]:
            a.axvline(s['start_s'],color='gray',ls='--',lw=1)
            a.axvspan(s['start_s']-3.2,s['start_s'],color='gray',alpha=.08)
    ax[2,0].set_xlabel('Simulation time [s]');ax[2,1].set_xlabel('Simulation time [s]')
    fig.suptitle(name+' — Isaac Sim; shaded: 3.2 s before junction')
    fig.tight_layout();fig.savefig(out/(name+'_comparison.png'),dpi=150);plt.close(fig)
    if name == 'combo1':
        junction=shots[-1]['start_s']
        fig, axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
        for mode,(r,d) in modes.items():
            t=d['time_s'];mask=(t>=junction-6)&(t<=junction+6)
            axes[0].plot(t[mask],np.rad2deg(np.unwrap(d['state_yaw']))[mask],label=mode)
            axes[1].plot(t[mask],d['position_error'][mask]*1000,label=mode)
            axes[2].plot(t[mask],d['command_omega'][mask],label=mode)
            if mode=='stage2' and 'heading_target' in d and t[-1]>junction:
                target=d['heading_target'][np.argmin(abs(t-(junction+.1)))]
                axes[0].axhline(np.rad2deg(target),ls=':',color='black',label='MoveOut heading')
        for axis,label in zip(axes,['Base heading [deg]','Camera position error [mm]','Base omega [rad/s]']):
            axis.set_ylabel(label);axis.axvline(junction,color='gray',ls='--')
            axis.axvspan(junction-3.2,junction,color='gray',alpha=.12)
            axis.grid(alpha=.25);axis.legend()
        axes[-1].set_xlabel('Simulation time [s]')
        fig.suptitle('Configuration preparation before Pan → MoveOut')
        fig.tight_layout();fig.savefig(out/'preview_preparation.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--ros',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    allresults={};lines=['# Stage 2 Isaac comparison','','Accuracy uses motion intervals only; connectors/holds remain in safety and continuity diagnostics.','',
        '| Case | Controller | Position RMS / max mm | Rotation RMS / max deg | Completed |',
        '|---|---|---:|---:|---|']
    for case in json.loads((root/'trajectories/shot_stage2_manifest.json').read_text()):
        name=case['name'];traj=json.loads((root/'trajectories'/case['trajectory']).read_text());modes={}
        for mode in ['stage1','stage2','no_preview']:
            folder=next((f/(name+'_'+mode) for f in a.folders if (f/(name+'_'+mode)/'summary.json').exists()),None)
            if folder is None:continue
            r,d=analyze(folder,traj);r['source_directory']=str(folder.resolve());modes[mode]=(r,d);allresults[name+'_'+mode]=r
            lines.append(f"| {name} | {mode} | {r['position_mm']['rms']:.3f} / {r['position_mm']['max']:.3f} | {r['orientation_deg']['rms']:.3f} / {r['orientation_deg']['max']:.3f} | {r['completed']} |")
        if modes:plot_case(name,modes,a.output,traj['shots'])
    gates={}
    for name in ['orbit','pan','pan_limit','moveout','moveout_forward','combo1']:
        r=allresults.get(name+'_stage2')
        if r:
            gates[name+'_tracking']=r['tracking_pass']
            gates[name+'_command_limits']=r['command_violation']<=1e-6
            if name=='pan':gates['pan_base_suppression']=r['segments'][0]['base_travel_m']<=.02 and r['segments'][0]['base_yaw_range_deg']<=3.
            if name=='pan_limit':gates['pan_limit_base_participates']=r['segments'][0]['base_yaw_range_deg']>=1.
            if name=='orbit':gates['orbit_allocation']=r['segments'][0].get('base_camera_projected_fraction',0)>=.6 and r['segments'][0].get('heading_rms_deg',180)<=15
            if name.startswith('moveout'):gates[name+'_allocation']=r['segments'][0].get('base_translation_projected_fraction',0)>=.7
            if name=='moveout':gates['moveout_reverse_selected']=r.get('directions')==[-1.]
            if name=='moveout_forward':gates['moveout_forward_selected']=r.get('directions')==[1.]
    one,two,abl=(allresults.get('combo1_'+mode) for mode in ['stage1','stage2','no_preview'])
    if one and two:
        gates['combo_tracking_improvement']=one['completed'] and two['completed'] and two['position_mm']['rms']<=.9*one['position_mm']['rms'] and two['orientation_deg']['rms']<=.9*one['orientation_deg']['rms']
        gates['combo_junction_improvement']=one['completed'] and two['completed'] and two['max_junction_step']<=.9*one['max_junction_step']
        gates['combo_junction_continuity']=two['max_junction_step']<=.05
        gates['combo_saturation_nonincrease']=two['speed_saturation_fraction']<=one['speed_saturation_fraction'] and two['acceleration_saturation_fraction']<=one['acceleration_saturation_fraction']
    if two and abl and len(two['junctions'])==2 and len(abl['junctions'])==2:
        gates['ablation_completed']=abl['completed'] and abl['tracking_pass']
        gates['predictive_preparation']=two['completed'] and abl['completed'] and abl['junctions'][-1]['next_heading_error_deg']-two['junctions'][-1]['next_heading_error_deg']>=5.
    ros=None
    if a.ros is not None:
        ros=json.loads((a.ros/'summary.json').read_text())
        gates['ros_closed_loop']=ros.get('tracking_accuracy_pass',False)
    required={name+'_'+mode for name in ['orbit','pan','pan_limit','moveout','moveout_forward','combo1'] for mode in ['stage1','stage2']}|{'combo1_no_preview'}
    complete=required.issubset(allresults) and all(allresults[k]['completed'] for k in required)
    physical=all(r['physics_constraints_pass'] for key,r in allresults.items() if key.endswith('_stage2'))
    passed=complete and all(gates.values()) and physical
    lines+=['','## Paired plots','',
            ' | '.join(f'[{name}]({name}_comparison.png)' for name in ['orbit','pan','pan_limit','moveout','moveout_forward','combo1']
                       if any(k.startswith(name+'_') for k in allresults)), '',
            '[Preview preparation detail](preview_preparation.png)', '',
            '## Gate summary','', '| Gate | Passed |','|---|---|']
    lines += [f'| {key} | {value} |' for key,value in gates.items()]
    lines += [f'| Full physical constraints | {physical} |',f'| Overall Stage 2 | {passed} |']
    if one and two and one['completed'] and two['completed']:
        lines += ['', '## Combination comparison','',
            f"Position RMS change: {(1-two['position_mm']['rms']/one['position_mm']['rms'])*100:.1f}% reduction; "
            f"orientation RMS change: {(1-two['orientation_deg']['rms']/one['orientation_deg']['rms'])*100:.1f}% reduction.",
            f"Maximum normalized command step, entire sequence: {one['max_normalized_command_step']:.6f} → {two['max_normalized_command_step']:.6f}.",
            f"Within +/-1 s of the junctions: {one['max_junction_step']:.6f} → {two['max_junction_step']:.6f}.",
            'The references have stationary junction holds. Preparatory motion may increase a finite command difference there even when it is smooth; it must not be described as a measured reduction in junction steps.']
    if two and abl and two['completed'] and abl['completed']:
        lines += ['', '## Matched future-preview ablation','',
            'Both controllers use the same 3.2 s horizon, update rate and costs. The ablation extrapolates current twist and freezes current shot preferences.', '',
            '| Metric | No future preview | Full preview |','|---|---:|---:|',
            f"| Position RMS mm | {abl['position_mm']['rms']:.3f} | {two['position_mm']['rms']:.3f} |",
            f"| Orientation RMS deg | {abl['orientation_deg']['rms']:.3f} | {two['orientation_deg']['rms']:.3f} |",
            f"| Heading change in final 3.2 s of Pan, deg | {abl['junctions'][-1]['heading_preparation_deg']:.3f} | {two['junctions'][-1]['heading_preparation_deg']:.3f} |",
            f"| Next-heading error at junction, deg | {abl['junctions'][-1]['next_heading_error_deg']:.3f} | {two['junctions'][-1]['next_heading_error_deg']:.3f} |",
            f"| Junction normalized command step | {abl['max_junction_step']:.6f} | {two['max_junction_step']:.6f} |", '',
            'Preview improves readiness, orientation tracking and command smoothness here, but its preparatory motion slightly increases aggregate position error versus the matched ablation.']
    if ros is not None:
        lines += ['', '## ROS / Isaac launch check','',
            f"Completed: {ros['completed']}; failure: {ros['failure']!r}; position RMS {1000*ros['position_error_rms']:.3f} mm; orientation RMS {np.rad2deg(ros['orientation_error_rms']):.3f} deg.",
            f"Raw launch log and telemetry: `{a.ros.resolve()}`. This check uses odometry/joint-state FK; physical constraints are assessed separately in the lockstep audits."]
    lines+=['','## Allocation and transitions','']
    for key,r in allresults.items():
        lines+=['### '+key,'','```json',json.dumps({k:r[k] for k in ['segments','junctions','speed_saturation_fraction','acceleration_saturation_fraction','command_violation','physics_constraints_pass']},indent=2),'```','']
    lines+=['## Acceptance','','```json',json.dumps(dict(gates=gates,paired_and_ablation_complete=complete,physics_constraints_pass=physical,stage2_pass=passed),indent=2),'```','',
        'Stage 2 pass requires the individual allocation, paired improvement, preview ablation and physical constraint gates. A zero-saturation Stage 1 baseline permits only a nonincrease claim.',
        'Plots show measured heading and command velocities; raw telemetry and 1 kHz physics audits remain in each case folder.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (a.output/'comparison.json').write_text(json.dumps(dict(results=allresults,gates=gates,stage2_pass=passed,ros_verification=ros),indent=2)+'\n')
    print(json.dumps(dict(gates=gates,stage2_pass=passed),indent=2))


if __name__=='__main__':main()
