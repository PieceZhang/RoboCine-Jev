#!/usr/bin/env python3
"""Audit measured PhysX logs; never apply a CPU integration certificate to them."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.cinematic_geometry import curve_quality
from jev_agent_wbc.whole_body_qp import Limits
from jev_agent_wbc.evaluation import save_json


def read(path):
    return json.loads(path.read_text())


def journal(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def physics_audit(folder):
    with gzip.open(folder/'physics.csv.gz', 'rt') as stream:
        data = np.genfromtxt(stream, delimiter=',', names=True)
    data = np.atleast_1d(data)
    if len(data)<2:
        return dict(samples=len(data), raw_maxima={}, raw_constraints_pass=None,
                    reason='insufficient_physical_samples')
    times = data['time_s']
    dt = np.diff(times)
    velocities = np.column_stack([data[f'measured_{i}'] for i in range(8)])
    wheels = np.column_stack([data[f'wheel_{i}'] for i in range(2)])
    limits = Limits()
    accelerations = np.diff(velocities, axis=0)/dt[:, None]
    wheel_accelerations = np.diff(wheels, axis=0)/dt[:, None]
    checks = dict(velocity=np.abs(velocities)/limits.velocity,
                  acceleration=np.abs(accelerations)/limits.acceleration,
                  wheel_speed=np.abs(wheels)/limits.wheel_speed,
                  wheel_acceleration=np.abs(wheel_accelerations)/limits.wheel_acceleration)
    stats = {}
    for key, utilization in checks.items():
        row, axis = np.unravel_index(np.argmax(utilization), utilization.shape)
        stamp = times[row+1] if 'acceleration' in key else times[row]
        stats[key] = dict(max_ratio=float(utilization[row, axis]), peak_time_s=float(stamp),
            peak_axis=int(axis), per_axis_max_ratio=utilization.max(axis=0).tolist(),
            over_limit_samples=int(np.sum(utilization.max(axis=1)>1.001)),
            sample_fraction_over_limit=float(np.mean(utilization.max(axis=1)>1.001)))
    # Recompute the existing hard gate from raw values, including additive
    # numeric tolerances. Diagnostic averaging below never changes this gate.
    maxima = dict(position_violation=float(data['position_violation'].max()),
        velocity_violation=float(np.maximum(np.abs(velocities)-limits.velocity, 0.).max()),
        wheel_speed_violation=float(np.maximum(np.abs(wheels)-limits.wheel_speed, 0.).max()),
        acceleration_violation=float(np.maximum(np.abs(accelerations)-limits.acceleration, 0.).max()),
        wheel_acceleration_violation=float(np.maximum(np.abs(wheel_accelerations)-limits.wheel_acceleration, 0.).max()),
        lateral_velocity=float(data['lateral_velocity'].max()))
    passed = all(v <= (.005 if k == 'lateral_velocity' else .001) for k,v in maxima.items())
    width = max(1, round(.02/float(np.median(dt))))
    n = len(velocities)//width
    averaged = velocities[:n*width].reshape(n,width,8).mean(axis=1)
    averaged_acceleration = np.diff(averaged,axis=0)/(width*np.median(dt))
    return dict(samples=len(data), duration_s=float(times[-1]), raw_maxima=maxima,
        raw_constraints_pass=passed, utilization=stats,
        diagnostic_20ms_mean_velocity_acceleration_max_ratio=float(
            (np.abs(averaged_acceleration)/limits.acceleration).max()) if n>1 else None,
        diagnostic_averaging_role='frequency diagnosis only; does not replace raw constraint acceptance')


def inspect(folder, baseline):
    summary = read(folder/'summary.json')
    rows = read(folder/'telemetry.json')
    runtime = read(folder/'runtime_scene.json')
    original = OriginalReference(folder/'original_task.json', runtime['subject']['position'])
    events, api = journal(folder/'events.jsonl'), journal(folder/'jev_api.jsonl')
    result = dict(case=runtime['name'], folder=str(folder.resolve()), completed=summary['completed'],
        elapsed_s=summary.get('elapsed_shot_s',0), failure=summary.get('failure',''),
        contacts=summary.get('contacts'), safety_pass=summary.get('safety_pass',False),
        installed_plans=summary.get('installed_plans',0), intent_success=summary.get('intent_success',False),
        api_calls=len(api), api_errors=sum('error' in r for r in api),
        actual_models=sorted({r['response']['model'] for r in api if r.get('response',{}).get('model')}),
        api_wall_s=dict(median=float(np.median([r['wall_s'] for r in api])),
            max=max(r['wall_s'] for r in api)) if api else None,
        event_counts=dict(Counter(e['event'] for e in events)),
        rejections=dict(Counter(e.get('error','') for e in events if e['event']=='proposal_rejected')),
        failures=[{k:v for k,v in e.items() if k!='evaluations'} for e in events
                  if e['event'] in ('run_failed','installation_rejected','math_failure')],
        candidate_rejection_counts=dict(Counter(v.get('rejected','accepted')
            for e in events for v in (e.get('evaluations') or []))),
        candidate_batches=[dict(time_s=e['time_s'], planner_wall_s=e.get('planner_wall_s'),
            cards=len(e.get('cards',[])), activation_times_s=[c['trajectory'].get('activation_s') for c in e.get('cards',[])])
            for e in events if e['event']=='candidate_set'],
        obstacle_pose_audit=summary.get('obstacle_pose_audit'),
        max_submitted_candidates=max((len(e.get('cards',[])) for e in events if e['event']=='candidate_set'), default=0))
    if baseline:
        cpu = baseline/runtime['name']/(runtime['name']+'_jev')
        result['same_task_and_scene'] = all(read(cpu/f)==read(folder/f)
            for f in ('original_task.json','runtime_scene.json'))
        old = read(cpu/'summary.json')
        result['cpu'] = {key:old.get(key) for key in ('completed','safety_pass','installed_plans','intent_success','min_truth_clearance_m','target_position_rms_m')}
    if not rows:
        return result
    times = np.array([r['t'] for r in rows])
    positions = np.array([r['camera_position'] for r in rows])
    targets = np.array([r['info']['target_position'] for r in rows])
    errors = np.linalg.norm(positions-targets,axis=1)
    class Actual:
        duration = min(original.duration, times[-1])
        activation_s = 0.
        def sample(self,t):
            return np.array([np.interp(t,times,positions[:,i]) for i in range(3)]),np.eye(3),np.zeros(6)
    actual = Actual(); actual.original = original
    quality = curve_quality(actual) if len(rows)>2 else None
    result.update(tracking_rms_m=float(np.sqrt(np.mean(errors**2))), tracking_max_m=float(errors.max()),
        tracking_peak_s=float(times[np.argmax(errors)]), tracking_terminal_m=float(errors[-1]),
        fk_camera_discrepancy_max_m=max(r['info'].get('fk_position_error_m',0.) for r in rows),
        min_clearance_m=summary['min_truth_clearance_m'], executed_prefix_curve=quality,
        full_curve_pass=bool(summary['completed'] and quality and quality['strict_full_curve_pass']),
        terminal_frame=summary.get('executed_camera_intent',{}).get('terminal_frame'),
        motion_trend=summary.get('executed_camera_intent',{}).get('motion_trend'),
        wbc_degraded_samples=sum(r['info'].get('tpr_status')=='degraded' for r in rows),
        camera_actuator_infeasible_samples=sum(r['info'].get('tpr_report',{}).get('camera_actuator_feasible') is False for r in rows),
        camera_actuator_measurement_missing_samples=sum('camera_actuator_feasible' not in r['info'].get('tpr_report',{}) for r in rows),
        control_loop_wall_ms=summary.get('control_loop_wall_ms'),
        physical=physics_audit(folder))
    result['physical']['matches_online_audit'] = (all(abs(v-summary['physical_audit']['maxima'][k])<1e-5
        for k,v in result['physical']['raw_maxima'].items()) if result['physical']['raw_maxima'] else None)
    angles=Rotation.from_matrix(np.array([r['base_rotation'] for r in rows])).as_euler('xyz')[:,:2]
    result['max_abs_base_roll_pitch_deg']=np.rad2deg(abs(angles).max(axis=0)).tolist()
    result['first_dynamic_track_time_s']=next((r['t'] for r in rows if r.get('runtime_tracks')),None)
    seen = [r['subject_track'] for r in rows if r.get('subject_track',{}).get('valid')]
    if seen:
        subject_errors = np.linalg.norm(np.array([r['position'] for r in seen])-runtime['subject']['position'],axis=1)
        result['subject_center_error'] = dict(median_m=float(np.median(subject_errors)), max_m=float(subject_errors.max()),
                                              role='offline fixture audit; never supplied as future planner truth')
    return result


def plots(results, baseline, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,9))
    fig2, axes = plt.subplots(len(results),2,figsize=(13,3*len(results)),squeeze=False)
    for i,r in enumerate(results):
        folder=Path(r['folder']); rows=read(folder/'telemetry.json')
        ax=fig.add_subplot(2,3,i+1,projection='3d')
        if not rows: continue
        t=np.array([v['t'] for v in rows]); p=np.array([v['camera_position'] for v in rows])
        target=np.array([v['info']['target_position'] for v in rows])
        original=OriginalReference(folder/'original_task.json',read(folder/'runtime_scene.json')['subject']['position'])
        ref=np.array([original.sample(float(s))[0] for s in np.linspace(0,original.duration,301)])
        ax.plot(*ref.T,'--',c='gray',label='Original');ax.plot(*p.T,c='#b04e27',label='PhysX actual')
        if baseline:
            cpu=read(baseline/r['case']/(r['case']+'_jev')/'telemetry.json')
            xyz=np.array([v['camera_position'] for v in cpu])
            ax.plot(*xyz.T,c='#007e85',label='CPU actual',alpha=.7)
        ax.set_title(r['case']+(f" {t[-1]:.1f}s recorded" if r['completed'] else f" STOP {t[-1]:.2f}s"))
        ax.set_xlabel('X m');ax.set_ylabel('Y m');ax.set_zlabel('Z m');ax.legend(fontsize=7)
        ax.set_box_aspect([max(.05,hi-lo) for lo,hi in (ax.get_xlim(),ax.get_ylim(),ax.get_zlim())])
        axes[i,0].plot(t,np.linalg.norm(p-target,axis=1)*1000)
        axes[i,0].set_ylabel(r['case']+'\ntracking mm');axes[i,0].grid(alpha=.2)
        axes[i,1].plot(t,[v['truth_clearance_m']*1000 for v in rows]);axes[i,1].axhline(25,c='red',ls='--')
        axes[i,1].set_ylabel('body clearance mm');axes[i,1].grid(alpha=.2)
        for a in axes[i]:a.set_xlabel('physical scene time s');a.set_xlim(0,8.4)
    fig.tight_layout();fig.savefig(output/'actual_paths_3d.png',dpi=150);plt.close(fig)
    fig2.tight_layout();fig2.savefig(output/'tracking_clearance.png',dpi=130);plt.close(fig2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    results=[inspect(folder,a.baseline) for folder in a.runs]
    save_json(a.output/'comparison.json',results)
    plots(results,a.baseline,a.output)
    for r in results:
        print(r['case'],r['completed'],r['elapsed_s'],r['installed_plans'],r['failure'])


if __name__=='__main__':
    main()
