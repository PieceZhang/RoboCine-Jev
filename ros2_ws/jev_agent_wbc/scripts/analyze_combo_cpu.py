#!/usr/bin/env python3
"""Independent CPU combo audit using the actual cubic obstacle motion.

A stopped/failed prefix never counts as a completed safe shot. Obstacle truth is
used only here, after execution, and never fed back into the Agent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.kc_svsdf import CollisionGeometry, advance_sequence
from jev_agent_wbc.self_collision import SelfCollision
from jev_agent_wbc.whole_body_qp import Limits
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER,JOINT_UPPER
from jev_agent_wbc.evaluation import save_json


def speed_bound(curve,start,end):
    """Exact component extrema of a piecewise quadratic Hermite velocity."""
    bound=np.zeros(3)
    for i in range(len(curve.x)-1):
        a=max(start,curve.x[i]);b=min(end,curve.x[i+1])
        if b<a:continue
        c3,c2,c1=curve.c[:3,i]
        lo,hi=a-curve.x[i],b-curve.x[i]
        for j in range(3):
            probes=[lo,hi]
            if abs(c3[j])>1e-12:
                root=-c2[j]/(3*c3[j])
                if lo<=root<=hi:probes.append(root)
            bound[j]=max(bound[j],max(abs(3*c3[j]*u*u+2*c2[j]*u+c1[j]) for u in probes))
    return float(np.linalg.norm(bound))


def audit(folder):
    summary=json.loads((folder/'summary.json').read_text())
    rows=json.loads((folder/'telemetry.json').read_text())
    scene=RuntimeScene(folder/'runtime_scene.json')
    api=[]
    for name in ('combo_jev_api.jsonl','jev_api.jsonl'):
        file=folder/name
        if file.exists():api.extend(json.loads(line) for line in file.read_text().splitlines())
    for file in sorted(folder.glob('phase_*/jev_api.jsonl')):
        api.extend(json.loads(line) for line in file.read_text().splitlines())
    successful=[r for r in api if not r.get('error') and r.get('response')]
    acceptance=summary.get('transfer_acceptance',summary.get('shot_acceptance',{}))
    result=dict(case=scene.document['name'],folder=str(folder.resolve()),completed=summary['completed'],
        failure=summary.get('failure'),elapsed_shot_s=summary.get('elapsed_shot_s') or 0.,
        planned_duration_s=json.loads((folder/'original_task.json').read_text())['shot_phase'][-1]['end_s'],
        installed_plans=summary.get('installed_plans',0),shot_accepted=acceptance.get('accepted',False),
        acceptance_type='configuration_transfer' if 'transfer_acceptance' in summary else 'camera_shot',
        failed_checks=acceptance.get('failed_checks',[]),
        api_attempts=len(api),successful_api_requests=len(successful),
        official_models=sorted({r['response'].get('model') for r in successful}),
        api_errors=[r.get('status',r.get('error')) for r in api if r.get('error')],
        wall_s=summary.get('wall_s'),
        scope='CPU constant-command robot integration and actual authored cubic obstacles; not PhysX/hardware',
        source_hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for name in ('telemetry.json','runtime_scene.json','original_task.json')})
    if len(rows)<2:
        return dict(result,prefix_safety_verified=None,full_safety_verified=False,
            reason='no_executed_interval')
    geometry=CollisionGeometry();self_collision=SelfCollision();limits=Limits()
    states=np.array([r['state'] for r in rows]);controls=np.array([r['command'] for r in rows])
    times=np.array([r['t'] for r in rows]);z=scene.document['base_z']
    minimum=float('inf');self_min=float('inf');integration_error=0.
    for i,(a,b) in enumerate(zip(times[:-1],times[1:])):
        dt=b-a;offsets=np.linspace(0.,dt,max(3,int(np.ceil(dt/.002))+1))
        poses=advance_sequence(states[i],controls[i],offsets)
        integration_error=max(integration_error,float(np.max(abs(poses[-1]-states[i+1]))))
        truth=[scene.obstacles(float(t)) for t in a+offsets]
        centers=np.array([[o['center'] for o in sample] for sample in truth])
        radii=np.array([o['radius'] for o in truth[0]])
        distances=geometry.distances_sequence(poses,centers,z)-radii[None,:,None]
        robot_speed=abs(controls[i,0])+abs(controls[i,1:])@geometry.reach
        obstacle_speed=max(speed_bound(curve,float(a),float(b)) for curve in scene.curves.values())
        bound=float(distances.min())-(robot_speed+obstacle_speed)*float(np.max(np.diff(offsets)))/2
        minimum=min(minimum,bound)
        self_min=min(self_min,float(self_collision.sweep(states[i],controls[i],controls[i],dt)))
    initial_control=np.asarray(scene.document.get('initial_control',np.zeros(8)))
    acceleration=np.diff(np.vstack([initial_control,controls]),axis=0)/.02
    speed=float(np.max(abs(controls)/limits.velocity));acc=float(np.max(abs(acceleration)/limits.acceleration))
    wheels=float(np.max(abs(controls@limits.wheels.T))/limits.wheel_speed)
    wheel_acc=float(np.max(abs(acceleration@limits.wheels.T))/limits.wheel_acceleration)
    margin=float(np.minimum(states[:,3:]-JOINT_LOWER,JOINT_UPPER-states[:,3:]).min())
    prefix=minimum>=.025 and self_min>=.004 and margin>=-1e-6 and max(speed,acc,wheels,wheel_acc)<=1.00001 and integration_error<1e-8
    compute=[r['info']['iteration_compute_wall_s'] for r in rows if 'iteration_compute_wall_s' in r['info']]
    return dict(result,prefix_safety_verified=bool(prefix),full_safety_verified=bool(prefix and summary['completed']),
        min_truth_continuous_clearance_lower_m=minimum,min_self_swept_clearance_m=self_min,
        min_joint_margin_rad=margin,max_velocity_ratio=speed,max_acceleration_ratio=acc,
        max_wheel_velocity_ratio=wheels,max_wheel_acceleration_ratio=wheel_acc,
        integration_consistency_max=integration_error,
        compute_wall_max_s=max(compute) if compute else None,
        over_20ms_steps=sum(v>.02 for v in compute),
        sim_to_wall_ratio=float(times[-1]/summary['wall_s']) if summary.get('wall_s') else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for folder in sorted(args.runs.glob('combo_*/*_jev')):
        if (folder/'summary.json').exists():
            row=audit(folder);results.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    provenance=json.loads((args.runs/'provenance.json').read_text()) if (args.runs/'provenance.json').exists() else {}
    requested=len(provenance.get('configuration',{}).get('cases',results))
    report=dict(cases=results,completed=sum(r['completed'] for r in results),
        accepted=sum(r['shot_accepted'] and r['full_safety_verified'] for r in results),
        full_safety_verified=sum(r['full_safety_verified'] for r in results),
        requested_cases=requested,missing_results=requested-len(results),pace=1,source=str(args.runs.resolve()))
    save_json(args.output/'audit.json',report)
    lines=['Combo CPU closed-loop audit','Source: '+str(args.runs.resolve()),
        'Official model: jev-1.14.0; pace=1; configured workers='+str(provenance.get('configuration',{}).get('parallel_cases','unknown'))+'.',
        'Independent safety: actual cubic obstacle motion with Lipschitz bounds between <=2 ms samples.',
        'A safe executed prefix is not a completed safe shot. CPU integration is not PhysX or hardware.', '']
    for r in results:
        lines.extend([r['case'],f"  Executed: {r['elapsed_shot_s']:.2f}/{r['planned_duration_s']:.2f} s; completed={r['completed']}",
            f"  Installed plans={r['installed_plans']}; official responses={r['successful_api_requests']}; models={r['official_models']}",
            f"  Full safety={r['full_safety_verified']}; prefix safety={r.get('prefix_safety_verified')}; {r['acceptance_type']} accepted={r['shot_accepted']}",
            f"  Failed checks={r['failed_checks']}; failure={r['failure']}",
            f"  Minimum continuous clearance lower bound={r.get('min_truth_continuous_clearance_lower_m')}", ''])
    lines.append(f"Completed {report['completed']}/{requested}; full safety {report['full_safety_verified']}/{requested}; accepted {report['accepted']}/{requested}; missing results {report['missing_results']}.")
    (args.output/'REPORT.txt').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
