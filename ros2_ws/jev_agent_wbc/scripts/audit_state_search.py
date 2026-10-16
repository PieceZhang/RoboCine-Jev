#!/usr/bin/env python3
"""Independent CPU execution audit, original-intent comparison and plot export."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.self_collision import SelfCollision
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.whole_body_qp import Limits,whole_body_kinematics
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER,JOINT_UPPER
from jev_agent_wbc.evaluation import save_json
from jev_agent_wbc.provenance import source_manifest


def audit(folder):
    rows=json.loads((folder/'telemetry.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    runtime=RuntimeScene(folder/'runtime_scene.json');z=runtime.document['base_z']
    limits=Limits();body=self_gap=float('inf');self_collision=SelfCollision()
    states=np.array([r['state'] for r in rows]);controls=np.array([r['command'] for r in rows])
    times=np.array([r['t'] for r in rows]);sigma=[]
    for i,row in enumerate(rows):
        _,_,jac=whole_body_kinematics(states[i],z)
        sigma.append(np.linalg.svd(np.array([1.,1.,1.,.35,.35,.35])[:,None]*jac,compute_uv=False)[-1])
        self_gap=min(self_gap,float(self_collision.distances(states[i]).min()))
        if i+1<len(rows):
            dt=times[i+1]-times[i]
            truth=DynamicCollision(runtime.truth_collision(times[i]),times[i],uncertainty_rate=0.)
            body=min(body,float(truth.sweep(states[i],controls[i],dt,z,max_samples=65)['lower'].min()))
            self_gap=min(self_gap,self_collision.sweep(states[i],controls[i],controls[i],dt))
    installs=[e for e in events if e['event']=='plan_installed']
    cards={c['id']:c for e in events for c in e.get('cards',[])}
    selection_cards=[c for e in events if e['event']=='candidate_set' for c in e.get('cards',[])]
    acceleration=np.diff(np.vstack([np.zeros(8),controls]),axis=0)/.02
    position_margin=float(np.min(np.minimum(states[:,3:]-JOINT_LOWER,JOINT_UPPER-states[:,3:])))
    speed=float(np.max(abs(controls)/limits.velocity));acc=float(np.max(abs(acceleration)/limits.acceleration))
    wheels=float(np.max(abs(controls@limits.wheels.T))/limits.wheel_speed)
    wheel_acc=float(np.max(abs(acceleration@limits.wheels.T))/limits.wheel_acceleration)
    active_times=[e['activation_s'] for e in installs]
    installed=[r for r in rows if active_times and r['t']>=min(active_times)]
    def errors(data,key):
        values=np.array([r['info'][key] for r in data])
        return None if not len(values) else dict(rms=float(np.sqrt(np.mean(values**2))),max=float(values.max()),p95=float(np.percentile(values,95)))
    return dict(case=runtime.document['name'],folder=str(folder.resolve()),completed=summary['completed'],
        installed_plans=len(installs),api_calls=summary['jev_calls'],
        model=summary.get('semantic_model'),source_backend=summary['backend'],
        sweep_scope='CPU constant-control intervals and authored constant-velocity sphere; not PhysX or hardware',
        min_truth_swept_clearance_m=body,min_self_swept_clearance_m=self_gap,
        min_joint_margin_rad=position_margin,min_scaled_singular_value=float(min(sigma)),
        max_velocity_ratio=speed,max_acceleration_ratio=acc,max_wheel_velocity_ratio=wheels,max_wheel_acceleration_ratio=wheel_acc,
        independent_safety_pass=bool(summary['completed'] and body>=.025 and self_gap>=.004 and position_margin>=-1e-6 and
            max(speed,acc,wheels,wheel_acc)<=1.00001),
        intent_success=summary.get('intent_success'),intent=summary.get('executed_camera_intent'),
        target_position_m=errors(rows,'position_error'),target_orientation_rad=errors(rows,'orientation_error'),
        installed_target_position_m=errors(installed,'position_error'),installed_target_orientation_rad=errors(installed,'orientation_error'),
        degraded_samples=sum(r['info'].get('tpr_status')=='degraded' for r in rows),
        preparation_status_samples=dict(Counter(r['info'].get('configuration_preparation',{}).get('status','inactive') for r in rows)),
        preparation_events=[e for e in events if e['event'] in ('configuration_preparation_planned','configuration_preparation_state')],
        selected_intent_violations=[cards[e['candidate_id']]['visual']['intent'].get('intent_violations',[]) for e in installs],
        candidates_presented=len(selection_cards),soft_violation_candidates=sum(not c['visual']['intent']['intent_complete'] for c in selection_cards),
        all_presented_candidates_body_checked=all(c.get('kinematic_feasibility',{}).get('whole_body_collision_checked') for c in selection_cards),
        rejection_reasons=dict(Counter(e.get('error') for e in events if e['event']=='proposal_rejected')),
        search_reasons=dict(Counter(x.get('rejected','passed') for e in events for x in (e.get('evaluations') or []))),
        configurations=dict(Counter(e['search_config'] for e in events if e['event']=='search_configuration')),
        final_summary=summary)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for folder in sorted(args.runs.glob('*_jev')):
        if not (folder/'summary.json').exists():continue
        result=audit(folder);results.append(result)
        print(result['case'],result['installed_plans'],result['independent_safety_pass'],result['installed_target_position_m'],flush=True)
    baseline=[]
    if args.baseline:
        for folder in sorted(args.baseline.glob('*_jev')):
            if (folder/'summary.json').exists():baseline.append(dict(case=folder.name[:-4],**json.loads((folder/'summary.json').read_text())))
    manifest=source_manifest(Path(__file__).resolve().parents[1])
    recorded=json.loads((args.runs/'provenance.json').read_text())
    save_json(args.output/'audit.json',dict(runs=results,baseline=baseline,source_matches_run=manifest['source_sha256']==recorded['source_sha256'],
        source_sha256=manifest['source_sha256'],recorded_source_sha256=recorded['source_sha256']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(results),2,figsize=(12,2.7*len(results)),squeeze=False)
    for i,result in enumerate(results):
        rows=json.loads((Path(result['folder'])/'telemetry.json').read_text());t=[r['t'] for r in rows]
        axes[i,0].plot(t,[r['info']['position_error']*1000 for r in rows],label='installed target position error (mm)')
        axes[i,0].axhline(15,ls='--',color='r',alpha=.5,label='FK screening tolerance')
        axes[i,0].set_title(result['case']);axes[i,0].legend(fontsize=7)
        axes[i,1].plot(t,[r['truth_clearance_m'] for r in rows],label='independent true body clearance (m)')
        axes[i,1].axhline(.025,color='r',ls='--',label='hard margin')
        axes[i,1].legend(fontsize=7)
        for ax in axes[i]:ax.set_xlabel('Original shot time (s)');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'tracking_and_clearance.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
