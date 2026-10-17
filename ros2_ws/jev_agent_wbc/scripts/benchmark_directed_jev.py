#!/usr/bin/env python3
"""Reproducible CPU single-candidate experiment, with observed-only Jev input.

Snapshot feasibility is reported separately from closed-loop execution. Seeds
perturb obstacle position/velocity and measured initial state, not the task.
Every run has a new directory, exact input fixtures, API journal and source copy.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def redundant_perturbation(state, draw, base_z, position, rotation):
    """Randomize the 3-D configuration null space, then restore exact camera FK."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from jev_agent_wbc.whole_body_qp import whole_body_kinematics
    scales=np.array([.01,.01,.025,.025,.02,.02,.02,.02,.02])
    state=np.asarray(state,float).copy()
    def jacobian(j):return np.column_stack([np.eye(6)[:,:2],j[:,1:]])
    _,_,j=whole_body_kinematics(state,base_z)
    scaled=jacobian(j)*scales
    null=draw-np.linalg.pinv(scaled)@(scaled@draw)
    null/=max(1.,np.max(abs(null)))
    state+=scales*null
    for _ in range(12):
        p,r,j=whole_body_kinematics(state,base_z)
        error=np.r_[position-p,Rotation.from_matrix(rotation@r.T).as_rotvec()]
        if np.max(abs(error))<1e-11:return state
        state+=scales*np.linalg.lstsq(jacobian(j)*scales,error,rcond=1e-12)[0]
    raise ValueError('redundant perturbation failed to preserve the initial camera pose')


def perturb(source, destination, seed, initial_mode='redundant'):
    import numpy as np
    from jev_agent_wbc.scenarios import NAMES
    rng=np.random.default_rng(seed)
    for name in NAMES:
        target=destination/name
        target.mkdir(parents=True)
        shutil.copy2(source/name/'task.json',target/'task.json')
        document=json.loads((source/name/'runtime_scene.json').read_text())
        state=np.asarray(document['initial_state'])
        draw=rng.uniform(-1.,1.,9)
        if initial_mode=='redundant':
            from jev_agent_wbc.reference import OriginalReference
            original=OriginalReference(source/name/'task.json',document['subject']['position'])
            p,r,_=original.sample(0.)
            changed=redundant_perturbation(state,draw,document['base_z'],p,r)
        elif initial_mode=='independent':
            changed=state+draw*np.array([.01,.01,.025,.025,.02,.02,.02,.02,.02])
        else:raise ValueError('unknown initial perturbation mode')
        delta=changed-state;document['initial_state']=changed.tolist()
        disturbances=[]
        for obstacle in document['obstacles']:
            offset=rng.uniform(-.03,.03,3)
            dv=rng.uniform(-.02,.02,3)
            for frame in obstacle['keyframes']:
                frame['position']=(np.asarray(frame['position'])+offset+(frame['t']-4.8)*dv).tolist()
                frame['velocity']=(np.asarray(frame['velocity'])+dv).tolist()
            disturbances.append(dict(position_at_4_8_s_m=offset.tolist(),velocity_m_s=dv.tolist()))
        document['perturbation']=dict(seed=seed,initial_mode=initial_mode,initial_state_delta=delta.tolist(),obstacles=disturbances)
        (target/'runtime_scene.json').write_text(json.dumps(document,indent=2)+'\n')


def run_snapshot(case, output, args):
    import numpy as np
    from jev_agent_wbc.scenarios import RuntimeScene
    from jev_agent_wbc.sensors import RuntimeLidar
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.task_space_planning import build,diagnose
    from jev_agent_wbc.representation_jvs import JVSGeometryBackend
    from jev_agent_wbc.directed_search import construction_evidence
    from jev_agent_wbc.search_configuration import offers
    from jev_agent_wbc.candidates import SearchBudget
    from jev_agent_wbc.whole_body_qp import Limits
    from jev_agent_wbc.jev import CandidateDecider,SystemOneClient
    from jev_agent_wbc.evaluation import save_json
    output.mkdir(parents=True)
    runtime=RuntimeScene(case/'runtime_scene.json');d=runtime.document
    state=np.array(d['initial_state']);z=d['base_z'];lidar=RuntimeLidar()
    for t in np.linspace(-.8,0.,9):observed=lidar.synthetic_scan(runtime,state,z,float(t))
    observed['subject_model']=d['subject']
    original=OriginalReference(case/'task.json',d['subject']['position'])
    budget=SearchBudget(single_candidate=True,activation_lead_s=args.lead,candidate_wall_budget_s=args.candidate_seconds)
    backend=JVSGeometryBackend(3);observation=backend.observe(state,0.,base_z=z)
    context=dict(objective='visible_intent_completion',time_s=0.,duration_s=8.,phase=original.phase(0.),
        intent=original.task['intent'],subject=original.task['subject'],reference_subject_position=original.subject.tolist(),
        robot_geometry=observation.summary(backend.obstacle_relations(observation,observed)),
        construction=construction_evidence(original,original,observed,0.,0. if args.startup else args.lead,
            mount_height_m=observation.to_dict()['world_from_mount']['translation_m'][2]))
    context['construction']['measured_initial_boundary']=args.startup
    if args.diagnose:
        diagnostic=diagnose(original,observed,state,np.zeros(8),0.,z,Limits(),budget=budget,measured_control=np.zeros(8))
        context['difficulty']=diagnostic['report']
        save_json(output/'diagnostic.json',diagnostic)
    save_json(output/'fixture.json',dict(state=state,scene=observed,context=context))
    client=SystemOneClient('https://api.typesafe.ai/v1/systemone','jev-1.14.0',timeout_s=20.,journal=output/'jev_api.jsonl')
    decider=CandidateDecider(client)
    first=decider.decide(dict(context,stage='strategy'))
    second=decider.decide(dict(context,stage='configuration',search_options=offers(first['strategy'],None)))
    save_json(output/'decisions.json',[first,second])
    start=time.monotonic()
    result=build(original,original,first['strategy'],observed,state,np.zeros(8),0.,z,Limits(),
        budget=budget,measured_control=np.zeros(8),commanded_q=state[3:],search_config=second['search_config'],
        construction_plan=second['construction_plan'],initial_boundary=args.startup)
    wall=time.monotonic()-start
    candidates=result.pop('candidates',[])
    result.pop('kinematic_prior',None)
    third=None
    if candidates:
        third=decider.decide(dict(context,stage='selection',candidates=[c.card() for c in candidates]))
    save_json(output/'decisions.json',[first,second,third])
    save_json(output/'result.json',dict(result,cards=[c.card() for c in candidates]))
    quality=candidates[0].consequences['curve_quality'] if candidates else {}
    return dict(case=case.name,construction=second['construction_plan'],generated=result.get('generated_proposals'),
        checked=len(result.get('evaluations',[])),accepted=len(candidates),
        first_candidate_feasible=len(candidates)==1 and len(result.get('evaluations',[]))==1,
        strict_curve_pass=quality.get('strict_full_curve_pass',False),
        failures=[e.get('rejected') for e in result.get('evaluations',[])],
        wall_s=wall,api_calls=client.calls,scope='snapshot_only_not_closed_loop')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--initial-mode',choices=('redundant','independent'),default='redundant',
        help='default preserves the original camera pose while perturbing redundant configuration')
    p.add_argument('--lead',type=float,default=1.2)
    p.add_argument('--candidate-seconds',type=float,default=20.)
    p.add_argument('--cases',nargs='+')
    p.add_argument('--diagnose',action='store_true')
    p.add_argument('--startup',action='store_true',help='construct the initial shot from measured camera pose before the clock starts')
    a=p.parse_args()
    os.environ['TYPESAFE_API_KEY_FILE']=str(a.key_file.resolve())
    from jev_agent_wbc.scenarios import NAMES
    from jev_agent_wbc.evaluation import save_json
    from jev_agent_wbc.provenance import source_manifest
    a.output.mkdir(parents=True,exist_ok=False)
    save_json(a.output/'provenance.json',source_manifest(ROOT,dict(seed=a.seed,initial_mode=a.initial_mode,lead=a.lead,model='jev-1.14.0',mode='official_jev_single_candidate_cpu')))
    shutil.copytree(ROOT/'jev_agent_wbc',a.output/'frozen_code'/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    # Lazy imports during later cases must use this same source snapshot.
    import jev_agent_wbc
    jev_agent_wbc.__path__=[str((a.output/'frozen_code'/'jev_agent_wbc').resolve())]
    perturb(ROOT/'scenarios',a.output/'scenarios',a.seed,a.initial_mode)
    rows=[]
    for name in a.cases or NAMES:
        row=run_snapshot(a.output/'scenarios'/name,a.output/name,a)
        rows.append(row);save_json(a.output/'summary.json',rows)
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
