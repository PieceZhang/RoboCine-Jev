#!/usr/bin/env python3
"""Offline, causal candidate diagnostics. Does not call Jev or authorize execution."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from jev_agent_wbc.candidates import SearchBudget,path_hash
from jev_agent_wbc.contracts import native
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.representation import Observation
from jev_agent_wbc.multi_robot_geometry import NativeKinematicModel
from jev_agent_wbc.multi_robot_planning import candidate_paths,lift,card,native_search_budget,CameraAttachmentScreen
from jev_agent_wbc.multi_robot_audit import camera_intrinsics,visibility_scene
from jev_agent_wbc.camera_intent import IntentSpec
from jev_agent_wbc.task_space_planning import camera_scene
from jev_agent_wbc.floating_camera import evaluate_path,intent_prediction


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def screen(directory,shot_time=None,family='joint'):
    source=directory/'avoidance'
    start=next(e['shot_start_s'] for e in rows(source/'events.jsonl')
               if e['event']=='scenario_bound')
    observations=[Observation.from_dict(o) for o in rows(source/'observations.jsonl')]
    maps=rows(source/'observed_maps.jsonl')
    # Earliest fresh public acquisition with a causally available confirmed
    # mover. Never synthesize a perfect obstacle from the authored scene.
    selected=None
    for obs in observations:
        if shot_time is not None and obs.stamp_s<start+shot_time:continue
        available=[m for m in maps if m['consumed_s']<=obs.stamp_s]
        if not available:continue
        snapshot=available[-1]['snapshot']
        if (0<=obs.stamp_s-snapshot['observation_stamp']<=.25
                and any(t['moving_confirmed'] for t in snapshot.get('tracks',[]))):
            selected=(obs,snapshot);break
    if selected is None:
        return dict(source=str(directory),error='no causal fresh moving-object observation')
    observation,snapshot=selected
    state=next(s for s in rows(source/'native_states.jsonl') if s['observation_id']==observation.id)
    if state['stamp_s']!=observation.stamp_s:raise ValueError('native acquisition mismatch')
    model=NativeKinematicModel(json.loads((source/'native_model.json').read_text()))
    model=model.recalibrated(state['body_poses'],state['joint_positions'],state.get('native_platform_state'))
    scene=json.loads((source/'runtime_scene.json').read_text())
    original=OriginalReference(json.loads((source/'original_task.json').read_text()),scene['subject']['position'])
    intrinsics=camera_intrinsics(json.loads((directory/'sensor_mounts.json').read_text())['camera'])
    specification=IntentSpec(original,scene['subject'],intrinsics)
    observed=camera_scene(snapshot,observation.stamp_s,start+8.)
    collision=dict(observed,observation_stamp=observed['observation_stamp']-start)
    view=visibility_scene(snapshot,observation.stamp_s,start+8.)
    view=dict(view,observation_stamp=view['observation_stamp']-start)
    budget=native_search_budget()
    activation=max(0.,observation.stamp_s-start+budget.activation_lead_s)
    result=dict(source=str(directory),physical_execution=False,semantic_selection=False,
        diagnostic_search_family=family,
        source_observation_id=observation.id,source_observation_stamp_s=observation.stamp_s,
        original_start_s=start,activation_s=activation,camera_screens=[],candidates=[])
    begun=time.monotonic();deadline=begun+budget.candidate_wall_budget_s;ordered=[]
    attachment_screen=CameraAttachmentScreen(model,observed,start,max(0.,observation.stamp_s-start),budget.prediction_step_s)
    for path in candidate_paths(original,family,'explore',activation):
        score,_=evaluate_path(path,collision,0.,8.,.06)
        intent=intent_prediction(specification,path,view,0.,step=.1)
        accepted=bool(intent['geometric_complete'] and intent['motion_trend']['pass_check'] and intent['camera_limits_pass'])
        attached_clearance=attachment_screen.clearance(path)
        result['camera_screens'].append(dict(path_id=path_hash(path),values=path.values.tolist(),
            program=getattr(path,'program',None),
            admitted_to_native_screen=accepted,camera=score.to_dict(),intent=intent,
            rigid_camera_attachment_clearance_m=attached_clearance))
        if accepted:ordered.append((attachment_screen.rank(attached_clearance,score),path,intent))
    result['camera_screen_wall_s']=time.monotonic()-begun
    ordered.sort(key=lambda e:e[0])
    for _,path,intent in ordered[:budget.max_rollouts]:
        try:
            report=lift(path,model.document,observed,start_s=start,
                progress_start_s=max(0.,observation.stamp_s-start),
                relaxed=directory.name.startswith(('g1_','go2_')),
                step=budget.prediction_step_s,deadline=deadline)
            report['cinematic_prediction']=intent
            result['candidates'].append(card(path,report,observation,start))
            if report['physical_valid']:break
        except TimeoutError as error:
            result['budget_error']=str(error);break
    result['wall_s']=time.monotonic()-begun
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shot-time',type=float,help='earliest acquisition relative to the recorded shot start')
    parser.add_argument('--family',choices=['framing','view','joint'],default='joint')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    directories=[args.source] if (args.source/'avoidance').is_dir() else sorted(
        d for d in args.source.iterdir() if (d/'avoidance/native_states.jsonl').is_file())
    results=[]
    for directory in directories:
        try:result=screen(directory,args.shot_time,args.family)
        except Exception as error:result=dict(source=str(directory),error=f'{type(error).__name__}: {error}')
        results.append(result)
        args.output.write_text(json.dumps(native(results),indent=2)+'\n')
        print(json.dumps(dict(source=str(directory),error=result.get('error'),
            admitted=sum(c['admitted_to_native_screen'] for c in result.get('camera_screens',[])),
            native_valid=sum(c['physical_valid'] for c in result.get('candidates',[])),
            budget_error=result.get('budget_error'),wall_s=result.get('wall_s'))),flush=True)


if __name__=='__main__':
    main()
