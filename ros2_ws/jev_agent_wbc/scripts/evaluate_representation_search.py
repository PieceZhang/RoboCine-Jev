#!/usr/bin/env python3
"""Matched recorded-state search diagnostic; no Agent or future obstacle truth.

Reconstructs the observed map by replaying sensor scans at the recorded robot
poses up to --time. RuntimeScene is used only by the current-time sensor, exactly
as in the CPU runner. It never supplies future obstacle keyframes to the planner.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings',type=Path,required=True)
    p.add_argument('--code',type=Path,default=root,help='package parent, optionally a saved source snapshot')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--time',type=float,default=.6)
    p.add_argument('--lead',type=float,default=.8)
    p.add_argument('--candidate-seconds',type=float,default=8.)
    p.add_argument('--configuration',choices=('local','base','timing','explore','executable'),default='base')
    p.add_argument('--case',default='all')
    a=p.parse_args();sys.path.insert(0,str(a.code.resolve()))
    import numpy as np
    from scipy.spatial.transform import Rotation
    from jev_agent_wbc.scenarios import RuntimeScene,NAMES
    from jev_agent_wbc.sensors import RuntimeLidar
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.task_space_planning import build,camera_scene
    from jev_agent_wbc.camera_intent import visibility,default_intrinsics
    from jev_agent_wbc.candidates import SearchBudget
    from jev_agent_wbc.whole_body_qp import Limits
    from jev_agent_wbc.contracts import digest
    from jev_agent_wbc.evaluation import save_json
    from jev_agent_wbc.provenance import source_manifest
    if not np.isfinite(a.time) or a.time<0:raise ValueError('nonnegative snapshot time required')
    a.output.mkdir(parents=True,exist_ok=False)
    save_json(a.output/'provenance.json',source_manifest(a.code,dict(
        recordings=str(a.recordings.resolve()),snapshot_s=a.time,activation_lead_s=a.lead,
        candidate_wall_budget_s=a.candidate_seconds,configuration=a.configuration,
        mode='matched_recorded_state_numeric_search_no_agent')))
    results=[]
    for case in NAMES if a.case=='all' else [a.case]:
        source=a.recordings/(case+'_jev')
        rows=json.loads((source/'telemetry.json').read_text())
        runtime=RuntimeScene(source/'runtime_scene.json');d=runtime.document;z=d['base_z'];lidar=RuntimeLidar()
        for t in np.linspace(-.8,0.,9):obs=lidar.synthetic_scan(runtime,np.array(d['initial_state']),z,float(t))
        history=[];selected=None
        for row_index,row in enumerate(rows):
            t=row['t']
            if t>a.time+1e-8:break
            if round(t/.02)%5==0:obs=lidar.synthetic_scan(runtime,np.array(row['state']),z,t)
            obs['subject_model']=d['subject']
            r=Rotation.from_quat(row['camera_quaternion_xyzw']).as_matrix()
            history.append(dict(t=t,camera_position=row['camera_position'],camera_quaternion_xyzw=row['camera_quaternion_xyzw'],
                progress_s=t,visual=visibility(np.array(row['camera_position']),r,camera_scene(obs,t,t),t,d['subject'],default_intrinsics())))
            selected=row;selected_index=row_index
        if selected is None:raise ValueError('recording has no snapshot')
        # This diagnostic requires an original-camera prefix, not a hidden
        # replacement of previously modified targets or motion clocks.
        for event in map(json.loads,(source/'events.jsonl').read_text().splitlines()):
            if event['event']=='plan_installed' and event['time_s']<=selected['t']:
                raise ValueError('choose a snapshot before the first installed modification')
        original=OriginalReference(source/'original_task.json',d['subject']['position'])
        state=np.array(selected['state']);velocity=(np.zeros(8) if selected_index==0 else np.array(rows[selected_index-1]['command']));stamp=selected['t']
        fixture=dict(state=state.tolist(),velocity=velocity.tolist(),scene=obs,history=history,
                     task=original.task,stamp_s=stamp,base_z=z)
        save_json(a.output/(case+'_fixture.json'),fixture)
        start=time.monotonic()
        result=build(original,original,'joint',obs,state,velocity,stamp,z,Limits(),
            history=history,measured_control=velocity,commanded_q=state[3:],
            search_config=a.configuration,budget=SearchBudget(max_proposals=16,activation_lead_s=a.lead,
                                                            candidate_wall_budget_s=a.candidate_seconds))
        candidates=result.pop('candidates',[])
        result.pop('kinematic_prior',None)
        result.update(case=case,fixture_sha256=digest(fixture),wall_s=time.monotonic()-start,
            candidate_count=len(candidates),complete_intent_candidates=sum(c.consequences['intent']['intent_complete'] for c in candidates),
            cards=[c.card() for c in candidates])
        save_json(a.output/(case+'_search.json'),result)
        results.append({k:v for k,v in result.items() if k not in ('cards','evaluations')})
        save_json(a.output/'summary.json',results)
        print(json.dumps(results[-1]),flush=True)


if __name__=='__main__':main()
