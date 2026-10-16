#!/usr/bin/env python3
"""Read-only re-evaluation of recorded whole-body demos; no planner or API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_state_search import audit
from jev_agent_wbc.camera_intent import IntentSpec, default_intrinsics, progress_at, visibility
from jev_agent_wbc.candidates import path_hash, restore_path
from jev_agent_wbc.evaluation import save_json
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.sensors import RuntimeLidar
from jev_agent_wbc.shot_acceptance import ACCEPTANCE_PROTOCOL, evaluate_shot
from jev_agent_wbc.task_space_planning import camera_scene


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def reconstruct_history(folder, rows, original, runtime):
    """Replay sensor observations at recorded states, without integrating motion."""
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    cards={c['id']:c for e in events for c in e.get('cards',[])}
    schedule=[]
    for e in events:
        if e['event'] not in ('plan_installed','target_replayed'):continue
        p=restore_path(original,cards[e['candidate_id']]['trajectory'])
        if path_hash(p)!=e['candidate_id']:raise ValueError('recorded trajectory hash mismatch')
        schedule.append((e['time_s'],p))
    schedule.sort(key=lambda e:e[0]);active=original;next_plan=0
    if (folder/'camera_history.json').is_file():
        end=schedule[-1][1].duration if schedule else original.duration
        return json.loads((folder/'camera_history.json').read_text()),end,'saved_camera_history'
    lidar=RuntimeLidar();z=runtime.document['base_z'];intrinsics=default_intrinsics()
    for t in np.linspace(-.8,0.,9):
        observed=lidar.synthetic_scan(runtime,np.asarray(runtime.document['initial_state']),z,float(t))
    history=[]
    for i,row in enumerate(rows):
        t=row['t']
        if i%5==0:observed=lidar.synthetic_scan(runtime,np.asarray(row['state']),z,t)
        while next_plan<len(schedule) and schedule[next_plan][0]<=t+1e-9:
            active=schedule[next_plan][1];next_plan+=1
        scene=camera_scene(dict(observed,subject_model=runtime.document['subject'],intrinsics=intrinsics),t,active.duration)
        p=np.asarray(row['camera_position']);r=Rotation.from_quat(row['camera_quaternion_xyzw']).as_matrix()
        history.append(dict(t=t,camera_position=p.tolist(),camera_quaternion_xyzw=row['camera_quaternion_xyzw'],
            progress_s=progress_at(active,t),visual=visibility(p,r,scene,t,runtime.document['subject'],intrinsics)))
    return history,active.duration,'recorded_states_and_recorded_plans_sensor_reconstruction'


def inspect(folder):
    rows=json.loads((folder/'telemetry.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    runtime=RuntimeScene(folder/'runtime_scene.json')
    original=OriginalReference(folder/'original_task.json',runtime.document['subject']['position'])
    history,motion_end,source=reconstruct_history(folder,rows,original,runtime)
    t=[r['t'] for r in history];p=[r['camera_position'] for r in history]
    rotations=[Rotation.from_quat(r['camera_quaternion_xyzw']).as_matrix() for r in history]
    visual=[r['visual'] for r in history];progress=[r['progress_s'] for r in history]
    legacy=IntentSpec(original,runtime.document['subject']).score(t,p,rotations,visual,progress)
    recorded=summary['executed_camera_intent']
    keys=['intent_complete','effective_coverage','longest_occlusion_s','in_frame_min','progress_s','phase_order_valid']
    differences={k:dict(recorded=recorded[k],recomputed=legacy[k]) for k in keys
                 if not np.isclose(recorded[k],legacy[k],rtol=0.,atol=1e-8)}
    if differences:raise ValueError(f'legacy reconstruction mismatch: {differences}')
    physical=audit(folder)
    safety_fields=['independent_safety_pass','sweep_scope','min_truth_swept_clearance_m',
        'min_self_swept_clearance_m','min_joint_margin_rad','min_scaled_singular_value',
        'max_velocity_ratio','max_acceleration_ratio','max_wheel_velocity_ratio','max_wheel_acceleration_ratio']
    result=evaluate_shot(original,runtime.document['subject'],t,p,rotations,visual,progress,
        safety_pass=physical['independent_safety_pass'],execution_ok=not summary.get('failure'),
        motion_end_s=motion_end,legacy_intent=legacy,
        safety_source='fresh_independent_CPU_whole_body_interval_sweep_and_actuator_audit')
    return dict(case=runtime.document['name'],source_folder=str(folder.resolve()),
        old_intent_success=bool(summary.get('legacy_reference_intent_success',summary['intent_success'])),
        new_acceptance=result,independent_safety={k:physical[k] for k in safety_fields},
        reconstruction=dict(source=source,legacy_metrics_match=True,checked_keys=keys),
        trajectory_modified=False,new_model_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--video',type=Path)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.runs.resolve()):
        parser.error('output must be outside the recorded run directory')
    folders=sorted(p.parent for p in args.runs.rglob('telemetry.json') if (p.parent/'summary.json').is_file())
    if not folders:parser.error('no recorded demos found')
    # Hash every original run input/output and the video, not only camera samples.
    files=sorted({p for f in folders for p in f.iterdir() if p.is_file()})
    if args.video:files.append(args.video)
    before={str(p.resolve()):sha256(p) for p in files}
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/'input_hashes_before.json',before)
    save_json(args.output/'acceptance_protocol.json',ACCEPTANCE_PROTOCOL)
    results=[]
    for folder in folders:
        result=inspect(folder);results.append(result)
        save_json(args.output/(result['case']+'.json'),result)
        print(result['case'],result['new_acceptance']['status'],result['new_acceptance']['failed_checks'],flush=True)
    after={str(p.resolve()):sha256(p) for p in files}
    if before!=after:raise RuntimeError('original artifacts changed during evaluation')
    summary=dict(protocol_version=ACCEPTANCE_PROTOCOL['version'],cases=len(results),
        accepted=sum(r['new_acceptance']['accepted'] for r in results),
        old_reference_intent_passed=sum(r['old_intent_success'] for r in results),
        per_check_passed={k:sum(r['new_acceptance']['checks'][k]['status']=='pass' for r in results)
                          for k in ACCEPTANCE_PROTOCOL['primary_checks']},
        original_artifacts_unchanged=True,hashed_files=len(files),new_model_calls=0,
        results=results)
    save_json(args.output/'results.json',summary)
    save_json(args.output/'input_hashes_after.json',after)
    source_files=[Path(__file__),ROOT/'scripts/audit_state_search.py',*sorted((ROOT/'jev_agent_wbc').glob('*.py'))]
    save_json(args.output/'evaluator_source_hashes.json',{str(p.resolve()):sha256(p) for p in source_files})
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
