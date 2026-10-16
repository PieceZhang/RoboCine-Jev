#!/usr/bin/env python3
"""Matched-camera geometry on/off installation and CPU WBC integration fixture.

Uses a distant synthetic obstacle and deterministic candidate selection. This is
an interface/priority experiment, NOT Jev performance or dynamic avoidance proof.
"""
import argparse
from concurrent.futures import Future
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.candidates import Approval,SearchBudget,path_hash
from jev_agent_wbc.contracts import digest
from jev_agent_wbc.engine import CinematicEngine
from jev_agent_wbc.evaluation import save_json,trajectory_summary
from jev_agent_wbc.kc_svsdf import advance
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.task_space_planning import build,verify
from jev_agent_wbc.whole_body_qp import Limits,whole_body_kinematics
from jev_agent_wbc.representation import Request,Observation
from jev_agent_wbc.representation_jvs import JVSGeometryBackend
from jev_agent_wbc.provenance import source_manifest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=('local','base'),default='base')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    runtime=json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'scenarios/movein/task.json',runtime['subject']['position'])
    initial=np.array(runtime['initial_state']);z=runtime['base_z']
    scene=dict(format='jvs_lidar_surfels/v1',observation_stamp=0.,surfel_centers=[[8.,8.,8.]],
        surfel_radius=.1,surfel_velocities=[[0.,0.,0.]],minimum_clearance_m=.025,subject_model=runtime['subject'])
    budget=SearchBudget(activation_lead_s=1.5,max_proposals=12,max_rollouts=3)
    result=build(original,original,'view',scene,initial,np.zeros(8),0.,z,Limits(),
        budget=budget,search_config=args.profile,measured_control=np.zeros(8))
    candidates=[c for c in result.get('candidates',[]) if c.path.program['route']!='continuation']
    if not candidates:
        from jev_agent_wbc.intent_paths import IntentPath
        from jev_agent_wbc.task_space_planning import screen,camera_scene
        from jev_agent_wbc.kinematic_feasibility import KinematicSearch
        from jev_agent_wbc.camera_intent import IntentSpec
        from jev_agent_wbc.candidates import Candidate
        from jev_agent_wbc.representation_jvs import candidate_geometry
        path=IntentPath(original,original,1.5,dict(route='above',amplitude_m=.03,extension_s=0.))
        search=KinematicSearch(initial,np.zeros(8),0.,z,Limits(),scene=scene,search_config=args.profile)
        checked=screen(path,search,IntentSpec(original,runtime['subject']),camera_scene(scene,0.,8.),budget,())
        if checked['error']:
            raise RuntimeError('explicit fixture path not feasible: '+checked['error'])
        candidate=Candidate(path,checked['value'],dict(intent=checked['metrics'],search_config=args.profile,
            source='explicit_geometric_fixture'),0.,'view',kinematics=checked['kinematics'])
        backend=JVSGeometryBackend();obs=backend.observe(initial,0.,base_z=z)
        candidate.consequences['geometry']=dict(candidate_geometry(candidate,backend,obs,z),box_budget=3)
    else:
        candidate=candidates[0]
    save_json(args.output/'candidate.json',candidate.card())
    save_json(args.output/'provenance.json',source_manifest(ROOT,dict(fixture='matched_camera_geometry_guidance',profile=args.profile)))
    results=[]
    # Use one exact camera path and witness for both executions.
    for enabled in (False,True):
        c=deepcopy(candidate)
        if not enabled:
            c.consequences['geometry']['request']['pose_preferences']=[]
            c.consequences['geometry']['request']['occupancy_preferences']=[]
        approval=Approval(c.id,digest(c.card()),path_hash(original),original.sha256,None,0)
        checked=verify(c,approval,original,scene,initial,np.zeros(8),0.,z,Limits(),budget=budget,measured_control=np.zeros(8))
        if checked['error']:raise RuntimeError(checked['error'])
        folder=args.output/('on' if enabled else 'off')
        engine=CinematicEngine(original,scene,folder,mode='rule',subject_model=runtime['subject'],
            budget=budget,geometry_preferences=enabled,decision_cooldown_s=1e9)
        # Exercise the real installation branch with a completed numerical job.
        # No waiting/clock race is being measured by this deterministic fixture.
        engine.math_future=Future();engine.math_future.set_result(checked);engine.math_kind='verify'
        engine.pipeline_started_wall=time.monotonic();engine.last_report_s=engine.last_attempt_s=0.
        state=initial.copy();rows=[];failure='';active_count=0
        geometry=candidate.consequences['geometry'];backend=JVSGeometryBackend()
        common=backend.compile(Request.from_dict(geometry['request']),Observation.from_dict(geometry['observation']),candidate.path)
        residuals=[];start=time.monotonic()
        try:
            for i in range(421):
                t=i*.02;observed=dict(scene,observation_stamp=t)
                u,info=engine.step(state,observed,t,.02,base_z=z,commanded_q=state[3:],measured_control=engine.wbc.previous.copy())
                active_count+=bool(info.get('tpr_report',{}).get('geometry_preference_active'))
                p,r,_=whole_body_kinematics(state,z)
                truth=float(DynamicCollision(observed,t).clearance(state,z).min())
                rows.append(dict(t=t,state=state.tolist(),command=u.tolist(),camera_position=p.tolist(),
                    camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),truth_clearance_m=truth,info=info))
                if i%25==0 and candidate.path.activation_s<=t<=candidate.path.duration:
                    residuals.append(common.feedback(backend.observe(state,t,base_z=z),'measured'))
                if i<420:state=advance(state,u,.02)
        except Exception as error:failure=f'{type(error).__name__}: {error}'
        finally:
            summary=trajectory_summary(rows,original,engine.summary(),failure)
            summary.update(fixture='deterministic_distant_obstacle_same_camera_path',preferences_enabled=enabled,
                geometry_active_samples=active_count,camera_path_id=candidate.id,wall_s=time.monotonic()-start,
                mean_common_preference_residual=float(np.mean([r['preference_residual_norm'] for r in residuals])) if residuals else None)
            save_json(folder/'summary.json',summary);save_json(folder/'geometry_results.json',residuals)
            save_json(folder/'telemetry.json',rows);engine.close();results.append(summary)
        print(json.dumps({k:summary[k] for k in ('preferences_enabled','completed','failure','installed_plans','geometry_active_samples','mean_common_preference_residual')}),flush=True)
    save_json(args.output/'runs.json',results)
    return int(not all(r['completed'] and r['installed_plans']==1 for r in results))


if __name__=='__main__':raise SystemExit(main())
