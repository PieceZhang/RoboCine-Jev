"""Independent ideal task-space camera, causal sphere observations, no robot state.

CPU diagnostic only: camera pose equals the active reference exactly. A spherical
camera body is audited independently; collision observations never deflect motion.
The same repair family, two Jev choices and exact-path binding are retained.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .candidates import Candidate, SearchBudget, path_hash, propose_paths
from .contracts import Validation, native, digest
from .evaluation import save_json, EVALUATION_PROTOCOL, smoothness
from .jev import AsyncJev, CandidateDecider, SystemOneClient, PhaseAwareDecider
from .perception import visual_consequences
from .provenance import source_manifest
from .reference import OriginalReference
from .scenarios import RuntimeScene, NAMES
from .sensors import RAYS, sphere_raycast


class CameraObserver:
    """Single-sphere fixture observer fitted from current rays, without future truth.

    No obstacle IDs, velocities or keyframes cross the observation boundary.
    Constant velocity is estimated from the past 0.8 s of fitted centers.
    """
    def __init__(self):
        self.history = deque(maxlen=9)

    def scan(self, runtime, position, t):
        if len(runtime.document['obstacles']) != 1:
            raise ValueError('camera diagnostic observer supports exactly one sphere')
        points = sphere_raycast(position, RAYS, runtime.obstacles(t))
        if len(points) >= 6:
            fit, _, rank, _ = np.linalg.lstsq(
                np.column_stack([2*points, np.ones(len(points))]), (points*points).sum(axis=1), rcond=None)
            radius2 = fit[3] + fit[:3]@fit[:3]
            if rank == 4 and radius2 > 0:
                self.history.append((t, fit[:3], np.sqrt(radius2)))
        if not self.history:
            return dict(observation_stamp=t, valid=False, centers=[], radii=[], velocities=[])
        stamp, center, radius = self.history[-1]
        times = np.array([x[0] for x in self.history])
        centers = np.array([x[1] for x in self.history])
        velocity = (np.linalg.lstsq(np.column_stack([times-times.mean(), np.ones(len(times))]),
                    centers, rcond=None)[0][0] if len(times) >= 2 else np.zeros(3))
        return dict(observation_stamp=stamp, valid=t-stamp <= .3,
            centers=[center.tolist()], radii=[float(radius)], velocities=[velocity.tolist()],
            source='current_camera_rays_sphere_fit_past_centroid_velocity', returns=len(points))


def predicted_clearances(positions, times, scene, radius):
    if not scene.get('valid') or not len(scene['centers']):
        return np.full(len(times), -1.)  # unknown cannot pass screening
    centers = np.asarray(scene['centers'])[None] + (
        np.asarray(times)[:, None, None]-scene['observation_stamp'])*np.asarray(scene['velocities'])[None]
    return (np.linalg.norm(np.asarray(positions)[:, None]-centers, axis=2)
            -np.asarray(scene['radii'])[None]
            -np.maximum(0.,np.asarray(times)-scene['observation_stamp'])[:,None]
             *np.asarray(scene.get('radius_rates',np.zeros(len(scene['radii']))))[None]-radius).min(axis=1)


def evaluate_path(path, scene, t, end, radius, step=.02, visual=False):
    times = np.unique(np.r_[np.arange(t, end, step), end])
    poses = [path.sample(float(at)) for at in times]
    positions = np.array([p for p, _, _ in poses])
    clearance = predicted_clearances(positions, times, scene, radius)
    # Exact closest approach on the piecewise-linear relative path, in addition
    # to samples. This prevents a fast crossing between two sample endpoints.
    swept = float(clearance.min())
    if scene.get('valid') and len(times) > 1:
        for center, velocity, obstacle_radius, rate in zip(scene['centers'],scene['velocities'],scene['radii'],
                scene.get('radius_rates',np.zeros(len(scene['radii'])))):
            relative = positions - (np.asarray(center)+(times-scene['observation_stamp'])[:,None]*velocity)
            delta = np.diff(relative, axis=0)
            u = np.clip(-np.sum(relative[:-1]*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-20),0,1)
            swept = min(swept,float(np.min(np.linalg.norm(relative[:-1]+u[:,None]*delta,axis=1)-obstacle_radius-radius
                -rate*np.maximum(0.,times[1:]-scene['observation_stamp']))))
    original = getattr(path,'original',path)
    nominal = np.array([original.sample(float(at))[0] for at in times])
    value = Validation(swept >= .025, 'ideal_camera_sampled_sweep', min_clearance_m=swept,
        joint_margin_rad=None, cost=float(np.mean(np.sum((positions-nominal)**2,axis=1))),
        samples=len(times), next_phase_checked=end >= original.duration)
    if visual:
        # Visual cards need only 10 Hz; collision checks above always use 50 Hz.
        indices = sorted(set([*range(0,len(times),max(1,round(.1/step))),len(times)-1]))
        value.camera_trace = [dict(t=float(times[i]), position=poses[i][0].tolist(),
            quaternion_xyzw=Rotation.from_matrix(poses[i][1]).as_quat().tolist(),
            joint_margin_rad=None) for i in indices]
    risk = next((float(at) for at,c in zip(times,clearance) if c < .105),None)
    return value, risk


def camera_candidates(original, active, strategy, scene, t, budget, radius):
    start = time.monotonic()
    if min(original.duration,t+budget.max_horizon_s)-t-budget.activation_lead_s < .6:
        return dict(error='insufficient_time_for_exact_plan_activation')
    caps = dict(radial_m=1.5,lateral_m=1.5,height_m=.65,yaw_rad=np.pi,pitch_rad=np.pi/2,roll_rad=np.pi/2)
    paths = propose_paths(original,active,strategy,t,budget,caps)
    end = min(original.duration,t+budget.max_horizon_s)
    ranked, evaluations = [], []
    for path in paths[:budget.max_proposals]:
        value, _ = evaluate_path(path,scene,t,end,radius)
        evaluations.append(dict(id=path_hash(path),coarse=value.to_dict()))
        if value.feasible:
            ranked.append((value.cost,path))
    candidates = []
    fy=.5/np.tan(np.deg2rad(27.5))
    for _, path in sorted(ranked,key=lambda pair:pair[0])[:budget.max_rollouts]:
        value, _ = evaluate_path(path,scene,t,end,radius,visual=True)
        consequences = visual_consequences(original,value.camera_trace,intrinsics=[fy*288/384,fy,.5,.5])
        consequences['source']='ideal_task_space_camera_projection_no_robot'
        candidates.append(Candidate(path,value,consequences,t,strategy))
        detail=value.to_dict();detail.pop('camera_trace',None)
        evaluations.append(dict(id=path_hash(path),full=detail))
    return dict(candidates=candidates,evaluations=evaluations,wall_s=time.monotonic()-start,
                error='' if candidates else 'no_camera_candidate_passed_collision_screening')


class FloatingAgent:
    def __init__(self, original, output, mode, budget, radius, model, api_url):
        self.original=self.active=original
        self.output=Path(output);self.mode=mode;self.budget=budget;self.radius=radius;self.model=model
        decider=CandidateDecider(SystemOneClient(url=api_url,model=model,journal=self.output/'jev_api.jsonl')) if mode=='jev' else PhaseAwareDecider()
        self.jev=AsyncJev(decider)
        self.pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'))
        self.future=None;self.pending={};self.last_attempt=-np.inf;self.report={};self.history=[]
        self.installed=0;self.started=None;self.outcomes=[];self.failures=[];self.feasible=0;self.attempts=0
        self.events=(self.output/'events.jsonl').open('w')
        self.write('initialized',0.,budget=asdict(budget),backend='ideal_floating_camera',camera_radius_m=radius,
            robot_state_used=False,actuator_constraints=False,obstacle_avoidance_enabled=False)

    def write(self,event,t,**kwargs):
        self.events.write(json.dumps(native(dict(event=event,time_s=t,**kwargs)),ensure_ascii=False,allow_nan=False)+'\n');self.events.flush()

    def reject(self,reason,t):
        self.write('proposal_rejected',t,error=reason)
        self.failures.append(dict(error=reason,time_s=t));self.last_attempt=t
        if self.started is not None:
            self.outcomes.append(dict(outcome='rejected',reason=reason,time_s=t,wall_s=time.monotonic()-self.started))
            self.started=None

    def evidence(self,t,stage):
        phase=self.original.phase(t)
        following=[p for p in self.original.task['shot_phase'] if p['start_s']>=phase['end_s']]
        return dict(stage=stage,time_s=t,duration_s=self.original.duration,phase=phase,
            next_phase=following[0] if following else None,intent=self.original.task['intent'],
            subject=self.original.task['subject'],event='difficulty',difficulty=self.report,
            active_modification=self.active is not self.original,execution_history=self.history[-4:],
            validation_feedback=self.failures[-3:],remaining_budget_s=self.original.duration-t,
            execution_model='ideal floating camera, exact target tracking; no base, joints or WBC',
            subject_evidence=dict(visual=dict(status='unknown'),geometry=dict(status='unknown')))

    def update(self,scene,t):
        if self.mode=='strict' or t>=self.original.duration:return
        if self.future is not None and self.future.done():
            result=self.future.result();self.future=None;self.attempts+=1
            if result.get('error'):
                self.reject(result['error'],t)
            else:
                self.feasible+=1
                cards=[c.card() for c in result['candidates']]
                self.write('candidate_set',t,cards=cards,evaluations=result['evaluations'],planner_wall_s=result['wall_s'])
                self.pending={c.id:c for c in result['candidates'] if t<c.path.activation_s}
                if not self.pending:self.reject('all_candidates_expired_before_selection',t)
                else:
                    evidence=dict(self.evidence(t,'selection'),candidates=[c.card() for c in self.pending.values()])
                    self.jev.submit(evidence);self.write('selection_requested',t,evidence=evidence)
        decision=self.jev.poll()
        if decision is not None:
            if decision.get('error'):self.reject('semantic: '+decision['error'],t)
            elif decision['stage']=='strategy':
                self.write('search_strategy',t,**decision)
                if decision['strategy']=='refuse':self.reject('search_refused',t)
                else:
                    self.parent=path_hash(self.active)
                    self.future=self.pool.submit(camera_candidates,self.original,self.active,decision['strategy'],scene,t,self.budget,self.radius)
            else:
                candidate=self.pending.get(decision['candidate_id']);self.pending={}
                if candidate is None:self.reject('selection_outside_pending_set',t)
                elif t>=candidate.path.activation_s:self.reject('candidate_expired_no_retiming_allowed',t)
                elif path_hash(candidate.path)!=candidate.id or path_hash(self.active)!=self.parent:self.reject('candidate_binding_changed',t)
                elif self.active is self.original and not np.any(candidate.path.values):
                    self.write('continuation_selected',t,candidate_id=candidate.id,decision=decision)
                    self.started=None;self.last_attempt=t
                else:
                    approval=dict(candidate_id=candidate.id,card_sha256=digest(candidate.card()),parent_id=self.parent)
                    self.write('candidate_approved',t,approval=approval,decision=decision)
                    refreshed,_=evaluate_path(candidate.path,scene,t,candidate.path.envelope.end_s,self.radius)
                    if not refreshed.feasible:self.reject('latest_camera_prediction_collision',t)
                    elif not scene['valid'] or t-scene['observation_stamp']>.3:self.reject('camera_observation_stale',t)
                    else:
                        self.active=candidate.path;self.installed+=1;self.last_attempt=t
                        latency=time.monotonic()-self.started
                        self.outcomes.append(dict(outcome='installed',time_s=t,wall_s=latency));self.started=None
                        self.write('plan_installed',t,candidate_id=candidate.id,approval=approval,
                            actual_parameters=self.active.values.tolist(),checks=dict(physical_valid='pass',state_fresh='pass',
                            target_authorized='pass',predicted_visual_valid='unknown',observed_visual_valid='unknown'),
                            end_to_end_wall_s=latency,latest_state_check=refreshed.to_dict())
        if (self.future is None and self.jev.future is None and t-self.last_attempt>=self.budget.decision_period_s
                and self.original.duration-t>self.budget.activation_lead_s+.8):
            value,risk=evaluate_path(self.active,scene,t,min(t+5,self.original.duration),self.radius,step=.05)
            self.last_attempt=t
            if risk is not None or not value.feasible:
                self.report=dict(issues=['predicted_camera_collision_or_low_clearance'],first_risk_s=risk,
                    min_clearance_m=value.min_clearance_m,joint_margin_rad=None,
                    local_repair_evidence='task-space translation families; no robot Jacobian')
                self.write('whole_body_difficulty',t,report=self.report,execution_model='camera_only')
                self.started=time.monotonic();evidence=self.evidence(t,'strategy')
                self.jev.submit(evidence);self.write('strategy_requested',t,evidence=evidence)

    def close(self):
        self.jev.close();self.pool.shutdown(wait=True,cancel_futures=True);self.events.close()


def run_case(case,output,mode,budget,radius,model,api_url,pace):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime=RuntimeScene(Path(case)/'runtime_scene.json')
    original=OriginalReference(Path(case)/'task.json',runtime.document['subject']['position'])
    save_json(output/'original_task.json',original.task);save_json(output/'runtime_scene.json',runtime.document)
    observer=CameraObserver()
    p0=original.sample(0.)[0]
    for t in np.linspace(-.8,0.,9):scene=observer.scan(runtime,p0,float(t))
    agent=FloatingAgent(original,output,mode,budget,radius,model,api_url)
    rows=[];failure='';begin=time.monotonic();dt=.02
    try:
        for i in range(round(original.duration/dt)+1):
            t=i*dt;before=agent.active.sample(t)[0]
            if i%5==0:scene=observer.scan(runtime,before,t)
            agent.update(scene,t)
            p,r,_=agent.active.sample(t) # Exact 1:1 pose assignment, no integration or robot FK.
            end=min(t+dt,original.duration)
            true_clearance=min(float(np.linalg.norm(agent.active.sample(float(at))[0]-obstacle['center'])-obstacle['radius']-radius)
                for at in np.linspace(t,end,9) for obstacle in runtime.obstacles(float(at)))
            op,ori,_=original.sample(t)
            info=dict(target_position=p.tolist(),position_error=0.,orientation_error=0.,joint_margin_rad=None,
                active_decision='' if agent.active is original else agent.active.envelope.decision_id,
                loop_wall_s=0.,dynamic_constraint_count=0,obstacle_avoidance_enabled=False)
            rows.append(dict(t=t,camera_position=p.tolist(),camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
                truth_clearance_m=true_clearance,info=info,observation_stamp=scene['observation_stamp']))
            if i%25==0:agent.history.append(dict(time_s=t,target_tracking_residual_m=0.,clearance_m=true_clearance,
                original_position_error_m=float(np.linalg.norm(p-op))))
            if pace>0:
                remaining=(i+1)*dt/pace-(time.monotonic()-begin)
                if remaining>0:time.sleep(remaining)
    except Exception as error:
        failure=f'{type(error).__name__}: {error}';agent.write('run_failed',len(rows)*dt,failure=failure)
    finally:
        elapsed=rows[-1]['t'] if rows else 0.
        positions=np.array([x['camera_position'] for x in rows])
        errors=[np.linalg.norm(np.asarray(x['camera_position'])-original.sample(x['t'])[0]) for x in rows]
        clearance=[x['truth_clearance_m'] for x in rows]
        completed=not failure and elapsed>=original.duration-1e-8
        safe=bool(clearance and min(clearance)>=.025)
        phase_ok=[]
        for phase in original.shots:
            matching=[x for x in rows if x['t']<=phase['end_s']+1e-8]
            row=matching[-1] if matching else None
            p,r,_=original.sample(phase['end_s'])
            phase_ok.append(bool(row and abs(row['t']-phase['end_s'])<1e-8 and
                np.linalg.norm(np.array(row['camera_position'])-p)<=.05 and
                (Rotation.from_quat(row['camera_quaternion_xyzw'])*Rotation.from_matrix(r).inv()).magnitude()<=.1))
        summary=dict(backend='ideal_floating_camera',mode=mode,semantic_model=model,completed=completed,
            safety_pass=safe,within_budget_success=completed and safe and all(phase_ok),failure=failure,
            installed_plans=agent.installed,jev_calls=agent.jev.decider.client.calls,wall_s=time.monotonic()-begin,
            elapsed_shot_s=elapsed,original_position_rms_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            min_truth_clearance_m=min(clearance) if clearance else None,collision_samples=sum(c<0 for c in clearance),
            clearance_violation_samples=sum(c<.025 for c in clearance),target_tracking_max_m=0.,target_tracking_max_rad=0.,
            camera_radius_m=radius,robot_state_used=False,obstacle_avoidance_enabled=False,
            collision_audit='ground-truth spheres sampled at 400 Hz; camera sphere; no control intervention',
            prediction_audit='50 Hz camera path with piecewise-linear relative-segment closest approach; not a continuous nonlinear certificate',
            observation_source='current ray hits and past fitted-center velocity; no future keyframes',
            decision_outcomes=agent.outcomes,original_duration_s=original.duration,
            execution_configuration=dict(pace=pace,activation_lead_s=budget.activation_lead_s),
            evaluation_protocol=EVALUATION_PROTOCOL)
        save_json(output/'summary.json',summary);save_json(output/'telemetry.json',rows);agent.close()
    print(json.dumps(dict(case=runtime.document['name'],mode=mode,completed=completed,safe=safe,installed=agent.installed,failure=failure)),flush=True)
    return summary


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',default='all');p.add_argument('--scenarios',type=Path,default=root/'scenarios')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['jev','rule','strict','both'],default='jev')
    p.add_argument('--model',default='perplexity');p.add_argument('--api-url',default='http://127.0.0.1:8000/v1/systemone')
    p.add_argument('--camera-radius',type=float,default=.06);p.add_argument('--pace',type=float,default=1.)
    p.add_argument('--activation-lead',type=float,default=1.5)
    a=p.parse_args()
    if not np.isfinite(a.camera_radius) or a.camera_radius<=0 or not np.isfinite(a.pace) or a.pace<0:p.error('invalid radius or pace')
    budget=SearchBudget(activation_lead_s=a.activation_lead)
    a.output.mkdir(parents=True,exist_ok=True)
    save_json(a.output/'provenance.json',source_manifest(root,{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
    results=[]
    for case in NAMES if a.case=='all' else [a.case]:
        if Path(case).name!=case or case in ('.','..'):p.error('invalid case name')
        for mode in ['strict','jev'] if a.mode=='both' else [a.mode]:
            results.append(run_case(a.scenarios/case,a.output/f'{case}_{mode}',mode,budget,a.camera_radius,a.model,a.api_url,a.pace))
            save_json(a.output/'runs.json',results)
    return 0 if all(x['completed'] for x in results) else 2


if __name__=='__main__':raise SystemExit(main())
