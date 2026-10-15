"""Independent ideal task-space camera, causal sphere observations, no robot state.

CPU diagnostic only: camera pose equals the active reference exactly. A spherical
camera body is audited independently; collision observations never deflect motion.
The agent proposes motion programs and selects by visible intent completion.
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

from .candidates import Candidate, SearchBudget, path_hash
from .camera_intent import (IntentSpec, POLICY, default_intrinsics, progress_at,
                            selection_key, visibility, subject_clearances)
from .intent_paths import propose_intent_paths, positions_at, pose_at
from .contracts import Validation, native, digest
from .evaluation import save_json, smoothness
from .jev import AsyncJev, CandidateDecider, SystemOneClient, PhaseAwareDecider
from .provenance import source_manifest
from .reference import OriginalReference
from .scenarios import RuntimeScene, NAMES
from .sensors import sphere_raycast
from .subject_state import (SubjectStateObserver, simulated_subject_returns, frame_state, frame_error, subject_observation_error)


# A floating camera has no robot-mounted scanner blind zone. This fixed
# omnidirectional CPU sensor is independent of obstacle positions and keyframes.
_azimuth, _elevation = np.meshgrid(np.linspace(-np.pi,np.pi,360,endpoint=False),
                                  np.linspace(-1.55,1.55,91))
CAMERA_RAYS = np.column_stack([(np.cos(_elevation)*np.cos(_azimuth)).ravel(),
    (np.cos(_elevation)*np.sin(_azimuth)).ravel(),np.sin(_elevation).ravel()])


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
        points = sphere_raycast(position, CAMERA_RAYS, runtime.obstacles(t))
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
    if not scene.get('valid'):
        return np.full(len(times), -1.)  # unknown cannot pass screening
    if not len(scene['centers']):
        return subject_clearances(positions,scene.get('subject_model'),radius)
    centers = np.asarray(scene['centers'])[None] + (
        np.asarray(times)[:, None, None]-scene['observation_stamp'])*np.asarray(scene['velocities'])[None]
    clearance = (np.linalg.norm(np.asarray(positions)[:, None]-centers, axis=2)
            -np.asarray(scene['radii'])[None]
            -np.maximum(0.,np.asarray(times)-scene['observation_stamp'])[:,None]
             *np.asarray(scene.get('radius_rates',np.zeros(len(scene['radii']))))[None]-radius).min(axis=1)
    return np.minimum(clearance,subject_clearances(positions,scene.get('subject_model'),radius))


def evaluate_path(path, scene, t, end, radius, step=.02, visual=False):
    times = np.r_[np.arange(t, end-1e-8, step), end]
    positions = positions_at(path,times)
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
    velocities = np.gradient(positions, times, axis=0) if len(times)>2 else np.zeros_like(positions)
    accelerations = np.gradient(velocities, times, axis=0) if len(times)>2 else np.zeros_like(positions)
    value = Validation(swept >= .025, 'ideal_camera_sampled_sweep', min_clearance_m=swept,
        joint_margin_rad=None, cost=float(np.mean(np.sum(accelerations**2,axis=1))),
        samples=len(times), next_phase_checked=end >= getattr(path,'duration',original.duration))
    if visual:
        # Visual cards need only 10 Hz; collision checks above always use 50 Hz.
        indices = sorted(set([*range(0,len(times),max(1,round(.1/step))),len(times)-1]))
        value.camera_trace = [dict(t=float(times[i]), position=positions[i].tolist(),
            quaternion_xyzw=Rotation.from_matrix(pose_at(path,float(times[i]))[1]).as_quat().tolist(),
            joint_margin_rad=None) for i in indices]
    risk = next((float(at) for at,c in zip(times,clearance) if c < .105),None)
    return value, risk


def intent_prediction(spec, path, scene, t, history=(), step=.05):
    prediction_end = path.duration+POLICY.terminal_hold_s
    # Include both confirmation-window boundaries explicitly. A monitoring
    # grid offset (e.g. t=2.78, step=.1) must not turn a full .4 s hold into .32 s.
    boundaries=[prediction_end]+([path.duration] if path.duration>=t else [])
    times = np.unique(np.round(np.r_[np.arange(t,prediction_end-1e-8,step),boundaries],9))
    poses = [pose_at(path,float(at)) for at in times]
    previous = [row for row in history if row['t'] < t-1e-8]
    positions = [row['camera_position'] for row in previous]+[pose[0] for pose in poses]
    rotations = [Rotation.from_quat(row['camera_quaternion_xyzw']).as_matrix() for row in previous]+[pose[1] for pose in poses]
    progress = [row['progress_s'] for row in previous]+[progress_at(path, at) for at in times]
    measurements = [row['visual'] for row in previous]+[
        visibility(pose[0], pose[1], scene, float(at), spec.model, spec.intrinsics) for at, pose in zip(times, poses)]
    all_times = np.r_[[row['t'] for row in previous], times]
    result = spec.score(all_times, positions, rotations, measurements, progress)
    velocity = np.gradient(np.array([pose[0] for pose in poses]), times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)
    angular = [(Rotation.from_matrix(b[1]@a[1].T).magnitude()/(tb-ta))
               for a, b, ta, tb in zip(poses[:-1], poses[1:], times[:-1], times[1:])]
    result['max_speed_m_s'] = float(np.linalg.norm(velocity, axis=1).max())
    result['max_acceleration_m_s2'] = float(np.linalg.norm(acceleration, axis=1).max())
    result['max_angular_speed_rad_s'] = float(max(angular, default=0.))
    result['camera_limits_pass'] = (result['max_speed_m_s'] <= POLICY.max_speed_m_s and
        result['max_acceleration_m_s2'] <= POLICY.max_acceleration_m_s2 and
        result['max_angular_speed_rad_s'] <= POLICY.max_angular_speed_rad_s)
    return result


def camera_candidates(original, active, strategy, scene, t, budget, radius, history=(), allow_time_extension=True,
                      kinematic_search=None, numerical_gate=None, search_config=None, proposal_transform=None):
    start = time.monotonic()
    if active.duration-t-budget.activation_lead_s < .3:
        return dict(error='insufficient_time_for_exact_plan_activation')
    observation = scene.get('subject_observation')
    observation_error=subject_observation_error(observation,original,t)
    if observation_error:return dict(error=observation_error)
    spec = IntentSpec(original, scene.get('subject_model'), scene.get('intrinsics'))
    paths = propose_intent_paths(original,active,strategy,t,budget,spec.model,spec.intrinsics,
                                 allow_time_extension,scene,kinematic_search,search_config)
    ranked, evaluations = [], []
    for path in paths:
        if numerical_gate is not None and evaluations and time.monotonic()-start>=budget.candidate_wall_budget_s:
            break
        if proposal_transform is not None:
            proposal=proposal_transform(path)
            if proposal.get('error'):
                evaluations.append(dict(id=path_hash(path),program=path.program,
                    physical=Validation(False,'projection_not_verified').to_dict(),
                    rejected=proposal['error'],failure_class='unverified_numerical_search',
                    kinematics=proposal.get('summary')))
                continue
            path=proposal['path']
        value, _ = evaluate_path(path,scene,t,path.duration+POLICY.terminal_hold_s,radius)
        detail = dict(id=path_hash(path), program=path.program, physical=value.to_dict())
        if value.feasible:
            metrics = intent_prediction(spec,path,scene,t,history)
            detail['intent'] = metrics
            if metrics['camera_limits_pass']:
                gate = None if numerical_gate is None else numerical_gate(path)
                if gate is not None:
                    detail['kinematics'] = gate['summary']
                    if gate.get('error'):
                        detail['rejected'] = gate['error']
                        detail['failure_class'] = gate['kinematics'].get('failure_class','unverified_numerical_search')
                        evaluations.append(detail)
                        continue
                    value, metrics = gate['value'], gate['metrics']
                    detail.update(physical=value.to_dict(), intent=metrics)
                detail['intent_violations'] = metrics.get('intent_violations', [])
                consequences = dict(intent=metrics, source='floating_camera_calibrated_geometry_no_wbc',
                    summary=metrics, prediction_interval_s=[t,path.duration],search_config=search_config)
                if gate is not None:
                    consequences['source'] = 'measured_seed_camera_with_sampled_kinematic_gate_no_wbc'
                candidate = Candidate(path,value,consequences,t,strategy)
                if gate is not None:
                    candidate.kinematics = gate['kinematics']
                ranked.append((selection_key(metrics,value.cost),candidate))
            else:
                detail['rejected'] = ('camera_limits' if not metrics['camera_limits_pass'] else
                    'terminal_frame_mismatch' if not metrics['terminal_frame']['pass_check'] else
                    'motion_trend_violation' if not metrics['motion_trend']['pass_check'] else 'incomplete_visible_intent')
                detail['failure_class'] = 'physical_limit'
        else:
            detail['rejected'] = 'camera_collision'
            detail['failure_class'] = 'physical_collision' if scene.get('valid') else 'unverified_observation'
        evaluations.append(detail)
        if numerical_gate is not None and ranked and time.monotonic()-start>=budget.candidate_wall_budget_s:
            # Publish an already certified candidate before its exact
            # activation expires. Additional posture branches must not hold
            # a usable result hostage. Latest-state verification still runs.
            break
    candidates = [c for _,c in sorted(ranked,key=lambda pair:pair[0])[:budget.max_rollouts]]
    return dict(candidates=candidates,evaluations=evaluations,wall_s=time.monotonic()-start,
                search_exhausted=len(evaluations)==len(paths), untested_proposals=len(paths)-len(evaluations),
                error='' if candidates else 'no_verified_executable_candidate')


class FloatingAgent:
    def __init__(self, original, output, mode, budget, radius, model, api_url):
        self.original=self.active=original
        self.output=Path(output);self.mode=mode;self.budget=budget;self.radius=radius;self.model=model
        decider=CandidateDecider(SystemOneClient(url=api_url,model=model,journal=self.output/'jev_api.jsonl')) if mode=='jev' else PhaseAwareDecider()
        self.jev=AsyncJev(decider)
        self.pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'))
        self.future=None;self.pending={};self.last_attempt=-np.inf;self.report={};self.history=[]
        self.executed=[];self.spec=None;self.subject_observation={}
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
        phase=self.original.phase(progress_at(self.active,t))
        following=[p for p in self.original.task['shot_phase'] if p['start_s']>=phase['end_s']]
        return dict(stage=stage,time_s=t,duration_s=self.active.duration,phase=phase,
            next_phase=following[0] if following else None,intent=self.original.task['intent'],
            subject=self.original.task['subject'],event='difficulty',difficulty=self.report,
            active_modification=self.active is not self.original,execution_history=self.history[-4:],
            validation_feedback=self.failures[-3:],remaining_budget_s=POLICY.max_duration_factor*self.original.duration-t,
            objective='visible_intent_completion',intent_specification=self.spec.to_dict() if self.spec else None,
            motion_progress_s=progress_at(self.active,t),
            execution_model='ideal floating camera, exact target tracking; no base, joints or WBC',
            subject_evidence=self.subject_observation,
            subject_state_comparison=self.subject_comparison(t))

    def subject_comparison(self,t):
        if self.spec is None:return dict(status='unknown')
        observation=self.subject_observation
        state=observation.get('state',dict(valid=False))
        progress=progress_at(self.active,t)
        reference=self.spec.reference_frame(progress)
        cp,cr=pose_at(self.active,t)
        candidate=frame_state(cp,cr,self.spec.model,self.spec.intrinsics)
        return dict(reference_at_progress=reference,candidate_prediction=candidate,
            actual_measurement=state,measurement_status=observation.get('status','unknown'),
            actual_vs_reference=frame_error(state,reference),actual_vs_candidate=frame_error(state,candidate),
            immutable_terminal_target=self.spec.reference_frame(self.original.duration),
            actual_vs_terminal=frame_error(state,self.spec.reference_frame(self.original.duration)),
            observation_stamp_s=observation.get('stamp_s'),reference_sha256=self.original.sha256)

    def update(self,scene,t):
        self.subject_observation=scene.get('subject_observation',{})
        if self.mode=='strict' or t>=self.active.duration:return
        if self.spec is None:
            self.spec=IntentSpec(self.original,scene.get('subject_model'),scene.get('intrinsics'))
        if self.future is not None and self.future.done():
            try:result=self.future.result()
            except Exception as error:result=dict(error=f'camera_planner: {type(error).__name__}: {error}')
            self.future=None;self.attempts+=1
            self.write('candidate_evaluation',t,evaluations=result.get('evaluations',[]),planner_wall_s=result.get('wall_s'))
            if result.get('error'):
                self.reject(result['error'],t)
                self.failures[-1]['candidate_feedback']=[dict(program=e['program'],
                    reason=e.get('rejected'),clearance_m=e['physical']['min_clearance_m'],
                    terminal_frame=e.get('intent',{}).get('terminal_frame'),motion_trend=e.get('intent',{}).get('motion_trend'),
                    effective_coverage=e.get('intent',{}).get('effective_coverage'),
                    longest_occlusion_s=e.get('intent',{}).get('longest_occlusion_s'))
                    for e in result.get('evaluations',[])][:16]
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
                    self.future=self.pool.submit(camera_candidates,self.original,self.active,decision['strategy'],scene,t,self.budget,self.radius,self.executed)
            else:
                candidate=self.pending.get(decision['candidate_id']);self.pending={}
                if candidate is None:self.reject('selection_outside_pending_set',t)
                elif t>=candidate.path.activation_s:self.reject('candidate_expired_requires_new_program',t)
                elif path_hash(candidate.path)!=candidate.id or path_hash(self.active)!=self.parent:self.reject('candidate_binding_changed',t)
                elif candidate.path.program['route']=='continuation':
                    self.write('continuation_selected',t,candidate_id=candidate.id,decision=decision)
                    self.started=None;self.last_attempt=t
                else:
                    approval=dict(candidate_id=candidate.id,card_sha256=digest(candidate.card()),parent_id=self.parent)
                    self.write('candidate_approved',t,approval=approval,decision=decision)
                    refreshed,_=evaluate_path(candidate.path,scene,t,candidate.path.envelope.end_s+POLICY.terminal_hold_s,self.radius)
                    current_intent=intent_prediction(self.spec,candidate.path,scene,t,self.executed)
                    if not refreshed.feasible:self.reject('latest_camera_prediction_collision',t)
                    elif not scene['valid'] or t-scene['observation_stamp']>.3:self.reject('camera_observation_stale',t)
                    elif subject_observation_error(scene.get('subject_observation'),self.original,t):
                        self.reject(subject_observation_error(scene.get('subject_observation'),self.original,t),t)
                    elif not current_intent['camera_limits_pass']:
                        self.reject('latest_camera_motion_limits',t)
                    else:
                        self.active=candidate.path;self.installed+=1;self.last_attempt=t
                        latency=time.monotonic()-self.started
                        self.outcomes.append(dict(outcome='installed',time_s=t,wall_s=latency));self.started=None
                        self.write('plan_installed',t,candidate_id=candidate.id,approval=approval,
                            program=self.active.program,actual_parameters=self.active.values.tolist(),
                            checks=dict(physical_valid='pass',state_fresh='pass',target_authorized='pass',
                            predicted_visual_valid='pass_calibrated_geometry' if current_intent['intent_complete'] else 'accepted_intent_violation',observed_visual_valid='unknown_RGB'),
                            end_to_end_wall_s=latency,latest_state_check=refreshed.to_dict(),intent=current_intent)
        if (self.future is None and self.jev.future is None and t-self.last_attempt>=self.budget.decision_period_s
                and self.active.duration-t>self.budget.activation_lead_s+.3):
            value,risk=evaluate_path(self.active,scene,t,self.active.duration+POLICY.terminal_hold_s,self.radius,step=.05)
            intent=intent_prediction(self.spec,self.active,scene,t,self.executed,step=.1)
            self.last_attempt=t
            if not value.feasible or not intent['intent_complete']:
                issues=[]
                if risk is not None or not value.feasible:issues.append('predicted_camera_collision_or_low_clearance')
                if not intent['intent_complete']:issues.append('incomplete_visible_intent')
                self.report=dict(issues=issues,first_risk_s=risk,intent=intent,
                    min_clearance_m=value.min_clearance_m,joint_margin_rad=None,
                    local_repair_evidence='intent-preserving spatial routes and time allocation; no WBC')
                self.write('camera_difficulty',t,report=self.report,execution_model='camera_only')
                self.started=time.monotonic();evidence=self.evidence(t,'strategy')
                self.jev.submit(evidence);self.write('strategy_requested',t,evidence=evidence)

    def close(self):
        if self.started is not None:
            self.outcomes.append(dict(outcome='unfinished_at_shot_end',
                time_s=self.executed[-1]['t'] if self.executed else 0.,wall_s=time.monotonic()-self.started))
            self.write('pipeline_unfinished',self.executed[-1]['t'] if self.executed else 0.)
        future=self.jev.future
        self.jev.close()
        if future is not None:
            try:future.result(timeout=4.)
            except Exception:pass
        self.pool.shutdown(wait=True,cancel_futures=True);self.events.close()


def run_case(case,output,mode,budget,radius,model,api_url,pace):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime=RuntimeScene(Path(case)/'runtime_scene.json')
    original=OriginalReference(Path(case)/'task.json',runtime.document['subject']['position'])
    subject_model=runtime.document['subject']
    intrinsics=default_intrinsics()
    specification=IntentSpec(original,subject_model,intrinsics)
    save_json(output/'original_task.json',original.task);save_json(output/'runtime_scene.json',runtime.document)
    save_json(output/'intent_specification.json',specification.to_dict())
    observer=CameraObserver()
    subject_observer=SubjectStateObserver(subject_model,intrinsics)
    p0=original.sample(0.)[0]
    for t in np.linspace(-.8,0.,9):scene=observer.scan(runtime,p0,float(t))
    agent=FloatingAgent(original,output,mode,budget,radius,model,api_url)
    rows=[];failure='';begin=time.monotonic();dt=.02
    try:
        for i in range(round((POLICY.max_duration_factor*original.duration+POLICY.terminal_hold_s)/dt)+1):
            loop_started=time.monotonic()
            t=i*dt;before=positions_at(agent.active,np.array([t]))[0]
            if i%5==0:scene=observer.scan(runtime,before,t)
            if i%5==0:
                bp,br=pose_at(agent.active,t)
                points=simulated_subject_returns(bp,CAMERA_RAYS,subject_model,runtime.obstacles(t))
                subject_observer.observe(points,bp,br,t)
            subject_observation=subject_observer.snapshot(t)
            scene.update(subject_model=subject_model,intrinsics=intrinsics,subject_observation=subject_observation)
            agent.update(scene,t)
            p,r=pose_at(agent.active,t) # Exact pose assignment; no twist calculation, integration or robot FK.
            end=min(t+dt,agent.active.duration+POLICY.terminal_hold_s)
            audit_times=np.linspace(t,end,9)
            audit_positions=positions_at(agent.active,audit_times)
            true_clearance=min(float(np.linalg.norm(at_position-obstacle['center'])-obstacle['radius']-radius)
                for at,at_position in zip(audit_times,audit_positions) for obstacle in runtime.obstacles(float(at)))
            true_clearance=min(true_clearance,float(subject_clearances(audit_positions,subject_model,radius).min()))
            op=original.translation(np.clip(t,0.,original.duration))
            info=dict(target_position=p.tolist(),position_error=0.,orientation_error=0.,joint_margin_rad=None,
                active_decision='' if agent.active is original else agent.active.envelope.decision_id,
                dynamic_constraint_count=0,obstacle_avoidance_enabled=False)
            rows.append(dict(t=t,camera_position=p.tolist(),camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
                truth_clearance_m=true_clearance,info=info,observation_stamp=scene['observation_stamp'],
                progress_s=progress_at(agent.active,t),subject_observation=subject_observation,
                visual=visibility(p,r,scene,t,subject_model,intrinsics)))
            # Only causal observed geometry is allowed into subsequent planning.
            # Independent true-scene visual auditing is performed after execution.
            agent.executed.append(rows[-1])
            info['loop_wall_s']=time.monotonic()-loop_started
            if i%25==0:agent.history.append(dict(time_s=t,target_tracking_residual_m=0.,clearance_m=true_clearance,
                original_position_error_m=float(np.linalg.norm(p-op))))
            if pace>0:
                remaining=(i+1)*dt/pace-(time.monotonic()-begin)
                if remaining>0:time.sleep(remaining)
            if t >= agent.active.duration+POLICY.terminal_hold_s-1e-8:break
    except Exception as error:
        failure=f'{type(error).__name__}: {error}';agent.write('run_failed',len(rows)*dt,failure=failure)
    finally:
        agent.close()
        elapsed=rows[-1]['t'] if rows else 0.
        positions=np.array([x['camera_position'] for x in rows])
        errors=[np.linalg.norm(np.asarray(x['camera_position'])-original.sample(x['t'])[0]) for x in rows]
        clearance=[x['truth_clearance_m'] for x in rows]
        completed=not failure and elapsed>=agent.active.duration+POLICY.terminal_hold_s-1e-8
        safe=bool(clearance and min(clearance)>=.025)
        rotations=[Rotation.from_quat(row['camera_quaternion_xyzw']).as_matrix() for row in rows]
        true_measurements=[]
        for row,r in zip(rows,rotations):
            obstacles=runtime.obstacles(row['t'])
            truth=dict(valid=True,observation_stamp=row['t'],centers=[o['center'] for o in obstacles],
                radii=[o['radius'] for o in obstacles],velocities=[o['velocity'] for o in obstacles])
            true_measurements.append(visibility(np.array(row['camera_position']),r,truth,row['t'],subject_model,intrinsics))
            row['truth_visual']=true_measurements[-1]
        intent=specification.score([row['t'] for row in rows],positions,rotations,true_measurements,
            [row['progress_s'] for row in rows]) if rows else dict(intent_complete=False,effective_coverage=0.)
        velocity=np.diff(positions,axis=0)/dt if len(rows)>1 else np.zeros((1,3))
        acceleration=np.diff(velocity,axis=0)/dt if len(rows)>2 else np.zeros((1,3))
        angular=[Rotation.from_matrix(b@a.T).magnitude()/dt for a,b in zip(rotations[:-1],rotations[1:])]
        kinematics=dict(max_speed_m_s=float(np.linalg.norm(velocity,axis=1).max()),
            max_acceleration_m_s2=float(np.linalg.norm(acceleration,axis=1).max()),
            max_angular_speed_rad_s=float(max(angular,default=0.)))
        limits_pass=(kinematics['max_speed_m_s']<=POLICY.max_speed_m_s and
            kinematics['max_acceleration_m_s2']<=POLICY.max_acceleration_m_s2 and
            kinematics['max_angular_speed_rad_s']<=POLICY.max_angular_speed_rad_s)
        loop_ms=np.array([row['info']['loop_wall_s']*1000 for row in rows])
        terminal_rows=[row for row in rows if row['t']>=agent.active.duration-1e-8]
        observed_errors=[frame_error(row['subject_observation'].get('state',{}),
            specification.reference_frame(original.duration)) for row in terminal_rows]
        observed_terminal=bool(terminal_rows and all(row['subject_observation']['status']=='valid' for row in terminal_rows)
            and all(e['valid'] and e['centroid_error']<=POLICY.terminal_centroid_tolerance
                and e['log_area_error']<=POLICY.terminal_log_area_tolerance
                and e['bbox_log_error']<=POLICY.terminal_bbox_log_tolerance
                and e['view_angle_error_rad']<=POLICY.terminal_view_tolerance_rad
                and (e['silhouette_axis_error_rad'] is None or e['silhouette_axis_error_rad']<=POLICY.terminal_silhouette_axis_tolerance_rad) for e in observed_errors))
        summary=dict(backend='ideal_floating_camera' ,mode=mode,semantic_model=model,completed=completed,
            safety_pass=safe,camera_intent_success=completed and safe and limits_pass and intent['intent_complete'] and observed_terminal,
            within_budget_success=completed and safe and limits_pass and intent['intent_complete'] and observed_terminal and elapsed<=POLICY.max_duration_factor*original.duration+POLICY.terminal_hold_s,
            observed_terminal_pass=observed_terminal,observed_terminal_errors=observed_errors,
            observed_terminal_distinct_scans=len({row['subject_observation'].get('stamp_s') for row in terminal_rows}),
            terminal_hold_s=POLICY.terminal_hold_s,motion_duration_s=agent.active.duration,
            camera_limits_pass=limits_pass,camera_kinematics=kinematics,
            control_loop_wall_ms=({k:float(v) for k,v in zip(('mean','p95','p99','max'),
                [loop_ms.mean(),np.percentile(loop_ms,95),np.percentile(loop_ms,99),loop_ms.max()])} if len(rows) else None),
            intent=intent,failure=failure,planner_model='floating_camera',
            installed_plans=agent.installed,jev_calls=agent.jev.decider.client.calls,wall_s=time.monotonic()-begin,
            elapsed_shot_s=elapsed,original_position_rms_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            min_truth_clearance_m=min(clearance) if clearance else None,collision_samples=sum(c<0 for c in clearance),
            clearance_violation_samples=sum(c<.025 for c in clearance),target_tracking_max_m=0.,target_tracking_max_rad=0.,
            camera_radius_m=radius,robot_state_used=False,obstacle_avoidance_enabled=False,
            collision_audit='ground-truth obstacle spheres and subject cylinder sampled at 400 Hz; camera sphere; no control intervention',
            prediction_audit='50 Hz camera path with piecewise-linear relative-segment closest approach; not a continuous nonlinear certificate',
            observation_source='current ray hits and past fitted-center velocity; no future keyframes',
            decision_outcomes=agent.outcomes,original_duration_s=original.duration,
            execution_configuration=dict(pace=pace,activation_lead_s=budget.activation_lead_s),
            evaluation_protocol=asdict(POLICY),time_extension_s=max(0.,elapsed-original.duration),
            visual_smoothness=smoothness([row['t'] for row in rows],positions,
                [row['camera_quaternion_xyzw'] for row in rows]),
            observed_visual_source='current simulated LiDAR subject returns fitted to calibrated cylinder; RGB/SAM not run',
            whole_body_execution='not_run')
        save_json(output/'summary.json',summary);save_json(output/'telemetry.json',rows)
    print(json.dumps(dict(case=runtime.document['name'],mode=mode,completed=completed,safe=safe,
        intent=intent['intent_complete'],coverage=intent['effective_coverage'],duration_s=elapsed,
        installed=agent.installed,failure=failure)),flush=True)
    return summary


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',default='all');p.add_argument('--scenarios',type=Path,default=root/'scenarios')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['jev','rule','strict','both'],default='jev')
    p.add_argument('--model',default='perplexity');p.add_argument('--api-url',default='http://127.0.0.1:8000/v1/systemone')
    p.add_argument('--camera-radius',type=float,default=.06);p.add_argument('--pace',type=float,default=1.)
    p.add_argument('--activation-lead',type=float,default=2.7)
    a=p.parse_args()
    if not np.isfinite(a.camera_radius) or a.camera_radius<=0 or not np.isfinite(a.pace) or a.pace<0:p.error('invalid radius or pace')
    budget=SearchBudget(activation_lead_s=a.activation_lead,max_proposals=16)
    a.output.mkdir(parents=True,exist_ok=True)
    save_json(a.output/'provenance.json',source_manifest(root,{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
    results=[]
    for case in NAMES if a.case=='all' else [a.case]:
        if Path(case).name!=case or case in ('.','..'):p.error('invalid case name')
        for mode in ['strict','jev'] if a.mode=='both' else [a.mode]:
            results.append(run_case(a.scenarios/case,a.output/f'{case}_{mode}',mode,budget,a.camera_radius,a.model,a.api_url,a.pace))
            save_json(a.output/'runs.json',results)
    return 0 if all(x['completed'] if x['mode']=='strict' else x['camera_intent_success'] for x in results) else 2


if __name__=='__main__':raise SystemExit(main())
