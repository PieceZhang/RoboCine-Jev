"""Slow event-driven semantic planning; continuously certified independent WBC."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import native, PARAMETERS, digest
from .jev import AsyncJev, CandidateDecider, PhaseAwareDecider, SystemOneClient
from .planning import joint_margin, ExecutionWBC, TrackingOnlyWBC
from .candidates import (SearchBudget, Approval, diagnose_bounded, build_candidates,
                         verify_candidate, installation_guard, path_hash, support_violation)
from .perception import SubjectEvidence
from .subject_state import compare_subject_evidence
from .shot_preview_worker import initialize_worker
from .whole_body_qp import Limits, whole_body_kinematics
from .safety import solve_with_certificate
from .kinematic_feasibility import POLICY as KINEMATIC_POLICY
from .phase_checkpoint import evaluate_phase as evaluate_phase_checkpoint


def geometry_feedback_snapshot(guidance,box_budget,state,t,base_z,base_rotation):
    """Advisory measured geometry; hard control preferences remain synchronous."""
    from .representation_jvs import JVSGeometryBackend
    backend=JVSGeometryBackend(box_budget)
    observation=backend.observe(state,t,base_z=base_z,base_rotation=base_rotation)
    feedback=guidance.feedback(observation,'measured')
    return feedback,json.dumps(native(feedback),ensure_ascii=False,allow_nan=False)


def needs_configuration_confirmation(diagnosis, feedback, t):
    """Allow one fresh diagnosis for transient local-model disagreement.

    The command still passes the independent per-step safety certificate. New
    geometry/visual risks always escalate immediately. Hard actuator or safety
    failures raise before feedback is queued. camera_actuator_feasible means
    exact camera-twist feasibility, not whether the issued command is safe.
    This gate expires after 0.8 s, including a stalled diagnosis worker.
    """
    if (diagnosis['report']['issues'] or not diagnosis.get('kinematics',{}).get('feasible')
            or diagnosis['report'].get('actual_tracking_error_m',float('inf'))>KINEMATIC_POLICY.position_tolerance_m
            or not feedback):
        return False
    for item in feedback:
        local=item.get('wbc_feasibility',{})
        stamp=item.get('time_s',-float('inf'))
        if (item.get('error')!='trajectory_preserving_configuration_degraded'
                or 'barrier_mode' not in local
                or local.get('first_predicted_conflict_s') is not None
                or local.get('pairs') or t-stamp>=.8
                or diagnosis['snapshot_time_s']-stamp>=.3):
            return False
    return True


def configure_reactive_progress(controller, original, active, enabled):
    """Use a local passage clock only for an unplanned single action.

    An installed candidate owns a verified camera clock and body allocation;
    the optional reactive policy must not retime that approval. Combinations
    retain their existing phase-aware controller. Disabled leaves backend
    settings untouched, including physical-runtime experiments.
    """
    if not enabled:
        return
    from .combo import is_combo
    unplanned = active is original and not is_combo(original)
    controller.body_aware = unplanned
    controller.observed_clock_enabled = unplanned


class CinematicEngine:
    def __init__(self, original, scene, output, *, mode='jev', api_url=None, model='perplexity', limits=None,
                 decision_cooldown_s=None, preview=False, budget=None, intrinsics=None,
                 decider=None, planning_obstacle_avoidance=None, planner_model="floating_camera", coordination_async=False,
                 subject_model=None, representation_boxes=3, geometry_preferences=True, runtime_policy=None, reactive_progress=False):
        if mode not in ('jev', 'strict', 'rule'):
            raise ValueError('mode must be jev, rule or strict')
        if planner_model not in ("wbc", "floating_camera"):
            raise ValueError("unknown planner model")
        self.planner_model = planner_model
        self.runtime_policy = runtime_policy
        self.reactive_progress = bool(reactive_progress)
        self.runtime_fault = None
        self.runtime_delays = set()
        self.runtime_retries = 0
        self.runtime_recovery_started = None
        self.last_execution_certificate = None
        self.runtime_retry_after = -float('inf')
        self.math_started_wall = None
        self.subject_model = subject_model
        from .representation_jvs import JVSGeometryBackend
        self.geometry_backend = JVSGeometryBackend(representation_boxes)
        self.geometry_preferences = geometry_preferences
        self.geometry_scene = None
        self.last_geometry_feedback_s = -float('inf')
        self.geometry_feedback_async=coordination_async
        self.geometry_feedback_future=None
        self.camera_history = []
        self.actual_feedback = None
        self.camera_output = None
        self.original = self.active = original
        self.mode, self.limits = mode, limits or Limits()
        # Planning may model avoidance independently of the execution controller.
        self.planning_limits = replace(self.limits, obstacle_avoidance=(
            self.limits.obstacle_avoidance if planning_obstacle_avoidance is None
            else planning_obstacle_avoidance))
        self.semantic_model = model if mode == 'jev' and decider is None else 'custom' if decider is not None else 'none'
        self.budget = budget or SearchBudget()
        self.cooldown = self.budget.decision_period_s if decision_cooldown_s is None else decision_cooldown_s
        self.intrinsics = intrinsics
        self.perception = SubjectEvidence(original.task['subject']['id'])
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if coordination_async:
            from .async_journal import AsyncJournal
            self.events=AsyncJournal(self.output/'events.jsonl')
        else:self.events = (self.output/'events.jsonl').open('w')
        if decider is None:
            client = SystemOneClient(model=model, journal=self.output/'jev_api.jsonl',
                **({} if api_url is None else {'url': api_url}))
            decider = PhaseAwareDecider() if mode in ('rule', 'strict') else CandidateDecider(client)
        self.jev = AsyncJev(decider)
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)
        self.math_future, self.math_kind = None, ''
        controller = TrackingOnlyWBC if not self.limits.obstacle_avoidance else ExecutionWBC
        self.wbc = controller(self.limits, scene=scene, horizon_steps=4, horizon_dt=.3, replan_s=.4)
        self.wbc.set_trajectory(original)
        self.wbc.geometry_backend = self.geometry_backend
        self.wbc.geometry_guidance = None
        if coordination_async and isinstance(self.wbc, ExecutionWBC):
            from .coordination import AsyncCoordinationPreview
            self.wbc.coordination = AsyncCoordinationPreview()
            self.wbc.coordination.set_trajectory(original)
            from .tpr_wbc import AsyncReconfigurationPreview
            self.wbc.reconfiguration = AsyncReconfigurationPreview()
            self.wbc.reconfiguration.reset(original)
        self.last_report_s = self.last_attempt_s = self.last_install_s = -float('inf')
        self.report, self.context, self.pipeline_parent = None, None, None
        self.pending_candidates = {}
        self.search_feedback = []
        from .configuration_preparation import ConfigurationMonitor
        self.configuration_monitor = ConfigurationMonitor()
        self.last_configuration_status = None
        self.active_candidate = None
        self.history, self.pending_feedback = [], []
        self.wbc_degraded_samples = 0
        self.revision, self.closed = 0, False
        self.validation_attempts = self.feasible_plans = self.installed_plans = self.rejected_stale = 0
        self.last_event_signature, self.pipeline_started_wall = None, None
        self.pipeline_latencies = []
        self.pipeline_outcomes = []
        self.last_visual_stamp, self.observed_visual_valid = None, 'unknown'
        self._write('initialized', 0., mode=mode, original_sha256=original.sha256,
            original_budget_s=original.duration, upstream_fields=list(original.task),
            architecture='exact_counterfactual/v2', budget=asdict(self.budget), planner_model=self.planner_model,
            representation='jev_geometry/v1', representation_boxes=representation_boxes, geometry_preferences=geometry_preferences,
            kinematic_gate='fk_constrained_continuous_lift/v1' if planner_model=='floating_camera' else None,
            obstacle_avoidance_enabled=self.limits.obstacle_avoidance,
            wbc_algorithm='TPR-WBC' if self.limits.obstacle_avoidance else 'tracking-only',
            legacy_allocation_preview_ignored=bool(preview and self.limits.obstacle_avoidance),
            reactive_progress_enabled=self.reactive_progress,
            planning_obstacle_avoidance_enabled=self.planning_limits.obstacle_avoidance,
            visual_intrinsics=intrinsics, semantic_policy=type(decider).__name__)

    def _write(self, event, t, **details):
        self.events.write(json.dumps(native(dict(event=event, time_s=t, **details)), ensure_ascii=False, allow_nan=False)+'\n')
        self.events.flush()

    def _feedback(self, error, t, finish_pipeline=True):
        from .execution_prediction import failure_evidence
        self.pending_feedback.append(dict(error=error, time_s=t, failure=failure_evidence(error,t)))
        self.pending_feedback = self.pending_feedback[-4:]
        self._write('proposal_rejected', t, error=error)
        if finish_pipeline:
            self.last_attempt_s = t
        if finish_pipeline and self.pipeline_started_wall is not None:
            self.pipeline_outcomes.append(dict(outcome='rejected', reason=error, time_s=t,
                wall_s=time.monotonic()-self.pipeline_started_wall))
            self.pipeline_started_wall = None

    def _submit_math(self, kind, function, *args, **kwargs):
        if self.planner_model == 'floating_camera':
            from . import task_space_planning
            function = {'diagnose': task_space_planning.diagnose,
                        'candidates': task_space_planning.build, 'verify': task_space_planning.verify}[kind]
            if kind == 'diagnose':
                kwargs.update(active=self.active, budget=self.budget,encode_journal=True,
                    accepted_intent=None if self.active_candidate is None else self.active_candidate.consequences.get('intent'))
            if kind == 'candidates':
                kwargs.update(representation_boxes=self.geometry_backend.box_budget,
                              geometry_preferences=self.geometry_preferences)
            kwargs['execution_model'] = getattr(self,'execution_model',None)
            kwargs['history'] = self.camera_history.copy()
            velocity = getattr(self.wbc, 'measured_control', None)
            kwargs['measured_control'] = None if velocity is None else np.asarray(velocity).copy()
            rotation = getattr(self, 'planning_base_rotation', None)
            kwargs['base_rotation'] = None if rotation is None else np.asarray(rotation).copy()
        self.math_kind = kind
        self.math_started_wall = time.monotonic()
        self.math_future = self.pool.submit(function, *args, **kwargs)

    def _refresh_verification(self, result, error, scene, state, t, base_z, commanded_q):
        """Retry one stale proof against fresh sensors, preserving approval/path.

        Never rebase activation or extend a lease. The new worker runs the
        complete verifier and its result still passes installation_guard.
        """
        refreshable={'latest_verification_expired',
            'camera_left_verified_state_neighborhood',
            'configuration_left_verified_kinematic_neighborhood'}
        if error not in refreshable or self.runtime_policy is None or not self.runtime_policy.recover_decisions:
            return False
        candidate=result['candidate'];approval=result['approval']
        attempts=getattr(self,'verification_refreshes',{})
        if (attempts.get(candidate.id,0)>=1 or self.math_future is not None
                or candidate.path.activation_s-t<=.2
                or path_hash(self.active)!=approval.parent_id
                or not 0<=t-float(scene['observation_stamp'])<=self.runtime_policy.sensor_timeout_s):
            return False
        attempts[candidate.id]=attempts.get(candidate.id,0)+1
        self.verification_refreshes=attempts
        self._submit_math('verify',verify_candidate,candidate,approval,self.original,
            scene,np.asarray(state).copy(),self.wbc.previous.copy(),t,base_z,self.planning_limits,
            evidence=self.perception.snapshot(t),intrinsics=self.intrinsics,budget=self.budget,
            commanded_q=None if commanded_q is None else np.asarray(commanded_q).copy())
        self._write('verification_refreshed',t,candidate_id=candidate.id,reason=error,
            attempt=attempts[candidate.id],activation_s=candidate.path.activation_s,
            approval_reused=True,full_verification_required=True)
        return True

    def _phase_recovery_checkpoint(self,t):
        recovery=getattr(self,'runtime_recovery_started',None)
        if recovery is None or not self.runtime_policy.recover_decisions:return False
        phases=getattr(getattr(self,'original',None),'task',{}).get('shot_phase',[])
        if len(phases)<2:return False
        certificate=getattr(self,'last_execution_certificate',None)
        observation=getattr(self,'geometry_scene',None)
        if (certificate is None or observation is None
                or not 0<=t-certificate['time_s']<=self.runtime_policy.sensor_timeout_s
                or not 0<=t-float(observation['observation_stamp'])<=self.runtime_policy.sensor_timeout_s
                or certificate['clearance_m']<self.wbc.collision.margin
                or certificate['self_clearance_m']<self.wbc.self_collision.margin):return False
        from .camera_intent import POLICY
        checked=getattr(self,'recovery_checkpoint_phases',set())
        pending=getattr(self,'recovery_checkpoint_pending',None)
        if pending is not None:
            if not pending['future'].done():return False
            self.recovery_checkpoint_pending=None
            index=pending['index'];checked.add(index);self.recovery_checkpoint_phases=checked
            if pending['episode_start']!=recovery['time_s']:return False
            try:result=pending['future'].result()
            except Exception as error:
                self._write('decision_checkpoint_unverified',t,index=index,reason=str(error))
                return False
            if result is None:return False
            self._write('decision_phase_checkpoint',t,index=index,phase_end_s=phases[index]['end_s'],
                accepted=bool(result['accepted']),failed_checks=result['failed_checks'],
                scope='provisional observed-camera phase checkpoint; final physical whole-shot audit unchanged')
            if result['accepted']:
                self.runtime_recovery_started=None;self.runtime_retries=0
                # Pending API/worker deadlines, physical latches, original task
                # deadlines and approval bindings remain in force.
                return True
        for index,phase in enumerate(phases[:-1]):
            boundary=phase['end_s']+POLICY.terminal_hold_s
            if index in checked or t<boundary or boundary<recovery['time_s']:continue
            # Reuse the existing bounded numerical worker. At most one phase
            # audit is queued; no camera-shape work blocks the measured servo.
            history=[row for row in self.camera_history if row['t']<=boundary+1e-8]
            try:
                future=self.pool.submit(evaluate_phase_checkpoint,self.original,phase,history,self.subject_model,
                    self.intrinsics,self.runtime_policy.sensor_timeout_s)
            except Exception as error:
                checked.add(index);self.recovery_checkpoint_phases=checked
                self._write('decision_checkpoint_unverified',t,index=index,reason=str(error))
                return False
            self.recovery_checkpoint_pending=dict(future=future,index=index,episode_start=recovery['time_s'])
            self._write('decision_phase_checkpoint_submitted',t,index=index,physics_waits=False)
            break
        return False

    def _runtime_deadline(self, t):
        if self.runtime_policy is None or self.runtime_fault is not None:
            return
        now = time.monotonic()
        recovery=getattr(self,'runtime_recovery_started',None)
        if recovery is not None and max(t-recovery['time_s'],now-recovery['wall_s'])>self.runtime_policy.recovery_timeout_s:
            self.runtime_fault=dict(reason='decision_recovery_exhausted',time_s=t,
                elapsed_wall_s=now-recovery['wall_s'])
            self.pending_candidates.clear()
            self._write('runtime_fault_latched',t,reason='decision_recovery_exhausted',
                action='bounded recovery expired; brake and require a new session')
            return
        if recovery is not None:self._phase_recovery_checkpoint(t)
        stages = [('planning_pipeline_timeout', self.pipeline_started_wall),
                  ('planning_'+self.math_kind+'_timeout',
                   self.math_started_wall if self.math_future is not None else None),
                  ('semantic_timeout', self.jev.submitted_wall if self.jev.future is not None else None)]
        for reason, started in stages:
            if started is not None and now-started > self.runtime_policy.planning_timeout_s:
                if self.runtime_policy.recover_decisions and now-started <= self.runtime_policy.recovery_timeout_s:
                    key=(reason,started)
                    if key not in self.runtime_delays:
                        self.runtime_delays.add(key)
                        self._write('decision_waiting',t,reason=reason,elapsed_wall_s=now-started,
                            action='continue certified execution; retain pending result for fresh verification')
                    continue
                self.runtime_fault = dict(reason=reason, time_s=t, elapsed_wall_s=now-started)
                self._write('runtime_fault_latched', t, **{k:v for k,v in self.runtime_fault.items() if k!='time_s'},
                            action='brake; discard late results; explicit rearm required')
                self.pending_candidates.clear()
                self._feedback(reason, t)
                break

    def _runtime_failure(self, reason, t):
        if self.runtime_policy is not None and self.runtime_fault is None:
            if self.runtime_policy.recover_decisions and getattr(self,'runtime_recovery_started',None) is None:
                self.runtime_recovery_started=dict(time_s=t,wall_s=time.monotonic(),reason=reason)
            if self.runtime_policy.recover_decisions and self.runtime_retries < self.runtime_policy.max_retries:
                self.runtime_retries += 1
                self._write('decision_retry',t,reason=reason,attempt=self.runtime_retries,
                    action='preserve active trajectory; rebuild from latest observation')
                return
            certificate=getattr(self,'last_execution_certificate',None)
            observation=getattr(self,'geometry_scene',None)
            fresh_execution=(certificate is not None and observation is not None
                and 0<=t-certificate['time_s']<=self.runtime_policy.sensor_timeout_s
                and 0<=t-float(observation['observation_stamp'])<=self.runtime_policy.sensor_timeout_s
                and certificate['clearance_m']>=self.wbc.collision.margin
                and certificate['self_clearance_m']>=self.wbc.self_collision.margin)
            if self.runtime_policy.recover_decisions and fresh_execution:
                # Failure to construct a replacement is not a revocation of
                # the last executable objective, including the authored path.
                # Continuation is provisional: THIS step must again pass all
                # WBC gates, and physical execution rechecks the current state.
                self.runtime_retry_after=t+2.
                self._write('decision_retry_budget_exhausted',t,reason=reason,
                    action='retain objective under fresh execution certificates; bounded recovery cooldown',
                    original_objective=self.active is self.original,certificate=certificate)
                return
            self.runtime_fault = dict(reason=reason, time_s=t)
            self.pending_candidates.clear()
            self._write('runtime_fault_latched', t, reason=reason,
                        action='brake; discard late results; explicit rearm required')

    def _evidence(self, t):
        from .representation import TaskContext
        phase = self.original.phase(t)
        following = [p for p in self.original.task['shot_phase'] if p['start_s'] >= phase['end_s']]
        recovery = self.active is not self.original and not self.report['report']['issues'] and t-self.last_install_s > 1.5
        feedback = self.actual_feedback
        earliest_change=min((c.path.activation_s for c in self.pending_candidates.values()
                             if c.path.activation_s>t),default=t+self.budget.activation_lead_s)
        geometry = None
        if feedback is not None:
            observation = self.geometry_backend.observe(feedback['state'],feedback['time_s'],
                base_z=self.geometry_base_z,base_rotation=getattr(self,'planning_base_rotation',None))
            geometry = observation.summary(self.geometry_backend.obstacle_relations(observation,self.geometry_scene))
            self._write('geometry_observation',t,observation=observation.to_dict(),observation_id=observation.id)
            if getattr(self,'execution_capability',None) is not None:
                geometry['execution_capability']=self.execution_capability
            guide = getattr(self.wbc,'geometry_guidance',None)
            if guide is not None:
                geometry['execution_feedback'] = guide.feedback(observation,'measured')
        construction={}
        if self.budget.single_candidate and self.planner_model=='floating_camera' and self.geometry_scene is not None:
            from .directed_search import construction_evidence
            construction=construction_evidence(self.original,self.active,self.geometry_scene,t,t+self.budget.activation_lead_s,
                mount_height_m=None if geometry is None else geometry['world_from_mount']['translation_m'][2])
            construction['bundle_construction'] = True
        return dict(planner_model=self.planner_model, actual_robot_feedback=self.actual_feedback,construction=construction,
            bundle_search_choices=True,
            robot_geometry=geometry,
            camera_task=TaskContext(self.original.sha256,t,self.original.duration,
                earliest_change,self.report.get('kinematics',{}).get('first_failure_s'),
                bool(self.active is not self.original and
                    not self.report['report'].get('intent',{}).get('intent_complete',False))).to_dict(),
            objective='visible_intent_completion' if self.planner_model == 'floating_camera' else 'legacy_reference_tracking',
            prediction_scope=('delayed WBC/drive prediction, empirical error envelope, bounded observed-safety horizon'
                if getattr(self,'execution_model',None) is not None else "state-seeded FK with observed whole-body/self-collision swept checks; no dynamics or future truth"
                if self.planner_model == "floating_camera" else "whole-body WBC rollout"),
            time_s=t, duration_s=self.original.duration, phase=phase, shot_phase=self.original.task['shot_phase'],
            next_phase=following[0] if following else None,
            event='validation_failed' if self.pending_feedback else 'recovery' if recovery else 'difficulty',
            subject=self.original.task['subject'], intent=self.original.task['intent'],
            reference_subject_position=self.original.subject.tolist(),
            active_modification=self.active is not self.original,
            difficulty=self.report['report'], subject_evidence=self.perception.snapshot(t),
            robot_capability=self.report.get('robot_capability'),
            subject_state_comparison=self._subject_comparison(t),
            active_plan=path_hash(self.active), execution_history=self.history[-4:],
            validation_feedback=self.pending_feedback[-3:], revision=self.revision,
            remaining_budget_s=max(0., self.original.duration-t))

    def _subject_comparison(self,t):
        return compare_subject_evidence(self.original,self.active,self.perception.snapshot(t),
            self.subject_model,self.intrinsics,t,getattr(self,'actual_feedback',None))

    def _monitor_visual(self, t):
        snapshot = self.perception.snapshot(t)
        visual = snapshot['visual']
        self.observed_visual_valid = 'unknown'
        if visual['status'] != 'valid':
            return
        stamp = visual['stamp_s']
        if stamp == self.last_visual_stamp:
            self.observed_visual_valid = getattr(self, '_last_visual_status', 'unknown')
            return
        self.last_visual_stamp = stamp
        if self.active_candidate is None:
            if visual.get('truncated') or visual.get('occluded'):
                self._feedback('new_visibility_evidence_requires_review', t, finish_pipeline=False)
            return
        prediction = self.active_candidate.consequences
        if 'intent' in prediction and (visual.get('occluded') or visual.get('truncated')):
            # Geometric intent cards have no timestamp-matched mask-center
            # samples. A new observed occlusion still requires replanning.
            self._write('visual_feedback', t, observed_visual_valid='unknown',
                cause='new_visibility_evidence_requires_review', acquisition_stamp_s=stamp,
                subject_evidence=snapshot)
            self._feedback('new_visibility_evidence_requires_review', t, finish_pipeline=False)
            return
        if 'intent' in prediction:
            comparison=self._subject_comparison(t)
            measured=comparison['errors'].get('candidate',{})
            center=measured.get('centroid_error');scale=measured.get('log_area_error')
            mismatch=(center is not None and center>.05) or (scale is not None and scale>.10)
            self.observed_visual_valid=self._last_visual_status='fail' if mismatch else 'unknown'
            self._write('visual_feedback',t,observed_visual_valid=self.observed_visual_valid,
                acquisition_stamp_s=stamp,subject_state_comparison=comparison,
                cause='measured_subject_frame_mismatch' if mismatch else 'partial_frame_consistency')
            if mismatch:self._feedback('measured_subject_frame_mismatch',t,finish_pipeline=False)
            return
        rows = prediction.get('samples', [])
        status, cause = 'unknown', snapshot['diagnosis']
        valid = [r for r in rows if r['projected_center'] is not None]
        # Causal comparison on the image's original capture time, not callback time.
        if valid and valid[0]['t'] <= stamp <= valid[-1]['t']:
            expected = np.array([np.interp(stamp, [r['t'] for r in valid],
                [r['projected_center'][i] for r in valid]) for i in range(2)])
            residual = float(np.linalg.norm(np.asarray(visual['centroid'])-expected))
            if visual.get('occluded') is False and not visual.get('truncated'):
                status = 'pass' if residual <= .05 else 'fail'
                cause = 'projection_mask_consistency' if status == 'pass' else 'perception_or_prediction_mismatch'
            elif visual.get('truncated') or visual.get('occluded'):
                # Unknown in the candidate cannot be retroactively declared an intent violation.
                cause = 'new_visibility_evidence_requires_review'
        # Center consistency alone cannot certify scale, view angle or occlusion.
        self.observed_visual_valid = self._last_visual_status = 'fail' if status == 'fail' else 'unknown'
        self._write('visual_feedback', t, observed_visual_valid=self.observed_visual_valid,
                    observed_composition_consistent=status, cause=cause,
                    acquisition_stamp_s=stamp, subject_evidence=snapshot)
        if status == 'fail' or cause == 'new_visibility_evidence_requires_review':
            self._feedback(cause, t, finish_pipeline=False)

    def set_camera_output(self, stamp_s, position, rotation, *, source='stabilized', max_age_s=0.):
        """Acquisition-time virtual camera geometry; robot state remains physical."""
        p, r = np.asarray(position, float), np.asarray(rotation, float)
        if (source not in ('stabilized', 'execution') or p.shape != (3,) or r.shape != (3, 3)
                or not np.isfinite(np.r_[stamp_s, max_age_s, p, r.ravel()]).all() or max_age_s < 0):
            raise ValueError('invalid camera output observation')
        self.camera_output = dict(stamp_s=float(stamp_s), position=p.copy(), rotation=r.copy(), source=source,
                                  max_age_s=float(max_age_s))

    def update(self, state, scene, t, base_z, commanded_q=None):
        if self.planner_model == 'floating_camera':
            scene = dict(scene, subject_model=self.subject_model or scene.get('subject_model'),
                         intrinsics=self.intrinsics)
        self.geometry_scene, self.geometry_base_z = scene, base_z
        self.revision += 1
        p,r,_ = whole_body_kinematics(state,base_z,getattr(self,'planning_base_rotation',None))
        camera_source = 'execution'
        camera_stamp = t
        if self.camera_output is not None:
            age = t-self.camera_output['stamp_s']
            if age < -1e-8 or age > self.camera_output['max_age_s']+1e-8:
                raise ValueError('Agent camera output is not aligned to the control timestamp')
            p, r = self.camera_output['position'], self.camera_output['rotation']
            camera_source = self.camera_output['source']
            camera_stamp = self.camera_output['stamp_s']
        self.actual_feedback = dict(time_s=t,state=np.asarray(state).tolist(),camera_position=p.tolist(),
            camera_output_source=camera_source, stamp_s=camera_stamp,
            camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
            target_position=self.active.sample(t)[0].tolist(),observation_stamp=scene['observation_stamp'])
        if self.planner_model == 'floating_camera' and (not self.camera_history or camera_stamp > self.camera_history[-1]['t']):
            from .task_space_planning import camera_scene
            from .camera_intent import visibility, progress_at, default_intrinsics
            observed = camera_scene(scene,t,self.active.duration)
            self.camera_history.append(dict(t=camera_stamp,camera_position=p.tolist(),
                camera_output_source=camera_source,
                camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
                progress_s=progress_at(self.active,camera_stamp),visual=visibility(p,r,observed,camera_stamp,
                    observed.get('subject_model'),self.intrinsics or default_intrinsics())))
        configure_reactive_progress(self.wbc, self.original, self.active,
                                    self.reactive_progress)
        self.wbc.update_collision_map(scene)
        self.wbc.set_time(t)
        self._monitor_visual(t)
        if self.planner_model=='floating_camera' and self.perception.snapshot(t)['geometry'].get('status')=='valid':
            comparison=self._subject_comparison(t)
            error=comparison['errors'].get('candidate',{})
            if (error.get('valid') and (error.get('centroid_error',0.)>.05 or error.get('log_area_error',0.)>.10)
                    and t-getattr(self,'last_subject_feedback_s',-float('inf'))>=.5):
                self.last_subject_feedback_s=t
                self._write('subject_state_feedback',t,comparison=comparison)
                self._feedback('measured_subject_frame_mismatch',t,finish_pipeline=False)
        self._runtime_deadline(t)
        if self.runtime_fault is not None or self.mode == 'strict' or t >= self.original.duration:
            return
        if self.math_future is not None and self.math_future.done():
            future, kind = self.math_future, self.math_kind
            self.math_future = None
            try:
                result = future.result()
            except Exception as error:
                result = dict(error=f'{type(error).__name__}: {error}')
            if kind == 'candidates':
                self.validation_attempts += 1
            if result.get('error'):
                self._runtime_failure('planning_failure: '+str(result['error']), t)
                if kind == 'candidates':
                    from collections import Counter
                    evaluations=result.get('evaluations',[])
                    self.search_feedback.append(dict(time_s=t,
                        activation_s=result.get('activation_s'),
                        reasons=dict(Counter(e.get('rejected','passed') for e in evaluations)),
                        untested_proposals=result.get('untested_proposals'), search_exhausted=result.get('search_exhausted'),
                        failures=[dict(program=e.get('program'),reason=e.get('rejected'),
                            first_failure_s=e.get('kinematics',{}).get('first_failure_s')) for e in evaluations[:6]]))
                    self.search_feedback=self.search_feedback[-3:]
                if kind == 'verify':
                    self.rejected_stale += 1
                self._feedback(result['error'], t)
                self._write('math_failure', t, kind=kind,
                            checks=result.get('checks'), visual_checks=result.get('visual_checks'),
                            common_prefix=result.get('common_prefix'),
                            kinematics=result.get('kinematics'),construction_diagnostics=result.get('construction_diagnostics'),kinematic_prior=result.get('kinematic_prior'),
                            untested_proposals=result.get('untested_proposals'),search_exhausted=result.get('search_exhausted'),
                            evaluations=result.get('evaluations'))
            elif kind == 'diagnose':
                encoded=result.pop('_encoded_journal',None)
                self.report = result
                if encoded is None:self._write('whole_body_difficulty',t,**result)
                else:
                    # Serialization runs in the numerical worker; the servo
                    # only appends its receipt time to the same full evidence.
                    self.events.write('{"event":"whole_body_difficulty","time_s":'+str(float(t))+','+encoded[1:]+'\n')
                    self.events.flush()
                if (self.planner_model=='floating_camera' and self.active is not self.original
                        and not result['report']['issues']):
                    kept=[]
                    for feedback in self.pending_feedback:
                        report=feedback.get('wbc_feasibility',{})
                        # Use the existing FK tracking tolerance for a local
                        # task-velocity disagreement with no reported proximity
                        # conflict. A hard 5 mm recovery threshold
                        # used to contradict a fresh, feasible 15 mm gate and
                        # replace a safe curve for a 5.4 mm transient error.
                        local_only=(feedback.get('error')=='trajectory_preserving_configuration_degraded'
                            and 'barrier_mode' in report
                            and report.get('first_predicted_conflict_s') is None and not report.get('pairs'))
                        recovery_limit=KINEMATIC_POLICY.position_tolerance_m if local_only else .005
                        recovered=(result['report'].get('actual_tracking_error_m',float('inf'))<recovery_limit
                            and result.get('kinematics',{}).get('feasible')
                            and result['snapshot_time_s']-feedback.get('time_s',t)>=.3
                            and feedback.get('error') in ('trajectory_preserving_configuration_degraded',
                                'no_verified_executable_candidate'))
                        if recovered:
                            self._write('recovered_execution_feedback_acknowledged',t,feedback=feedback,
                                reason='fresh_safe_diagnosis_and_measured_tracking_recovered')
                            continue
                        if (feedback.get('error')=='trajectory_preserving_configuration_warning'
                                and report.get('camera_task_feasible')
                                and report.get('first_predicted_conflict_s') is None):
                            self._write('configuration_warning_acknowledged',t,feedback=feedback,
                                reason='fresh_diagnosis_and_actual_tracking_preserve_approved_curve')
                        else:kept.append(feedback)
                    self.pending_feedback=kept
                if t-result['snapshot_time_s'] > self.budget.max_prediction_age_s:
                    self._feedback('diagnosis_expired', t)
                elif (self.active is not self.original and
                        needs_configuration_confirmation(result,self.pending_feedback,t)):
                    self._write('configuration_confirmation_pending',t,
                        snapshot_time_s=result['snapshot_time_s'],
                        reason='transient_local_model_disagreement_requires_fresh_diagnosis',
                        feedback=self.pending_feedback)
                elif (result['report']['issues'] or self.pending_feedback):
                    self.context = self._evidence(t)
                    signature = (self.context['event'], self.context['phase']['name'],
                        tuple(result['report']['issues']), path_hash(self.active),
                        self.pending_feedback[-1]['error'] if self.pending_feedback else '')
                    # Retry unchanged evidence at a bounded rate after a refusal/timeout.
                    if signature != self.last_event_signature or t-self.last_attempt_s >= 2*self.cooldown:
                        self.last_event_signature = signature
                        self.last_attempt_s = t
                        self.pipeline_started_wall = time.monotonic()
                        evidence = dict(self.context, stage='strategy')
                        self.jev.submit(evidence)
                        self._write('strategy_requested', t, evidence=evidence)
            elif kind == 'candidates':
                candidates = result['candidates']
                self.feasible_plans += bool(candidates)
                self.pending_candidates = {c.id: c for c in candidates if t < c.path.activation_s}
                self._write('candidate_set', t, cards=[c.card() for c in candidates],
                    evaluations=result['evaluations'], planner_wall_s=result['wall_s'],seed_wall_s=result.get('seed_wall_s'),
                    untested_proposals=result.get('untested_proposals'), search_exhausted=result.get('search_exhausted'),
                    kinematic_prior=result.get('kinematic_prior'),construction_plan=result.get('construction_plan'),
                    generated_proposals=result.get('generated_proposals'))
                if not self.pending_candidates:
                    self.rejected_stale += 1
                    self._feedback('all_candidates_expired_before_selection', t)
                else:
                    self.context = self._evidence(t)
                    evidence = dict(self.context, stage='selection',
                        candidates=[c.card() for c in self.pending_candidates.values()])
                    self.jev.submit(evidence)
                    self._write('selection_requested', t, evidence=evidence)
            elif kind == 'verify':
                error = installation_guard(result, self.active, scene, state, t, self.budget,
                    measured_control=getattr(self.wbc,'measured_control',None),previous=self.wbc.previous,
                    commanded_q=commanded_q,base_rotation=getattr(self,'planning_base_rotation',None),base_z=base_z)
                error = error or support_violation(result['approval'], self.perception.snapshot(t))
                if error:
                    self.rejected_stale += 1
                    refreshed=self._refresh_verification(result,error,scene,state,t,base_z,commanded_q)
                    if not refreshed:self._feedback(error, t)
                    self._write('installation_rejected', t, error=error,
                                diagnostics=result.get('installation_diagnostics'),fresh_verification_pending=refreshed)
                else:
                    candidate = result['candidate']
                    self.active = candidate.path
                    self.active_candidate = candidate
                    self.wbc.set_trajectory(self.active)
                    self.wbc.kinematic_reference=result.get('kinematics')
                    execution_forecast=(result.get('kinematics') or {}).get('execution_prediction')
                    if execution_forecast is not None:
                        self._write('execution_prediction_installed',t,candidate_id=candidate.id,prediction=execution_forecast)
                    self.wbc.geometry_guidance=result.get('geometry_guidance')
                    self.configuration_monitor.reset(candidate.id,result.get('kinematics'))
                    self.wbc.input_generation = getattr(self.wbc, 'input_generation', 0)+1
                    self.wbc.route, self.wbc.route_map_time = None, -np.inf
                    self.wbc.comfort_cache.clear()
                    self.wbc.guide_costs.clear()
                    self.last_install_s = t
                    self.installed_plans += 1
                    self.runtime_retries=0
                    self.runtime_recovery_started=None
                    self.pending_feedback.clear()
                    latency = time.monotonic()-self.pipeline_started_wall
                    self.pipeline_latencies.append(latency)
                    self.pipeline_outcomes.append(dict(outcome='installed', time_s=t, wall_s=latency))
                    self.pipeline_started_wall = None
                    check = result['validation'].to_dict()
                    check.pop('camera_trace', None)
                    self._write('plan_installed', t, candidate_id=candidate.id,
                        approval=asdict(result['approval']),
                        envelope=(dict(program=self.active.program) if hasattr(self.active,'program') else self.active.envelope.to_dict()),
                        activation_s=self.active.activation_s, optimizer_return_s=self.active.return_s,
                        actual_parameters=dict(zip(PARAMETERS, self.active.values)),
                        checks=result['checks'], visual_checks=result['visual_checks'],
                        installation_diagnostics=result.get('installation_diagnostics'),
                        latest_state_check=check, end_to_end_wall_s=latency)
                    self._write('configuration_preparation_planned',t,candidate_id=candidate.id,
                        preparation=result.get('kinematics',{}).get('preparation'),
                        accepted_intent_violations=result.get('intent',{}).get('intent_violations',[]))
        if self.runtime_fault is not None:
            return
        semantic = self.jev.poll()
        if semantic is not None:
            if semantic.get('stage') == 'strategy' and semantic.get('bundled_construction'):
                self._write('search_strategy', t, **semantic)
                semantic = dict(semantic, stage='configuration')
            if 'error' in semantic:
                self._runtime_failure('semantic_failure: '+str(semantic['error']), t)
                self._feedback('semantic: '+semantic['error'], t)
            elif semantic['stage'] == 'strategy':
                self._write('search_strategy', t, **semantic)
                if semantic['strategy'] == 'refuse':
                    self._feedback('search_refused', t)
                elif self.planner_model=='floating_camera':
                    from .search_configuration import offers
                    options=offers(semantic['strategy'],self.report.get('robot_capability'),self.search_feedback)
                    evidence=dict(self._evidence(t),stage='configuration',search_options=options)
                    self.jev.submit(evidence)
                    self._write('search_configuration_requested',t,evidence=evidence)
                else:
                    self.pipeline_parent = path_hash(self.active)
                    self._submit_math('candidates', build_candidates, self.original, self.active,
                        semantic['strategy'], scene, np.asarray(state).copy(), self.wbc.previous.copy(),
                        t, base_z, self.planning_limits, evidence=self.perception.snapshot(t),
                        intrinsics=self.intrinsics, budget=self.budget,
                        commanded_q=None if commanded_q is None else np.asarray(commanded_q).copy())
            elif semantic['stage']=='configuration':
                self._write('search_configuration',t,**semantic)
                self.pipeline_parent=path_hash(self.active)
                self._submit_math('candidates',build_candidates,self.original,self.active,
                    semantic['strategy'],scene,np.asarray(state).copy(),self.wbc.previous.copy(),
                    t,base_z,self.planning_limits,evidence=self.perception.snapshot(t),
                    intrinsics=self.intrinsics,budget=self.budget,search_config=semantic['search_config'],
                    **({'construction_plan':semantic['construction_plan']} if 'construction_plan' in semantic else {}),
                    commanded_q=None if commanded_q is None else np.asarray(commanded_q).copy())
            elif semantic['stage'] == 'selection':
                candidate = self.pending_candidates.get(semantic['candidate_id'])
                geometry_request={} if candidate is None else candidate.consequences.get('geometry',{}).get('request',{})
                changes_geometry=bool(geometry_request.get('pose_preferences') or geometry_request.get('occupancy_preferences'))
                if candidate is None:
                    self._feedback('selection_not_in_pending_candidate_set', t)
                elif not changes_geometry and ((hasattr(candidate.path,'program') and candidate.path.program['route']=='continuation') or
                      (not hasattr(candidate.path,'program') and self.active is self.original and not np.any(candidate.path.values))):
                    # Selecting the unchanged target is a continuation decision,
                    # not an installed repair or evidence of task success.
                    self._write('continuation_selected', t, candidate_id=candidate.id, decision=semantic)
                    self.pipeline_outcomes.append(dict(outcome='continued', time_s=t,
                        wall_s=time.monotonic()-self.pipeline_started_wall))
                    self.pipeline_started_wall = None
                    self.last_attempt_s = t
                else:
                    approval = Approval(candidate.id, digest(candidate.card()), self.pipeline_parent,
                        self.original.sha256, semantic['selected_probability'], self.context['revision'],
                        {key: self.context['subject_evidence'][key]['status'] for key in ('visual', 'geometry')})
                    self._write('candidate_approved', t, approval=asdict(approval), decision=semantic)
                    self._submit_math('verify', verify_candidate, candidate, approval, self.original,
                        scene, np.asarray(state).copy(), self.wbc.previous.copy(), t, base_z, self.planning_limits,
                        evidence=self.perception.snapshot(t), intrinsics=self.intrinsics, budget=self.budget,
                        commanded_q=None if commanded_q is None else np.asarray(commanded_q).copy())
                self.pending_candidates = {}
        if (self.runtime_fault is None and t>=self.runtime_retry_after and self.math_future is None and self.jev.future is None and
                t-max(self.last_report_s, self.last_attempt_s) >= self.cooldown and
                self.original.duration-t > self.budget.activation_lead_s+.8):
            self._submit_math('diagnose', diagnose_bounded, self.original, scene, np.asarray(state).copy(),
                self.wbc.previous.copy(), t, base_z, self.planning_limits,
                commanded_q=None if commanded_q is None else np.asarray(commanded_q).copy())
            self.last_report_s = t
    def warm_workers(self):
        self.pool.submit(initialize_worker).result(timeout=30.)
        for name in ('coordination','reconfiguration'):
            worker=getattr(self.wbc,name,None)
            if hasattr(worker,'warm'):worker.warm()

    def step(self, state, scene, t, dt, base_z=0., base_rotation=None, commanded_q=None, measured_control=None,
             measured_base_twist=None):
        started = time.monotonic()
        cpu_started=time.thread_time()
        self.wbc.measured_control = None if measured_control is None else np.asarray(measured_control, float).copy()
        self.planning_base_rotation = None if base_rotation is None else np.asarray(base_rotation,float).copy()
        self.wbc.measured_base_twist = None if measured_base_twist is None else np.asarray(measured_base_twist, float).copy()
        self.update(state, scene, t, base_z, commanded_q)
        updated_wall=time.monotonic()
        target, rotation, feedforward = self.active.sample(t)
        brake = (self.runtime_fault is not None or t >= self.original.duration
                 or t-float(scene['observation_stamp']) > .6)
        if brake and t < self.original.duration:
            self._write('sensor_stale_braking', t, observation_stamp=scene['observation_stamp'])
        try:
            command, info = solve_with_certificate(self.wbc, state, target, rotation, feedforward, dt,
                base_z=base_z, base_rotation=base_rotation, commanded_q=commanded_q,
                trajectory_time=t, brake=brake)
        except RuntimeError as error:
            from .execution_prediction import failure_evidence
            self._write('wbc_feasibility',t,status='failed',camera_task_feasible=False,
                        trajectory_id=path_hash(self.active),reason=str(error),
                        failure=failure_evidence(str(error),t,source='controller_constraint_check'),
                        diagnostics=getattr(error,'diagnostics',None),
                        safety_guaranteed=False,command_published=False)
            raise
        solved_wall=time.monotonic()
        self.last_execution_certificate=dict(time_s=t,
            clearance_m=float(info.get('command_sweep_lower_m',-float('inf'))),
            self_clearance_m=float(info.get('self_clearance_lower_m',-float('inf'))))
        guide = getattr(self.wbc,'geometry_guidance',None)
        if self.geometry_feedback_future is not None and self.geometry_feedback_future.done():
            future,self.geometry_feedback_future=self.geometry_feedback_future,None
            try:
                feedback,encoded=future.result()
                self.events.write('{"event":"geometry_execution_feedback","time_s":'+str(float(t))+','+encoded[1:]+'\n')
                self.events.flush()
                if guide is not None and feedback['request_id']==guide.request.id:
                    info['geometry_feedback']={k:v for k,v in feedback.items() if k!='geometry'}
            except Exception as error:
                self._write('geometry_feedback_failed',t,reason=str(error))
        if (guide is not None and guide.request.start_s <= t <= guide.request.end_s
                and t-self.last_geometry_feedback_s >= .5 and self.geometry_feedback_future is None):
            if self.geometry_feedback_async:
                from copy import copy
                snapshot=copy(guide);snapshot.camera_path=None
                self.geometry_feedback_future=self.pool.submit(geometry_feedback_snapshot,snapshot,
                    self.geometry_backend.box_budget,np.asarray(state).copy(),t,base_z,
                    None if base_rotation is None else np.asarray(base_rotation).copy())
            else:
                snapshot = self.geometry_backend.observe(state,t,base_z=base_z,base_rotation=base_rotation)
                feedback = guide.feedback(snapshot,'measured')
                self._write('geometry_execution_feedback',t,**feedback)
                info['geometry_feedback'] = {k:v for k,v in feedback.items() if k!='geometry'}
            self.last_geometry_feedback_s = t
        self.wbc_degraded_samples += int(info.get('tpr_status') == 'degraded')
        preparation=self.configuration_monitor.update(state,
            measured_control if measured_control is not None else self.wbc.previous,t,info,self.wbc.limits)
        info['configuration_preparation']=preparation
        if preparation['status']!=self.last_configuration_status:
            self._write('configuration_preparation_state',t,**preparation)
            self.last_configuration_status=preparation['status']
        if info.get('tpr_status') in ('warning', 'degraded'):
            if not self.pending_feedback or t-self.pending_feedback[-1].get('time_s', -1.) >= .5:
                report = info['tpr_report']
                from .execution_prediction import failure_evidence
                self.pending_feedback.append(dict(error='trajectory_preserving_configuration_'+info['tpr_status'],
                    time_s=t,wbc_feasibility=report,failure=failure_evidence(
                        'trajectory_preserving_configuration_'+info['tpr_status'],t,source='controller_constraint_check')))
                self._write('wbc_feasibility', t, **report)
        if 'primary_qp_failure' in info:
            self._write('certified_safety_fallback', t, reason=info['primary_qp_failure'],
                        clearance_m=info['command_sweep_lower_m'])
            if not self.pending_feedback or t-self.pending_feedback[-1].get('time_s', -1.) >= .5:
                self.pending_feedback.append(dict(error=info['primary_qp_failure'], time_s=t))
        p, r, _ = whole_body_kinematics(state, base_z, base_rotation)
        if self.camera_output is not None:
            p, r = self.camera_output['position'], self.camera_output['rotation']
        original_p, original_r, _ = self.original.sample(t)
        parameters = np.zeros(len(PARAMETERS)) if self.active is self.original else self.active.parameters(t)
        info.update(jev_busy=self.jev.future is not None, math_busy=self.math_future is not None,
                    runtime_fault=self.runtime_fault,
                    active_decision='' if self.active is self.original else self.active.envelope.decision_id,
                    observed_visual_valid=self.observed_visual_valid,
                    parameters=parameters.tolist(), original_position=original_p.tolist(),
                    target_position=target.tolist(), joint_margin_rad=joint_margin(np.asarray(state)),
                    loop_wall_s=time.monotonic()-started, fixed_progress_s=min(t, self.original.duration))
        if not self.history or t-self.history[-1]['time_s'] >= .5:
            reference_view = original_p-self.original.subject
            actual_view = p-self.original.subject
            denom = max(np.linalg.norm(reference_view)*np.linalg.norm(actual_view), 1e-9)
            self.history.append(dict(time_s=t, original_position_error_m=float(np.linalg.norm(p-original_p)),
                camera_output_source='execution' if self.camera_output is None else self.camera_output['source'],
                original_orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(r@original_r.T).as_rotvec())),
                target_tracking_residual_m=float(np.linalg.norm(p-target)),
                view_angle_proxy_rad=float(np.arccos(np.clip(reference_view@actual_view/denom, -1., 1.))),
                area_scale_proxy_ratio=float((np.linalg.norm(reference_view)/max(np.linalg.norm(actual_view), 1e-6))**2),
                visual_proxy_source='runtime subject position and declared camera output pose; no SAM; not evaluation masks',
                clearance_m=info.get('command_sweep_lower_m'), joint_margin_rad=info['joint_margin_rad'],
                actual_parameters=parameters.tolist(), remaining_budget_s=max(0., self.original.duration-t)))
        info['control_thread_cpu_s']=time.thread_time()-cpu_started
        info['engine_stage_wall_s']=dict(observation_and_agent=updated_wall-started,
            control=solved_wall-updated_wall,feedback_and_record=time.monotonic()-solved_wall)
        return command, info

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.pipeline_started_wall is not None:
            self.pipeline_outcomes.append(dict(outcome='unfinished_at_close',
                wall_s=time.monotonic()-self.pipeline_started_wall))
            self.pipeline_started_wall = None
        self.jev.close()
        if self.geometry_feedback_future is not None and self.runtime_policy is None:
            try:
                feedback,_=self.geometry_feedback_future.result(timeout=30.)
                self._write('geometry_execution_feedback',(self.actual_feedback or {}).get('time_s',feedback['stamp_s']),
                    **feedback,drained_at_close=True)
            except Exception as error:
                self._write('geometry_feedback_failed',self.last_geometry_feedback_s,reason=str(error))
            self.geometry_feedback_future=None
        if self.math_future:
            self.math_future.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
        if hasattr(self.wbc, 'close'):
            self.wbc.close()
        self.events.close()

    def summary(self):
        return dict(mode=self.mode, planner_model=self.planner_model, semantic_model=self.semantic_model, jev_calls=self.jev.decider.client.calls,
            runtime_fault=self.runtime_fault,
            kinematic_gate='fk_constrained_continuous_lift/v1' if self.planner_model=='floating_camera' else None,
            wbc_algorithm='TPR-WBC' if self.limits.obstacle_avoidance else 'tracking-only',
            wbc_degraded_samples=self.wbc_degraded_samples,
            validation_attempts=self.validation_attempts, feasible_plans=self.feasible_plans,
            planning_feasibility_rate=None if not self.validation_attempts else self.feasible_plans/self.validation_attempts,
            representation='jev_geometry/v1', representation_boxes=self.geometry_backend.box_budget,
            geometry_preferences=self.geometry_preferences, installed_plans=self.installed_plans, stale_rejections=self.rejected_stale,
            original_sha256=self.original.sha256, original_duration_s=self.original.duration,
            architecture='exact_counterfactual/v2', agent_period_s=self.cooldown,
            end_to_end_wall_s=self.pipeline_latencies, decision_outcomes=list(self.pipeline_outcomes),
            pending_decision_wall_s=None if self.pipeline_started_wall is None else time.monotonic()-self.pipeline_started_wall,
            observed_visual_valid=self.observed_visual_valid)
