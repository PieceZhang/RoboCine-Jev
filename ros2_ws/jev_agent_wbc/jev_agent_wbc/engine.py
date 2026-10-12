"""Event-driven asynchronous Jev -> bounded planning -> independent safety WBC."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import native, PARAMETERS
from .jev import AsyncJev, SemanticDecider, SystemOneClient
from .planning import (unrelaxed_prediction, useful_directions, time_windows,
                       optimize_envelope, validate_rollout, permission_violation, joint_margin, ExecutionWBC)
from .dynamic_predictive_qp import AsyncDynamicPredictiveQP
from .reference import DeformedReference
from .shot_preview_worker import initialize_worker
from .whole_body_qp import Limits, whole_body_kinematics
from .safety import solve_with_certificate


def diagnose(original, scene, state, previous, t, base_z, limits, commanded_q=None):
    report = unrelaxed_prediction(original, scene, state, previous, t, base_z, limits, commanded_q=commanded_q)
    directions, caps = useful_directions(original, scene, state, t, base_z, limits, report, previous, commanded_q)
    return dict(report=report, directions=directions, caps=caps, snapshot_time_s=t)


class CinematicEngine:
    def __init__(self, original, scene, output, *, mode='jev', api_url=None, limits=None,
                 decision_cooldown_s=.8, preview=False):
        if mode not in ('jev', 'strict'):
            raise ValueError('mode must be jev or strict')
        self.original = self.active = original
        self.mode, self.limits = mode, limits or Limits()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = (self.output/'events.jsonl').open('w')
        client = SystemOneClient(journal=self.output/'jev_api.jsonl', **({} if api_url is None else {'url': api_url}))
        self.jev = AsyncJev(SemanticDecider(client))
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)
        self.math_future = None
        self.math_kind = ''
        controller = AsyncDynamicPredictiveQP if preview else ExecutionWBC
        self.wbc = controller(self.limits, scene=scene, horizon_steps=4, horizon_dt=.3, replan_s=.4)
        self.wbc.set_trajectory(original)
        self.last_report_s = -float('inf')
        self.last_decision_s = -float('inf')
        self.last_attempt_s = -float('inf')
        self.cooldown = decision_cooldown_s
        self.report = None
        self.history = []
        self.pending_feedback = []
        self.revision = 0
        self.last_install_s = -float('inf')
        self.closed = False
        self.failures = []
        self.validation_attempts = self.feasible_plans = self.installed_plans = 0
        self.rejected_stale = 0
        self.last_event_signature = None
        self._write('initialized', 0., mode=mode, original_sha256=original.sha256,
                    original_budget_s=original.duration, upstream_fields=list(original.task))

    def _write(self, event, t, **details):
        self.events.write(json.dumps(native(dict(event=event, time_s=t, **details)), ensure_ascii=False, allow_nan=False)+'\n')
        self.events.flush()

    def _submit_diagnosis(self, scene, state, t, base_z, commanded_q=None):
        self.math_kind = 'diagnose'
        self.math_future = self.pool.submit(diagnose, self.original, scene, np.asarray(state).copy(),
            self.wbc.previous.copy(), t, base_z, self.limits,
            None if commanded_q is None else np.asarray(commanded_q).copy())
        self.last_report_s = t

    def _evidence(self, diagnostic, scene, state, t):
        task = self.original.task
        event = 'difficulty'
        if self.pending_feedback:
            event = 'validation_failed'
        elif self.active is not self.original:
            event = 'recovery' if not diagnostic['report']['issues'] or t-self.last_install_s > 1.5 else 'difficulty'
        phase = self.original.phase(t)
        following = [p for p in task['shot_phase'] if p['start_s'] >= phase['end_s']]
        return dict(event=event, time_s=t, duration_s=self.original.duration, reference_sha256=self.original.sha256,
            phase=phase, next_phase=following[0] if following else None, subject=task['subject'], intent=task['intent'],
            reference_samples=[dict(t=float(s), position=self.original.sample(s)[0].tolist(),
                quaternion_xyzw=Rotation.from_matrix(self.original.sample(s)[1]).as_quat().tolist())
                for s in np.linspace(t, self.original.duration, 5)],
            runtime_subject_position=self.original.subject.tolist(),
            difficulty=diagnostic['report'], directions=diagnostic['directions'], physical_caps=diagnostic['caps'],
            robot_state=np.asarray(state).tolist(), runtime_tracks=scene.get('tracks', []),
            time_windows=time_windows(t, self.original.duration, diagnostic['report'], self.limits),
            active_envelope=None if self.active is self.original else self.active.envelope.to_dict(),
            current_absolute_parameters=dict(zip(PARAMETERS, np.zeros(len(PARAMETERS)) if self.active is self.original else self.active.parameters(t))),
            execution_history=self.history[-4:], validation_feedback=self.pending_feedback[-3:], revision=self.revision)

    def _event_signature(self, result, evidence):
        report = result['report']
        clearance = report['min_clearance_m']
        clearance_band = int(np.floor(clearance/.1)) if np.isfinite(clearance) else None
        return (evidence['event'], evidence['phase']['name'], tuple(report['issues']), clearance_band,
                int(np.floor(report['joint_margin_rad']/.1)),
                '' if self.active is self.original else self.active.envelope.decision_id,
                self.pending_feedback[-1].get('error', '') if self.pending_feedback else '')

    def update(self, state, scene, t, base_z, commanded_q=None):
        self.revision += 1
        self.wbc.update_collision_map(scene)
        self.wbc.set_time(t)
        if self.mode != 'jev' or t >= self.original.duration-.4:
            return
        if self.math_future is not None and self.math_future.done():
            future, kind = self.math_future, self.math_kind
            self.math_future = None
            try:
                result = future.result()
            except Exception as error:
                result = dict(error=f'{type(error).__name__}: {error}')
            if kind == 'diagnose':
                if 'error' in result:
                    self._write('diagnosis_error', t, detail=result)
                else:
                    self.report = result
                    self._write('whole_body_difficulty', t, **result)
                    wants_recovery = self.active is not self.original and t-self.last_install_s > 1.5
                    if ((result['report']['issues'] or self.pending_feedback or wants_recovery) and
                            t-self.last_attempt_s >= self.cooldown and self.jev.future is None):
                        evidence = self._evidence(result, scene, state, t)
                        signature = self._event_signature(result, evidence)
                        if signature != self.last_event_signature and self.jev.submit(evidence):
                            self.last_event_signature = signature
                            self.last_attempt_s = t
                            self._write('jev_requested', t, evidence=evidence)
            elif kind == 'optimize':
                self.validation_attempts += 1
                if 'error' in result:
                    self.pending_feedback.append(result)
                    self._write('planning_rejected', t, **result)
                else:
                    self.feasible_plans += 1
                    path = result['path']
                    # The original envelope authorizes an interval, not an exact onset.
                    # Rebase a late mathematical proposal within that interval, then
                    # recheck the latest observation/robot state before installation.
                    try:
                        activation = max(t+.02, path.envelope.start_s)
                        current = None if self.active is self.original else self.active
                        path = DeformedReference(self.original, path.envelope, path.values, current, activation, path.return_s)
                        semantic_error = permission_violation(path, t, self.original.duration)
                        if semantic_error:
                            raise ValueError(semantic_error)
                        verification = validate_rollout(path, scene, state, self.wbc.previous, t,
                            base_z, self.limits, horizon=min(.2, self.original.duration-t), step=.04,
                            commanded_q=commanded_q)
                        if not verification.feasible:
                            raise ValueError(verification.reason)
                        if t-result['snapshot_time_s'] > 2.4:
                            raise ValueError('whole-body prediction is older than its horizon')
                    except (ValueError, RuntimeError) as error:
                        self.rejected_stale += 1
                        feedback = dict(error='latest_state_validation: '+str(error))
                        self.pending_feedback.append(feedback)
                        self._write('latest_state_rejected', t, **feedback)
                    else:
                        self.active = path
                        self.wbc.set_trajectory(path)
                        self.wbc.input_generation = getattr(self.wbc, 'input_generation', 0)+1
                        self.wbc.route = None
                        self.wbc.route_map_time = -np.inf
                        self.wbc.comfort_cache.clear()
                        self.wbc.guide_costs.clear()
                        self.last_install_s = t
                        self.installed_plans += 1
                        self.pending_feedback.clear()
                        self._write('plan_installed', t, envelope=path.envelope.to_dict(), activation_s=path.activation_s,
                            optimizer_return_s=path.return_s,
                            actual_parameters=dict(zip(PARAMETERS, path.values)),
                            prediction=result['validation'].to_dict(), latest_state_check=verification.to_dict(),
                            optimizer_wall_s=result['wall_s'], evaluations=result['evaluations'])
        semantic = self.jev.poll()
        if semantic is not None:
            if 'error' in semantic:
                self._write('jev_unavailable_or_refused', t, **semantic)
                self.pending_feedback.append(dict(error=semantic['error'], time_s=t))
            elif self.math_future is None:
                envelope = semantic['envelope']
                self._write('jev_permission', t, envelope=envelope.to_dict(), wall_s=semantic['wall_s'], recovery=semantic['recovery'])
                self.math_kind = 'optimize'
                self.math_future = self.pool.submit(optimize_envelope, self.original, self.active,
                    envelope, semantic['recovery'], scene, np.asarray(state).copy(), self.wbc.previous.copy(),
                    t, base_z, self.limits, commanded_q=None if commanded_q is None else np.asarray(commanded_q).copy())
                self.last_decision_s = t
            else:
                self._write('permission_discarded_busy', t)
        if (self.math_future is None and self.jev.future is None and t-self.last_report_s >= self.cooldown
                and self.original.duration-t > .8):
            self._submit_diagnosis(scene, state, t, base_z, commanded_q)

    def step(self, state, scene, t, dt, base_z=0., base_rotation=None, commanded_q=None):
        started = time.monotonic()
        self.update(state, scene, t, base_z, commanded_q)
        target, rotation, feedforward = self.active.sample(t)
        brake = t >= self.original.duration or t-float(scene['observation_stamp']) > .6
        if brake and t < self.original.duration:
            self._write('sensor_stale_braking', t, observation_stamp=scene['observation_stamp'])
        command, info = solve_with_certificate(self.wbc, state, target, rotation, feedforward, dt,
            base_z=base_z, base_rotation=base_rotation, commanded_q=commanded_q,
            trajectory_time=t, brake=brake)
        if 'primary_qp_failure' in info:
            self._write('certified_safety_fallback', t, reason=info['primary_qp_failure'],
                        clearance_m=info['command_sweep_lower_m'])
            if not self.pending_feedback or t-self.pending_feedback[-1].get('time_s', -1.) >= .5:
                self.pending_feedback.append(dict(error=info['primary_qp_failure'], time_s=t))
        p, r, _ = whole_body_kinematics(state, base_z, base_rotation)
        original_p, original_r, _ = self.original.sample(t)
        parameters = np.zeros(len(PARAMETERS)) if self.active is self.original else self.active.parameters(t)
        info.update(jev_busy=self.jev.future is not None, math_busy=self.math_future is not None,
                    active_decision='' if self.active is self.original else self.active.envelope.decision_id,
                    parameters=parameters.tolist(), original_position=original_p.tolist(),
                    target_position=target.tolist(), joint_margin_rad=joint_margin(np.asarray(state)),
                    loop_wall_s=time.monotonic()-started, fixed_progress_s=min(t, self.original.duration))
        if not self.history or t-self.history[-1]['time_s'] >= .5:
            reference_view = original_p-self.original.subject
            actual_view = p-self.original.subject
            denom = max(np.linalg.norm(reference_view)*np.linalg.norm(actual_view), 1e-9)
            self.history.append(dict(time_s=t, original_position_error_m=float(np.linalg.norm(p-original_p)),
                original_orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(r@original_r.T).as_rotvec())),
                target_tracking_residual_m=float(np.linalg.norm(p-target)),
                view_angle_proxy_rad=float(np.arccos(np.clip(reference_view@actual_view/denom, -1., 1.))),
                area_scale_proxy_ratio=float((np.linalg.norm(reference_view)/max(np.linalg.norm(actual_view), 1e-6))**2),
                visual_proxy_source='runtime subject position and measured FK; no SAM; not evaluation masks',
                clearance_m=info.get('command_sweep_lower_m'), joint_margin_rad=info['joint_margin_rad'],
                actual_parameters=parameters.tolist(), remaining_budget_s=max(0., self.original.duration-t)))
        return command, info

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.jev.close()
        if self.math_future:
            self.math_future.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
        if hasattr(self.wbc, 'close'):
            self.wbc.close()
        self.events.close()

    def summary(self):
        return dict(mode=self.mode, jev_calls=self.jev.decider.client.calls,
            validation_attempts=self.validation_attempts, feasible_plans=self.feasible_plans,
            planning_feasibility_rate=None if not self.validation_attempts else self.feasible_plans/self.validation_attempts,
            installed_plans=self.installed_plans, stale_rejections=self.rejected_stale,
            original_sha256=self.original.sha256, original_duration_s=self.original.duration)
