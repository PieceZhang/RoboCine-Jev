"""Bounded counterfactuals, exact-plan approval and independent installation gates.

Search domains are NOT semantic permissions. An Envelope is used to construct a
proposal; only a hash-bound Approval allows that proposal into the controller.
"""
from dataclasses import dataclass, field
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .contracts import Envelope, PARAMETERS, digest, native
from .planning import (physical_caps, rank_candidate, validate_rollout,
                       permission_violation, unrelaxed_prediction)
from .reference import DeformedReference
from .reference import look_at
from .dynamic_collision import DynamicCollision
from .whole_body_qp import whole_body_kinematics
from .kc_svsdf import advance
from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER
from .perception import visual_consequences


STRATEGIES = {
    'framing': 'Explore radius/height changes with original screen framing compensation; compare scale and view costs.',
    'view': 'Explore lateral/height changes with framing compensation; compare view angle and next-phase reserves.',
    'joint': 'Explore coupled radial-height and lateral-height repairs; combinations need their own approval.',
    'restore': 'Compare keeping the active plan with a smooth return to the immutable original.',
}


@dataclass(frozen=True)
class SearchBudget:
    max_proposals: int = 8
    max_rollouts: int = 3
    activation_lead_s: float = 4.
    prediction_step_s: float = .1
    max_prediction_age_s: float = 4.
    max_install_age_s: float = 1.2
    max_horizon_s: float = 8.
    decision_period_s: float = .5
    candidate_wall_budget_s: float = 2.
    # Numerical model-consistency allowances, NOT authored shot tolerances.
    composition_slack: float = .015
    scale_slack: float = .04
    angle_slack_rad: float = .03
    progress_slack: float = .04

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (self.max_proposals, self.max_rollouts)):
            raise ValueError('candidate counts must be integers')
        if not 2 <= self.max_rollouts <= self.max_proposals <= 16:
            raise ValueError('candidate budget must satisfy 2 <= rollouts <= proposals <= 16')
        if not .04 <= self.prediction_step_s <= .1:
            raise ValueError('prediction step outside bounded range')
        if not .2 <= self.decision_period_s <= 5.:
            raise ValueError('agent period must be in [.2, 5] seconds, independent of WBC')
        if not .2 <= self.activation_lead_s <= 5.:
            raise ValueError('activation lead must be in [.2, 5] seconds')
        if not self.activation_lead_s+.6 <= self.max_horizon_s <= 12.:
            raise ValueError('bounded horizon must accommodate activation and be at most 12 seconds')
        for value in (self.max_prediction_age_s, self.max_install_age_s, self.candidate_wall_budget_s,self.composition_slack,
                      self.scale_slack, self.angle_slack_rad, self.progress_slack):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('all timing and consistency allowances must be finite and positive')


def path_spec(path):
    if hasattr(path, 'to_spec'):
        return path.to_spec()
    if not isinstance(path, DeformedReference):
        return dict(original_sha256=path.sha256)
    return dict(original_sha256=path.original.sha256, values=path.values.tolist(),
        activation_s=path.activation_s, return_s=path.return_s, envelope=path.envelope.to_dict(),
        previous=path_spec(path.previous) if path.previous is not None else None)


def restore_path(original, spec):
    """Reconstruct an exact logged target, rejecting a different original task."""
    if spec['original_sha256'] != original.sha256:
        raise ValueError('replay original reference hash mismatch')
    if spec.get('kind') == 'intent_program/v1':
        from .intent_paths import IntentPath
        return IntentPath.from_spec(original, spec)
    if spec.get('kind') == 'fk_camera_program/v1':
        from .fk_camera_path import FKCameraPath
        return FKCameraPath.from_spec(original,spec)
    if 'values' not in spec:
        return original
    from .contracts import Envelope
    return DeformedReference(original, Envelope(**spec['envelope']), spec['values'],
        previous=restore_path(original,spec['previous']) if spec.get('previous') else None,
        activation_s=spec['activation_s'],return_s=spec.get('return_s'))


def path_hash(path):
    return digest(path_spec(path))


@dataclass
class Candidate:
    path: object
    validation: object
    consequences: dict
    snapshot_time_s: float
    strategy: str
    id: str = ''
    kinematics: dict | None = None

    def __post_init__(self):
        self.id = path_hash(self.path)

    def card(self):
        numeric = self.validation.to_dict()
        numeric.pop('camera_trace', None)
        numeric.pop('terminal_state', None)
        consequences = {k: v for k, v in self.consequences.items() if k != 'samples'}
        intent_path = hasattr(self.path, 'program')
        card = dict(id=self.id, strategy=self.strategy, trajectory=path_spec(self.path),
            physical_valid=self.validation.feasible, prediction=numeric,
            visual=consequences, valid_before_s=self.path.activation_s,
            approval_scope=('exact camera motion program, directed intent intervals and physical timing; no WBC prediction'
                if intent_path else 'this exact camera target path and full activation/hold/recovery process only; robot execution is predicted, not prescribed'))
        if self.kinematics is not None:
            from .kinematic_feasibility import summary
            card['kinematic_feasibility'] = summary(self.kinematics)
        return native(card)


@dataclass(frozen=True)
class Approval:
    candidate_id: str
    card_sha256: str
    parent_id: str
    reference_sha256: str
    selected_probability: float | None
    evidence_revision: int
    observation_support: dict = field(default_factory=dict)


def support_violation(approval, evidence):
    """Losing evidence used for approval requires a new semantic decision."""
    evidence = evidence or {}
    for source, status in approval.observation_support.items():
        if status == 'valid' and evidence.get(source, {}).get('status') != 'valid':
            return 'approved_observation_became_unknown:'+source
    return None


def diagnose_bounded(original, scene, state, previous, t, base_z, limits, commanded_q=None):
    report = unrelaxed_prediction(original, scene, state, previous, t, base_z, limits,
                                  horizon=5., step=.1, commanded_q=commanded_q)
    predicted = np.asarray(report.pop('conflict_state', report['predicted_state']))
    at = report['first_risk_s']
    report['local_repair_evidence'] = local_repair_evidence(
        original, scene, predicted, min(t+5., original.duration) if at is None else at, base_z)
    return dict(report=report, snapshot_time_s=t, physical_caps=physical_caps(state, limits, original.duration-t))


def local_repair_evidence(original, scene, state, t, base_z):
    """Nine inexpensive Jacobian perturbations at the predicted difficulty.

    These rank search directions, not trajectories: no execution permission,
    actuator feasibility, future visibility or safe path is inferred here.
    """
    p, rotation, jac = whole_body_kinematics(state, base_z)
    radial = p-original.subject
    radial[2] = 0.
    radial /= max(np.linalg.norm(radial), 1e-8)
    axes = np.column_stack([radial, rotation[:, 1], [0., 0., 1.]])
    collision = DynamicCollision(scene, t)
    initial_clearance = float(collision.clearance(state, base_z).min())
    result = []
    operators = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
                 (1,0,1), (1,0,-1), (0,1,1)]
    for coefficients in operators:
        offset = axes@np.asarray(coefficients)*.06
        aim = look_at(p+offset, original.subject)@look_at(p, original.subject).T
        differential = np.linalg.lstsq(jac, np.r_[offset, Rotation.from_matrix(aim).as_rotvec()], rcond=1e-4)[0]
        differential *= min(1., .15/max(abs(differential[1:]).max(), 1e-9))
        moved = advance(state, differential, 1.)
        if np.any(moved[3:] < JOINT_LOWER+.01) or np.any(moved[3:] > JOINT_UPPER-.01):
            continue
        actual = whole_body_kinematics(moved, base_z)[0]-p
        gain = float(collision.clearance(moved, base_z).min())-initial_clearance
        result.append(dict(direction=list(coefficients), clearance_change_m=gain,
            realized_translation_m=float(np.linalg.norm(actual)),
            linearization_residual_m=float(np.linalg.norm(actual-offset)),
            family='joint' if np.count_nonzero(coefficients)>1 else
                   'view' if coefficients[1] else 'framing'))
    return dict(at_s=float(t), baseline_clearance_m=initial_clearance, probes=result,
        meaning='local whole-body Jacobian sensitivity only; compute complete candidates before authorizing')


def propose_paths(original, active, strategy, t, budget, caps):
    """Shared task-space repair family; caps are supplied by the execution model."""
    activation = t + budget.activation_lead_s
    prediction_end = min(original.duration, t + budget.max_horizon_s)
    remaining = prediction_end - activation
    blend = min(1., remaining / 2.)
    current = None if active is original else active
    # Small finite family; no assumption that acceptance is monotone in magnitude.
    vectors = [np.zeros(6)]
    axes = {'framing': [(0, 1), (0, -1), (2, 1), (2, -1)],
            'view': [(1, 1), (1, -1), (2, 1), (2, -1)],
            'joint': [(0, 1, 2, 1), (0, 1, 2, -1), (1, 1, 2, 1), (1, -1, 2, 1)],
            'restore': []}[strategy]
    for fraction in (.12, .3):
        for operator in axes:
            v = np.zeros(6)
            for i in range(0, len(operator), 2):
                axis, sign = operator[i:i+2]
                v[axis] = sign*fraction*caps[PARAMETERS[axis]]
            vectors.append(v)
    proposals = []
    for i, values in enumerate(vectors):
        bounds = {name: [min(0., float(value)), max(0., float(value))]
                  for name, value in zip(PARAMETERS, values)}
        if current:
            for name, (lo, hi) in current.envelope.bounds.items():
                bounds[name] = [min(bounds[name][0], lo), max(bounds[name][1], hi)]
        env = Envelope(original.sha256, f'proposal-{t:.6f}-{strategy}-{i}', 'candidate_search',
            t, activation, prediction_end, blend, bounds,
            ('subject_aim', 'fixed_progress', 'horizon'),
            'retain' if prediction_end == original.duration else 'recover',
            rationale='Unapproved numerical proposal; exact candidate approval required',
            parent_id='' if current is None else current.envelope.decision_id)
        path = DeformedReference(original, env, values, current, activation)
        proposals.append(path)
    # Continuing the active path is the explicit 'refuse change' action and does
    # not consume a counterfactual branch. Reserve one proposal for recovery.
    if len(vectors) > 1:
        last = proposals[-1]
        proposals = proposals[:budget.max_proposals-1]
        proposals.append(DeformedReference(original, last.envelope, last.values, current,
                                           activation, prediction_end-blend))
    return proposals


def build_candidates(original, active, strategy, scene, state, previous, t, base_z, limits,
                     evidence=None, intrinsics=None, budget=None, commanded_q=None):
    budget = budget or SearchBudget()
    begin = time.monotonic()
    if strategy not in STRATEGIES:
        return dict(error='unknown_search_strategy')
    activation = t+budget.activation_lead_s
    prediction_end = min(original.duration, t+budget.max_horizon_s)
    remaining = prediction_end-activation
    if remaining < .6:
        return dict(error='insufficient_time_for_exact_plan_activation')
    blend = min(1., remaining/2.)
    caps = physical_caps(state, limits, original.duration-t)
    proposals = propose_paths(original, active, strategy, t, budget, caps)
    # Every proposal follows active identically before activation. Compute that
    # prefix once, then rank AFTER activation, where the operators actually differ.
    prefix = {}
    common = validate_rollout(active, scene, state, previous, t, base_z, limits,
        step=budget.prediction_step_s, horizon=activation-t, commanded_q=commanded_q,
        record_trace=True, checkpoint=prefix)
    if not common.feasible:
        detail = common.to_dict()
        detail.pop('camera_trace', None)
        return dict(error='common_prefix_not_physically_valid', evaluations=[],
                    common_prefix=detail, wall_s=time.monotonic()-begin)
    ranked, evaluations = [], []
    for path in proposals[:budget.max_proposals]:
        coarse = rank_candidate(path, scene, prefix['state'], prefix['controller'].previous,
                                activation, base_z, limits, horizon=min(2.4, remaining),
                                commanded_q=prefix['commanded_q'])
        evaluations.append(dict(id=path_hash(path), coarse=coarse.to_dict()))
        if coarse.feasible:
            ranked.append((coarse.cost, path))
    # Keep zero/continuation, then diverse numerical alternatives; bounded expensive work.
    mandatory = [p for _, p in ranked if np.linalg.norm(p.values) < 1e-8]
    finalists = mandatory[:1]
    supports = set()
    for _, path in sorted(ranked, key=lambda pair: pair[0]):
        if len(finalists) >= budget.max_rollouts:
            break
        support = (tuple(np.flatnonzero(path.values)), path.return_s is not None)
        if path not in finalists and support not in supports:
            finalists.append(path)
            supports.add(support)
    candidates = []
    for path in finalists:
        if path is active:
            continue
        full = validate_rollout(path, scene, state, previous, t, base_z, limits,
            step=budget.prediction_step_s, horizon=prediction_end-t,
            check_terminal=prediction_end == original.duration, commanded_q=commanded_q,
            record_trace=True, resume=prefix)
        detail = full.to_dict()
        detail.pop('camera_trace', None)
        evaluations.append(dict(id=path_hash(path), full=detail))
        task_preserved = (full.last_task_degradation_s is None or
                          full.last_task_degradation_s < activation-1e-9)
        if full.feasible and task_preserved:
            consequences = visual_consequences(original, full.camera_trace, evidence, intrinsics)
            candidate = Candidate(path, full, consequences, t, strategy)
            # Continuing an already active exact plan does not have a future activation.
            # It is a refusal-to-change alternative, not a new install candidate.
            if path is not active:
                candidates.append(candidate)
    return dict(candidates=candidates, evaluations=evaluations, snapshot_time_s=t,
                wall_s=time.monotonic()-begin,
                error='' if candidates else 'no_candidate_passed_physical_validation')


def contract_comparison(approved, refreshed, budget):
    """No retrospective widening. Compare re-prediction to approved consequences."""
    gates = {}
    slacks = dict(composition_error=budget.composition_slack,
        range_scale_error=budget.scale_slack, view_angle_error_rad=budget.angle_slack_rad,
        orientation_error_rad=budget.angle_slack_rad, phase_progress_error=budget.progress_slack)
    for key, slack in slacks.items():
        a, b = approved['summary'][key], refreshed['summary'][key]
        gates[key] = 'unknown' if a is None or b is None else ('pass' if b <= a+slack else 'fail')
    old_duration = approved['summary'].get('center_out_of_frame_duration_s')
    new_duration = refreshed['summary'].get('center_out_of_frame_duration_s')
    gates['center_visibility'] = ('unknown' if old_duration is None or new_duration is None else
        'pass' if new_duration <= old_duration+1e-8 else 'fail')
    gates['occlusion'] = 'unknown'
    # Protect phase-local maxima too; a global maximum must not hide a new late-phase failure.
    for phase, value in refreshed['phases'].items():
        previous = approved['phases'].get(phase)
        if previous:
            for key, slack in slacks.items():
                a, b = previous[key], value[key]
                gates[phase+':'+key] = 'unknown' if a is None or b is None else ('pass' if b <= a+slack else 'fail')
    return gates


def verify_candidate(candidate, approval, original, scene, state, previous, t, base_z, limits,
                     evidence=None, intrinsics=None, budget=None, commanded_q=None):
    """Worker-only suffix prediction. Never block the control callback on a rollout."""
    budget = budget or SearchBudget()
    begin = time.monotonic()
    checks = dict(physical_valid='unknown', target_authorized='fail',
                  predicted_visual_valid='unknown', state_fresh='fail', observed_visual_valid='unknown')
    if (candidate.id != approval.candidate_id or path_hash(candidate.path) != approval.candidate_id or
            digest(candidate.card()) != approval.card_sha256 or original.sha256 != approval.reference_sha256):
        return dict(error='approval_binding_mismatch', checks=checks)
    checks['target_authorized'] = 'pass'
    support_error = support_violation(approval, evidence)
    if support_error:
        return dict(error=support_error, checks=checks)
    if not 0 <= t-candidate.snapshot_time_s <= budget.max_prediction_age_s or t >= candidate.path.activation_s:
        return dict(error='candidate_expired_no_retiming_allowed', checks=checks)
    if not -1e-8 <= t-float(scene['observation_stamp']) <= .6:
        return dict(error='scene_stale', checks=checks,
                    scene_age_s=t-float(scene['observation_stamp']))
    checks['state_fresh'] = 'pass'
    full = validate_rollout(candidate.path, scene, state, previous, t, base_z, limits,
        step=budget.prediction_step_s, horizon=candidate.path.envelope.end_s-t,
        check_terminal=candidate.path.envelope.end_s == original.duration,
        commanded_q=commanded_q, record_trace=True, fine_until_s=t+budget.max_install_age_s)
    checks['physical_valid'] = 'pass' if full.feasible else 'fail'
    task_preserved = (full.last_task_degradation_s is None or
                      full.last_task_degradation_s < max(t, candidate.path.activation_s)-1e-9)
    checks['wbc_task_preserved'] = 'pass' if task_preserved else 'fail'
    consequences = visual_consequences(original, full.camera_trace, evidence, intrinsics)
    visual = contract_comparison(candidate.consequences, consequences, budget)
    checks['predicted_visual_valid'] = ('fail' if 'fail' in visual.values() else
                                     'unknown' if 'unknown' in visual.values() else 'pass')
    error = '' if full.feasible and task_preserved and checks['predicted_visual_valid'] != 'fail' else 'latest_prediction_rejected'
    lost = any(value is not None and consequences['summary'].get(key) is None
               for key, value in candidate.consequences['summary'].items())
    lost |= (candidate.consequences['subject_source'] != 'runtime_initial_calibration' and
             consequences['subject_source'] == 'runtime_initial_calibration')
    if lost:
        error = 'approved_evidence_became_unknown'
    return dict(error=error, candidate=candidate, approval=approval, checks=checks, visual_checks=visual,
        validation=full, snapshot_time_s=t, scene=scene, wall_s=time.monotonic()-begin)


def installation_guard(result, active, scene, state, t, budget, *, measured_control=None, previous=None,
                       commanded_q=None, base_rotation=None, base_z=None):
    """Cheap bounded checks after the worker returns; no rebasing an approved path."""
    candidate, approval = result['candidate'], result['approval']
    if np.asarray(state).shape != (9,) or not np.isfinite(state).all() or not np.isfinite(t):
        return 'invalid_latest_robot_state'
    if result.get('error'):
        return result['error']
    if path_hash(candidate.path) != approval.candidate_id or digest(candidate.card()) != approval.card_sha256:
        return 'approval_binding_mismatch'
    if path_hash(active) != approval.parent_id:
        return 'active_parent_changed'
    if t >= candidate.path.activation_s or not 0 <= t-result['snapshot_time_s'] <= budget.max_install_age_s:
        return 'latest_verification_expired'
    if not -1e-8 <= t-float(scene['observation_stamp']) <= .6:
        return 'scene_stale'
    trace = result['validation'].camera_trace
    if not trace:
        return 'missing_execution_prediction'
    times = np.array([row['t'] for row in trace])
    if result.get('prediction_model') == 'floating_camera':
        latest_z = result['base_z'] if base_z is None else base_z
        p,r,_ = whole_body_kinematics(state,latest_z,base_rotation)
        positions = np.array([row['position'] for row in trace])
        expected = np.array([np.interp(t,times,positions[:,j]) for j in range(3)])
        from scipy.spatial.transform import Slerp
        predicted_rotation = Slerp(times,Rotation.from_quat([row['quaternion_xyzw'] for row in trace]))(np.clip(t,times[0],times[-1]))
        position_error = float(np.linalg.norm(p-expected))
        angle_error = float((Rotation.from_matrix(r)*predicted_rotation.inv()).magnitude())
        result['installation_diagnostics'] = dict(verification_age_s=t-result['snapshot_time_s'],
            camera_position_residual_m=position_error,camera_orientation_residual_rad=angle_error,
            whole_body_prediction='not_performed')
        if position_error > .04 or angle_error > .08:
            return 'camera_left_verified_state_neighborhood'
        if 'kinematics' in result:
            from .kinematic_feasibility import installation_error, KinematicSearch
            velocity = measured_control if measured_control is not None else previous
            if velocity is None:
                return 'missing_latest_velocity_for_kinematic_gate'
            error = installation_error(result['kinematics'], state, velocity, t, result['kinematic_limits'])
            if error:
                return error
            if velocity is not None:
                latest = KinematicSearch(state, previous if previous is not None else velocity,
                    t, latest_z, result['kinematic_limits'], measured_control=measured_control,
                    commanded_q=commanded_q, base_rotation=base_rotation)
                error = latest.initial_error()
                _, lo, hi = latest.constraints(latest.state, latest.velocity, .02, t,
                                               latest.commanded_q, initial=True)
                if error or np.any(lo > hi):
                    return 'latest_kinematic_state_invalid:'+(error or 'braking_or_acceleration_infeasible')
            result['installation_diagnostics']['kinematic_state'] = 'pass'
    else:
        predicted = np.array([row['state'] for row in trace])
        expected = np.array([np.interp(t, times, predicted[:, j]) for j in range(9)])
        error = np.asarray(state)-expected
        error[2] = (error[2]+np.pi)%(2*np.pi)-np.pi
        result['installation_diagnostics'] = dict(verification_age_s=t-result['snapshot_time_s'],
            base_position_residual_m=float(np.linalg.norm(error[:2])),
            max_angular_state_residual_rad=float(np.max(np.abs(error[2:]))), state_residual=error.tolist())
        if np.linalg.norm(error[:2]) > .04 or np.max(np.abs(error[2:])) > .08:
            return 'robot_left_verified_state_neighborhood'
    old = result['scene']
    # Compare observed occupancies after constant-velocity extrapolation. Fresh
    # additions invalidate the proposal even if the moving-track ID was reused.
    def cloud(snapshot):
        p = np.asarray(snapshot.get('surfel_centers', []), float).reshape(-1, 3)
        v = np.asarray(snapshot.get('surfel_velocities', np.zeros_like(p)), float)
        return p+v*(t-float(snapshot['observation_stamp']))
    a, b = cloud(old), cloud(scene)
    if max(len(a), len(b)) > 8192:
        return 'scene_exceeds_installation_check_budget'
    if bool(len(a)) != bool(len(b)):
        return 'scene_occupancy_changed'
    if len(a):
        distance, nearest = cKDTree(a).query(b)
        if max(distance.max(), cKDTree(b).query(a)[0].max()) > .06:
            return 'scene_occupancy_changed'
        old_radii = np.asarray(old.get('surfel_radii', np.full(len(a), old.get('surfel_radius', .025))))
        new_radii = np.asarray(scene.get('surfel_radii', np.full(len(b), scene.get('surfel_radius', .025))))
        if np.any(new_radii-old_radii[nearest] > .01):
            return 'scene_extent_changed'
        old_velocity = np.asarray(old.get('surfel_velocities', np.zeros_like(a)))
        new_velocity = np.asarray(scene.get('surfel_velocities', np.zeros_like(b)))
        if np.any(np.linalg.norm(new_velocity-old_velocity[nearest], axis=1) > .1):
            return 'scene_velocity_changed'
    if scene.get('minimum_clearance_m') != old.get('minimum_clearance_m'):
        return 'scene_safety_margin_changed'
    return None
