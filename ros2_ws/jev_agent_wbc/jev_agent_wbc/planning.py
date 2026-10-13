"""Whole-body difficulty reports, useful directions and bounded trajectory search."""
from dataclasses import replace
from copy import deepcopy
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import PARAMETERS, Validation
from .reference import DeformedReference
from .whole_body_qp import WholeBodyQP, Limits, whole_body_kinematics
from .dynamic_predictive_qp import DynamicPredictiveQP
from .dynamic_collision import DynamicCollision
from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER
from .kc_svsdf import advance
from .safety import solve_with_certificate
from .actuation import advance_command, certify_control_ramp


def joint_margin(state):
    return float(np.min(np.minimum(state[3:]-JOINT_LOWER, JOINT_UPPER-state[3:])))


class ExecutionWBC(DynamicPredictiveQP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .coordination import CoordinationPreview
        self.coordination = CoordinationPreview()
        self.coordination_reference = None
        self.measured_control = None
        self.measured_base_twist = None
        from .tpr_wbc import ReconfigurationPreview
        self.reconfiguration = ReconfigurationPreview()
        self.trajectory_preserving = self.limits.obstacle_avoidance
        from .self_collision import SelfCollision
        self.self_collision = SelfCollision()

    def set_trajectory(self, trajectory):
        super().set_trajectory(trajectory)
        from .candidates import path_hash
        self.received_trajectory_id=path_hash(trajectory) if hasattr(trajectory,'sha256') else 'received'
        self.coordination.set_trajectory(trajectory)
        self.reconfiguration.reset(trajectory)
        if self.coordination.enabled or self.trajectory_preserving:
            # The fast QP must retain the original deadline even while the
            # asynchronous allocator is late or unavailable.
            self.limits = replace(self.limits, deadline_s=trajectory.duration)

    def solve(self, state, target_p, target_r, feedforward, dt, **kwargs):
        t = kwargs.get('trajectory_time', 0.)
        # Candidate activation is also the receipt boundary for its predictive
        # configuration task. Before it, predict the currently active parent;
        # otherwise an uninstalled candidate changes the shared execution prefix.
        received = self.trajectory
        while isinstance(received, DeformedReference) and t < received.activation_s-1e-9:
            received = received.previous or received.original
        self.coordination.set_trajectory(received)
        self.reconfiguration.reset(received)
        old_command = self.previous.copy()
        self.coordination_reference = None
        if not kwargs.get('brake', False) and not kwargs.get('preparing', False):
            self.coordination_reference = self.coordination.update(
                np.asarray(state), self.previous,
                np.asarray(state)[3:] if kwargs.get('commanded_q') is None else np.asarray(kwargs['commanded_q']),
                t, dt, kwargs.get('base_z', 0.), self.limits, kwargs.get('base_rotation'),
                measured_control=self.measured_control)
        if self.trajectory_preserving:
            from .tpr_wbc import solve_tpr
            command, info = solve_tpr(self, state, target_p, target_r, feedforward, dt, **kwargs)
        else:
            command, info = super().solve(state, target_p, target_r, feedforward, dt, **kwargs)
        info.update(self.coordination.info)
        info['coordination_active'] = self.coordination_reference is not None
        if self.coordination_reference is not None:
            info['coordination_reference'] = self.coordination_reference.tolist()
        if self.coordination.enabled and not self.trajectory_preserving:
            p, r, jacobian = whole_body_kinematics(state, kwargs.get('base_z', 0.), kwargs.get('base_rotation'))
            error = np.r_[np.asarray(target_p)-p, Rotation.from_matrix(target_r@r.T).as_rotvec()]
            requested, _ = self.camera_task(error, feedforward, self.hold_weight(t))
            info['camera_velocity_residual'] = (jacobian@command-requested).tolist()
            info['previous_actuator_command'] = old_command.tolist()
            velocity = old_command if self.measured_control is None else self.measured_control
            stop = velocity[2:]**2/(2*self.limits.acceleration[2:])
            reserve = np.where(velocity[2:] >= 0., JOINT_UPPER-np.asarray(state)[3:], np.asarray(state)[3:]-JOINT_LOWER)-stop
            info['joint_braking_reserve_rad'] = reserve.tolist()
            info['base_yaw_stopping_angle_rad'] = float(velocity[1]**2/(2*self.limits.acceleration[1]))
            info['braking_velocity_source'] = 'measured' if self.measured_control is not None else 'previous_command'
        return command, info

    def coordination_objective(self, hessian, gradient, j, task, desired):
        if not self.coordination.enabled:
            return hessian, gradient
        # Future camera motion replaces conflicting instantaneous heading and
        # comfort goals. The current camera task and every safety row remain.
        weight = np.array([20000., 20000., 20., 20., 20., 20., 20., 20.])
        return (j.T@(task[:, None]*j)+np.diag(weight)+.15*np.eye(8),
                -j.T@(task*desired)-weight*(self.coordination_reference if self.coordination_reference is not None else np.zeros(8))-.15*self.previous)

    def close(self):
        if hasattr(self.reconfiguration, 'close'):
            self.reconfiguration.close()
        if hasattr(self.coordination, 'close'):
            self.coordination.close()

    def replan_due(self, trajectory_time):
        return False


class TrackingOnlyWBC(ExecutionWBC):
    """Same shot-aware WBC task/actuator objectives, with avoidance disabled."""
    obstacle_avoidance = False

    def __init__(self, limits, scene, **kwargs):
        if limits.obstacle_avoidance:
            raise ValueError('tracking-only controller requires obstacle_avoidance=False')
        super().__init__(limits, scene=scene, **kwargs)
        self.collision_mode = 'none'

    def solve(self, *args, **kwargs):
        command, info = super().solve(*args, **kwargs)
        info.update(dynamic_constraint_count=0, obstacle_avoidance_enabled=False,
                    collision_audit_only=True)
        return command, info


def make_wbc(limits, scene, reference, preview=False):
    controller_type = ExecutionWBC if limits.obstacle_avoidance else TrackingOnlyWBC
    controller = controller_type(limits, scene=scene, preview=preview,
                                     horizon_steps=4, horizon_dt=.3, replan_s=.4)
    controller.set_trajectory(reference)
    return controller


def physical_caps(state, limits, remaining):
    """Conservative numerical SEARCH domains, not permissions or visual tolerances."""
    _, _, j = whole_body_kinematics(state)
    reach = np.abs(j)@limits.velocity*min(remaining, 2.)
    # Global caps avoid a singular Jacobian declaring an axis forever impossible.
    arm_span = float(np.sum(np.linalg.norm(j[:3, 2:], axis=0)*(JOINT_UPPER-JOINT_LOWER)))
    linear = max(.02, min(limits.velocity[0]*remaining+arm_span, 1.5))
    angular = min(np.pi, float(np.max(reach[3:]))+limits.velocity[1]*remaining)
    return dict(radial_m=linear, lateral_m=linear,
                height_m=max(.02, min(arm_span, .65)),
                yaw_rad=angular, pitch_rad=min(angular, np.pi/2), roll_rad=min(angular, np.pi/2))


def unrelaxed_prediction(original, scene, state, previous, t, base_z, limits, horizon=3.5, step=.1,
                         commanded_q=None):
    """Predict unconstrained-to-obstacles tracking; audit it against moving geometry.

    The rollout still respects actuator/position constraints. This report exposes
    what the requested camera path would require, before safety changes it.
    """
    robot = np.asarray(state, float).copy()
    q_command = robot[3:].copy() if commanded_q is None else np.asarray(commanded_q).copy()
    qp = WholeBodyQP(limits)
    qp.previous = np.asarray(previous).copy()
    end = min(original.duration, t+horizon)
    result = dict(min_clearance_m=float('inf'), first_risk_s=None, joint_margin_rad=joint_margin(robot),
                  velocity_ratio=0., acceleration_ratio=0., tracking_residual_m=0.,
                  heading_disadvantage=0., predicted_state=robot.tolist(), failure='')
    collision = DynamicCollision(scene, t)
    for now in np.arange(t, end-1e-8, step):
        dt = min(step, end-now)
        p, r, ff = original.sample(now)
        old = qp.previous.copy()
        qp.time_s = now
        try:
            u, info = qp.solve(robot, p, r, ff, dt, base_z=base_z, commanded_q=q_command)
        except RuntimeError as error:
            result['failure'] = str(error)
            break
        collision.set_time(now)
        sweep = collision.sweep(robot, u, dt, base_z, max_samples=7)
        clearance = float(sweep['lower'].min())
        result['min_clearance_m'] = min(result['min_clearance_m'], clearance)
        if clearance < collision.margin+.08 and result['first_risk_s'] is None:
            result['first_risk_s'] = float(now)
            result['conflict_state'] = robot.tolist()
        result['velocity_ratio'] = max(result['velocity_ratio'], float(np.max(np.abs(u)/limits.velocity)))
        result['acceleration_ratio'] = max(result['acceleration_ratio'], float(np.max(np.abs(u-old)/(dt*limits.acceleration))))
        result['tracking_residual_m'] = max(result['tracking_residual_m'], info['position_error'])
        result['joint_margin_rad'] = min(result['joint_margin_rad'], joint_margin(robot))
        if len(collision.centers):
            closest = collision.centers[np.argmin(np.linalg.norm(collision.centers[:, :2]-robot[:2], axis=1))]
            direction = robot[:2]-closest[:2]
            result['heading_disadvantage'] = float(abs(np.cross([np.cos(robot[2]), np.sin(robot[2])],
                direction/max(np.linalg.norm(direction), 1e-9))))
        robot = advance_command(robot, old, u, dt, limits.command_ramp)
        q_command += dt*(.5*(old[2:]+u[2:]) if limits.command_ramp else u[2:])
    result['predicted_state'] = robot.tolist()
    issues = []
    if result['first_risk_s'] is not None:
        issues.append('predicted_dynamic_collision_or_low_clearance')
    if result['joint_margin_rad'] < .16:
        issues.append('joint_reserve_low')
    if result['velocity_ratio'] > .9:
        issues.append('velocity_limited')
    if result['acceleration_ratio'] > .95:
        issues.append('acceleration_limited')
    if result['heading_disadvantage'] > .8 and result['first_risk_s'] is not None:
        issues.append('nonholonomic_heading_unfavorable_for_retreat')
    if result['failure']:
        issues.append('whole_body_prediction_failed')
    result['issues'] = issues
    return result


class DirectionProbe:
    """Unapproved mathematical probe; never passed to the execution controller."""
    def __init__(self, original, parameter, amount, start, blend):
        self.original, self.parameter, self.amount = original, parameter, amount
        self.start, self.blend, self.duration = start, blend, original.duration

    def sample(self, t):
        from .reference import smoothstep, look_at
        def pose(now):
            p, r, _ = self.original.sample(now)
            v = self.amount*smoothstep((now-self.start)/self.blend)
            axis = np.zeros(3)
            if self.parameter == 'radial_m':
                axis = p-self.original.subject
                axis[2] = 0.
                axis /= max(np.linalg.norm(axis), 1e-9)
            elif self.parameter == 'lateral_m':
                axis = r[:, 1]
            elif self.parameter == 'height_m':
                axis[2] = 1.
            if self.parameter.endswith('_m'):
                oldp = p.copy()
                p += v*axis
                r = look_at(p, self.original.subject)@look_at(oldp, self.original.subject).T@r
            else:
                axis = np.zeros(3)
                axis[{'yaw_rad': 2, 'pitch_rad': 1, 'roll_rad': 0}[self.parameter]] = v
                r = r@Rotation.from_rotvec(axis).as_matrix()
            return p, r
        p, r = pose(t)
        h = 1e-4
        pa, ra = pose(max(0., t-h))
        pb, rb = pose(min(self.duration, t+h))
        width = min(self.duration, t+h)-max(0., t-h)
        return p, r, np.r_[(pb-pa)/width, Rotation.from_matrix(rb@ra.T).as_rotvec()/width]


def useful_directions(original, scene, state, t, base_z, limits, report, previous=None, commanded_q=None):
    """Short nonholonomic, actuator-constrained probes; no semantic permissions.

    Translational probes compensate aim only to estimate their visual effects.
    Jev must independently authorize that protection and all numerical amounts.
    """
    caps = physical_caps(state, limits, original.duration-t)
    previous = np.zeros(8) if previous is None else previous
    horizon = min(3.5, original.duration-t)
    baseline = rank_candidate(original, scene, state, previous, t, base_z, limits, horizon, commanded_q)
    directions = []
    for name in PARAMETERS:
        for sign in (-1, 1):
            cap = caps[name]
            if name == 'height_m':
                # Triangle inequality gives a model-derived upper reach bound.
                from .jvs_kinematics import JOINT_ORIGINS, CAMERA_OFFSET
                ceiling = base_z+np.linalg.norm(JOINT_ORIGINS, axis=1).sum()+np.linalg.norm(CAMERA_OFFSET)
                height = original.sample(min(original.duration, t+horizon))[0][2]
                cap = min(cap, max(0., ceiling-height) if sign > 0 else max(0., height-base_z-.05))
            probe = min(.12 if name.endswith('_m') else .18, .2*cap)
            if probe < 1e-5:
                continue
            path = DirectionProbe(original, name, sign*probe, t, min(.8, horizon/2))
            outcome = rank_candidate(path, scene, state, previous, t, base_z, limits, horizon, commanded_q)
            # Discard directions whose apparent clearance gain comes mainly
            # from failing to realize the proposed camera change. This is a
            # numerical probe effectiveness test, not a semantic tolerance.
            tracking_effective = (outcome.max_tracking_error_m <= baseline.max_tracking_error_m+.4*probe
                                  if name.endswith('_m') else
                                  outcome.max_orientation_error_rad <= baseline.max_orientation_error_rad+.4*probe)
            gain = outcome.min_clearance_m-baseline.min_clearance_m
            margin_gain = outcome.joint_margin_rad-baseline.joint_margin_rad
            if not np.isfinite(gain+margin_gain):
                continue
            distance = max(np.linalg.norm(original.sample(t)[0]-original.subject), 1e-6)
            directions.append(dict(id=name+('_positive' if sign > 0 else '_negative'), parameter=name,
                sign=sign, unit='m' if name.endswith('_m') else 'rad', physical_cap=cap, probe_amount=probe,
                estimated_clearance_gain_m=gain, estimated_joint_margin_gain_rad=margin_gain,
                estimated_tracking_residual_m=outcome.max_tracking_error_m,
                estimated_view_angle_change_rad=probe/distance if name.endswith('_m') else probe,
                estimated_area_scale_ratio=(distance/(distance+sign*probe))**2 if name == 'radial_m' else None,
                rough_joint_feasible=outcome.feasible and tracking_effective,
                estimate_kind='nonholonomic_actuator_constrained_rollout_not_collision_certificate',
                translation_probe_aim='compensated_subject_framing_for_estimation_only'))
    directions.sort(key=lambda d: (d['rough_joint_feasible'],
                    d['estimated_clearance_gain_m']+.35*d['estimated_joint_margin_gain_rad']), reverse=True)
    return [d for d in directions if d['rough_joint_feasible'] and
            (d['estimated_clearance_gain_m'] > .001 or d['estimated_joint_margin_gain_rad'] > .001)][:6], caps


def time_windows(t, duration, report, limits):
    available = duration-t
    # Timings are derived from remaining budget and predicted encounter, not shot-specific tolerances.
    encounter = report.get('first_risk_s')
    encounter = t+.5*available if encounter is None else encounter
    blend = max(.25, min(1.2, available/4.))
    early = min(t+.1, duration-2.*blend)
    late = min(max(early, encounter-blend), duration-2.*blend)
    return dict(
        immediate=dict(start_s=early, end_s=duration, blend_s=blend),
        encounter=dict(start_s=late, end_s=duration, blend_s=blend),
        local=dict(start_s=early, end_s=min(duration, max(early+2*blend, encounter+2*blend)), blend_s=blend))


def permission_violation(reference, start, end, step=.02):
    """Check absolute semantic permissions, including activation and recovery.

    Bounds follow the explicit envelope; local direction is checked on a fixed
    20 ms grid. This target-path check is distinct from measured tracking error.
    """
    if not isinstance(reference, DeformedReference):
        return None
    times = np.unique(np.r_[np.arange(start, end, step), end])
    if not all(reference.envelope.contains(reference.parameters(t), atol=1e-6,
            settled=t >= reference.activation_s+reference.envelope.blend_s) for t in times):
        return 'deformation_outside_absolute_permission'
    if 'motion_direction' in reference.envelope.protected:
        for a, b in zip(times[:-1], times[1:]):
            if b < reference.activation_s:
                continue
            nominal = reference.original.sample(b)[0]-reference.original.sample(a)[0]
            changed = reference.sample(b)[0]-reference.sample(a)[0]
            if np.linalg.norm(nominal) > 1e-5 and float(nominal@changed) < -1e-7:
                return 'protected_motion_direction_reversed'
    return None


def validate_rollout(reference, scene, state, previous, t, base_z, limits,
                     horizon=2.4, step=.08, check_terminal=False, commanded_q=None,
                     record_trace=False, checkpoint=None, resume=None, fine_until_s=None):
    """Nonholonomic WBC rollout + swept dynamic collision and physical residuals.

    Reference tracking error is an objective and reported residual, never a
    secretly assigned semantic tolerance. The fast QP rechecks every real step.
    """
    end = reference.duration if check_terminal else min(reference.duration, t+horizon)
    violation = permission_violation(reference, t, end)
    if violation:
        return Validation(False, violation, cost=1e4)
    controller = make_wbc(limits, scene, reference, preview=False)
    # Fast validation uses the same constrained execution QP, without repeated long MPC solves.
    controller.last_plan = float('inf')
    controller.previous = np.asarray(previous, float).copy()
    robot = np.asarray(state, float).copy()
    q_command = robot[3:].copy() if commanded_q is None else np.asarray(commanded_q).copy()
    original = getattr(reference, 'original', reference)
    validation = Validation(True, 'verified_prediction', joint_margin_rad=joint_margin(robot), cost=0.)
    if resume is not None:
        # Branch an identical already-computed execution prefix. This is an exact
        # computation reuse, not a reduced-dynamics approximation or safety skip.
        controller = deepcopy(resume['controller'])
        controller.set_trajectory(reference)
        robot = resume['state'].copy()
        q_command = resume['commanded_q'].copy()
        validation = deepcopy(resume['validation'])
        t = resume['time_s']
        if validation.camera_trace and abs(validation.camera_trace[-1]['t']-t) < 1e-8:
            validation.camera_trace.pop()
    actual_positions, original_positions = [], []
    # Selected-plan verification resolves the installation interval at the
    # execution period; the distant suffix keeps the cheaper preview step.
    fine_end = t if fine_until_s is None else min(end, max(t, fine_until_s))
    intervals = [(now, min(.02, fine_end-now)) for now in np.arange(t, fine_end-1e-8, .02)]
    intervals += [(now, min(step, end-now)) for now in np.arange(fine_end, end-1e-8, step)]
    for now, dt in intervals:
        controller.set_time(now)
        target, rotation, twist = reference.sample(now)
        old = controller.previous.copy()
        try:
            command, info = solve_with_certificate(controller, robot, target, rotation, twist, dt,
                base_z=base_z, commanded_q=q_command, trajectory_time=now)
            if limits.command_ramp:
                info['command_sweep_lower_m'] = certify_control_ramp(scene, robot, old, command, now, dt,
                    base_z, steps=max(2, round(dt/.01)), constant_certificate=info['command_sweep_lower_m'])
        except RuntimeError as error:
            validation.feasible, validation.reason = False, str(error)
            validation.cost += 1e4
            return validation
        if info['command_sweep_lower_m'] < controller.collision.margin:
            validation.feasible, validation.reason = False, 'predicted_tracking_collision'
            validation.min_clearance_m = info['command_sweep_lower_m']
            return validation
        validation.samples += 1
        if info.get('tpr_status') == 'degraded':
            validation.task_degraded_samples += 1
            validation.last_task_degradation_s = float(now)
        validation.min_clearance_m = min(validation.min_clearance_m, info['command_sweep_lower_m'])
        validation.joint_margin_rad = min(validation.joint_margin_rad, joint_margin(robot))
        validation.max_velocity_ratio = max(validation.max_velocity_ratio, float(np.max(np.abs(command)/limits.velocity)))
        validation.max_acceleration_ratio = max(validation.max_acceleration_ratio,
                                               float(np.max(np.abs(command-old)/(dt*limits.acceleration))))
        validation.max_tracking_error_m = max(validation.max_tracking_error_m, info['position_error'])
        validation.max_orientation_error_rad = max(validation.max_orientation_error_rad, info['orientation_error'])
        p, r, _ = whole_body_kinematics(robot, base_z)
        if record_trace:
            validation.camera_trace.append(dict(t=float(now), position=p.tolist(),
                quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
                state=robot.tolist(), joint_margin_rad=joint_margin(robot)))
        op, ori, _ = original.sample(now)
        original_error = np.linalg.norm(p-op)**2+.15*np.linalg.norm(Rotation.from_matrix(r@ori.T).as_rotvec())**2
        deficit = max(0., .16-joint_margin(robot))
        safety_reserve = max(0., .10-info['command_sweep_lower_m'])
        deformation = np.linalg.norm(target-op)**2+.15*np.linalg.norm(Rotation.from_matrix(rotation@ori.T).as_rotvec())**2
        tracking = np.linalg.norm(p-target)**2+.15*np.linalg.norm(Rotation.from_matrix(r@rotation.T).as_rotvec())**2
        validation.original_loss_integral += dt*original_error
        validation.target_tracking_loss_integral += dt*tracking
        # Execution quality is secondary to original-reference preservation.
        # Tracking an easier altered target is not itself an intent benefit.
        # These are soft optimization weights, not semantic acceptance limits.
        validation.cost += dt*(original_error+2.*tracking+.10*deformation+.6*deficit**2+2.*safety_reserve**2+
                               .0001*np.linalg.norm((command-old)/dt)**2)
        actual_positions.append(p)
        original_positions.append(op)
        robot = advance_command(robot, old, command, dt, limits.command_ramp)
        q_command += dt*(.5*(old[2:]+command[2:]) if limits.command_ramp else command[2:])
    validation.terminal_state = robot.tolist()
    if record_trace:
        p, r, _ = whole_body_kinematics(robot, base_z)
        validation.camera_trace.append(dict(t=float(end), position=p.tolist(),
            quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
            state=robot.tolist(), joint_margin_rad=joint_margin(robot)))
    validation.original_objective = validation.original_loss_integral
    if check_terminal:
        endpoint, er, _ = whole_body_kinematics(robot, base_z)
        nominal_endpoint, nominal_er, _ = original.sample(end)
        terminal_loss = 2.*(np.linalg.norm(endpoint-nominal_endpoint)**2+
            .15*np.linalg.norm(Rotation.from_matrix(er@nominal_er.T).as_rotvec())**2)
        validation.original_objective += terminal_loss
        validation.cost += terminal_loss
    validation.next_phase_checked = check_terminal or any(t < s['start_s'] < end for s in reference.shots)
    if checkpoint is not None:
        checkpoint.update(controller=controller, state=robot.copy(), commanded_q=q_command.copy(),
                          validation=deepcopy(validation), time_s=end)
    return validation


def rank_candidate(reference, scene, state, previous, t, base_z, limits, horizon=2.4, commanded_q=None):
    """Cheap physical-QP ranking only; a positive score is NOT a safety certificate."""
    controller = WholeBodyQP(limits)
    controller.previous = np.asarray(previous).copy()
    robot = np.asarray(state).copy()
    q_command = robot[3:].copy() if commanded_q is None else np.asarray(commanded_q).copy()
    original = getattr(reference, 'original', reference)
    collision = DynamicCollision(scene, t)
    result = Validation(True, 'coarse_ranking_only', joint_margin_rad=joint_margin(robot), cost=0.)
    end = min(reference.duration, t+horizon)
    for now in np.arange(t, end-1e-8, .1):
        dt = min(.1, end-now)
        p, r, ff = reference.sample(now)
        old = controller.previous.copy()
        controller.time_s = now
        try:
            u, info = controller.solve(robot, p, r, ff, dt, base_z=base_z, commanded_q=q_command)
        except RuntimeError as error:
            return Validation(False, str(error))
        collision.set_time(now+.5*dt)
        clearance = float(collision.clearance(advance(robot, u, .5*dt), base_z).min())
        actual, actual_r, _ = whole_body_kinematics(robot, base_z)
        op, ori, _ = original.sample(now)
        cost = np.linalg.norm(actual-op)**2+.15*np.linalg.norm(Rotation.from_matrix(actual_r@ori.T).as_rotvec())**2
        cost += 20.*max(0., .10-clearance)**2+.6*max(0., .16-joint_margin(robot))**2
        cost += .1*np.linalg.norm(p-op)**2
        cost += 2.*(np.linalg.norm(actual-p)**2+.15*np.linalg.norm(Rotation.from_matrix(actual_r@r.T).as_rotvec())**2)
        result.cost += cost*dt
        result.min_clearance_m = min(result.min_clearance_m, clearance)
        result.joint_margin_rad = min(result.joint_margin_rad, joint_margin(robot))
        result.max_tracking_error_m = max(result.max_tracking_error_m, info['position_error'])
        result.max_orientation_error_rad = max(result.max_orientation_error_rad, info['orientation_error'])
        result.samples += 1
        robot = advance_command(robot, old, u, dt, limits.command_ramp)
        q_command += dt*(.5*(old[2:]+u[2:]) if limits.command_ramp else u[2:])
    return result


def optimize_envelope(original, active, envelope, recovery, scene, state, previous, t, base_z, limits,
                      activation_lead=.1, commanded_q=None):
    """Small bounded candidate search; minimizes actual ORIGINAL-reference loss.

    It includes zero and interior amounts, and validates full remaining-budget
    prediction of finalists. Enlarging bounds never implies using their maximum.
    """
    begin = time.monotonic()
    activation = max(envelope.start_s, t+activation_lead)
    if activation+envelope.blend_s > envelope.end_s:
        return dict(error='permission_expired_before_smooth_activation', evaluations=[])
    envelope.validate(original.sha256, original.duration, physical_caps(state, limits, original.duration-t))
    current = np.zeros(len(PARAMETERS)) if active is original else active.parameters(activation)
    directions = []
    for i, name in enumerate(PARAMETERS):
        lo, hi = (envelope.settled_bounds or envelope.bounds).get(name, [0., 0.])
        if abs(lo)+abs(hi) > 1e-8:
            axis = np.zeros(len(PARAMETERS))
            axis[i] = hi if hi > abs(lo) else lo
            directions.append(axis)
    candidates = [np.zeros(len(PARAMETERS)), current]
    if recovery != 'revoke':
        fractions = (.25, .5, .8, 1.) if recovery != 'tighten' else (.125, .25, .5)
        for axis in directions:
            candidates.extend(axis*f for f in fractions)
        if len(directions) > 1:
            combined = np.sum(directions, axis=0)
            candidates.extend(combined*f for f in (.25, .5, .8))
    unique = {tuple(np.round(p, 8)): p for p in candidates if envelope.contains(p, settled=True)}
    ranked, evaluations = [], []
    for values in unique.values():
        returns = [None]
        if envelope.terminal == 'retain' and np.linalg.norm(values) > 1e-8:
            returns.append(envelope.end_s-envelope.blend_s)
        for return_s in returns:
            path = DeformedReference(original, envelope, values, None if active is original else active, activation, return_s)
            outcome = rank_candidate(path, scene, state, previous, t, base_z, limits,
                                     horizon=min(2.4, original.duration-t), commanded_q=commanded_q)
            evaluations.append(dict(values=values.tolist(), return_s=return_s, coarse_ranking=outcome.to_dict()))
            if outcome.feasible:
                ranked.append((outcome.cost, path, outcome))
    # Always compare against zero permission usage in the SAME full WBC model.
    # Coarse collision penalties alone must not force unnecessary deformation.
    ordered = sorted(ranked, key=lambda entry: entry[0])
    finalists = [entry for entry in ordered if np.linalg.norm(entry[1].values) < 1e-8][:1]
    # Retain/recover variants can have identical coarse scores before recovery
    # begins. Do not spend every full-budget slot on the same small amplitude:
    # that can incorrectly reject an envelope whose larger interior or boundary
    # values make the whole-body plan feasible.
    amplitudes = {}
    for entry in ordered:
        if np.linalg.norm(entry[1].values) >= 1e-8:
            amplitudes.setdefault(tuple(np.round(entry[1].values, 8)), entry)
    distinct = list(amplitudes.values())
    finalists += distinct[:2]
    if distinct:
        boundary = max(distinct, key=lambda entry: np.linalg.norm(entry[1].values))
        if not any(entry[1] is boundary[1] for entry in finalists):
            finalists.append(boundary)
        # Check its alternative in-budget recovery too: coarse ranking may
        # end before that recovery begins and cannot distinguish the variants.
        alternative = next((entry for entry in ordered
            if np.allclose(entry[1].values, boundary[1].values, atol=1e-9, rtol=0.)
            and entry[1].return_s != boundary[1].return_s), None)
        if alternative is not None:
            finalists.append(alternative)
    feasible = []
    for _, path, outcome in finalists:
        full = validate_rollout(path, scene, state, previous, t, base_z, limits,
                                 check_terminal=True, commanded_q=commanded_q)
        evaluations.append(dict(values=path.values.tolist(), return_s=path.return_s,
                                full_budget=True, validation=full.to_dict()))
        if full.feasible:
            # Physical feasibility is already a hard gate. Among verified
            # candidates, first minimize predicted loss against the immutable
            # original, including terminal recovery. A lower tracking residual
            # to an altered target cannot compensate for worse original loss.
            feasible.append((full.original_objective, full.cost, path, full))
    if feasible:
        _, _, path, full = min(feasible, key=lambda entry: entry[:2])
        return dict(path=path, validation=full, evaluations=evaluations, wall_s=time.monotonic()-begin,
                    snapshot_time_s=t, snapshot_state=np.asarray(state).tolist())
    failures = []
    for candidate in evaluations:
        outcome = candidate.get('validation', candidate.get('coarse_ranking', {}))
        if not outcome.get('feasible', False):
            failures.append(dict(values=candidate['values'], return_s=candidate.get('return_s'),
                reason=outcome.get('reason'), min_clearance_m=outcome.get('min_clearance_m'),
                joint_margin_rad=outcome.get('joint_margin_rad'),
                tracking_residual_m=outcome.get('max_tracking_error_m')))
    return dict(error='no_candidate_passed_whole_body_validation', evaluations=evaluations,
                rejection_summary=failures[-3:], wall_s=time.monotonic()-begin)
