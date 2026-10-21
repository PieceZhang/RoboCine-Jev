"""Conditional execution forecasts for the jvs_mock physical control chain.

The kinematic witness is a configuration preference, not the executed trajectory.
This module runs the execution WBC, delayed command transport and the same motor
slew as the physical adapter. Residual envelopes are empirical engineering bounds;
they are deliberately not advertised as probabilities or safety certificates.
"""
from collections import deque
from dataclasses import asdict, dataclass, field, replace
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .whole_body_qp import whole_body_kinematics
from .kc_svsdf import advance


def failure_evidence(reason, at=None, *, source='prediction', measured=False):
    """Separate observed symptoms from a claimed physical root cause."""
    text = str(reason).lower()
    if any(s in text for s in ('transport_unverified', 'connection', 'http', 'network')):
        category, action = 'network_unverified', 'retry_same_request_with_bounded_backoff'
    elif any(s in text for s in ('expired', 'stale', 'latency', 'command_age', 'takeover')):
        category, action = 'time_alignment', 'refresh_state_and_predict_takeover'
    elif any(s in text for s in ('envelope_exceeded', 'model_residual')):
        category, action = 'execution_model_mismatch', 'widen_uncertainty_and_refresh_forecast'
    elif any(s in text for s in ('collision', 'clearance', 'sweep', 'contact','tube')):
        category, action = 'safety_coverage', 'change_occupied_space_or_timing'
    elif any(s in text for s in ('velocity_limit', 'acceleration_limit', 'braking', 'reserve')):
        category, action = 'execution_capacity', 'reduce_demand_or_prepare_configuration'
    elif any(s in text for s in ('qp', 'solver', 'numerical', 'wall_budget')):
        category, action = 'numerical_unverified', 'inspect_constraints_and_solver_status'
    elif 'hard constraint' in text or 'hard safety constraints infeasible' in text:
        category, action = 'constraint_conflict', 'inspect_active_actuator_collision_and_task_rows'
    elif any(s in text for s in ('intent', 'terminal', 'camera_limits', 'tracking','executed_direction','natural_curve','subject_frame')):
        category, action = 'intent_execution_mismatch', 'compare_target_deformation_and_execution_error'
    elif any(s in text for s in ('single_candidate', 'interface','configuration_contract')):
        category, action = 'interface_configuration', 'repair_adapter_contract'
    elif 'configuration' in text:
        category, action = 'execution_capacity', 'inspect_configuration_reserve_and_measured_tracking'
    else:
        category, action = 'unknown', 'retain_counterexample_for_diagnosis'
    return dict(category=category, first_anomaly_s=at, symptom=str(reason),
                source=source, root_cause_confirmed=False, measured=bool(measured),
                recommended_action=action)


@dataclass
class ExecutionModel:
    robot: str = 'jvs_mock'
    controller: str = 'TPR-WBC/physical_slew/v1'
    command_mode: str = 'adaptive'
    delay_s: float = .04
    delay_jitter_s: float = .02
    preview_delay_s: float = .4
    response_s: list = field(default_factory=lambda: [.0025, .005, 0., 0., 0., 0., 0., 0.])
    residual_velocity: list = field(default_factory=lambda: [.003, .008, .006, .006, .006, .006, .006, .006])
    bias_velocity: list = field(default_factory=lambda: [0.] * 8)
    calibrated_velocity_abs_max: list | None = None
    position_floor_m: float = .003
    trusted_horizon_s: float = 1.5
    snapshot_time_s: float = 0.
    samples: int = 0
    breaches: int = 0
    revision: int = 0
    delay_aware: bool = True
    body_aware: bool = False
    evidence: str = 'engineering_prior_not_calibrated'

    def __post_init__(self):
        if (self.robot != 'jvs_mock' or self.controller != 'TPR-WBC/physical_slew/v1' or
                self.command_mode not in ('velocity', 'acceleration', 'adaptive')):
            raise ValueError('execution model robot/command contract mismatch')
        for name in ('response_s', 'residual_velocity', 'bias_velocity'):
            value = np.asarray(getattr(self, name), float)
            if value.shape != (8,) or not np.isfinite(value).all():
                raise ValueError('execution model requires eight finite ' + name)
            if name != 'bias_velocity' and np.any(value < 0):
                raise ValueError('negative execution response or residual')
        if self.calibrated_velocity_abs_max is not None:
            support=np.asarray(self.calibrated_velocity_abs_max,float)
            if support.shape!=(8,) or not np.isfinite(support).all() or np.any(support<0):
                raise ValueError('invalid calibrated operating range')
        if not np.isfinite([self.delay_s, self.delay_jitter_s, self.preview_delay_s,self.position_floor_m,
                            self.trusted_horizon_s, self.snapshot_time_s]).all():
            raise ValueError('nonfinite execution model')
        if min(self.delay_s, self.delay_jitter_s,self.preview_delay_s,self.position_floor_m,self.snapshot_time_s)<0 or self.trusted_horizon_s<=0:
            raise ValueError('invalid execution time or uncertainty')

    def to_dict(self):
        return asdict(self)

    def public(self, state, velocity, limits, base_z=0., base_rotation=None):
        """Task-space consequences; joint identities and limits stay private."""
        _, _, j = whole_body_kinematics(state, base_z, base_rotation)
        v = np.asarray(velocity)
        braking = float(np.max(abs(v) / limits.acceleration))
        available = np.maximum(0., limits.velocity - abs(v))
        return dict(schema='jev_execution_capability/v1', robot=self.robot, controller=self.controller,
            state_stamp_s=self.snapshot_time_s, valid_until_s=self.snapshot_time_s + .1,
            expected_command_effect_s=self.snapshot_time_s + self.delay_s,
            command_delay_s=self.delay_s, delay_jitter_s=self.delay_jitter_s,
            optional_configuration_preview_delay_s=self.preview_delay_s,
            optional_preview_delay_evidence='engineering timing prior; actual freshness and pose gates are replayed',
            recovery_time_s=braking + self.delay_s,
            camera_twist=(j @ v).tolist(),
            twist_frame='world',twist_order=['vx','vy','vz','wx','wy','wz'],
            directional_velocity_support=(abs(j) @ available).tolist(),
            directional_support_scope='outer_support_only; directions share actuators and cannot be combined independently',
            velocity_reserve_fraction=float(np.min(available / limits.velocity)),
            validity_conditions=['same_robot_and_controller', 'fresh_measured_state_and_map',
                'command_age_within_100ms_lease', 'no_unmodelled_contact_or_wheel_slip'],
            prediction=dict(trusted_horizon_s=self.trusted_horizon_s,
                position_floor_m=self.position_floor_m,
                residual_twist_envelope=(abs(j) @ np.asarray(self.residual_velocity)).tolist(),
                bias_twist=(j @ np.asarray(self.bias_velocity)).tolist(),
                evidence=self.evidence, samples=self.samples, envelope_breaches=self.breaches,
                operating_point_within_calibrated_velocity_range=(None if self.calibrated_velocity_abs_max is None else
                    bool(np.all(abs(v)<=np.asarray(self.calibrated_velocity_abs_max)+np.asarray(self.residual_velocity)))),
                revision=self.revision, probability=None,
                scope='conditional empirical envelope; distant forecasts extrapolated; fresh physical checks mandatory'))


def motor_step(command, velocity, state, limits, dt):
    """Same bounded drive target as MotorTargets, in generalized coordinates."""
    from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER
    desired = np.asarray(command, float).copy()
    a = limits.acceleration[2:]
    low = np.maximum(state[3:] - JOINT_LOWER - limits.position_margin - .005, 0.)
    high = np.maximum(JOINT_UPPER - limits.position_margin - .005 - state[3:], 0.)
    desired[2:] = np.clip(desired[2:], -(np.sqrt((a*dt)**2 + 2*a*low)-a*dt),
                         np.sqrt((a*dt)**2 + 2*a*high)-a*dt)
    delta = np.clip(desired - velocity, -limits.acceleration*dt, limits.acceleration*dt)
    delta[:2] /= max(1., float(np.max(abs(limits.wheels @ delta))) / (limits.wheel_acceleration*dt))
    return velocity + delta


def response_step(state, velocity, command, limits, model, dt=.01):
    target = motor_step(command, velocity, state, limits, dt)
    tau = np.asarray(model.response_s)
    gain = np.ones(8)
    np.divide(-dt, tau, out=gain, where=tau > 0)
    gain = np.where(tau > 0, -np.expm1(gain), 1.)
    final = velocity + gain * (target - velocity)
    final += np.asarray(model.bias_velocity) * min(1., dt/.02)
    # Bias is a measured residual, not an additional control authority.
    # PhysX advances configuration with the new step velocity. A trapezoidal
    # CPU integrator creates a systematic half-step position lag even when
    # predicted encoder velocities are correct.
    return advance(state, final, dt), final, target


def relative_self_reach(self_collision):
    """Per-pair joint displacement support; common rigid transforms cancel."""
    geometry=self_collision.geometry
    masks=geometry.shape_joint_mask
    relative=np.logical_xor(masks[self_collision.left],masks[self_collision.right])
    return relative*geometry.reach[1:]


class ExecutionObserver:
    """Causal residual/latency updates. Bounds widen immediately, never shrink."""
    def __init__(self, model=None):
        self.model = model or ExecutionModel()
        self.previous = None
        self.delays = deque(maxlen=64)
        self.last_command_sequence=None
        self.last_evidence = None

    def observe(self, stamp, state, velocity, command, limits, *, command_delay_s=None,command_sequence=None,transition=None):
        m = self.model
        self.last_evidence = None
        if self.previous is not None and stamp<=self.previous[0]:
            return None
        innovation=None
        bias_observation=None
        if transition is not None:
            # Use the actually applied motor target, not a proposed command
            # that may have been rejected or superseded by the physical guard.
            at=float(transition['stamp_s']);dt=float(transition['dt_s'])
            if abs(stamp-at)<=1e-6 and 0<dt<=.02:
                old_v=np.asarray(transition['previous_velocity'],float)
                target=np.asarray(transition['drive_target'],float)
                actual=np.asarray(transition['measured_velocity'],float)
                if all(v.shape==(8,) and np.isfinite(v).all() for v in (old_v,target,actual)):
                    tau=np.asarray(m.response_s)
                    gain=np.ones(8);positive=tau>0
                    gain[positive]=-np.expm1(-dt/tau[positive])
                    predicted=old_v+gain*(target-old_v)+np.asarray(m.bias_velocity)*min(1.,dt/.02)
                    innovation=actual-predicted
                    bias_observation=np.asarray(m.bias_velocity)+innovation/min(1.,dt/.02)
        elif self.previous is not None:
            old_t, old_state, old_v, old_command = self.previous
            dt = stamp - old_t
            if 0 < dt <= .1:
                predicted, v = old_state.copy(), old_v.copy()
                steps = max(1, int(np.ceil(dt/.01)))
                for _ in range(steps):
                    predicted, v, _ = response_step(predicted, v, old_command, limits, m, dt/steps)
                innovation = np.asarray(velocity) - v
                bias_observation=np.asarray(m.bias_velocity)+innovation/(dt/.02)
        if innovation is not None:
            bound=np.asarray(m.residual_velocity)
            breached=bool(np.any(abs(innovation)>bound))
            m.samples+=1
            if breached:
                m.breaches+=1;m.revision+=1
                self.last_evidence=failure_evidence('model_residual_envelope_exceeded',stamp,
                    source='measured_one_step_velocity',measured=True)
            m.residual_velocity=np.maximum(bound,1.2*abs(innovation)).tolist()
            m.bias_velocity=(.95*np.asarray(m.bias_velocity)+.05*bias_observation).tolist()
            m.evidence='online_one_step_residuals_not_joint_trajectory_calibration'
        new_packet=command_sequence is None or command_sequence!=self.last_command_sequence
        if new_packet and command_delay_s is not None and np.isfinite(command_delay_s) and 0 <= command_delay_s <= .1:
            self.delays.append(float(command_delay_s))
            m.delay_s = max(.02, float(np.quantile(self.delays, .5)))
            # A few recent fast packets cannot revoke calibrated delay
            # headroom and abruptly increase the feedback gain.
            m.delay_jitter_s = max(m.delay_jitter_s, .01, float(max(self.delays)-m.delay_s))
            self.last_command_sequence=command_sequence
        m.snapshot_time_s = float(stamp)
        self.previous = (stamp, np.asarray(state).copy(), np.asarray(velocity).copy(), np.asarray(command).copy())
        return self.last_evidence


def delayed_reconfiguration(delay):
    """Replay the real asynchronous preview's freshness/pose checks in time.

    A future is evaluated lazily at its modeled completion time. Forecasting
    cannot use a configuration solution before the physical worker could return
    it, and a short forecast need not compute an unavailable future result.
    """
    from .tpr_wbc import AsyncReconfigurationPreview
    class Future:
        def __init__(self,owner,function,args):
            self.owner,self.function,self.args=owner,function,args
            self.ready_at=owner.virtual_time+delay
        def done(self):return self.owner.virtual_time>=self.ready_at-1e-9
        def result(self):return self.function(*self.args)
    class Pool:
        def __init__(self,owner):self.owner=owner
        def submit(self,function,*args):return Future(self.owner,function,args)
        def shutdown(self,**kwargs):pass
    class Preview(AsyncReconfigurationPreview):
        def __init__(self):
            super().__init__();self.virtual_time=0.;self.pool=Pool(self)
        def update(self,c,state,velocity,t,*args):
            self.virtual_time=t
            return super().update(c,state,velocity,t,*args)
    return Preview()


def forecast(path, witness, scene, state, velocity, previous, t, base_z, limits, model,
             *, base_rotation=None, end_s=None, wall_budget_s=30., safety_horizon_s=None):
    """Full WBC/transport/drive prediction; all times are physical times.

    Unknown distant safety is explicit. A failing swept bound in the supported
    prefix is a rejection, never a score penalty. This is not a PhysX substitute.
    """
    from .planning import make_wbc, joint_margin
    from .dynamic_collision import DynamicCollision
    from .self_collision import SelfCollision
    from .safety import solve_with_certificate
    model = ExecutionModel(**model) if isinstance(model, dict) else model
    started = time.monotonic()
    scene=dict(scene,format=scene.get('format','jvs_lidar_surfels/v1'))
    end = path.duration + .4 if end_s is None else float(end_s)
    robot, speed = np.asarray(state, float).copy(), np.asarray(velocity, float).copy()
    tilt = None if base_rotation is None else Rotation.from_euler('z', -robot[2]).as_matrix() @ base_rotation
    controller = make_wbc(replace(limits, obstacle_avoidance=True), scene, path, preview=False)
    controller.reconfiguration=delayed_reconfiguration(model.preview_delay_s)
    controller.reconfiguration.reset(path)
    controller.kinematic_reference = witness
    controller.body_aware = model.body_aware
    # Physical limits have no cinematic deadline; the commanded trajectory
    # does. Mixing these produces an artificial empty terminal intersection.
    controller.measured_limits = replace(limits, deadline_s=None)
    controller.feedback_latency_s = model.delay_s+model.delay_jitter_s if model.delay_aware else 0.
    controller.expected_command_delay_s = model.delay_s if model.delay_aware else 0.
    controller.execution_safety_delay_s=min(.1,model.delay_s+model.delay_jitter_s) if model.delay_aware else 0.
    controller.previous = np.asarray(previous).copy()
    collision, self_collision = DynamicCollision(scene, t), SelfCollision()
    self_reach=relative_self_reach(self_collision)
    horizon = model.trusted_horizon_s if safety_horizon_s is None else safety_horizon_s
    observation_stamp=float(scene.get('observation_stamp',t))
    safety_until = min(end, max(t, observation_stamp + horizon))
    input_error=None
    if not -1e-8<=t-observation_stamp<=.6:
        input_error='stale_or_future_observation_for_execution_prediction'
    elif not -1e-8<=t-model.snapshot_time_s<=.1:
        input_error='stale_or_future_execution_model_snapshot'
    held = np.asarray(previous).copy()
    held_accelerated = False
    pending = deque()
    trace, failures = [], []
    min_body = min_self = float('inf')
    first_extrapolated_risk=None
    safety_forecast_until=t
    max_error = max_rotation = max_ratio = deformation_loss = tracking_loss = 0.
    outside_calibration=False
    calibration_domain_exit=None
    tube = model.position_floor_m
    joint_tube = np.full(6, .0001)
    uncertainty = np.asarray(model.residual_velocity)
    complete = True
    control_dt, physics_dt = .02, .01
    original = getattr(path, 'original', path)
    try:
        for index in range(int(np.ceil((end-t)/control_dt)) + 1):
            now = min(end, t + index*control_dt)
            rotation = None if tilt is None else Rotation.from_euler('z', robot[2]).as_matrix() @ tilt
            p, r, jac = whole_body_kinematics(robot, base_z, rotation)
            target, target_r, ff = path.sample(now)
            tracking = float(np.linalg.norm(p-target))
            angular = float(Rotation.from_matrix(r @ target_r.T).magnitude())
            max_error, max_rotation = max(max_error, tracking), max(max_rotation, angular)
            if model.calibrated_velocity_abs_max is not None:
                outside_calibration |= bool(np.any(abs(speed)>
                    np.asarray(model.calibrated_velocity_abs_max)+np.asarray(model.residual_velocity)))
                if outside_calibration and calibration_domain_exit is None:
                    calibration_domain_exit=float(now)
            jitter=model.delay_jitter_s*(abs(speed[0])+collision.geometry.reach@abs(speed[1:]))
            trace.append(dict(t=float(now), position=p.tolist(), quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
                state=robot.tolist(), control=speed.tolist(), tube_radius_m=float(tube+jitter),
                safety_supported=bool(now<=safety_until), joint_margin_rad=joint_margin(robot)))
            if input_error:
                failures.append(failure_evidence(input_error,now));complete=False
                break
            if now >= end-1e-9:
                break
            if time.monotonic()-started > wall_budget_s:
                failures.append(failure_evidence('forecast_wall_budget_exhausted', now))
                complete = False
                break
            dt = min(control_dt, end-now)
            controller.set_time(now)
            controller.previous = speed.copy()
            controller.measured_control = speed.copy()
            try:
                command, info = solve_with_certificate(controller, robot, target, target_r, ff, control_dt,
                    base_z=base_z, base_rotation=rotation, commanded_q=robot[3:], trajectory_time=now,
                    brake=now>=path.duration)
            except RuntimeError as error:
                # Exactly the measured adapter's bounded terminal brake. It
                # cannot turn a late camera into successful terminal framing.
                if path.duration-.4 <= now <= path.duration+.4+1e-8:
                    saved_limits=controller.limits
                    try:
                        controller.limits=replace(saved_limits,deadline_s=None)
                        command,info=solve_with_certificate(controller,robot,target,target_r,np.zeros(6),control_dt,
                            base_z=base_z,base_rotation=rotation,commanded_q=robot[3:],trajectory_time=now,brake=True)
                    except RuntimeError as brake_error:
                        failures.append(failure_evidence(str(brake_error),now))
                        complete=False
                        break
                    finally:
                        controller.limits=saved_limits
                else:
                    failures.append(failure_evidence(str(error), now))
                    complete = False
                    break
            accelerated = model.command_mode in ('acceleration', 'adaptive')
            payload = (command-speed)/control_dt if accelerated else command
            # The physical supervisor polls at 50 Hz and latches ONE velocity
            # endpoint for its two 100 Hz drive steps. Re-anchoring acceleration
            # at every physics substep would apply small accelerations twice.
            delivery=now+np.ceil(model.delay_s/control_dt-1e-9)*control_dt
            pending.append((delivery, payload.copy()))
            while pending and pending[0][0] <= now+1e-9:
                _, held = pending.popleft()
                held_accelerated = accelerated
            requested = held
            if held_accelerated:
                acceleration = np.clip(held, -limits.acceleration, limits.acceleration).copy()
                acceleration[:2] /= max(1., float(np.max(abs(limits.wheels @ acceleration)))/limits.wheel_acceleration)
                requested = np.clip(speed + control_dt*acceleration, -limits.velocity, limits.velocity)
                requested[:2] /= max(1., float(np.max(abs(limits.wheels @ requested)))/limits.wheel_speed)
            for substep in range(int(round(dt/physics_dt))):
                physical_t = now + substep*physics_dt
                before, old_speed = robot.copy(), speed.copy()
                robot, speed, motor = response_step(robot, speed, requested, limits, model, physics_dt)
                integrated_velocity = speed
                # Conservative geometry displacement bound, not camera-only.
                growth = physics_dt*(uncertainty[0] + collision.geometry.reach @ uncertainty[1:])
                jitter = model.delay_jitter_s*(abs(speed[0]) + collision.geometry.reach @ abs(speed[1:]))
                tube += float(growth)
                joint_tube += physics_dt*uncertainty[2:]
                envelope = tube + float(jitter)
                col = collision.at_time(physical_t-t)
                sweep = col.sweep(before, integrated_velocity, physics_dt, base_z, max_samples=3, base_rotation=rotation)
                lower = float(np.min(sweep['lower'])) - envelope
                # Common platform/arm-prefix motion cancels in self distance.
                # A world-position envelope must not be subtracted twice
                # from rigidly related links (e.g. base and first arm link).
                joint_error = joint_tube + model.delay_jitter_s*abs(speed[2:])
                d0,d1=self_collision.distances_sequence([before,robot])
                # A global relative-chain displacement bound remains valid
                # through nearest-capsule feature changes. Local distance
                # Jacobians alone cannot certify an uncertainty neighbourhood.
                uncertainty_bound=self_reach@joint_error
                between = physics_dt*self_reach@np.maximum(abs(old_speed[2:]),abs(speed[2:]))
                self_lower = float(np.min(np.minimum(d0,d1)-uncertainty_bound-between))
                min_body, min_self = min(min_body, lower), min(min_self, self_lower)
                safety_forecast_until=physical_t+physics_dt
                risky=lower < collision.margin or self_lower < self_collision.margin
                if risky and physical_t>safety_until and first_extrapolated_risk is None:
                    first_extrapolated_risk=physical_t
                if risky and physical_t<=safety_until:
                    failures.append(dict(failure_evidence('execution_tube_clearance_unverified', physical_t),
                        body_lower_m=lower,self_lower_m=self_lower,body_required_m=collision.margin,
                        self_required_m=self_collision.margin))
                    complete = False
                    break
                max_ratio = max(max_ratio, float(np.max(abs(motor)/limits.velocity)))
            if not complete:
                break
            tracking_loss += dt*tracking**2
            deformation_loss += dt*float(np.linalg.norm(target-original.sample(now)[0]))**2
    finally:
        controller.close()
    terminal = trace[-1] if trace else None
    return dict(schema='jev_execution_prediction/v1', model=model.to_dict(), snapshot_time_s=float(t),
        end_s=end, predicted_until_s=None if terminal is None else terminal['t'], complete=complete,
        feasible=complete and not failures, failures=failures, trace=trace,
        max_tracking_error_m=max_error, max_orientation_error_rad=max_rotation,
        max_velocity_ratio=max_ratio, min_body_clearance_m=None if not np.isfinite(min_body) else min_body,
        min_self_clearance_m=None if not np.isfinite(min_self) else min_self,
        target_deformation_integral=deformation_loss, execution_tracking_integral=tracking_loss,
        safety_forecast_until_s=float(safety_forecast_until),first_extrapolated_risk_s=first_extrapolated_risk,
        safety_supported_until_s=float(min(safety_until,safety_forecast_until)), safety_unknown_after_s=float(safety_until) if safety_until<end else None,
        safety_scope='continuous_swept_engineering_envelope_on_observed_geometry; not a probabilistic guarantee',
        outside_calibrated_velocity_range=outside_calibration if model.calibrated_velocity_abs_max is not None else None,
        calibration_domain_exit_s=calibration_domain_exit,
        calibration_scope='one-step residual data; engineering safety sweeps beyond this operating range are uncalibrated',
        conditioning=dict(configuration_preview='empty_initial_cache; delayed results use actual freshness and pose gates',
            preview_delay_s=model.preview_delay_s,obstacle_motion='current measured extrapolation; unobserved changes unknown'),
        terminal=terminal, wall_s=time.monotonic()-started)


def public_prediction(report):
    result={key:value for key,value in report.items() if key not in ('trace','model','terminal')}
    if 'intent' in result:
        intent=result['intent']
        result['intent']={key:intent[key] for key in ('intent_complete','camera_limits_pass',
            'max_speed_m_s','max_acceleration_m_s2','max_angular_speed_rad_s','in_frame_min',
            'longest_occlusion_s') if key in intent}
        result['intent']['terminal_frame']={key:value for key,value in intent.get('terminal_frame',{}).items()
            if key not in ('target','predicted_or_actual')}
        result['intent']['motion_trend']=intent.get('motion_trend')
    trace=report.get('trace',[])
    indices=sorted(set([0,len(trace)//2,len(trace)-1])) if trace else []
    result['predicted_camera']=[{key:trace[i][key] for key in
        ('t','position','quaternion_xyzw','tube_radius_m','safety_supported')} for i in indices]
    result['confidence']=dict(evidence=report['model']['evidence'],probability=None,
        model_revision=report['model']['revision'],robot=report['model']['robot'],controller=report['model']['controller'])
    return result
