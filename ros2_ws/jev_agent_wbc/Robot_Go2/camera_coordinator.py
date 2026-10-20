"""Camera-primary, receding-horizon base command support for native Piper WBC.

Only velocity/EE task inputs change. Authored WORLD camera samples and all
eighteen learned joint actions remain untouched. Prediction is a support guard,
not a dynamics/contact/collision certificate.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def operating_posture_weight(stamp_s,start_s,end_s,weight=.2,ramp_s=.5,stop_taper_s=.5):
    """Absolute-time soft preference, removed before the terminal stand hold."""
    if not np.isfinite([stamp_s,start_s,end_s,weight,ramp_s,stop_taper_s]).all() or end_s<=start_s or not 0<=weight<=1. or min(ramp_s,stop_taper_s)<=0:
        raise ValueError('finite absolute task times and bounded positive preference parameters required')
    smooth=lambda u:np.clip(u,0.,1.)**2*(3.-2.*np.clip(u,0.,1.))
    return float(weight*smooth((stamp_s-start_s)/ramp_s)*(1.-smooth((stamp_s-end_s)/stop_taper_s)))


def load_backend(path=None):
    spec = importlib.util.spec_from_file_location('go2_coordinator_backend', path or ROOT/'wbc_backend.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class CameraCoordinator:
    def __init__(self, T_ee_camera_optical, *, backend=None, limits=(.2, .2, .2), horizon_s=.8,response_profile='legacy',
                 prediction_profile='legacy'):
        self.backend = backend or load_backend()
        self.extrinsic = self.backend.rigid(T_ee_camera_optical)
        self.limits = np.asarray(limits, float); self.horizon_s = float(horizon_s)
        if (self.limits.shape != (3,) or not np.isfinite(self.limits).all() or np.any(self.limits <= 0)
                or not np.isfinite(self.horizon_s) or self.horizon_s <= 0):
            raise ValueError('finite positive command limits and horizon required')
        if prediction_profile not in ('legacy','persistent_bias_12_v1','bounded_ramp_history_v2'):
            raise ValueError('unknown prediction profile')
        self.prediction_profile=prediction_profile
        if prediction_profile!='legacy':self.horizon_s=1.2
        self.bounded_prediction_class=None
        if prediction_profile=='bounded_ramp_history_v2':
            if not np.array_equal(self.limits,[.2,.2,.2]):raise ValueError('original native limits required by bounded input forecast')
            spec=importlib.util.spec_from_file_location('go2_bounded_body_prediction',ROOT/'bounded_body_prediction.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.bounded_prediction_class=module.BoundedBodyPrediction
        self.domain=self.backend.PolicyDomain()
        check = self.domain.check([.5, 0., .5, 1., 0., 0., 0.])
        if check['status'] != 'inside_outer_domain_unverified':
            raise ValueError('pinned upstream domain unavailable')
        self.xyz = np.asarray(check['xyz_outer_bounds'])
        self.euler = np.asarray(check['euler_outer_bounds_rad'])
        # Operating reserve, without changing original joint/configuration limits.
        self.reserve = np.array([.035, .035, .015])
        if response_profile not in ('legacy','measured_backward_v1'):raise ValueError('unknown measured response profile')
        self.response_profile=response_profile;self.input_history=[]

    def prediction_intervals(self,horizon):
        if not np.isfinite(horizon) or not 0<=horizon<=self.horizon_s+1e-9:
            raise ValueError('finite bounded prediction horizon required')
        if self.prediction_profile=='legacy':return np.linspace(0.,horizon,5)
        # Retain the original .8s five nodes and append 1.0/1.2s. A shorter
        # original Task horizon retains its old five nodes; never extrapolate.
        return np.unique(np.r_[np.linspace(0.,min(horizon,.8),5),
                               np.linspace(min(horizon,.8),horizon,3)])

    def future_plan(self,velocity,preferred,future_policy_velocity,posture_weight):
        if (not posture_weight and (self.prediction_profile=='legacy' or future_policy_velocity is None)):
            return future_policy_velocity
        return np.clip((preferred if future_policy_velocity is None else future_policy_velocity)+velocity-preferred,
                       -self.limits,self.limits)

    def observe_applied_input(self,stamp_s,command):
        command=np.asarray(command,float)
        if command.shape!=(3,) or not np.isfinite(command).all() or np.any(np.abs(command)>self.limits+1e-9):
            raise ValueError('finite actually applied native input required')
        if self.input_history and stamp_s<=self.input_history[-1][0]:raise ValueError('monotonic actual input stamps required')
        self.input_history.append((float(stamp_s),command.copy()))
        self.input_history=self.input_history[-32:]

    def history_displacement(self,stamp,measured,initial,future,elapsed):
        gain,tau,delay=.55,.35,.15
        prefix=min(delay,elapsed);position=0.;velocity=float(measured);start=0.
        old=[(t+delay-stamp,float(u[0])) for t,u in self.input_history if t<=stamp]
        input_value=next((u for t,u in reversed(old) if t<=0.),0.)
        events=[(t,u) for t,u in old if 0.<t<prefix]
        for end,new_input in [*events,(prefix,input_value)]:
            interval=end-start;one_minus=-np.expm1(-interval/tau)
            position+=velocity*tau*one_minus+gain*input_value*(interval-tau*one_minus)
            velocity=velocity*(1-one_minus)+gain*input_value*one_minus
            input_value=new_input;start=end
        if elapsed>delay:
            position+=self.delayed_ramp_displacement(velocity,initial,future,elapsed-delay,gain=gain,tau=tau,delay=0.)
        return position

    @staticmethod
    def delayed_ramp_displacement(measured,initial_input,future_input,elapsed,gain=.55,tau=.35,delay=.15,slew=.5):
        """Operating approximation from measured backward gain and response lag.

        First-order response to a delayed bounded input ramp, integrated in
        closed form. This is not a certified learned-policy dynamics model.
        """
        transient=min(delay,elapsed);duration=max(0.,elapsed-delay)
        displacement=measured*transient
        ramp=min(duration,abs(future_input-initial_input)/slew)
        rate=np.sign(future_input-initial_input)*slew
        decay=np.exp(-ramp/tau);one_minus=-np.expm1(-ramp/tau)
        displacement+=measured*tau*one_minus+gain*initial_input*(ramp-tau*one_minus)+gain*rate*(ramp*ramp/2-tau*ramp+tau*tau*one_minus)
        velocity=measured*decay+gain*initial_input*one_minus+gain*rate*(ramp-tau*one_minus)
        remaining=duration-ramp;one_minus=-np.expm1(-remaining/tau)
        displacement+=velocity*tau*one_minus+gain*future_input*(remaining-tau*one_minus)
        return displacement

    def predicted_link(self, link, body, velocity, elapsed, measured_velocity,future_policy_velocity=None,stamp_s=None):
        # Account for measured inertia during the first .15s, then requested
        # velocity. Roll/pitch and base height remain measured physical state.
        transient = min(.15, elapsed)
        displacement_body = measured_velocity[:2]*transient+velocity[:2]*(elapsed-transient)
        if (self.response_profile=='measured_backward_v1' and future_policy_velocity is not None
                and (velocity[0]<0 or future_policy_velocity[0]<0 or measured_velocity[0]<-.01)):
            displacement_body[0]=(self.history_displacement(stamp_s,measured_velocity[0],velocity[0],future_policy_velocity[0],elapsed)
                if stamp_s is not None else self.delayed_ramp_displacement(measured_velocity[0],velocity[0],future_policy_velocity[0],elapsed))
        angle = measured_velocity[2]*transient+velocity[2]*(elapsed-transient)
        yaw = Rotation.from_euler('z', angle).as_matrix()
        middle = Rotation.from_euler('z', angle/2).as_matrix()
        root = body.copy(); root[:3, :3] = yaw@body[:3, :3]
        root[:3, 3] += middle@body[:3, :3]@np.r_[displacement_body, 0.]
        predicted = link.copy(); predicted[:3, :3] = yaw@link[:3, :3]
        predicted[:3, 3] = root[:3, 3]+yaw@(link[:3, 3]-body[:3, 3])
        return predicted

    def reject_without_forecast(self,camera_optical,T_world_link0,previous_applied,*,command_dt=.02,reason='command_stale'):
        """Fresh current conversion and original slew brake; no future claim.

        This is the explicit rejected-command path when a terminal context
        cannot be built. Invalid history must not be passed to another model.
        Native correction/recovery, FCL and integral rollback still run later.
        """
        previous=np.asarray(previous_applied,float)
        if (previous.shape!=(3,)or not np.isfinite(previous).all()or np.any(abs(previous)>self.limits+1e-9)
                or not np.isfinite(command_dt)or command_dt<=0 or reason not in ('command_stale','tracking_limit')):
            raise ValueError('finite actual input, interval and explicit rejected-command reason required')
        delta=np.array([.5,.5,.8])*command_dt
        brake=np.clip(np.zeros(3),np.maximum(-self.limits,previous-delta),np.minimum(self.limits,previous+delta))
        current=self.backend.camera_to_mixed(camera_optical,self.backend.rigid(T_world_link0),self.extrinsic)
        return dict(base_command=brake,native_command=current,camera_supported=False,base_velocity_feasible=False,
            feasible=False,reason=reason,camera_reference_changed=False,current_domain=self.domain.check(current),
            prediction_horizon_s=0.,predicted_min_domain_margin=None,prediction_intervals_s=[],
            forecast_status='unavailable_not_evaluated',planned_slew_brake_model=False,bounded_ramp_history_model=False,
            interpretation='current native conversion and original slew brake only; no forecast support claim')

    def command(self, stamp_s, camera_sample, T_world_link0, T_world_body, preferred_body_velocity,
                measured_body_velocity, *, max_lookahead_s=None,future_policy_velocity=None,previous_applied=None,command_dt=.02,
                native_posture_x_m=None,posture_weight=0.,prediction_provider=None,locked_input=False):
        link, body = self.backend.rigid(T_world_link0), self.backend.rigid(T_world_body)
        preferred, measured = np.asarray(preferred_body_velocity, float), np.asarray(measured_body_velocity, float)
        if (preferred.shape != (3,) or measured.shape != (3,) or not np.isfinite([preferred, measured]).all()
                or np.any(np.abs(preferred) > self.limits+1e-8)):
            raise ValueError('finite supported body velocities required')
        if (prediction_provider is not None and not callable(prediction_provider))or type(locked_input)is not bool:
            raise ValueError('callable private model and boolean lock required')
        if future_policy_velocity is not None:
            future_policy_velocity=np.asarray(future_policy_velocity,float)
            if future_policy_velocity.shape!=(3,) or not np.isfinite(future_policy_velocity).all() or np.any(np.abs(future_policy_velocity)>self.limits+1e-9):
                raise ValueError('finite bounded future native-input plan required')
        if not np.isfinite(posture_weight) or not 0<=posture_weight<=1.:
            raise ValueError('bounded nonnegative soft posture weight required')
        if prediction_provider is not None and posture_weight:
            raise ValueError('private terminal model owns its subordinate posture proposal')
        if (posture_weight and native_posture_x_m is None) or (native_posture_x_m is not None and
                (not np.isfinite(native_posture_x_m) or not self.xyz[0,0]<=native_posture_x_m<=self.xyz[0,1])):
            raise ValueError('soft native posture target must lie inside original outer x bounds')
        low,high=-self.limits.copy(),self.limits.copy()
        if previous_applied is not None:
            previous_applied=np.asarray(previous_applied,float)
            if previous_applied.shape!=(3,) or not np.isfinite(previous_applied).all() or not np.isfinite(command_dt) or command_dt<=0:
                raise ValueError('finite previous input and positive control interval required')
            delta=np.array([.5,.5,.8])*command_dt
            low=np.maximum(low,previous_applied-delta);high=np.minimum(high,previous_applied+delta)
        if locked_input:
            low=np.clip(preferred,low,high);high=low.copy()
        brake=np.clip(np.zeros(3),low,high)
        current = self.backend.camera_to_mixed(camera_sample(stamp_s), link, self.extrinsic)
        current_check = self.domain.check(current)
        if current_check['status'] == 'out_of_policy_domain':
            return dict(base_command=brake, native_command=current, camera_supported=False,
                        base_velocity_feasible=False, feasible=False, reason='out_of_policy_domain',
                        camera_reference_changed=False, current_domain=current_check)
        horizon = self.horizon_s if max_lookahead_s is None else min(self.horizon_s, max_lookahead_s)
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError('finite nonnegative lookahead required')
        intervals = self.prediction_intervals(horizon)
        reserve_ramp_horizon=min(horizon,.8) if self.prediction_profile!='legacy' else horizon
        inverse_extrinsic = np.linalg.inv(self.extrinsic)
        targets = [self.backend.rigid(camera_sample(stamp_s+t))@inverse_extrinsic for t in intervals]
        bounded_prediction=None
        if self.bounded_prediction_class is not None and prediction_provider is None:
            if previous_applied is None or not np.isclose(command_dt,.02,atol=1e-12,rtol=0):
                raise ValueError('actual previous input and original50Hz interval required')
            bounded_prediction=self.bounded_prediction_class(self.backend,stamp_s=stamp_s,body=body,link=link,
                measured_velocity=measured,previous_applied=previous_applied,input_history=self.input_history,
                prediction_intervals=intervals,pending_mode='linear_residual')
        def future_plan(velocity):
            # The from-now input bias persists through this short prediction;
            # the next real tick is still recomputed from fresh measured state.
            return self.future_plan(velocity,preferred,future_policy_velocity,posture_weight)
        def predict(velocity,elapsed):
            if prediction_provider is not None:return self.backend.rigid(prediction_provider(velocity,float(elapsed)))
            if bounded_prediction is not None:
                target=future_plan(velocity)
                return bounded_prediction.predict_link(velocity,velocity if target is None else target,float(elapsed))
            return self.predicted_link(link,body,velocity,elapsed,measured,future_plan(velocity),stamp_s)
        def bounded_derivative(function,at,value):
            columns=[];epsilon=1e-5
            for axis in range(3):
                sample=at.copy();sample[axis]=min(high[axis],at[axis]+epsilon)
                if abs(sample[axis]-at[axis])<1e-12:sample[axis]=max(low[axis],at[axis]-epsilon)
                distance=sample[axis]-at[axis]
                columns.append((function(sample)-value)/distance if abs(distance)>1e-12 else np.zeros_like(value))
            return np.asarray(columns).T
        def margins(velocity):
            values = []
            for elapsed, target in zip(intervals, targets):
                predicted = predict(velocity,elapsed)
                inverse_rotation = predicted[:3, :3].T
                position = np.r_[(inverse_rotation@(target[:3, 3]-predicted[:3, 3]))[:2], target[2, 3]]
                angles = Rotation.from_matrix(inverse_rotation@target[:3, :3]).as_euler('xyz')
                # Current pose cannot be moved instantly to the reserve; ramp
                # the reserve over the same horizon, preserving outer bounds.
                reserve_fraction=min(1.,elapsed/max(reserve_ramp_horizon,1e-9))
                reserve = self.reserve*reserve_fraction
                values.extend(position-self.xyz[:, 0]-reserve)
                values.extend(self.xyz[:, 1]-position-reserve)
                values.extend(angles-self.euler[0]-.015*reserve_fraction)
                values.extend(self.euler[1]-angles-.015*reserve_fraction)
            return np.asarray(values)
        scale = self.limits
        def posture_error(velocity):
            predicted=predict(velocity,horizon)
            x=(predicted[:3,:3].T@(targets[-1][:3,3]-predicted[:3,3]))[0]
            return float((x-native_posture_x_m)/.1)
        center = np.clip(preferred, low, high); accepted = False
        if locked_input:
            accepted=bool(margins(center).min()>=-1e-7)
        # Bounded local linearization of three velocity inputs. Check the actual
        # nonlinear predictions after each solve; never accept the linear model
        # alone. This avoids repeated Python FK within the optimizer's line search.
        for _ in range(3):
            if locked_input:break
            value = margins(center)
            if value.min() >= -1e-7 and np.max(np.abs(center-preferred)) < 1e-9 and not posture_weight:
                accepted = True; break
            epsilon = 1e-5
            if value.min()>=-1e-7 and posture_weight:
                # This soft preference concerns native x only. A bounded
                # one-coordinate quadratic avoids an unnecessary 3-D solve
                # when the camera-domain guard already accepts the input.
                soft=posture_error(center)
                derivative=(bounded_derivative(posture_error,center,soft)[0] if bounded_prediction is not None else
                    (posture_error(center+np.array([epsilon,0.,0.]))-soft)/epsilon)
                candidate=center.copy()
                candidate[0]=np.clip((preferred[0]/scale[0]**2+
                    posture_weight*derivative*(derivative*center[0]-soft))/
                    (1/scale[0]**2+posture_weight*derivative**2),low[0],high[0])
                if margins(candidate).min()>=-1e-7:
                    center=candidate;accepted=True;break
            if prediction_provider is None and bounded_prediction is None:
                jacobian=np.column_stack([(margins(center+np.eye(3)[i]*epsilon)-value)/epsilon for i in range(3)])
            else:
                # A real planned first input must satisfy the same slew and
                # deadline as execution. Differentiate on the admissible side.
                jacobian=bounded_derivative(margins,center,value)
            at = center.copy()
            soft=posture_error(at) if posture_weight else 0.
            soft_jac=(bounded_derivative(posture_error,at,soft) if bounded_prediction is not None else
                np.array([(posture_error(at+np.eye(3)[i]*epsilon)-soft)/epsilon for i in range(3)])) if posture_weight else np.zeros(3)
            objective=lambda v:float(np.sum(((v-preferred)/scale)**2)+posture_weight*(soft+soft_jac@(v-at))**2)
            gradient=lambda v:2*(v-preferred)/scale**2+2*posture_weight*(soft+soft_jac@(v-at))*soft_jac
            result = minimize(objective, at, jac=gradient, method='SLSQP',
                              bounds=list(zip(low,high)),
                              constraints=[dict(type='ineq', fun=lambda v: value+jacobian@(v-at),
                                                jac=lambda v: jacobian)], options=dict(maxiter=15, ftol=1e-9))
            if not result.success or not np.isfinite(result.x).all():
                break
            center = result.x
            if margins(center).min() >= -1e-7:
                accepted = True; break
        applied = center if accepted or locked_input else brake
        requested_supported = bool(accepted and np.max(np.abs(applied-preferred)) <= .015)
        return dict(base_command=applied, native_command=current, camera_supported=accepted,
                    base_velocity_feasible=requested_supported, feasible=None if accepted else False,
                    reason=('accepted' if requested_supported else 'tracking_limit') if accepted else 'solver_infeasible',
                    camera_reference_changed=False, predicted_min_domain_margin=float(margins(applied).min()),
                    preferred_base_command=preferred.tolist(), prediction_horizon_s=horizon,
                    response_profile=self.response_profile,
                    prediction_profile=self.prediction_profile,prediction_intervals_s=intervals.tolist(),
                    planned_slew_brake_model=prediction_provider is not None,locked_input=locked_input,
                    bounded_ramp_history_model=bounded_prediction is not None,
                    planned_future_body_target=(np.asarray(future_plan(applied)).tolist()
                        if bounded_prediction is not None and future_plan(applied) is not None else None),
                    reserve_ramp_horizon_s=reserve_ramp_horizon,
                    native_posture_preference=dict(enabled=bool(posture_weight),x_link0_m=native_posture_x_m,
                        weight=float(posture_weight),predicted_normalized_x_error=posture_error(applied) if posture_weight else None,
                        interpretation='soft local operating preference only; not a training or stability domain certificate',
                        original_hard_workspace_unchanged=True),
                    interpretation='native domain support only; dynamic tracking/contact/collision still require measured validation')


def feedback_native_command(backend, reference_optical, measured_optical, link0, extrinsic,
                            position_gain=.35, orientation_gain=.25,domain=None):
    """Small task-input feedback; the independent authored camera is unchanged."""
    reference, measured = backend.rigid(reference_optical), backend.rigid(measured_optical)
    corrected = reference.copy()
    correction = position_gain*(reference[:3, 3]-measured[:3, 3])
    length = np.linalg.norm(correction)
    corrected[:3, 3] += correction*min(1., .025/max(length, 1e-12))
    rotation = orientation_gain*Rotation.from_matrix(reference[:3, :3]@measured[:3, :3].T).as_rotvec()
    length = np.linalg.norm(rotation)
    corrected[:3, :3] = Rotation.from_rotvec(rotation*min(1., .05/max(length, 1e-12))).as_matrix()@reference[:3, :3]
    native = backend.camera_to_mixed(corrected, link0, extrinsic)
    check = domain.check(native) if domain else backend.check_upstream_domain(native)
    if check['status'] != 'inside_outer_domain_unverified':
        fallback=backend.camera_to_mixed(reference, link0, extrinsic)
        actual=domain.check(fallback) if domain else backend.check_upstream_domain(fallback)
        actual['rejected_feedback_correction']=dict(native_command=native.tolist(),domain_check=check)
        return fallback,actual
    return native, check


class NativeCameraParking:
    """Hysteretic native-input hold after measured braking, never retargeting.

    Mixed-frame commands otherwise change with every measured link0 jitter.
    The authored WORLD pose is still evaluated independently on every sample.
    """
    def __init__(self):
        self.command=None;self.base=None;self.transitions=0

    def update(self,native_command,T_world_reference,T_world_camera,T_world_base,measured_velocity,*,parking):
        from scipy.spatial.transform import Rotation
        native=np.asarray(native_command,float)
        reference,camera,base=map(lambda x:np.asarray(x,float),(T_world_reference,T_world_camera,T_world_base))
        measured=np.asarray(measured_velocity,float)
        position=float(np.linalg.norm(reference[:3,3]-camera[:3,3]))
        angle=float(Rotation.from_matrix(reference[:3,:3]@camera[:3,:3].T).magnitude())
        release=not parking or position>.040 or angle>.12
        if self.base is not None:
            release=release or np.linalg.norm(base[:3,3]-self.base[:3,3])>.020 or Rotation.from_matrix(base[:3,:3]@self.base[:3,:3].T).magnitude()>.04
        if self.command is not None and release:
            self.command=None;self.base=None;self.transitions+=1
        if (self.command is None and parking and position<.030 and angle<.09
                and np.linalg.norm(measured[:2])<.02 and abs(measured[2])<.03):
            self.command=native.copy();self.base=base.copy();self.transitions+=1
        return (native.copy() if self.command is None else self.command.copy()),dict(
            active=self.command is not None,transitions=self.transitions,
            world_position_error_m=position,world_orientation_error_rad=angle,
            original_world_reference_modified=False)


class NativeDomainRecovery:
    """Reject invalid EE inputs while recovering from fresh measured geometry.

    The last legal native posture is a declared hold, never a substituted WORLD
    reference. A bounded base input may recover the original requested domain;
    after two seconds it brakes and keeps reporting the original task failure.
    """
    def __init__(self,domain,backend,limits=(.2,.2,.2),budget_s=2.):
        self.domain,self.backend=domain,backend;self.limits=np.asarray(limits,float)
        self.budget=float(budget_s);self.last=None;self.started=None

    def seed(self,native):
        native=np.asarray(native,float)
        if self.domain.check(native)['status']!='inside_outer_domain_unverified':
            raise ValueError('recovery requires an explicitly legal native hold input')
        self.last=native.copy()

    def update(self,native,T_world_link0,T_world_base,stamp_s,previous_applied,dt):
        native=np.asarray(native,float);check=self.domain.check(native)
        previous=np.asarray(previous_applied,float);delta=np.array([.5,.5,.8])*dt
        if check['status']=='inside_outer_domain_unverified':
            was_recovering=self.started is not None;self.seed(native);self.started=None
            return native,None,dict(active=False,recovered_from_fresh_measurement=was_recovering,requested_domain=check)
        if self.last is None:raise RuntimeError('No legal native posture available for domain failure hold')
        if self.started is None:self.started=float(stamp_s)
        elapsed=float(stamp_s)-self.started
        requested=np.zeros(3)
        unsupported_axes=sorted(set(check.get('violations',[]))-{'x_link0','y_link0','yaw_link0'})
        if elapsed<self.budget and not unsupported_axes:
            # Moving the base toward the requested EE position moves the
            # request back toward the middle of the original mixed domain.
            xyz=np.asarray(check['xyz_outer_bounds']);center=xyz[:2].mean(axis=1)
            local=np.r_[native[:2]-center,0.]
            world=np.asarray(T_world_link0)[:3,:3]@local
            body=np.asarray(T_world_base)[:3,:3].T@world
            requested[:2]=np.clip(body[:2]/.55,-self.limits[:2],self.limits[:2])
            if 'yaw_link0' in check.get('violations',[]):
                yaw=Rotation.from_quat(native[[4,5,6,3]]).as_euler('xyz')[2]
                requested[2]=np.clip(yaw-np.mean(check['euler_outer_bounds_rad']),-self.limits[2],self.limits[2])
        applied=np.clip(requested,np.maximum(-self.limits,previous-delta),np.minimum(self.limits,previous+delta))
        return self.last.copy(),applied,dict(active=True,requested_native_command=native.tolist(),requested_domain=check,
            recovery_elapsed_s=elapsed,recovery_budget_s=self.budget,recovery_timed_out=elapsed>=self.budget,
            unsupported_recovery_axes=unsupported_axes,
            original_world_reference_modified=False,actual_native_hold=True)
