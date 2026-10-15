"""Numerical camera search prior and continuous differential-IK witnesses.

Uses the robot FK/J_eff and actuator inequalities, never the execution WBC's
solve/avoidance policy. A local polytope is a search prior; only a sampled,
state-seeded, time-parameterized lift can pass the kinematic gate. This is not a
global reachability proof, dynamic model, or whole-body collision certificate.
"""
from dataclasses import dataclass, replace

import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER
from .kc_svsdf import advance
from .whole_body_qp import WholeBodyQP, whole_body_kinematics


@dataclass(frozen=True)
class KinematicPolicy:
    step_s: float = .04
    position_tolerance_m: float = .015
    orientation_tolerance_rad: float = .035
    # Angular rows in radians are converted to metres with this length. The
    # singular threshold therefore has an explicit, fixed normalization.
    characteristic_length_m: float = .35
    min_singular_value: float = .015
    prior_horizon_s: float = .25
    max_samples: int = 401

    def __post_init__(self):
        values = [self.step_s, self.position_tolerance_m, self.orientation_tolerance_rad,
                  self.characteristic_length_m, self.min_singular_value, self.prior_horizon_s]
        if not np.isfinite(values).all() or min(values) <= 0 or self.step_s > .1:
            raise ValueError('invalid kinematic policy')
        if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int) or self.max_samples < 2:
            raise ValueError('kinematic max_samples must be an integer >= 2')


POLICY = KinematicPolicy()


def checked_vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result.copy()


class KinematicSearch:
    """Immutable measured seed; each proposed path gets its own continuous lift."""
    def __init__(self, state, previous, t, base_z, limits, *, measured_control=None,
                 commanded_q=None, base_rotation=None, policy=POLICY, scene=None, search_config=None):
        self.state = checked_vector(state, 9, 'state')
        self.command = checked_vector(previous, 8, 'previous command')
        self.velocity = checked_vector(previous if measured_control is None else measured_control,
                                       8, 'current velocity')
        self.velocity_source = 'previous_command_fallback' if measured_control is None else 'measured_control'
        self.commanded_q = None if commanded_q is None else checked_vector(commanded_q, 6, 'commanded joints')
        if not np.isfinite([t, base_z]).all():
            raise ValueError('invalid kinematic timestamp or base height')
        self.t, self.base_z, self.limits, self.policy = float(t), float(base_z), limits, policy
        self.tilt = None
        if base_rotation is not None:
            br = np.asarray(base_rotation, float)
            if (br.shape != (3, 3) or not np.isfinite(br).all() or
                    not np.allclose(br.T@br, np.eye(3), atol=1e-6) or np.linalg.det(br) < .999999):
                raise ValueError('invalid measured base rotation')
            self.tilt = Rotation.from_euler('z', -self.state[2]).as_matrix()@br
        self.weight = np.array([1., 1., 1., *([policy.characteristic_length_m]*3)])
        self.p, self.r, self.j = self.fk(self.state)
        self._prior = None
        from .search_configuration import profile
        self.config = profile(search_config)
        self.scene = scene
        self.collision = self.self_collision = None
        if scene is not None:
            from .dynamic_collision import DynamicCollision
            from .self_collision import SelfCollision
            self.collision = DynamicCollision(dict(scene,format=scene.get('format','jvs_lidar_surfels/v1')), self.t)
            self.self_collision = SelfCollision()

    def fk(self, state):
        br = None if self.tilt is None else Rotation.from_euler('z', state[2]).as_matrix()@self.tilt
        return whole_body_kinematics(state, self.base_z, br)

    def sigma(self, jacobian):
        return float(np.linalg.svd(self.weight[:, None]*jacobian, compute_uv=False)[-1])

    def constraints(self, state, velocity, dt, at, commanded_q=None, initial=False, limits=None):
        # Reuse only the pure actuator inequalities, not a WBC solve or policy.
        bounds = WholeBodyQP(limits or self.limits)
        bounds.previous = velocity.copy()
        bounds.time_s = at
        a, lo, hi = bounds.bounds(state[3:], dt, commanded_q)
        if initial:
            # Measured acceleration and command slew must both be respected.
            bounds.previous = self.command.copy()
            _, cl, ch = bounds.bounds(state[3:], dt, commanded_q)
            lo, hi = np.maximum(lo, cl), np.minimum(hi, ch)
        return a, lo, hi

    def initial_error(self):
        q = self.state[3:]
        if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
            return 'initial_joint_position_limit'
        if (np.any(abs(self.velocity) > self.limits.velocity+1e-6) or
                np.any(abs(self.limits.wheels@self.velocity) > self.limits.wheel_speed+1e-6)):
            return 'initial_velocity_limit'
        if self.sigma(self.j) < self.policy.min_singular_value:
            return 'initial_near_singular_configuration'
        return ''

    def prior(self):
        if self._prior is not None:
            return self._prior
        a, lo, hi = self.constraints(self.state, self.velocity, self.policy.prior_horizon_s,
                                    self.t, self.commanded_q, initial=True)
        error = self.initial_error()
        if np.any(lo > hi):
            error = error or 'initial_braking_or_acceleration_infeasible'
        # Store the coupled polytope, not just independent Cartesian bounds.
        self._prior = dict(model='differential_drive_fk_polytope/v1', snapshot_time_s=self.t,
            velocity_source=self.velocity_source, state=self.state.tolist(), velocity=self.velocity.tolist(),
            camera_position=self.p.tolist(), camera_quaternion_xyzw=Rotation.from_matrix(self.r).as_quat().tolist(),
            effective_jacobian=self.j.tolist(), control_matrix=a.tolist(), control_lower=lo.tolist(),
            control_upper=hi.tolist(), horizon_s=self.policy.prior_horizon_s,
            min_singular_value=self.sigma(self.j), error=error,
            meaning='local search prior only; continuous sampled lift required; no collision certificate')
        return self._prior

    def amplitude(self, route, requested, radial):
        """Directional short-horizon support suggests smaller search seeds.

        It never certifies/rejects a distant pose. The floor keeps exploration
        possible when a later base turn changes instantaneous capability.
        """
        # Preserve the outer exploration tier. A short local linearization
        # must never exclude a multi-second path that turns/reconfigures later.
        if requested > .12:
            return requested
        prior = self.prior()
        left = np.cross(radial, [0., 0., 1.])
        direction = dict(left=left, right=-left, outer=radial, inner=-radial,
                         above=np.array([0., 0., 1.]), below=np.array([0., 0., -1.])).get(route)
        if direction is None or not requested:
            return requested
        a = np.asarray(prior['control_matrix'])
        lo, hi = np.asarray(prior['control_lower']), np.asarray(prior['control_upper'])
        support = 0.
        if not prior['error']:
            result = linprog(-direction@self.j[:3], A_ub=np.vstack([a, -a]), b_ub=np.r_[hi, -lo],
                             bounds=[(None, None)]*8, method='highs')
            if result.success:
                support = max(0., float(-result.fun))*self.policy.prior_horizon_s
        scale = np.clip(support/.22, .15, 1.)
        return float(requested*scale)

    def lift(self, path, end):
        # One greedy inverse-kinematics allocation is not a reachability
        # proof. Try a bounded set of continuous redundant allocations, each
        # seeded from the SAME measured configuration and actuator velocity.
        import time
        started=time.monotonic()
        best=None; accepted=[]; attempts=[]
        allocations = self.config['allocations'] if self.config else ('balanced','path_heading','base_assisted','arm_assisted')
        for allocation in allocations:
            report=self._lift(path,end,allocation)
            report['allocation']=allocation
            attempts.append(dict(allocation=allocation,feasible=report['feasible'],reason=report['reason'],
                                 first_failure_s=report.get('first_failure_s')))
            if report['feasible'] and self.collision is not None:
                from .configuration_preparation import preparation
                underlying=getattr(path,'path',path)
                report['preparation']=preparation(report,max(self.t,getattr(underlying,'activation_s',self.t)),self.limits)
                accepted.append(report)
                if time.monotonic()-started>.8 or report['preparation'].get('duration_s')==0.:
                    break
                continue
            if report['feasible'] or report.get('first_failure_s',self.t)<=self.t:
                return report
            if best is None or report.get('first_failure_s',self.t)>best.get('first_failure_s',self.t):
                best=report
        if accepted:
            best=min(accepted,key=lambda r:(r['preparation']['status']!='verified',
                r['preparation'].get('duration_s') if r['preparation'].get('duration_s') is not None else float('inf'),
                -r['min_joint_margin_rad']))
        best['allocation_attempts']=attempts
        best['planning_wall_s']=time.monotonic()-started
        return best

    def _lift(self, path, end, allocation, project_camera=False):
        """Find one continuous bounded control witness, relinearizing each step.

        Position/velocity/braking bounds, discrete actuator acceleration and
        coupled wheel limits are checked at every step. Nonlinear FK residuals
        and singular values are audited at endpoints and midpoints. Failure
        means this local search found no witness, not global impossibility.
        """
        cfg, lim = self.policy, self.limits
        underlying=getattr(path,'path',path)
        activation=getattr(underlying,'activation_s',self.t)
        seed_trace=getattr(underlying,'joint_seed',[])
        if seed_trace:
            seed_times=np.array([row['t'] for row in seed_trace])
            seed_states=np.array([row['state'] for row in seed_trace])
            seed_controls=np.array([row['control'] for row in seed_trace])
        report = dict(feasible=False, reason='', snapshot_time_s=self.t, end_s=float(end),
            velocity_source=self.velocity_source, step_s=cfg.step_s,
            position_tolerance_m=cfg.position_tolerance_m, orientation_tolerance_rad=cfg.orientation_tolerance_rad,
            singular_value_floor=cfg.min_singular_value, characteristic_length_m=cfg.characteristic_length_m,
            min_singular_value=self.sigma(self.j), min_joint_margin_rad=float(min(
                np.min(self.state[3:]-JOINT_LOWER), np.min(JOINT_UPPER-self.state[3:]))),
            max_velocity_ratio=0., max_acceleration_ratio=0., max_wheel_velocity_ratio=0.,
            max_wheel_acceleration_ratio=0., max_position_error_m=0., max_orientation_error_rad=0.,
            max_constraint_residual=0., samples=0, trace=[],
            scope='sampled continuous kinematic witness; no WBC, dynamics or whole-body collision prediction')
        if self.collision is not None:
            report.update(scope='continuous FK lift with observed whole-body and self-collision swept bounds; no dynamics or future truth',
                whole_body_collision_checked=True, min_body_clearance_m=1e6, min_self_clearance_m=1e6)
        def reject(reason, at):
            report.update(reason=reason, first_failure_s=float(at), failure_class=(
                'physical_collision' if reason in ('whole_body_collision','self_collision') else
                'physical_limit' if reason in ('initial_joint_position_limit','initial_velocity_limit','joint_position_limit') else
                'unverified_numerical_search'),
                interpretation='this allocation failed verification; not a global infeasibility proof')
            return report
        error = self.initial_error()
        if error:
            return reject(error, self.t)
        if self.collision is not None:
            br=None if self.tilt is None else Rotation.from_euler('z',self.state[2]).as_matrix()@self.tilt
            gap=float(self.collision.clearance(self.state,self.base_z,br).min())
            self_gap=float(self.self_collision.distances(self.state).min())
            if gap<self.collision.margin or self_gap<self.self_collision.margin:
                return reject('whole_body_collision' if gap<self.collision.margin else 'self_collision',self.t)
        if not np.isfinite(end) or end < self.t:
            return reject('invalid_kinematic_horizon', self.t)
        count = max(1, int(np.ceil((end-self.t)/cfg.step_s)))
        if count+1 > cfg.max_samples:
            return reject('kinematic_sample_budget_exceeded', self.t)
        times = np.linspace(self.t, end, count+1) if end > self.t else np.array([self.t])
        state, velocity = self.state.copy(), self.velocity.copy()
        cq = None if self.commanded_q is None else self.commanded_q.copy()
        # Require braking by the shot end; a terminal hold does not extend it.
        deadline = path.duration if lim.deadline_s is None else min(path.duration, lim.deadline_s)
        bounded_limits = replace(lim, deadline_s=deadline)
        # Generation reserves leave room for measured-state correction during
        # execution. These tighten the search; actual hard limits are unchanged.
        projection_limits=replace(bounded_limits,position_margin=max(lim.position_margin,.12),
            velocity=.85*lim.velocity,acceleration=.75*lim.acceleration,
            wheel_speed=.9*lim.wheel_speed,wheel_acceleration=.8*lim.wheel_acceleration) if project_camera else bounded_limits
        solver = None
        solver_shape = None
        preview_preference=np.zeros(8)
        rows, cols = np.triu_indices(8)
        def audit(x, u, at):
            p, r, j = self.fk(x)
            tp, tr = path.sample(float(at))[:2]
            ep = float(np.linalg.norm(p-tp))
            er = float(Rotation.from_matrix(r@tr.T).magnitude())
            sigma = self.sigma(j)
            margin = float(min(np.min(x[3:]-JOINT_LOWER), np.min(JOINT_UPPER-x[3:])))
            report['min_singular_value'] = min(report['min_singular_value'], sigma)
            report['min_joint_margin_rad'] = min(report['min_joint_margin_rad'], margin)
            report['max_position_error_m'] = max(report['max_position_error_m'], ep)
            report['max_orientation_error_rad'] = max(report['max_orientation_error_rad'], er)
            if margin < -1e-6:
                return 'joint_position_limit'
            if sigma < cfg.min_singular_value:
                return 'near_singular_configuration'
            if (ep > cfg.position_tolerance_m or er > cfg.orientation_tolerance_rad) and not (project_camera and at>=activation):
                return 'kinematic_tracking_unreachable'
            return ''
        def record(at):
            p, r, _ = self.fk(state)
            report['trace'].append(dict(t=float(at), state=state.tolist(), control=velocity.tolist(),
                position=p.tolist(), quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist()))
            report['samples'] += 1
        error = audit(state, velocity, self.t)
        record(self.t)
        if error:
            return reject(error, self.t)
        for i, (at, after) in enumerate(zip(times[:-1], times[1:])):
            dt = float(after-at)
            a, lo, hi = self.constraints(state, velocity, dt, float(at), cq, initial=i == 0,
                limits=projection_limits if at>=activation else bounded_limits)
            if np.any(lo > hi):
                return reject('braking_or_acceleration_infeasible', at)
            p, r, j = self.fk(state)
            tp, tr = path.sample(float(after))[:2]
            desired = np.r_[tp-p, Rotation.from_matrix(tr@r.T).as_rotvec()]/dt
            factor = .5 if lim.command_ramp else 1.
            offset = .5*j@velocity if lim.command_ramp else np.zeros(6)
            if self.collision is not None and not (seed_trace and at>=activation):
                from .tpr_wbc import witnesses, primitive_values
                from .shot_predictive_qp import integration_matrix
                br = None if self.tilt is None else Rotation.from_euler('z',state[2]).as_matrix()@self.tilt
                collision = self.collision.at_time(float(at)-self.t)
                centers, radii, indices, distances, gradients = witnesses(collision,state,self.base_z,br)
                all_rows = gradients@integration_matrix(state[2])
                all_motion=collision.primitive_motion(state,centers,self.base_z,br)
                rate=all_rows@velocity+all_motion
                mask = (distances < .35) | (rate < -.1)
                rows_collision=all_rows[mask]
                motion=all_motion[mask]
                eps=.002
                plus,minus=advance(state,velocity,eps),advance(state,velocity,-eps)
                rp=None if br is None else Rotation.from_euler('z',velocity[1]*eps).as_matrix()@br
                rm=None if br is None else Rotation.from_euler('z',-velocity[1]*eps).as_matrix()@br
                dp=primitive_values(collision.geometry,plus,centers+collision.velocities[indices]*eps,self.base_z,rp)-radii[:,None]
                dm=primitive_values(collision.geometry,minus,centers-collision.velocities[indices]*eps,self.base_z,rm)-radii[:,None]
                curvature=(dp-2*distances+dm)/eps**2
                gap=distances-collision.margin-.006
                lows=np.maximum(rows_collision@velocity+dt*(-curvature-4*rate-4*gap)[mask],(-2*gap-all_motion)[mask])
                sd, sg = self.self_collision.terms(state)
                smask = sd < .08
                rows_collision = np.vstack([rows_collision,(sg@integration_matrix(state[2]))[smask]])
                lows = np.r_[lows,-(sd[smask]-self.self_collision.margin-.002)/max(dt,.1)]
                lo = np.r_[lo, lows-(.5*rows_collision@velocity if lim.command_ramp else 0.)]
                hi = np.r_[hi, np.full(len(lows),np.inf)]
                a = np.vstack([a,factor*rows_collision])
                # Active witness rows and their sparsity change with geometry.
                solver = None
            elif self.collision is not None:
                # A control seed is checked by the nonlinear swept oracle
                # below. Re-imposing a different sufficient local barrier can
                # reject the same already separated motion without adding a
                # physical constraint. Actuator bounds and FK checks remain.
                br=None if self.tilt is None else Rotation.from_euler('z',state[2]).as_matrix()@self.tilt
                collision=self.collision.at_time(float(at)-self.t)
            jw = self.weight[:, None]*j*factor
            # Damped differential IK; a small centering objective keeps the
            # continuous branch away from limits without permitting task jumps.
            metric=(np.ones(8) if allocation=='balanced' else
                    np.r_[.08,.2,np.ones(6)] if allocation in ('base_assisted','path_heading','anticipatory','base_left','base_right','base_retreat') else np.r_[3.,2.,np.ones(6)])
            # The joint-range midpoint can be a straight-arm singularity.
            # Preserve the measured bent posture in alternate allocations.
            anchor=(JOINT_LOWER+JOINT_UPPER)/2 if allocation=='balanced' else self.state[3:]
            center = np.r_[0., 0., (.15 if allocation=='balanced' else .6)*(anchor-state[3:])]
            if allocation in ('path_heading','anticipatory') and (not seed_trace or at<activation):
                planar=path.sample(float(after))[2][:2]
                if np.linalg.norm(planar)>.005:
                    angle=(np.arctan2(planar[1],planar[0])-state[2]+np.pi)%(2*np.pi)-np.pi
                    if abs(angle)>np.pi/2:
                        angle=angle-np.sign(angle)*np.pi
                    center[0]=planar@np.array([np.cos(state[2]),np.sin(state[2])])
                    center[1]=1.5*angle
            if (allocation=='anticipatory' and self.collision is not None and not seed_trace and
                    at>=getattr(getattr(path,'path',path),'activation_s',self.t)):
                # Future path/obstacle geometry supplies a redundancy gradient.
                # This is only a seed preference; all task residuals, actuator
                # constraints and nonlinear sweeps below still have to pass.
                if i%4==0:
                    inverse=np.linalg.pinv(j,rcond=1e-5)
                    null=np.eye(8)-inverse@j
                    preview_preference=np.zeros(8)
                    for horizon in (.5,1.0):
                        future_t=min(float(at)+horizon,path.duration)
                        future_p,future_r=path.sample(future_t)[:2]
                        move=np.r_[future_p-p,Rotation.from_matrix(future_r@r.T).as_rotvec()]
                        nominal=np.clip(inverse@move/horizon,-lim.velocity,lim.velocity)
                        future_state=advance(state,nominal,horizon)
                        future_rotation=None if self.tilt is None else Rotation.from_euler('z',future_state[2]).as_matrix()@self.tilt
                        forecast=self.collision.at_time(future_t-self.t)
                        _,_,_,d,gc=witnesses(forecast,future_state,self.base_z,future_rotation)
                        rows_future=gc@integration_matrix(future_state[2])
                        ids=np.flatnonzero(d.ravel()<.18)
                        if len(ids):
                            rr=rows_future.reshape(-1,8)[ids]@null
                            lack=np.clip(.18-d.ravel()[ids],0.,.35)
                            preview_preference+=rr.T@(lack/np.maximum(np.sum(rr*rr,axis=1),.02))/max(1,len(ids))
                        # Predict direction-dependent joint stopping reserve.
                        q=future_state[3:]
                        reserve=np.minimum(q-JOINT_LOWER,JOINT_UPPER-q)
                        pull=np.where(q<(JOINT_LOWER+JOINT_UPPER)/2,1.,-1.)*np.maximum(0.,.25-reserve)
                        preview_preference+=null@np.r_[0.,0.,pull]
                    preview_preference=np.clip(2.*preview_preference,-.5*lim.velocity,.5*lim.velocity)
                center+=preview_preference
            if allocation in ('base_left','base_right','base_retreat'):
                # Distinct continuous redundant branches, not independent IK
                # jumps. The fixed camera equality is audited at every step.
                center[:2]=[-.2,0.] if allocation=='base_retreat' else [.12,.65 if allocation=='base_left' else -.65]
            if seed_trace and at>=activation:
                expected=np.array([np.interp(at,seed_times,seed_states[:,j]) for j in range(9)])
                delta=expected-state;delta[2]=(delta[2]+np.pi)%(2*np.pi)-np.pi
                center=np.array([np.interp(after,seed_times,seed_controls[:,j]) for j in range(8)])
                center+=.8*np.r_[delta[:2]@np.array([np.cos(state[2]),np.sin(state[2])]),delta[2:]]
                metric=np.ones(8)
            h = jw.T@jw+np.diag(metric)*1e-5
            g = -jw.T@(self.weight*(desired-offset))-1e-5*metric*center
            if seed_trace and at>=activation:
                # Reproduce the verified continuous control seed instead of
                # solving its discretized camera chord into a different IK
                # branch. Corrections are bounded by the same constraints;
                # strict nonlinear FK audits still reject any lost tracking.
                h=np.eye(8)
                g=-center
            pm = sparse.csc_matrix((h[rows, cols], (rows, cols)), shape=(8, 8))
            if solver_shape != a.shape:
                solver = None
            if solver is None:
                solver = osqp.OSQP()
                solver_shape = a.shape
                solver.setup(P=pm, q=g, A=sparse.csc_matrix(a), l=lo, u=hi, verbose=False,
                    eps_abs=1e-8, eps_rel=1e-8, max_iter=4000, polish=False, warm_start=True)
            else:
                solver.update(Px=pm.data, q=g, l=lo, u=hi)
            result = solver.solve()
            if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
                # Use the shared convex solver's feasible-start refinement;
                # an iteration limit is not evidence of unreachable geometry.
                from .tpr_wbc import qp
                try:
                    control,_=qp(h,g,a,lo,hi,velocity)
                except RuntimeError as error:
                    report['solver_failure']=str(error)
                    return reject('kinematic_qp_infeasible', at)
            else:
                control = result.x.copy()
            residual = float(max(0., np.max(lo-a@control), np.max(a@control-hi)))
            report['max_constraint_residual'] = max(report['max_constraint_residual'], residual)
            if residual > 1e-6:
                return reject('kinematic_constraint_residual', at)
            report['max_velocity_ratio'] = max(report['max_velocity_ratio'], float(np.max(abs(control)/lim.velocity)))
            report['max_acceleration_ratio'] = max(report['max_acceleration_ratio'], float(np.max(abs(control-velocity)/(dt*lim.acceleration))))
            report['max_wheel_velocity_ratio'] = max(report['max_wheel_velocity_ratio'], float(np.max(abs(lim.wheels@control))/lim.wheel_speed))
            report['max_wheel_acceleration_ratio'] = max(report['max_wheel_acceleration_ratio'], float(np.max(abs(lim.wheels@(control-velocity)))/(dt*lim.wheel_acceleration)))
            mid_control = .75*velocity+.25*control if lim.command_ramp else control
            mid = advance(state, mid_control, dt/2)
            error = audit(mid, mid_control, (at+after)/2)
            if error:
                return reject(error, (at+after)/2)
            integrated = .5*(velocity+control) if lim.command_ramp else control
            if self.collision is not None:
                proof=collision.sweep(state,integrated,dt,self.base_z,max_samples=33,base_rotation=br)
                body=float(proof['lower'].min())
                if lim.command_ramp:
                    body-=dt*(abs(control[0]-velocity[0])+np.abs(control[1:]-velocity[1:])@collision.geometry.reach)/2
                self_gap=self.self_collision.sweep(state,velocity if lim.command_ramp else control,control,dt)
                report['min_body_clearance_m']=min(report['min_body_clearance_m'],body)
                report['min_self_clearance_m']=min(report['min_self_clearance_m'],self_gap)
                if body < collision.margin or self_gap < self.self_collision.margin:
                    return reject('whole_body_collision' if body<collision.margin else 'self_collision',at)
            state = advance(state, integrated, dt)
            if cq is not None:
                cq = cq+dt*integrated[2:]
            velocity = control
            error = audit(state, velocity, after)
            record(after)
            if error:
                return reject(error, after)
        report.update(feasible=True, reason='sampled_kinematic_witness',
                      camera_projected_for_generation=bool(project_camera))
        return report


def summary(report):
    return {k: v for k, v in report.items() if k != 'trace'}


def installation_error(report, state, velocity, t, limits):
    """Reject a changed redundant configuration even when camera FK is identical."""
    trace = report.get('trace', [])
    if not report.get('feasible') or not trace or not trace[0]['t'] <= t <= trace[-1]['t']:
        return 'missing_fresh_kinematic_witness'
    times = [row['t'] for row in trace]
    states = np.array([row['state'] for row in trace])
    expected = np.array([np.interp(t, times, states[:, j]) for j in range(9)])
    delta = np.asarray(state)-expected
    delta[2] = (delta[2]+np.pi)%(2*np.pi)-np.pi
    if np.linalg.norm(delta[:2]) > .025 or np.max(abs(delta[2:])) > .05:
        return 'configuration_left_verified_kinematic_neighborhood'
    if velocity is not None:
        actual = checked_vector(velocity, 8, 'installation velocity')
        controls = np.array([row['control'] for row in trace])
        predicted = np.array([np.interp(t, times, controls[:, j]) for j in range(8)])
        # The witness and the independent WBC may choose different redundant
        # allocations after the shared measured seed. Two admissible velocity
        # paths can separate by at most 2*a*age (plus one sampling interval).
        # Comparing with a single fixed-step allowance wrongly prescribes the
        # witness controls to the execution WBC. State/braking gates still apply.
        age = max(0., t-report['snapshot_time_s'])
        allowance = limits.acceleration*(2*age+report['step_s'])
        if np.max(abs(actual-predicted)/allowance) > 1.+1e-6:
            return 'velocity_left_verified_kinematic_neighborhood'
    return None
