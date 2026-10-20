"""Trajectory-preserving predictive reconfiguration (TPR-WBC).

Eight executable inputs and a six-dimensional camera task. The fast hierarchy
uses endpoint velocities (a=(u-u_measured)/dt). Safety and actuator rows remain
hard. Ordered combo actions can advance along their geometric path to prepare
body clearance; exact-camera tracking remains the ordinary single-shot mode.
Prediction is a local SQP, not a recursive-feasibility guarantee.
"""
import time
import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize, LinearConstraint
from scipy.spatial.transform import Rotation

from .whole_body_qp import whole_body_kinematics
from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER
from .shot_predictive_qp import integration_matrix
from .kc_svsdf import advance
from .kc_svsdf import primitive_sdf
from .coordination import CoordinationPreview
from .actuation import certify_control_ramp


def task_margin_witness(command, J, desired, A, lo, hi, R, low, scale):
    """A feasible exact-task point proves a lower bound on the margin LP.

    Project away the task QP's numerical equality residual, then check every
    actuator row again. Only a bound above the warning threshold can bypass
    the LP; small/negative margins still need its optimum for classification.
    """
    point = command + np.linalg.lstsq(J, desired-J@command, rcond=None)[0]
    values = A@point
    if (np.max(np.abs(J@point-desired)) > 1e-10 or
            np.any(values < lo-1e-10) or np.any(values > hi+1e-10)):
        return None
    lower = float(np.min((R@point-low)/scale))
    return lower if lower >= .05+1e-8 else None


def response_coefficients(tau, duration, ramp=True):
    """u_end = decay*u_measured + old_weight*cmd_old + gain*cmd_new."""
    tau=np.asarray(tau,float)
    decay=np.exp(-duration/np.maximum(tau,1e-12))
    gain=1-tau/duration*(1-decay) if ramp else 1-decay
    old_weight=1-decay-gain
    return decay,old_weight,gain


def response_velocity(old,command0,command1,tau,duration,t):
    """Exact velocity of a first-order actuator under a linear command ramp."""
    tau=np.asarray(tau,float)
    decay=np.exp(-t/np.maximum(tau,1e-12))
    slope=(command1-command0)/duration
    value=command0+slope*(t-tau)+(old-command0+tau*slope)*decay
    return np.where(tau > 0.,value,command0+slope*t)


def feedback_frequency(nominal,dt,latency=0.):
    """Bound delayed feedback gain; actuator and safety constraints are unchanged."""
    if not np.isfinite(latency) or latency<0:
        raise ValueError('feedback latency must be finite and nonnegative')
    return min(nominal,.6/dt,.35/latency if latency>0 else float('inf'))


def small_qp(H, g, A, lo, hi):
    """Solve the identical small convex problem without cubic enumeration."""
    from .small_convex_qp import solve
    return solve(H,g,A,lo,hi)


def qp(H, g, A, lo, hi, warm=None):
    try:
        return _solve_qp(H, g, A, lo, hi, warm)
    except RuntimeError as error:
        # Preserve the final failing mathematical problem for independent
        # replay. This does not turn an infeasible command into a fallback.
        error.qp_problem = dict(H=H, g=g, A=A, lo=lo, hi=hi)
        raise


def _solve_qp(H, g, A, lo, hi, warm=None):
    if np.any(lo > hi):
        raise RuntimeError('TPR hard constraint intersection is empty')
    # Eliminate exact equalities with an undamped SVD. Besides making the
    # hierarchy exact, this avoids ADMM fighting six task rows against nearly
    # parallel active actuator faces. The original rows are checked afterward.
    original = A, lo, hi
    equal = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    offset, Z = np.zeros(len(g)), np.eye(len(g))
    if np.any(equal):
        offset = np.linalg.lstsq(A[equal], lo[equal], rcond=1e-10)[0]
        if np.max(np.abs(A[equal]@offset-lo[equal])) > 1e-7:
            raise RuntimeError('TPR camera/actuator equalities conflict')
        Z = null_space(A[equal], rcond=1e-10)
        g, H = Z.T@(g+H@offset), Z.T@H@Z
        lo, hi = lo[~equal]-A[~equal]@offset, hi[~equal]-A[~equal]@offset
        A = A[~equal]@Z
    if not Z.shape[1]:
        value = original[0]@offset
        residual = max(0., float(np.max(original[1]-value)), float(np.max(value-original[2])))
        if residual > 1e-6:
            raise RuntimeError('TPR equality solution violates safety')
        return offset, residual
    norm = np.linalg.norm(A, axis=1)
    nonzero = norm > 1e-10
    if np.any(lo[~nonzero] > 1e-7) or np.any(hi[~nonzero] < -1e-7):
        raise RuntimeError('TPR task-conditioned safety set empty')
    A, lo, hi = A[nonzero]/norm[nonzero, None], lo[nonzero]/norm[nonzero], hi[nonzero]/norm[nonzero]
    objective_scale = max(1., float(np.max(np.abs(H))), float(np.max(np.abs(g))))
    H, g = H/objective_scale, g/objective_scale
    if len(g) <= 2 and len(A) < 150 and np.linalg.eigvalsh(H).min() > 1e-10:
        reduced = small_qp(H, g, A, lo, hi)
        answer = offset+Z@reduced
        AA, ll, hh = original
        residual = max(0., float(np.max(ll-AA@answer)), float(np.max(AA@answer-hh)))
        if residual > 1e-6:
            raise RuntimeError('TPR reduced QP residual')
        return answer, residual
    solver = osqp.OSQP()
    solver.setup(P=sparse.csc_matrix(np.triu(H)), q=g, A=sparse.csc_matrix(A),
                 l=lo, u=hi, verbose=False, eps_abs=1e-10, eps_rel=1e-10,
                 eps_prim_inf=1e-9, max_iter=20000, polish=True)
    if warm is not None:
        solver.warm_start(x=Z.T@(warm-offset))
    result = solver.solve()
    valid = (result.x is not None and result.y is not None
             and np.isfinite(np.asarray(result.x, float)).all()
             and np.isfinite(np.asarray(result.y, float)).all())
    stationarity = (np.inf if not valid else
                    np.max(np.abs(H@result.x+g+A.T@result.y)))
    converged = valid and (result.info.status_val in (1, 2) or
        (result.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED') and stationarity < 1e-6))
    answer = offset+Z@result.x if valid else offset
    AA,ll,hh = original
    residual = max(0., float(np.max(ll-AA@answer)), float(np.max(AA@answer-hh)))
    if (not converged or residual > 1e-6) and len(g) <= 32:
        # Older bundled OSQP versions can stall on nearly parallel faces.
        # A dense active-set refinement solves the IDENTICAL convex problem.
        # OSQP's infeasibility certificate is finite but is NOT a primal
        # iterate. Start refinement at a feasible point of the same polytope,
        # and distinguish infeasibility from numerical non-convergence.
        upper, lower = np.isfinite(hi), np.isfinite(lo)
        feasibility=linprog(np.zeros(len(g)),A_ub=np.vstack([A[upper],-A[lower]]),
            b_ub=np.r_[hi[upper],-lo[lower]],bounds=[(None,None)]*len(g),method='highs')
        if feasibility.status == 2:
            raise RuntimeError('TPR hard safety constraints infeasible (independent LP)')
        seed = feasibility.x if feasibility.success else (np.zeros(len(g)) if warm is None else Z.T@(warm-offset))
        refined = minimize(lambda x: .5*x@H@x+g@x, seed, jac=lambda x:H@x+g,
                           constraints=LinearConstraint(A,lo,hi), method='SLSQP',
                           options=dict(ftol=1e-12,maxiter=300))
        if refined.success and np.isfinite(refined.x).all():
            answer=offset+Z@refined.x
            residual=max(0.,float(np.max(ll-AA@answer)),float(np.max(AA@answer-hh)))
            converged=True
    if not converged:
        raise RuntimeError('TPR QP: '+result.info.status)
    if not np.isfinite(answer).all() or residual > 1e-6:
        raise RuntimeError(f'TPR hard residual {residual}')
    return answer, residual


def qp_with_reserve(H,g,A,lo,hi,warm,rows,targets,weight):
    """Penalize a clearance-rate deficit, never surplus safe separation."""
    count=len(rows)
    if not count:return qp(H,g,A,lo,hi,warm)
    size=len(g)
    extended_H=np.zeros((size+count,size+count));extended_H[:size,:size]=H
    extended_H[size:,size:]=weight*np.eye(count)
    extended_A=np.vstack([np.pad(A,((0,0),(0,count))),
        np.column_stack([rows,np.eye(count)]),np.column_stack([np.zeros((count,size)),np.eye(count)])])
    seed=np.r_[warm,np.maximum(0.,targets-rows@warm)]
    command,residual=qp(extended_H,np.r_[g,np.zeros(count)],extended_A,
        np.r_[lo,targets,np.zeros(count)],np.r_[hi,np.full(2*count,np.inf)],seed)
    return command[:size],residual


def witnesses(collision, state, z=0., rotation=None,with_motion=False):
    centers, radii = collision.active_obstacles(state, z, rotation, per_body=None)
    # Incoming movers must not be lost by a static nearest-neighbour selection.
    indices = np.unique(np.r_[collision.tree.query(centers, k=1)[1],
                             np.flatnonzero(collision.track_ids > 0)])
    centers, radii = collision.centers[indices], collision.radii[indices]
    result=collision.geometry.primitive_distances(state,centers,z,rotation,with_normals=with_motion)
    d,G=result[:2]
    values=centers,radii,indices,d-radii[:,None],G
    if with_motion:
        motion=np.einsum('npi,ni->np',result[2],collision.velocities[indices])-collision.radius_rates[indices,None]
        motion[collision.track_ids[indices]<=0]=0.
        return (*values,motion)
    return values


def primitive_values(geometry,state,centers,z=0.,rotation=None):
    """Distance-only curvature probes, avoiding unused point Jacobians."""
    frames=geometry.frames(state,z,rotation)[0]
    return geometry.primitive_arrays(frames,centers)


def motion_evidence(scene):
    """Small causal model identity, independent of map revision/point density."""
    return dict(stamp=float(scene.get('observation_stamp', 0.)), tracks={
        int(row['id']): (np.asarray(row['center'], float).copy(),
                         np.asarray(row['velocity'], float).copy())
        for row in scene.get('tracks', [])})


def motion_changed(previous, current):
    if previous is None:
        return True
    a, b = previous['tracks'], current['tracks']
    if a.keys() != b.keys():
        return True
    age = max(0., current['stamp']-previous['stamp'])
    return any(np.linalg.norm(b[key][1]-velocity) > .08 or
               np.linalg.norm(b[key][0]-(center+velocity*age)) > .035
               for key, (center, velocity) in a.items())


class ReconfigurationPreview:
    """Short horizon, relinearized nonholonomic configuration SQP.

    Every knot has a camera equality, joint/velocity/acceleration/wheel bounds,
    and all nearby primitive-obstacle inequalities. Nonlinear interpolation is
    checked before accepting a plan. Plans only supply a secondary preference.
    """
    def __init__(self, horizon=1.5, nodes=6, period=.4):
        self.horizon, self.nodes, self.period = horizon, nodes, period
        self.updated = -np.inf
        self.times = self.controls = self.states = None
        self.trajectory = None
        self.anchor = None
        self.info = {}

    def reset(self, trajectory):
        if self.trajectory is not trajectory:
            self.trajectory = trajectory
            self.updated = -np.inf
            self.times = self.controls = self.states = None
            self.anchor = None
            self.info = dict(tpr_prediction_valid=False)

    def sample(self, t):
        return np.array([np.interp(t, self.times, self.controls[:, j]) for j in range(8)])

    def update(self, c, state, velocity, t, dt, z, rotation, commanded_q):
        if self.trajectory is None or t >= self.trajectory.duration-1e-9:
            return np.zeros(8)
        observed = motion_evidence(c.collision.scene)
        changed = motion_changed(getattr(self, 'observed_motion', None), observed)
        if t-self.updated < self.period-1e-8 and not changed:
            return np.zeros(8) if self.controls is None else self.sample(t+dt)
        started = time.perf_counter()
        self.updated = t
        self.observed_motion = observed
        body_mode=getattr(c,'body_aware',False) or getattr(c,'body_adaptation_seen',False)
        n = max(self.nodes,10) if body_mode else self.nodes
        horizon=max(self.horizon,2.5) if body_mode else self.horizon
        end = min(t+horizon, self.trajectory.duration)
        h = (end-t)/n
        times = t+np.arange(1, n+1)*h
        nominal = (np.tile(velocity, (n, 1)) if self.controls is None else
                   np.array([self.sample(s) for s in times]))
        nv = 8*n
        E = [np.pad(np.eye(8), ((0, 0), (8*k, nv-8*(k+1)))) for k in range(n)]
        tilt = None if rotation is None else Rotation.from_euler('z', -state[2]).as_matrix()@rotation
        limit = c.limits
        response_active=bool(np.any(limit.response_time > 0.))
        if self.anchor is None:
            self.anchor = state.copy()
        offset = np.zeros(6) if commanded_q is None else np.asarray(commanded_q)-state[3:]
        qlo = np.maximum(JOINT_LOWER+limit.position_margin, JOINT_LOWER+limit.position_margin-offset)
        qhi = np.minimum(JOINT_UPPER-limit.position_margin, JOINT_UPPER-limit.position_margin-offset)
        failure = ''
        accepted = False
        first_conflict = None
        from .combo import is_combo
        from .body_anticipation import path_frame
        from types import SimpleNamespace
        geometric=getattr(c,'body_aware',False) or getattr(c,'body_adaptation_seen',False)
        initial_p,initial_r,_=whole_body_kinematics(state,z,rotation)
        initial_frame=path_frame(SimpleNamespace(),self.trajectory,initial_p,t) if geometric else None
        capture_position=np.zeros(3) if initial_frame is None else initial_p-initial_frame[0]
        capture_rotation=np.zeros(3) if initial_frame is None else Rotation.from_matrix(initial_r@initial_frame[1].T).as_rotvec()
        capture_duration=min(1.,end-t)
        def capture_frame(frame,elapsed):
            tp,tr,transform,tangent,progress=frame
            u=np.clip(elapsed/max(capture_duration,1e-6),0.,1.)
            weight=1.-(10*u**3-15*u**4+6*u**5)
            return tp+weight*capture_position,Rotation.from_rotvec(weight*capture_rotation).as_matrix()@tr,transform,tangent,progress
        if geometric and (self.controls is None or changed):
            from .body_anticipation import preview_seed
            nominal=preview_seed(c,self.trajectory,state,velocity,t,h,n,z,rotation)
        for iteration in range(6 if geometric else 4):
            H, g = np.eye(nv)*1e-5, np.zeros(nv)
            rows, lows, highs = [], [], []
            x = state.copy(); affine = x.copy(); sensitivity = np.zeros((9, nv))
            old, oldE, oldaff = velocity, np.zeros((8, nv)), velocity
            oldcmd=c.previous.copy() if response_active and hasattr(c,'previous') else velocity.copy()
            oldcmdE,oldcmdaff=np.zeros((8,nv)),oldcmd.copy()
            command_q_affine=state[3:].copy() if commanded_q is None else np.asarray(commanded_q).copy()
            command_q_sensitivity=np.zeros((6,nv))
            projector=SimpleNamespace()

            def constrain(M, low, high):
                rows.append(M); lows.append(np.broadcast_to(low, M.shape[0])); highs.append(np.broadcast_to(high, M.shape[0]))

            def cost(M, target, weight):
                nonlocal H, g
                H += weight*M.T@M; g -= weight*M.T@target

            for k, select in enumerate(E):
                diff = select-oldE
                constrain(select, -limit.velocity, limit.velocity)
                constrain(diff, oldaff-limit.acceleration*h, oldaff+limit.acceleration*h)
                constrain(limit.wheels@select, -limit.wheel_speed, limit.wheel_speed)
                constrain(limit.wheels@diff, limit.wheels@oldaff-limit.wheel_acceleration*h,
                          limit.wheels@oldaff+limit.wheel_acceleration*h)
                if limit.deadline_s is not None:
                    remaining = max(0., limit.deadline_s-times[k])
                    constrain(select, -limit.acceleration*remaining, limit.acceleration*remaining)
                    constrain(limit.wheels@select, -limit.wheel_acceleration*remaining, limit.wheel_acceleration*remaining)
                cost(select, np.zeros(8), .1*h)
                cost(diff, oldaff, .04/h)
                decay,beta,gain=response_coefficients(limit.response_time,h)
                cmdE=(select-decay[:,None]*oldE-beta[:,None]*oldcmdE)/gain[:,None]
                cmdaff=(-decay*oldaff-beta*oldcmdaff)/gain
                cmd=(nominal[k]-decay*old-beta*oldcmd)/gain
                if response_active:
                    constrain(cmdE,-limit.velocity-cmdaff,limit.velocity-cmdaff)
                    constrain(limit.wheels@cmdE,-limit.wheel_speed-limit.wheels@cmdaff,limit.wheel_speed-limit.wheels@cmdaff)
                    constrain(cmdE-oldcmdE,-limit.acceleration*h-cmdaff+oldcmdaff,limit.acceleration*h-cmdaff+oldcmdaff)
                    constrain(limit.wheels@(cmdE-oldcmdE),-limit.wheel_acceleration*h-limit.wheels@(cmdaff-oldcmdaff),
                              limit.wheel_acceleration*h-limit.wheels@(cmdaff-oldcmdaff))
                    responding=limit.response_time > 0.
                    tau=limit.response_time[responding]
                    constrain((cmdE-select)[responding]/tau[:,None],
                              -limit.acceleration[responding]-cmdaff[responding]/tau,
                              limit.acceleration[responding]-cmdaff[responding]/tau)
                    constrain((oldcmdE-oldE)[responding]/tau[:,None],
                              -limit.acceleration[responding]-(oldcmdaff-oldaff)[responding]/tau,
                              limit.acceleration[responding]-(oldcmdaff-oldaff)[responding]/tau)
                    command_q_affine += .5*h*(oldcmdaff[2:]+cmdaff[2:])
                    command_q_sensitivity += .5*h*(oldcmdE[2:]+cmdE[2:])
                    constrain(command_q_sensitivity,JOINT_LOWER+limit.position_margin-command_q_affine,
                              JOINT_UPPER-limit.position_margin-command_q_affine)
                # The integral of tau*u_dot+u=command is exact. Only planar
                # pose integration uses the same midpoint approximation as before.
                avg=limit.response_time/h*(old-nominal[k])+.5*(oldcmd+cmd)
                avgE=limit.response_time[:,None]/h*(oldE-select)+.5*(oldcmdE+cmdE)
                avgaff=limit.response_time/h*oldaff+.5*(oldcmdaff+cmdaff)
                xn,A,B=CoordinationPreview.step(x,avg,avg,h)
                affine=A@affine+xn-A@x+2*B@(avgaff-avg)
                sensitivity=A@sensitivity+2*B@avgE
                x = xn
                br = None if tilt is None else Rotation.from_euler('z', x[2]).as_matrix()@tilt
                p, r, J = whole_body_kinematics(x, z, br)
                C = np.zeros((6, 9)); C[:2, :2] = np.eye(2); C[:, 2:] = J[:, 1:]
                tp, tr, _ = self.trajectory.sample(times[k])
                frame=path_frame(projector,self.trajectory,p,times[k]) if geometric else None
                timed_position=tp.copy()
                if frame is not None:tp,tr,transform,tangent,progress=capture_frame(frame,times[k]-t)
                error = np.r_[tp-p, Rotation.from_matrix(tr@r.T).as_rotvec()]
                target = error-C@(affine-x)
                if frame is None:
                    constrain(C@sensitivity, target, target)
                else:
                    constrain(transform@C@sensitivity,transform@target,transform@target)
                    cost(tangent[None,:]@C[:3]@sensitivity,
                        [tangent@(timed_position-p-C[:3]@(affine-x))],.5*h)
                    progress_knots,progress_points=projector.progress_cache
                    remaining_points=np.vstack([frame[0],progress_points[progress_knots>progress]])
                    remaining_arc=float(np.linalg.norm(np.diff(remaining_points,axis=0),axis=1).sum())
                    constrain(tangent[None,:]@J[:3]@select,0.,2.*remaining_arc)
                constrain(sensitivity[3:], qlo-affine[3:], qhi-affine[3:])
                # Bounded configuration anchoring recovers compensation range
                # after a mover passes; it can never compete with camera rows.
                cost(sensitivity[2:], self.anchor[2:]-affine[2:], .25*h)
                # Convex piecewise upper envelopes of signed stopping distance.
                for j in range(6):
                    for sign in (-1., 1.):
                        edges = np.linspace(0., limit.velocity[j+2], 5)
                        for a, b in zip(edges[:-1], edges[1:]):
                            slope = (a+b)/(2*limit.acceleration[j+2])
                            intercept = -a*b/(2*limit.acceleration[j+2])
                            bound = qhi[j]-affine[j+3] if sign > 0 else affine[j+3]-qlo[j]
                            constrain(sign*sensitivity[j+3:j+4]+slope*sign*select[j+2:j+3], -np.inf, bound-intercept)
                    # Only replenish near-limit reserve; no endless homing/escape.
                    # Response delay plus stopping distance makes reserve
                    # depletion visible BEFORE the joint reaches a face.
                    for sign in (-1.,1.):
                        speed=max(0.,sign*nominal[k,j+2])
                        stop=.2*speed+speed**2/(2*limit.acceleration[j+2])
                        bound=(qhi[j]-.2) if sign > 0 else -(qlo[j]+.2)
                        if sign*x[j+3]+stop > bound:
                            derivative=(.2+speed/limit.acceleration[j+2]) if speed > 0 else 0.
                            M=sign*sensitivity[j+3:j+4]+derivative*sign*select[j+2:j+3]
                            target=bound-sign*affine[j+3]-stop+derivative*speed
                            cost(M,[target],8.*h)
                future = c.collision.at_time(times[k]-t)
                fc, fr, fi, d, G = witnesses(future, x, z, br)
                mask = d < .28
                if np.any(mask):
                    gradients = G[mask]
                    M = gradients@sensitivity
                    correction = gradients@(affine-x)
                    constrain(M, future.margin+.006-d[mask]-correction, np.inf)
                    # Use the same relative-motion acceleration barrier as the
                    # fast filter, rather than planning merely separated nodes
                    # that arrive with no braking authority. Recompute at every
                    # SQP knot; the nonholonomic S and normals are never frozen.
                    R = gradients@integration_matrix(x[2])
                    motion = future.primitive_motion(x, fc, z, br)[mask]
                    ee = .002
                    xp, xm = advance(x, nominal[k], ee), advance(x, nominal[k], -ee)
                    brp = None if br is None else Rotation.from_euler('z', nominal[k,1]*ee).as_matrix()@br
                    brm = None if br is None else Rotation.from_euler('z', -nominal[k,1]*ee).as_matrix()@br
                    dp = primitive_values(future.geometry,xp,fc+future.velocities[fi]*ee,z,brp)-fr[:,None]
                    dm = primitive_values(future.geometry,xm,fc-future.velocities[fi]*ee,z,brm)-fr[:,None]
                    curvature = ((dp-2*d+dm)/ee**2)[mask]
                    from .body_anticipation import barrier_gains
                    gains=barrier_gains(future,fi)[mask]
                    hocbf = R@((select-oldE)/h)+2*gains[:,None]*(R@select)+gains[:,None]**2*M
                    bound = (-curvature-2*gains*motion-gains**2*(d[mask]-future.margin-.006)
                             -gains**2*correction+R@oldaff/h)
                    constrain(hocbf, bound, np.inf)
                    reserve_scale=np.maximum(np.linalg.norm(R,axis=1),1e-5)
                    reserve_target=bound+.05*reserve_scale
                    fragile=hocbf@nominal.ravel() < reserve_target
                    if np.any(fragile):
                        # Saturating hinge: seek a finite common-control
                        # barrier reserve, then stop rewarding extra clearance.
                        cost(hocbf[fragile],reserve_target[fragile],.8*h)
                    active = d[mask] < future.margin+.04
                    if np.any(active):
                        cost(M[active], future.margin+.04-d[mask][active]-correction[active], 35.*h)
                    if d.min() < future.margin and first_conflict is None:
                        first_conflict = float(times[k])
                sd, sG = c.self_collision.terms(x)
                smask = sd < .08
                constrain(sG[smask]@sensitivity,
                          c.self_collision.margin+.002-sd[smask]-sG[smask]@(affine-x), np.inf)
                old, oldE, oldaff = nominal[k], select, np.zeros(8)
                oldcmd,oldcmdE,oldcmdaff=cmd,cmdE,cmdaff
            try:
                answer, residual = qp(H, g, np.vstack(rows), np.concatenate(lows), np.concatenate(highs), nominal.ravel())
            except RuntimeError as error:
                failure = str(error); break
            change = float(np.max(np.abs(answer-nominal.ravel())))
            nominal = answer.reshape(n, 8)
            accepted = True
            if change < 1e-4:
                break
        maxp = maxr = 0.; clearance = np.inf
        states = [state.copy()]
        if accepted:
            x, old = state.copy(), velocity.copy()
            projector=SimpleNamespace()
            oldcmd=c.previous.copy() if response_active and hasattr(c,'previous') else velocity.copy()
            accumulated_error = 0.
            # Nonlinear re-integration and between-knot collision validation.
            for k, u in enumerate(nominal):
                decay,beta,gain=response_coefficients(limit.response_time,h)
                cmd=(u-decay*old-beta*oldcmd)/gain
                subdivisions=3
                for sub in range(subdivisions):
                    a=response_velocity(old,oldcmd,cmd,limit.response_time,h,sub*h/subdivisions)
                    b=response_velocity(old,oldcmd,cmd,limit.response_time,h,(sub+1)*h/subdivisions)
                    br = None if tilt is None else Rotation.from_euler('z', x[2]).as_matrix()@tilt
                    future = c.collision.at_time((k+sub/subdivisions)*h)
                    proof = future.sweep(x, .5*(a+b), h/subdivisions, z, max_samples=9, base_rotation=br)
                    # Extra displacement bound for the endpoint velocity ramp.
                    ramp_error = h/subdivisions*(abs(b[0]-a[0])+np.abs(b[1:]-a[1:])@future.geometry.reach)/2
                    accumulated_error += ramp_error
                    clearance = min(clearance, float(proof['lower'].min())-accumulated_error)
                    if c.self_collision.sweep(x, a, b, h/subdivisions) < c.self_collision.margin:
                        accepted = False
                    x = CoordinationPreview.step(x, a, b, h/subdivisions)[0]
                    if c.self_collision.distances(x).min() < c.self_collision.margin:
                        accepted = False
                    br = None if tilt is None else Rotation.from_euler('z', x[2]).as_matrix()@tilt
                    p, r, _ = whole_body_kinematics(x, z, br)
                    tp, tr, _ = self.trajectory.sample(t+(k+(sub+1)/subdivisions)*h)
                    frame=path_frame(projector,self.trajectory,p,t+(k+(sub+1)/subdivisions)*h) if geometric else None
                    if frame is not None:tp,tr=capture_frame(frame,(k+(sub+1)/subdivisions)*h)[:2]
                    maxp = max(maxp, float(np.linalg.norm(p-tp)))
                    maxr = max(maxr, float(np.linalg.norm(Rotation.from_matrix(tr@r.T).as_rotvec())))
                states.append(x.copy()); old = u;oldcmd=cmd
            accepted = accepted and clearance >= c.collision.margin and maxp <= .003 and maxr <= .006
            if not accepted:
                failure = 'nonlinear prediction validation failed'
        if accepted:
            self.times = np.r_[t, times]
            self.controls = np.vstack([velocity, nominal]); self.states = np.array(states)
            first_conflict = None
        else:
            self.times = self.controls = self.states = None
        self.info = dict(tpr_prediction_valid=bool(accepted), tpr_prediction_failure=failure,
                         tpr_prediction_horizon_s=float(end-t), tpr_prediction_s=time.perf_counter()-started,
                         tpr_prediction_position_m=maxp, tpr_prediction_orientation_rad=maxr,
                         tpr_prediction_clearance_m=None if not np.isfinite(clearance) else clearance,
                         tpr_prediction_path_progress_free=bool(geometric),
                         tpr_prediction_capture_offset_m=float(np.linalg.norm(capture_position)),
                         tpr_prediction_capture_duration_s=capture_duration,
                         tpr_first_conflict_s=first_conflict,
                         tpr_prediction_scope='local measured constant-velocity obstacle model; no recursive guarantee')
        return np.zeros(8) if self.controls is None else self.sample(t+dt)


def compute_reconfiguration(preview, scene, physical_time, limits, state, velocity, previous_command, t, dt, z, rotation, commanded_q, body_aware=False):
    from types import SimpleNamespace
    from .dynamic_collision import DynamicCollision
    from .self_collision import SelfCollision
    model = SimpleNamespace(collision=DynamicCollision(scene, physical_time), limits=limits,previous=previous_command,
                            self_collision=SelfCollision(),body_aware=body_aware)
    preview.update(model, state, velocity, t, dt, z, rotation, commanded_q)
    # Return only the result, not the trajectory and its growing pose caches.
    return SimpleNamespace(**{name:getattr(preview,name) for name in
        ('updated','times','controls','states','info','anchor')})


class AsyncReconfigurationPreview(ReconfigurationPreview):
    """One worker and one in-flight request; the servo never waits for SQP."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = self.pending = None
        self.generation = 0
        self.pending_generation = -1
        self.requested = -np.inf
        self.accepted = self.rejected = 0
        self.motion_replans = 0
        self.pending_motion = self.accepted_motion = self.requested_motion = None

    def warm(self):
        """Start the numerical worker before entering the control loop."""
        if self.pool is None:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            from .shot_preview_worker import initialize_worker
            self.pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker)
            self.pool.submit(initialize_worker).result(timeout=30.)

    def reset(self, trajectory):
        if trajectory is not self.trajectory:
            self.generation += 1
            self.requested = -np.inf
        super().reset(trajectory)

    def accept(self, result, state, t):
        if self.pending_generation != self.generation or t-result.updated > .6:
            return False
        if result.controls is None or t > result.times[-1]:
            return False
        expected = np.array([np.interp(t, result.times, result.states[:, j]) for j in range(9)])
        difference = state-expected
        difference[2] = np.arctan2(np.sin(difference[2]), np.cos(difference[2]))
        return np.linalg.norm(difference[:2]) < .035 and np.max(np.abs(difference[2:])) < .12

    def update(self, c, state, velocity, t, dt, z, rotation, commanded_q):
        observed = motion_evidence(c.collision.scene)
        changed = motion_changed(self.requested_motion, observed)
        if self.controls is not None and motion_changed(self.accepted_motion, observed):
            self.controls = self.times = self.states = None
            self.info.update(tpr_prediction_valid=False,
                             tpr_prediction_failure='observed obstacle motion changed')
        if self.pending is not None and self.pending.done():
            pending, self.pending = self.pending, None
            try:
                result = pending.result()
                same_motion = not motion_changed(self.pending_motion, observed)
                if same_motion and self.accept(result, state, t):
                    for name in ('updated', 'times', 'controls', 'states', 'info', 'anchor'):
                        setattr(self, name, getattr(result, name))
                    self.accepted += 1
                    self.accepted_motion = self.pending_motion
                else:
                    self.rejected += 1
                    self.info.update(tpr_prediction_valid=False,
                                     tpr_prediction_failure=('observed obstacle motion changed' if not same_motion else
                                         result.info.get('tpr_prediction_failure') or 'stale or mismatched asynchronous prediction'))
            except Exception as error:
                self.rejected += 1
                self.info.update(tpr_prediction_valid=False, tpr_prediction_failure=str(error))
        if self.controls is not None and t-self.updated > .6:
            self.controls = self.times = self.states = None
            self.info.update(tpr_prediction_valid=False, tpr_prediction_failure='prediction expired')
        if self.pending is None and (changed or t-self.requested >= self.period-1e-8):
            if self.pool is None:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor
                from .shot_preview_worker import initialize_worker
                self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)
            snapshot = ReconfigurationPreview(self.horizon, self.nodes, self.period)
            snapshot.observed_motion = self.accepted_motion
            for name in ('trajectory', 'times', 'controls', 'anchor'):
                setattr(snapshot, name, getattr(self, name))
            self.pending = self.pool.submit(compute_reconfiguration, snapshot, c.collision.scene,
                c.physical_time, c.limits, state.copy(), velocity.copy(), c.previous.copy(), t, dt, z,
                None if rotation is None else np.asarray(rotation).copy(),
                None if commanded_q is None else np.asarray(commanded_q).copy(),
                getattr(c,'body_aware',False) or getattr(c,'body_adaptation_seen',False))
            self.requested, self.pending_generation = t, self.generation
            self.requested_motion = self.pending_motion = observed
            self.motion_replans += int(changed)
        self.info.update(tpr_prediction_async=True, tpr_prediction_pending=self.pending is not None,
                         tpr_prediction_accepted=self.accepted, tpr_prediction_rejected=self.rejected,
                         tpr_motion_replans=self.motion_replans,
                         tpr_prediction_age_s=None if self.controls is None else t-self.updated)
        return np.zeros(8) if self.controls is None else self.sample(t+dt)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


def solve_tpr(c, state, target_p, target_r, feedforward, dt, *, base_z=0., base_rotation=None,
              commanded_q=None, brake=False, trajectory_time=0., preparing=False, free_transfer_control=None, **unused):
    started = time.perf_counter()
    timings={};stage_started=started
    def mark(name):
        nonlocal stage_started
        now=time.perf_counter();timings[name]=now-stage_started;stage_started=now
    if not np.isfinite(dt) or not 0 < dt <= .1:
        raise ValueError('control dt must be in (0, 0.1]')
    state = np.asarray(state, float)
    if free_transfer_control is not None:
        free_transfer_control=np.asarray(free_transfer_control,float)
        if free_transfer_control.shape!=(8,) or not np.isfinite(free_transfer_control).all():
            raise ValueError('free transfer requires eight finite actuator preferences')
        if c.reconfiguration.trajectory is not None or c.limits.deadline_s is not None:
            raise ValueError('free transfer cannot override an installed camera shot or deadline')
    previous = c.previous.copy()
    measured = previous if c.measured_control is None else np.asarray(c.measured_control, float)
    if measured.shape != (8,) or not np.isfinite(measured).all():
        raise ValueError('measured velocity must contain eight finite values')
    previous_prediction=getattr(c,'last_predicted_actual_velocity',None)
    model_error=None if previous_prediction is None or c.measured_control is None else (measured-previous_prediction)
    c.time_s = trajectory_time
    p, r, J = whole_body_kinematics(state, base_z, base_rotation)
    received = c.reconfiguration.trajectory
    original=getattr(received,'original',received)
    from .combo import is_combo
    ordered=received is not None and is_combo(original)
    authored = received is not None and bool(getattr(original, 'task', {}).get('shot_phase'))
    if ordered and any(np.linalg.norm(row['velocity'])>.04 for row in c.collision.scene.get('tracks',[])):
        c.body_adaptation_seen=True
    combo=received is not None and (getattr(c,'body_aware',False) or (ordered and getattr(c,'body_adaptation_seen',False)))
    terminal_pose=getattr(c,'terminal_camera_pose',None)
    if terminal_pose is not None and received is not None and trajectory_time>=received.duration:
        # Once the measured frame is inside the unchanged terminal band,
        # confirm that frame without restarting the finished camera move.
        # Redundant body motion remains available under all physical checks.
        target_p,target_r=terminal_pose;feedforward=np.zeros(6);combo=False
    semantic_axis=None;feedback_progress=trajectory_time
    if combo and not brake and not preparing:
        from .body_anticipation import phase_axis,tracking_progress,local_forward_target
        original=getattr(received,'original',received)
        semantic_axis=phase_axis(original,original.phase(trajectory_time),p)
        feedback_progress=tracking_progress(c,received,p,trajectory_time)
        if semantic_axis is not None:
            feedback_progress=local_forward_target(c,received,feedback_progress)
        if abs(feedback_progress-trajectory_time)>1e-7:
            target_p,target_r,feedforward=received.sample(feedback_progress)
        if semantic_axis is not None:
            geometric_position,target_r,geometric_twist=received.sample(c.achieved_progress)
            from .reference import look_at
            # Preserve the authored subject-relative viewing offset at the
            # actually measured camera position during body avoidance.
            target_r=look_at(p,original.subject)@look_at(geometric_position,original.subject).T@target_r
            # Feed forward along the locally achieved tangent. A clock ahead
            # of a hairpin otherwise commands the opposite translation while
            # the path constraint requires forward progress.
            feedforward=np.asarray(geometric_twist)
    yield_info=dict(active=False)
    if authored and getattr(c,'observed_clock_enabled',False) and not brake and not preparing:
        from .observed_clock import update as observed_yield
        yield_info=observed_yield(c,received,state,measured,trajectory_time,base_z,base_rotation)
        if yield_info['active']:
            target_p,target_r,_=received.sample(yield_info['progress'])
            feedforward=received.sample(yield_info['progress'])[2]*yield_info['rate']
    error = np.r_[np.asarray(target_p)-p, Rotation.from_matrix(target_r@r.T).as_rotvec()]
    ff = np.asarray(feedforward, float)
    if ff.shape != (6,) or not np.isfinite(ff).all() or not np.isfinite(error).all():
        raise ValueError('invalid camera task')
    eps = .002
    plus, minus = advance(state, measured, eps), advance(state, measured, -eps)
    rp = None if base_rotation is None else Rotation.from_euler('z', measured[1]*eps).as_matrix()@base_rotation
    rm = None if base_rotation is None else Rotation.from_euler('z', -measured[1]*eps).as_matrix()@base_rotation
    zp = zm = base_z
    camera_velocity = J@measured
    base_twist = getattr(c, 'measured_base_twist', None)
    if base_twist is not None:
        base_twist = np.asarray(base_twist, float)
        if base_twist.shape != (6,) or not np.isfinite(base_twist).all():
            raise ValueError('measured world base twist must contain six finite values')
        nominal_linear = np.r_[measured[0]*np.cos(state[2]),measured[0]*np.sin(state[2]),0.]
        delta_angular = base_twist[3:]-np.array([0.,0.,measured[1]])
        camera_velocity += np.r_[base_twist[:3]-nominal_linear+np.cross(delta_angular,p-np.r_[state[:2],base_z]),delta_angular]
        plus[:2], minus[:2] = state[:2]+eps*base_twist[:2],state[:2]-eps*base_twist[:2]
        zp, zm = base_z+eps*base_twist[2],base_z-eps*base_twist[2]
        rotation = Rotation.from_euler('z',state[2]).as_matrix() if base_rotation is None else base_rotation
        rp = Rotation.from_rotvec(eps*base_twist[3:]).as_matrix()@rotation
        rm = Rotation.from_rotvec(-eps*base_twist[3:]).as_matrix()@rotation
    Jdot = (whole_body_kinematics(plus, zp, rp)[2]-whole_body_kinematics(minus, zm, rm)[2])/(2*eps)
    ff_acc = np.zeros(6)
    received = c.reconfiguration.trajectory
    if received is not None and not brake and not preparing:
        ta, tb = max(0., feedback_progress-eps), min(received.duration, feedback_progress+eps)
        if tb > ta:
            from .intent_paths import twists_at
            va,vb=twists_at(received,[ta,tb])
            ff_acc=(vb-va)/(tb-ta)
        if combo and semantic_axis is not None:
            ta=max(0.,c.achieved_progress-eps);tb=min(received.duration,c.achieved_progress+eps)
            if tb>ta:
                va,vb=twists_at(received,[ta,tb]);ff_acc=(vb-va)/(tb-ta)
                pa=received.sample(ta)[0];pb=received.sample(tb)[0]
                planned_gaze=Rotation.from_matrix(look_at(pb,original.subject)@look_at(pa,original.subject).T).as_rotvec()/(tb-ta)
                live_gaze=Rotation.from_matrix(
                    look_at(p+eps*camera_velocity[:3],original.subject)@
                    look_at(p-eps*camera_velocity[:3],original.subject).T).as_rotvec()/(2*eps)
                # The viewing direction follows the measured camera, including
                # its slower arrival after avoidance, rather than a stale
                # planned angular speed that leaves a terminal centroid bias.
                ff[3:]=live_gaze+geometric_twist[3:]-planned_gaze
                ff_acc[3:]=0.
    if yield_info['active']:ff_acc=np.zeros(6)
    feedback_latency=float(getattr(c,'feedback_latency_s',0.))
    frequency = feedback_frequency(float(getattr(c, 'tpr_servo_frequency', 12.)),dt,feedback_latency)
    desired_acc = ff_acc+2*frequency*(ff-camera_velocity)+frequency**2*error
    desired = J@measured+dt*(desired_acc-Jdot@measured)
    if semantic_axis is not None:
        desired[:3]-=semantic_axis*min(0.,float(semantic_axis@desired[:3]))
    # The actuator envelope is anchored to measured velocity, and ALSO to the
    # last command, so servo mismatch is never hidden by resetting history.
    A, lo, hi = c.bounds(state[3:], dt, commanded_q)
    tau=c.limits.response_time
    response_active=bool(np.any(tau > 0.))
    decay,beta,gain=response_coefficients(tau,dt,c.limits.command_ramp)
    response_offset=decay*measured+beta*previous
    command_scale=1/gain
    command_offset=-response_offset/gain
    if response_active:
        lo,hi=lo-A@command_offset,hi-A@command_offset
        A=A*command_scale[None,:]
        responding=tau > 0.
        initial_acceleration=(previous[responding]-measured[responding])/tau[responding]
        if np.any(np.abs(initial_acceleration) > c.limits.acceleration[responding]+1e-6):
            raise RuntimeError('identified actuator response is outside its acceleration envelope')
        endpoint_rows=np.diag((command_scale[responding]-1)/tau[responding])@np.eye(8)[responding]
        endpoint_offset=command_offset[responding]/tau[responding]
        A=np.vstack([A,endpoint_rows])
        lo=np.r_[lo,-c.limits.acceleration[responding]-endpoint_offset]
        hi=np.r_[hi,c.limits.acceleration[responding]-endpoint_offset]
    if c.measured_control is not None:
        c.previous = measured.copy()
        original_limits=c.limits
        try:
            c.limits=getattr(c,'measured_limits',None) or original_limits
            if response_active:
                from dataclasses import replace
                # The command must stop at the deadline. An identified servo
                # can still coast; report/certify that actual response instead
                # of declaring its noisy measured velocity exactly zero.
                c.limits=replace(c.limits,deadline_s=None)
            Am, lm, hm = c.bounds(state[3:], dt, commanded_q)
        finally:
            c.previous = previous
            c.limits=original_limits
        A, lo, hi = np.vstack([A, Am]), np.r_[lo, lm], np.r_[hi, hm]
    actuator_A, actuator_lo, actuator_hi = A.copy(), lo.copy(), hi.copy()
    centers, radii, indices, distances, gradients, motion = witnesses(c.collision, state, base_z, base_rotation,with_motion=True)
    rows = gradients@integration_matrix(state[2])
    rate = rows@measured+motion
    cp = centers+c.collision.velocities[indices]*eps
    cm = centers-c.collision.velocities[indices]*eps
    geometry=c.collision.geometry
    # Reuse the two FK chains just evaluated for Jdot and batch their SDFs.
    frame_pair=[geometry.frames(x,z,r)[0] for x,z,r in ((plus,zp,rp),(minus,zm,rm))]
    pp=np.array([[p for p,_ in frames] for frames in frame_pair])
    rr=np.array([[r for _,r in frames] for frames in frame_pair])
    dp,dm=geometry._primitive_arrays(pp,rr,np.array([cp,cm]))-radii[None,:,None]
    if base_twist is not None:
        rate = (dp-dm)/(2*eps)-c.collision.radius_rates[indices,None]
    curvature = (dp-2*distances+dm)/eps**2
    h = distances-c.collision.margin-.006
    # Acceleration HOCBF: hddot+2*k*hdot+k^2*h >= 0. All pairs
    # can activate; near-feature switches retain individual primitive rows.
    k = 2.
    # Moving crossing normals rotate; the static-wall exponential gain
    # demands immediate radial retreat far outside the collision envelope.
    # Use the existing moving-surface gain, retaining independent nonlinear
    # swept certificates, actuator bounds and the original separation margin.
    from .body_anticipation import barrier_gains
    gain=barrier_gains(c.collision,indices)
    lower_acc = -curvature-2*gain*rate-gain*gain*h
    active = (distances < .35) | (rate < -.1)
    R = rows[active]
    velocity_low = (-gain*h-motion)[active]
    low = np.maximum(R@measured+dt*lower_acc[active], velocity_low)
    sd, sG = c.self_collision.terms(state)
    sR = sG@integration_matrix(state[2])
    srate = sR@measured
    sd_plus,sd_minus=c.self_collision.distances_sequence([plus,minus])
    scurvature = (sd_plus-2*sd+sd_minus)/eps**2
    smask = sd < .08
    self_velocity_low = -k*(sd[smask]-c.self_collision.margin-.002)
    slow = np.maximum(sR[smask]@measured+dt*(-scurvature[smask]-2*k*srate[smask]-k*k*(sd[smask]-c.self_collision.margin-.002)), self_velocity_low)
    direct_low = np.r_[velocity_low,self_velocity_low]
    R, low = np.vstack([R, sR[smask]]), np.r_[low, slow]
    A, lo, hi = np.vstack([A, R]), np.r_[lo, low], np.r_[hi, np.full(len(R), np.inf)]
    preferred = measured.copy()
    predictive_command=measured.copy()
    mark('task_and_barriers')
    kinematic_preference_active=False
    kinematic_recovery_preference=False
    if not brake and not preparing:
        preferred = c.reconfiguration.update(c, state, measured, trajectory_time, dt, base_z, base_rotation, commanded_q)
        predictive_command=preferred.copy()
        if c.coordination_reference is not None:
            # Cross-phase allocation is only a subordinate configuration goal.
            preferred = c.coordination_reference
        witness=getattr(c,'kinematic_reference',None)
        if (witness and witness.get('feasible') and witness.get('trace') and
                witness['trace'][0]['t']<=trajectory_time<=witness['trace'][-1]['t']):
            # Installation authorizes the verified redundant preparation too.
            # The camera prefix stays unchanged until its exact activation;
            # delaying body preparation until then wastes the verified lead.
            trace=witness['trace'];knots=np.array([row['t'] for row in trace])
            states=np.array([row['state'] for row in trace]);controls=np.array([row['control'] for row in trace])
            # Redundancy belongs to the camera's achieved position on a bow.
            # A wall-clock sample across its turn can demand the opposite
            # configuration while the camera is still traversing that turn.
            witness_time=(getattr(c,'achieved_progress',feedback_progress) if getattr(c,'arrival_reanchored',False)
                else feedback_progress) if combo else trajectory_time
            expected=np.array([np.interp(witness_time,knots,states[:,j]) for j in range(9)])
            delta=expected-state;delta[2]=(delta[2]+np.pi)%(2*np.pi)-np.pi
            if np.linalg.norm(delta[:2])<=.25 and np.max(abs(delta[2:]))<=.5:
                preferred=np.array([np.interp(witness_time+dt,knots,controls[:,j]) for j in range(8)])
                correction=2.*np.r_[delta[:2]@np.array([np.cos(state[2]),np.sin(state[2])]),delta[2:]]
                preferred+=np.clip(correction,-.25*c.limits.velocity,.25*c.limits.velocity)
                kinematic_preference_active=True
            elif combo:
                # Outside the witness neighbourhood, its controls are not an
                # execution certificate. Its comfortable configuration may
                # still serve as a bounded recovery goal under fresh checks.
                displacement=delta[:2]
                heading=np.array([np.cos(state[2]),np.sin(state[2])])
                # Recover the verified pose branch, including heading. A
                # shortest turn toward position alone can select the opposite
                # arm configuration and prevent backing out of a completed bow.
                drive=np.clip(2.*displacement@heading,-.25,.25)
                turn=1.5*delta[2]
                preferred=np.r_[drive,np.clip(turn,-.8,.8),np.clip(.8*delta[3:],-.4,.4)]
                kinematic_recovery_preference=True
    elif brake:
        preferred = np.zeros(8)
    clock_guide_preference=yield_info.get('body_control_preference')
    if clock_guide_preference is not None and not brake and not preparing:
        preferred=np.asarray(clock_guide_preference)
    mark('prediction')
    W = np.array([1., 1., 1., .35, .35, .35])
    JW = W[:, None]*J
    # First try exact task equality; this is also an exact level-one solution.
    metric = np.array([1., .5, .2, .2, .2, .2, .2, .2])
    if kinematic_preference_active:
        # Track the validated redundant allocation within the exact task/safety
        # set. Excessive blending with the old velocity accumulated posture
        # drift and exhausted the very reserve the planner had checked.
        metric=np.full(8,10.)
    if c.coordination_reference is not None:
        metric[:2] = 200.
    H = np.diag(metric+.1)
    g = -metric*preferred-.1*measured
    from .body_anticipation import base_preference,path_frame
    escape_axis=semantic_axis
    if combo and semantic_axis is not None:
        frame=path_frame(c,received,p,trajectory_time)
        if frame is not None:escape_axis=frame[3]
    escape_key=(id(received),original.phase(trajectory_time)['name'] if combo else None)
    if getattr(c,'body_escape_key',None)!=escape_key:
        c.body_escape_key=escape_key;c.body_escape_memory={}
    body_preference,body_info=base_preference(c.collision,state,measured,base_z,base_rotation,escape_axis,c.body_escape_memory,
        camera_locked=free_transfer_control is None)
    from .combo import is_combo
    anticipatory=(rate<-.02)&(distances<.65)&(distances+1.5*np.minimum(rate,0.)<.12)
    # Keep at most two witnesses per articulated part; point density must not
    # multiply the importance of one observed obstacle.
    reserve_mask=np.zeros_like(anticipatory)
    for body in range(8):
        candidates=np.argwhere(anticipatory&(c.collision.geometry.shape_bodies[None,:]==body))
        if len(candidates):
            order=np.argsort([distances[i,j]+rate[i,j] for i,j in candidates])[:2]
            for i,j in candidates[order]:reserve_mask[i,j]=True
    reserve_R=rows[reserve_mask]
    reserve_target=np.maximum(0.,-motion[reserve_mask]-1.2*(distances[reserve_mask]-.10))
    body_priority=bool((body_preference is not None or len(reserve_R)) and not brake and not preparing and
                       combo and not yield_info['active'])
    body_info['whole_body_reserve_rows']=int(len(reserve_R))
    body_info['observed_yield']=yield_info
    manifold_rows=np.empty((0,8));manifold_target=np.empty(0)
    forward_limit=np.inf
    if combo and semantic_axis is not None:
        original=getattr(received,'original',received);phase=original.phase(trajectory_time)
        first=received.sample(phase['start_s'])[0];last=received.sample(phase['end_s'])[0]
        chord=last-first;length=np.linalg.norm(chord)
        if length>1e-6:
            tangent=chord/length
            transverse=np.eye(3)-np.outer(tangent,tangent)
            path_key=(id(received),phase['name'])
            if getattr(c,'path_shape_key',None)!=path_key:
                from .intent_paths import positions_at
                probes=positions_at(received,np.linspace(phase['start_s'],phase['end_s'],41))
                c.path_shape_straight=bool(np.max(np.linalg.norm((probes-first)@transverse,axis=1))<.003)
                c.path_shape_key=path_key
            if c.path_shape_straight:
                # For a straight authored action, use its geometric path as
                # a manifold while allowing forward progress to advance. This
                # preserves the shot instead of producing a reactive zigzag.
                _,_,basis=np.linalg.svd(tangent[None,:])
                normals=basis[1:]
                manifold_rows=normals@J[:3]
                transverse_acceleration=-8.*camera_velocity[:3]+16.*(first-p)
                manifold_target=normals@(J[:3]@measured+dt*(transverse_acceleration-Jdot[:3]@measured))
                forward_limit=np.sqrt(2*.5*max(0.,float((last-p)@tangent)))
            else:
                at=float(np.clip(getattr(c,'achieved_progress',trajectory_time),phase['start_s'],phase['end_s']))
                ta=max(phase['start_s'],at-.01);tb=min(phase['end_s'],at+.01)
                tangent=received.sample(tb)[0]-received.sample(ta)[0]
                if np.linalg.norm(tangent)>1e-8:
                    tangent/=np.linalg.norm(tangent)
                    _,_,basis=np.linalg.svd(tangent[None,:]);normals=basis[1:]
                    manifold_rows=normals@J[:3]
                    knots,points=c.progress_cache
                    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
                    valid=np.flatnonzero(lengths>1e-6)
                    curvature=np.zeros(3)
                    if len(valid)>2:
                        nearest=int(np.argmin(abs((knots[valid]+knots[valid+1])/2-at)))
                        left,right=valid[max(0,nearest-1)],valid[min(len(valid)-1,nearest+1)]
                        if right>left:
                            ta=(points[left+1]-points[left])/lengths[left]
                            tb=(points[right+1]-points[right])/lengths[right]
                            separation=.5*(lengths[left]+lengths[right])+lengths[left+1:right].sum()
                            curvature=(tb-ta)/max(separation,1e-6)
                    transverse_acceleration=(curvature*(camera_velocity[:3]@tangent)**2
                        -8.*camera_velocity[:3]+16.*(received.sample(at)[0]-p))
                    manifold_target=normals@(J[:3]@measured+dt*(transverse_acceleration-Jdot[:3]@measured))
                    remaining=np.vstack([received.sample(at)[0],points[knots>at]])
                    arc=float(np.linalg.norm(np.diff(remaining,axis=0),axis=1).sum())
                    forward_limit=np.sqrt(2*.5*arc)
            # Slow forward progress smoothly before exhausting the camera's
            # geometric interval; joint acceleration limits alone do not imply
            # a fixed camera stopping acceleration at every configuration.
            remaining_feature=max(0.,float((last-p)@semantic_axis))
            forward_limit=min(forward_limit,2.*remaining_feature)
    if body_preference is not None and not brake and not preparing:
        # Only the subordinate configuration objective changes: exact camera
        # equality, all body barriers and actuator limits still take priority.
        H[:2,:2]+=np.diag([60.,40.])
        g[:2]-=np.array([60.,40.])*body_preference
    geometry_preference_active = False
    guidance = getattr(c, 'geometry_guidance', None)
    if (guidance is not None and not brake and not preparing
            and guidance.request.start_s <= trajectory_time <= guidance.request.end_s
            and (guidance.request.pose_preferences or guidance.request.occupancy_preferences)):
        geometry_H, geometry_g = c.geometry_backend.quadratic_guidance(
            guidance,state,trajectory_time,base_z,base_rotation)
        H = H + geometry_H
        g = g + geometry_g
        geometry_preference_active = True
    mark('geometry_preferences')
    def hierarchy(bounds_lo):
        if free_transfer_control is not None:
            # A connector is a configuration goal, with no camera task. Reuse
            # every physical row and the mandatory nonlinear certificates.
            transfer_H=np.diag(np.r_[40.,20.,np.full(6,4.)] if getattr(c,'free_configuration_preview',False) else np.r_[4.,2.,np.ones(6)])
            transfer_g=-transfer_H@free_transfer_control
            if body_preference is not None and not getattr(c,'free_configuration_preview',False):
                transfer_H[:2,:2]+=np.diag([20.,10.])
                transfer_g[:2]-=np.array([20.,10.])*body_preference
            answer,residual=qp_with_reserve(transfer_H,transfer_g,A,bounds_lo,hi,previous,
                reserve_R,reserve_target,25./max(1,len(reserve_R)))
            return answer,residual,True
        if body_priority or (combo and len(manifold_rows) and not brake and not preparing):
            # Imminent chassis crossings need motion BEFORE the instantaneous
            # distance barrier activates. A camera equality must not consume
            # all of that acceleration reserve. The semantic direction remains
            # a constraint; camera pose tracking becomes a subordinate cost.
            axis=semantic_axis
            AA,ll,hh=A,bounds_lo,hi
            if axis is not None:
                AA=np.vstack([A,axis@J[:3]]);ll=np.r_[bounds_lo,0.];hh=np.r_[hi,forward_limit]
            if len(manifold_rows):
                AA=np.vstack([AA,manifold_rows]);ll=np.r_[ll,manifold_target];hh=np.r_[hh,manifold_target]
            task_weights=(np.array([1.,1.,1.,3.,3.,3.]) if body_priority else np.array([10.,10.,10.,30.,30.,30.]))
            task_J=task_weights[:,None]*J
            primary_H=task_J.T@task_J+.05*np.eye(8)
            primary_g=-task_J.T@(task_weights*desired)-.05*preferred
            posture_null=(np.eye(8) if body_priority else np.eye(8)-np.linalg.pinv(J,rcond=1e-6)@J)
            def posture_cost(metric,goal):
                nonlocal primary_H,primary_g
                projected=posture_null.T@metric@posture_null
                primary_H+=projected;primary_g-=projected@goal
            if clock_guide_preference is not None and not body_priority:
                posture_cost(5.*np.eye(8),preferred)
            elif kinematic_preference_active and not body_priority:
                posture_cost((10. if getattr(c,'arrival_reanchored',False) else 1.)*np.eye(8),preferred)
            elif kinematic_recovery_preference and not body_priority:
                recovery_weights=np.r_[5.,5.,np.ones(6)]
                posture_cost(np.diag(recovery_weights),preferred)
            body_info['configuration_recovery_goal_active']=kinematic_recovery_preference
            predictive_preference=bool(c.reconfiguration.info.get('tpr_prediction_valid') and
                c.reconfiguration.info.get('tpr_prediction_path_progress_free'))
            # Away from observed hazards, the complete current-state FK lift
            # is the better redundancy guide. A short path-progress preview
            # must not pull a freshly re-anchored arrival off that branch.
            if kinematic_preference_active and not body_priority and getattr(c,'arrival_reanchored',False):
                predictive_preference=False
            if yield_info['active'] and not c.reconfiguration.info.get('tpr_prediction_path_progress_free'):predictive_preference=False
            if predictive_preference:
                posture_cost(10.*np.eye(8),predictive_command)
            body_info['verified_path_preview_preference']=predictive_preference
            anchor_key=(id(getattr(received,'original',received)),original.phase(trajectory_time)['name'])
            if getattr(c,'posture_anchor_key',None)!=anchor_key:
                c.posture_anchor_key=anchor_key;c.posture_anchor=state[3:].copy()
            joint_margin=np.minimum(state[3:]-JOINT_LOWER,JOINT_UPPER-state[3:])
            # Recover redundant joint reserve before a crossing leaves the
            # camera unable to settle at the authored deadline. This is a
            # posture objective only; actuator/collision rows and the camera
            # manifold retain their existing bounds and priority.
            reserve_weights=12.*np.maximum(0.,(.30-joint_margin)/.30)**2
            restoration=np.clip(.8*(c.posture_anchor-state[3:]),-.4,.4)
            posture_cost(np.diag(np.r_[0.,0.,reserve_weights]),np.r_[0.,0.,restoration])
            if body_preference is not None and not predictive_preference:
                primary_H[:2,:2]+=np.diag([10.,5.])
                primary_g[:2]-=np.array([10.,5.])*body_preference
            reserve_rows=reserve_R if not predictive_preference else np.empty((0,8))
            def primary_solve(matrix,lower,upper):
                return qp_with_reserve(primary_H,primary_g,matrix,lower,upper,previous,
                    reserve_rows,reserve_target,25./max(1,len(reserve_rows)))
            try:
                answer,residual=primary_solve(AA,ll,hh)
            except RuntimeError:
                # If an exact path equality is momentarily outside the
                # acceleration envelope, retain its geometric objective.
                # Dropping it entirely caused a large, unnecessary detour.
                primary_H+=100.*manifold_rows.T@manifold_rows
                primary_g-=100.*manifold_rows.T@manifold_target
                body_info['path_manifold_feasible']=False
                try:
                    if axis is None:raise RuntimeError('no directed axis')
                    answer,residual=primary_solve(np.vstack([A,axis@J[:3]]),
                        np.r_[bounds_lo,0.],np.r_[hi,forward_limit])
                    body_info['direction_constraint_feasible']=True
                except RuntimeError:
                    answer,residual=primary_solve(A,bounds_lo,hi)
                    body_info['direction_constraint_feasible']=False
            return answer,residual,bool(np.max(np.abs(J@answer-desired))<=2e-6)
        try:
            answer, residual = qp(H, g, np.vstack([A, J]), np.r_[bounds_lo, desired], np.r_[hi, desired], previous)
            return answer, residual, True
        except RuntimeError:
            # No task slack penalty competes with posture: minimize task alone,
            # then freeze its unique six-dimensional output in level two.
            primary, _ = qp(JW.T@JW, -JW.T@(W*desired), A, bounds_lo, hi, previous)
            fixed = J@primary
            try:
                command, violation = qp(H, g, np.vstack([A, J]), np.r_[bounds_lo, fixed], np.r_[hi, fixed], primary)
            except RuntimeError:
                # Level one's already checked solution is safe and task-optimal.
                # Losing only a posture refinement must not become an escape mode.
                command = primary
                violation = max(0., float(np.max(bounds_lo-A@command)), float(np.max(A@command-hi)))
            return command, violation, np.max(np.abs(fixed-desired)) <= 2e-6
    barrier_mode = 'acceleration_and_velocity'
    try:
        command, violation, task_feasible = hierarchy(lo)
    except RuntimeError:
        # A changed map or a mandatory terminal deceleration can invalidate
        # the derivative assumption of the acceleration HOCBF even when the
        # actual separation constraint remains viable. Re-solve against the
        # direct relative-distance barrier, retaining EVERY actuator bound.
        # The nonlinear swept/self certificates below remain mandatory.
        low = direct_low
        lo = np.r_[actuator_lo,low]
        try:
            command, violation, task_feasible = hierarchy(lo)
        except RuntimeError:
            # A new measured velocity can invalidate a sufficient derivative
            # inequality without invalidating physical separation. Use the
            # available acceleration to recover its reserve, with a finite-step
            # distance constraint. This command is still rejected below unless
            # BOTH nonlinear swept certificates meet their unchanged margins.
            step_low = np.r_[(-h/dt-motion)[active], self_velocity_low]
            lo = np.r_[actuator_lo, step_low]
            deficient = R@measured < direct_low
            recovery_rows, recovery_target = R[deficient], direct_low[deficient]
            recovery_H = np.eye(8)+100.*recovery_rows.T@recovery_rows
            recovery_g = -100.*recovery_rows.T@recovery_target
            try:
                command, violation = qp(recovery_H, recovery_g, A, lo, hi, previous)
                task_feasible = False
                low = step_low
                barrier_mode = 'certified_swept_recovery'
            except RuntimeError as error:
                au,al=np.isfinite(actuator_hi),np.isfinite(actuator_lo)
                actuator=linprog(np.zeros(8),A_ub=np.vstack([actuator_A[au],-actuator_A[al]]),
                    b_ub=np.r_[actuator_hi[au],-actuator_lo[al]],bounds=[(None,None)]*8,method='highs')
                error.diagnostics=dict(kind='control_constraints_infeasible',time_s=trajectory_time,
                    actuator_bounds_feasible=bool(actuator.success),
                    current_observed_clearance_m=float(distances.min()),
                    joint_margin_rad=float(np.min(np.minimum(state[3:]-JOINT_LOWER,JOINT_UPPER-state[3:]))),
                    deadline_s=c.limits.deadline_s,active_collision_rows=len(R),
                    map_age_s=max(0.,c.physical_time-c.collision.stamp),
                    barriers_attempted=['acceleration_and_velocity','velocity','finite_step_with_nonlinear_sweep'])
                raise
        else:
            barrier_mode = 'certified_velocity_barrier'
    # Task-conditioned JOINT margin, a single LP over all active rows.
    rho = None
    actuator_task_feasible = None
    rho_method = 'linear_program_optimum'
    if len(R) and free_transfer_control is None:
        scale = np.maximum(np.linalg.norm(R, axis=1)*dt, 1e-5)
        if task_feasible:
            rho = task_margin_witness(command,J,desired,actuator_A,actuator_lo,actuator_hi,R,low,scale)
        if rho is not None:
            actuator_task_feasible = True
            rho_method = 'verified_feasible_point_lower_bound'
        else:
            upper_mask, lower_mask = np.isfinite(actuator_hi), np.isfinite(actuator_lo)
            Aub = np.vstack([np.c_[actuator_A[upper_mask], np.zeros(upper_mask.sum())],
                             np.c_[-actuator_A[lower_mask], np.zeros(lower_mask.sum())], np.c_[-R, scale]])
            bub = np.r_[actuator_hi[upper_mask], -actuator_lo[lower_mask], -low]
            result = linprog(np.r_[np.zeros(8), -1.], A_ub=Aub, b_ub=bub,
                             A_eq=np.c_[J, np.zeros(6)], b_eq=desired, bounds=[(None, None)]*9, method='highs')
            if result.success:
                rho = float(result.x[-1])
                actuator_task_feasible = True
            elif result.status == 2:
                actuator_task_feasible = False
    predicted_velocity=command.copy()
    if response_active:
        command=command_scale*predicted_velocity+command_offset
    mark('constrained_qp')
    proof = c.collision.sweep(state, predicted_velocity, dt, base_z, tolerance=.0005, max_samples=33, base_rotation=base_rotation)
    clearance = float(proof['lower'].min())
    if response_active:
        # A stable first-order response remains in the componentwise convex
        # hull of measured velocity and both command endpoints. This bounds
        # the whole curved response against the constant-velocity certificate.
        delta=np.maximum.reduce([np.abs(measured-predicted_velocity),np.abs(previous-predicted_velocity),np.abs(command-predicted_velocity)])
        clearance-=dt*(delta[0]+delta[1:]@c.collision.geometry.reach)
        clearance-=dt*dt*max(abs(previous[0]),abs(command[0]),abs(measured[0]))*delta[1]
    elif c.limits.command_ramp:
        clearance = certify_control_ramp(c.collision.scene, state, measured, command, c.physical_time,
            dt, base_z, steps=8, constant_certificate=clearance, base_rotation=base_rotation)
    if clearance < c.collision.margin:
        error=RuntimeError('TPR command fails swept clearance certificate; safe control not certified')
        error.diagnostics=dict(kind='swept_collision_certificate_failed',time_s=trajectory_time,
            current_observed_clearance_m=float(distances.min()),certified_lower_m=clearance,
            required_clearance_m=c.collision.margin,map_age_s=max(0.,c.physical_time-c.collision.stamp))
        raise error
    self_clearance = c.self_collision.sweep(state, measured, predicted_velocity, dt)
    if response_active:
        self_clearance-=2*dt*delta[2:]@c.self_collision.geometry.reach[1:]
    if self_clearance < c.self_collision.margin:
        raise RuntimeError('TPR command fails self-collision certificate')
    c.previous = command.copy()
    c.last_predicted_actual_velocity=predicted_velocity.copy()
    reserve = np.where(measured[2:] >= 0., JOINT_UPPER-state[3:], state[3:]-JOINT_LOWER)-measured[2:]**2/(2*c.limits.acceleration[2:])
    prediction = c.reconfiguration.info
    warning = barrier_mode != 'acceleration_and_velocity' or (rho is not None and rho < .05) or bool(prediction.get('tpr_prediction_failure')) or reserve.min() < .08
    status = 'normal' if task_feasible and not warning else 'warning' if task_feasible else 'degraded'
    involved = [dict(link=c.collision.geometry.shapes[j][1], obstacle_id=int(c.collision.track_ids[indices[i]]),
                     clearance_m=float(distances[i, j])) for i, j in np.argwhere(active & (distances < .15))]
    report = dict(status=status, camera_task_feasible=bool(task_feasible), rho=rho,
                  camera_actuator_feasible=actuator_task_feasible,
                  rho_definition='joint minimum normalized active-barrier residual',
                  rho_method=rho_method,
                  barrier_mode=barrier_mode,
                  kinematic_preference_active=kinematic_preference_active,
                  geometry_preference_active=geometry_preference_active,
                  trajectory_id=getattr(c,'received_trajectory_id','received'),
                  first_predicted_conflict_s=prediction.get('tpr_first_conflict_s'), pairs=involved,
                  joint_braking_reserve_rad=reserve.tolist(),
                  velocity_source='measured' if c.measured_control is not None else 'previous_command',
                  velocity_mismatch=float(np.max(np.abs(measured-previous))),
                  velocity_prediction_error=None if model_error is None else model_error.tolist(),
                  observation_age_s=max(0.,c.physical_time-c.collision.stamp),
                  actuator_response_time_s=tau.tolist(),
                  actual_acceleration_limits=(getattr(c,'measured_limits',None) or c.limits).acceleration.tolist(),
                  full_base_twist_measured=base_twist is not None,
                  self_clearance_lower_m=self_clearance,
                  scope='local model feasibility; agent owns task-space changes')
    if free_transfer_control is not None:
        status='free_transfer'
        report.update(status=status,camera_task_feasible=None,camera_actuator_feasible=None,
            rho_definition=None,scope='configuration transfer; camera task not requested; physical constraints and swept checks retained')
    mark('continuous_certificates')
    info = dict(position_error=float(np.linalg.norm(error[:3])), orientation_error=float(np.linalg.norm(error[3:])),
                feedback_latency_s=feedback_latency,servo_frequency_rad_s=frequency,
                free_camera_transfer=free_transfer_control is not None,
                body_anticipation=body_info,
                feedback_progress_s=feedback_progress,
                solver_stage_wall_s=timings,
                qp_violation=violation, solve_s=time.perf_counter()-started, command_position_violation=0.,
                command_diff_drive_violation=0., command_sweep_lower_m=clearance, clearance_m=float(distances.min()),
                dynamic_constraint_count=int(np.sum(active & (c.collision.track_ids[indices,None] > 0))),
                self_constraint_count=int(smask.sum()), safety_constraint_count=len(R), collision_mode='tpr',
                obstacle_avoidance_enabled=True, tpr_status=status, tpr_report=report,
                self_clearance_lower_m=self_clearance,
                predicted_actual_velocity=predicted_velocity.tolist(),
                camera_velocity_residual=(J@predicted_velocity-desired).tolist(),
                camera_acceleration_residual=((J@predicted_velocity-desired)/dt).tolist(),
                joint_braking_reserve_rad=reserve.tolist(), previous_actuator_command=previous.tolist(),
                braking_velocity_source=report['velocity_source'], **prediction)
    return command, info
