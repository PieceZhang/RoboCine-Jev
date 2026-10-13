"""Trajectory-preserving predictive reconfiguration (TPR-WBC).

Eight executable inputs, a six-dimensional camera task, and no escape heading.
The fast hierarchy is expressed in endpoint velocities (a=(u-u_measured)/dt).
Safety and actuator rows are hard; the second QP freezes the first QP's camera
output. Prediction is a local SQP, not a recursive-feasibility guarantee.
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


def small_qp(H, g, A, lo, hi):
    """Exact active-set enumeration for the usual two self-motion inputs."""
    C = np.vstack([A[np.isfinite(hi)], -A[np.isfinite(lo)]])
    b = np.r_[hi[np.isfinite(hi)], -lo[np.isfinite(lo)]]
    inv = np.linalg.inv(H)
    free = -inv@g
    if not len(b) or np.all(C@free <= b+1e-9):
        return free
    if len(g) == 1:
        low = np.max(b[C[:, 0] < 0]/C[C[:, 0] < 0, 0], initial=-np.inf)
        high = np.min(b[C[:, 0] > 0]/C[C[:, 0] > 0, 0], initial=np.inf)
        if low > high+1e-8:
            raise RuntimeError('TPR task-conditioned safety set empty')
        return np.array([np.clip(free[0], low, high)])
    normal = C@inv
    projected = free-(C@free-b)[:, None]*normal/np.sum(normal*C, axis=1)[:, None]
    i, j = np.triu_indices(len(b), 1)
    determinant = C[i, 0]*C[j, 1]-C[i, 1]*C[j, 0]
    keep = np.abs(determinant) > 1e-10
    i, j, determinant = i[keep], j[keep], determinant[keep]
    intersections = np.c_[(b[i]*C[j, 1]-C[i, 1]*b[j])/determinant,
                           (C[i, 0]*b[j]-b[i]*C[j, 0])/determinant]
    candidates = np.vstack([projected, intersections])
    feasible = np.all(C@candidates.T <= b[:, None]+1e-8, axis=0)
    if not np.any(feasible):
        raise RuntimeError('TPR task-conditioned safety set empty')
    candidates = candidates[feasible]
    values = .5*np.einsum('ni,ij,nj->n', candidates, H, candidates)+candidates@g
    return candidates[np.argmin(values)]


def qp(H, g, A, lo, hi, warm=None):
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
    if (not converged or residual > 1e-6) and len(g) <= 16:
        # Older bundled OSQP versions can stall on nearly parallel faces.
        # A dense active-set refinement solves the IDENTICAL convex problem.
        seed = result.x if valid else np.zeros(len(g)) if warm is None else Z.T@(warm-offset)
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


def witnesses(collision, state, z=0., rotation=None):
    centers, radii = collision.active_obstacles(state, z, rotation, per_body=None)
    # Incoming movers must not be lost by a static nearest-neighbour selection.
    indices = np.unique(np.r_[collision.tree.query(centers, k=1)[1],
                             np.flatnonzero(collision.track_ids > 0)])
    centers, radii = collision.centers[indices], collision.radii[indices]
    d, G = collision.geometry.primitive_distances(state, centers, z, rotation)
    return centers, radii, indices, d-radii[:, None], G


def primitive_values(geometry,state,centers,z=0.,rotation=None):
    """Distance-only curvature probes, avoiding unused point Jacobians."""
    frames=geometry.frames(state,z,rotation)[0]
    values=[]
    for body,_,kind,size,offset,local_rotation in geometry.shapes:
        p,r=frames[body]
        values.append(primitive_sdf((centers-(p+r@offset))@(r@local_rotation),kind,size))
    return np.array(values).T


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
            self.info = dict(tpr_prediction_valid=False)

    def sample(self, t):
        return np.array([np.interp(t, self.times, self.controls[:, j]) for j in range(8)])

    def update(self, c, state, velocity, t, dt, z, rotation, commanded_q):
        if self.trajectory is None or t >= self.trajectory.duration-1e-9:
            return np.zeros(8)
        if t-self.updated < self.period-1e-8:
            return np.zeros(8) if self.controls is None else self.sample(t+dt)
        started = time.perf_counter()
        self.updated = t
        n = self.nodes
        end = min(t+self.horizon, self.trajectory.duration)
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
        for iteration in range(4):
            H, g = np.eye(nv)*1e-5, np.zeros(nv)
            rows, lows, highs = [], [], []
            x = state.copy(); affine = x.copy(); sensitivity = np.zeros((9, nv))
            old, oldE, oldaff = velocity, np.zeros((8, nv)), velocity
            oldcmd=c.previous.copy() if response_active and hasattr(c,'previous') else velocity.copy()
            oldcmdE,oldcmdaff=np.zeros((8,nv)),oldcmd.copy()
            command_q_affine=state[3:].copy() if commanded_q is None else np.asarray(commanded_q).copy()
            command_q_sensitivity=np.zeros((6,nv))

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
                error = np.r_[tp-p, Rotation.from_matrix(tr@r.T).as_rotvec()]
                target = error-C@(affine-x)
                constrain(C@sensitivity, target, target)
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
                    hocbf = R@((select-oldE)/h+4*select)+4*M
                    bound = (-curvature-4*motion-4*(d[mask]-future.margin-.006)
                             -4*correction+R@oldaff/h)
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
                         tpr_first_conflict_s=first_conflict,
                         tpr_prediction_scope='local measured constant-velocity obstacle model; no recursive guarantee')
        return np.zeros(8) if self.controls is None else self.sample(t+dt)


def compute_reconfiguration(preview, scene, physical_time, limits, state, velocity, previous_command, t, dt, z, rotation, commanded_q):
    from types import SimpleNamespace
    from .dynamic_collision import DynamicCollision
    from .self_collision import SelfCollision
    model = SimpleNamespace(collision=DynamicCollision(scene, physical_time), limits=limits,previous=previous_command,
                            self_collision=SelfCollision())
    preview.update(model, state, velocity, t, dt, z, rotation, commanded_q)
    return preview


class AsyncReconfigurationPreview(ReconfigurationPreview):
    """One worker and one in-flight request; the servo never waits for SQP."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = self.pending = None
        self.generation = 0
        self.pending_generation = -1
        self.requested = -np.inf
        self.accepted = self.rejected = 0

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
        if self.pending is not None and self.pending.done():
            pending, self.pending = self.pending, None
            try:
                result = pending.result()
                if self.accept(result, state, t):
                    for name in ('updated', 'times', 'controls', 'states', 'info', 'anchor'):
                        setattr(self, name, getattr(result, name))
                    self.accepted += 1
                else:
                    self.rejected += 1
                    self.info.update(tpr_prediction_valid=False,
                                     tpr_prediction_failure=result.info.get('tpr_prediction_failure') or 'stale or mismatched asynchronous prediction')
            except Exception as error:
                self.rejected += 1
                self.info.update(tpr_prediction_valid=False, tpr_prediction_failure=str(error))
        if self.controls is not None and t-self.updated > .6:
            self.controls = self.times = self.states = None
            self.info.update(tpr_prediction_valid=False, tpr_prediction_failure='prediction expired')
        if self.pending is None and t-self.requested >= self.period-1e-8:
            if self.pool is None:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor
                from .shot_preview_worker import initialize_worker
                self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)
            snapshot = ReconfigurationPreview(self.horizon, self.nodes, self.period)
            for name in ('trajectory', 'times', 'controls', 'anchor'):
                setattr(snapshot, name, getattr(self, name))
            self.pending = self.pool.submit(compute_reconfiguration, snapshot, c.collision.scene,
                c.physical_time, c.limits, state.copy(), velocity.copy(), c.previous.copy(), t, dt, z,
                None if rotation is None else np.asarray(rotation).copy(),
                None if commanded_q is None else np.asarray(commanded_q).copy())
            self.requested, self.pending_generation = t, self.generation
        self.info.update(tpr_prediction_async=True, tpr_prediction_pending=self.pending is not None,
                         tpr_prediction_accepted=self.accepted, tpr_prediction_rejected=self.rejected,
                         tpr_prediction_age_s=None if self.controls is None else t-self.updated)
        return np.zeros(8) if self.controls is None else self.sample(t+dt)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


def solve_tpr(c, state, target_p, target_r, feedforward, dt, *, base_z=0., base_rotation=None,
              commanded_q=None, brake=False, trajectory_time=0., preparing=False, **unused):
    started = time.perf_counter()
    if not np.isfinite(dt) or not 0 < dt <= .1:
        raise ValueError('control dt must be in (0, 0.1]')
    state = np.asarray(state, float)
    previous = c.previous.copy()
    measured = previous if c.measured_control is None else np.asarray(c.measured_control, float)
    if measured.shape != (8,) or not np.isfinite(measured).all():
        raise ValueError('measured velocity must contain eight finite values')
    previous_prediction=getattr(c,'last_predicted_actual_velocity',None)
    model_error=None if previous_prediction is None or c.measured_control is None else (measured-previous_prediction)
    c.time_s = trajectory_time
    p, r, J = whole_body_kinematics(state, base_z, base_rotation)
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
        ta, tb = max(0., trajectory_time-eps), min(received.duration, trajectory_time+eps)
        if tb > ta:
            ff_acc = (received.sample(tb)[2]-received.sample(ta)[2])/(tb-ta)
    frequency = min(float(getattr(c, 'tpr_servo_frequency', 12.)), .6/dt)
    desired_acc = ff_acc+2*frequency*(ff-camera_velocity)+frequency**2*error
    desired = J@measured+dt*(desired_acc-Jdot@measured)
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
    centers, radii, indices, distances, gradients = witnesses(c.collision, state, base_z, base_rotation)
    rows = gradients@integration_matrix(state[2])
    motion = c.collision.primitive_motion(state, centers, base_z, base_rotation)
    rate = rows@measured+motion
    cp = centers+c.collision.velocities[indices]*eps
    cm = centers-c.collision.velocities[indices]*eps
    dp = primitive_values(c.collision.geometry,plus,cp,zp,rp)-radii[:, None]
    dm = primitive_values(c.collision.geometry,minus,cm,zm,rm)-radii[:, None]
    if base_twist is not None:
        rate = (dp-dm)/(2*eps)-c.collision.radius_rates[indices,None]
    curvature = (dp-2*distances+dm)/eps**2
    h = distances-c.collision.margin-.006
    # Acceleration HOCBF: hddot+2*k*hdot+k^2*h >= 0. All pairs
    # can activate; near-feature switches retain individual primitive rows.
    k = 2.
    lower_acc = -curvature-2*k*rate-k*k*h
    active = (distances < .35) | (rate < -.1)
    R = rows[active]; low = R@measured+dt*lower_acc[active]
    sd, sG = c.self_collision.terms(state)
    sR = sG@integration_matrix(state[2])
    srate = sR@measured
    scurvature = (c.self_collision.distances(plus)-2*sd+c.self_collision.distances(minus))/eps**2
    smask = sd < .08
    slow = sR[smask]@measured+dt*(-scurvature[smask]-2*k*srate[smask]-k*k*(sd[smask]-c.self_collision.margin-.002))
    R, low = np.vstack([R, sR[smask]]), np.r_[low, slow]
    A, lo, hi = np.vstack([A, R]), np.r_[lo, low], np.r_[hi, np.full(len(R), np.inf)]
    preferred = measured.copy()
    if not brake and not preparing:
        preferred = c.reconfiguration.update(c, state, measured, trajectory_time, dt, base_z, base_rotation, commanded_q)
        if c.coordination_reference is not None:
            # Cross-phase allocation is only a subordinate configuration goal.
            preferred = c.coordination_reference
    elif brake:
        preferred = np.zeros(8)
    W = np.array([1., 1., 1., .35, .35, .35])
    JW = W[:, None]*J
    # First try exact task equality; this is also an exact level-one solution.
    metric = np.array([1., .5, .2, .2, .2, .2, .2, .2])
    if c.coordination_reference is not None:
        metric[:2] = 200.
    H = np.diag(metric+.1)
    g = -metric*preferred-.1*measured
    try:
        command, violation = qp(H, g, np.vstack([A, J]), np.r_[lo, desired], np.r_[hi, desired], previous)
        task_feasible = True
    except RuntimeError:
        # No task slack penalty competes with posture: minimize task alone,
        # then freeze its unique six-dimensional output in level two.
        primary, _ = qp(JW.T@JW, -JW.T@(W*desired), A, lo, hi, previous)
        fixed = J@primary
        try:
            command, violation = qp(H, g, np.vstack([A, J]), np.r_[lo, fixed], np.r_[hi, fixed], primary)
        except RuntimeError:
            # Level one's already checked solution is safe and task-optimal.
            # Losing only a posture refinement must not become an escape mode.
            command = primary
            violation = max(0., float(np.max(lo-A@command)), float(np.max(A@command-hi)))
        task_feasible = np.max(np.abs(fixed-desired)) <= 2e-6
    # Task-conditioned JOINT margin, a single LP over all active rows.
    rho = None
    actuator_task_feasible = None
    if len(R):
        scale = np.maximum(np.linalg.norm(R, axis=1)*dt, 1e-5)
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
        raise RuntimeError('TPR command fails swept clearance certificate; safe control not certified')
    self_clearance = c.self_collision.sweep(state, measured, predicted_velocity, dt)
    if response_active:
        self_clearance-=2*dt*delta[2:]@c.self_collision.geometry.reach[1:]
    if self_clearance < c.self_collision.margin:
        raise RuntimeError('TPR command fails self-collision certificate')
    c.previous = command.copy()
    c.last_predicted_actual_velocity=predicted_velocity.copy()
    reserve = np.where(measured[2:] >= 0., JOINT_UPPER-state[3:], state[3:]-JOINT_LOWER)-measured[2:]**2/(2*c.limits.acceleration[2:])
    prediction = c.reconfiguration.info
    warning = (rho is not None and rho < .05) or bool(prediction.get('tpr_prediction_failure')) or reserve.min() < .08
    status = 'normal' if task_feasible and not warning else 'warning' if task_feasible else 'degraded'
    involved = [dict(link=c.collision.geometry.shapes[j][1], obstacle_id=int(c.collision.track_ids[indices[i]]),
                     clearance_m=float(distances[i, j])) for i, j in np.argwhere(active & (distances < .15))]
    report = dict(status=status, camera_task_feasible=bool(task_feasible), rho=rho,
                  camera_actuator_feasible=actuator_task_feasible,
                  rho_definition='joint minimum normalized acceleration-barrier residual',
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
    info = dict(position_error=float(np.linalg.norm(error[:3])), orientation_error=float(np.linalg.norm(error[3:])),
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
