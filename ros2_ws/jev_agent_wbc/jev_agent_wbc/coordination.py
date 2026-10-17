"""Acceleration-aware allocation for continuous translation-to-rotation shots.

This is a soft kinematic preview, never a collision or task-feasibility certificate.
The measured state and integrated motor target are both constrained. The fast
execution QP retains its original actuator bounds and swept collision gate.
"""
import time
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from .whole_body_qp import whole_body_kinematics
from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER


class CoordinationPreview:
    def __init__(self):
        self.trajectory = None
        self.enabled = False
        self.updated = -np.inf
        self.times = None
        self.controls = None
        self.states = None
        self.info = {}

    def set_trajectory(self, reference):
        if reference is self.trajectory:
            return
        self.trajectory = reference
        # Use geometry, not a Pan label, to recognize a stationary-position
        # orientation task. A target-locked translating arc is not stationary.
        self.enabled = False
        for shot in reference.shots:
            poses = [reference.sample(s) for s in np.linspace(shot['start_s'], shot['end_s'], 9)]
            fixed = np.max(np.ptp([p[0] for p in poses], axis=0)) < 1e-4
            turning = np.linalg.norm(Rotation.from_matrix(poses[-1][1]@poses[0][1].T).as_rotvec()) > .02
            self.enabled |= fixed and turning
        self.updated = -np.inf
        self.controls = self.times = self.states = None
        self.info = {}

    @staticmethod
    def step(x, old, new, h):
        """Midpoint integration of an endpoint-velocity ramp, with derivatives."""
        average = .5*(old+new)
        yaw = x[2]+.5*h*average[1]
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        tangent = np.array([-np.sin(yaw), np.cos(yaw)])
        result = x.copy()
        result[:2] += h*average[0]*direction
        result[2:] += h*average[1:]
        A = np.eye(9)
        A[:2, 2] = h*average[0]*tangent
        B = np.zeros((9, 8))
        B[:2, 0] = .5*h*direction
        B[:2, 1] = .25*h*h*average[0]*tangent
        B[2:, 1:] = .5*h*np.eye(7)
        return result, A, B

    def update(self, state, previous, commanded_q, t, dt, base_z, limits, base_rotation=None, measured_control=None):
        if not self.enabled or t >= self.trajectory.duration-1e-9:
            return None
        if t-self.updated < .4-1e-8 and self.controls is not None:
            return self.sample(t+dt)
        start = time.perf_counter()
        measured = previous if measured_control is None else np.asarray(measured_control, float)
        if measured.shape != (8,) or not np.isfinite(measured).all():
            raise ValueError('measured control must contain eight finite velocities')
        end = min(self.trajectory.duration, t+8.)
        n = min(20, max(2, int(np.ceil((end-t)/.4))))
        h = (end-t)/n
        times = t+np.arange(1, n+1)*h
        if self.controls is None:
            nominal = np.tile(previous, (n, 1))
        else:
            nominal = np.array([self.sample(s) for s in times])
        nv = 8*n
        selections = [np.pad(np.eye(8), ((0, 0), (8*k, nv-8*(k+1)))) for k in range(n)]
        tilt = (None if base_rotation is None else
                Rotation.from_euler('z', -state[2]).as_matrix()@base_rotation)
        failure = ''
        accepted = None
        # Successive linearization about a feasible velocity warm start; no IK
        # branch selection and no independent base/arm command clipping.
        for iteration in range(6 if self.controls is None else 3):
            H = np.eye(nv)*1e-6
            g = np.zeros(nv)
            rows, lows, highs = [], [], []
            affine = xbar = state.copy()
            sensitivity = np.zeros((9, nv))
            old = measured.copy()
            old_select = np.zeros((8, nv))
            old_affine = measured.copy()

            def cost(M, target, weight):
                nonlocal H, g
                w = np.broadcast_to(weight, (M.shape[0],))
                H += M.T@(w[:, None]*M)
                g -= M.T@(w*target)

            def constrain(M, lo, hi):
                rows.append(M)
                lows.append(np.broadcast_to(lo, (M.shape[0],)))
                highs.append(np.broadcast_to(hi, (M.shape[0],)))

            for k, select in enumerate(selections):
                difference = select-old_select
                constrain(select, -limits.velocity, limits.velocity)
                constrain(limits.wheels@select, -limits.wheel_speed, limits.wheel_speed)
                constraint_previous = previous if k == 0 else np.zeros(8)
                constrain(difference, constraint_previous-limits.acceleration*h, constraint_previous+limits.acceleration*h)
                constrain(limits.wheels@difference,
                          limits.wheels@constraint_previous-limits.wheel_acceleration*h,
                          limits.wheels@constraint_previous+limits.wheel_acceleration*h)
                cost(select, np.zeros(8), h*np.array([2., 20., .1, .1, .1, .1, .1, .1]))
                cost(difference, constraint_previous, .05/h)
                xnext, A, B = self.step(xbar, old, nominal[k], h)
                affine = A@affine+B@old_affine+xnext-A@xbar-B@(old+nominal[k])
                sensitivity = A@sensitivity+B@(old_select+select)
                xbar = xnext
                rotation = None if tilt is None else Rotation.from_euler('z', xbar[2]).as_matrix()@tilt
                p, r, J = whole_body_kinematics(xbar, base_z, rotation)
                C = np.zeros((6, 9)); C[:2, :2] = np.eye(2); C[:, 2:] = J[:, 1:]
                tp, tr, _ = self.trajectory.sample(times[k])
                error = np.r_[tp-p, Rotation.from_matrix(tr@r.T).as_rotvec()]
                cost(C@sensitivity, error-C@(affine-xbar), h*np.array([100000.]*3+[30000.]*3))
                # A reserve throughout the next task, rather than a neutral
                # posture goal at the current camera pose. Hinge costs only
                # activate near a physical face and include BOTH servo paths.
                offset = commanded_q-state[3:]+.5*h*(previous[2:]-measured[2:])
                joint_lower = np.maximum(JOINT_LOWER, JOINT_LOWER+limits.position_margin-offset)
                joint_upper = np.minimum(JOINT_UPPER, JOINT_UPPER-limits.position_margin-offset)
                qaff = affine[3:]
                constrain(sensitivity[3:], joint_lower-qaff, joint_upper-qaff)
                for j in range(6):
                    qnom = xbar[j+3]
                    safe_low, safe_high = JOINT_LOWER[j]+.2, JOINT_UPPER[j]-.2
                    if qnom < safe_low or qnom > safe_high:
                        target = np.clip(qnom, safe_low, safe_high)
                        cost(sensitivity[3+j:4+j], np.array([target-qaff[j]]), 80.*h)
                    # Secant upper envelopes for signed stopping distance;
                    # include terminal endpoint speed, not just position.
                    acc, vmax = limits.acceleration[j+2], limits.velocity[j+2]
                    edges = np.linspace(0., vmax, 5)
                    for sign in (-1., 1.):
                        for a, b in zip(edges[:-1], edges[1:]):
                            slope, intercept = (a+b)/(2*acc), -a*b/(2*acc)
                            M = sign*sensitivity[3+j:4+j]+slope*sign*select[j+2:j+3]
                            bound = joint_upper[j]-qaff[j] if sign > 0 else qaff[j]-joint_lower[j]
                            constrain(M, -np.inf, bound-intercept)
                remaining = max(0., self.trajectory.duration-times[k])
                constrain(select, -limits.acceleration*remaining, limits.acceleration*remaining)
                constrain(limits.wheels@select, -limits.wheel_acceleration*remaining, limits.wheel_acceleration*remaining)
                old, old_select, old_affine = nominal[k], select, np.zeros(8)
            M = np.vstack(rows); lo, hi = np.concatenate(lows), np.concatenate(highs)
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(np.triu(H)), q=g, A=sparse.csc_matrix(M), l=lo, u=hi,
                         verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=16000, polish=True)
            solver.warm_start(x=nominal.ravel())
            result = solver.solve()
            if result.x is None or result.info.status_val not in (1, 2, osqp.constant('OSQP_MAX_ITER_REACHED')) or not np.isfinite(result.x).all():
                failure = result.info.status
                break
            residual = float(max(0., np.max(lo-M@result.x), np.max(M@result.x-hi)))
            stationarity = float(np.max(np.abs(H@result.x+g+M.T@result.y))/max(1., np.max(np.abs(g)), np.max(np.abs(H@result.x))))
            if residual > 1e-6 or stationarity > 1e-5:
                failure = f'{result.info.status}: residual={residual}, stationarity={stationarity}'
                break
            nominal = result.x.reshape(n, 8)
            accepted = nominal.copy()
            accepted_residual, accepted_stationarity = residual, stationarity
        self.updated = t
        self.info = dict(coordination_active=True, coordination_horizon_end_s=float(end),
                         coordination_solve_s=time.perf_counter()-start,
                         coordination_iterations=iteration+1, coordination_failure=failure,
                         coordination_certificate=False,
                         coordination_velocity_source='measured' if measured_control is not None else 'previous_command',
                         coordination_velocity_mismatch=float(np.max(np.abs(measured-previous))))
        if accepted is None:
            # Failed/stale previews never become a command or safety claim.
            self.controls = self.times = None
            return None
        self.times = np.r_[t, times]
        self.controls = np.vstack([previous, accepted])
        self.info['coordination_constraint_residual'] = accepted_residual
        self.info['coordination_stationarity'] = accepted_stationarity
        predicted = state.copy(); old = measured.copy(); minimum = np.inf
        max_position = max_orientation = 0.
        states = [state.copy()]
        for s, u in zip(times, accepted):
            predicted = self.step(predicted, old, u, h)[0]
            states.append(predicted.copy())
            minimum = min(minimum, float(np.min(np.minimum(predicted[3:]-JOINT_LOWER, JOINT_UPPER-predicted[3:]))))
            rotation = None if tilt is None else Rotation.from_euler('z', predicted[2]).as_matrix()@tilt
            pp, rr, _ = whole_body_kinematics(predicted, base_z, rotation)
            tp, tr, _ = self.trajectory.sample(s)
            max_position = max(max_position, float(np.linalg.norm(tp-pp)))
            max_orientation = max(max_orientation, float(np.linalg.norm(Rotation.from_matrix(tr@rr.T).as_rotvec())))
            old = u
        self.states = np.asarray(states)
        self.info.update(coordination_predicted_joint_margin_rad=minimum,
                         coordination_predicted_position_max_m=max_position,
                         coordination_predicted_orientation_max_rad=max_orientation)
        return self.sample(t+dt)

    def sample(self, t):
        return np.array([np.interp(t, self.times, self.controls[:, j]) for j in range(8)])


def compute_coordination(preview, *args, **kwargs):
    preview.update(*args, **kwargs)
    return preview


class AsyncCoordinationPreview(CoordinationPreview):
    """Live ROS preview: one spawned worker, bounded age and state acceptance.

    Lockstep CPU/PhysX harnesses intentionally use the synchronous class above
    to compare numerical behavior without scheduler-dependent plan arrivals.
    """
    def __init__(self):
        super().__init__()
        self.pool = self.pending = None
        self.generation = 0
        self.pending_generation = 0
        self.pending_t = -np.inf
        self.requested = -np.inf
        self.accepted = self.rejected = 0

    def warm(self):
        if self.enabled and self.pool is None:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            from .shot_preview_worker import initialize_worker
            self.pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker)
            self.pool.submit(initialize_worker).result(timeout=30.)

    def set_trajectory(self, reference):
        if reference is not self.trajectory:
            self.generation += 1
            self.requested = -np.inf
        super().set_trajectory(reference)

    def accept(self, result, state, t):
        if self.pending_generation != self.generation or t-self.pending_t > .6:
            return False
        if result.controls is None or t > result.times[-1]:
            return False
        predicted = np.array([np.interp(t, result.times, result.states[:, j]) for j in range(9)])
        difference = state-predicted
        difference[2] = np.arctan2(np.sin(difference[2]), np.cos(difference[2]))
        return (np.linalg.norm(difference[:2]) <= .08 and abs(difference[2]) <= .2
                and np.max(np.abs(difference[3:])) <= .25)

    def update(self, state, previous, commanded_q, t, dt, base_z, limits, base_rotation=None, measured_control=None):
        if not self.enabled or t >= self.trajectory.duration-1e-9:
            return None
        if self.pending is not None and self.pending.done():
            finished, self.pending = self.pending, None
            try:
                result = finished.result()
                if self.accept(result, state, t):
                    for name in ('updated', 'times', 'controls', 'states', 'info'):
                        setattr(self, name, getattr(result, name))
                    self.accepted += 1
                else:
                    self.rejected += 1
            except Exception as error:
                self.rejected += 1
                self.info['coordination_failure'] = f'{type(error).__name__}: {error}'
        if self.controls is not None and t-self.updated > .6:
            self.controls = self.times = None
        if self.pending is None and t-self.requested >= .4-1e-8:
            if self.pool is None:
                import multiprocessing
                from concurrent.futures import ProcessPoolExecutor
                from .shot_preview_worker import initialize_worker
                self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)
            snapshot = CoordinationPreview()
            for name in ('trajectory', 'enabled', 'times', 'controls'):
                setattr(snapshot, name, getattr(self, name))
            self.pending = self.pool.submit(compute_coordination, snapshot,
                np.asarray(state).copy(), previous.copy(), commanded_q.copy(), t, dt, base_z, limits,
                base_rotation=None if base_rotation is None else base_rotation.copy(),
                measured_control=None if measured_control is None else measured_control.copy())
            self.pending_generation, self.pending_t = self.generation, t
            self.requested = t
        self.info.update(coordination_async=True, coordination_pending=self.pending is not None,
                         coordination_accepted=self.accepted, coordination_rejected=self.rejected,
                         coordination_age_s=None if self.controls is None else t-self.updated)
        return None if self.controls is None else self.sample(t+dt)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
