"""Velocity-level differential-drive WBC. All actuator limits enter one QP.

State: [x, y, yaw, q1..q6]; control: [v, omega, dq1..dq6].
The nonholonomic constraint is eliminated exactly by sdot = S(yaw) u.
No inverse-kinematics solution or post-solve clipping is used.
"""
from dataclasses import dataclass, field
import time

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER, forward_kinematics


@dataclass
class Limits:
    wheel_radius: float = 0.07
    wheel_separation: float = 0.21
    wheel_speed: float = 15.0
    wheel_acceleration: float = 20.0
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.2, 1., 1., 1., 1.2, 1.2, 1.5]))
    acceleration: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.5, 2., 2., 2., 2., 2., 2.]))
    position_margin: float = 0.01
    command_ramp: bool = False
    deadline_s: float = None
    obstacle_avoidance: bool = True

    def __post_init__(self):
        for name in ('velocity', 'acceleration'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (8,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f'{name} must contain eight positive finite limits')
            setattr(self, name, value)
        values = [self.wheel_radius, self.wheel_separation, self.wheel_speed, self.wheel_acceleration]
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError('wheel geometry/limits must be positive and finite')
        if not 0 <= self.position_margin < np.min(JOINT_UPPER - JOINT_LOWER) / 2:
            raise ValueError('invalid joint position margin')
        if self.deadline_s is not None and (not np.isfinite(self.deadline_s) or self.deadline_s <= 0):
            raise ValueError('execution deadline must be positive and finite')

    @property
    def wheels(self):
        matrix = np.zeros((2, 8))
        matrix[:, :2] = [[1, -self.wheel_separation / 2], [1, self.wheel_separation / 2]]
        return matrix / self.wheel_radius


def whole_body_kinematics(state, base_z=0., base_rotation=None):
    state = np.asarray(state, dtype=float)
    if state.shape != (9,) or not np.all(np.isfinite(state)):
        raise ValueError('state must contain nine finite values')
    yaw = state[2]
    rotation = Rotation.from_euler('z', yaw).as_matrix() if base_rotation is None else base_rotation
    p, r, arm_j = forward_kinematics(state[3:], with_jacobian=True)
    offset = rotation @ p
    jacobian = np.zeros((6, 8))
    jacobian[:3, 0] = [np.cos(yaw), np.sin(yaw), 0]
    jacobian[:3, 1] = np.cross([0, 0, 1], offset)
    jacobian[5, 1] = 1
    jacobian[:3, 2:] = rotation @ arm_j[:3]
    jacobian[3:, 2:] = rotation @ arm_j[3:]
    return np.array([state[0], state[1], base_z]) + offset, rotation @ r, jacobian


class WholeBodyQP:
    def __init__(self, limits=None):
        self.limits = limits or Limits()
        self.previous = np.zeros(8)
        self.time_s = 0.
        self.solver = None
        self.weight = np.array([1., 1., 1., .35, .35, .35])
        self.regularization = np.array([.002, .002, .001, .001, .001, .001, .001, .001])
        self.smoothing = .008

    def bounds(self, q, dt, commanded_q=None):
        lim = self.limits
        lower = np.maximum(-lim.velocity, self.previous - lim.acceleration * dt)
        upper = np.minimum(lim.velocity, self.previous + lim.acceleration * dt)
        remaining = None if lim.deadline_s is None else max(0., lim.deadline_s-self.time_s-dt)
        if remaining is not None:
            # Reserve enough deceleration time to reach zero command by the
            # original budget. A late model response can never extend motion.
            lower = np.maximum(lower, -lim.acceleration*remaining)
            upper = np.minimum(upper, lim.acceleration*remaining)
        # Discrete stopping envelope: dt*|dq| + dq^2/(2*a) <= distance.
        # Constraining both measured and integrated target positions prevents drift
        # in the downstream position servo from invalidating command limits.
        # Reserve the margin on the commanded path for servo tracking error.
        # Imposing that *same* interior boundary on the measured path consumes
        # this reserve and can make braking infeasible for a tiny servo overshoot.
        paths = [(q, lim.position_margin)] if commanded_q is None else [(q, 0.), (commanded_q, lim.position_margin)]
        for positions, margin in paths:
            lower[2:] = np.maximum(lower[2:], (JOINT_LOWER + margin - positions) / dt)
            upper[2:] = np.minimum(upper[2:], (JOINT_UPPER - margin - positions) / dt)
            low_distance = np.maximum(positions - JOINT_LOWER - margin, 0.)
            high_distance = np.maximum(JOINT_UPPER - margin - positions, 0.)
            a = lim.acceleration[2:]
            if lim.command_ramp:
                # Motor targets ramp from previous to new velocity during dt.
                # Reserve the old half-step BEFORE solving the remaining
                # half-step + continuous stopping-distance inequality. Using
                # dt*new alone underestimates travel while decelerating.
                def ramp_stop_bound(distance, initial):
                    positive = np.maximum(initial, 0.)
                    positive[positive <= 1e-6] = 0.  # same accepted velocity residual, avoids division by tiny distance
                    if commanded_q is not None and margin > 0:
                        # A nanometre-scale position error at the EXTRA command
                        # reserve can amplify into a large velocity-bound gap
                        # when stopping inside dt. Repair only this numerical
                        # reserve deficit; the measured physical joint bounds
                        # (margin == 0) and all acceleration bounds stay exact.
                        stopping_distance = positive**2/(2*a)
                        deficit = stopping_distance-distance
                        numerical = (deficit > 0.) & (deficit <= 1e-7)
                        distance = np.where(numerical, stopping_distance, distance)
                    remaining = distance-.5*dt*positive
                    bound = -.5*a*dt+np.sqrt((.5*a*dt)**2+2*a*np.maximum(remaining, 0.))
                    # If stopping occurs INSIDE this interval, a negative final
                    # velocity can still be safe: maximum travel is
                    # dt*v0^2/(2*(v0-v1)), not the endpoint displacement.
                    crossing = remaining < 0.
                    bound[crossing] = positive[crossing]-dt*positive[crossing]**2/(2*np.maximum(distance[crossing], 1e-14))
                    return bound
                lower[2:] = np.maximum(lower[2:], -ramp_stop_bound(low_distance, -self.previous[2:]))
                upper[2:] = np.minimum(upper[2:], ramp_stop_bound(high_distance, self.previous[2:]))
            else:
                stop_dt = dt
                lower[2:] = np.maximum(lower[2:], a * stop_dt - np.sqrt((a * stop_dt)**2 + 2*a*low_distance))
                upper[2:] = np.minimum(upper[2:], -a * stop_dt + np.sqrt((a * stop_dt)**2 + 2*a*high_distance))
        wheel_previous = lim.wheels @ self.previous
        wl = np.maximum(-lim.wheel_speed, wheel_previous - lim.wheel_acceleration * dt)
        wu = np.minimum(lim.wheel_speed, wheel_previous + lim.wheel_acceleration * dt)
        if remaining is not None:
            wl = np.maximum(wl, -lim.wheel_acceleration*remaining)
            wu = np.minimum(wu, lim.wheel_acceleration*remaining)
        lower, upper = np.r_[lower, wl], np.r_[upper, wu]
        # Adjacent exact stopping/acceleration faces can differ by the accepted
        # 1e-6 command residual from the preceding solve. Only reconcile an
        # intersection whose midpoint still meets BOTH original faces within
        # that same numerical tolerance; larger gaps remain infeasible.
        numerical = (lower > upper) & (lower-upper <= 2e-6)
        middle = .5*(lower[numerical]+upper[numerical])
        lower[numerical] = upper[numerical] = middle
        return np.vstack([np.eye(8), lim.wheels]), lower, upper

    def solve(self, state, target_p, target_r, feedforward, dt, *, base_z=0., base_rotation=None,
              commanded_q=None, brake=False):
        if not np.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('control dt must be in (0, 0.1] seconds')
        p, r, j = whole_body_kinematics(state, base_z, base_rotation)
        error = np.r_[np.asarray(target_p) - p, Rotation.from_matrix(target_r @ r.T).as_rotvec()]
        desired = np.asarray(feedforward) + 3. * error
        if not np.all(np.isfinite(desired)):
            raise ValueError('non-finite camera reference')
        jw = self.weight[:, None] * j
        h = jw.T @ jw + np.diag(self.regularization) + self.smoothing * np.eye(8)
        g = -jw.T @ (self.weight * desired) - self.smoothing * self.previous
        if brake:
            h, g = np.eye(8), np.zeros(8)
        a, lower, upper = self.bounds(np.asarray(state)[3:], dt, commanded_q)
        if np.any(lower > upper):
            raise RuntimeError('infeasible hard bounds: state outside braking envelope')
        # Preserve a fixed dense upper-triangle sparsity pattern for OSQP updates.
        rows, cols = np.triu_indices(8)
        p_matrix = sparse.csc_matrix((h[rows, cols], (rows, cols)), shape=(8, 8))
        start = time.perf_counter()
        if self.solver is None:
            self.solver = osqp.OSQP()
            self.solver.setup(P=p_matrix, q=g, A=sparse.csc_matrix(a), l=lower, u=upper,
                              verbose=False, eps_abs=1e-8, eps_rel=1e-8,
                              max_iter=4000, polish=True, warm_start=True)
        else:
            self.solver.update(Px=p_matrix.data, q=g, l=lower, u=upper)
        result = self.solver.solve()
        elapsed = time.perf_counter() - start
        if result.info.status_val != 1 or result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f'QP failed: {result.info.status}')
        violation = float(max(0., np.max(lower - a @ result.x), np.max(a @ result.x - upper)))
        if violation > 1e-6:
            raise RuntimeError(f'QP hard constraint residual {violation}')
        self.previous = result.x.copy()
        next_q = (np.asarray(state)[3:] if commanded_q is None else commanded_q) + dt*result.x[2:]
        return result.x.copy(), dict(position_error=float(np.linalg.norm(error[:3])),
                                     orientation_error=float(np.linalg.norm(error[3:])),
                                     qp_violation=violation, solve_s=elapsed,
                                     command_position_violation=float(max(0., np.max(JOINT_LOWER-next_q), np.max(next_q-JOINT_UPPER))),
                                     command_diff_drive_violation=0.,
                                     solver_iterations=int(result.info.iter))
