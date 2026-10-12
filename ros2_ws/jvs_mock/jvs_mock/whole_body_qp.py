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
        self.solver = None
        self.weight = np.array([1., 1., 1., .35, .35, .35])
        self.regularization = np.array([.002, .002, .001, .001, .001, .001, .001, .001])
        self.smoothing = .008

    def bounds(self, q, dt, commanded_q=None):
        lim = self.limits
        lower = np.maximum(-lim.velocity, self.previous - lim.acceleration * dt)
        upper = np.minimum(lim.velocity, self.previous + lim.acceleration * dt)
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
            lower[2:] = np.maximum(lower[2:], a * dt - np.sqrt((a * dt)**2 + 2*a*low_distance))
            upper[2:] = np.minimum(upper[2:], -a * dt + np.sqrt((a * dt)**2 + 2*a*high_distance))
        wheel_previous = lim.wheels @ self.previous
        wl = np.maximum(-lim.wheel_speed, wheel_previous - lim.wheel_acceleration * dt)
        wu = np.minimum(lim.wheel_speed, wheel_previous + lim.wheel_acceleration * dt)
        return np.vstack([np.eye(8), lim.wheels]), np.r_[lower, wl], np.r_[upper, wu]

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
