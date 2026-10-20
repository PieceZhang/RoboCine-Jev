"""Constrained camera resolved-rate control for a nonholonomic mobile arm.

Inputs/outputs use SI units and world-frame spatial twists [linear, angular].
Only v, yaw rate and actual arm/torso joint speeds are optimization variables.
There is no independently commanded lateral base velocity.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.transform import Rotation, Slerp


def validate_trajectory(data, dt=.01):
    """Validate external references before any simulator is created."""
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError('trajectory needs at least two samples')
    stamps = np.asarray([row['t_s'] for row in data], dtype=float)
    positions = np.asarray([row['position_m'] for row in data], dtype=float)
    quaternions = np.asarray([row['quaternion_xyzw'] for row in data], dtype=float)
    if (stamps.shape != (len(data),) or positions.shape != (len(data), 3)
            or quaternions.shape != (len(data), 4)
            or not all(np.isfinite(x).all() for x in (stamps, positions, quaternions))
            or stamps[0] != 0 or np.any(np.diff(stamps) <= 0) or stamps[-1] < dt
            or not np.allclose(np.linalg.norm(quaternions, axis=1), 1., atol=1e-5, rtol=0)):
        raise ValueError('trajectory requires increasing finite times, Nx3 positions and unit XYZW quaternions')
    return stamps, positions, Slerp(stamps, Rotation.from_quat(quaternions))


def wheel_speeds(v, yaw_rate, radius, separation):
    if radius <= 0 or separation <= 0:
        raise ValueError('wheel geometry must be positive')
    return np.array([v-yaw_rate*separation/2, v+yaw_rate*separation/2])/radius


def camera_jacobian(base_transform, camera_position, link_jacobian, offset_world):
    """Shift the measured EE Jacobian to optical centre, add admissible base DOFs."""
    base_transform = np.asarray(base_transform)
    forward, up = base_transform[:3, 0], base_transform[:3, 2]
    base = np.zeros((6, 2))
    base[:3, 0] = forward
    base[:3, 1] = np.cross(up, np.asarray(camera_position)-base_transform[:3, 3])
    base[3:, 1] = up
    arm = np.array(link_jacobian, dtype=float, copy=True)
    arm[:3] += np.cross(arm[3:].T, np.asarray(offset_world)).T
    return np.column_stack([base, arm])


class CameraController:
    """Convex quadratic velocity task with hard actuator/position/acceleration bounds.

    This is a local kinematic WBC, not an MPC or collision-free path planner.
    An infeasible optimization never sends its last candidate to the robot.
    """
    def __init__(self, joints, radius, separation, dt=.01):
        self.home = np.array([j['home'] for j in joints])
        self.lower = np.array([j['lower'] for j in joints])
        self.upper = np.array([j['upper'] for j in joints])
        self.speed = np.r_[.22, .35, [j['velocity'] for j in joints]]
        self.accel = np.r_[.45, .65, [j.get('acceleration', 1.) for j in joints]]
        self.previous = np.zeros(len(joints)+2)
        self.radius, self.separation, self.dt = radius, separation, dt
        self.wheel_limit = 8.

    def velocity(self, q, jacobian, error, feedforward=None, arm_only=False):
        q, J, error = np.asarray(q), np.asarray(jacobian), np.asarray(error)
        n = len(self.previous)
        if J.shape != (6, n) or q.shape != self.home.shape or error.shape != (6,):
            raise ValueError('incompatible WBC dimensions')
        if not all(np.isfinite(a).all() for a in (q, J, error, self.previous)):
            raise ValueError('nonfinite WBC input')
        target = (np.zeros(6) if feedforward is None else np.asarray(feedforward)) + np.r_[4.*error[:3], 3.*error[3:]]
        lo = np.maximum(-self.speed, self.previous-self.accel*self.dt)
        hi = np.minimum(self.speed, self.previous+self.accel*self.dt)
        # Limit a command using measured joints, including prismatic torso.
        lo[2:] = np.maximum(lo[2:], (self.lower+.002-q)/self.dt)
        hi[2:] = np.minimum(hi[2:], (self.upper-.002-q)/self.dt)
        # Keep enough distance to stop under the same acceleration constraint.
        # For inward speed s: distance >= s*dt + s**2/(2*a). The positive
        # quadratic root accounts for the next held command as well as braking.
        acceleration = self.accel[2:]
        def stoppable_speed(distance):
            return np.sqrt((acceleration*self.dt)**2+2*acceleration*np.maximum(distance, 0.))-acceleration*self.dt
        lo[2:] = np.maximum(lo[2:], -stoppable_speed(q-self.lower-.002))
        hi[2:] = np.minimum(hi[2:], stoppable_speed(self.upper-.002-q))
        if arm_only:
            lo[:2] = hi[:2] = 0.
        if np.any(lo > hi):
            raise RuntimeError('joint recovery conflicts with bounded acceleration; brake')
        # Prefer redundancy/base motion BEFORE the hard stopping envelope.
        # Encoder/drive tracking fluctuations make riding that envelope brittle.
        # This is a smooth cost, not a reduction of the original joint limits.
        lower_distance, upper_distance = q-self.lower, self.upper-q
        zone = np.minimum(.25, .2*(self.upper-self.lower))
        pressure = np.maximum(0., 1.-np.minimum(lower_distance, upper_distance)/zone)**2
        pressure = np.minimum(pressure, 1.)
        away = np.where(lower_distance <= upper_distance, 1., -1.)
        weights = np.r_[.06, .03, .002+.5*pressure]
        H = J.T@J + np.diag(weights)
        preferred = np.r_[0., 0., -.12*(q-self.home)+.3*pressure*away]
        b = J.T@target + weights*preferred
        wheels = np.zeros((2, n))
        wheels[:, :2] = np.array([[1., -self.separation/2], [1., self.separation/2]])/self.radius
        solution = minimize(lambda u: .5*u@H@u-b@u, np.clip(self.previous, lo, hi),
                            jac=lambda u: H@u-b, method='SLSQP', bounds=Bounds(lo, hi),
                            constraints=[LinearConstraint(wheels, -self.wheel_limit, self.wheel_limit)],
                            options={'maxiter': 60, 'ftol': 1e-10})
        u = solution.x
        if (not solution.success or not np.isfinite(u).all() or np.any(u < lo-1e-7)
                or np.any(u > hi+1e-7) or np.any(np.abs(wheels@u) > self.wheel_limit+1e-7)):
            raise RuntimeError(f'WBC solver rejected: {solution.message}')
        self.previous = u.copy()
        return u
