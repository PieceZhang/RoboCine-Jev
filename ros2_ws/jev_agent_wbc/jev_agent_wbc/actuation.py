"""Certify a physically interpolated command against the latest dynamic map."""
import numpy as np
from scipy.spatial.transform import Rotation

from .dynamic_collision import DynamicCollision
from .kc_svsdf import advance


def advance_command(state, old, new, duration, ramp=False):
    if not ramp:
        return advance(state, new, duration)
    steps = max(1, int(np.ceil(duration/.02)))
    robot = np.asarray(state).copy()
    for i in range(steps):
        u = old+(i+.5)/steps*(new-old)
        robot = advance(robot, u, duration/steps)
    return robot


def certify_control_ramp(scene, state, old, new, t, dt, base_z=0., steps=10, constant_certificate=None,
                         base_rotation=None):
    robot = np.asarray(state).copy()
    collision = DynamicCollision(scene, t)
    if constant_certificate is not None:
        delta = np.abs(new-old)
        # Conservative perturbation of the already-certified constant command.
        error = dt*(delta[0]+delta[1:]@collision.geometry.reach)
        error += dt*dt*max(abs(old[0]), abs(new[0]))*delta[1]
        lower = constant_certificate-error
        if lower >= collision.margin:
            return float(lower)
    minimum = float('inf')
    interval = dt/steps
    accumulated_error = 0.
    change = np.abs(new-old)/steps
    for i in range(steps):
        command = old+(i+.5)/steps*(new-old)
        collision = DynamicCollision(scene, t+i*interval)
        rotation = (None if base_rotation is None else
                    Rotation.from_euler('z', robot[2]-state[2]).as_matrix()@base_rotation)
        certificate = collision.sweep(robot, command, interval, base_z, max_samples=3, base_rotation=rotation)
        # Midpoint commands approximate the real velocity ramp. Bound the
        # within-interval difference AND its accumulated positional drift;
        # refining time samples alone must never be mistaken for a certificate.
        local_error = interval*(change[0]+change[1:]@collision.geometry.reach)
        local_error += interval**2*max(abs(old[0]), abs(new[0]))*change[1]
        accumulated_error += local_error
        minimum = min(minimum, float(certificate['lower'].min())-accumulated_error)
        if minimum < collision.margin:
            raise RuntimeError('interpolated motor command fails dynamic clearance certificate')
        robot = advance(robot, command, interval)
    return minimum
