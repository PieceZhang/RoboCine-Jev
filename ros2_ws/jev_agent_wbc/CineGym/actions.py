"""Common physical velocity action contract for both execution backends."""
import numpy as np
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER


def validate_action(action, state, previous, dt, limits):
    u = np.asarray(action, float)
    if u.shape != (8,) or not np.isfinite(u).all():
        raise ValueError('action must contain eight finite physical velocities')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('action interval must be positive and finite')
    previous, state = np.asarray(previous), np.asarray(state)
    wheels, old_wheels = limits.wheels@u, limits.wheels@previous
    if (np.any(abs(u) > limits.velocity+1e-7) or
        np.any(abs(u-previous) > limits.acceleration*dt+1e-7) or
        np.any(abs(wheels) > limits.wheel_speed+1e-7) or
        np.any(abs(wheels-old_wheels) > limits.wheel_acceleration*dt+1e-7)):
        raise ValueError('action exceeds actuator velocity or acceleration limits')
    q = state[3:]+u[2:]*dt
    if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
        raise ValueError('action would exceed joint position limits')
    return u.copy()
