"""Causal continuous drive targets for a sparse ROS joint-command stream."""
import math

import numpy as np


class JointCommandFilter:
    """Give wheels and arm velocities the same critically damped response.

    A new message changes the reference, never the current target position or
    velocity. Zero frequency bypasses filtering and forwards native targets.
    The filter runs in physical time and changes drive targets only.
    """

    def __init__(self, position, lower, upper, speed, frequency_hz, timeout=.15):
        self.lower, self.upper, self.speed = [np.asarray(v, float) for v in (lower, upper, speed)]
        if (not np.isfinite(frequency_hz) or frequency_hz < 0
                or not np.isfinite(timeout) or timeout <= 0
                or self.lower.ndim != 1 or not len(self.lower)
                or self.lower.shape != self.upper.shape or self.speed.shape != self.lower.shape
                or not all(np.all(np.isfinite(a)) for a in (self.lower, self.upper, self.speed))
                or np.any(self.upper-self.lower <= .01) or np.any(self.speed <= 0)):
            raise ValueError('invalid command-filter limits or frequency')
        self.omega, self.timeout = 2*math.pi*frequency_hz, timeout
        self.accepted = self.rejected = self.unsynchronized = self.bounded_steps = 0
        self.maximum_age = 0.
        self.reset(position)

    def reset(self, position):
        position = np.asarray(position, float)
        if (position.shape != self.lower.shape or not np.all(np.isfinite(position))
                or np.any(position < self.lower-1e-6) or np.any(position > self.upper+1e-6)):
            raise ValueError('initial arm position must lie within finite joint limits')
        self.position = position.copy()
        self.velocity = np.zeros_like(self.position)
        self.wheels = np.zeros(2)
        self.filtered_velocity = np.zeros(len(self.position)+2)
        self.acceleration = np.zeros_like(self.filtered_velocity)
        self.command = None

    def push(self, stamp, position, velocity, wheels, time):
        if not np.isfinite(stamp) or not np.isfinite(time) or abs(stamp-time) > self.timeout:
            self.unsynchronized += 1
            return False
        q, v, w = [np.asarray(a, float).copy() for a in (position, velocity, wheels)]
        if (q.shape != self.position.shape or v.shape != q.shape or w.shape != (2,)
                or not all(np.all(np.isfinite(a)) for a in (q, v, w))
                or np.any(q < self.lower-1e-6) or np.any(q > self.upper+1e-6)
                or np.any(np.abs(v) > self.speed+1e-6) or np.any(np.abs(w) > 15.+1e-6)
                or (self.command is not None and stamp < self.command[0])):
            self.rejected += 1
            return False
        self.command = float(stamp), q, v, w
        self.accepted += 1
        return True

    def step(self, time, dt):
        if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('physical timestep must be positive and finite')
        if self.command is None:
            return self.position.copy(), self.velocity.copy(), self.wheels.copy()
        stamp, q, v, wheels = self.command
        age = max(0., time-stamp)
        self.maximum_age = max(self.maximum_age, age)
        if self.omega == 0.:
            self.position = q.copy()
            self.velocity = v.copy() if age <= self.timeout else np.zeros_like(v)
            self.wheels = wheels.copy() if age <= self.timeout else np.zeros(2)
            return self.position.copy(), self.velocity.copy(), self.wheels.copy()
        # For this critical velocity filter, stopping_position is the exact
        # endpoint reached after a zero velocity reference. Its derivative is
        # the input velocity, so it can track the controller's integrated path
        # while the wheels and arm retain the same continuous response.
        stopping_position = (self.position+2*self.velocity/self.omega
                             +self.acceleration[2:]/self.omega**2)
        expected_position = np.clip(q+v*min(age,.05), self.lower+.005, self.upper-.005)
        if age > self.timeout:
            expected_position, v, wheels = q, np.zeros_like(v), np.zeros(2)
        arm_reference = np.clip(v+4*(expected_position-stopping_position), -self.speed, self.speed)
        # Reserve physical travel for the existing filter tail, rather than
        # clipping only when the instantaneous target reaches a joint stop.
        arm_reference = np.clip(arm_reference,
            (self.lower+.005-stopping_position)/dt,
            (self.upper-.005-stopping_position)/dt)
        reference = np.r_[wheels, arm_reference]
        error = self.filtered_velocity-reference
        coefficient = self.acceleration+self.omega*error
        decay = math.exp(-self.omega*dt)
        new_velocity = reference+(error+coefficient*dt)*decay
        new_acceleration = (self.acceleration-self.omega*coefficient*dt)*decay
        new_position = self.position+.5*(self.velocity+new_velocity[2:])*dt
        # Retain physical bounds even for a reversal at the end of travel.
        bounded = np.clip(new_position, np.maximum(self.lower, self.position-self.speed*dt),
                          np.minimum(self.upper, self.position+self.speed*dt))
        clipped = bounded != new_position
        if np.any(clipped):
            self.bounded_steps += 1
            new_velocity[2:] = np.where(clipped, (bounded-self.position)/dt, new_velocity[2:])
            new_acceleration[2:] = np.where(clipped, 0., new_acceleration[2:])
        self.filtered_velocity, self.acceleration = new_velocity, new_acceleration
        self.position, self.velocity, self.wheels = bounded, new_velocity[2:].copy(), new_velocity[:2].copy()
        return self.position.copy(), self.velocity.copy(), self.wheels.copy()
