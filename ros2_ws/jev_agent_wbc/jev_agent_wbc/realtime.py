"""Transport-independent, latched actuator watchdog with two independent clocks.

Zero is a brake request to the acceleration-limited drive, not a proof that a
moving obstacle cannot hit a stopped robot. Reset requires a new armed session.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RuntimePolicy:
    command_timeout_s: float = .10
    sensor_timeout_s: float = .20
    planning_timeout_s: float = 1.
    startup_timeout_s: float = .20

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('runtime deadlines must be finite and positive')


class CommandWatchdog:
    def __init__(self, policy, wall):
        self.policy = policy
        self.started_wall = wall
        self.packet = None
        self.sequence = -1
        self.fault = None
        self.emergency_stop_required = None
        self.rejected = 0

    def trip(self, reason, t, wall):
        if self.fault is None:
            self.fault = dict(reason=reason, time_s=float(t), wall_s=float(wall))

    def escalate(self, reason, t, wall):
        self.trip(reason, t, wall)
        if self.emergency_stop_required is None:
            self.emergency_stop_required = dict(reason=reason, time_s=float(t), wall_s=float(wall),
                action='external safety controller / operator intervention required')

    def accept(self, packet, t, wall):
        if self.fault is not None:
            return False
        try:
            values = np.asarray(packet['command'], float)
            stamp, sent = float(packet['t']), float(packet['observation_wall'])
            sequence = packet['sequence']
            valid = (values.shape == (8,) and np.isfinite(values).all()
                     and math.isfinite(stamp) and math.isfinite(sent)
                     and isinstance(sequence, int) and sequence > self.sequence
                     and -1e-8 <= t-stamp <= self.policy.command_timeout_s
                     and 0 <= wall-sent <= self.policy.command_timeout_s)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            self.rejected += 1
            return False
        self.packet = dict(packet, command=values.copy())
        self.sequence = sequence
        return True

    def command(self, t, wall, sensor_stamp, *, alive=True):
        if not alive:
            self.trip('controller_process_exited', t, wall)
        if not math.isfinite(sensor_stamp) or not -1e-8 <= t-sensor_stamp <= self.policy.sensor_timeout_s:
            self.trip('sensor_stale', t, wall)
        if self.packet is None:
            if max(t, wall-self.started_wall) > self.policy.startup_timeout_s:
                self.trip('first_command_timeout', t, wall)
        elif (t-self.packet['t'] > self.policy.command_timeout_s
              or wall-self.packet['observation_wall'] > self.policy.command_timeout_s):
            self.trip('command_timeout', t, wall)
        if self.fault is not None or self.packet is None:
            return np.zeros(8)
        return self.packet['command'].copy()


def realtime_assessment(timing, watchdog_fault, shot_accepted):
    """A rendered simulation-time movie cannot establish wall-clock success."""
    timing_pass = bool(timing['pace'] == 1. and timing['max_tick_wall_s'] <= .02
                       and timing['max_lag_s'] <= .02)
    return dict(**timing, timing_pass=timing_pass, watchdog_fault=watchdog_fault,
                success=bool(timing_pass and watchdog_fault is None and shot_accepted))
