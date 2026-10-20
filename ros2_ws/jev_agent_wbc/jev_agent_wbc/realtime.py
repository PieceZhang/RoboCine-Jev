"""Transport-independent actuator leases with two independent clocks.

Zero is a brake request to the acceleration-limited drive, not a proof that a
moving obstacle cannot hit a stopped robot. Recoverable gaps require new,
sensor-bound commands; hard faults require a new armed session.
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
    recover_decisions: bool = False
    recovery_timeout_s: float = 15.
    max_retries: int = 3

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for k,v in vars(self).items() if k != 'recover_decisions'):
            raise ValueError('runtime deadlines must be finite and positive')
        if type(self.max_retries) is not int:
            raise ValueError('max_retries must be an integer')


class CommandWatchdog:
    def __init__(self, policy, wall):
        self.policy = policy
        self.started_wall = wall
        self.packet = None
        self.sequence = -1
        self.fault = None
        self.emergency_stop_required = None
        self.rejected = 0
        self.waiting = None
        self.sensor_waiting = None
        self.certificate_waiting = None
        self.recoveries = 0

    def trip(self, reason, t, wall):
        if self.fault is None:
            self.fault = dict(reason=reason, time_s=float(t), wall_s=float(wall))

    def escalate(self, reason, t, wall):
        self.trip(reason, t, wall)
        if self.emergency_stop_required is None:
            self.emergency_stop_required = dict(reason=reason, time_s=float(t), wall_s=float(wall),
                action='external safety controller / operator intervention required')

    def accept(self, packet, t, wall):
        self._certificate_deadline(t,wall)
        if self.waiting and max(t-self.waiting['time_s'],wall-self.waiting['wall_s']) > self.policy.recovery_timeout_s:
            self.trip('command_recovery_exhausted',t,wall)
        if self.sensor_waiting and max(t-self.sensor_waiting['time_s'],wall-self.sensor_waiting['wall_s'])>self.policy.recovery_timeout_s:
            self.trip('sensor_recovery_exhausted',t,wall)
        if self.fault is not None:
            return False
        try:
            values = np.asarray(packet['command'], float)
            stamp, sent = float(packet['t']), float(packet['observation_wall'])
            sequence = packet['sequence']
            valid = (values.shape == (8,) and np.isfinite(values).all()
                     and math.isfinite(stamp) and math.isfinite(sent)
                     and isinstance(sequence, int) and sequence > self.sequence
                     and -1e-8 <= t-stamp <= self.policy.command_timeout_s+1e-9
                     and 0 <= wall-sent <= self.policy.command_timeout_s+1e-9)
            acceleration=packet.get('control_acceleration')
            if acceleration is not None:
                acceleration=np.asarray(acceleration,float)
                valid &= acceleration.shape==(8,) and np.isfinite(acceleration).all()
            if 'sensor_stamp' in packet:
                sensor=float(packet['sensor_stamp'])
                valid &= math.isfinite(sensor) and sensor<=stamp+1e-8
            if self.sensor_waiting is not None:
                sensor=float(packet.get('sensor_stamp',float('-inf')))
                valid &= (math.isfinite(sensor) and sensor>self.sensor_waiting['sensor_stamp']
                          and -1e-8<=t-sensor<=self.policy.sensor_timeout_s+1e-9)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            self.rejected += 1
            return False
        self.packet = dict(packet, command=values.copy())
        self.sequence = sequence
        if self.waiting is not None:
            self.recoveries += 1
            self.waiting = None
        if self.sensor_waiting is not None:
            self.recoveries += 1
            self.sensor_waiting = None
        return True

    def command(self, t, wall, sensor_stamp, *, alive=True):
        self._certificate_deadline(t,wall)
        if not alive:
            self.trip('controller_process_exited', t, wall)
        if not math.isfinite(sensor_stamp) or not -1e-8 <= t-sensor_stamp <= self.policy.sensor_timeout_s+1e-9:
            if self.policy.recover_decisions and math.isfinite(sensor_stamp) and sensor_stamp<=t:
                if self.sensor_waiting is None:
                    self.sensor_waiting=dict(reason='sensor_stale',time_s=float(t),wall_s=float(wall),sensor_stamp=float(sensor_stamp))
            else:
                self.trip('sensor_stale', t, wall)
        if self.sensor_waiting and max(t-self.sensor_waiting['time_s'],wall-self.sensor_waiting['wall_s'])>self.policy.recovery_timeout_s:
            self.trip('sensor_recovery_exhausted',t,wall)
        if self.packet is None:
            if max(t, wall-self.started_wall) > self.policy.startup_timeout_s:
                self._command_gap('first_command_timeout', t, wall)
        elif (t-self.packet['t'] > self.policy.command_timeout_s+1e-9
              or wall-self.packet['observation_wall'] > self.policy.command_timeout_s+1e-9):
            self._command_gap('command_timeout', t, wall)
        if self.waiting and max(t-self.waiting['time_s'],wall-self.waiting['wall_s']) > self.policy.recovery_timeout_s:
            self.trip('command_recovery_exhausted',t,wall)
        if (self.fault is not None or self.packet is None or self.waiting is not None
                or self.sensor_waiting is not None or self.certificate_waiting is not None):
            return np.zeros(8)
        return self.packet['command'].copy()

    def wait_for_certificate(self, reason, t, wall, sensor_stamp, required_track_ids=()):
        if not self.policy.recover_decisions:
            self.escalate(reason,t,wall)
        elif self.fault is None and self.certificate_waiting is None:
            self.certificate_waiting=dict(reason=reason,time_s=float(t),wall_s=float(wall),
                sensor_stamp=float(sensor_stamp),sequence=self.sequence,
                required_track_ids=sorted(set(int(x) for x in required_track_ids if x>0)))
        elif self.fault is None and self.certificate_waiting is not None:
            self.certificate_waiting['required_track_ids']=sorted(set(
                self.certificate_waiting.get('required_track_ids',()))|set(int(x) for x in required_track_ids if x>0))
        self._certificate_deadline(t,wall)

    def _certificate_deadline(self,t,wall):
        waiting=self.certificate_waiting
        if waiting and max(t-waiting['time_s'],wall-waiting['wall_s'])>self.policy.recovery_timeout_s:
            self.escalate('certificate_recovery_exhausted',t,wall)

    def certificate_candidate(self,t,wall,sensor_stamp):
        """Unissued proposal for the actuator's complete current-state gates."""
        self._certificate_deadline(t,wall)
        waiting=self.certificate_waiting;packet=self.packet
        if (waiting is None or packet is None or self.fault is not None
                or self.waiting is not None or self.sensor_waiting is not None):return None
        sensor=float(packet.get('sensor_stamp',float('-inf')))
        if (packet['sequence']<=waiting['sequence'] or packet['t']<=waiting['time_s']
                or not math.isfinite(sensor_stamp)
                or not -1e-8<=t-sensor_stamp<=self.policy.sensor_timeout_s+1e-9
                or sensor<=waiting['sensor_stamp'] or sensor_stamp<=waiting['sensor_stamp']
                or not -1e-8<=t-sensor<=self.policy.sensor_timeout_s+1e-9
                or not -1e-8<=t-packet['t']<=self.policy.command_timeout_s+1e-9
                or not 0<=wall-packet['observation_wall']<=self.policy.command_timeout_s+1e-9):return None
        return packet['command'].copy()

    def certificate_passed(self,t,wall,sensor_stamp,fresh_track_ids=()):
        # Only the independent actuator may call this AFTER every sweep gate.
        if self.certificate_candidate(t,wall,sensor_stamp) is None:return False
        # A forgotten/expired obstacle is unknown, not observed free space.
        if not set(self.certificate_waiting.get('required_track_ids',())).issubset(fresh_track_ids):return False
        self.certificate_waiting=None;self.recoveries+=1
        return True

    def _command_gap(self, reason, t, wall):
        if not self.policy.recover_decisions:
            self.trip(reason,t,wall)
        elif self.waiting is None:
            # Expired commands are never extended. A fresh, ordered packet must
            # still pass the actuator's current-state and ramp certificates.
            self.waiting=dict(reason=reason,time_s=float(t),wall_s=float(wall))


def realtime_assessment(timing, watchdog_fault, shot_accepted):
    """A rendered simulation-time movie cannot establish wall-clock success."""
    timing_pass = bool(timing['pace'] == 1. and timing['max_tick_wall_s'] <= .02
                       and timing['max_lag_s'] <= .02)
    return dict(**timing, timing_pass=timing_pass, watchdog_fault=watchdog_fault,
                success=bool(timing_pass and watchdog_fault is None and shot_accepted))
