"""Continuous path clock: keep shot geometry while allowing the base to catch up."""
import numpy as np


class ShotClock:
    def __init__(self):
        self.time = 0.
        self.rate = 0.

    def advance(self, dt, position_error, orientation_error, transition=False,
                heading_error=0., reach=.25, minimum_rate=.15,
                position_scale=.1, orientation_scale=.12, preparing=False):
        # Never skip waypoints to catch up with wall time. A quintic camera
        # path stays continuous and its feedforward uses this same clock rate.
        if (not np.isfinite(position_scale) or not np.isfinite(orientation_scale)
                or position_scale <= 0. or orientation_scale <= 0.):
            raise ValueError('clock accuracy scales must be positive and finite')
        # Smooth error-energy governor, tied to the requested accuracy. The
        # old flat 10 cm deadband accelerated to full speed even at 7 cm error
        # with a wrist on its stop. Slow earlier, without skipping poses or
        # adding arrival/recovery states; executed commands still use the QP.
        lag = (position_error/position_scale)**2+(orientation_error/orientation_scale)**2
        desired = float(np.clip(1./(1.+lag), minimum_rate, 1.))
        if transition:
            # Spend the existing hold on the upcoming base orientation and
            # compact arm posture. Bounded slowdown avoids unending pauses.
            preparation = .7/(1.+(heading_error/.35)**2+(max(reach-.25, 0.)/.10)**2)
            desired = min(desired, float(np.clip(preparation, min(.2, minimum_rate), .7)))
            if preparing:
                # Parking can still require transverse travel when heading and
                # arm reach already look ready. Keep time for that measured move.
                desired = min(desired, minimum_rate)
        self.rate += float(np.clip(desired-self.rate, -.4*dt, .4*dt))
        self.time += self.rate*dt
        return self.time, self.rate


class FixedMotionShotClock(ShotClock):
    """Execute action intervals at their specified rate; retain adaptive holds.

    Neither collision risk nor tracking error changes an action's duration.
    Split a tick at action boundaries so a slow connector cannot consume any
    of the following action at its connector rate.
    """
    motion_timing_mode = 'fixed_physical_time'

    def __init__(self):
        super().__init__()
        self.trajectory = None

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def advance(self, dt, position_error, orientation_error, transition=False,
                heading_error=0., reach=.25, **kwargs):
        if self.trajectory is None:
            raise RuntimeError('set the trajectory before advancing the motion clock')
        if not np.isfinite(dt) or dt <= 0.:
            raise ValueError('clock dt must be positive and finite')
        intervals = [(s.get('motion_start_s', s['start_s']),
                      s.get('motion_end_s', s['end_s'])) for s in self.trajectory.shots]
        remaining = float(dt)
        while remaining > 1e-12:
            end = next((b for a, b in intervals if a <= self.time < b), None)
            if end is not None:
                used = min(remaining, end-self.time)
                self.time += used
                self.rate = 1.
                remaining -= used
                continue
            next_start = min((a for a, _ in intervals if a > self.time), default=np.inf)
            previous = self.time
            super().advance(remaining, position_error, orientation_error,
                self.trajectory.transition(previous) is not None, heading_error, reach, **kwargs)
            if self.time <= next_start:
                break
            # The hold crossed into an action during this tick. Keep its
            # physical duration, then run the remainder at the action rate.
            used = (next_start-previous)/self.rate
            self.time = next_start
            remaining -= used
        return self.time, self.rate
