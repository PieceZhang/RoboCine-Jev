"""Measured body-velocity command servo for unchanged learned legged policies.

This changes only the supported policy input. It never edits joint actions,
root state, original joint limits or actuator effort/velocity limits.
"""
import numpy as np


class VelocityServo:
    def __init__(self, limits=(.5, .5, 1.), kp=(1., 1., .8), ki=(1.2, 1.2, 1.),
                 integral_limit=(.2, .2, .3), slew=(.5, .5, .8), filter_s=.10,feedforward_gain=(1.,1.,1.),negative_feedforward_gain=None):
        self.limits = np.asarray(limits, float)
        self.kp, self.ki = np.asarray(kp, float), np.asarray(ki, float)
        self.integral_limit, self.slew = np.asarray(integral_limit, float), np.asarray(slew, float)
        self.filter_s = float(filter_s)
        self.feedforward_gain=np.asarray(feedforward_gain,float)
        self.negative_feedforward_gain=np.asarray(negative_feedforward_gain if negative_feedforward_gain is not None else feedforward_gain,float)
        for vector in (self.limits, self.kp, self.ki, self.integral_limit, self.slew,self.feedforward_gain,self.negative_feedforward_gain):
            if vector.shape != (3,) or not np.isfinite(vector).all() or np.any(vector <= 0):
                raise ValueError('finite positive three-axis servo parameters required')
        if not np.isfinite(self.filter_s) or self.filter_s < 0:
            raise ValueError('finite nonnegative filter_s required')
        self.integral = np.zeros(3); self.command = np.zeros(3); self.filtered = None
        self.integral_before_update=self.integral.copy()

    def update(self, desired_body, measured_body, dt, *, pose_error_body=None):
        desired, measured = np.asarray(desired_body, float), np.asarray(measured_body, float)
        if (desired.shape != (3,) or measured.shape != (3,) or not np.isfinite([desired, measured]).all()
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError('finite desired/measured body vx,vy,wz and positive dt required')
        if np.any(np.abs(desired) > self.limits+1e-9):
            raise ValueError('desired velocity exceeds supported command range')
        self.integral_before_update=self.integral.copy()
        if pose_error_body is not None:
            anchor = np.asarray(pose_error_body, float)
            if anchor.shape != (3,) or not np.isfinite(anchor).all():
                raise ValueError('finite body xy/yaw stop anchor error required')
            desired = np.clip(desired+np.clip(1.5*anchor, -.05, .05), -self.limits, self.limits)
        alpha = dt/(self.filter_s+dt)
        self.filtered = measured.copy() if self.filtered is None else self.filtered+alpha*(measured-self.filtered)
        error = desired-self.filtered
        candidate = np.clip(self.integral+self.ki*error*dt, -self.integral_limit, self.integral_limit)
        raw = desired*np.where(desired<0,self.negative_feedforward_gain,self.feedforward_gain)+self.kp*error+candidate
        low = np.maximum(-self.limits, self.command-self.slew*dt)
        high = np.minimum(self.limits, self.command+self.slew*dt)
        bounded = np.clip(raw, low, high)
        # Integrate only directions that can actually be applied. Unwind while
        # saturated, including slew saturation; a stop cannot retain windup.
        accept = ((raw >= low) & (raw <= high)) | ((raw > high) & (error < 0)) | ((raw < low) & (error > 0))
        self.integral = np.where(accept, candidate, self.integral)
        self.command = bounded
        return bounded.copy()

    def applied(self, actual_command, *, transient_hold=False):
        """Back-calculate when a camera/workspace guard reduces policy input."""
        actual = np.asarray(actual_command, float)
        if actual.shape != (3,) or not np.isfinite(actual).all() or np.any(np.abs(actual) > self.limits+1e-9):
            raise ValueError('finite supported applied policy input required')
        discrepancy = actual-self.command
        # A camera guard can veto positive motion indefinitely. Unwind existing
        # integral toward zero, without accumulating an opposite hidden request.
        if transient_hold:
            # A brief missing module result is not a permanent workspace veto.
            # Freeze integration for this tick; retain the actual command for
            # the next slew bound. Existing safety vetoes use normal unwinding.
            self.integral=self.integral_before_update.copy()
        else:
            unwind = self.integral*discrepancy < 0
            self.integral = np.where(unwind, np.sign(self.integral)*np.maximum(
                0., np.abs(self.integral)-.5*np.abs(discrepancy)), self.integral)
        self.command = actual.copy()

    def reset(self):
        self.integral[:] = 0.; self.command[:] = 0.; self.filtered = None
        self.integral_before_update=self.integral.copy()


def stop_pose_error(anchor, current):
    """World xy/yaw anchor error expressed in the current physical body axes."""
    from scipy.spatial.transform import Rotation
    anchor, current = np.asarray(anchor, float), np.asarray(current, float)
    translation = current[:3, :3].T@(anchor[:3, 3]-current[:3, 3])
    # A locomotion policy exposes yaw rate, not independent roll/pitch control.
    yaw = Rotation.from_matrix(anchor[:3, :3]@current[:3, :3].T).as_rotvec()[2]
    return np.r_[translation[:2], yaw]


def parking_transition(anchor,current,filtered_velocity,stop_age_s,parking,*,reset_anchor_on_entry=False):
    """Measured brake-to-stand transition with an optional fresh pose latch.

    A camera trajectory is not an input and cannot be modified here. The latch
    removes braking displacement from the subsequent drift monitor exactly at
    measured stand entry. Actual later 25mm/50mrad drift still exits stand.
    """
    anchor,current=np.asarray(anchor,float),np.asarray(current,float)
    velocity=np.asarray(filtered_velocity,float)
    entered=bool(not parking and stop_age_s>.5 and np.linalg.norm(velocity[:2])<.02 and abs(velocity[2])<.03)
    if entered:
        parking=True
        if reset_anchor_on_entry:anchor=current.copy()
    error=stop_pose_error(anchor,current)
    exited=bool(parking and (np.linalg.norm(error[:2])>.025 or abs(error[2])>.05))
    if exited:parking=False
    return parking,anchor,dict(entered=entered,exited=exited,anchor_error_body=error.tolist(),
        reset_anchor_on_entry=bool(entered and reset_anchor_on_entry))


def g1_profile(name):
    if name=='baseline':return {}
    if name=='low_gain':
        return dict(kp=(.35,.5,.4),ki=(.35,.5,.4),filter_s=.12,feedforward_gain=(1.10,1.,1.))
    raise ValueError('unknown G1 measured-velocity profile')


def go2_profile(name):
    if name=='legacy':return {}
    if name=='measured_backward_v1':return dict(negative_feedforward_gain=(1/.55,1.,1.))
    raise ValueError('unknown Go2 measured-velocity profile')
