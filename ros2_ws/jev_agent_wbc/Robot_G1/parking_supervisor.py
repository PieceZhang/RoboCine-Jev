"""Explicit native-input brake/stand/park modes for the unchanged leg policy.

Standing starts from a bounded slew to literal zero. It does not depend on a
filtered walking PI first producing an apparent zero crossing. PARK is only a
measured status; all original physical acceptance thresholds remain external.
"""
import numpy as np


class ParkingSupervisor:
    def __init__(self, brake_s=.5, stable_s=.2, zero_settle_s=.25,
                 slew=(.5,.5,.8), linear_speed_mps=.02, yaw_speed_radps=.03):
        self.brake_s,self.stable_s,self.zero_settle_s=map(float,(brake_s,stable_s,zero_settle_s))
        self.slew=np.asarray(slew,float)
        self.linear_speed_mps,self.yaw_speed_radps=map(float,(linear_speed_mps,yaw_speed_radps))
        if not np.isfinite([self.brake_s,self.stable_s,self.zero_settle_s,self.linear_speed_mps,self.yaw_speed_radps]).all() or min(self.brake_s,self.stable_s,self.zero_settle_s,self.linear_speed_mps,self.yaw_speed_radps)<=0:
            raise ValueError('positive finite declared parking parameters required')
        if self.slew.shape!=(3,) or not np.isfinite(self.slew).all() or np.any(self.slew<=0):
            raise ValueError('positive finite three-axis slew required')
        self.mode='BRAKE';self.phase_start=None;self.zero_start=None;self.stable_start=None;self.anchor=None

    def update(self,stamp_s,body,measured_body_velocity,brake_command,previous_applied,dt):
        from scipy.spatial.transform import Rotation
        body=np.asarray(body,float);measured=np.asarray(measured_body_velocity,float)
        brake,previous=np.asarray(brake_command,float),np.asarray(previous_applied,float)
        if body.shape!=(4,4) or measured.shape!=(3,) or brake.shape!=(3,) or previous.shape!=(3,) or not np.isfinite(np.r_[body.ravel(),measured,brake,previous,stamp_s,dt]).all() or dt<=0:
            raise ValueError('finite measured pose/velocity, native inputs and positive dt required')
        if self.phase_start is None:self.phase_start=float(stamp_s)
        before=self.mode;exited=False
        drift=np.zeros(3)
        if self.anchor is not None:
            drift=np.r_[(body[:3,:3].T@(self.anchor[:3,3]-body[:3,3]))[:2],
                Rotation.from_matrix(self.anchor[:3,:3]@body[:3,:3].T).as_rotvec()[2]]
        if self.mode=='PARK' and (np.linalg.norm(drift[:2])>.025 or abs(drift[2])>.05):
            self.mode='BRAKE';self.phase_start=float(stamp_s);self.zero_start=None;self.stable_start=None;exited=True
        if self.mode=='BRAKE' and stamp_s-self.phase_start>=self.brake_s-1e-10:
            self.mode='SETTLE';self.zero_start=None;self.stable_start=None
        desired=brake if self.mode=='BRAKE' else np.zeros(3)
        applied=previous+np.clip(desired-previous,-self.slew*dt,self.slew*dt)
        literal_zero=bool(np.max(np.abs(applied))<=1e-12)
        stable=bool(np.linalg.norm(measured[:2])<self.linear_speed_mps and abs(measured[2])<self.yaw_speed_radps)
        entered=False
        if self.mode=='SETTLE':
            if literal_zero:
                if self.zero_start is None:self.zero_start=float(stamp_s)
                if stable:
                    if self.stable_start is None:self.stable_start=float(stamp_s)
                else:self.stable_start=None
                if self.stable_start is not None and stamp_s-self.stable_start>=self.stable_s-1e-10 and stamp_s-self.zero_start>=self.zero_settle_s-1e-10:
                    self.mode='PARK';self.anchor=body.copy();entered=True
            else:self.zero_start=None;self.stable_start=None
        return applied,dict(mode=self.mode,previous_mode=before,entered=entered,exited=exited,
            literal_zero_input=literal_zero,instantaneous_velocity_stable=stable,
            stable_since_s=self.stable_start,zero_since_s=self.zero_start,
            anchor_error_body=drift.tolist(),camera_task_modified=False,
            declared_brake_s=self.brake_s,declared_stable_s=self.stable_s,
            declared_zero_settle_s=self.zero_settle_s)
