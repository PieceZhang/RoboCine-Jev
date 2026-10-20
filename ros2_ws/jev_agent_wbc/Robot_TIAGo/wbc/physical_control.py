"""Isaac-side math/guards for the genuine isolated Giskard adapter.

No optimizer executes inside Isaac. The independent guard only bounds/brakes
native commands under the same original actuator and stopping-distance policy.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def validate_trajectory(data,dt=.01):
    ts = np.asarray([x['t_s'] for x in data],float)
    ps = np.asarray([x['position_m'] for x in data],float)
    qs = np.asarray([x['quaternion_xyzw'] for x in data],float)
    if (len(ts)<2 or ts[0]!=0 or np.any(np.diff(ts)<=0) or ts[-1]<dt
            or ps.shape!=(len(ts),3) or qs.shape!=(len(ts),4)
            or not all(np.isfinite(x).all() for x in (ts,ps,qs))
            or not np.allclose(np.linalg.norm(qs,axis=1),1.,atol=1e-6,rtol=0)):
        raise ValueError('Invalid authored optical trajectory')
    return ts,ps,Slerp(ts,Rotation.from_quat(qs))


def wheel_speeds(v,yaw_rate,radius,separation):
    return np.array([v-yaw_rate*separation/2,v+yaw_rate*separation/2])/radius


def spatial_jacobian_at_offset(jacobian,offset_world):
    """Translate a world-expressed [linear;angular] Jacobian to another point.

    ``offset_world`` points from the old reference point to the new point.
    PhysX dense articulation Jacobians use each link's COM; collision-model
    point Jacobians require its actor origin. Both linear and angular columns
    are retained; this is a reference-point shift, not a frame rotation.
    Batched (...,6,n) inputs are supported.
    """
    result=np.array(jacobian,dtype=float,copy=True)
    offset=np.asarray(offset_world,dtype=float)
    if result.shape[-2]!=6 or offset.shape!=result.shape[:-2]+(3,):
        raise ValueError('Expected (...,6,n) Jacobian and (...,3) world offset')
    result[...,:3,:]+=np.cross(np.swapaxes(result[...,3:,:],-1,-2),offset[...,None,:]).swapaxes(-1,-2)
    return result


class WheelParkingServo:
    """Latch physical wheel encoders only after an explicit stop request.

    This changes only low-level motor stiffness. The upstream velocity target
    and its unchanged acceleration limits remain authoritative. Any nonzero
    native command releases the position drive immediately. No target changes
    while latched, no wheel teleport, and no external force is applied.
    """
    def __init__(self,kp=300.,speed_threshold=.08,settled_samples=3):
        self.kp=float(kp);self.speed_threshold=float(speed_threshold)
        self.settled_samples=int(settled_samples);self.slow_samples=0
        self.target=None;self.events=[]

    def update(self,stamp,requested,issued_wheel_rates,measured_q,measured_qd):
        rate,q,qd=map(np.asarray,(issued_wheel_rates,measured_q,measured_qd))
        zero=bool(np.max(np.abs(rate))<1e-9)
        if not requested or not zero:
            if self.target is not None:
                self.events.append(dict(stamp_s=float(stamp),mode='velocity',reason='nonzero native command' if not zero else 'parking not requested'))
            self.target=None;self.slow_samples=0
        elif self.target is None:
            self.slow_samples=self.slow_samples+1 if np.max(np.abs(qd))<self.speed_threshold else 0
            if self.slow_samples>=self.settled_samples:
                self.target=q.copy()
                self.events.append(dict(stamp_s=float(stamp),mode='position_hold',target_rad=self.target.tolist(),
                    measured_velocity_radps=qd.tolist(),reason='explicit park; native zero; measured wheels decelerated'))
        return dict(active=self.target is not None,kp=self.kp if self.target is not None else 0.,
            target_rad=None if self.target is None else self.target.copy())

    def manifest(self):
        return dict(position_kp=self.kp,latch_speed_threshold_radps=self.speed_threshold,
            latch_consecutive_samples=self.settled_samples,target='one actual measured encoder snapshot after deceleration; unchanged while parked',
            release='any nonzero native issued wheel command, or explicit parking request removed',
            velocity_acceleration_effort_caps='unchanged; no external joint forces')


def camera_jacobian(base_transform,camera_position,link_jacobian,offset_world):
    base = np.zeros((6,2))
    base[:3,0]=base_transform[:3,0]
    base[:3,1]=np.cross(base_transform[:3,2],camera_position-base_transform[:3,3])
    base[3:,1]=base_transform[:3,2]
    arm=np.array(link_jacobian,float,copy=True)
    arm[:3]+=np.cross(arm[3:].T,offset_world).T
    return np.c_[base,arm]


class CameraController:
    """Limits only: the genuine solver lives in another process."""
    def __init__(self,joints,radius,separation,dt=.01):
        self.home=np.array([j['home'] for j in joints])
        self.lower=np.array([j['lower'] for j in joints])
        self.upper=np.array([j['upper'] for j in joints])
        self.speed=np.r_[.22,.35,[j['velocity'] for j in joints]]
        self.accel=np.r_[.45,.65,[j.get('acceleration',1.) for j in joints]]
        self.previous=np.zeros(10)
        self.dt=dt
        self.camera_request=None

    def guard(self,q,command):
        q=np.asarray(q)
        lo=np.maximum(-self.speed,self.previous-self.accel*self.dt)
        hi=np.minimum(self.speed,self.previous+self.accel*self.dt)
        a=self.accel[2:]
        stop=lambda d:np.sqrt((a*self.dt)**2+2*a*np.maximum(d,0.))-a*self.dt
        lo[2:]=np.maximum.reduce([lo[2:],(self.lower+.002-q)/self.dt,-stop(q-self.lower-.002)])
        hi[2:]=np.minimum.reduce([hi[2:],(self.upper-.002-q)/self.dt,stop(self.upper-.002-q)])
        if np.any(lo>hi+1e-9):
            raise RuntimeError('Independent measured-state stopping guard infeasible')
        return np.clip(command,lo,hi)

    def measured_motion_check(self,q,rates):
        """Independent raw physical feedback; never substitutes command state.

        The 1e-4 speed tolerance is numerical/servo reporting tolerance, not a
        changed command limit. Joint stopping reserve uses the unchanged issued
        acceleration policy as a conservative runtime condition, not a dynamics
        or force capability certificate.
        """
        q,rates=np.asarray(q),np.asarray(rates)
        if q.shape!=(8,) or rates.shape!=(10,) or not np.isfinite([*q,*rates]).all():
            raise RuntimeError('Invalid independent measured physical state')
        excess=np.maximum(np.abs(rates)-self.speed,0.)
        if np.any(excess>1e-4):
            raise RuntimeError('Measured physical velocity exceeds unchanged limits: '+repr(excess.tolist()))
        direction_distance=np.where(rates[2:]>=0.,self.upper-.002-q,q-self.lower-.002)
        required=rates[2:]**2/(2*self.accel[2:])+np.abs(rates[2:])*self.dt
        reserve=direction_distance-required
        if np.any(reserve < -1e-6):
            raise RuntimeError('Measured physical velocity lacks joint stopping reserve: '+repr(reserve.tolist()))
        return dict(raw_measured_velocity_limits_checked=True,max_speed_excess=float(excess.max()),
            numerical_velocity_tolerance=1e-4,minimum_joint_stopping_reserve_mixed_units=float(reserve.min()),
            acceleration_source='issued-command policy; no measured acceleration capability certificate')


class PhysicalDofGuard:
    """Raw feedback checks for every real DOF, including wheels/passive holds."""
    def __init__(self,names,urdf_joints,wheel_names):
        self.names=list(names);self.lower=[];self.upper=[];self.speed=[];self.units=[]
        joints={node.get('name'):node for node in urdf_joints}
        for name in self.names:
            node=joints[name];limit=node.find('limit')
            if limit is None:raise ValueError('Missing original physical limits for '+name)
            continuous=node.get('type')=='continuous'
            self.lower.append(-np.inf if continuous else float(limit.get('lower')))
            self.upper.append(np.inf if continuous else float(limit.get('upper')))
            self.speed.append(min(float(limit.get('velocity')),8.) if name in wheel_names else float(limit.get('velocity')))
            self.units.append('m' if node.get('type')=='prismatic' else 'rad')
        self.lower,self.upper,self.speed=map(np.asarray,(self.lower,self.upper,self.speed))

    def check(self,q,qd):
        q,qd=np.asarray(q),np.asarray(qd)
        if q.shape!=self.speed.shape or qd.shape!=q.shape or not np.isfinite([q,qd]).all():
            raise RuntimeError('Invalid raw full-articulation feedback')
        position_excess=np.maximum(np.maximum(self.lower-q,q-self.upper),0.)
        speed_excess=np.maximum(np.abs(qd)-self.speed,0.)
        if np.any(position_excess>1e-5) or np.any(speed_excess>1e-4):
            raise RuntimeError('Raw full-articulation native limits exceeded: '+repr(dict(
                names=self.names,position_excess=position_excess.tolist(),speed_excess=speed_excess.tolist())))
        return dict(raw_physical_dofs_checked=len(q),max_position_excess=float(position_excess.max()),
            max_velocity_excess=float(speed_excess.max()),max_velocity_fraction=float(np.max(np.abs(qd)/self.speed)))

    def manifest(self):
        return dict(dof_names=self.names,joint_units=self.units,
            lower=[float(v) if np.isfinite(v) else None for v in self.lower],
            upper=[float(v) if np.isfinite(v) else None for v in self.upper],speed=self.speed.tolist(),
            position_numerical_tolerance=1e-5,velocity_numerical_tolerance=1e-4,
            passive_behavior='original fixed hold targets and original physical gains/effort caps; no invented acceleration limits')
