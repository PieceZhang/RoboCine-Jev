"""Explicit camera targets proposed from a constrained continuous robot motion.

Projection changes the proposed target, never the installed controller's task
priority. Every resulting interpolant must pass a fresh strict FK/collision lift.
"""
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from .intent_paths import IntentPath


class FKCameraPath(IntentPath):
    def __init__(self, seed, geometry, joint_seed=None):
        self.seed=seed
        self.original,self.previous=seed.original,seed.previous
        self.activation_s,self.duration=seed.activation_s,seed.duration
        self.subject,self.model,self.intrinsics=seed.subject,seed.model,seed.intrinsics
        self.geometry=geometry
        self.joint_seed=[] if joint_seed is None else joint_seed
        self.program=dict(seed.program,route='fk_projection',seed_route=seed.program['route'],
            numerical_model='constrained_joint_motion_then_explicit_camera_target')
        self.return_s=None
        self.values=np.zeros(6)
        self.envelope=SimpleNamespace(end_s=self.duration,decision_id='fk-projection',
            blend_s=self.duration-self.activation_s,terminal='audited_soft_intent',bounds={})
        self.translation=CubicSpline(geometry['times'],geometry['positions'],
            bc_type=((1,np.asarray(geometry['start_velocity'])),(1,np.zeros(3))))
        self.rotation=RotationSpline(geometry['times'],Rotation.from_quat(geometry['quaternions']))

    @classmethod
    def from_lift(cls, seed, report):
        if not report.get('feasible') or not report.get('camera_projected_for_generation'):
            raise ValueError('projection requires a complete constrained robot motion')
        trace=[r for r in report['trace'] if seed.activation_s+1e-8<r['t']<seed.duration-1e-8]
        p,r,v=seed.previous.sample(seed.activation_s)
        endpoint=min(report['trace'],key=lambda row:abs(row['t']-seed.duration))
        return cls(seed,dict(times=[seed.activation_s]+[row['t'] for row in trace]+[seed.duration],
            positions=[p.tolist()]+[row['position'] for row in trace]+[endpoint['position']],
            quaternions=[Rotation.from_matrix(r).as_quat().tolist()]+[row['quaternion_xyzw'] for row in trace]+[endpoint['quaternion_xyzw']],
            start_velocity=v[:3].tolist()),report['trace'])

    def progress_time(self,t):
        return self.seed.progress_time(t)

    def pose(self,t):
        return self.sample(t)[:2]

    def poses_at(self,times):
        from .intent_paths import poses_at
        times=np.asarray(times,float);before=times<=self.activation_s
        p=np.empty((len(times),3));r=np.empty((len(times),3,3))
        p[before],r[before]=poses_at(self.previous,times[before])
        if np.any(~before):
            at=np.minimum(times[~before],self.duration)
            p[~before],r[~before]=self.translation(at),self.rotation(at).as_matrix()
        return p,r

    def sample(self,t):
        if t<=self.activation_s:
            return self.previous.sample(t)
        at=min(t,self.duration)
        p,r=self.translation(at),self.rotation(at).as_matrix()
        twist=np.r_[self.translation(at,1),r@self.rotation(at,1)] if t<self.duration else np.zeros(6)
        return p,r,twist

    def to_spec(self):
        from .candidates import path_spec
        return dict(kind='fk_camera_program/v1',original_sha256=self.original.sha256,
            activation_s=self.activation_s,duration_s=self.duration,program=self.program,
            seed=path_spec(self.seed),geometry=self.geometry,joint_seed=self.joint_seed)

    @classmethod
    def from_spec(cls,original,spec):
        from .candidates import restore_path
        return cls(restore_path(original,spec['seed']),spec['geometry'],spec.get('joint_seed'))
