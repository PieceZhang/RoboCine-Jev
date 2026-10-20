"""Source-URDF kinematics for the explicitly limited Giskard solver fixture.

This is an adapter task model, not an alternative optimizer or a PhysX model.
The differential base has forward speed and yaw rate, never lateral speed.
"""
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from wbc_backend import ARM_JOINTS, CAMERA_FRAME, ROOT, model_urdf


def transform(xyz=(0.,0.,0.), rpy=(0.,0.,0.)):
    T = np.eye(4)
    T[:3,:3] = Rotation.from_euler('xyz',rpy).as_matrix()
    T[:3,3] = xyz
    return T


class TiagoKinematics:
    def __init__(self):
        document = ET.fromstring(model_urdf())
        self.profile = json.loads((ROOT/'robot.json').read_text())
        self.joints = {j.get('name'):j for j in document.findall('joint')}
        by_child = {j.find('child').get('link'):j for j in self.joints.values()}
        chain, link = [], CAMERA_FRAME
        while link != self.profile['base_link']:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        self.chain = tuple(reversed(chain))
        active = tuple(j.get('name') for j in self.chain if j.get('type')!='fixed')
        if active != ARM_JOINTS:
            raise ValueError(f'Unexpected source camera chain: {active}')
        self.home = np.array([j['home'] for j in self.profile['joints']])
        self.lower = np.array([j['lower'] for j in self.profile['joints']])
        self.upper = np.array([j['upper'] for j in self.profile['joints']])
        self.velocity = np.array([j['velocity'] for j in self.profile['joints']])
        self.wheel_velocity = min(float(self.joints[n].find('limit').get('velocity'))
            for n in self.profile['left_wheels']+self.profile['right_wheels'])

    def pose_and_jacobian(self, T_world_base, q):
        """World spatial [linear;angular] 6x10 Jacobian: v,yaw,torso,arm1..7."""
        T_world_base, q = np.asarray(T_world_base),np.asarray(q)
        if q.shape!=(8,) or T_world_base.shape!=(4,4):
            raise ValueError('Full SE(3) base pose and eight torso/arm coordinates required')
        T = T_world_base.copy()
        axes,origins,types = [],[],[]
        index = 0
        for joint in self.chain:
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            T = T@transform(xyz,rpy)
            kind = joint.get('type')
            if kind=='fixed':
                continue
            axis_element = joint.find('axis')
            axis = np.fromstring(axis_element.get('xyz','1 0 0'),sep=' ') if axis_element is not None else np.array([1.,0.,0.])
            axis = axis/np.linalg.norm(axis)
            axes.append(T[:3,:3]@axis)
            origins.append(T[:3,3].copy())
            types.append(kind)
            motion = np.eye(4)
            if kind=='prismatic':
                motion[:3,3] = axis*q[index]
            elif kind in ('revolute','continuous'):
                motion[:3,:3] = Rotation.from_rotvec(axis*q[index]).as_matrix()
            else:
                raise ValueError(f'Unsupported source joint type: {kind}')
            T = T@motion
            index += 1
        J = np.zeros((6,10))
        up = T_world_base[:3,2]
        J[:3,0] = T_world_base[:3,0]
        J[:3,1] = np.cross(up,T[:3,3]-T_world_base[:3,3])
        J[3:,1] = up
        for i,(axis,origin,kind) in enumerate(zip(axes,origins,types),2):
            J[:3,i] = axis if kind=='prismatic' else np.cross(axis,T[:3,3]-origin)
            J[3:,i] = 0. if kind=='prismatic' else axis
        return T,J

    @staticmethod
    def integrate(T_world_base,q,rates,dt):
        """Exact planar differential arc, retaining supplied base height/tilt."""
        rates = np.asarray(rates)
        if rates.shape!=(10,) or dt<=0:
            raise ValueError('Ten actuator rates and positive dt required')
        v,yaw = rates[:2]
        angle = yaw*dt
        local = np.array([v*dt,0.,0.]) if abs(yaw)<1e-10 else np.array([
            v*np.sin(angle)/yaw,v*(1.-np.cos(angle))/yaw,0.])
        T = np.asarray(T_world_base).copy()
        T[:3,3] += T[:3,:3]@local
        T[:3,:3] = T[:3,:3]@Rotation.from_rotvec([0.,0.,angle]).as_matrix()
        return T,np.asarray(q)+rates[2:]*dt


def pose_error(target,current):
    return np.r_[target[:3,3]-current[:3,3],
        (Rotation.from_matrix(target[:3,:3])*Rotation.from_matrix(current[:3,:3]).inv()).as_rotvec()]
