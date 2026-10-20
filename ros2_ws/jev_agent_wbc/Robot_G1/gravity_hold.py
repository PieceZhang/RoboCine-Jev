"""Prototype measured-state G1 gravity, using only NumPy and URDF XML.

This is the static gravity of the existing calibrated30-body point-mass model.
It does not change PhysX masses, reconstruct a solver packet, produce velocities,
or authorize a trajectory/lease. A caller must supply the current measured29q
and pelvis rotation afresh. No torque result is cached. The existing total
PD+gravity effort clamp remains the caller's responsibility, applied once.

There is deliberately no Pinocchio, SciPy, Torch, ROS, or Kit import here.
The original gravity model and calibration remain unchanged.
"""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


UPPER_JOINT_NAMES=('waist_yaw_joint',)+tuple(
    f'{side}_{joint}_joint'for side in ('left','right')for joint in
    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'))


def _array(value,shape,name):
    """Reject boolean/string metadata instead of silently coercing it."""
    if isinstance(value,np.ndarray):
        if value.dtype.kind not in 'fiu':raise ValueError(name+' must be numeric, not boolean or string')
        result=np.asarray(value,dtype=float)
    else:
        source=np.asarray(value,dtype=object)
        if any(isinstance(x,(bool,np.bool_))or not isinstance(x,(int,float,np.integer,np.floating))for x in source.flat):
            raise ValueError(name+' must be numeric, not boolean or string')
        result=np.asarray(value,dtype=float)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError(name+' has invalid shape or nonfinite values')
    return result


def _rotation(value):
    source=np.asarray(value)
    if source.shape==(4,4):
        T=_array(value,(4,4),'measured pelvis transform')
        if np.max(abs(T[3]-[0.,0.,0.,1.]))>1e-9:raise ValueError('invalid homogeneous pelvis transform')
        R=T[:3,:3]
    else:R=_array(value,(3,3),'measured pelvis rotation')
    if np.max(abs(R.T@R-np.eye(3)))>1e-6 or abs(np.linalg.det(R)-1.)>1e-6:
        raise ValueError('proper measured pelvis rotation required')
    return R


def _triple(value):
    if value is None:return np.zeros(3)
    result=np.asarray([float(x)for x in value.split()])
    if result.shape!=(3,)or not np.isfinite(result).all():raise ValueError('invalid URDF origin/axis')
    return result


def _rpy(angles):
    r,p,y=angles;cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
        [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


class GravityHold:
    """G1-specific, gravity-only prototype; not connected to the physical loop.

    ``torque(joints, pelvis_R_or_T)`` returns a dictionary for all29 joints.
    ``upper_torque`` returns only the existing waist-yaw+14arm owners.
    ``torque_vector`` accepts exactly ``joint_names`` order; a PhysX caller must
    prepare its named permutation once, never assume its interleaved order.
    Freshness is an acquisition/caller property, not inferred from q values.
    """
    def __init__(self,urdf,metadata):
        urdf=Path(urdf);content=urdf.read_bytes()
        if isinstance(metadata,(str,Path)):metadata=json.loads(Path(metadata).read_text())
        if (not isinstance(metadata,Mapping)or metadata.get('schema')!='g1_actual_gravity_only/v1'
                or metadata.get('gravity_only')is not True or metadata.get('rotational_inertia_used')is not False
                or metadata.get('physical_parameters_changed')is not False
                or hashlib.sha256(content).hexdigest()!=metadata.get('source_urdf_sha256')):
            raise ValueError('existing actual gravity metadata must bind the unchanged URDF')
        tree=ET.fromstring(content);links={x.get('name')for x in tree.findall('link')}
        all_joints=tree.findall('joint');children={x.find('child').get('link')for x in all_joints}
        if links-children!={'pelvis'}:raise ValueError('expected original fixed-base G1 pelvis topology')
        joints=[x for x in all_joints if x.get('type')!='fixed']
        if len(joints)!=29 or any(x.get('type')not in ('revolute','continuous')for x in joints):
            raise ValueError('expected original29 revolute G1 joints')
        self.joint_names=tuple(x.get('name')for x in joints)
        self._joint_set=set(self.joint_names)
        if (len(self._joint_set)!=29 or not set(UPPER_JOINT_NAMES)<=self._joint_set
                or set(metadata.get('calibration_joint_positions',{}))!=self._joint_set):
            raise ValueError('complete original29/camera-manipulator15 joint names required')
        self.upper_joint_names=UPPER_JOINT_NAMES
        self._upper_indices=np.array([self.joint_names.index(n)for n in UPPER_JOINT_NAMES])
        bodies=('pelvis',)+tuple(x.find('child').get('link')for x in joints)
        actual=metadata.get('bodies',{})
        if len(set(bodies))!=30 or set(actual)!=set(bodies):
            raise ValueError('all30 actual bodies must be root or original active-joint children')
        body_indices={n:i for i,n in enumerate(bodies)}
        parents=[];translations=[];rotations=[];axes=[]
        for joint in joints:
            parent=joint.find('parent').get('link')
            if parent not in body_indices:raise ValueError('unreviewed fixed intermediary in actual G1 topology')
            parents.append(body_indices[parent]);origin=joint.find('origin');axis=joint.find('axis')
            translations.append(_triple(origin.get('xyz')if origin is not None else None))
            rotations.append(_rpy(_triple(origin.get('rpy')if origin is not None else None)))
            direction=_triple(axis.get('xyz')if axis is not None else '1 0 0')
            norm=float(np.linalg.norm(direction))
            if abs(norm-1.)>1e-12:raise ValueError('original normalized revolute axes required')
            axes.append(direction/norm)
        self._parents=np.r_[0,parents].astype(int)
        self._xyz=np.asarray(translations);self._origin_R=np.asarray(rotations);self._axes=np.asarray(axes)
        self._depth_groups=[];known={0};remaining=set(range(1,30))
        while remaining:
            group=sorted(i for i in remaining if self._parents[i]in known)
            if not group:raise ValueError('disconnected or cyclic actual gravity body tree')
            self._depth_groups.append(np.asarray(group));known.update(group);remaining.difference_update(group)
        masses=[];centers=[]
        for index,name in enumerate(bodies):
            body=actual[name];mass=_array([body['mass_kg']],(1,),'actual mass')[0]
            if mass<=0:raise ValueError('positive actual point masses required')
            expected='universe'if index==0 else self.joint_names[index-1]
            if body.get('parent_joint')!=expected:raise ValueError('actual body/active joint parent mismatch')
            offset=_array(body['T_pin_body_from_actor'],(4,4),'calibrated actor-to-body offset')
            _rotation(offset)
            center=_array(body['center_actor_m'],(3,),'actor-local COM translation')
            masses.append(mass);centers.append(offset[:3,3]+offset[:3,:3]@center)
        self._masses=np.asarray(masses);self._centers=np.asarray(centers)
        total=_array([metadata.get('total_actual_mass_kg')],(1,),'actual total mass')[0]
        if abs(self._masses.sum()-total)>1e-7:raise ValueError('all30 actual masses must account exactly once')
        # This static incidence matrix sums actual descendant point moments.
        # It excludes the pelvis from every active joint and never imports any
        # old URDF/fixed-hand inertia into the actual gravity calculation.
        ancestor=np.zeros((29,30))
        for body in range(1,30):
            node=body
            while node:
                ancestor[node-1,body]=1.;node=int(self._parents[node])
        self._ancestor=ancestor;self._subtree_mass=ancestor@self._masses
        x,y,z=self._axes.T;zero=np.zeros(29)
        self._skew=np.stack([zero,-z,y,z,zero,-x,-y,x,zero],axis=-1).reshape(29,3,3)
        identity=np.eye(3)
        self._rodrigues_second=self._axes[:,:,None]*self._axes[:,None,:]-identity
        self._identity=identity
        self._manifest=dict(schema='g1_actual_numpy_gravity_hold/v1',prototype_only=True,connected_to_runner=False,
            gravity_only=True,rotational_inertia_used=False,physical_parameters_changed=False,
            source_urdf_sha256=metadata['source_urdf_sha256'],native_dump_sha256=metadata['native_dump_sha256'],
            source_metadata_sha256=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
            actual_mass_kg=float(total),actual_body_count=30,joint_names=list(self.joint_names),
            upper_joint_names=list(UPPER_JOINT_NAMES),gravity_mps2=9.81,
            torque_sign='Compensation +dU/dq in Nm; physical gravity is its negative.',
            caller_contract='Current measured q/pelvisR every call; no stale solver packet, velocity or lease reuse. Apply the existing total PD+g clamp once.')
        for value in (self._parents,self._xyz,self._origin_R,self._axes,self._masses,self._centers,
                      self._ancestor,self._subtree_mass,self._skew,self._rodrigues_second,self._identity,self._upper_indices):
            value.flags.writeable=False

    def _named_q(self,joints):
        if not isinstance(joints,Mapping)or set(joints)!=self._joint_set:
            raise ValueError('current measured named29 joint configuration required')
        return _array([joints[n]for n in self.joint_names],(29,),'measured joint positions')

    def torque_vector(self,positions,pelvis_rotation):
        q=_array(positions,(29,),'measured joint positions');R_world=_rotation(pelvis_rotation)
        # Independent analytic revolute FK, grouped by tree depth. No prior
        # result or trajectory derivative is read or retained between calls.
        own_R=self._identity+np.sin(q)[:,None,None]*self._skew+(1.-np.cos(q))[:,None,None]*self._rodrigues_second
        local_R=self._origin_R@own_R
        rotations=np.empty((30,3,3));rotations[0]=self._identity
        positions_body=np.zeros((30,3))
        for group in self._depth_groups:
            parent=self._parents[group];parent_R=rotations[parent]
            rotations[group]=parent_R@local_R[group-1]
            positions_body[group]=positions_body[parent]+np.einsum('nij,nj->ni',parent_R,self._xyz[group-1])
        centers=positions_body+np.einsum('nij,nj->ni',rotations,self._centers)
        first_moment=self._ancestor@(self._masses[:,None]*centers)
        levers=first_moment-self._subtree_mass[:,None]*positions_body[1:]
        world_up_in_pelvis=R_world.T@np.array([0.,0.,9.81])
        moments=np.cross(levers,world_up_in_pelvis)
        axes=np.einsum('nij,nj->ni',rotations[1:],self._axes)
        return np.einsum('ni,ni->n',axes,moments)

    def torque(self,joints,pelvis_rotation):
        values=self.torque_vector(self._named_q(joints),pelvis_rotation)
        return dict(zip(self.joint_names,values.tolist()))

    def upper_torque(self,joints,pelvis_rotation):
        values=self.torque_vector(self._named_q(joints),pelvis_rotation)
        return dict(zip(UPPER_JOINT_NAMES,values[self._upper_indices].tolist()))

    def manifest(self):return json.loads(json.dumps(self._manifest))
