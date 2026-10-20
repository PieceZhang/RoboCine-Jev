"""Gravity-only Pin model from actual PhysX masses and actor-local COMs.

Rotational inertia is deliberately unused: this point-mass model must never be
used for inverse dynamics, IK, collision geometry, or simulated physics.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np


def rigid(T):
    T=np.asarray(T,float)
    if (T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1])
            or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-6)):
        raise ValueError('proper finite body transform required')
    return T


def configuration(model,joints):
    import pinocchio as pin
    q=pin.neutral(model)
    for name in list(model.names)[1:]:
        if name not in joints or not np.isfinite(joints[name]):raise ValueError('finite measured joint missing: '+name)
        q[model.joints[model.getJointId(name)].idx_q]=joints[name]
    return q


def calibrate(urdf,dump,native_dump_sha256):
    import pinocchio as pin
    from scipy.spatial.transform import Rotation
    model=pin.buildModelFromUrdf(str(urdf));names=dump['body_names']
    if (dump.get('schema')!='g1_default_upper_startup_physx/v1'
            or len(names)!=30 or len(set(names))!=len(names)):
        raise ValueError('expected complete actual 30-body height-model measurement')
    if dump.get('conventions',{}).get('coms_axes')!='articulation, body_names, xyz/qx/qy/qz/qw; COM principal-axes frame relative to rigid-body-prim actor frame':
        raise ValueError('actor-local measured COM translation convention is not established')
    masses=np.asarray(dump['masses']['values'],float)[0]
    centers=np.asarray(dump['coms']['values'],float)[0,:,:3]
    actors=np.asarray(dump['actor_world_poses_xyzw']['values'],float)[0]
    if (masses.shape!=(len(names),) or centers.shape!=(len(names),3) or actors.shape!=(len(names),7)
            or not np.isfinite(masses).all() or not np.isfinite(centers).all() or not np.isfinite(actors).all()
            or np.any(masses<=0)):
        raise ValueError('actual named masses/COMs/actor poses are incomplete')
    poses=np.broadcast_to(np.eye(4),(len(names),4,4)).copy()
    poses[:,:3,:3]=Rotation.from_quat(actors[:,3:]).as_matrix();poses[:,:3,3]=actors[:,:3]
    joint_names=dump['joint_names'];positions=np.asarray(dump['joint_positions']['values'],float)
    if (len(set(joint_names))!=29 or set(joint_names)!=set(list(model.names)[1:])
            or positions.shape!=(1,29) or not np.isfinite(positions).all()
            or not np.allclose(np.linalg.norm(actors[:,3:],axis=1),1.,rtol=0,atol=1e-5)):
        raise ValueError('actual joint configuration or actor quaternion is incomplete')
    joints=dict(zip(joint_names,positions[0]))
    q=configuration(model,joints);data=model.createData();pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
    pelvis=poses[names.index('pelvis')];bodies={}
    for i,name in enumerate(names):
        fid=model.getFrameId(name)
        if fid>=model.nframes:raise ValueError('actual body not present in original model: '+name)
        offset=np.linalg.inv(pelvis@data.oMf[fid].np)@poses[i]
        bodies[name]=dict(mass_kg=float(masses[i]),center_actor_m=centers[i].tolist(),
            T_pin_body_from_actor=rigid(offset).tolist(),parent_joint=model.names[model.frames[fid].parent])
    return dict(schema='g1_actual_gravity_only/v1',gravity_only=True,rotational_inertia_used=False,
        physical_parameters_changed=False,source_urdf_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
        source_usd_sha256=dump['source_usd']['sha256'],native_dump_sha256=native_dump_sha256,
        total_actual_mass_kg=float(masses.sum()),bodies=bodies,
        calibration_joint_positions={n:float(v) for n,v in joints.items()},T_world_pelvis=pelvis.tolist(),
        interpretation='Actual point masses and actor-local COM translations only. Principal COM quaternion/inertia axes are not used. All original fused masses are cleared to avoid fixed-link duplication.')


class ActualGravity:
    def __init__(self,urdf,metadata):
        import pinocchio as pin
        if (metadata.get('schema')!='g1_actual_gravity_only/v1' or metadata.get('gravity_only') is not True
                or metadata.get('rotational_inertia_used') is not False
                or metadata.get('physical_parameters_changed') is not False
                or hashlib.sha256(Path(urdf).read_bytes()).hexdigest()!=metadata['source_urdf_sha256']):
            raise ValueError('gravity-only metadata does not match the pinned URDF')
        self.model=pin.buildModelFromUrdf(str(urdf));self.metadata=copy.deepcopy(metadata)
        if len(metadata['bodies'])!=30:raise ValueError('actual height model requires all 30 bodies')
        for i in range(self.model.njoints):self.model.inertias[i]=pin.Inertia.Zero()
        total=0.
        for name,body in metadata['bodies'].items():
            fid=self.model.getFrameId(name)
            if fid>=self.model.nframes:raise ValueError('unknown calibrated body')
            frame=self.model.frames[fid];mass=float(body['mass_kg']);center=np.asarray(body['center_actor_m'],float)
            if (mass<=0 or not np.isfinite(mass) or center.shape!=(3,) or not np.isfinite(center).all()
                    or self.model.names[frame.parent]!=body['parent_joint']):
                raise ValueError('invalid mass/COM/fixed-body parent')
            placement=frame.placement*pin.SE3(rigid(body['T_pin_body_from_actor']))
            # Zero rotational inertia is intentional and harmless for gravity.
            point=placement.act(pin.Inertia(mass,center,np.zeros((3,3))))
            self.model.inertias[frame.parent]+=point;total+=mass
        if not np.isclose(total,metadata['total_actual_mass_kg'],rtol=0,atol=1e-7):
            raise ValueError('gravity mass accounting mismatch')
        self.data=self.model.createData()

    def torque(self,joints,T_world_pelvis):
        import pinocchio as pin
        q=configuration(self.model,joints);pelvis=rigid(T_world_pelvis)
        self.model.gravity.linear=pelvis[:3,:3].T@np.array([0.,0.,-9.81])
        values=pin.computeGeneralizedGravity(self.model,self.data,q)
        return {n:float(values[self.model.joints[self.model.getJointId(n)].idx_v]) for n in list(self.model.names)[1:]}

    def verify_native(self,body_names,masses,centers_actor,body_poses,joints,T_world_pelvis):
        """Fail closed if the current native plant differs from calibration.

        Call once before physical time. Body poses and COM translations must be
        actual actor-frame native measurements; no authored USD pose substitutes.
        """
        import pinocchio as pin
        names=list(body_names);masses=np.asarray(masses,float);centers=np.asarray(centers_actor,float)
        if (len(set(names))!=len(names) or set(names)!=set(self.metadata['bodies'])
                or masses.shape!=(len(names),) or centers.shape!=(len(names),3)
                or not np.isfinite(masses).all() or not np.isfinite(centers).all()
                or set(body_poses)!=set(names)):
            raise ValueError('native gravity verification requires all named physical bodies')
        q=configuration(self.model,joints);data=self.model.createData()
        pin.forwardKinematics(self.model,data,q);pin.updateFramePlacements(self.model,data)
        pelvis=rigid(T_world_pelvis);mass_errors=[];center_errors=[];pose_errors=[]
        for i,name in enumerate(names):
            body=self.metadata['bodies'][name]
            predicted=pelvis@data.oMf[self.model.getFrameId(name)].np@np.asarray(body['T_pin_body_from_actor'])
            measured=rigid(body_poses[name])
            mass_errors.append(abs(masses[i]-body['mass_kg']))
            center_errors.append(float(np.linalg.norm(centers[i]-body['center_actor_m'])))
            pose_errors.append(float(np.max(np.abs(predicted-measured))))
        report=dict(body_count=len(names),measured_total_mass_kg=float(masses.sum()),
            mass_max_error_kg=float(max(mass_errors)),actor_com_max_error_m=float(max(center_errors)),
            body_fk_max_component_error=float(max(pose_errors)),
            gravity_only=True,physical_parameters_changed=False,rotational_inertia_used=False)
        report['passed']=bool(report['mass_max_error_kg']<=1e-5 and report['actor_com_max_error_m']<=1e-6
                              and report['body_fk_max_component_error']<=2e-5)
        if not report['passed']:raise ValueError('current native mass/COM/FK mismatch: '+str(report))
        return report

    def potential(self,joints,T_world_pelvis):
        import pinocchio as pin
        q=configuration(self.model,joints);data=self.model.createData()
        pin.forwardKinematics(self.model,data,q);pin.updateFramePlacements(self.model,data)
        pelvis=rigid(T_world_pelvis);energy=0.
        for name,body in self.metadata['bodies'].items():
            T=pelvis@data.oMf[self.model.getFrameId(name)].np@np.asarray(body['T_pin_body_from_actor'])
            position=T[:3,:3]@body['center_actor_m']+T[:3,3]
            energy+=body['mass_kg']*9.81*position[2]
        return float(energy)
