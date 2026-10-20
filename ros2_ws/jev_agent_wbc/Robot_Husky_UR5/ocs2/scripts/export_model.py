#!/usr/bin/env python3
"""Extract a POE arm model from exact physical USD joint axes, without Kit."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    site = Path(sys.executable).resolve().parents[1] / 'lib/python3.11/site-packages/isaacsim'
    usd = next((site/'extscache').glob('omni.usd.libs-*'))
    if not os.environ.get('JEV_OCS2_USD_READY'):
        env = dict(os.environ, JEV_OCS2_USD_READY='1')
        env['LD_LIBRARY_PATH'] = ':'.join([str(Path(sys.executable).resolve().parents[1]/'lib'),str(usd/'bin'),str(site/'kit'),env.get('LD_LIBRARY_PATH','')])
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    sys.path.insert(0,str(usd))
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdGeom, UsdPhysics
    spec = json.loads((ROOT/'ocs2/backend.json').read_text())
    model = ROOT/spec['usd']
    stage = Usd.Stage.Open(str(model))
    prims = list(Usd.PrimRange(stage.GetDefaultPrim()))
    cache = UsdGeom.XformCache()
    bodies = {p.GetName():p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    base = np.asarray(cache.GetLocalToWorldTransform(bodies[spec['base_link']])).T
    ee = np.asarray(cache.GetLocalToWorldTransform(bodies[spec['ee_link']])).T
    joint_prims = {p.GetName():p for p in prims if p.IsA(UsdPhysics.Joint)}
    def local(position, quaternion):
        a=np.eye(4);v=quaternion.GetImaginary();a[:3,:3]=Rotation.from_quat([v[0],v[1],v[2],quaternion.GetReal()]).as_matrix();a[:3,3]=position;return a
    joints=[]
    for name in spec['arm_joints']:
        joint=UsdPhysics.RevoluteJoint(joint_prims[name])
        parent=stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0]);child=stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        p=np.asarray(cache.GetLocalToWorldTransform(parent)).T
        c=np.asarray(cache.GetLocalToWorldTransform(child)).T
        j0=p@local(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get())
        j1=c@local(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get())
        axis_index={'X':0,'Y':1,'Z':2}[joint.GetAxisAttr().Get()]
        relative=np.linalg.inv(j0)@j1
        r=Rotation.from_matrix(relative[:3,:3]).as_rotvec();rest=float(r[axis_index])
        assert np.linalg.norm(relative[:3,3])<2e-6 and np.linalg.norm(np.delete(r,axis_index))<2e-5,name
        b=np.linalg.inv(base)@j0
        omega=b[:3,axis_index];v=-np.cross(omega,b[:3,3])
        joints.append(dict(name=name,screw_omega=omega.tolist(),screw_v=v.tolist(),q_at_authored_pose=rest,lower=float(np.deg2rad(joint.GetLowerLimitAttr().Get())),upper=float(np.deg2rad(joint.GetUpperLimitAttr().Get()))))
    source=ROOT/spec['sensor_config'];sensor=json.loads(source.read_text())['robots'][spec['robot_key']]['camera']
    forward=np.array(sensor['forward_parent']);up=np.array(sensor['up_parent']);optical=np.eye(4);optical[:3,:3]=np.column_stack([np.cross(forward,up),-up,forward]);optical[:3,3]=sensor['translation_m']
    parent=np.asarray(cache.GetLocalToWorldTransform(bodies[sensor['parent']])).T
    ee_camera=np.linalg.inv(ee)@parent@optical
    data=dict(schema='usd_poe_arm_model/v1',model=spec['usd'],model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),sensor_config=spec['sensor_config'],sensor_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),base_link=spec['base_link'],ee_link=spec['ee_link'],joints=joints,T_base_ee_at_authored_pose=(np.linalg.inv(base)@ee).tolist(),T_ee_camera_optical=ee_camera.tolist(),T_world_base_at_authored_pose=base.tolist(),T_world_camera_at_authored_pose=(parent@optical).tolist(),scope='Exact USD arm joint kinematics; no collision-distance solver or wheel-contact dynamics.')
    (ROOT/'ocs2/model.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(robot=spec['robot_key'],joint_count=len(joints),minimum_lower_limit=min(j['lower'] for j in joints),camera_offset=ee_camera[:3,3].tolist()),indent=2))


if __name__=='__main__':
    main()
