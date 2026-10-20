#!/usr/bin/env python3
"""CPU/native-USD preparation of exact body transforms and conservative guards."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
site=Path(sys.executable).resolve().parents[1]/'lib/python3.11/site-packages/isaacsim'
usd=next((site/'extscache').glob('omni.usd.libs-*'))
if not os.environ.get('JEV_OCS2_USD_READY'):
    env=dict(os.environ,JEV_OCS2_USD_READY='1')
    env['LD_LIBRARY_PATH']=':'.join([str(Path(sys.executable).resolve().parents[1]/'lib'),str(usd/'bin'),str(site/'kit'),env.get('LD_LIBRARY_PATH','')])
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
sys.path[:0]=[str(usd),str(ROOT.parent),str(ROOT.parent/'scripts')]
import numpy as np
from scipy.linalg import expm
from pxr import Usd,UsdGeom,UsdPhysics
from jev_agent_wbc.wbc_physical import MeasuredColliderGeometry
from ocs2_wheeled_safety import CollisionGuard,add_convex_vertices

spec=json.loads((ROOT/'ocs2/backend.json').read_text());model=json.loads((ROOT/'ocs2/model.json').read_text())
stage=Usd.Stage.Open(str(ROOT/spec['usd']));root=stage.GetDefaultPrim()
bodies={p.GetName():p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
parents={};children={};joints=[]
for prim in Usd.PrimRange(root):
    if not prim.IsA(UsdPhysics.Joint):continue
    joint=UsdPhysics.Joint(prim);a,b=joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()
    if not a or not b:continue
    p,c=stage.GetPrimAtPath(a[0]).GetName(),stage.GetPrimAtPath(b[0]).GetName()
    parents[c]=p;joints.append((p,c,prim.IsA(UsdPhysics.FixedJoint)))
    if prim.GetName() in spec['arm_joints']:children[c]=spec['arm_joints'].index(prim.GetName())+1
def prefix(name):
    seen=set();value=0
    while name not in seen:
        seen.add(name);value=max(value,children.get(name,0))
        if name not in parents:break
        name=parents[name]
    return value
geometry=MeasuredColliderGeometry.from_stage(stage,root.GetPath(),[name for name in bodies if prefix(name)],bodies)
add_convex_vertices(stage,geometry)
guard=CollisionGuard(geometry.shapes,joints)
cache=UsdGeom.XformCache();authored={name:np.asarray(cache.GetLocalToWorldTransform(p)).T for name,p in bodies.items()}
invbase=np.linalg.inv(authored[spec['base_link']]);body_model=[dict(name=name,prefix=prefix(name),T_base_body=(invbase@T).tolist()) for name,T in authored.items()]
prefixes=[np.eye(4)]
for q,j in zip(spec['initial_arm'],model['joints']):
    w=np.asarray(j['screw_omega']);v=np.asarray(j['screw_v']);hat=np.zeros((4,4));hat[:3,:3]=[[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]];hat[:3,3]=v
    prefixes.append(prefixes[-1]@expm(hat*(q-j['q_at_authored_pose'])))
home={b['name']:prefixes[b['prefix']]@np.array(b['T_base_body']) for b in body_model}
distance,pair=guard.check(home)
model.update(body_kinematics=body_model,collision_shapes=geometry.shapes,collision_pairs=guard.pairs)
out=ROOT/'ocs2/evidence/live_model_static.json';out.write_text(json.dumps(model,indent=2)+'\n')
report=dict(robot=spec['robot'],scope='CPU conservative collision-shell guard at original profile home; no physics',shapes=len(geometry.shapes),pairs=len(guard.pairs),minimum_separation_lower_bound_m=distance,closest_pair=pair,passed=bool(distance>=.005))
(ROOT/'ocs2/evidence/live_geometry_static.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
