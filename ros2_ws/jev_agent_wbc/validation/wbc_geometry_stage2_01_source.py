from pathlib import Path
import json,hashlib
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from jev_agent_wbc.wbc_physical import MeasuredColliderGeometry
from jev_agent_wbc.robot_sensor_rig import load_config,optical_transform
p=Path.cwd()
configs={
 'Robot_Husky_UR5':dict(model='assets/husky_ur5_v2.usda',key='husky',base='husky_base_link',first='ur5_shoulder_pan_joint',sensors='sensors.json'),
 'Robot_Ridgeback_UR5':dict(model='assets/Isaac/Robots/Clearpath/RidgebackUr/ridgeback_ur5.usd',key='ridgeback',base='base_link',first='ur_arm_shoulder_pan_joint',sensors='../config/robot_sensor_mounts.json'),
 'Robot_TIAGo':dict(model='assets/usd/tiago.usd',key='tiago',base='base_link',first='torso_lift_joint',sensors='sensors.json'),
 'Robot_G1':dict(model='assets/G1/g1.usd',key='g1',base='pelvis',first='waist_yaw_joint',sensors='../config/robot_sensor_mounts.json'),
 'Robot_Go2':dict(model='vendor/LeggedManip_Lab/source/LeggedManip_Lab/LeggedManip_Lab/assets/go2_piper/go2_piper.usd',key='go2',base='base',first='joint1',sensors='../config/robot_sensor_mounts.json'),
}
results=[]
for name,config in configs.items():
 model=p/name/config['model'];stage=Usd.Stage.Open(str(model));root=stage.GetDefaultPrim()
 if not root: raise RuntimeError('missing model default root')
 prims=list(Usd.PrimRange(root,Usd.TraverseInstanceProxies()));scale=UsdGeom.GetStageMetersPerUnit(stage)
 bodies={x.GetName():x for x in prims if x.HasAPI(UsdPhysics.RigidBodyAPI)}
 graph={};first=None
 for prim in prims:
  if prim.IsA(UsdPhysics.Joint):
   j=UsdPhysics.Joint(prim);a=j.GetBody0Rel().GetTargets();b=j.GetBody1Rel().GetTargets()
   if a and b:graph.setdefault(str(a[0]),set()).add(str(b[0]))
   if prim.GetName()==config['first'] and b:first=str(b[0])
 if first is None: raise RuntimeError(name+' missing first controlled joint')
 descendants={first};pending=[first]
 while pending:
  for child in graph.get(pending.pop(),set()):
   if child not in descendants:descendants.add(child);pending.append(child)
 manip={x.GetName() for x in bodies.values() if str(x.GetPath()) in descendants}
 geometry=MeasuredColliderGeometry.from_stage(stage,root.GetPath(),manip)
 cache=UsdGeom.XformCache();poses={}
 for body,prim in bodies.items():
  m=np.asarray(cache.GetLocalToWorldTransform(prim)).T.copy();m[:3,3]*=scale
  u,_,v=np.linalg.svd(m[:3,:3]);m[:3,:3]=u@np.diag([1,1,np.linalg.det(u@v)])@v
  poses[body]=m
 if config['base'] not in poses:config['base']=next(n for n in poses if n.lower() in ['base_link','trunk'])
 sensors,_=load_config(config['key'],p/name/config['sensors']);camera=poses[sensors['camera']['parent']]@optical_transform(sensors['camera'])
 obs=geometry.observe(0.,poses[config['base']],camera,poses)
 bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],useExtentsHint=False,ignoreVisibility=True)
 containment=[]
 for shape in geometry.shape_world_bounds(poses):
  actual=bbox.ComputeWorldBound(stage.GetPrimAtPath(shape['name'])).ComputeAlignedRange()
  low=np.asarray(actual.GetMin())*scale;high=np.asarray(actual.GetMax())*scale
  containment.append(bool(np.all(low>=shape['low']-1e-8) and np.all(high<=shape['high']+1e-8)))
 results.append(dict(robot=name,asset=str(model),asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
  shape_count=len(geometry.shapes),measured_body_count=len(geometry.body_names),all_authored_collision_bounds_contained=all(containment),
  manipulator_bodies=sorted(manip),representation_parts=list(obs.occupancy),observation=obs.to_dict(),geometry=geometry.manifest()))
 print(name,'colliders',len(geometry.shapes),'contained',all(containment))
out=p/'validation/wbc_geometry_stage2_01.json';source=out.with_name(out.stem+'_source.py')
with source.open('xb') as f:f.write(Path(__file__).read_bytes())
with out.open('x') as f:f.write(json.dumps(dict(schema='wbc_geometry_static_audit/v1',scope='authored geometry containment only; real dynamic pose feedback is validated in each physics loop',all_passed=all(x['all_authored_collision_bounds_contained'] for x in results),robots=results,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')
