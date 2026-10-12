#!/usr/bin/env python3
"""Inspect the real composed USD in Isaac; does not certify obstacle avoidance."""
import argparse
import json
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--output',required=True)
args=p.parse_args();sys.argv=[sys.argv[0]]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.Open(args.stage)
    meshes=[];colliders=[];animated=[];unloaded=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):meshes.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):colliders.append(str(prim.GetPath()))
        if not prim.IsLoaded():unloaded.append(str(prim.GetPath()))
        if any(a.GetNumTimeSamples()>1 for a in prim.GetAttributes()):animated.append(str(prim.GetPath()))
    report=dict(stage=str(Path(args.stage).resolve()),up_axis=str(UsdGeom.GetStageUpAxis(stage)),
                meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),meshes=len(meshes),collision_prims=len(colliders),
                animated_prims=len(animated),unloaded_prims=unloaded,collider_paths=colliders,
                scope='Isaac-composed USD inspection only; no full-room avoidance claim',
                mapping_to_wbc='Y-up (x,y,z) -> Z-up (x,-z,y); yaw0 -> -pi/2',
                required_next_steps=['mesh/voxel obstacle-map ingestion','collision-schema coverage audit',
                                     'room trajectory and initial robot configuration validation'])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('STAGE_INSPECTION '+json.dumps(report),flush=True)
finally:app.close()
