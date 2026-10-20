#!/usr/bin/env python3
"""Read-only contact instrumentation around the unchanged native physics runner.

Enables contact reports, never modifies collision offsets, pairs, drives or
commands. This is a diagnostic run, not a new readiness acceptance entry.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROBOT=Path(__file__).resolve().parents[2]
PACKAGE=ROBOT.parent
spec=importlib.util.spec_from_file_location('native_contact_diagnostic_runner',PACKAGE/'scripts/ocs2_wheeled_runtime.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
original_run=runner.run


def run(args):
    import numpy as np
    from pxr import Usd,UsdGeom,UsdUtils,UsdPhysics,PhysxSchema,PhysicsSchemaTools
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx import get_physx_simulation_interface,get_physx_cooking_interface
    snapshot=args.output/'sources/contact_diagnostic.py';shutil.copy2(__file__,snapshot)
    provenance_path=args.output/'provenance.json';provenance=json.loads(provenance_path.read_text())
    provenance['contact_diagnostic']=dict(source='sources/contact_diagnostic.py',sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),physical_parameters_changed=False,positive_readiness_claim=False)
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    original_reset=World.reset;subscription=None;stream=(args.output/'contacts.jsonl').open('w',buffering=1)
    def reset(world,*positional,**kwargs):
        nonlocal subscription
        stage=world.stage
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        def contact(headers,data):
            for header in headers:
                points=data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]
                impulses=[list(map(float,p.impulse)) for p in points]
                actors=[str(PhysicsSchemaTools.intToSdfPath(header.actor0)),str(PhysicsSchemaTools.intToSdfPath(header.actor1))]
                # Keep all self contacts and material impulses, without paying
                # for a verbose wheel-ground report at every quiet tick.
                if (all(p.startswith('/World/Robot/') for p in actors)
                        or any(np.linalg.norm(v)>2. for v in impulses)):
                    stream.write(json.dumps(dict(reported_world_time_s=float(world.current_time),timestamp_scope='World.current_time within PhysX contact callback; may precede the after-step sample by one 10ms tick',actors=actors,colliders=[str(PhysicsSchemaTools.intToSdfPath(header.collider0)),str(PhysicsSchemaTools.intToSdfPath(header.collider1))],points=[dict(position=list(map(float,p.position)),normal=list(map(float,p.normal)),impulse=list(map(float,p.impulse)),separation=float(p.separation)) for p in points]))+'\n')
        subscription=get_physx_simulation_interface().subscribe_contact_report_events(contact)
        result=original_reset(world,*positional,**kwargs)
        sim=SimulationManager.get_physics_sim_view();properties=[]
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
            body=sim.create_rigid_body_view(str(prim.GetPath()))
            if not body.count:continue
            properties.append(dict(body=str(prim.GetPath()),count=int(body.count),max_shapes=int(body.max_shapes),contact_offsets=np.asarray(body.get_contact_offsets()).tolist(),rest_offsets=np.asarray(body.get_rest_offsets()).tolist(),material_properties=np.asarray(body.get_material_properties()).tolist()))
        collision_properties=[];cooked=[];stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            api=PhysxSchema.PhysxCollisionAPI(prim)
            raw_contact=api.GetContactOffsetAttr().Get();raw_rest=api.GetRestOffsetAttr().Get()
            collision_properties.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),mesh_approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),contact_offset_usd=raw_contact if raw_contact is not None and np.isfinite(raw_contact) else 'automatic',rest_offset_usd=raw_rest if raw_rest is not None and np.isfinite(raw_rest) else 'automatic'))
            if prim.IsA(UsdGeom.Mesh) and collision_properties[-1]['enabled']:
                def ready(status,convexes,path=str(prim.GetPath()),mesh=prim):
                    cooked.append(dict(path=path,status=str(status),coordinate_scope='Native cooking API representation; raw returned coordinates retained alongside USD local points and transform for explicit frame/scale audit',source_mesh_points=np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(),float).tolist(),T_world_usd_prim=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh)).T.tolist(),convexes=[dict(vertices=[list(map(float,v)) for v in c.vertices],indices=list(c.indices),polygons=[dict(index_base=p.index_base,num_vertices=p.num_vertices,plane=list(map(float,p.plane))) for p in c.polygons]) for c in convexes]))
                # Synchronous at reset, before the motion timer and first
                # commanded step. No cooking wait pauses a running trajectory.
                get_physx_cooking_interface().request_convex_collision_representation(stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),run_asynchronously=False,on_result=ready)
        (args.output/'resolved_shape_properties.json').write_text(json.dumps(dict(scope='Actual PhysX tensor readback after reset; reports enabled only, no contact/material modifications',bodies=properties,usd_collision_properties=collision_properties),indent=2)+'\n')
        (args.output/'cooked_collision_representation.json').write_text(json.dumps(dict(scope='Actual native cooking API query during initialization, not the raw USD mesh convex hull and not contact skin; no modification of cooking parameters',shapes=cooked),indent=2)+'\n')
        return result
    World.reset=reset
    try:return original_run(args)
    finally:World.reset=original_reset;subscription=None;stream.close()


runner.run=run
if __name__=='__main__':raise SystemExit(runner.main())
