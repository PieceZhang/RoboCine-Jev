#!/usr/bin/env python3
"""Native USD-only gate on a preserved scene; never starts Kit or physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROBOT=Path(__file__).resolve().parents[2]
PACKAGE=ROBOT.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.child:
        cache=Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/isaacsim/extscache'
        usd=next(cache.glob('omni.usd.libs-*'))
        env=dict(os.environ);env['PYTHONPATH']=str(usd)+os.pathsep+env.get('PYTHONPATH','')
        env['LD_LIBRARY_PATH']=str(Path(sys.prefix)/'lib')+os.pathsep+str(usd/'bin')+os.pathsep+env.get('LD_LIBRARY_PATH','')
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child'],env)
    sys.path.insert(0,str(PACKAGE/'scripts'))
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf
    from ocs2_wheeled_scene import repair_single_ground,audit_single_ground
    args.output.mkdir(parents=True,exist_ok=False)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source_hash=digest(args.scene);stage=Usd.Stage.Open(str(args.scene));root='/World/Ground'
    active=[str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    material=UsdShade.Material.Get(stage,'/World/GroundMaterial')
    floor=float(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root+'/collisionPlane'))[3][2])
    def robot_spec():
        layer=Sdf.Layer.CreateAnonymous();Sdf.CreatePrimInLayer(layer,'/World');Sdf.CopySpec(stage.GetRootLayer(),'/World/Robot',layer,'/World/Robot')
        return hashlib.sha256(layer.ExportToString().encode()).hexdigest()
    robot_before=robot_spec()
    try:audit_single_ground(stage,root,material,floor);error_before=None
    except ValueError as exc:error_before=str(exc)
    report=repair_single_ground(stage,root,material,floor);robot_after=robot_spec()
    scene=args.output/'repaired_scene.usda';stage.Export(str(scene))
    reopened=Usd.Stage.Open(str(scene));reopened_report=audit_single_ground(reopened,root,UsdShade.Material.Get(reopened,'/World/GroundMaterial'),floor)
    result=dict(scope='Native USD CPU construction gate on preserved actual scene; no PhysX run and no readiness pass',passed=bool(error_before and len(active)==2 and robot_before==robot_after and source_hash==digest(args.scene) and reopened_report['passed']),original_active_ground_colliders=active,original_gate_rejection=error_before,repaired=report,robot_spec_before_sha256=robot_before,robot_spec_after_sha256=robot_after,original_scene_unchanged=source_hash==digest(args.scene),sources_sha256={str(args.scene.resolve()):source_hash,str(Path(__file__).resolve()):digest(__file__),str(PACKAGE/'scripts/ocs2_wheeled_scene.py'):digest(PACKAGE/'scripts/ocs2_wheeled_scene.py')},exported_scene_sha256=digest(scene))
    (args.output/'single_ground_cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
