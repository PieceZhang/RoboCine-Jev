#!/usr/bin/env python3
"""Inspect the pinned source USD without Kit or GPU; never edits the source."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def usd_bootstrap():
    site = Path(sys.executable).resolve().parents[1] / 'lib/python3.11/site-packages/isaacsim'
    usd = next((site / 'extscache').glob('omni.usd.libs-*'))
    if not os.environ.get('HUSKY_USD_READY'):
        env = dict(os.environ, HUSKY_USD_READY='1')
        env['LD_LIBRARY_PATH'] = ':'.join([str(Path(sys.executable).resolve().parents[1] / 'lib'), str(usd / 'bin'), str(site / 'kit'), env.get('LD_LIBRARY_PATH', '')])
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    sys.path.insert(0, str(usd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/source_inspection.json')
    args = parser.parse_args()
    usd_bootstrap()
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(ROOT / 'assets/source/husky_ur5.usd'))
    result = dict(default_prim=str(stage.GetDefaultPrim().GetPath()), meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage), up_axis=str(UsdGeom.GetStageUpAxis(stage)), roots=[], bodies=[], joints=[], graphs=[], collisions=[], root_children=[dict(path=str(p.GetPath()), type=p.GetTypeName()) for p in stage.GetPseudoRoot().GetChildren()], layer_assets=stage.GetRootLayer().GetExternalAssetDependencies())
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            result['roots'].append(str(prim.GetPath()))
        if 'Graph' in prim.GetTypeName():
            result['graphs'].append(dict(path=str(prim.GetPath()), type=prim.GetTypeName()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            b = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            record = dict(path=str(prim.GetPath()), type=prim.GetTypeName(), enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(), min=list(b.GetMin()), max=list(b.GetMax()), empty=b.IsEmpty(), shape_attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName() in ['axis','radius','height','extent','size'] or a.GetName().startswith('xformOp:')})
            if prim.IsA(UsdGeom.Mesh):
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if points:
                    record['point_bounds_local'] = dict(min=[min(p[i] for p in points) for i in range(3)], max=[max(p[i] for p in points) for i in range(3)])
            result['collisions'].append(record)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            t = cache.GetLocalToWorldTransform(prim)
            result['bodies'].append(dict(path=str(prim.GetPath()), xyz=list(t.ExtractTranslation()), matrix=[list(row) for row in t], physical_attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(('physics:', 'physxArticulation:'))}))
        if prim.IsA(UsdPhysics.Joint):
            result['joints'].append(dict(path=str(prim.GetPath()), type=prim.GetTypeName(), attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(('physics:', 'drive:'))}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
