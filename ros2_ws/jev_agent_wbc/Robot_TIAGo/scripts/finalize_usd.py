#!/usr/bin/env python3
"""Author TIAGo-specific default prim and explicit passive-support material.

No robot pose is controlled here. All subsequent motion uses physical drives.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    python = Path(os.environ.get('ISAAC_PYTHON', '/home/w84384651/miniconda3/envs/env_isaaclab/bin/python'))
    site = python.resolve().parents[1] / 'lib/python3.11/site-packages/isaacsim'
    usd = next((site/'extscache').glob('omni.usd.libs-*'))
    if not os.environ.get('TIAGO_USD_READY'):
        env = dict(os.environ, TIAGO_USD_READY='1')
        env['LD_LIBRARY_PATH'] = ':'.join([str(python.resolve().parents[1]/'lib'), str(usd/'bin'), str(site/'kit'), env.get('LD_LIBRARY_PATH', '')])
        os.execve(str(python), [str(python), __file__], env)
    sys.path.insert(0, str(usd))
    from pxr import Sdf, Usd, UsdPhysics, UsdShade
    path = ROOT / 'assets/usd/tiago.usd'
    stage = Usd.Stage.Open(str(path))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/tiago'))
    articulation = stage.GetPrimAtPath('/tiago/base_link')
    articulation.CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool).Set(True)
    articulation.CreateAttribute('physxArticulation:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(64)
    articulation.CreateAttribute('physxArticulation:solverVelocityIterationCount', Sdf.ValueTypeNames.Int).Set(16)
    filtered, absent = [], []
    for pair in ET.parse(ROOT/'urdf/tiago.srdf').findall('disable_collisions'):
        first, second = pair.attrib['link1'], pair.attrib['link2']
        p0, p1 = stage.GetPrimAtPath('/tiago/'+first), stage.GetPrimAtPath('/tiago/'+second)
        if not p0 or not p1:
            absent.append(pair.attrib)
            continue
        UsdPhysics.FilteredPairsAPI.Apply(p0).CreateFilteredPairsRel().AddTarget(p1.GetPath())
        filtered.append(pair.attrib)
    material = UsdShade.Material.Define(stage, '/tiago/PassiveSupportMaterial')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    # The official URDF fixes its caster joints. Nearly frictionless supports
    # represent a passive caster approximation; this is explicitly reported.
    api.CreateStaticFrictionAttr(.001)
    api.CreateDynamicFrictionAttr(.001)
    api.CreateRestitutionAttr(0.)
    material.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
    material.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('min')
    drive_units = []
    for prim in stage.Traverse():
        for axis in ('angular', 'linear'):
            drive = UsdPhysics.DriveAPI(prim, axis)
            if drive and prim.HasAPI(UsdPhysics.DriveAPI, axis):
                drive.CreateTypeAttr('force')
                drive_units.append(dict(joint=str(prim.GetPath()), axis=axis))
    for name in ('wheel_left_joint', 'wheel_right_joint'):
        joint = stage.GetPrimAtPath('/tiago/joints/'+name)
        # URDF friction is a static resisting torque; PhysX's jointFriction is
        # a load-scaled coefficient. Directly importing PAL's 2.0 locks wheels.
        # Preserve physical effort bounds and velocity damping, omit that
        # incompatible Coulomb coefficient instead of raising actuator limits.
        joint.CreateAttribute('physxJoint:jointFriction', Sdf.ValueTypeNames.Float).Set(0.)
        UsdPhysics.DriveAPI(joint, 'angular').CreateTypeAttr('force')
    bound = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and '/caster_' in str(prim.GetPath()):
            # Imported geometry is instanceable; bind on its rigid-body parent.
            parent = prim
            while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            if parent:
                UsdShade.MaterialBindingAPI.Apply(parent).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
                bound.append(str(parent.GetPath()))
    stage.GetRootLayer().Save()
    report = dict(default_prim='/tiago', passive_support_collision_bodies=sorted(set(bound)),
        support_model='official fixed caster geometry with low-friction support approximation',
        support_static_friction=.001, support_dynamic_friction=.001,
        support_friction_combine_mode='min', wheel_joint_friction_coefficient=0.,
        wheel_friction_note='URDF static torque 2 Nm is not numerically compatible with PhysX load-scaled coefficient; omitted, viscous damping and official effort retained',
        drive_type='force', force_drive_joints=drive_units,
        drive_reason='PD gains use force/torque units; importer acceleration drives scale gains by small link inertia, producing passive drift and insufficient wheel torque',
        enabled_self_collisions=True, self_collision_source='urdf/tiago.srdf (official PAL tiago_moveit_config)',
        self_collision_allowed_pairs=filtered, absent_srdf_link_pairs=absent,
        rigid_bodies=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()),
        usd_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), physics_executed=False)
    (ROOT/'assets/usd/tiago.finalize.json').write_text(json.dumps(report, indent=2)+'\n')
    source = ROOT/'assets/usd'
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix in ('.usd','.usda','.usdc'))
    digest = hashlib.sha256()
    for asset in files:
        digest.update(str(asset.relative_to(source)).encode())
        digest.update(asset.read_bytes())
    closure_id = digest.hexdigest()
    frozen = ROOT/'assets/releases'/closure_id
    if not frozen.exists():
        shutil.copytree(source, frozen)
    closure = dict(schema='tiago_usd_closure/v1', entry='assets/usd/tiago.usd',
        closure_sha256=closure_id, frozen_asset_directory=str(frozen.relative_to(ROOT)),
        source_lock_sha256=hashlib.sha256((ROOT/'SOURCE_LOCK.json').read_bytes()).hexdigest(),
        preparation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_executed=False,
        files=[dict(path=str(asset.relative_to(ROOT)),sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),bytes=asset.stat().st_size,
                    layer_text_sha256=hashlib.sha256(Sdf.Layer.FindOrOpen(str(asset)).ExportToString().encode()).hexdigest()) for asset in files])
    (ROOT/'USD_MANIFEST.json').write_text(json.dumps(closure,indent=2)+'\n')
    (frozen/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
