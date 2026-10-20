#!/usr/bin/env python3
"""Verify exact upstream bytes and author a local, ROS-independent USD layer."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

from inspect_source import ROOT, usd_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Retrieve missing pinned source files')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'ASSET_MANIFEST.json').read_text())
    for record in manifest['files']:
        path = ROOT / record['path']
        if not path.exists() and args.download:
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(record['url'], timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError(f'Upstream hash differs from pinned bytes: {record["source_path"]}')
            path.write_bytes(data)
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; use --download')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Source hash mismatch: {path}')
    usd_bootstrap()
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    import numpy as np
    from scipy.spatial.transform import Rotation
    source = Sdf.Layer.FindOrOpen(str(ROOT / manifest['source_entry']))
    refs = sorted(set(re.findall(r'@([^@\n]+)@', source.ExportToString())) - {'OmniPBR.mdl'})
    if refs:
        raise ValueError(f'Unexpected external source dependencies: {refs}')
    path = ROOT / manifest['entry']
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, '/husky_ur5').GetPrim()
    root.GetReferences().AddReference('./source/husky_ur5.usd')
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    # ROS subscriptions in the vendor sample otherwise compete with this runner.
    for name in ['HuskyControl', 'JointState', 'JointCommand']:
        stage.OverridePrim('/husky_ur5/' + name).SetActive(False)
    # The task controls the UR5 camera, not the vendor demo's 2F-140 fingers.
    # Preserve their authored geometry and mass as a rigid tool. A tree of exact
    # fixed joints avoids resetting an underactuated closed loop to invalid q=0.
    source_stage = Usd.Stage.Open(str(ROOT / manifest['source_entry']))
    xforms = UsdGeom.XformCache()
    repaired = []
    links = {p.attrib['name']:p for p in ET.parse(ROOT / 'assets/source/husky_ur5.urdf').getroot().findall('link')}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        inertial = links.get(prim.GetName())
        inertial = None if inertial is None else inertial.find('inertial')
        if inertial is None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.))
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1., 0., 0., 0.))
            continue
        attr = inertial.find('inertia').attrib
        tensor = np.array([[float(attr['ixx']),float(attr['ixy']),float(attr['ixz'])],
                           [float(attr['ixy']),float(attr['iyy']),float(attr['iyz'])],
                           [float(attr['ixz']),float(attr['iyz']),float(attr['izz'])]])
        eigenvalues, axes = np.linalg.eigh(tensor)
        if np.min(eigenvalues) <= 0.:
            raise ValueError(f'Nonpositive source inertia: {prim.GetPath()}')
        if np.linalg.det(axes) < 0.:
            axes[:, 0] *= -1.
        origin = inertial.find('origin')
        xyz = np.fromstring(origin.get('xyz','0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy','0 0 0'), sep=' ')
        q = Rotation.from_matrix(Rotation.from_euler('xyz', rpy).as_matrix() @ axes).as_quat()
        mass_api.CreateMassAttr(float(inertial.find('mass').get('value')))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*xyz))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*eigenvalues))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        repaired.append(dict(path=str(prim.GetPath()), action='restore_full_mass_and_inertia_from_pinned_urdf'))
    # The imported source authored extents already in metres and then scaled
    # them again, which corrupts BBox/ground and footprint audits. Author exact
    # chassis primitives from the same pinned URDF, with unit-cube extents.
    stage.OverridePrim('/husky_ur5/husky_base_link/collisions').SetActive(False)
    for index, collision in enumerate(links['husky_base_link'].findall('collision')):
        size = np.fromstring(collision.find('geometry/box').get('size'), sep=' ')
        origin = collision.find('origin')
        xyz = np.fromstring(origin.get('xyz','0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy','0 0 0'), sep=' ')
        cube = UsdGeom.Cube.Define(stage, f'/husky_ur5/husky_base_link/VerifiedChassisCollision{index}')
        cube.CreateSizeAttr(1.)
        cube.CreateExtentAttr([Gf.Vec3f(-.5), Gf.Vec3f(.5)])
        cube.CreatePurposeAttr('guide')
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        q = Rotation.from_euler('xyz', rpy).as_quat()
        xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        repaired.append(dict(path=str(cube.GetPath()), action='exact_pinned_URDF_chassis_collision_box', size_m=size.tolist()))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            half = np.full(3, cylinder.GetRadiusAttr().Get())
            half[{'X':0, 'Y':1, 'Z':2}[cylinder.GetAxisAttr().Get()]] = cylinder.GetHeightAttr().Get()/2
            cylinder.CreateExtentAttr([Gf.Vec3f(*(-half)), Gf.Vec3f(*half)])
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        name = prim.GetName()
        if joint.GetExcludeFromArticulationAttr().Get():
            if prim.IsA(UsdPhysics.FixedJoint):
                joint.CreateExcludeFromArticulationAttr(False)
            else:
                prim.SetActive(False)
                repaired.append(dict(path=str(prim.GetPath()), action='deactivate_redundant_gripper_loop'))
                continue
        if prim.IsA(UsdPhysics.RevoluteJoint) and not name.startswith(('husky_', 'ur5_')):
            p0, p1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(p0) != 1 or len(p1) != 1:
                raise ValueError(f'Unexpected gripper body relationship: {prim.GetPath()}')
            parent = xforms.GetLocalToWorldTransform(source_stage.GetPrimAtPath(p0[0]))
            child = xforms.GetLocalToWorldTransform(source_stage.GetPrimAtPath(p1[0]))
            relative = child * parent.GetInverse()  # USD uses row vectors.
            prim.SetTypeName('PhysicsFixedJoint')
            fixed = UsdPhysics.FixedJoint(prim)
            fixed.CreateLocalPos0Attr(Gf.Vec3f(relative.ExtractTranslation()))
            q = relative.ExtractRotationQuat()
            fixed.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
            fixed.CreateLocalPos1Attr(Gf.Vec3f(0.))
            fixed.CreateLocalRot1Attr(Gf.Quatf(1., 0., 0., 0.))
            fixed.CreateExcludeFromArticulationAttr(False)
            repaired.append(dict(path=str(prim.GetPath()), action='lock_unused_gripper_in_authored_pose'))
    # Enable physical self contacts. Filter only rigid parts joined in the same
    # fixed assembly; adjacent moving links retain their authored joint filter.
    component = {str(p.GetPath()):str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    def find(path):
        while component[path] != path:
            path = component[path]
        return path
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(a) == 1 and len(b) == 1:
            component[find(str(a[0]))] = find(str(b[0]))
    filtered = []
    for body in component:
        pairs = [other for other in component if other != body and find(body) == find(other)]
        if pairs:
            UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(body)).CreateFilteredPairsRel().SetTargets(pairs)
            filtered.append(dict(body=body, fixed_assembly_only=pairs))
    stage.GetPrimAtPath('/husky_ur5/husky_base_link').CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool).Set(True)
    stage.GetRootLayer().Save()
    (ROOT / 'assets/repairs.json').write_text(json.dumps(dict(source_unchanged=True, repairs=repaired, physical_self_contacts_enabled=True, fixed_assembly_pair_filters=filtered), indent=2) + '\n')
    print(f'Verified {len(manifest["files"])} files and generated {path}')


if __name__ == '__main__':
    main()
