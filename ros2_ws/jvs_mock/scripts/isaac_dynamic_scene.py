"""USD composition checks and static collision authoring for dynamic runs only."""
import hashlib
import json
from pathlib import Path


def prepare_static_scene(stage, audit_file, origin, radius=4.5):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade

    root = stage.GetPrimAtPath('/World/Simview')
    paths = []
    bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    material = UsdShade.Material.Define(stage, '/World/DynamicRoomPhysicsMaterial')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.5)
    api.CreateDynamicFrictionAttr(.5)
    api.CreateRestitutionAttr(0.)
    # Imported furniture assets contain nested, enabled rigid bodies and door
    # joints. This stage is the fixed room; only the JSON actors move. Disabling
    # their imported dynamics keeps both RTX and PhysX at the authored pose.
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Gprim):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
            continue
        path = str(prim.GetPath())
        bound = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = bound.GetMin(), bound.GetMax()
        distance = sum((max(low[k], min(origin[k], high[k]))-origin[k])**2 for k in (0, 1))**.5
        if distance > radius:
            # Keep the complete visual/RTX scene. Physical colliders cover the
            # local workspace; the independent audit verifies the robot stays
            # inside this region with its complete body extent.
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
            continue
        if '/floor/' in path.lower() or 'groundplane' in path.lower():
            # The runner's Z=0 plane supplies floor contact. Duplicate source
            # floor triangles cause redundant caster contacts and GPU warnings.
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            continue
        if any('/' + name + '/' in path for name in ('CamBody', 'Annotations')):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility(0) == UsdGeom.Tokens.invisible:
            continue
        # Static triangle meshes preserve concave furniture and open doorways.
        # Author in the new stage's root layer; the referenced USDA is immutable.
        if prim.IsInstanceProxy():
            raise ValueError(f'static collision authoring needs a non-instance prim: {path}')
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(.002)
        api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
        paths.append(path)
    if not paths:
        raise ValueError('the selected USDA has no visible scene collision geometry')
    layers = {str(Path(layer.realPath).resolve()): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
              for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    report = dict(static_collider_count=len(paths), static_collider_paths=paths,
                  collision_region_origin=list(origin), collision_region_radius_m=radius,
                  usd_layer_sha256=layers, referenced_layers_unchanged=None)
    if audit_file:
        target = Path(audit_file).with_name('usd_scene_audit.json')
        target.write_text(json.dumps(report, indent=2) + '\n')
    print(f'JVS dynamic USDA: {len(paths)} static colliders; {len(layers)} composed source layers', flush=True)
    return report


def finish_static_scene_audit(report, audit_file):
    if report is None or not audit_file:
        return
    report['referenced_layers_unchanged'] = all(Path(path).is_file() and
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        for path, expected in report['usd_layer_sha256'].items())
    Path(audit_file).with_name('usd_scene_audit.json').write_text(json.dumps(report, indent=2) + '\n')
