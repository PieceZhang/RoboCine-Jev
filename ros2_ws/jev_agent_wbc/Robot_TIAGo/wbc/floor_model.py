"""One analytic ground collider with the existing intended physical material."""


def audit_ground(stage,ground_path,material_path):
    from pxr import Usd,UsdPhysics,UsdShade,UsdGeom
    rows=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ground_path)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        physical=bool(material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI))
        rows.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),
            collision_enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
            bound_material=str(material.GetPath()) if material else None,material_is_physical=physical,
            world_transform=str(UsdGeom.XformCache().GetLocalToWorldTransform(prim))))
    material=UsdPhysics.MaterialAPI(stage.GetPrimAtPath(material_path))
    properties=dict(static_friction=material.GetStaticFrictionAttr().Get(),
        dynamic_friction=material.GetDynamicFrictionAttr().Get(),restitution=material.GetRestitutionAttr().Get())
    enabled=[row for row in rows if row['collision_enabled']]
    return dict(colliders=rows,intended_material=material_path,intended_material_properties=properties,
        enabled_collider_count=len(enabled),passed=bool(len(enabled)==1 and enabled[0]['path']==ground_path+'/collisionPlane'
            and enabled[0]['type']=='Plane' and enabled[0]['bound_material']==material_path and enabled[0]['material_is_physical']))


def configure_single_ground(stage,ground_path,material_path):
    from pxr import UsdPhysics,UsdShade
    before=audit_ground(stage,ground_path,material_path)
    expected={ground_path+'/geom',ground_path+'/collisionPlane'}
    if {row['path'] for row in before['colliders']}!=expected:
        raise RuntimeError('Unexpected SDK ground geometry; do not silently remove any new collider')
    mesh=stage.GetPrimAtPath(ground_path+'/geom');plane=stage.GetPrimAtPath(ground_path+'/collisionPlane')
    if mesh.GetTypeName()!='Mesh' or plane.GetTypeName()!='Plane':
        raise RuntimeError('Expected original visual Mesh plus analytic Plane')
    UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr().Set(False)
    UsdPhysics.CollisionAPI(plane).CreateCollisionEnabledAttr().Set(True)
    UsdShade.MaterialBindingAPI.Apply(plane).Bind(UsdShade.Material.Get(stage,material_path),materialPurpose='physics')
    after=audit_ground(stage,ground_path,material_path)
    properties_unchanged=before['intended_material_properties']==after['intended_material_properties']
    transforms_unchanged=all(a['world_transform']==b['world_transform'] for a,b in zip(before['colliders'],after['colliders']))
    result=dict(schema='tiago_single_ground_plane/v1',before=before,after=after,
        intended_material_values_unchanged=properties_unchanged,floor_transforms_unchanged=transforms_unchanged,
        operation='original analytic plane enabled and bound to original ground material; geom Mesh retained as visual only',
        passed=bool(after['passed'] and properties_unchanged and transforms_unchanged))
    if not result['passed']:raise RuntimeError('Single analytic ground static gate failed')
    return result
