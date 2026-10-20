"""Explicit single-ground construction and read-only USD acceptance gates."""
import math


def audit_single_ground(stage, root, material, height_m):
    """Fail closed on duplicate physical floors or an unbound ground material."""
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
    prim=stage.GetPrimAtPath(root)
    active=[p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    analytic=stage.GetPrimAtPath(root+'/collisionPlane')
    if [str(p.GetPath()) for p in active]!=[str(analytic.GetPath())]:
        raise ValueError('ground must contain exactly its one analytic collisionPlane: '+str([str(p.GetPath()) for p in active]))
    if not analytic.IsA(UsdGeom.Plane) or UsdGeom.Plane(analytic).GetAxisAttr().Get()!='Z':
        raise ValueError('ground collider must be the original analytic Z plane')
    bound,_=UsdShade.MaterialBindingAPI(analytic).ComputeBoundMaterial(materialPurpose='physics')
    if not bound or bound.GetPath()!=material.GetPath():raise ValueError('analytic ground physics material mismatch')
    transform=UsdGeom.XformCache().GetLocalToWorldTransform(analytic)
    if not math.isfinite(height_m) or abs(float(transform[3][2])-height_m)>1e-6:
        raise ValueError('analytic ground height differs from the declared physical floor')
    visual=stage.GetPrimAtPath(root+'/geom')
    if UsdPhysics.CollisionAPI(visual).GetCollisionEnabledAttr().Get():raise ValueError('visual ground mesh still collides')
    physics=UsdPhysics.MaterialAPI(material.GetPrim())
    return dict(schema='ocs2_single_ground/v1',passed=True,active_ground_colliders=[str(analytic.GetPath())],
                visual_mesh_collision_enabled=False,physics_material=str(bound.GetPath()),
                static_friction=float(physics.GetStaticFrictionAttr().Get()),
                dynamic_friction=float(physics.GetDynamicFrictionAttr().Get()),
                restitution=float(physics.GetRestitutionAttr().Get()),height_m=float(height_m),
                scope='one analytic ground; original material values; robot collision geometry/pairs and all contact offsets untouched')


def repair_single_ground(stage, root, material, height_m):
    """Correct GroundPlane's duplicate visual-mesh collider before physics reset."""
    from pxr import UsdGeom,UsdPhysics,UsdShade
    visual=stage.GetPrimAtPath(root+'/geom');analytic=stage.GetPrimAtPath(root+'/collisionPlane')
    if (not visual.IsA(UsdGeom.Mesh) or not analytic.IsA(UsdGeom.Plane)
            or not visual.HasAPI(UsdPhysics.CollisionAPI) or not analytic.HasAPI(UsdPhysics.CollisionAPI)):
        raise ValueError('expected the SDK ground mesh and existing analytic collider')
    UsdPhysics.CollisionAPI(visual).CreateCollisionEnabledAttr(False)
    UsdShade.MaterialBindingAPI.Apply(analytic).Bind(material,bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose='physics')
    return audit_single_ground(stage,root,material,height_m)
