"""Read-only resolved USD material evidence and PhysX combine arithmetic."""
import math


COMBINE_PRIORITY={'average':0,'min':1,'multiply':2,'max':3}
OFFICIAL_COMBINE_SOURCE='https://nvidia-omniverse.github.io/PhysX/physx/latest/_api_build/structPxCombineMode.html'


def combine(a,b,mode_a,mode_b):
    if mode_a not in COMBINE_PRIORITY or mode_b not in COMBINE_PRIORITY:
        raise ValueError('unknown PhysX material combine mode')
    if not math.isfinite(a) or not math.isfinite(b) or min(a,b)<0:
        raise ValueError('ordinary finite nonnegative material coefficients required')
    mode=max((mode_a,mode_b),key=COMBINE_PRIORITY.__getitem__)
    return mode,{'average':lambda:(a+b)/2,'min':lambda:min(a,b),
        'multiply':lambda:a*b,'max':lambda:max(a,b)}[mode]()


def audit_stage(stage,robot_root='/World/Robot',ground_root='/World/Ground'):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade

    def shapes(root):
        parent=stage.GetPrimAtPath(root)
        result=[]
        for p in Usd.PrimRange(parent,Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Gprim):continue
            owner=p
            while owner and owner.GetPath().HasPrefix(parent.GetPath()) and not owner.HasAPI(UsdPhysics.CollisionAPI):
                owner=owner.GetParent()
            if owner and owner.GetPath().HasPrefix(parent.GetPath()) and owner.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(owner).GetCollisionEnabledAttr().Get() is not False:
                result.append(p)
        return result

    def material(p):
        # Render-only materials do not create a PxMaterial. Walk ancestors,
        # then use the actual PhysicsScene-bound default, never a guessed mu.
        def resolve(consumer):
            while consumer and not consumer.IsPseudoRoot():
                bound,relation=UsdShade.MaterialBindingAPI(consumer).ComputeBoundMaterial('physics')
                if bound and bound.GetPrim().HasAPI(UsdPhysics.MaterialAPI):return bound,relation
                consumer=consumer.GetParent()
            return None,None
        bound,relation=resolve(p);scope='shape_or_ancestor_physics_material'
        if not bound:
            scenes=[x for x in stage.Traverse() if x.IsA(UsdPhysics.Scene)]
            if len(scenes)==1:bound,relation=resolve(scenes[0]);scope='explicit_physics_scene_default'
        if not bound:
            return dict(shape=str(p.GetPath()),resolved=False,
                bound_material=None,reason='No resolved PhysicsMaterialAPI or explicit single-scene default; coefficients not inferred')
        api=UsdPhysics.MaterialAPI(bound.GetPrim())
        def mode(name):
            attr=bound.GetPrim().GetAttribute('physxMaterial:'+name+'CombineMode')
            return str(attr.Get() or 'average') if attr else 'average'
        return dict(shape=str(p.GetPath()),resolved=True,material=str(bound.GetPath()),
            binding_scope=scope,
            binding_relationship=str(relation.GetPath()),
            static_friction=float(api.GetStaticFrictionAttr().Get()),
            dynamic_friction=float(api.GetDynamicFrictionAttr().Get()),
            restitution=float(api.GetRestitutionAttr().Get()),
            friction_combine=mode('friction'),restitution_combine=mode('restitution'))
    robots=[material(p) for p in shapes(robot_root)]
    grounds=[material(p) for p in shapes(ground_root)]
    pairs=[]
    for robot in robots:
        for ground in grounds:
            if not robot['resolved'] or not ground['resolved']:continue
            static_mode,static=combine(robot['static_friction'],ground['static_friction'],robot['friction_combine'],ground['friction_combine'])
            _,dynamic=combine(robot['dynamic_friction'],ground['dynamic_friction'],robot['friction_combine'],ground['friction_combine'])
            restitution_mode,restitution=combine(robot['restitution'],ground['restitution'],robot['restitution_combine'],ground['restitution_combine'])
            pairs.append(dict(robot_shape=robot['shape'],ground_shape=ground['shape'],
                effective_friction_mode=static_mode,effective_static_friction=static,
                effective_dynamic_friction=dynamic,effective_restitution_mode=restitution_mode,
                effective_restitution=restitution))
    return dict(schema='legged_resolved_contact_materials/v1',robot_materials=robots,ground_materials=grounds,
        expected_contact_pair_materials=pairs,
        all_collision_materials_explicitly_resolved=bool(robots and grounds and all(r['resolved'] for r in robots+grounds)),
        official_combine_source=OFFICIAL_COMBINE_SOURCE,
        official_default_material_source='https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/population/Material.html',
        geometric_contact_or_dynamic_stability_verified=False)
