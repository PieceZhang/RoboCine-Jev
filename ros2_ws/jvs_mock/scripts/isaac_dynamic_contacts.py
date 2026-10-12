"""Report all relevant physical contacts while excluding the supporting floor."""


def configure_contact_audit(stage):
    from pxr import PhysxSchema, UsdPhysics
    robot, moving, static = [], [], []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith('/jvs_mock/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            robot.append(prim)
        elif path.startswith('/World/DynamicObstacles/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            moving.append(prim)
        elif (path.startswith('/World/Simview/') and prim.HasAPI(UsdPhysics.CollisionAPI)
              and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
              and '/floor/' not in path.lower() and 'groundplane' not in path.lower()):
            static.append(prim)
        if prim.IsA(UsdPhysics.Scene):
            api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            api.CreateReportKinematicStaticPairsAttr(True)
            api.CreateReportKinematicKinematicPairsAttr(True)
    for bodies, opponents in ((robot, moving), (moving, robot)):
        targets = [prim.GetPath() for prim in static+opponents]
        for prim in bodies:
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr(0.)
            api.CreateReportPairsRel().SetTargets(targets)
    return dict(robot_report_bodies=len(robot), moving_report_bodies=len(moving),
                reported_static_colliders=len(static), supporting_floor_contacts_excluded=True,
                kinematic_static_pairs=True, kinematic_kinematic_pairs=True)
