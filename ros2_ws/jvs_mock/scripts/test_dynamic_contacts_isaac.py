#!/usr/bin/env python3
"""Positive Isaac contact probes for the actual dynamic audit configuration."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.argv[:] = [sys.argv[0]]
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
try:
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics, PhysicsSchemaTools
    from omni.physx import get_physx_simulation_interface
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import RigidPrim
    from isaac_dynamic_contacts import configure_contact_audit

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    scene = UsdPhysics.Scene.Define(stage, '/World/physicsScene')
    scene.CreateGravityMagnitudeAttr(0.)
    for path, position, rigid in [('/jvs_mock/base_link', [1., 0., .5], True),
                                 ('/World/DynamicObstacles/Post', [5., 0., .5], True),
                                 ('/World/Simview/Environment/Furniture', [3., 0., .5], False),
                                 ('/World/groundPlane', [1., 0., -.1], False)]:
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(.2)
        shape.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        if rigid:
            body = UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
            body.CreateKinematicEnabledAttr(True)
    configuration = configure_contact_audit(stage)
    events = []
    def contacts(headers, data):
        events.extend([dict(actors=[str(PhysicsSchemaTools.intToSdfPath(h.actor0)),
                                    str(PhysicsSchemaTools.intToSdfPath(h.actor1))],
                            event=str(h.type)) for h in headers])
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(contacts)
    simulation = SimulationContext(physics_dt=.001, rendering_dt=1/60, stage_units_in_meters=1.,
                                   physics_prim_path='/World/physicsScene')
    simulation.initialize_physics()
    robot = RigidPrim('/jvs_mock/base_link', name='probe_robot', reset_xform_properties=False)
    obstacle = RigidPrim('/World/DynamicObstacles/Post', name='probe_post', reset_xform_properties=False)
    robot.initialize()
    obstacle.initialize()
    simulation.play()
    def sample(robot_position, obstacle_position):
        start = len(events)
        for body, position in ((robot, robot_position), (obstacle, obstacle_position)):
            targets = body._physics_view.get_transforms().copy()
            targets[:, :3] = position
            body._physics_view.set_kinematic_targets(targets, np.array([0], np.uint32))
        for _ in range(50):
            simulation.step(render=True)
        return events[start:]
    probes = {}
    probes['floor'] = sample([1., 0., 0.], [5., 0., .5])
    sample([1., 0., .5], [5., 0., .5])
    probes['robot_static'] = sample([3., 0., .5], [5., 0., .5])
    sample([1., 0., .5], [5., 0., .5])
    probes['robot_dynamic'] = sample([5., 0., .5], [5., 0., .5])
    sample([1., 0., .5], [5., 0., .5])
    probes['obstacle_static'] = sample([1., 0., .5], [3., 0., .5])
    results = {name: any(all(any(prefix in actor for actor in event['actors']) for prefix in prefixes)
                         for event in probes[name]) for name, prefixes in
               [('robot_static', ['/jvs_mock/', '/World/Simview/']),
                ('robot_dynamic', ['/jvs_mock/', '/World/DynamicObstacles/']),
                ('obstacle_static', ['/World/DynamicObstacles/', '/World/Simview/'])]}
    results['floor_excluded'] = not any('/World/groundPlane' in e['actors'] for e in probes['floor'])
    result = dict(configuration=configuration, probes=probes, checks=results, passed=all(results.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('CONTACT_PROBES '+json.dumps(result), flush=True)
    simulation.stop()
    subscription = None
    if not result['passed']:
        raise RuntimeError('contact probe failed')
finally:
    app.close()
