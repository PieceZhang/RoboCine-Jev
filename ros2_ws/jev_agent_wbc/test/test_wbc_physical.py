"""Measured body geometry and physical task/stop invariants."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.representation import PARTS, Pose, Request, Box
from jev_agent_wbc.wbc_protocol import CameraTrajectory, Capabilities, TimedTask, semantic_camera
from jev_agent_wbc.wbc_physical import MeasuredColliderGeometry, LiveTaskSession, audit_stop, constraint_feedback


def matrix(position=(0., 0., 0.), rpy=(0., 0., 0.)):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    out[:3, 3] = position
    return out


def geometry():
    return MeasuredColliderGeometry([
        dict(name='floor_body', body='base', part=PARTS[0], center_body=[0, 0, 0], half_extents_body=[.3, .2, .1]),
        dict(name='wrist_body', body='wrist', part=PARTS[1], center_body=[0, 0, .1], half_extents_body=[.1, .1, .2]),
    ])


def observation(stamp):
    base, camera = matrix((.1, .2, .7), (.2, -.3, 1.)), matrix((.3, .1, 1.), (-.2, .4, 1.2))
    return geometry().observe(stamp, base, camera, dict(base=base, wrist=camera))


def test_measured_rotations_and_every_shape_remain_in_representation():
    shape = geometry()
    poses = dict(base=matrix((1., 2., .7), (.2, -.3, 1.)), wrist=matrix((.4, 2., 1.), (-.2, .4, 1.2)))
    result = shape.observe(10., poses['base'], poses['wrist'], poses, box_budget=1)
    np.testing.assert_allclose(result.world_from_mount.rotation, poses['base'][:3, :3], atol=1e-15)
    for raw in shape.shapes:
        local = Box(tuple(raw['center_body']), tuple(raw['half_extents_body'])).vertices
        transform = poses[raw['body']]
        world = local@transform[:3,:3].T+transform[:3,3]
        points = result.world_from_mount.inverse().apply(world)
        assert np.max(result.occupancy[raw['part']][0].distance(points)) < 1e-14
    assert result.source == 'measured_fk'
    assert all(name not in json.dumps(result.to_dict()) for name in ['floor_body', 'wrist_body'])


def test_omitted_collider_body_cannot_be_replaced_by_authored_pose():
    with pytest.raises(ValueError, match='missing acquisition'):
        geometry().world_bounds(dict(base=np.eye(4)))
    with pytest.raises(ValueError, match='both functional'):
        MeasuredColliderGeometry([geometry().shapes[0]])


def test_session_exact_time_optical_axes_feedback_and_terminal_hold():
    observed = observation(10.)
    path = CameraTrajectory(10., (0., 1.), (observed.world_from_camera,
        observed.world_from_camera.compose(Pose((.02, .01, .03)))))
    task = TimedTask('physical-1', 10., observed.id, path, .03, .08)
    session = LiveTaskSession(Capabilities('native-test', True, frozenset({'timed_camera_se3'})))
    assert session.accept(task, observed, 10.)['supported']
    optical, _ = session.reference(10.5)
    np.testing.assert_allclose(semantic_camera(Pose.from_rt(optical[:3, :3], optical[:3, 3])).rotation,
                               path.sample(10.5).rotation, atol=1e-15)
    assert session.feedback(observation(10.5))['stamp_s'] == 10.5
    with pytest.raises(ValueError, match='strictly increase'):
        session.feedback(observation(10.5))
    with pytest.raises(ValueError, match='expired'):
        session.reference(11.1)
    held = session.feedback(observation(11.1), hold_terminal=True)
    assert held['tracking_mode'] == 'explicit_terminal_hold' and held['reference_stamp_s'] == 11.
    assert held['stamp_s'] == 11.1


def test_session_cannot_accept_unimplemented_body_request():
    observed = observation(10.)
    path = CameraTrajectory(10., (0., 1.), (observed.world_from_camera, observed.world_from_camera))
    request = Request(observed.id, path.id, 10., 11., occupancy_preferences=(dict(
        part=PARTS[0], frame='world', relation='avoid_region',
        region=Box((1, 0, 0), (.1, .1, .1)).to_dict(), margin_m=.02, weight=1.),)).to_dict()
    task = TimedTask('body-1', 10., observed.id, path, .03, .08, request)
    session = LiveTaskSession(Capabilities('native-test', True, frozenset({'timed_camera_se3'})))
    outcome = session.accept(task, observed, 10.)
    assert not outcome['supported'] and outcome['feasible'] is None
    assert session.task is None


def stop_data():
    stamps = np.arange(101)*.01
    bases = np.broadcast_to(np.eye(4), (101, 4, 4)).copy()
    cameras = bases.copy()
    joints = np.zeros((101, 2))
    return stamps, bases, cameras, joints


def test_strict_stop_uses_complete_window_and_all_measured_axes():
    times, bases, cameras, joints = stop_data()
    assert audit_stop(times, bases, cameras, joints, ['rad', 'm'], [0, 0, 0], [0, 0, 0])['passed']
    # Endpoint equality must not hide intermediate rotation, height or torso motion.
    cameras[75, :3, :3] = Rotation.from_rotvec([.02, 0, 0]).as_matrix()
    assert not audit_stop(times, bases, cameras, joints, ['rad', 'm'], [0, 0, 0], [0, 0, 0])['passed']
    cameras[75] = np.eye(4)
    bases[75, 2, 3] = .006
    assert not audit_stop(times, bases, cameras, joints, ['rad', 'm'], [0, 0, 0], [0, 0, 0])['passed']
    bases[75] = np.eye(4)
    joints[75, 1] = .003
    assert not audit_stop(times, bases, cameras, joints, ['rad', 'm'], [0, 0, 0], [0, 0, 0])['passed']


@pytest.mark.parametrize('window', [-1., 0., .1, float('nan'), True])
def test_stop_window_cannot_be_shortened_to_pass(window):
    times, bases, cameras, joints = stop_data()
    with pytest.raises(ValueError):
        audit_stop(times, bases, cameras, joints, ['rad', 'm'], [0, 0, 0], [0, 0, 0], window_s=window)


def test_low_displacement_does_not_hide_nonzero_final_velocity():
    times, bases, cameras, joints = stop_data()
    assert not audit_stop(times, bases, cameras, joints, ['rad', 'm'], [.021, 0, 0], [0, 0, 0])['passed']
    assert not audit_stop(times, bases, cameras, joints, ['rad', 'm'], [0, 0, 0], [0, .031, 0])['passed']


def test_stale_or_missing_constraint_evidence_is_unknown():
    evidence = dict(self_collision=dict(status='pass', stamp_s=1., scope='measured_collision_model', clearance_m=.02))
    assert constraint_feedback(evidence, 1.01)['self_collision']['status'] == 'pass'
    result = constraint_feedback(evidence, 1.2)
    assert result['self_collision']['status'] == 'unknown'
    assert result['environment']['status'] == 'unknown'
    assert result['environment']['reason'] == 'missing_evidence'
    with pytest.raises(ValueError, match='negative clearance'):
        constraint_feedback(dict(self_collision=dict(status='pass', stamp_s=1., scope='measured_collision_model', clearance_m=-.01)), 1.)
    with pytest.raises(ValueError):
        constraint_feedback(dict(actuation=dict(status='pass', stamp_s=1., scope='measured_native_limits', joint1_limit=1.)), 1.)


def test_native_usd_colliders_include_guide_and_parent_api_and_ignore_visuals():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, '/Robot')
    for body in ('base', 'wrist'):
        prim = UsdGeom.Xform.Define(stage, '/Robot/'+body).GetPrim()
        if body == 'base':
            UsdGeom.Xformable(prim).AddRotateXYZOp().Set((23., -18., 41.))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        cube = UsdGeom.Cube.Define(stage, '/Robot/'+body+'/CollisionCube')
        cube.CreateSizeAttr(.1)
        UsdGeom.Imageable(cube).CreatePurposeAttr('guide')
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        # Large unrelated visual meshes must not inflate robot collider evidence.
        visual = UsdGeom.Cube.Define(stage, '/Robot/'+body+'/Visual')
        visual.CreateSizeAttr(10.)
    group = UsdGeom.Xform.Define(stage, '/Robot/wrist/CollisionGroup')
    UsdPhysics.CollisionAPI.Apply(group.GetPrim())
    UsdGeom.Sphere.Define(stage, '/Robot/wrist/CollisionGroup/Sphere').CreateRadiusAttr(.03)
    UsdGeom.Xform.Define(stage, '/Prototype')
    inherited = UsdGeom.Cube.Define(stage, '/Prototype/InstancedCollider')
    inherited.CreateSizeAttr(.04)
    UsdPhysics.CollisionAPI.Apply(inherited.GetPrim())
    instance = UsdGeom.Xform.Define(stage, '/Robot/wrist/Instance').GetPrim()
    instance.GetReferences().AddInternalReference('/Prototype')
    instance.SetInstanceable(True)
    model = MeasuredColliderGeometry.from_stage(stage, root.GetPath(), ['wrist'])
    assert len(model.shapes) == 4 and set(model.body_names) == {'base', 'wrist'}
    assert any('InstancedCollider' in shape['name'] for shape in model.shapes)
    assert max(max(shape['half_extents_body']) for shape in model.shapes) < .051
