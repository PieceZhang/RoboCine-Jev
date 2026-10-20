"""Sensor calibration, closest-hit shadows and acquisition-time full-body TF."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_agent_wbc.robot_sensor_rig import (
    OPTICAL_FROM_USD, RobotSensorRig, SensorAudit, lidar_directions, load_config,
    optical_transform, pose_transform, transform,
)


@pytest.mark.parametrize('name', ['g1', 'go2', 'ridgeback'])
def test_optical_and_usd_forward_match_explicit_mount_direction(name):
    spec, _ = load_config(name)
    camera = spec['camera']
    local = optical_transform(camera)
    np.testing.assert_allclose(local[:3, :3].T @ local[:3, :3], np.eye(3), atol=1e-15)
    assert np.linalg.det(local[:3, :3]) == pytest.approx(1.)
    np.testing.assert_allclose(local[:3, :3] @ [0, 0, 1], camera['forward_parent'])
    np.testing.assert_allclose(local[:3, :3] @ [0, -1, 0], camera['up_parent'])
    np.testing.assert_allclose((local @ OPTICAL_FROM_USD)[:3, :3] @ [0, 0, -1], camera['forward_parent'])
    assert spec['lidars'] and len({s['name'] for s in spec['lidars']}) == len(spec['lidars'])
    rays = lidar_directions(spec['lidars'][0])
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1., atol=1e-15)
    assert np.any(np.isclose(rays[:, 2], 0.))


def make_rig(name='go2', auxiliary=False):
    rig = object.__new__(RobotSensorRig)
    rig.spec, _ = load_config(name)
    rig.root = '/World/Robot'
    rig.camera_spec = rig.spec['camera']
    rig.camera_local = optical_transform(rig.camera_spec)
    specs = deepcopy(rig.spec['lidars'])
    if auxiliary:
        specs.append(dict(specs[0], name='low', translation_m=[.2, 0., -.1], rpy_deg=[5, 10, 20]))
    rig.lidars = [dict(spec=s, local=transform(s['translation_m'], Rotation.from_euler('xyz', s['rpy_deg'], degrees=True).as_matrix()),
                       rays=lidar_directions(s)) for s in specs]
    return rig


class NearestSurfaces:
    """A near body shadow followed by an external surface; no passthrough."""
    def __init__(self):
        self.calls = 0

    def raycast_closest(self, origin, ray, maximum):
        self.calls += 1
        origin, ray = np.asarray(origin), np.asarray(ray)
        distance = .02 if ray[0] > .5 else 2.
        return dict(hit=True, distance=distance, position=origin+distance*ray,
                    collision='/World/Robot/arm/collision' if distance < .05 else '/World/Wall/collision')


def test_full_roll_pitch_and_multiple_scanners_share_acquisition_stamp():
    rig = make_rig(auxiliary=True)
    body = transform([2., -1., .7], Rotation.from_euler('xyz', [.4, -.3, .8]).as_matrix())
    poses = {name: body for name in rig.parent_names}
    query = NearestSurfaces()
    scan = rig.sample(1.25, poses, query)
    assert query.calls == sum(len(item['rays']) for item in rig.lidars)
    for packet, item in zip(scan['lidars'], rig.lidars):
        world = body @ item['local']
        np.testing.assert_allclose(packet['T_world_lidar'], world)
        assert packet['stamp_s'] == scan['stamp_s'] == 1.25
        assert packet['acquisition_duration_s'] == 0.
        points = np.asarray(packet['points_sensor'])
        np.testing.assert_allclose(packet['points_world'], points @ world[:3, :3].T + world[:3, 3])
        # A yaw-only conversion would fail both ray classification and this Z check.
        assert np.ptp(np.asarray(packet['points_world'])[:, 2]) > 1.
        shadow = np.asarray(packet['self_blocked_mask'])
        assert shadow.any() and (~shadow).any()
        assert not np.asarray(packet['valid_environment_mask'])[shadow].any()
        assert np.all(np.asarray(packet['ranges_m'])[shadow] < item['spec']['min_range_m'])


def test_arm_camera_moves_without_moving_body_lidar():
    rig = make_rig()
    poses = {name: np.eye(4) for name in rig.parent_names}
    before = rig.sample(0., poses, NearestSurfaces())
    poses[rig.camera_spec['parent']] = transform([1., 2., 3.], Rotation.from_euler('y', .8).as_matrix())
    after = rig.sample(.5, poses, NearestSurfaces())
    np.testing.assert_array_equal(before['lidars'][0]['T_world_lidar'], after['lidars'][0]['T_world_lidar'])
    assert not np.allclose(before['T_world_camera_optical'], after['T_world_camera_optical'])


def test_wxyz_measured_pose_preserves_tilt():
    rotation = Rotation.from_euler('xyz', [.3, -.4, .6])
    np.testing.assert_allclose(pose_transform([1, 2, 3], rotation.as_quat()[[3, 0, 1, 2]]),
                               transform([1, 2, 3], rotation.as_matrix()))


def test_failed_rgb_diagnostic_retains_separate_unverified_result(tmp_path):
    import json
    from types import SimpleNamespace
    def unavailable(*args):
        raise RuntimeError('renderer unavailable')
    rig = SimpleNamespace(manifest=lambda: {}, capture_camera=unavailable)
    audit = SensorAudit(rig, tmp_path)
    result = audit.finish(1.5, render=lambda: None)
    assert not result['camera_render_verified']
    assert not result['full_motion_range_verified']
    assert 'renderer unavailable' in result['camera_render_error']
    assert json.loads((tmp_path/'sensor_audit.json').read_text()) == result
    assert (tmp_path/'camera_render_error.txt').is_file()


@pytest.mark.parametrize('stamp', [-1., float('nan'), float('inf')])
def test_invalid_acquisition_clock_is_rejected(stamp):
    with pytest.raises(ValueError, match='timestamp'):
        make_rig().sample(stamp, {}, NearestSurfaces())


@pytest.mark.parametrize('name', ['g1', 'go2', 'ridgeback'])
def test_actual_usd_assembly_has_correct_parent_and_renderer_axis(name):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    spec, _ = load_config(name)
    asset = ROOT/spec['asset']
    if not asset.is_file():
        pytest.skip('local robot asset is not downloaded')
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World/Robot').GetReferences().AddReference(str(asset))
    physics_before = [str(p.GetPath()) for p in stage.Traverse()
                      if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)]
    rig = RobotSensorRig(stage, '/World/Robot', name)
    physics_after = [str(p.GetPath()) for p in stage.Traverse()
                     if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)]
    assert physics_after == physics_before
    camera = stage.GetPrimAtPath(rig.camera_path)
    assert camera.GetTypeName() == 'Camera'
    parent = camera.GetParent().GetParent()
    assert parent.GetName() == spec['camera']['parent']
    cache = UsdGeom.XformCache()
    world_camera = np.asarray(cache.GetLocalToWorldTransform(camera)).T
    world_parent = np.asarray(cache.GetLocalToWorldTransform(parent)).T
    np.testing.assert_allclose(world_camera, world_parent @ rig.camera_local @ OPTICAL_FROM_USD, atol=1e-7)
    # A point in optical +Z projects to the centre of the authored camera frustum.
    point = world_parent @ rig.camera_local @ [0., 0., 1., 1.]
    usd_point = np.linalg.inv(world_camera) @ point
    np.testing.assert_allclose(usd_point, [0., 0., -1., 1.], atol=1e-7)
    lidar = stage.GetPrimAtPath(rig.lidars[0]['path'])
    assert lidar.GetParent().GetName() == spec['lidars'][0]['parent']
