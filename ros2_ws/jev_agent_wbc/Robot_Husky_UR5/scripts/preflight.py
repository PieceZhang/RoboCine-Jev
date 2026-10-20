#!/usr/bin/env python3
"""Read-only structural check of the actual composed robot, without Kit/GPU."""
import hashlib
import json
from pathlib import Path
import sys

from inspect_source import ROOT, usd_bootstrap


def main():
    usd_bootstrap()
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    sys.path.insert(0, str(ROOT.parent))
    from jev_agent_wbc.robot_sensor_rig import load_config, lidar_directions, optical_transform
    spec = json.loads((ROOT / 'robot.json').read_text())
    manifest = json.loads((ROOT / 'ASSET_MANIFEST.json').read_text())
    for item in manifest['files']:
        if hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Source hash mismatch: {item["path"]}')
    stage = Usd.Stage.Open(str(ROOT / spec['usd']))
    root = stage.GetDefaultPrim()
    physical = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    by_name = {p.GetName():p for p in physical}
    articulation = stage.GetPrimAtPath(str(root.GetPath()) + '/' + spec['articulation_root'].strip('/'))
    assert articulation.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert articulation.GetAttribute('physxArticulation:enabledSelfCollisions').Get() is True
    for name in [spec['base_link'], spec['ee_link']]:
        assert name in by_name
    joints = {p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    expected = set(spec['left_wheels'] + spec['right_wheels'] + [j['name'] for j in spec['joints']])
    assert set(joints) == expected, f'Unexpected remaining free DOFs: {set(joints) ^ expected}'
    for item in spec['joints']:
        joint = UsdPhysics.RevoluteJoint(joints[item['name']])
        assert np.isclose(np.deg2rad(joint.GetLowerLimitAttr().Get()), item['lower'])
        assert np.isclose(np.deg2rad(joint.GetUpperLimitAttr().Get()), item['upper'])
        assert item['lower'] < item['home'] < item['upper']
    for prim in physical:
        mass = UsdPhysics.MassAPI(prim)
        assert np.isfinite(list(mass.GetCenterOfMassAttr().Get())).all(), prim.GetPath()
        assert float(mass.GetMassAttr().Get()) > 0., prim.GetPath()
        q = mass.GetPrincipalAxesAttr().Get()
        assert np.isclose(q.GetLength(), 1.), prim.GetPath()
    inertial = UsdPhysics.MassAPI(by_name['husky_inertial_link'])
    assert np.all(np.asarray(inertial.GetDiagonalInertiaAttr().Get()) > .5)
    assert np.isclose(UsdPhysics.MassAPI(by_name['husky_top_chassis_link']).GetMassAttr().Get(), .01)
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    chassis_boxes = []
    for index, expected_size in enumerate([[.9874,.5709,.12375], [.78992,.5709,.10375]]):
        prim = stage.GetPrimAtPath(str(root.GetPath())+f'/husky_base_link/VerifiedChassisCollision{index}')
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        actual_size = np.asarray(bounds.GetMax()) - np.asarray(bounds.GetMin())
        assert np.allclose(actual_size, expected_size, atol=1e-6), actual_size
        chassis_boxes.append(actual_size.tolist())
    sensor, sensor_hash = load_config(spec['robot_key'], ROOT / spec['sensors'])
    assert sensor['asset'] == spec['usd'], 'sensor asset metadata differs from the selected physical model'
    assert len(sensor['lidars']) == 2
    scan_keys = ['horizontal_samples', 'elevation_deg', 'min_range_m', 'max_range_m']
    assert all(sensor['lidars'][0][key] == sensor['lidars'][1][key] for key in scan_keys)
    for body in [sensor['camera']['parent'], *(lidar['parent'] for lidar in sensor['lidars'])]:
        assert body in by_name
    optical_transform(sensor['camera'])
    rays = [len(lidar_directions(lidar)) for lidar in sensor['lidars']]
    result = dict(passed=True, kind='CPU static structure only; no physical or rendered acceptance',
                  source_revision=manifest['revision'], verified_files=len(manifest['files']),
                  active_rolling_wheel_dofs=4, active_arm_dofs=len(spec['joints']),
                  fixed_gripper=True, self_contacts_enabled=True,
                  measured_USD_chassis_collision_dimensions_m=chassis_boxes,
                  lidar_beams_per_scan=rays, sensor_config_sha256=sensor_hash,
                  composed_usd_sha256=hashlib.sha256((ROOT / spec['usd']).read_bytes()).hexdigest())
    path = ROOT / 'results/preflight.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
