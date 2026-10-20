#!/usr/bin/env python3
"""Build a directly openable USD sensor overlay without editing tested physics."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    python = Path(os.environ.get('ISAAC_PYTHON','/home/w84384651/miniconda3/envs/env_isaaclab/bin/python'))
    site = python.resolve().parents[1]/'lib/python3.11/site-packages/isaacsim'
    usd = next((site/'extscache').glob('omni.usd.libs-*'))
    if not os.environ.get('TIAGO_SENSOR_USD_READY'):
        env = dict(os.environ,TIAGO_SENSOR_USD_READY='1')
        env['LD_LIBRARY_PATH'] = ':'.join([str(python.resolve().parents[1]/'lib'),str(usd/'bin'),str(site/'kit'),env.get('LD_LIBRARY_PATH','')])
        os.execve(str(python),[str(python),__file__],env)
    sys.path[:0] = [str(usd),str(ROOT.parent)]
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics
    from jev_agent_wbc.robot_sensor_rig import RobotSensorRig
    output = ROOT/'assets/tiago_with_sensors.usda'
    if output.exists():
        previous = ROOT/'SENSOR_ASSET_MANIFEST.json'
        expected = json.loads(previous.read_text()).get('sha256') if previous.exists() else None
        if hashlib.sha256(output.read_bytes()).hexdigest()!=expected:
            raise FileExistsError(f'{output} has changes outside this generator; preserve it before rebuilding')
        output.unlink()
    closure = json.loads((ROOT/'USD_MANIFEST.json').read_text())
    geometry = ROOT/closure['frozen_asset_directory']/'tiago.usd'
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths.append(str(geometry.relative_to(output.parent)))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/tiago'))
    rig = RobotSensorRig(stage,'/tiago','tiago',ROOT/'sensors.json')
    stage.GetRootLayer().Save()
    manifest = rig.manifest()
    manifest.update(asset='assets/tiago_with_sensors.usda',sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        geometry_reference=str(geometry.relative_to(ROOT)),geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
        physics_modified=False,physical_validation='See measured motion and independent coverage results; building this overlay is a CPU structural check',
        scanner_execution='Body Xforms define scanner extrinsics; run.sh performs actual closest-hit queries and saves point clouds.')
    if not stage.GetPrimAtPath(rig.camera_path).IsA(UsdGeom.Camera):
        raise ValueError('missing actual USD camera')
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if roots!=['/tiago/base_link']:
        raise ValueError(f'physics articulation changed: {roots}')
    manifest['articulation_roots']=roots
    cache = UsdGeom.XformCache()
    def matrix(path):
        return np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
    camera_relative = np.linalg.inv(matrix('/tiago/gripper_link'))@matrix(rig.camera_path)
    expected = rig.camera_local@np.diag([1.,-1.,-1.,1.])
    np.testing.assert_allclose(camera_relative,expected,atol=1e-12)
    for lidar in rig.lidars:
        parent = '/tiago/'+lidar['spec']['parent']
        np.testing.assert_allclose(np.linalg.inv(matrix(parent))@matrix(lidar['path']),lidar['local'],atol=1e-12)
    scan_keys = ('horizontal_samples','elevation_deg','min_range_m','max_range_m')
    if any(any(lidar['spec'][key]!=rig.lidars[0]['spec'][key] for key in scan_keys) for lidar in rig.lidars[1:]):
        raise ValueError('scanner parameters differ')
    manifest['cpu_checks']=dict(camera_parent_pose_and_optical_axes=True,lidar_parent_poses=True,
        two_identical_scanners=len(rig.lidars)==2,articulation_preserved=True)
    (ROOT/'SENSOR_ASSET_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(asset=str(output),camera=rig.camera_path,lidar_count=len(rig.lidars),physics_modified=False),indent=2))


if __name__=='__main__':
    main()
