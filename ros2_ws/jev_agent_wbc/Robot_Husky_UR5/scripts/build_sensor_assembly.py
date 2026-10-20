#!/usr/bin/env python3
"""Author an inspectable robot + sensor USD without changing the runtime model."""
import hashlib
import json
from pathlib import Path
import sys

from inspect_source import ROOT, usd_bootstrap


def main():
    usd_bootstrap()
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom
    sys.path.insert(0, str(ROOT.parent))
    from jev_agent_wbc.robot_sensor_rig import RobotSensorRig, OPTICAL_FROM_USD
    profile = json.loads((ROOT / 'robot.json').read_text())
    model, sensors = ROOT / profile['usd'], ROOT / profile['sensors']
    model_hash, config_hash = (hashlib.sha256(p.read_bytes()).hexdigest() for p in [model, sensors])
    target = ROOT / 'assets/husky_with_sensors.usda'
    if target.exists():
        target.unlink()
    stage = Usd.Stage.CreateNew(str(target))
    root = UsdGeom.Xform.Define(stage, '/husky_ur5').GetPrim()
    root.GetReferences().AddReference('./' + model.name)
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    rig = RobotSensorRig(stage, str(root.GetPath()), profile['robot_key'], sensors)
    root.CreateAttribute('cine:inspectionAssembly', Sdf.ValueTypeNames.Bool).Set(True)
    root.CreateAttribute('cine:modelSha256', Sdf.ValueTypeNames.String).Set(model_hash)
    root.CreateAttribute('cine:sensorConfigSha256', Sdf.ValueTypeNames.String).Set(config_hash)
    for item in rig.lidars:
        prim, spec = stage.GetPrimAtPath(item['path']), item['spec']
        for key, value, kind in [
            ('horizontalSamples', spec['horizontal_samples'], Sdf.ValueTypeNames.Int),
            ('elevationDegrees', spec['elevation_deg'], Sdf.ValueTypeNames.DoubleArray),
            ('minRangeMeters', spec['min_range_m'], Sdf.ValueTypeNames.Double),
            ('maxRangeMeters', spec['max_range_m'], Sdf.ValueTypeNames.Double),
            ('requiresRunnerForAcquisition', True, Sdf.ValueTypeNames.Bool),
        ]:
            prim.CreateAttribute('cine:' + key, kind).Set(value)
    stage.GetRootLayer().Save()
    # Inspect the authored result again, including the parent-relative matrices.
    inspected = Usd.Stage.Open(str(target))
    cache = UsdGeom.XformCache()
    camera = inspected.GetPrimAtPath(rig.camera_path)
    optical = camera.GetParent()
    assert camera.IsA(UsdGeom.Camera)
    assert np.allclose(np.asarray(UsdGeom.Xformable(optical).GetLocalTransformation()).T, rig.camera_local)
    assert np.allclose(np.asarray(UsdGeom.Xformable(camera).GetLocalTransformation()).T, OPTICAL_FROM_USD)
    parent_world = np.asarray(cache.GetLocalToWorldTransform(optical.GetParent())).T
    optical_world = np.asarray(cache.GetLocalToWorldTransform(optical)).T
    assert np.allclose(optical_world, parent_world @ rig.camera_local)
    for item in rig.lidars:
        prim = inspected.GetPrimAtPath(item['path'])
        assert np.allclose(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T, item['local'])
        assert prim.GetAttribute('cine:horizontalSamples').Get() == item['spec']['horizontal_samples']
        assert list(prim.GetAttribute('cine:elevationDegrees').Get()) == item['spec']['elevation_deg']
    assert len(rig.lidars) == 2
    assert hashlib.sha256(model.read_bytes()).hexdigest() == model_hash
    assert hashlib.sha256(sensors.read_bytes()).hexdigest() == config_hash
    record = dict(schema='husky_sensor_assembly/v1', passed=True,
        scope='CPU USD hierarchy/extrinsics only; physical acquisition and rendering use run.sh',
        assembly=str(target.relative_to(ROOT)), assembly_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        model=profile['usd'], model_sha256=model_hash, config=profile['sensors'], config_sha256=config_hash,
        camera=rig.camera_path, camera_parent=rig.camera_spec['parent'],
        T_parent_optical=rig.camera_local.tolist(), lidar_paths=[x['path'] for x in rig.lidars],
        lidar_local_transforms=[x['local'].tolist() for x in rig.lidars],
        runtime_model_and_config_unchanged=True)
    (ROOT / 'assets/sensor_assembly.json').write_text(json.dumps(record, indent=2) + '\n')
    output = ROOT / 'results/sensor_assembly_preflight.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
