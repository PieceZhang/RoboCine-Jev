#!/usr/bin/env python3
"""Author inspectable USD sensor assemblies without starting physics or Kit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=['g1', 'go2', 'ridgeback', 'all'], default='all')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing output is never overwritten.')
    args = parser.parse_args()
    from pxr import Usd, UsdGeom
    from jev_agent_wbc.robot_sensor_rig import RobotSensorRig, load_config
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (['g1', 'go2', 'ridgeback'] if args.robot == 'all' else [args.robot]):
        spec, _ = load_config(name, args.config)
        asset = (ROOT/spec['asset']).resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        path = args.output/f'{name}_sensors.usda'
        stage = Usd.Stage.CreateNew(str(path.resolve()))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(world.GetPrim())
        root = '/World/Robot'
        stage.DefinePrim(root).GetReferences().AddReference(str(asset))
        rig = RobotSensorRig(stage, root, name, args.config)
        stage.GetRootLayer().Save()
        manifest = rig.manifest()
        manifest.update(physics_executed=False, rendering_executed=False, asset=str(asset))
        (args.output/f'{name}_sensor_mounts.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(path, flush=True)


if __name__ == '__main__':
    main()
