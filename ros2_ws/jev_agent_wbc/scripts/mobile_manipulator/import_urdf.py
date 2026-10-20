#!/usr/bin/env python3
"""Import a locally resolved authentic URDF as a free wheel-driven articulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--merge-fixed', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    a.urdf, a.output = a.urdf.resolve(), a.output.resolve()
    if not a.urdf.is_file():
        raise FileNotFoundError(a.urdf)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    os.environ.setdefault('ACCEPT_EULA', 'Y')
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, limit_cpu_threads=8))
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.asset.importer.urdf')
        import omni.kit.commands
        ok, config = omni.kit.commands.execute('URDFCreateImportConfig')
        if not ok:
            raise RuntimeError('URDFCreateImportConfig failed')
        config.set_fix_base(False)
        config.set_merge_fixed_joints(a.merge_fixed)
        config.set_self_collision(True)
        config.set_import_inertia_tensor(True)
        config.set_distance_scale(1.)
        config.set_density(0.)
        config.set_default_drive_strength(0.)
        config.set_default_position_drive_damping(0.)
        ok, root = omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=str(a.urdf),
                                           import_config=config, dest_path=str(a.output))
        if not ok or not a.output.is_file():
            raise RuntimeError(f'URDF import failed: {root}')
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(str(a.output))
        report = dict(urdf=str(a.urdf), urdf_sha256=hashlib.sha256(a.urdf.read_bytes()).hexdigest(),
                      usd=str(a.output), usd_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                      import_root=str(root), fix_base=False, self_collision=True,
                      merge_fixed_joints=a.merge_fixed, physics_executed=False,
                      articulation_roots=[str(x.GetPath()) for x in stage.Traverse() if x.HasAPI(UsdPhysics.ArticulationRootAPI)],
                      rigid_bodies=[str(x.GetPath()) for x in stage.Traverse() if x.HasAPI(UsdPhysics.RigidBodyAPI)])
        a.output.with_suffix('.import.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    if '--worker' in sys.argv or '--help' in sys.argv or '-h' in sys.argv:
        main()
    else:
        import subprocess
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--output', type=Path)
        a, _ = parser.parse_known_args()
        if a.output and a.output.exists():
            raise SystemExit(f'Output already exists; choose a new file: {a.output}')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'])
        report = a.output.with_suffix('.import.json') if a.output else None
        valid = False
        if report and report.is_file() and a.output.is_file():
            data = json.loads(report.read_text())
            valid = bool(data.get('rigid_bodies') and data.get('articulation_roots')
                         and data['usd_sha256'] == hashlib.sha256(a.output.read_bytes()).hexdigest())
        raise SystemExit(0 if result.returncode == 0 and valid else 2)
