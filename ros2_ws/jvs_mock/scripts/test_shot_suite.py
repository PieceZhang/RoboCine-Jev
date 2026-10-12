#!/usr/bin/env python3
"""Paired Isaac Stage 1 / shot MPC / no-preview ablation with identical physics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isaac-python', default='/home/w84384651/miniconda3/envs/env_isaaclab/bin/python')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['orbit','pan','pan_limit','moveout','moveout_forward','combo1'])
    parser.add_argument('--controllers', nargs='+', choices=['stage1','stage2','no_preview'], default=['stage1','stage2'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cases = {c['name']: c for c in json.loads((root/'trajectories/shot_stage2_manifest.json').read_text())}
    args.output.mkdir(parents=True, exist_ok=True)
    index = args.output/'suite.json'
    results = json.loads(index.read_text()) if index.exists() else {}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in args.cases:
        for mode in args.controllers:
            key = name+'_'+mode
            folder = args.output/key
            folder.mkdir(exist_ok=False)
            case = cases[name]
            command = [sys.executable, str(root/'scripts/jvs_isaac_bootstrap.py'), '--isaac-python', args.isaac_python,
                '--runner', str(root/'scripts/test_wbc_isaac.py'), '--', '--xacro', str(root/'urdf/jvs_mock.urdf.xacro'),
                '--headless', 'true', '--enable-lidar', 'false', '--solver-type', 'PGS',
                '--trajectory', str(root/'trajectories'/case['trajectory']), '--output', str(folder.resolve()),
                '--controller', mode, '--initial-q', *map(str, case['initial_q'])]
            with (folder/'isaac.log').open('w') as log:
                try:
                    code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            report = folder/'summary.json'
            summary = json.loads(report.read_text()) if report.exists() else dict(completed=False, failure='no report')
            results[key] = dict(returncode=code, summary=summary)
            print(key, summary.get('completed'), summary.get('failure'), summary.get('position_error_rms'), summary.get('orientation_error_rms'), flush=True)
            (args.output/'suite.json').write_text(json.dumps(results, indent=2)+'\n')
    return 0 if all(v['summary'].get('tracking_accuracy_pass', False) for v in results.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
