#!/usr/bin/env python3
"""Run regular camera shots (or selected cases) in Isaac with separate reports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isaac-python', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--solver-type', choices=['PGS', 'TGS'], default='PGS')
    parser.add_argument('--cases', nargs='+', default=['orbit_ccw10', 'orbit_cw15',
                        'pan_left20', 'pan_right30', 'moveout_040', 'moveout_080'])
    parser.add_argument('--criterion', choices=['tracking', 'stage1'], default='tracking',
                        help='Tracking precision by default; stage1 additionally requires physical constraints')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in args.cases:
        folder = args.output / case
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / 'isaac.log').open('w') as log:
            command = [sys.executable, str(package/'scripts/jvs_isaac_bootstrap.py'),
                '--isaac-python', args.isaac_python, '--runner', str(package/'scripts/test_wbc_isaac.py'), '--',
                '--xacro', str(package/'urdf/jvs_mock.urdf.xacro'), '--headless', 'true', '--enable-lidar', 'false',
                '--solver-type', args.solver_type,
                '--trajectory', str(package/f'trajectories/wbc_{case}.json'), '--output', str(folder.resolve())]
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300.)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
        report = folder / 'summary.json'
        summary = json.loads(report.read_text()) if report.exists() else {'completed': False, 'failure': 'no report'}
        results[case] = dict(returncode=code, summary=summary)
        print(f'{case}: returncode={code}, tracking_accuracy_pass={summary.get("tracking_accuracy_pass", False)}, '
              f'stage1_pass={summary.get("stage1_pass", False)}', flush=True)
        (args.output / 'suite.json').write_text(json.dumps(results, indent=2)+'\n')
    key = 'tracking_accuracy_pass' if args.criterion == 'tracking' else 'stage1_pass'
    return 0 if all(r['summary'].get(key, False) for r in results.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
