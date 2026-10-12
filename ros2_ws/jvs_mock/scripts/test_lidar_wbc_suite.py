#!/usr/bin/env python3
"""Run the three requested trajectories with both collision sources in Isaac.

Run after sourcing ROS and the workspace. Each run owns its simulator and a
private DDS domain. Results are never reused from a previous output directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain', type=int, default=224)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--sources', nargs='+', choices=['groundtruth', 'lidar'],
                        default=['groundtruth', 'lidar'])
    parser.add_argument('--shots', nargs='+', choices=['shot_orbit', 'shot_moveout',
                        'shot_combo1', 'shot_combo2', 'shot_combo3', 'shot_combo4'],
                        default=['shot_orbit', 'shot_moveout', 'shot_combo1'])
    parser.add_argument('--stage-view', type=Path)
    parser.add_argument('--reference-mesh', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output directory')
    args.output.mkdir(parents=True)
    package = Path(__file__).resolve().parents[1]
    runner = package / 'scripts/test_wbc_ros_isaac.py'
    results = []
    for shot in args.shots:
        for source in args.sources:
            folder = args.output / shot / source
            command = [sys.executable, str(runner), '--output', str(folder), '--timeout', str(args.timeout),
                '--ros-domain-id', str(args.domain), '--launch', 'jvs_kc_wbc_isaac.launch.py',
                '--trajectory', str(package / 'trajectories' / (shot + '.json')),
                '--mesh-source', source, '--criterion', 'tracking']
            if args.stage_view:
                command += ['--stage-view', str(args.stage_view)]
            if args.reference_mesh:
                command += ['--mesh-cache-file', str(args.reference_mesh)]
            print(f'RUN {shot} {source}', flush=True)
            completed = subprocess.run(command, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
            results.append(dict(shot=shot, source=source, returncode=completed.returncode))
            (args.output / 'runs.json').write_text(json.dumps(results, indent=2) + '\n')
    if args.reference_mesh and args.shots == ['shot_orbit', 'shot_moveout', 'shot_combo1'] and len(results) == 6 and all(
            (args.output / row['shot'] / row['source'] / 'tracking_summary.json').is_file() for row in results):
        return subprocess.call([sys.executable, str(package / 'scripts/compare_lidar_wbc.py'),
            '--runs', str(args.output), '--reference-mesh', str(args.reference_mesh)])
    return 0 if all(row['returncode'] == 0 for row in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
