#!/usr/bin/env python3
"""Compare six real ROS runs and audit both sources against the same USD surfels.

Each input folder must contain tracking_summary.json and telemetry.csv. The
reference mesh is used only after execution. This is a sampled geometry audit,
not a physics contact or continuous-time certificate of the observed environment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from jvs_mock.kc_svsdf import SweptCollision


def audit(folder, reference):
    summary = json.loads((folder / 'tracking_summary.json').read_text())
    with (folder / 'telemetry.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    minimum = np.inf
    violations = 0
    for row in rows:
        state = np.array([float(row['state_' + name]) for name in
                          ['x', 'y', 'yaw'] + [f'q{i}' for i in range(1, 7)]])
        rotation = Rotation.from_rotvec([float(row['base_rotvec_' + axis]) for axis in 'xyz']).as_matrix()
        clearance = float(reference.clearance(state, float(row['base_z']), rotation).min())
        minimum = min(minimum, clearance)
        violations += clearance < reference.margin
    summary.update(audit_samples=len(rows), reference_min_clearance_m=float(minimum) if rows else None,
                   reference_clearance_violations=violations,
                   audit_scope='measured states at control samples against groundtruth surface voxels')
    if rows and 'lidar_voxels' in rows[0]:
        summary.update(lidar_scans_integrated=int(rows[-1]['lidar_revision']),
            lidar_map_updates_observed=len({row['lidar_revision'] for row in rows}),
            lidar_age_max_s=max(float(row['lidar_age_s']) for row in rows),
            lidar_update_p95_ms=float(np.quantile([float(row['lidar_update_ms']) for row in rows], .95)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path,
                        help='contains shot_{orbit,moveout,combo1}/{groundtruth,lidar}')
    parser.add_argument('--reference-mesh', required=True)
    parser.add_argument('--position-slack-m', type=float, default=.03)
    parser.add_argument('--orientation-slack-rad', type=float, default=.03)
    parser.add_argument('--duration-ratio', type=float, default=1.25)
    parser.add_argument('--shot-completion-slack', type=float, default=.25)
    args = parser.parse_args()
    reference = SweptCollision(args.reference_mesh)
    report = dict(criteria=dict(position_rms_slack_m=args.position_slack_m,
        orientation_rms_slack_rad=args.orientation_slack_rad, duration_ratio=args.duration_ratio,
        shot_completion_ratio_slack=args.shot_completion_slack,
        clearance='both sources >= configured clearance at every recorded sample'), trajectories={})
    for shot in ('shot_orbit', 'shot_moveout', 'shot_combo1'):
        runs = {source: audit(args.runs / shot / source, reference) for source in ('groundtruth', 'lidar')}
        gt, lidar = runs['groundtruth'], runs['lidar']
        checks = dict(completed=all(r['completed'] and not r['failure'] for r in runs.values()),
            tracking=all(r.get('tracking_accuracy_pass', False) for r in runs.values()),
            shot_motions=all(r.get('shot_motions_pass', False) for r in runs.values()),
            command_continuity=all(r.get('command_continuity_pass', False) for r in runs.values()),
            avoidance=all(r['audit_samples'] > 0 and r['reference_clearance_violations'] == 0 for r in runs.values()))
        gt_shots, lidar_shots = gt.get('shots', []), lidar.get('shots', [])
        checks['shot_completion_comparison'] = (bool(gt_shots) and len(gt_shots) == len(lidar_shots)
            and all(a['type'] == b['type'] and abs(a['completion_ratio']-b['completion_ratio']) <= args.shot_completion_slack
                    for a, b in zip(gt_shots, lidar_shots)))
        checks['shot_direction'] = all(s['motion_direction_pass'] for s in gt_shots + lidar_shots)
        for field, slack in [('position_error_rms', args.position_slack_m),
                             ('orientation_error_rms', args.orientation_slack_rad)]:
            checks[field] = (field in gt and field in lidar and lidar[field] <= gt[field] + slack)
        checks['duration'] = ('execution_duration_s' in gt and 'execution_duration_s' in lidar
            and lidar['execution_duration_s'] <= args.duration_ratio * gt['execution_duration_s'])
        # Keep the existing absolute trajectory acceptance visible even when
        # the groundtruth baseline itself fails it. Relative comparability is
        # a separate result, never a substitute for that acceptance.
        comparable = all(value for key, value in checks.items() if key not in ('tracking', 'shot_motions'))
        report['trajectories'][shot] = dict(runs=runs, checks=checks,
            comparable=comparable, passed=all(checks.values()))
    report['comparable'] = all(item['comparable'] for item in report['trajectories'].values())
    report['passed'] = all(item['passed'] for item in report['trajectories'].values())
    (args.runs / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
