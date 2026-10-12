#!/usr/bin/env python3
"""Verify temporary base obstruction against a measured unobstructed run.

Simulator truth and the baseline are used only after closed-loop execution.
Neither enters the controller, its LiDAR map, or its reference trajectory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_dynamic_demos import surface
from jvs_mock.kc_svsdf import CollisionGeometry


STATE_KEYS = ['state_x', 'state_y', 'state_yaw', *[f'state_q{i}' for i in range(1, 7)]]


def read_rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def audit(folder, baseline):
    rows = read_rows(folder/'telemetry.csv')
    reference_rows = read_rows(baseline/'telemetry.csv')
    reference_summary = json.loads((baseline/'tracking_summary.json').read_text())
    reference_valid = bool(reference_summary.get('completed') and not reference_summary.get('failure'))
    same_trajectory = (hashlib.sha256((folder/'trajectory.json').read_bytes()).hexdigest()
                       == hashlib.sha256((baseline/'trajectory.json').read_bytes()).hexdigest())
    path = np.array([float(r['time_s']) for r in rows])
    physical = np.array([float(r['wall_elapsed_s']) for r in rows])
    ref_path = np.array([float(r['time_s']) for r in reference_rows])
    ref_state = np.array([[float(r[k]) for k in STATE_KEYS] for r in reference_rows])
    ref_state[:, 2] = np.unwrap(ref_state[:, 2])
    nominal = np.column_stack([np.interp(path, ref_path, ref_state[:, i]) for i in range(9)])
    actual = np.array([[float(r[k]) for k in STATE_KEYS] for r in rows])
    active = np.array([r['motion_active'] == '1' and r['phase'] == 'tracking' for r in rows])
    dt = np.array([float(r['dt_s']) for r in rows])
    base_deviation = np.linalg.norm(actual[:, :2]-nominal[:, :2], axis=1)
    side = np.column_stack([-np.sin(nominal[:, 2]), np.cos(nominal[:, 2])])
    lateral_deviation = np.abs(np.sum((actual[:, :2]-nominal[:, :2])*side, axis=1))
    heading_deviation = np.abs(np.arctan2(np.sin(actual[:, 2]-nominal[:, 2]), np.cos(actual[:, 2]-nominal[:, 2])))
    nominal_v = np.interp(path, ref_path, [float(r['measured_v']) for r in reference_rows])
    camera = np.array([[float(r['camera_'+k]) for k in 'xyz'] for r in rows])
    camera_speed = np.r_[0., np.linalg.norm(np.diff(camera, axis=0), axis=1)/np.diff(physical)]
    arm_speed = np.linalg.norm([[float(r[f'measured_dq{i}']) for i in range(1, 7)] for r in rows], axis=1)
    stopped = (np.abs([float(r['measured_v']) for r in rows]) <= .025)
    stopped &= (np.abs([float(r['measured_omega']) for r in rows]) <= .12)
    arm_tracking = stopped & (np.abs(nominal_v) >= .04) & (camera_speed >= .005) & (arm_speed >= .02)
    arm_tracking &= np.array([float(r['position_error']) <= .05 and abs(float(r['path_rate'])-1.) < 1e-8 for r in rows])
    scene = json.loads((folder/'obstacle_scene.json').read_text())
    truth = read_rows(folder/'obstacle_truth.csv')
    epoch = json.loads((folder/'dynamic_epoch.json').read_text())['epoch_s']
    geometry = CollisionGeometry()
    base_columns = [i for i, shape in enumerate(geometry.shapes) if shape[0] == 0]
    reports = []
    for obstacle in scene['obstacles']:
        selected = [r for r in truth if r['obstacle'] == obstacle['name']]
        times = [float(r['simulation_time_s']) for r in selected]
        centers = np.column_stack([np.interp(physical+epoch, times, [float(r[k]) for r in selected]) for k in 'xyz'])
        points, coverage = surface(obstacle['model'])
        gap = np.full(len(rows), np.inf)
        for i in np.flatnonzero(active):
            tilt = Rotation.from_rotvec([float(rows[i]['base_rotvec_'+k]) for k in 'xyz']).as_matrix()
            distances = geometry.primitive_distances(nominal[i], points+centers[i], float(rows[i]['base_z']), tilt)[0]
            gap[i] = float(distances[:, base_columns].min()-coverage)
        blocked = active & (gap < .025)
        duration = float(dt[blocked].sum())
        changed = blocked & ((lateral_deviation >= .03) | (heading_deviation >= .15))
        paused = blocked & arm_tracking
        clear_later = active & (gap >= .05) & (physical > physical[blocked][-1]) if np.any(blocked) else np.zeros(len(rows), bool)
        released = bool(dt[clear_later].sum() >= .5)
        blocking_pass = bool(duration >= .10 and released and (np.any(changed) or dt[paused].sum() >= .10))
        reports.append(dict(obstacle=obstacle['name'],
            nominal_base_clearance_lower_m=float(gap.min()),
            nominal_base_blocked_duration_s=duration,
            maximum_base_deviation_during_block_m=float(base_deviation[blocked].max()) if np.any(blocked) else 0.,
            maximum_lateral_base_deviation_during_block_m=float(lateral_deviation[blocked].max()) if np.any(blocked) else 0.,
            maximum_heading_deviation_during_block_rad=float(heading_deviation[blocked].max()) if np.any(blocked) else 0.,
            base_pause_with_arm_tracking_duration_s=float(dt[paused].sum()),
            base_bypassed_obstruction=bool(np.any(changed)),
            base_paused_with_arm_tracking=bool(dt[paused].sum() >= .10),
            first_blocked_physical_time_s=float(physical[blocked][0]) if np.any(blocked) else None,
            last_blocked_physical_time_s=float(physical[blocked][-1]) if np.any(blocked) else None,
            obstruction_released= released,
            passed=blocking_pass))
    result = dict(format='jvs_dynamic_blockage_audit/v1', baseline_directory=str(baseline.resolve()),
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_completed=reference_valid, same_camera_trajectory=same_trajectory,
        baseline_telemetry_sha256=hashlib.sha256((baseline/'telemetry.csv').read_bytes()).hexdigest(),
        run_telemetry_sha256=hashlib.sha256((folder/'telemetry.csv').read_bytes()).hexdigest(),
        obstacles=reports,
        temporary_base_blockage_pass=bool(reference_valid and same_trajectory and any(r['passed'] for r in reports)),
        interpretation='Compare base collision geometry at equal camera-path progress. '
            'Each demo must contain a temporary nominal base gap below 2.5 cm coinciding '
            'with at least 3 cm of lateral displacement, 0.15 rad of base turning, '
            'or a measured base pause while the arm follows the moving camera reference. '
            'Other moving actors are diagnosed individually and may stay in the free aisle.')
    (folder/'blockage_report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    reports = {name: audit(args.runs/name, args.baseline/name) for name in ['crossing', 'overtake', 'two_movers']}
    (args.runs/'blockage_reports.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps({name: dict(passed=r['temporary_base_blockage_pass'], obstacles=r['obstacles'])
                      for name, r in reports.items()}, indent=2))
    return 0 if all(r['temporary_base_blockage_pass'] for r in reports.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
