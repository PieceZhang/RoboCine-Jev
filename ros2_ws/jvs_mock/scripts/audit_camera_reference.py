#!/usr/bin/env python3
"""Measure slow camera drift against the intended shot, without detrending it."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jvs_mock.camera_stability import load_physics
from jvs_mock.wbc_trajectory import CameraTrajectory


def orientation_errors(actual, reference):
    relative = reference.inv()*actual
    matrix = relative.as_matrix()
    return (np.rad2deg(np.arctan2(matrix[:, 2, 1], matrix[:, 2, 2])),
            np.rad2deg(relative.magnitude()))


def summarize(folder, window=(7., 11.), roll_limit_deg=.5):
    folder = Path(folder)
    reference = json.loads((folder/'reference.json').read_text())
    document = json.loads((folder/'trajectory.json').read_text())
    trajectory = CameraTrajectory(folder/'trajectory.json',
        np.array(reference['initial_camera_position']), np.array(reference['initial_camera_rotation']),
        np.array(reference['initial_base_rotation']), transition_options=reference['transition_options'])
    with (folder/'telemetry.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    control_time = np.array([float(row['wall_elapsed_s']) for row in rows])
    path_time = np.array([float(row['time_s']) for row in rows])
    time, position, rotation = load_physics(folder)
    summary = json.loads((folder/'tracking_summary.json').read_text())
    if (not summary.get('completed') or summary.get('failure')
            or time[0] > window[0] or min(time[-1], control_time[-1]) < window[1]):
        raise ValueError('a completed recording covering the full requested window is required')
    spawn = document['initial_base_pose']
    initial = Rotation.from_euler('z', spawn['yaw']-np.pi/2)
    origin = np.array([spawn['x'], -spawn['z'], spawn['height']])
    grid = np.arange(max(0., time[0], control_time[0]), min(time[-1], control_time[-1]), .001)
    actual = initial.inv()*Slerp(time, rotation)(grid)
    position = initial.inv().apply(np.column_stack([
        np.interp(grid, time, position[:, i]) for i in range(3)])-origin)
    progress = np.interp(grid, control_time, path_time)
    samples = [trajectory.sample(t) for t in progress]
    planned = Rotation.from_matrix(np.array([s[1] for s in samples]))
    roll, angle = orientation_errors(actual, planned)
    position_error = np.linalg.norm(position-np.array([s[0] for s in samples]), axis=1)
    roll_rate = np.gradient(np.unwrap(np.deg2rad(roll)), grid)*180/np.pi
    windows = {}
    for label, mask in [('requested_window', (grid >= window[0]) & (grid <= window[1])),
                        ('shooting', trajectory.motion_mask(progress))]:
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError('recording does not cover the requested window or shooting')
        worst = indices[np.argmax(abs(roll[mask]))]
        windows[label] = dict(samples=int(mask.sum()),
            roll_rms_deg=float(np.sqrt(np.mean(roll[mask]**2))),
            roll_peak_deg=float(np.max(abs(roll[mask]))), worst_roll_time_s=float(grid[worst]),
            orientation_peak_deg=float(angle[mask].max()),
            roll_rate_peak_deg_s=float(np.max(abs(roll_rate[mask]))),
            position_rms_cm=float(100*np.sqrt(np.mean(position_error[mask]**2))),
            position_peak_cm=float(100*position_error[mask].max()),
            pass_=bool(np.max(abs(roll[mask])) <= roll_limit_deg))
    return dict(format='jvs_camera_reference_audit/v1', result_directory=str(folder.resolve()),
        source='unfiltered physical camera pose, compared with reference at measured path progress',
        physical_window_s=list(window), roll_limit_deg=roll_limit_deg,
        source_sha256={name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for name in ('camera_physics.csv', 'telemetry.csv', 'reference.json', 'trajectory.json')},
        windows=windows, reference_roll_pass=all(w['pass_'] for w in windows.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path, nargs='+')
    parser.add_argument('--window', type=float, nargs=2, default=(7., 11.))
    parser.add_argument('--roll-limit-deg', type=float, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.window[0] < args.window[1] or not np.isfinite(args.roll_limit_deg) or args.roll_limit_deg <= 0:
        parser.error('invalid window or roll limit')
    reports = [summarize(folder, args.window, args.roll_limit_deg) for folder in args.records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
