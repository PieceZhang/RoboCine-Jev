#!/usr/bin/env python3
"""Measure camera motion and discontinuous base guidance during combo holds.

The high-pass metrics use a 0.2 s Gaussian trend on a 50 Hz resampling of
physical simulation time. They describe recorded motion, not a safety or
full-physics certificate. Trajectories are resolved by recorded content hashes.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_record import resolve_recorded_trajectory
from jvs_mock.whole_body_qp import whole_body_kinematics


def inspect(folder):
    folder = Path(folder)
    package = Path(__file__).resolve().parents[1]
    metadata = json.loads((folder/'run_metadata.json').read_text())
    trajectory_path = resolve_recorded_trajectory(package, metadata, folder.name, folder)
    trajectory = json.loads(trajectory_path.read_text())
    reference_path = folder/'reference.json'
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        trajectory['shots'] = reference.get('execution_shots', trajectory['shots'])
    with (folder/'telemetry.csv').open() as stream:
        rows = list(csv.DictReader(stream))

    def column(name):
        return np.array([float(row[name]) for row in rows])

    time = column('time_s')
    clock = column('wall_elapsed_s')
    positions = np.column_stack([column('camera_'+axis) for axis in 'xyz'])
    forward = np.column_stack([column('camera_forward_'+axis) for axis in 'xyz'])
    headings = column('heading_target')
    directions = column('moveout_direction')
    windows = []
    traces = []
    for previous, following in zip(trajectory['shots'], trajectory['shots'][1:]):
        start = previous.get('motion_end_s', previous['end_s'])
        end = following.get('motion_start_s', following['start_s'])
        ids = np.flatnonzero((time >= start) & (time < end))
        if end <= start or len(ids) < 3:
            continue
        # Include the incoming/outgoing sample when checking boundary jumps.
        boundary = np.arange(max(0, ids[0]-1), min(len(time), ids[-1]+2))
        heading_steps = np.abs((np.diff(headings[boundary])+np.pi) % (2*np.pi)-np.pi)
        grid = np.arange(clock[ids[0]], clock[ids[-1]], .02)
        position = np.column_stack([np.interp(grid, clock[ids], positions[ids, axis])
                                    for axis in range(3)])
        direction = np.column_stack([np.interp(grid, clock[ids], forward[ids, axis])
                                     for axis in range(3)])
        rotations = []
        for index in ids:
            row = rows[index]
            state = np.array([float(row['state_'+name]) for name in ['x', 'y', 'yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']])
            base = Rotation.from_rotvec([float(row['base_rotvec_'+axis]) for axis in 'xyz']).as_matrix()
            rotations.append(whole_body_kinematics(state, float(row['base_z']), base)[1])
        relative = Rotation.from_matrix(np.array(rotations)@rotations[0].T).as_rotvec()
        orientation = np.column_stack([np.interp(grid, clock[ids], relative[:, axis]) for axis in range(3)])
        rotation_highpass = orientation-gaussian_filter1d(orientation, .2/.02, axis=0)
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        trend = gaussian_filter1d(position, .2/.02, axis=0)
        view_trend = gaussian_filter1d(direction, .2/.02, axis=0)
        view_trend /= np.maximum(np.linalg.norm(view_trend, axis=1, keepdims=True), 1e-12)
        highpass = position-trend
        view_error = np.arccos(np.clip(np.sum(direction*view_trend, axis=1), -1., 1.))
        interior = (grid > grid[0]+.4) & (grid < grid[-1]-.4)
        if not np.any(interior):
            interior = np.ones(len(grid), dtype=bool)
        name = previous['type']+' -> '+following['type']
        windows.append(dict(junction=name, start_s=start, end_s=end,
            physical_duration_s=float(clock[ids[-1]]-clock[ids[0]]),
            position_span_mm=float(1000*np.linalg.norm(np.ptp(positions[ids], axis=0))),
            position_highpass_rms_mm=float(1000*np.sqrt(np.mean(np.sum(highpass[interior]**2, axis=1)))),
            view_highpass_rms_deg=float(np.rad2deg(np.sqrt(np.mean(view_error[interior]**2)))),
            orientation_highpass_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(rotation_highpass[interior]**2, axis=1))))),
            maximum_camera_step_mm=float(1000*np.max(np.linalg.norm(np.diff(positions[ids], axis=0), axis=1))),
            maximum_heading_step_deg=float(np.rad2deg(np.max(heading_steps))),
            heading_steps_over_15_deg=int(np.count_nonzero(heading_steps > np.deg2rad(15.))),
            drive_branch_changes=int(np.count_nonzero(np.diff(directions[ids]))),
            position_error_max_mm=float(1000*np.max(column('position_error')[ids])),
            minimum_command_clearance_mm=float(1000*np.min(column('command_sweep_lower_m')[ids]))))
        traces.append((name, clock[ids]-clock[ids[0]], positions[ids]-positions[ids[0]],
                       np.unwrap(headings[ids]), column('command_omega')[ids]))
    final_summary = json.loads((folder/'summary.json').read_text())
    captured_summary = folder/'tracking_summary.json'
    summary = (json.loads(captured_summary.read_text()) if captured_summary.exists()
               else final_summary)
    result = dict(record=str(folder.resolve()), trajectory=Path(metadata.get('trajectory', trajectory_path)).name,
        trajectory_sha256=metadata['trajectory_sha256'],
        completed=summary.get('completed'), failure=summary.get('failure'),
        acceptance_summary=captured_summary.name if captured_summary.exists() else 'summary.json',
        shutdown_completed=final_summary.get('completed'), shutdown_failure=final_summary.get('failure'),
        tracking_accuracy_pass=summary.get('tracking_accuracy_pass'),
        approximate_success=summary.get('approximate_success'),
        position_error_max_mm=1000*summary['position_error_max'], windows=windows)
    return result, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    options = parser.parse_args()
    inspected = [inspect(folder) for folder in options.records]
    report = dict(metric_scope='ROS control-rate samples; 0.2 s Gaussian high-pass, 50 Hz resampling',
                  records=[result for result, _ in inspected])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    if options.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        count = sum(len(traces) for _, traces in inspected)
        figure, axes = plt.subplots(count, 3, figsize=(13, 2.6*count), squeeze=False)
        index = 0
        for result, traces in inspected:
            label = Path(result['record']).parent.name+'/'+Path(result['record']).name
            for name, clock, position, heading, omega in traces:
                row = axes[index]
                for axis, color in enumerate(['#297aaf', '#c17138', '#439970']):
                    row[0].plot(clock, 1000*position[:, axis], color=color, label='xyz'[axis])
                row[0].set_ylabel(label+'\n'+name+'\nPosition [mm]')
                row[0].legend(loc='upper right', ncol=3, fontsize=8)
                row[1].plot(clock, np.rad2deg(heading), color='#297aaf')
                row[1].set_ylabel('Base heading target [deg]')
                row[2].plot(clock, omega, color='#c17138')
                row[2].set_ylabel('Command omega [rad/s]')
                for axis in row:
                    axis.set_xlabel('Simulation time since hold began [s]')
                    axis.grid(alpha=.2)
                index += 1
        figure.tight_layout()
        options.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(options.plot, dpi=150)
        plt.close(figure)
    for result, _ in inspected:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
