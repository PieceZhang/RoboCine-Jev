"""Measure rapid camera motion on a physical-time grid, including image roll."""
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from .whole_body_qp import whole_body_kinematics


def motion_residual(time, position, rotation, *, fps=30., cutoff_hz=3.):
    time = np.asarray(time, dtype=float)
    position = np.asarray(position, dtype=float)
    if len(time) < 20 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError('at least 20 strictly increasing physical timestamps required')
    if position.shape != (len(time), 3) or not np.all(np.isfinite(position)):
        raise ValueError('camera positions must be finite Nx3 values')
    if not 0 < cutoff_hz < fps / 2 or time[-1] - time[0] < 1.:
        raise ValueError('invalid cutoff, sampling rate or duration')
    grid = np.arange(time[0], time[-1], 1. / fps)
    sampled_position = np.column_stack([np.interp(grid, time, position[:, i]) for i in range(3)])
    sampled_rotation = Slerp(time, rotation)(grid)
    quaternion = sampled_rotation.as_quat()
    # Keep a continuous quaternion lift across pi; componentwise sign selection
    # against w would introduce false spikes during large chassis turns.
    for i in range(1, len(quaternion)):
        if np.dot(quaternion[i-1], quaternion[i]) < 0:
            quaternion[i] *= -1
    filter_ = butter(4, cutoff_hz, fs=fps, output='sos')
    trend = sosfiltfilt(filter_, quaternion, axis=0)
    trend /= np.linalg.norm(trend, axis=1, keepdims=True)
    orientation = (sampled_rotation * Rotation.from_quat(trend).inv()).as_rotvec()
    centered_position = detrend(sampled_position, axis=0, type='linear')
    position_residual = centered_position - sosfiltfilt(filter_, centered_position, axis=0)
    return grid, position_residual, orientation


def load_telemetry(folder):
    with (Path(folder) / 'telemetry.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    column = lambda name: np.array([float(row[name]) for row in rows])
    time = column('wall_elapsed_s')
    position = np.column_stack([column('camera_' + a) for a in 'xyz'])
    base = Rotation.from_rotvec(np.column_stack([column('base_rotvec_' + a) for a in 'xyz']))
    rotation = []
    for row, br in zip(rows, base.as_matrix()):
        state = np.array([float(row['state_' + name]) for name in
                          ['x', 'y', 'yaw', *[f'q{i}' for i in range(1, 7)]]])
        rotation.append(whole_body_kinematics(state, float(row['base_z']), br)[1])
    return time, position, Rotation.from_matrix(rotation), rows


def load_physics(folder):
    with (Path(folder) / 'camera_physics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if np.isfinite(float(row['scene_time_s']))]
    column = lambda name: np.array([float(row[name]) for row in rows])
    time = column('scene_time_s')
    position = np.column_stack([column('camera_' + a) for a in 'xyz'])
    rotation = Rotation.from_quat(np.column_stack([column('camera_' + a) for a in ['qx', 'qy', 'qz', 'qw']]))
    return time, position, rotation


def summarize_physics(folder, *, cutoff_hz=3., rms_limit_deg=.02, peak_limit_deg=.08,
                      wobble_cutoff_hz=1., wobble_rms_limit_deg=.25, wobble_peak_limit_deg=.75):
    """Audit the unfiltered camera body at video, viewport and physics rates.

    Scene time zero is the recording origin. Control samples supply phase
    labels only; all camera poses come from the live rigid-body measurements.
    Slow closed-loop wobble must pass independently of the rapid-shake test.
    """
    folder = Path(folder)
    physical_time, position, rotation = load_physics(folder)
    control_time, _, _, rows = load_telemetry(folder)
    first = max(0, int(np.searchsorted(physical_time, 0., side='right')) - 1)
    physical_time = physical_time[first:].copy()
    position, rotation = position[first:], rotation[first:]
    if physical_time[0] < 0.:
        physical_time[0] = 0.
    report = dict(format='jvs_physical_camera_stability/v1',
        result_directory=str(folder.resolve()),
        source='unfiltered camera_link rigid-body world pose captured every PhysX step',
        physical_pose_overrides=False, image_stabilization=False, cutoff_hz=cutoff_hz,
        source_sha256={name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                       for name in ('camera_physics.csv', 'telemetry.csv')},
        excluded_video_edges_s=.5,
        thresholds=dict(orientation_rms_deg=rms_limit_deg, orientation_peak_deg=peak_limit_deg),
        slow_wobble_thresholds=dict(cutoff_hz=wobble_cutoff_hz,
            orientation_rms_deg=wobble_rms_limit_deg, orientation_peak_deg=wobble_peak_limit_deg),
        frequencies={})
    for frequency in (30., 60., 1000.):
        grid, dp, dr = motion_residual(physical_time, position, rotation,
                                      fps=frequency, cutoff_hz=cutoff_hz)
        angle = np.rad2deg(np.linalg.norm(dr, axis=1))
        distance = 1000. * np.linalg.norm(dp, axis=1)
        _, wobble_dp, wobble_dr = motion_residual(physical_time, position, rotation,
            fps=frequency, cutoff_hz=wobble_cutoff_hz)
        wobble_angle = np.rad2deg(np.linalg.norm(wobble_dr, axis=1))
        wobble_distance = 1000. * np.linalg.norm(wobble_dp, axis=1)
        # Use the last control sample reached by the recording clock.
        selected = np.clip(np.searchsorted(control_time, grid, side='right') - 1,
                           0, len(control_time) - 1)
        turn = np.array([abs(float(rows[i]['command_omega'])) > .15 for i in selected])
        hold = np.array([float(rows[i]['transition_active']) > .5 for i in selected])
        shooting = np.array([float(rows[i]['motion_active']) > .5 for i in selected])
        interior = (grid >= .5) & (grid <= control_time[-1] - .5)
        windows = {}
        wobble_windows = {}
        for name, condition in (('all', np.ones(len(grid), bool)), ('turning', turn),
                                ('hold', hold), ('turning_hold', turn & hold),
                                ('shooting', shooting)):
            mask = interior & condition
            windows[name] = dict(samples=int(mask.sum()))
            wobble_windows[name] = dict(samples=int(mask.sum()))
            if mask.any():
                windows[name].update(
                    orientation_rms_deg=float(np.sqrt(np.mean(angle[mask]**2))),
                    orientation_peak_deg=float(angle[mask].max()),
                    position_rms_mm=float(np.sqrt(np.mean(distance[mask]**2))),
                    position_peak_mm=float(distance[mask].max()))
                wobble_windows[name].update(
                    orientation_rms_deg=float(np.sqrt(np.mean(wobble_angle[mask]**2))),
                    orientation_peak_deg=float(wobble_angle[mask].max()),
                    position_rms_mm=float(np.sqrt(np.mean(wobble_distance[mask]**2))),
                    position_peak_mm=float(wobble_distance[mask].max()))
        checked = [item for item in windows.values() if item['samples']]
        passed = bool(checked and all(
            item['orientation_rms_deg'] <= rms_limit_deg
            and item['orientation_peak_deg'] <= peak_limit_deg for item in checked))
        wobble_checked = [wobble_windows[name] for name in ('hold', 'turning_hold', 'shooting')
                          if wobble_windows[name]['samples']]
        wobble_passed = bool(wobble_checked and all(
            item['orientation_rms_deg'] <= wobble_rms_limit_deg
            and item['orientation_peak_deg'] <= wobble_peak_limit_deg for item in wobble_checked))
        worst_ids = np.flatnonzero(interior)[np.argsort(angle[interior])[-8:]]
        report['frequencies'][str(int(frequency))] = dict(windows=windows,
            slow_wobble_windows=wobble_windows, fast_shake_pass=passed,
            slow_wobble_pass=wobble_passed, pass_=passed and wobble_passed,
            worst_samples=[dict(time_s=float(grid[i]), orientation_deg=float(angle[i]),
                                position_mm=float(distance[i])) for i in worst_ids])
    report['camera_stability_pass'] = all(item['pass_'] for item in report['frequencies'].values())
    return report


def summarize(folder, *, fps=30., cutoff_hz=3., rms_limit_deg=.02, peak_limit_deg=.08):
    time, position, rotation, rows = load_telemetry(folder)
    grid, dp, dr = motion_residual(time, position, rotation, fps=fps, cutoff_hz=cutoff_hz)
    column = lambda name: np.array([float(row[name]) for row in rows])
    interior = (grid >= .5) & (grid <= time[-1] - .5)
    turn = np.abs(np.interp(grid, time, column('command_omega'))) > .15
    hold = np.interp(grid, time, column('transition_active')) > .5
    shooting = np.interp(grid, time, column('motion_active')) > .5
    angle = np.rad2deg(np.linalg.norm(dr, axis=1))
    distance = 1000 * np.linalg.norm(dp, axis=1)

    def statistics(mask):
        if not mask.any():
            return dict(samples=0)
        return dict(samples=int(mask.sum()),
            orientation_rms_deg=float(np.sqrt(np.mean(angle[mask]**2))),
            orientation_peak_deg=float(np.max(angle[mask])),
            position_rms_mm=float(np.sqrt(np.mean(distance[mask]**2))),
            position_peak_mm=float(np.max(distance[mask])))

    windows = {name: statistics(interior & mask) for name, mask in (
        ('all', np.ones(len(grid), bool)), ('turning', turn), ('hold', hold),
        ('turning_hold', turn & hold), ('shooting', shooting))}
    worst = np.argsort(angle[interior])[-8:]
    worst_ids = np.flatnonzero(interior)[worst]
    result = dict(format='jvs_camera_stability/v1', result_directory=str(Path(folder).resolve()),
        source='measured full base pose and measured joint angles; URDF camera FK',
        fps=fps, cutoff_hz=cutoff_hz, filter='zero-phase fourth-order Butterworth; full SO(3) residual',
        excluded_edges_s=.5, turn_command_threshold_radps=.15,
        thresholds=dict(orientation_rms_deg=rms_limit_deg, orientation_peak_deg=peak_limit_deg),
        windows=windows, worst_samples=[dict(time_s=float(grid[i]),
            orientation_deg=float(angle[i]), position_mm=float(distance[i])) for i in worst_ids])
    checked = [windows[name] for name in ('turning', 'hold', 'shooting') if windows[name]['samples']]
    result['camera_stability_pass'] = bool(checked and all(
        item['orientation_rms_deg'] <= rms_limit_deg and item['orientation_peak_deg'] <= peak_limit_deg
        for item in checked))
    audit = Path(folder) / 'camera_physics.csv'
    if audit.exists():
        pt, pp, pr = load_physics(folder)
        pg, pdp, pdr = motion_residual(pt, pp, pr, fps=1000., cutoff_hz=cutoff_hz)
        physical = (pg >= .5) & (pg <= time[-1] - .5)
        residual = np.rad2deg(np.linalg.norm(pdr[physical], axis=1))
        result['physics'] = dict(samples=len(pt), median_step_s=float(np.median(np.diff(pt))),
            orientation_rms_deg=float(np.sqrt(np.mean(residual**2))),
            orientation_peak_deg=float(residual.max()),
            position_rms_mm=float(1000*np.sqrt(np.mean(np.sum(pdp[physical]**2, axis=1)))))
    return result
