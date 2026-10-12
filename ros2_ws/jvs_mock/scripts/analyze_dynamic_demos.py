#!/usr/bin/env python3
"""Independent truth-surface audit and report figures for real Isaac demo runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from jvs_mock.kc_svsdf import CollisionGeometry, primitive_sdf
from jvs_mock.dynamic_scene import model_axis_rotation


def read_rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def audit_motion_timing(rows, shots):
    """Check recorded action rates and elapsed durations on the physical clock."""
    physical = np.array([float(r['wall_elapsed_s']) for r in rows])
    path = np.array([float(r['time_s']) for r in rows])
    rates = np.array([float(r['path_rate']) for r in rows])
    tolerance = max((float(r['dt_s']) for r in rows), default=.02)+.002
    results = []
    for shot in shots:
        a = shot.get('motion_start_s', shot['start_s'])
        b = shot.get('motion_end_s', shot['end_s'])
        selected = (path >= a) & (path <= b)
        pairs = selected[1:] & selected[:-1]
        rate_error = float(np.max(np.abs(rates[selected]-1.))) if np.any(selected) else np.inf
        clock_error = float(np.max(np.abs(np.diff(path)[pairs]-np.diff(physical)[pairs]))) if np.any(pairs) else np.inf
        covered = bool(len(path) and path[0] <= a and path[-1] >= b)
        elapsed = float(np.interp(b, path, physical)-np.interp(a, path, physical)) if covered else None
        passed = bool(covered and rate_error <= 1e-8 and clock_error <= 1e-6
                      and abs(elapsed-(b-a)) <= tolerance)
        results.append(dict(type=shot['type'], planned_duration_s=b-a, physical_duration_s=elapsed,
            maximum_rate_error=rate_error, maximum_clock_increment_error_s=clock_error, passed=passed))
    return dict(motion_timing_pass=bool(results and all(r['passed'] for r in results)),
                motion_timing_tolerance_s=tolerance, motion_timing=results)


def match_moving_tracks(history, truth_tracks, maximum_separation=.30):
    """Audit actual detections at measurement time and forecasts separately.

    A cached detection is one observation, even when many map snapshots reuse
    it during occlusion. Extrapolation error does not create a new detection;
    it remains visible in the forecast statistics and detailed mismatch log.
    Collision acceptance independently uses actual robot/obstacle geometry
    and every physics-step contact report, rather than estimated centres.
    """
    def nearest(center, stamp):
        candidates = []
        for name, (physical, positions, velocities) in truth_tracks.items():
            position = np.array([np.interp(stamp, physical, positions[:, i]) for i in range(3)])
            velocity = np.array([np.interp(stamp, physical, velocities[:, i]) for i in range(3)])
            candidates.append((float(np.linalg.norm(np.asarray(center)[:2]-position[:2])), velocity, name))
        return min(candidates, key=lambda candidate: candidate[0])

    seen, measurement_errors, velocity_errors = set(), [], []
    matched, unmatched, forecast_errors, forecast_misses, ages = 0, 0, [], [], []
    by_obstacle = {name: 0 for name in truth_tracks}
    for observation in history:
        for track in observation['tracks']:
            if not track['moving']:
                continue
            stamp = observation['stamp']
            measured_at = track.get('measurement_stamp', stamp)
            predicted_error, _, predicted_name = nearest(track['center'], stamp)
            age = float(max(0., stamp-measured_at))
            forecast_errors.append(predicted_error)
            ages.append(age)
            if predicted_error >= maximum_separation:
                forecast_misses.append(dict(stamp=stamp, track_id=track['id'],
                    obstacle=predicted_name, separation_m=predicted_error,
                    prediction_age_s=age, measurement_stamp=measured_at,
                    surfaces=track.get('surfaces')))
            key = (track['id'], measured_at)
            if key in seen:
                continue
            seen.add(key)
            separation, velocity, name = nearest(track.get('measured_center', track['center']), measured_at)
            measurement_errors.append(separation)
            if separation >= maximum_separation:
                unmatched += 1
            elif np.linalg.norm(velocity) > .02:
                matched += 1
                by_obstacle[name] += 1
                velocity_errors.append(float(np.linalg.norm(np.asarray(track['velocity'])-velocity)))
    return dict(motion_matching_basis='distinct (track_id, measurement_stamp); measured centre at measurement time',
        matched_moving_observations=matched, unmatched_moving_observations=unmatched,
        matched_moving_observations_by_obstacle=by_obstacle,
        measured_moving_observations=len(seen),
        measured_center_error_max_m=max(measurement_errors, default=None),
        matched_velocity_error_rms_m_s=(float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else None),
        predicted_center_error_max_m=max(forecast_errors, default=None),
        maximum_prediction_age_s=max(ages, default=0.),
        predicted_unmatched_moving_observations=len(forecast_misses),
        predicted_unmatched_moving_observation_details=forecast_misses,
        center_matching_threshold_m=maximum_separation)


def surface(model, spacing=.025):
    if model['type'] == 'box':
        size = np.array(model['size'])
        axes = [np.linspace(-s / 2, s / 2, max(2, int(np.ceil(s / spacing)) + 1)) for s in size]
        points = []
        for fixed in range(3):
            other = [index for index in range(3) if index != fixed]
            for sign in (-1, 1):
                for a in axes[other[0]]:
                    for b in axes[other[1]]:
                        p = np.zeros(3)
                        p[fixed], p[other[0]], p[other[1]] = sign * size[fixed] / 2, a, b
                        points.append(p)
        return np.array(points), spacing / np.sqrt(2)
    radius = model['radius']
    angles = np.linspace(0, 2 * np.pi, max(12, int(np.ceil(2 * np.pi * radius / spacing))), endpoint=False)
    if model['type'] == 'cylinder':
        height = model['height']
        zs = np.linspace(-height / 2, height / 2, max(2, int(np.ceil(height / spacing)) + 1))
        side = [[radius * np.cos(a), radius * np.sin(a), z] for z in zs for a in angles]
        rings = np.linspace(0, radius, max(2, int(np.ceil(radius / spacing)) + 1))
        caps = [[r * np.cos(a), r * np.sin(a), z] for z in (-height / 2, height / 2) for r in rings for a in angles]
        return np.array(side + caps) @ model_axis_rotation(model).T, spacing / np.sqrt(2)
    polar = np.linspace(0, np.pi, max(4, int(np.ceil(np.pi * radius / spacing)) + 1))
    return np.array([[radius * np.sin(p) * np.cos(a), radius * np.sin(p) * np.sin(a), radius * np.cos(p)]
                     for p in polar for a in angles]), spacing / np.sqrt(2)


def static_audit(reference_path, folder, collision, states, rows):
    with np.load(reference_path) as data:
        reference_metadata = json.loads(str(data['metadata']))
        points, coverage = data['points'].astype(float), float(data['coverage'])
    scene_audit = json.loads((folder/'usd_scene_audit.json').read_text())
    if reference_metadata['usd_layer_sha256'] != scene_audit['usd_layer_sha256']:
        raise ValueError(f'static reference differs from the USD layers actually simulated: {folder}')
    spawn = json.loads((folder/'trajectory.json').read_text())['initial_base_pose']
    rotation = Rotation.from_euler('z', spawn['yaw'] - np.pi/2).as_matrix()
    origin = np.array([spawn['x'], -spawn['z'], spawn['height']])
    local = (points-origin) @ rotation
    tree = cKDTree(local)
    clearances = np.empty(len(rows))
    low, high = np.array(reference_metadata['roi_low']), np.array(reference_metadata['roi_high'])
    for i, (row, state) in enumerate(zip(rows, states)):
        tilt = Rotation.from_rotvec([float(row[f'base_rotvec_{axis}']) for axis in 'xyz']).as_matrix()
        frames = collision.frames(state, float(row['base_z']), tilt)[0]
        centers = np.array([frame[0] for frame in frames])
        native = centers @ rotation.T + origin
        if np.any(native[:, :2] < low[:2]+.35) or np.any(native[:, :2] > high[:2]-.35) or np.max(native[:, 2]) > high[2]-.35:
            raise ValueError('robot left the independently sampled USD reference volume')
        selected = sorted(set(index for batch in tree.query_ball_point(centers, .8) for index in batch))
        clearances[i] = (collision.distances(state, local[selected], float(row['base_z']), tilt).min()-coverage
                         if selected else .4)
    return clearances, reference_metadata


def analyze(folder, make_animation=False, static_reference=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metadata = json.loads((folder / 'run.json').read_text())
    summary = json.loads((folder / 'tracking_summary.json').read_text())
    rows = read_rows(folder / 'telemetry.csv')
    rows = [row for row in rows if float(row['wall_elapsed_s']) <= summary.get('execution_duration_s', np.inf) + .1]
    truth = read_rows(folder / 'obstacle_truth.csv')
    epoch = json.loads((folder / 'dynamic_epoch.json').read_text())['epoch_s']
    scene_path = Path(metadata['scene'])
    saved_scene = folder / 'obstacle_scene.json'
    scenario = json.loads((saved_scene if saved_scene.exists() else scene_path).read_text())
    states = np.array([[float(row[key]) for key in ['state_x', 'state_y', 'state_yaw',
                       *[f'state_q{i}' for i in range(1, 7)]]] for row in rows])
    actual_time = np.array([float(row['wall_elapsed_s']) for row in rows])
    error = np.array([float(row['position_error']) for row in rows])
    collision = CollisionGeometry()
    static_clearances, static_metadata = None, None
    if static_reference is not None:
        static_clearances, static_metadata = static_audit(static_reference, folder, collision, states, rows)
    clearances = np.full(len(rows), np.inf)
    obstacle_paths = {}
    truth_tracks = {}
    for obstacle in scenario['obstacles']:
        selected = [row for row in truth if row['obstacle'] == obstacle['name']]
        physical = np.array([float(row['simulation_time_s']) for row in selected])
        positions = np.array([[float(row[a]) for a in 'xyz'] for row in selected])
        velocities = np.array([[float(row['v' + a]) for a in 'xyz'] for row in selected])
        truth_tracks[obstacle['name']] = (physical, positions, velocities)
        interpolated = np.column_stack([np.interp(actual_time + epoch, physical, positions[:, i]) for i in range(3)])
        points, coverage = surface(obstacle['model'])
        for i, (row, state, center) in enumerate(zip(rows, states, interpolated)):
            tilt = Rotation.from_rotvec([float(row[f'base_rotvec_{axis}']) for axis in 'xyz']).as_matrix()
            distance = collision.distances(state, points + center, float(row['base_z']), tilt).min() - coverage
            # A robot entirely contained in a large obstacle has no nearby
            # obstacle-surface samples. Its body centres still detect overlap.
            frames = collision.frames(state, float(row['base_z']), tilt)[0]
            body_centers = np.array([frames[body][0] + frames[body][1] @ p
                                    for body, _, _, _, p, _ in collision.shapes])
            model = obstacle['model']
            dimensions = (np.asarray(model['size']) if model['type'] == 'box' else
                          np.array([model['radius'], model['height']]) if model['type'] == 'cylinder' else
                          np.array([model['radius']]))
            inside = primitive_sdf((body_centers - center) @ model_axis_rotation(model), model['type'], dimensions).min()
            if inside < 0:
                distance = min(distance, inside)
            clearances[i] = min(clearances[i], distance)
        obstacle_paths[obstacle['name']] = interpolated
    motion_audit = match_moving_tracks(json.loads((folder / 'track_history.json').read_text()), truth_tracks)
    contacts_path = folder / 'dynamic_contacts.json'
    contacts = json.loads(contacts_path.read_text()) if contacts_path.exists() else {}
    usd_path = folder/'usd_scene_audit.json'
    usd_audit = json.loads(usd_path.read_text()) if usd_path.exists() else {}
    static_scope_complete = False
    obstacle_scope_complete = False
    maximum_extent, maximum_obstacle_extent = None, None
    if usd_audit.get('collision_region_radius_m'):
        maximum_extent = 0.
        for row, state in zip(rows, states):
            tilt = Rotation.from_rotvec([float(row[f'base_rotvec_{axis}']) for axis in 'xyz']).as_matrix()
            frames = collision.frames(state, float(row['base_z']), tilt)[0]
            for body, _, kind, size, p, rotation in collision.shapes:
                center = frames[body][0]+frames[body][1]@p
                radius = (np.linalg.norm(size)/2 if kind=='box' else
                          np.hypot(size[0], size[1]/2) if kind=='cylinder' else
                          np.hypot(max(size[:2]), size[2]/2) if kind=='frustum' else size[0])
                maximum_extent = max(maximum_extent, np.linalg.norm(center[:2])+radius)
        static_scope_complete = bool(maximum_extent < usd_audit['collision_region_radius_m'])
        maximum_obstacle_extent = 0.
        for obstacle in scenario['obstacles']:
            model = obstacle['model']
            radius = (np.linalg.norm(model['size'])/2 if model['type'] == 'box' else
                      np.hypot(model['radius'], model['height']/2) if model['type'] == 'cylinder' else
                      model['radius'])
            positions = truth_tracks[obstacle['name']][1]
            maximum_obstacle_extent = max(maximum_obstacle_extent,
                float(np.linalg.norm(positions[:, :2], axis=1).max())+radius)
        obstacle_scope_complete = bool(maximum_obstacle_extent < usd_audit['collision_region_radius_m'])
    tracks = np.array([int(row['dynamic_tracks']) for row in rows])
    # Old recorded runs retain their original yielding evidence. New runs
    # have no risk-dependent path-rate ceiling and must prove fixed action time.
    yield_rate = np.array([float(row.get('dynamic_yield_rate', 1.)) for row in rows])
    path_rate = np.array([float(row['path_rate']) for row in rows])
    reference = json.loads((folder/'reference.json').read_text())
    timing_mode = reference.get('motion_timing_mode', 'adaptive')
    fixed_motion = timing_mode == 'fixed_physical_time'
    timing_audit = audit_motion_timing(rows, reference['execution_shots']) if fixed_motion else {}
    active = np.array([bool(int(row['motion_active'])) for row in rows])
    errors = error[active] if np.any(active) else error
    report = dict(name=folder.name, trajectory=Path(metadata['trajectory']).name,
        stage_view=metadata.get('stage_view'),
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        static_reference_sha256=(hashlib.sha256(Path(static_reference).read_bytes()).hexdigest()
                                 if static_reference is not None else None),
        scenario=scene_path.name, completed=summary.get('completed', False), failure=summary.get('failure', ''),
        verification='real ROS 2 / Isaac Sim; truth surfaces at recorded control states',
        dynamic_source='live timestamped dual LiDAR only; truth CSV used solely in this audit',
        position_rms_cm=float(np.sqrt(np.mean(errors ** 2)) * 100), position_peak_cm=float(np.max(errors) * 100),
        orientation_rms_rad=summary.get('orientation_error_rms'),
        execution_duration_s=summary.get('execution_duration_s'),
        minimum_sampled_dynamic_clearance_lower_m=float(clearances.min()),
        minimum_command_sweep_lower_m=min(float(row['command_sweep_lower_m']) for row in rows),
        maximum_tracks=int(tracks.max()), detected_moving_samples=int(np.sum(tracks > 0)),
        yielding_samples=int(np.sum(yield_rate < .95)),
        motion_timing_mode=timing_mode, **timing_audit,
        **motion_audit,
        physics_contact_audit_enabled=contacts.get('enabled', False),
        physics_contact_coverage_complete=bool(contacts.get('simulation_time_s', -1.) >= actual_time.max() + epoch - .002),
        audited_physics_steps=contacts.get('physics_steps', 0),
        robot_obstacle_contact_events=len(contacts.get('robot_obstacle_events', [])),
        robot_static_contact_events=len(contacts.get('robot_static_events', [])),
        obstacle_static_contact_events=len(contacts.get('obstacle_static_events', [])),
        static_collider_count=usd_audit.get('static_collider_count', 0),
        static_collision_scope_complete=static_scope_complete,
        obstacle_collision_scope_complete=obstacle_scope_complete,
        maximum_robot_extent_from_spawn_m=float(maximum_extent) if maximum_extent is not None else None,
        maximum_obstacle_extent_from_spawn_m=float(maximum_obstacle_extent) if maximum_obstacle_extent is not None else None,
        usd_referenced_layers_unchanged=usd_audit.get('referenced_layers_unchanged', False),
        minimum_sampled_static_clearance_lower_m=float(static_clearances.min()) if static_clearances is not None else None,
        static_reference_metadata=static_metadata,
        maximum_lidar_age_s=max(float(row['lidar_age_s']) for row in rows),
        maximum_command_constraint_violation=max(float(v) for k, v in summary.items()
            if k.startswith('max_command_') and 'violation' in k),
        tracking_accuracy_pass=summary.get('tracking_accuracy_pass', False),
        cinematic_pass=summary.get('approximate_success', False),
        physics_constraints_pass=summary.get('sampled_checks_pass', False),
        source_sha256=metadata.get('source_sha256', {}), sources_unchanged=metadata.get('sources_unchanged', False))
    report['dynamic_demo_pass'] = bool(report['completed'] and not report['failure']
        and report['minimum_sampled_dynamic_clearance_lower_m'] >= .025
        and report['minimum_command_sweep_lower_m'] >= .025 and report['detected_moving_samples'] > 0
        and all(count >= 3 for count in report['matched_moving_observations_by_obstacle'].values())
        and report['matched_moving_observations'] >= 3 and report['unmatched_moving_observations'] == 0
        and (report['motion_timing_pass'] if fixed_motion else report['yielding_samples'] > 0)
        and report['physics_contact_audit_enabled'] and report['robot_obstacle_contact_events'] == 0
        and report['physics_contact_coverage_complete'] and report['cinematic_pass']
        and report['tracking_accuracy_pass']
        and report['sources_unchanged'] and report['maximum_command_constraint_violation'] <= 1e-6)
    if usd_audit:
        report['dynamic_demo_pass'] &= bool(report['static_collider_count'] > 0
            and report['static_collision_scope_complete'] and report['obstacle_collision_scope_complete']
            and static_clearances is not None
            and report['usd_referenced_layers_unchanged'] and report['robot_static_contact_events'] == 0
            and report['obstacle_static_contact_events'] == 0)
    if static_clearances is not None:
        report['dynamic_demo_pass'] &= bool(static_clearances.min() >= .025)
    (folder / 'dynamic_report.json').write_text(json.dumps(report, indent=2) + '\n')
    audit_arrays = dict(physical_time=actual_time, dynamic_clearance_lower=clearances,
                       tracks=tracks, path_rate=path_rate)
    if not fixed_motion:
        audit_arrays['yield_rate'] = yield_rate
    if static_clearances is not None:
        audit_arrays['static_clearance_lower'] = static_clearances
    np.savez_compressed(folder / 'dynamic_audit.npz', **audit_arrays)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    ax = axes[0, 0]
    if static_reference is not None:
        with np.load(static_reference) as data:
            p = data['points']
        spawn = json.loads((folder/'trajectory.json').read_text())['initial_base_pose']
        r = Rotation.from_euler('z', spawn['yaw'] - np.pi/2).as_matrix()
        p = (p[p[:, 2] < .75] - [spawn['x'], -spawn['z'], spawn['height']]) @ r
        ax.scatter(p[::3, 0], p[::3, 1], s=.3, color='#aaaaaa', rasterized=True, label='USDA furniture surfaces')
    ax.plot(states[:, 0], states[:, 1], color='#225ea8', lw=2, label='Robot base (measured)')
    ax.scatter(states[0, 0], states[0, 1], c='black', marker='s', label='Start')
    for name, positions in obstacle_paths.items():
        model = next(obstacle['model'] for obstacle in scenario['obstacles'] if obstacle['name'] == name)
        color = model.get('color', '#e6550d')
        ax.plot(positions[:, 0], positions[:, 1], '--', color=color, lw=2, label=name)
        previous_label = None
        for t in np.linspace(max(0., actual_time.min()), min(45., actual_time.max()), 5):
            i = np.argmin(abs(actual_time - t))
            if previous_label is not None and np.linalg.norm(positions[i, :2] - previous_label) < .1:
                continue
            previous_label = positions[i, :2]
            ax.scatter(*positions[i, :2], color=color, s=25)
            ax.annotate(f'{t:.0f}s', positions[i, :2], xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Initial-base forward (m)', ylabel='Initial-base left (m)', title=report['trajectory'])
    all_paths = np.vstack([states[:, :2], *[p[:, :2] for p in obstacle_paths.values()]])
    low, high = all_paths.min(axis=0)-.65, all_paths.max(axis=0)+.65
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8)
    axes[0, 1].plot(actual_time, error * 100, label='Camera position error')
    axes[0, 1].axhline(5, ls='--', c='gray', label='5 cm reference')
    axes[0, 1].set(xlabel='Physical time after scene start (s)', ylabel='cm', title='Camera trajectory tracking')
    axes[0, 1].legend()
    axes[1, 0].plot(actual_time, clearances * 100, label='Independent dynamic truth lower bound')
    if static_clearances is not None:
        axes[1, 0].plot(actual_time, static_clearances * 100, label='Independent USDA furniture lower bound', alpha=.8)
    axes[1, 0].plot(actual_time, [float(row['command_sweep_lower_m']) * 100 for row in rows], label='LiDAR command sweep lower bound', alpha=.7)
    axes[1, 0].axhline(2.5, ls='--', c='red', label='Required clearance')
    axes[1, 0].set(xlabel='Physical time (s)', ylabel='cm', ylim=(0, 100),
                   title='Whole-body clearance (0–100 cm view)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(actual_time, tracks, label='Motion-classified tracks')
    axes[1, 1].plot(actual_time, path_rate, label='Executed camera-path rate')
    if not fixed_motion:
        axes[1, 1].plot(actual_time, yield_rate, label='Historical dynamic path-rate ceiling')
    axes[1, 1].set(xlabel='Physical time (s)', title='Perception and action timing')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    figure.suptitle(f"Dynamic KC-WBC: {folder.name}")
    figure.savefig(folder / 'report.png', dpi=160)
    figure.savefig(folder / 'report.pdf')
    plt.close(figure)
    if make_animation:
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Circle
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(states[:, 0], states[:, 1], color='#225ea8', alpha=.3)
        all_points = np.vstack([states[:, :2], *[p[:, :2] for p in obstacle_paths.values()]])
        low, high = all_points.min(axis=0) - .4, all_points.max(axis=0) + .4
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect='equal',
               xlabel='Forward (m)', ylabel='Left (m)', title='Measured Isaac playback (top view)')
        robot = Circle(states[0, :2], .18, color='#225ea8', alpha=.8)
        ax.add_patch(robot)
        movers = {}
        for obstacle in scenario['obstacles']:
            model = obstacle['model']
            radius = model.get('radius', np.linalg.norm(model.get('size', [.2, .2])[:2]) / 2)
            movers[obstacle['name']] = Circle(obstacle_paths[obstacle['name']][0, :2], radius,
                                              color=model.get('color', [.9, .3, .1]), alpha=.8)
            ax.add_patch(movers[obstacle['name']])
        title = ax.text(.02, .97, '', transform=ax.transAxes, va='top')
        indices = np.searchsorted(actual_time, np.arange(actual_time[0], actual_time[-1], .35)).clip(0, len(rows) - 1)
        def frame(index):
            i = indices[index]
            robot.center = states[i, :2]
            for name, patch in movers.items():
                patch.center = obstacle_paths[name][i, :2]
            title.set_text(f't={actual_time[i]:.1f}s  camera error={error[i]*100:.1f} cm')
            return [robot, *movers.values(), title]
        animation = FuncAnimation(fig, frame, frames=len(indices), interval=70, blit=True)
        animation.save(folder / 'playback.gif', writer=PillowWriter(fps=14))
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--animation', action='store_true')
    parser.add_argument('--static-reference', type=Path, help='independent dense USDA surface reference, never a controller input')
    args = parser.parse_args()
    folders = [args.runs] if (args.runs / 'run.json').exists() else sorted(
        path.parent for path in args.runs.glob('*/run.json')
        if (path.parent/'tracking_summary.json').is_file())
    reports = [analyze(folder, args.animation, args.static_reference) for folder in folders]
    (args.runs / 'comparison.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
    return 0 if reports and all(report['dynamic_demo_pass'] for report in reports) else 2


if __name__ == '__main__':
    raise SystemExit(main())
