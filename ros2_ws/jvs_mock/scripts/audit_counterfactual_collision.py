#!/usr/bin/env python3
"""Prove simultaneous solid overlap on a measured unobstructed trajectory.

This is an offline geometric counterfactual, not a simulated contact experiment.
No baseline, actor trajectory, or truth data is supplied to the controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from analyze_dynamic_demos import read_rows, surface
from jvs_mock.dynamic_scene import DynamicScene, model_axis_rotation
from jvs_mock.kc_svsdf import CollisionGeometry, primitive_sdf

STATE_KEYS = ['state_x', 'state_y', 'state_yaw', *[f'state_q{i}' for i in range(1, 7)]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_collision_planes():
    """Use the imported URDF STL hull, rather than its circumscribed SDF proxy."""
    path = Path(__file__).resolve().parents[1] / 'meshes/base_frustum.stl'
    data = path.read_bytes()
    count = struct.unpack('<I', data[80:84])[0]
    if len(data) == 84 + 50 * count:
        triangles = np.frombuffer(data, offset=84, count=count,
            dtype=np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]))
        vertices = triangles['vertices'].reshape(-1, 3)
    else:
        vertices = np.array([[float(value) for value in line.split()[1:]]
                             for line in data.decode().splitlines() if line.strip().startswith('vertex ')])
    return ConvexHull(np.unique(vertices, axis=0)).equations, path


def evaluate(baseline, scene_path, body, min_encounters=1):
    if min_encounters < 1:
        raise ValueError('min_encounters must be positive')
    rows = read_rows(baseline / 'telemetry.csv')
    summary_path = baseline / 'tracking_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    spawn = json.loads((baseline / 'trajectory.json').read_text())['initial_base_pose']
    scene = DynamicScene(scene_path)
    geometry = CollisionGeometry()
    shapes = [shape for shape in geometry.shapes if
              (shape[1] == 'base_link' if body == 'base' else shape[1] in [f'link{i}' for i in range(1, 7)])]
    base_planes, base_mesh = base_collision_planes() if body == 'base' else (None, None)
    selected = [row for row in rows if row['phase'] == 'tracking' and row['motion_active'] == '1']
    reports = []
    for actor in scene.obstacles:
        model = actor['model']
        points, _ = surface(model, spacing=.0125)
        dimensions = (model['size'] if model['type'] == 'box' else
                      [model['radius'], model['height']] if model['type'] == 'cylinder' else [model['radius']])
        gaps, links = [], []
        for row in selected:
            t = float(row['wall_elapsed_s'])
            center = scene.to_initial_base(scene.sample(actor['name'], t)[0], spawn)
            x = np.array([float(row[key]) for key in STATE_KEYS])
            tilt = Rotation.from_rotvec([float(row['base_rotvec_' + a]) for a in 'xyz']).as_matrix()
            frames = geometry.frames(x, float(row['base_z']), tilt)[0]
            distances = []
            for b, link, kind, size, p, rotation in shapes:
                bp, br = frames[b]
                solid_center = bp + br @ p
                # No coverage, safety margin, or inflation is subtracted.
                # Negative means a sampled actor surface lies inside the
                # actual URDF collision solid, proving intersecting volumes.
                local = (points + center - solid_center) @ (br @ rotation)
                distances_to_solid = (np.max(local @ base_planes[:, :3].T + base_planes[:, 3], axis=1)
                                      if link == 'base_link' else primitive_sdf(local, kind, size))
                gap = float(distances_to_solid.min())
                inside = float(primitive_sdf((solid_center - center) @ model_axis_rotation(model),
                                            model['type'], dimensions))
                if inside < 0:
                    gap = min(gap, inside)
                distances.append((gap, link))
            gap, link = min(distances)
            gaps.append(gap)
            links.append(link)
        gaps = np.array(gaps)
        overlap = gaps < -.005
        times = np.array([float(row['wall_elapsed_s']) for row in selected])
        duration = float(sum(float(row['dt_s']) for row, hit in zip(selected, overlap) if hit))
        worst = int(np.argmin(gaps))
        hit = np.flatnonzero(overlap)
        # Respect the recorded sampling period: a continuous low-rate overlap
        # must not count as a new encounter at every sample. Still split gaps
        # left by inactive motion intervals or missing telemetry.
        periods = np.array([float(row['dt_s']) for row in selected])
        gap_limit = np.maximum(.1, 1.5 * periods[hit[1:]])
        groups = np.split(hit, np.flatnonzero((np.diff(hit) > 1) | (np.diff(times[hit]) > gap_limit)) + 1)
        intervals = [dict(start_s=float(times[group[0]]), end_s=float(times[group[-1]]),
                          duration_s=float(sum(float(selected[i]['dt_s']) for i in group)))
                     for group in groups if len(group)]
        encounter_count = sum(interval['duration_s'] >= .1 for interval in intervals)
        reports.append(dict(obstacle=actor['name'], body_group=body,
            minimum_uninflated_solid_gap_m=float(gaps[worst]), worst_link=links[worst],
            worst_physical_time_s=float(times[worst]), overlap_duration_s=duration,
            first_overlap_physical_time_s=float(times[overlap][0]) if np.any(overlap) else None,
            last_overlap_physical_time_s=float(times[overlap][-1]) if np.any(overlap) else None,
            overlap_intervals=intervals,
            qualifying_encounter_count=encounter_count,
            simultaneous_collision_pass=bool(encounter_count >= min_encounters)))
    metadata = json.loads((baseline / 'run.json').read_text())
    return dict(format='jvs_counterfactual_solid_collision/v1',
        baseline_directory=str(baseline.resolve()),
        baseline_completed=bool(summary.get('completed') and not summary.get('failure')),
        scene=str(scene_path.resolve()), scene_sha256=digest(scene_path),
        baseline_telemetry_sha256=digest(baseline / 'telemetry.csv'),
        baseline_trajectory_sha256=digest(baseline / 'trajectory.json'),
        baseline_controller_sources_sha256={p: v for p, v in metadata['source_sha256'].items()
                                            if p.endswith(('.py', '.yaml'))},
        audit_script_sha256=digest(__file__), obstacles=reports,
        base_collision_mesh_sha256=digest(base_mesh) if base_mesh else None,
        required_encounter_count=min_encounters,
        simultaneous_collision_pass=any(item['simultaneous_collision_pass'] for item in reports),
        collision_definition='Same physical elapsed time; each distinct encounter requires at least 5 mm of uninflated solid penetration for at least 0.1 s. '
            'Robot link cylinders use URDF collision geometry; base uses the actual STL convex hull imported by PhysX. '
            'A historical center-path crossing or safety-buffer overlap cannot pass.')


def same_runtime_sources(first, second):
    """Compare complete named runtime inputs across isolated package roots."""
    required = {'dynamic_whole_body_controller.py', 'whole_body_controller.py',
                'whole_body_controller.yaml', 'shot_whole_body_controller.yaml'}
    def named(metadata):
        pairs = [(Path(p).name, h) for p, h in metadata['source_sha256'].items()
                 if p.endswith(('.py', '.yaml'))]
        if len({p for p, _ in pairs}) != len(pairs):
            return {}
        return dict(pairs)
    a, b = named(first), named(second)
    return required <= a.keys() and required <= b.keys() and a == b


def compare_response(report, run, baseline):
    metadata = json.loads((run / 'run.json').read_text())
    nominal_metadata = json.loads((baseline / 'run.json').read_text())
    same_sources = same_runtime_sources(metadata, nominal_metadata)
    rows = read_rows(run / 'telemetry.csv')
    reference = read_rows(baseline / 'telemetry.csv')
    ref_path = np.array([float(row['time_s']) for row in reference])
    ref_state = np.array([[float(row[k]) for k in STATE_KEYS] for row in reference])
    ref_state[:, 2] = np.unwrap(ref_state[:, 2])
    path = np.array([float(row['time_s']) for row in rows])
    nominal = np.column_stack([np.interp(path, ref_path, ref_state[:, i]) for i in range(9)])
    actual = np.array([[float(row[k]) for k in STATE_KEYS] for row in rows])
    times = np.array([float(row['wall_elapsed_s']) for row in rows])
    windows = [(interval['start_s'], interval['end_s']) for r in report['obstacles']
               if r['simultaneous_collision_pass'] for interval in r['overlap_intervals']]
    conflict = np.zeros(len(rows), bool)
    for start, end in windows:
        conflict |= (times >= start) & (times <= end)
    conflict &= np.array([row['phase'] == 'tracking' and row['motion_active'] == '1' for row in rows])
    joints = np.linalg.norm(actual[:, 3:] - nominal[:, 3:], axis=1)
    base = np.linalg.norm(actual[:, :2] - nominal[:, :2], axis=1)
    maximum_arm = float(joints[conflict].max()) if np.any(conflict) else 0.
    maximum_base = float(base[conflict].max()) if np.any(conflict) else 0.
    report.update(same_controller_sources=same_sources,
                  same_camera_trajectory=digest(run / 'trajectory.json') == digest(baseline / 'trajectory.json'),
                  actual_shot_motion_during_conflict_s=float(sum(float(row['dt_s']) for row, hit in zip(rows, conflict) if hit)),
                  maximum_arm_joint_posture_change_during_conflict_rad=maximum_arm,
                  maximum_base_deviation_during_conflict_m=maximum_base,
                  arm_posture_changed_during_conflict=maximum_arm >= .05,
                  response_comparison='Arm and base differences at equal camera-path progress; collision proof uses equal physical time.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--body', choices=['arm', 'base'], required=True)
    parser.add_argument('--min-encounters', type=int, default=1,
                        help='Required separate solid-collision intervals for one obstacle (each >= 0.1 s)')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.min_encounters < 1:
        parser.error('--min-encounters must be positive')
    report = evaluate(args.baseline, args.scene, args.body, args.min_encounters)
    if args.run:
        compare_response(report, args.run, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['baseline_completed'] and report['simultaneous_collision_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
