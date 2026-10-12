#!/usr/bin/env python3
"""Check retiming geometry and actuator-speed witnesses without starting a scene.

The speed witnesses use recorded configurations from an earlier execution.
They do not certify acceleration, collision avoidance or the new closed loop.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jvs_mock.wbc_trajectory import CameraTrajectory
from jvs_mock.whole_body_qp import Limits, whole_body_kinematics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(package, original, name, recorded_run=None):
    path = package/'trajectories'/f'{name}.json'
    action_path = path.with_name(f'{name}_actions.json')
    before_path = original/path.name
    before = json.loads(before_path.read_text())
    after = json.loads(path.read_text())
    old_actions = json.loads((original/action_path.name).read_text())
    actions = json.loads(action_path.read_text())
    p, r, _ = whole_body_kinematics(np.zeros(9), .002)
    make = lambda source: CameraTrajectory(source, p, r, np.eye(3))
    old, new = make(before_path), make(path)
    report = dict(trajectory=path.name, original_sha256=digest(before_path),
        trajectory_sha256=digest(path), actions_sha256=digest(action_path),
        original_duration_s=before['duration_s'], duration_s=after['duration_s'],
        original_pose_samples_retained=[s['pose'] for s in before['samples']] == [s['pose'] for s in after['samples']],
        action_pose_samples_retained=[[s['pose'] for s in shot['samples']] for shot in old_actions['shots']] ==
                                    [[s['pose'] for s in shot['samples']] for shot in actions['shots']],
        initial_base_pose_identical=before['initial_base_pose'] == after['initial_base_pose'],
        retiming=after['retiming'], shots=[])
    for a, b in zip(old.shots, new.shots):
        fraction = np.linspace(0., 1., 201)
        old_samples = [old.sample(a['motion_start_s']+f*(a['motion_end_s']-a['motion_start_s'])) for f in fraction]
        new_samples = [new.sample(b['motion_start_s']+f*(b['motion_end_s']-b['motion_start_s'])) for f in fraction]
        poses = [np.array([s[0] for s in samples]) for samples in (old_samples, new_samples)]
        def motion(values):
            vector = values-new.subject
            if a['type'] == 'moveout':
                return float(np.linalg.norm(vector[-1, :2])-np.linalg.norm(vector[0, :2]))
            return float(np.arctan2(np.cross(vector[0, :2], vector[-1, :2]), vector[0, :2]@vector[-1, :2]))
        report['shots'].append(dict(type=a['type'], original_motion_s=a['motion_end_s']-a['motion_start_s'],
            motion_s=b['motion_end_s']-b['motion_start_s'],
            original_path_length_m=float(np.linalg.norm(np.diff(poses[0], axis=0), axis=1).sum()),
            path_length_m=float(np.linalg.norm(np.diff(poses[1], axis=0), axis=1).sum()),
            original_signed_motion=motion(poses[0]), signed_motion=motion(poses[1]),
            max_geometry_difference_m=float(np.linalg.norm(poses[1]-poses[0], axis=1).max()),
            max_rotation_difference_rad=max(float(np.linalg.norm(Rotation.from_matrix(y[1]@x[1].T).as_rotvec()))
                                           for x, y in zip(old_samples, new_samples))))
    times = np.linspace(0., new.duration, 5001)
    twists = np.array([new.sample(t)[2] for t in times])
    accelerations = np.diff(twists, axis=0)/np.diff(times)[:, None]
    report.update(peak_camera_linear_m_s=float(np.linalg.norm(twists[:, :3], axis=1).max()),
        peak_camera_angular_rad_s=float(np.linalg.norm(twists[:, 3:], axis=1).max()),
        peak_camera_linear_acceleration_m_s2=float(np.linalg.norm(accelerations[:, :3], axis=1).max()),
        peak_camera_angular_acceleration_rad_s2=float(np.linalg.norm(accelerations[:, 3:], axis=1).max()))
    report['geometry_pass'] = (report['original_pose_samples_retained'] and report['action_pose_samples_retained']
        and report['initial_base_pose_identical'] and all(s['max_geometry_difference_m'] < 1e-7
                and s['max_rotation_difference_rad'] < 1e-7 for s in report['shots']))
    if recorded_run is not None:
        reference = json.loads((recorded_run/'reference.json').read_text())
        prior = CameraTrajectory(recorded_run/'trajectory.json',
            np.array(reference['initial_camera_position']), np.array(reference['initial_camera_rotation']),
            np.array(reference['initial_base_rotation']), transition_options=reference.get('transition_options'))
        current = CameraTrajectory(path, np.array(reference['initial_camera_position']),
            np.array(reference['initial_camera_rotation']), np.array(reference['initial_base_rotation']),
            transition_options=reference.get('transition_options'))
        rows = [row for row in csv.DictReader((recorded_run/'telemetry.csv').open()) if row['phase'] == 'tracking']
        limit = Limits()
        actuator = np.vstack((np.eye(8), limit.wheels))
        capacities = np.r_[limit.velocity, [limit.wheel_speed]*2]
        inequalities = np.vstack((np.c_[actuator, -capacities], np.c_[-actuator, -capacities]))
        witnesses = []
        for a, b in zip(prior.shots, current.shots):
            for fraction in np.linspace(0., 1., 101):
                t = a['motion_start_s']+fraction*(a['motion_end_s']-a['motion_start_s'])
                row = min(rows, key=lambda row: abs(float(row['time_s'])-t))
                state = np.array([float(row[f'state_{key}']) for key in ('x', 'y', 'yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
                base_rotation = Rotation.from_rotvec([float(row[f'base_rotvec_{key}']) for key in 'xyz']).as_matrix()
                _, _, jacobian = whole_body_kinematics(state, float(row['base_z']), base_rotation)
                new_t = b['motion_start_s']+fraction*(b['motion_end_s']-b['motion_start_s'])
                twist = current.sample(new_t)[2]
                result = linprog(np.r_[np.zeros(8), 1.], A_ub=inequalities, b_ub=np.zeros(20),
                    A_eq=np.c_[jacobian, np.zeros(6)], b_eq=twist,
                    bounds=[(None, None)]*8+[(0., None)], method='highs')
                witnesses.append(dict(type=b['type'], time_s=float(new_t),
                    solved=bool(result.success), maximum_limit_fraction=float(result.fun) if result.success else None))
        report['instantaneous_speed_witness'] = dict(recorded_run=str(recorded_run.resolve()),
            scope='Camera feedforward only at recorded prior configurations; no acceleration/collision/closed-loop certification.',
            samples=len(witnesses), all_solved=all(x['solved'] for x in witnesses),
            maximum_limit_fraction=max(x['maximum_limit_fraction'] for x in witnesses if x['solved']),
            hardware_speed_limits=limit.velocity.tolist(), wheel_speed_limit_rad_s=limit.wheel_speed,
            within_hardware_speed_limits=all(x['solved'] and x['maximum_limit_fraction'] <= 1.+1e-8 for x in witnesses))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--recorded-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.package, args.original, args.name, args.recorded_run)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['geometry_pass']:
        raise SystemExit('Retiming changed camera geometry.')


if __name__ == '__main__':
    main()
