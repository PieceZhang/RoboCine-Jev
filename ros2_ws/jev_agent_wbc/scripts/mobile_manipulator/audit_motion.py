#!/usr/bin/env python3
"""Independently audit saved physical samples; no simulator or control execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rotation_span_rad(rotations):
    """Largest geodesic angle between any two recorded orientations."""
    rotations = np.asarray(rotations)
    cosine = (np.einsum('aij,bij->ab', rotations, rotations)-1.)/2.
    return float(np.arccos(np.clip(cosine, -1., 1.)).max())


def audit(folder):
    folder = Path(folder)
    acceptance = json.loads((folder/'acceptance.json').read_text())
    config = json.loads((folder/'configuration.json').read_text())
    provenance = json.loads((folder/'provenance.json').read_text())
    spec, names = config['spec'], config['dof_names']
    joints = [names.index(j['name']) for j in spec['joints']]
    lower = np.array([j['lower'] for j in spec['joints']])
    upper = np.array([j['upper'] for j in spec['joints']])
    source_checks = {}
    for number, (source, expected) in enumerate(provenance['sources'].items()):
        frozen = folder/'sources'/f'{number}_{Path(source).name}'
        source_checks[source] = frozen.is_file() and hashlib.sha256(frozen.read_bytes()).hexdigest() == expected
    count, previous, phases = 0, 0., set()
    finite, continuous, limits, transforms = True, True, True, True
    positions, orientations, last = [], [], []
    with (folder/'samples.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            stamp, q = row['t_s'], np.asarray(row['q'])
            phases.add(row['phase'])
            continuous &= abs(stamp-previous-.01) < 1e-7
            finite &= bool(np.isfinite(q).all() and np.isfinite(row['qd']).all())
            limits &= bool(np.all(q[joints] >= lower-.015) and np.all(q[joints] <= upper+.015))
            for key in ['T_world_base', 'T_world_camera']:
                matrix = np.asarray(row[key])
                transforms &= bool(matrix.shape == (4, 4) and np.isfinite(matrix).all()
                    and np.allclose(matrix[3], [0, 0, 0, 1])
                    and np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-5)
                    and np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-5))
            if row['phase'] == 'whole_body_camera':
                actual, reference = np.asarray(row['T_world_camera']), np.asarray(row['reference'])
                positions.append(np.linalg.norm(actual[:3, 3]-reference[:3, 3]))
                angle = np.arccos(np.clip((np.trace(reference[:3, :3]@actual[:3, :3].T)-1)/2, -1, 1))
                orientations.append(angle)
            if row['phase'] == 'final_stop':
                last.append(row)
            previous, count = stamp, count+1
    stop = False
    base_rotation_span = camera_rotation_span = None
    if len(last) >= 51:
        tail = last[-51:]
        base = np.array([np.asarray(r['T_world_base'])[:3, 3] for r in tail])
        arm = np.array([np.asarray(r['q'])[joints] for r in tail])
        camera = np.array([np.asarray(r['T_world_camera'])[:3, 3] for r in tail])
        base_rotation_span = rotation_span_rad([np.asarray(r['T_world_base'])[:3, :3] for r in tail])
        camera_rotation_span = rotation_span_rad([np.asarray(r['T_world_camera'])[:3, :3] for r in tail])
        tolerance = np.array([.002 if config.get('controlled_joint_units', {}).get(j['name']) == 'm' else .01 for j in spec['joints']])
        stop = bool(np.linalg.norm(np.ptp(base, axis=0)) < .005 and np.all(np.ptp(arm, axis=0) < tolerance)
                    and np.linalg.norm(np.ptp(camera, axis=0)) < .005
                    and base_rotation_span <= .01 and camera_rotation_span <= .01)
    position_rmse = float(np.sqrt(np.mean(np.square(positions)))) if positions else None
    orientation_rmse = float(np.sqrt(np.mean(np.square(orientations)))) if orientations else None
    wbc_pass = bool(position_rmse is not None and orientation_rmse is not None
                    and position_rmse < .03 and orientation_rmse < .08)
    reported = next((row for row in acceptance.get('phases', []) if row.get('phase') == 'whole_body_camera'), {})
    reported_consistent = bool(position_rmse is not None and orientation_rmse is not None
                               and np.isclose(position_rmse, reported.get('position_rmse_m', np.nan), atol=1e-7, rtol=1e-5)
                               and np.isclose(orientation_rmse, reported.get('orientation_rmse_rad', np.nan), atol=1e-7, rtol=1e-5))
    required = {'forward', 'backward', 'left', 'right', 'diagonal_ne', 'diagonal_sw', 'independent_arm', 'whole_body_camera', 'final_stop'}
    motion = bool(acceptance.get('complete') and acceptance.get('passed') and not acceptance.get('diagnostic_only')
                  and all(source_checks.values()) and source_checks and finite and continuous and limits and transforms
                  and required <= phases and count*.01 == acceptance['simulated_seconds'] and stop
                  and wbc_pass and reported_consistent)
    result = dict(schema='mobile_manipulator_motion_audit/v1', complete=True,
        motion_acceptance_verified=motion, sample_count=count, continuous_100hz_timestamps=bool(continuous),
        finite_measured_state=bool(finite), joint_limits_pass=bool(limits), rigid_transforms_pass=bool(transforms),
        frozen_source_hashes=source_checks, required_motion_phases_present=required <= phases, measured_stop_pass=stop,
        final_half_second_base_rotation_span_rad=base_rotation_span,
        final_half_second_camera_rotation_span_rad=camera_rotation_span,
        stop_rotation_tolerance_rad=.01, recomputed_wbc_error_thresholds_pass=wbc_pass,
        reported_wbc_errors_consistent=reported_consistent,
        recomputed_position_rmse_m=position_rmse, recomputed_orientation_rmse_rad=orientation_rmse,
        sensor_full_range_verified=False, avoidance_verified=False,
        scope='Saved physical motion only; use the independent posture/raw-scan audit for sensor evidence.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = audit(a.folder)
    if a.output:
        if a.output.exists():
            raise FileExistsError(a.output)
        a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['motion_acceptance_verified'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
