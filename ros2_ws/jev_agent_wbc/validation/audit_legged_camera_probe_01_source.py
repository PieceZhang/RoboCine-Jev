#!/usr/bin/env python3
"""Independently recompute G1/Go2 short-probe errors from saved physical poses.

An internally consistent failed trial is still an auditable trial. Tracking
acceptance is a separate field and never implies general camera/avoidance support.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def audit(folder):
    folder = Path(folder)
    summary_file = folder/'summary.json'
    summary = json.loads(summary_file.read_text())
    g1 = (folder/'measured.jsonl').is_file()
    raw_file = folder/('measured.jsonl' if g1 else 'measurements.jsonl')
    rows = [json.loads(line) for line in raw_file.read_text().splitlines() if line.strip()]
    selected = ([r for r in rows if r.get('phase') == 'pink_camera'] if g1 else
                [r for r in rows if 'T_world_camera_optical_reference' in r and r['t'] >= 3.])
    errors = []
    position, angle = [], []
    stamps = [r['stamp_s' if g1 else 't'] for r in rows]
    if len(stamps) < 2 or not np.allclose(np.diff(stamps), .02, rtol=0, atol=1e-9):
        errors.append('missing_or_nonmonotonic_50hz_measurements')
    if not selected:
        errors.append('no_camera_task_samples')
    for row in selected:
        measured = np.asarray(row['T_world_camera_optical' if g1 else 'T_world_camera_optical_measured'])
        reference = np.asarray(row['T_world_reference_optical' if g1 else 'T_world_camera_optical_reference'])
        for matrix in (measured, reference):
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                    or not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                    or abs(np.linalg.det(matrix[:3, :3])-1) > 1e-5):
                errors.append('invalid_physical_or_reference_transform')
        p = float(np.linalg.norm(measured[:3, 3]-reference[:3, 3]))
        a = float(Rotation.from_matrix(reference[:3, :3]@measured[:3, :3].T).magnitude())
        position.append(p); angle.append(a)
        if (abs(p-row['position_error_m' if g1 else 'camera_position_error_m']) > 1e-9
                or abs(a-row['orientation_error_rad' if g1 else 'camera_orientation_error_rad']) > 1e-9):
            errors.append('stored_error_does_not_match_physical_pose')
    p_rmse = float(np.sqrt(np.mean(np.square(position)))) if position else None
    a_rmse = float(np.sqrt(np.mean(np.square(angle)))) if angle else None
    declared = summary if g1 else summary['cases'][0]
    for value, key in ((p_rmse, 'position_rmse_m' if g1 else 'camera_position_rmse_m'),
                       (a_rmse, 'orientation_rmse_rad' if g1 else 'camera_orientation_rmse_rad')):
        if value is None or declared.get(key) is None or abs(value-declared[key]) > 1e-9:
            errors.append('summary_metric_mismatch:'+key)
    decoded = []
    for name in ('camera_rgb.png', 'camera_direction_probe.png'):
        path = folder/name
        if path.is_file():
            with Image.open(path) as picture:
                picture.load()
                decoded.append(dict(file=name, size=list(picture.size), mode=picture.mode,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(schema='legged_camera_probe_independent_audit/v1',
                source_directory=str(folder.resolve()), audit_passed=not errors,
                errors=sorted(set(errors)), physical_samples=len(rows), camera_samples=len(selected),
                sample_scope='G1 all active camera samples; Go2 existing declared t>=3s evaluation window',
                position_rmse_m=p_rmse, orientation_rmse_rad=a_rmse,
                tracking_thresholds=dict(position_rmse_m=.05, orientation_rmse_rad=.15),
                local_tracking_pass=bool(not errors and p_rmse < .05 and a_rmse < .15),
                general_continuous_tracking_verified=False, avoidance_verified=False,
                target_semantics='held target actually issued for the preceding control interval',
                decoded_images=decoded,
                input_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (summary_file, raw_file, Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.result_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['audit_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
