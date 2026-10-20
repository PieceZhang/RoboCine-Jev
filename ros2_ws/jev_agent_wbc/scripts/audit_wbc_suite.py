#!/usr/bin/env python3
"""Independently check motion coverage and every stop in a physical WBC suite.

Use alongside audit_wbc_physical.py, which validates immutable tasks, feedback,
source snapshots and timing. Neither audit certifies perception or avoidance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_wbc_physical import digest, finite, pick, rigid, strict_stop


DIRECTIONS = {
    'husky': ('forward', 'backward', 'left', 'right', 'diagonal_ne', 'diagonal_sw'),
    'tiago': ('forward', 'backward', 'left', 'right', 'diagonal_ne', 'diagonal_sw'),
    'ridgeback': ('forward', 'backward', 'left', 'right', 'diagonal_ne', 'diagonal_sw', 'yaw_ccw', 'yaw_cw'),
    'g1': ('forward', 'backward', 'left', 'right', 'diagonal_left', 'diagonal_right', 'yaw_left', 'yaw_right', 'camera_shot'),
    'go2': ('forward', 'backward', 'left', 'right', 'diagonal_left', 'diagonal_right', 'yaw_left', 'yaw_right', 'camera_shot'),
}


def canonical(row):
    return dict(stamp=float(finite(pick(row, 'stamp_s', 't_s'), ())),
        base=rigid(row['T_world_base']), camera=rigid(pick(row, 'T_world_camera_optical', 'T_world_camera')),
        q=finite(pick(row, 'joint_positions', 'q')),
        linear=finite(pick(row, 'base_linear_velocity_world', 'base_linear_velocity'), (3,)),
        angular=finite(pick(row, 'base_angular_velocity_world', 'base_angular_velocity'), (3,)))


def motion(rows, name, legged):
    """Recompute direction sign/displacement or the original legged speed gate."""
    if len(rows) < 2:
        raise ValueError('missing measured movement '+name)
    if legged:
        tail = rows[round(len(rows)/2):]
        desired = finite(tail[0]['requested_body_velocity'], (3,))
        if any(not np.array_equal(finite(r['requested_body_velocity'], (3,)), desired) for r in tail):
            raise ValueError('requested velocity changed within case '+name)
        measured = finite([r['measured_body_velocity'] for r in tail])
        mean = np.mean(measured, axis=0)
        return dict(requested_body_velocity=desired.tolist(), measured_body_velocity=mean.tolist(),
            passed=bool(np.all(np.abs(mean-desired) <= np.maximum(.025, .25*np.abs(desired)))))
    matrices = np.array([rigid(r['T_world_base']) for r in rows])
    delta = matrices[-1, :3, 3]-matrices[0, :3, 3]
    if name.startswith('yaw_'):
        yaw = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
        signed = float((yaw[-1]-yaw[0])*(1 if name == 'yaw_ccw' else -1))
        return dict(signed_yaw_rad=signed, passed=signed > .7)
    angle, sign = {'forward': (0., 1.), 'backward': (0., -1.), 'left': (np.pi/2, 1.),
        'right': (-np.pi/2, 1.), 'diagonal_ne': (np.pi/4, 1.), 'diagonal_sw': (-3*np.pi/4, 1.)}[name]
    along = float(delta[:2] @ [np.cos(angle), np.sin(angle)]*sign)
    lateral = float(abs(delta[:2] @ [-np.sin(angle), np.cos(angle)]))
    return dict(along_m=along, lateral_error_m=lateral, passed=bool(along > .25 and lateral < .08))


def rotation_span(matrices):
    quaternions = Rotation.from_matrix(np.asarray(matrices)[:, :3, :3]).as_quat()
    minimum_abs_dot = float(np.min(np.abs(quaternions @ quaternions.T)))
    return float(2*np.arccos(np.clip(minimum_abs_dot, 0., 1.)))


def excitation(rows):
    """A nontrivial shot must visibly respond in both translation and rotation."""
    moving = [r for r in rows if r.get('phase') == 'move']
    desired = np.array([rigid(r['T_world_reference_optical']) for r in moving])
    actual = np.array([rigid(r['T_world_camera_optical']) for r in moving])
    def errors(poses):
        position = np.linalg.norm(poses[:, :3, 3]-desired[:, :3, 3], axis=1)
        rotation = Rotation.from_matrix(poses[:, :3, :3] @ desired[:, :3, :3].transpose(0, 2, 1)).magnitude()
        return np.sqrt(np.mean(np.column_stack([position, rotation])**2, axis=0))
    rms = errors(actual)
    baseline = errors(np.repeat(actual[:1], len(actual), axis=0))
    reference_span = [float(np.linalg.norm(np.ptp(desired[:, :3, 3], axis=0))), rotation_span(desired)]
    measured_span = [float(np.linalg.norm(np.ptp(actual[:, :3, 3], axis=0))), rotation_span(actual)]
    ratio = np.divide(rms, baseline, out=np.full(2, np.inf), where=baseline > 0)
    return dict(reference_span_m_rad=reference_span, measured_span_m_rad=measured_span,
        rmse_m_rad=rms.tolist(), initial_pose_hold_baseline_rmse_m_rad=baseline.tolist(),
        error_to_hold_baseline_ratio=ratio.tolist(),
        passed=bool(reference_span[0] >= .1-1e-8 and reference_span[1] >= .3-1e-8
            and np.all(ratio <= .65) and np.all(rms < [.05, .15])))


def independent_arm(rows, configuration, robot):
    """Check every controlled arm/torso axis, actual base drift and TIAGo home."""
    selected = [i for i, r in enumerate(rows) if r.get('case', r.get('phase')) == 'independent_arm']
    if not selected or selected[0] == 0:
        raise ValueError('independent arm initial measurement missing')
    initial = canonical(rows[selected[0]-1])
    group = [rows[i] for i in selected]
    names = group[0]['joint_names']
    if robot == 'tiago':
        joints = configuration['spec']['joints']
        arm_names = [j['name'] for j in joints]
        home = finite([j['home'] for j in joints])
        lower, upper = (finite([j[k] for j in joints]) for k in ('lower', 'upper'))
        amplitude = np.minimum(.12, .4*np.minimum(home-lower, upper-home))
        end_rows = [r for r in rows if r.get('phase') == 'independent_arm_measured_stop']
        if not end_rows:
            raise ValueError('independent arm home/stop measurement missing')
        end = canonical(end_rows[-1])
    else:
        arm_names = configuration['robot']['arm_joints']
        lower, upper = finite(configuration['arm_model_limits']).T
        arm_ids = [names.index(n) for n in arm_names]
        start = initial['q'][arm_ids]
        amplitude = np.minimum(.12, .4*np.minimum(start-lower, upper-start))
        end = canonical(group[-1])
    if len(arm_names) != (8 if robot == 'tiago' else 6):
        raise ValueError('missing independent arm/torso axes')
    arm_ids = [names.index(n) for n in arm_names]
    positions = finite([pick(r, 'joint_positions', 'q') for r in group])[:, arm_ids]
    span = np.ptp(positions, axis=0)
    drift = float(np.linalg.norm(end['base'][:2, 3]-initial['base'][:2, 3]))
    result = dict(joint_names=arm_names, measured_span=span.tolist(),
        original_amplitude=amplitude.tolist(), base_drift_m=drift,
        passed=bool(np.all(amplitude > 0) and np.all(span > .5*amplitude) and drift < .03))
    if robot == 'tiago':
        units = group[0]['joint_units']
        tolerances = np.array([.002 if units[i] == 'm' else .01 for i in arm_ids])
        home_error = np.abs(end['q'][arm_ids]-home)
        result.update(measured_home_error=home_error.tolist(), home_tolerance=tolerances.tolist())
        result['passed'] = bool(result['passed'] and np.all(home_error < tolerances))
    return result


def audit_suite(folder, robot, *, camera_only=False):
    folder = Path(folder).resolve()
    errors, checks = [], {}
    source_files = {'suite_auditor': digest(Path(__file__)),
        'physical_auditor': digest(Path(__file__).with_name('audit_wbc_physical.py'))}
    try:
        name = 'summary.json' if (folder/'summary.json').is_file() else 'acceptance.json'
        summary = json.loads((folder/name).read_text())
        source_files[name] = digest(folder/name)
        log = next(folder/n for n in ('measurements.jsonl', 'samples.jsonl') if (folder/n).is_file())
        source_files[log.name] = digest(log)
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        legged = robot in ('g1', 'go2')
        if summary.get('diagnostic_only') is True and not camera_only:
            errors.append('diagnostic_is_not_full_suite')
        if summary.get('all_pass', summary.get('passed')) is not True:
            errors.append('native_acceptance_failed')
        units = rows[0].get('joint_units')
        if units is None and legged:
            units = ['rad']*len(rows[0]['joint_positions'])
        if not units:
            raise ValueError('missing all measured joint units')
        phases = summary.get('cases', summary.get('phases', []))
        native = {p.get('case', p.get('phase')): p for p in phases}
        required = ('camera_6d',) if camera_only else DIRECTIONS[robot]
        for case in required:
            group = [r for r in rows if r.get('case', r.get('phase')) == case]
            moving = [r for r in group if r.get('phase') != 'stop'] if legged else group
            if case not in native or native[case].get('passed') is not True:
                errors.append('missing_or_failed_native_phase:'+case)
            direction = excitation(group) if camera_only else motion(moving, case, legged)
            if legged:
                stopping = [r for r in group if r.get('phase') == 'stop']
            else:
                label = case+('_measured_stop' if robot == 'tiago' else '_stop')
                stopping = [r for r in rows if r.get('case', r.get('phase')) == label]
                if label not in native or native[label].get('passed') is not True:
                    errors.append('missing_or_failed_native_stop:'+case)
            stop = strict_stop([canonical(r) for r in stopping], units)
            checks[case] = dict(movement=direction, strict_stop=stop,
                passed=bool(direction['passed'] and stop['passed']))
        if not legged and not camera_only:
            group = [r for r in rows if r.get('case', r.get('phase')) == 'independent_arm']
            if len(group) < 2 or native.get('independent_arm', {}).get('passed') is not True:
                raise ValueError('independent arm phase missing or failed')
            stop_label = 'independent_arm_measured_stop' if robot == 'tiago' else 'arm_stop'
            stopping = [r for r in rows if r.get('case', r.get('phase')) == stop_label]
            configuration_path = folder/'configuration.json'
            source_files[configuration_path.name] = digest(configuration_path)
            arm = independent_arm(rows, json.loads(configuration_path.read_text()), robot)
            checks['independent_arm'] = dict(movement=arm,
                strict_stop=strict_stop([canonical(r) for r in stopping], units))
            checks['independent_arm']['passed'] = bool(arm['passed']
                and checks['independent_arm']['strict_stop']['passed'])
        if not checks or not all(c['passed'] for c in checks.values()):
            errors.append('one_or_more_measured_phases_failed')
    except (ValueError, KeyError, TypeError, StopIteration, IndexError, OSError) as exc:
        errors.append(str(exc))
    return dict(schema='independent_wbc_suite_audit/v1', folder=str(folder), robot=robot,
        camera_excitation_only=camera_only, audit_errors=errors, checks=checks,
        passed=bool(not errors and checks), sources_sha256=source_files,
        scope='measured direction coverage and each complete strict stop; use physical audit for task, feedback, safety and pace; sensing and avoidance unknown')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--robot', required=True, choices=DIRECTIONS)
    parser.add_argument('--camera-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_suite(args.folder, args.robot, camera_only=args.camera_only)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload, end='')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
