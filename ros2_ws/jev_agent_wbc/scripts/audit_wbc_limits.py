#!/usr/bin/env python3
"""Recompute actual joint position/speed bounds from raw physical WBC logs.

Run together with the physical and suite audits. A command inside its bound, or
a controller's reported pass, cannot excuse an out-of-bound measured velocity.
This audit does not interpret projected joint reaction as motor torque.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_wbc_physical import digest, finite, pick


def read_limits(folder, robot):
    path = folder/('configuration.json' if robot == 'tiago' else 'actuator_audit.json')
    data = json.loads(path.read_text())
    if robot == 'tiago':
        guard = data['raw_full_dof_guard']
        names, units = guard['dof_names'], guard['joint_units']
        # None is explicit continuous rotation, not missing evidence.
        if any((lo is None) != (hi is None) for lo, hi in zip(guard['lower'], guard['upper'])):
            raise ValueError('half-specified continuous joint bound')
        bounds = [[-np.inf if lo is None else lo, np.inf if hi is None else hi]
            for lo, hi in zip(guard['lower'], guard['upper'])]
        speed = guard['speed']
        position_tolerance = finite(guard['position_numerical_tolerance'], ()).item()
        absolute_speed_tolerance = finite(guard['velocity_numerical_tolerance'], ()).item()
        relative_speed_tolerance = 0.
    else:
        names = data['joint_names']
        units = data.get('joint_units', ['rad']*len(names))
        bounds = data['position_limits' if robot in ('g1', 'go2') else 'physical_limits']
        speed = data['velocity_limits' if robot in ('g1', 'go2') else 'velocity_caps']
        position_tolerance = 1e-6 if robot in ('g1', 'go2') else 0.
        absolute_speed_tolerance, relative_speed_tolerance = 0., 1e-5
    if (not names or len(set(names)) != len(names) or len(units) != len(names)
            or any(not isinstance(n, str) or not n for n in names)
            or any(u not in ('m', 'rad') for u in units)):
        raise ValueError('invalid full joint name/unit coverage')
    bounds = np.asarray(bounds, float)
    speed = finite(speed, (len(names),))
    if (bounds.shape != (len(names), 2) or np.any(np.isnan(bounds))
            or np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(speed <= 0)
            or position_tolerance < 0 or absolute_speed_tolerance < 0):
        raise ValueError('invalid original position/speed bounds')
    return dict(path=path, names=names, units=units, bounds=bounds, speed=speed,
        position_tolerance=position_tolerance, absolute_speed_tolerance=absolute_speed_tolerance,
        relative_speed_tolerance=relative_speed_tolerance)


def audit_limits(folder, robot):
    folder = Path(folder).resolve()
    errors, violations, hashes = [], [], {'limits_auditor': digest(Path(__file__)),
        'physical_auditor': digest(Path(__file__).with_name('audit_wbc_physical.py'))}
    maximum_ratio = 0.
    minimum_margin = None
    rows_checked = 0
    try:
        limits = read_limits(folder, robot)
        hashes[limits['path'].name] = digest(limits['path'])
        log = next(folder/n for n in ('measurements.jsonl', 'samples.jsonl') if (folder/n).is_file())
        hashes[log.name] = digest(log)
        previous_stamp = None
        for line in log.open():
            if not line.strip():
                continue
            row = json.loads(line)
            stamp = float(finite(pick(row, 'stamp_s', 't_s'), ()))
            if previous_stamp is not None and stamp <= previous_stamp:
                raise ValueError('non-increasing measured joint clock')
            previous_stamp = stamp
            if row['joint_names'] != limits['names']:
                raise ValueError('measured joint name/order mismatch')
            if row.get('joint_units', limits['units']) != limits['units']:
                raise ValueError('measured joint unit mismatch')
            count = len(limits['names'])
            q = finite(pick(row, 'joint_positions', 'q'), (count,))
            qd = finite(pick(row, 'measured_joint_velocity', 'qd'), (count,))
            margin = np.minimum(q-limits['bounds'][:, 0], limits['bounds'][:, 1]-q)
            speed = np.abs(qd)
            if robot in ('g1', 'go2'):
                peak = finite(row['measured_joint_velocity_peak_abs'], (count,))
                if np.any(peak < 0) or np.any(peak+1e-8 < speed):
                    raise ValueError('invalid substep speed peak')
                speed = peak
                substep_margin = float(finite(row['minimum_substep_joint_margin_rad'], ()))
                if substep_margin > float(np.min(margin))+1e-7:
                    raise ValueError('invalid substep position margin')
                if substep_margin < -limits['position_tolerance']:
                    violations.append(dict(stamp_s=stamp, kind='substep_position', margin=substep_margin))
            current_margin = float(np.min(margin))
            if np.isfinite(current_margin):
                minimum_margin = current_margin if minimum_margin is None else min(minimum_margin, current_margin)
            maximum_ratio = max(maximum_ratio, float(np.max(speed/limits['speed'])))
            bad_position = np.flatnonzero(margin < -limits['position_tolerance'])
            bad_speed = np.flatnonzero(speed > limits['speed']*(1+limits['relative_speed_tolerance'])
                +limits['absolute_speed_tolerance'])
            for index in bad_position:
                violations.append(dict(stamp_s=stamp, kind='position', joint=limits['names'][index],
                    actual=float(q[index]), lower=float(limits['bounds'][index, 0]),
                    upper=float(limits['bounds'][index, 1])))
            for index in bad_speed:
                violations.append(dict(stamp_s=stamp, kind='speed', joint=limits['names'][index],
                    actual_abs=float(speed[index]), cap=float(limits['speed'][index])))
            rows_checked += 1
        if not rows_checked:
            errors.append('no_measured_joint_samples')
        if violations:
            errors.append('measured_joint_limit_violation')
    except (ValueError, KeyError, TypeError, StopIteration, IndexError, OSError) as exc:
        errors.append(str(exc))
    return dict(schema='independent_wbc_joint_limits_audit/v1', folder=str(folder), robot=robot,
        rows_checked=rows_checked, maximum_actual_speed_ratio=maximum_ratio,
        minimum_sampled_joint_margin=minimum_margin, violation_count=len(violations),
        violations=violations, audit_errors=errors, passed=bool(rows_checked and not errors),
        sources_sha256=hashes,
        scope='actual all-joint positions and velocities, including logged legged physics substeps; use physical audit for source provenance, task and feedback; motor torque and environment unverified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--robot', required=True, choices=('husky', 'ridgeback', 'tiago', 'g1', 'go2'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_limits(args.folder, args.robot)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload, end='')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
