#!/usr/bin/env python3
"""Read-only audit of immutable WBC tasks against measured physical trajectories.

Recomputes reference interpolation, optical axes, errors and strict stopping.
It deliberately does not import a controller's acceptance or stop implementation.
Passing this audit does not certify environmental perception or avoidance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from jev_agent_wbc.wbc_protocol import TimedTask
from jev_agent_wbc.representation import Observation, Pose, Box


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, shape=None):
    # JSON booleans must not be interpreted as measured numbers.
    def has_bool(v):
        return isinstance(v, bool) or (isinstance(v, (list, tuple)) and any(has_bool(x) for x in v))
    if has_bool(value):
        raise ValueError('boolean measurement')
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError('invalid finite measurement shape')
    return array


def rigid(value):
    matrix = finite(value, (4, 4))
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(rotation)-1) > 1e-5):
        raise ValueError('invalid measured SE3')
    return matrix


def pose(value):
    translation = finite(value['translation_m'], (3,))
    quaternion = finite(value['quaternion_xyzw'], (4,))
    if abs(np.linalg.norm(quaternion)-1) > 1e-6:
        raise ValueError('invalid pose quaternion')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def angle(a, b):
    return float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())


def reference(task):
    """Independent reconstruction directly from original serialized knots."""
    path = task['camera_trajectory']
    times = finite([x['t_s'] for x in path['samples']])
    transforms = np.array([pose(x['pose']) for x in path['samples']])
    orientations = Slerp(times, Rotation.from_matrix(transforms[:, :3, :3]))
    # T_semantic_optical is the inverse of optical_from_semantic in the contract.
    semantic_optical = np.eye(4)
    semantic_optical[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    start = float(path['start_s'])

    def sample(stamp):
        relative = float(np.clip(stamp-start, times[0], times[-1]))
        result = np.eye(4)
        result[:3, :3] = orientations(relative).as_matrix()
        result[:3, 3] = [np.interp(relative, times, transforms[:, axis, 3]) for axis in range(3)]
        return result @ semantic_optical
    return sample, start, start+times[-1]


def strict_stop(rows, units):
    """All measured axes and all joints over a complete final half-second."""
    times = finite([r['stamp'] for r in rows])
    if len(times) < 2 or times[-1]-times[0] < .5-1e-8 or np.any(np.diff(times) <= 0):
        raise ValueError('incomplete stop window')
    first = max(0, np.searchsorted(times, times[-1]-.5, side='right')-1)
    tail = rows[first:]
    base = np.array([r['base'] for r in tail])
    camera = np.array([r['camera'] for r in tail])
    joints = finite([r['q'] for r in tail])
    if joints.shape[1:] != (len(units),) or not units or any(u not in ('m', 'rad') for u in units):
        raise ValueError('all joint units required')
    def rotation_span(matrices):
        return max(angle(a, b) for i, a in enumerate(matrices) for b in matrices[i:])
    metrics = dict(measured_window_s=float(times[-1]-times[first]),
        base_position_span_m=float(np.linalg.norm(np.ptp(base[:, :3, 3], axis=0))),
        camera_position_span_m=float(np.linalg.norm(np.ptp(camera[:, :3, 3], axis=0))),
        base_rotation_span_rad=rotation_span(base), camera_rotation_span_rad=rotation_span(camera),
        joint_span=np.ptp(joints, axis=0).tolist(),
        final_linear_speed_mps=float(np.linalg.norm(rows[-1]['linear'])),
        final_angular_speed_radps=float(np.linalg.norm(rows[-1]['angular'])))
    metrics['passed'] = bool(metrics['base_position_span_m'] < .005
        and metrics['camera_position_span_m'] < .005 and metrics['base_rotation_span_rad'] < .01
        and metrics['camera_rotation_span_rad'] < .01 and metrics['final_linear_speed_mps'] < .02
        and metrics['final_angular_speed_radps'] < .03
        and np.all(np.asarray(metrics['joint_span']) < [.002 if u == 'm' else .01 for u in units]))
    return metrics


def pick(row, *keys):
    for key in keys:
        if key in row:
            return row[key]
    raise ValueError('missing measured field '+keys[0])


def audit(folder):
    folder = Path(folder).resolve()
    errors, cases = [], []
    files = {'auditor': digest(Path(__file__))}
    def read(name):
        path = folder/name
        files[name] = digest(path)
        return json.loads(path.read_text())
    def reject(message):
        if message not in errors:
            errors.append(message)
    try:
        summary_name = 'summary.json' if (folder/'summary.json').is_file() else 'acceptance.json'
        summary, provenance = read(summary_name), read('provenance.json')
        hashes = provenance.get('frozen_sha256', {})
        # Historical wheels encode both the snapshot location and its checksum.
        if not hashes:
            hashes = {v['snapshot']: v['sha256'] for v in provenance.get('sources', {}).values()
                      if isinstance(v, dict) and set(v) >= {'snapshot', 'sha256'}}
        if not hashes:
            reject('missing_frozen_source_hashes')
        for name, expected in hashes.items():
            path = (folder/name if Path(name).parts[:1] == ('sources',) else folder/'sources'/name).resolve()
            if not path.is_relative_to(folder/'sources') or not path.is_file() or digest(path) != expected:
                reject('frozen_source_missing_or_changed')
        tasks_name = 'timed_tasks.json' if (folder/'timed_tasks.json').is_file() else 'public_task.json'
        tasks = read(tasks_name)
        tasks = tasks if isinstance(tasks, list) else [tasks]
        parsed = {}
        for original in tasks:
            task = TimedTask.from_dict(original)
            if task.task_id in parsed:
                raise ValueError('duplicate original task id')
            parsed[task.task_id] = (task, reference(original))
        if not parsed:
            reject('no_original_timed_tasks')
        ordered_tasks = sorted(parsed.values(), key=lambda item: item[1][1])
        name = next((n for n in ('measurements.jsonl', 'samples.jsonl', 'measured.jsonl') if (folder/n).is_file()), None)
        if name is None:
            raise ValueError('missing physical measurements')
        files[name] = digest(folder/name)
        raw = [json.loads(line) for line in (folder/name).read_text().splitlines() if line.strip()]
        public = {}
        if (folder/'public_feedback.jsonl').is_file():
            files['public_feedback.jsonl'] = digest(folder/'public_feedback.jsonl')
            for line in (folder/'public_feedback.jsonl').read_text().splitlines():
                entry = json.loads(line)
                feedback = entry.get('feedback', entry)
                public[round(float(feedback['stamp_s']), 8)] = feedback
        rows, names, units = [], None, None
        observations = {}
        if (folder/'accepted_observations.json').is_file():
            for value in read('accepted_observations.json'):
                observation = Observation.from_dict(value)
                if observation.source != 'measured_fk':
                    reject('task_admission_observation_not_measured')
                observations[observation.id] = observation
        for entry in raw:
            stamp = float(finite(pick(entry, 'stamp_s', 't_s'), ()))
            q = finite(pick(entry, 'joint_positions', 'q'))
            if q.ndim != 1 or not len(q):
                raise ValueError('invalid full joint positions')
            row = dict(stamp=stamp, base=rigid(entry['T_world_base']),
                camera=rigid(pick(entry, 'T_world_camera_optical', 'T_world_camera')), q=q,
                linear=finite(pick(entry, 'base_linear_velocity_world', 'base_linear_velocity'), (3,)),
                angular=finite(pick(entry, 'base_angular_velocity_world', 'base_angular_velocity'), (3,)),
                raw=entry, feedback=entry.get('task_feedback', public.get(round(stamp, 8))))
            observation_value = entry.get('measured_observation')
            if observation_value is None and row['feedback']:
                observation_value = row['feedback'].get('observation')
            if observation_value is not None:
                observation = Observation.from_dict(observation_value)
                if observation.source != 'measured_fk' or abs(observation.stamp_s-stamp) > 1e-8:
                    reject('observation_not_bound_to_measurement_stamp')
                observations[observation.id] = observation
            row_names = entry.get('joint_names')
            if row_names is not None:
                if len(row_names) != len(q) or len(set(row_names)) != len(q):
                    reject('joint_names_do_not_cover_full_measurement')
                if names is not None and row_names != names:
                    reject('joint_order_changed')
                names = row_names
            if 'joint_units' in entry:
                if units is not None and units != entry['joint_units']:
                    reject('joint_units_changed')
                units = entry['joint_units']
            rows.append(row)
        if units is None and summary.get('robot') in ('g1', 'go2'):
            units = ['rad']*len(rows[0]['q'])
        if units is None:
            reject('missing_all_joint_units')
        if names is None:
            reject('missing_all_joint_names')
        actuator_file = 'actuator_audit.json' if (folder/'actuator_audit.json').is_file() else 'configuration.json'
        actuator = read(actuator_file)
        expected_names = actuator.get('joint_names', actuator.get('dof_names'))
        if not expected_names or names != expected_names:
            reject('joint_measurements_do_not_cover_physical_articulation')
        expected_counts = {'g1': 29, 'go2': 18}
        if summary.get('robot') in expected_counts and len(names or []) != expected_counts[summary['robot']]:
            reject('physical_dof_count_mismatch')
        stamps = np.array([r['stamp'] for r in rows])
        if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
            raise ValueError('nonmonotonic physical measurements')
        interval = float(np.median(np.diff(stamps)))
        if abs(stamps[0]-interval) > 1e-7:
            reject('physical_log_start_incomplete')
        if np.max(np.diff(stamps)) > interval*1.5+1e-8:
            reject('physical_measurement_gap')
        known_rates = {'legged_adaptive_physical/v1': 50., 'ocs2_wheeled_physical_acceptance/v1': 100.}
        measured_hz = provenance.get('measurement_hz', known_rates.get(summary.get('schema')))
        if measured_hz is None or float(finite(measured_hz, ())) <= 0:
            reject('missing_measurement_rate')
        elif not np.allclose(np.diff(stamps), 1./measured_hz, atol=1e-7, rtol=0):
            reject('physical_samples_missing_or_rate_changed')
        grouped = {key: [] for key in parsed}
        feedback_count = 0
        for row in rows:
            entry, feedback = row['raw'], row['feedback']
            # Acceptance occurs at an existing measured tick. Its first executed
            # sample is the following tick, so that boundary may still be the
            # previous task's terminal sample.
            active = [item[0].task_id for item in ordered_tasks if item[1][1] < row['stamp']-1e-8]
            if active and (feedback is None or feedback.get('task_id') != active[-1]):
                reject('active_task_feedback_missing_or_wrong_task')
            if feedback is None:
                continue
            if feedback.get('task_id') not in parsed:
                reject('feedback_not_bound_to_original_task'); continue
            task, (sample, start, end) = parsed[feedback['task_id']]
            if row['stamp'] < start-1e-8:
                reject('feedback_before_task_start'); continue
            feedback_count += 1
            if abs(float(finite(feedback['stamp_s'], ()))-row['stamp']) > 1e-8:
                reject('feedback_measurement_stamp_mismatch')
            if feedback.get('camera_path_id') != task.trajectory.id or feedback.get('camera_reference_changed') is not False:
                reject('camera_reference_binding_changed')
            desired = sample(row['stamp'])
            position_error = float(np.linalg.norm(row['camera'][:3, 3]-desired[:3, 3]))
            orientation_error = angle(row['camera'], desired)
            for source in (entry, feedback):
                if 'position_error_m' in source:
                    if (abs(float(finite(source['position_error_m'], ()))-position_error) > 1e-6
                            or abs(float(finite(source['orientation_error_rad'], ()))-orientation_error) > 1e-6):
                        reject('stored_error_differs_from_original_task_measurement')
            logged_ref = entry.get('T_world_reference_optical', entry.get('reference'))
            if logged_ref is not None and not np.allclose(rigid(logged_ref), desired, rtol=0, atol=1e-6):
                reject('logged_reference_differs_from_original_timed_task')
            actual = pose(feedback['actual_camera'])
            optical_to_semantic = np.eye(4)
            optical_to_semantic[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
            if not np.allclose(actual, row['camera'] @ optical_to_semantic, atol=1e-6, rtol=0):
                reject('feedback_camera_differs_from_raw_optical_measurement')
            geometry = feedback.get('geometry', {})
            # A complete public two-part summary retains every compressed box;
            # it can reconstruct its acquisition-time Observation for binding.
            try:
                snapshot = Observation(float(finite(geometry['stamp_s'], ())),
                    Pose.from_dict(geometry['world_from_mount']), Pose.from_dict(geometry['mount_from_camera']),
                    {part: tuple(Box.from_dict(x) for x in geometry['parts'][part]['regions'])
                        for part in ('mobile_platform', 'camera_manipulator')}, geometry['source'])
                if (snapshot.id != geometry['observation_id'] or snapshot.source != 'measured_fk'
                        or abs(snapshot.stamp_s-row['stamp']) > 1e-8
                        or not np.allclose(pose(snapshot.world_from_camera.to_dict()), actual, atol=1e-6, rtol=0)):
                    reject('feedback_geometry_not_bound_to_actual_measurement')
                observations[snapshot.id] = snapshot
            except (ValueError, KeyError, TypeError):
                reject('incomplete_measured_two_part_geometry_feedback')
            if row['stamp'] > end+1e-8 and (feedback.get('tracking_mode') != 'explicit_terminal_hold'
                    or abs(float(finite(feedback.get('reference_stamp_s', -1), ()))-end) > 1e-8):
                reject('terminal_reference_hold_not_explicit')
            constraints = feedback.get('constraints', {})
            if set(constraints) != {'configuration', 'actuation', 'self_collision', 'environment', 'policy_domain'}:
                reject('missing_constraint_feedback_families')
            for value in constraints.values():
                if value.get('status') in ('pass', 'fail') and (value.get('stamp_s') is None
                        or not 0 <= row['stamp']-float(finite(value['stamp_s'], ())) <= .1+1e-8):
                    reject('constraint_evidence_not_fresh')
            grouped[task.task_id].append(dict(row, errors=[position_error, orientation_error]))
        for key, (task, (_, start, end)) in parsed.items():
            admission = observations.get(task.observation_id)
            if (admission is None or admission.source != 'measured_fk'
                    or not -1e-8 <= task.issued_at_s-admission.stamp_s <= .1+1e-8):
                reject('task_admission_not_bound_to_fresh_measured_observation')
            segment = grouped[key]
            moving = [r for r in segment if r['stamp'] <= end+1e-8]
            stopped = [r for r in segment if r['stamp'] > end+1e-8]
            if not moving or moving[0]['stamp'] > start+interval*1.5 or moving[-1]['stamp'] < end-interval*1.5:
                reject('task_physical_interval_incomplete'); continue
            rms = np.sqrt(np.mean(np.array([r['errors'] for r in moving])**2, axis=0))
            final_error = segment[-1]['errors']
            camera_pass = bool(np.all(rms < [task.position_tolerance_m, task.orientation_tolerance_rad])
                and np.all(np.asarray(final_error) < [task.position_tolerance_m, task.orientation_tolerance_rad]))
            stop = strict_stop(stopped, units) if units is not None else dict(passed=False)
            cases.append(dict(task_id=key, measured_samples=len(segment), moving_samples=len(moving),
                position_rmse_m=float(rms[0]), orientation_rmse_rad=float(rms[1]), final_error=final_error,
                task_tolerances=[task.position_tolerance_m, task.orientation_tolerance_rad],
                camera_tracking_pass=camera_pass, strict_stop=stop, passed=bool(camera_pass and stop['passed'])))
        elapsed = float(finite(summary.get('wall_seconds', summary.get('wall_s')), ()))
        sim_seconds = float(finite(summary.get('sim_seconds', summary.get('simulated_seconds', summary.get('simulation_s'))), ()))
        timing_pass = elapsed > 0 and sim_seconds > 0 and elapsed <= sim_seconds*1.05
        if abs(stamps[-1]-sim_seconds) > interval*1.5:
            reject('summary_simulation_time_differs_from_physical_log')
        run_complete = summary.get('run_complete', summary.get('complete')) is True
        if not run_complete:
            reject('physical_run_incomplete')
        return dict(schema='independent_wbc_physical_audit/v1', folder=str(folder), audit_errors=errors,
            audit_passed=not errors, run_complete=run_complete, measured_samples=len(rows),
            feedback_samples=feedback_count, measured_interval_s=interval, timing_pass=bool(timing_pass),
            tasks=cases, all_tasks_pass=bool(cases) and all(c['passed'] for c in cases),
            physical_acceptance_pass=bool(not errors and timing_pass and cases and all(c['passed'] for c in cases)),
            sources_sha256=files, scope='immutable timed camera tasks, measured SE3, all joint stopping, freshness and pace; '
                'direction coverage, contacts, rendering, sensing and avoidance require separate evidence')
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        reject(str(exc))
        return dict(schema='independent_wbc_physical_audit/v1', folder=str(folder), audit_passed=False,
            physical_acceptance_pass=False, audit_errors=errors, tasks=cases, sources_sha256=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload, end='')
    return 0 if result['physical_acceptance_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
