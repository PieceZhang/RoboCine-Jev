"""Fixed original-time evaluation. Missing observations never become zero error."""
import json
from pathlib import Path

import numpy as np

from .contracts import digest, native


EVALUATION_PROTOCOL = dict(version='cinematic_eval/v1.1', correspondence='original_elapsed_time',
    position_completion_m=.05, orientation_completion_rad=.10,
    collision_margin_m=.025, constraint_numeric_tolerance=1e-6,
    smoothness='raw uniform-time poses; translation jerk and world angular acceleration RMS',
    mask_invalid='null; separately count missing, empty, segmentation failure, occlusion and frame edge',
    semantic_intent_assessment='not_equivalent_to_geometry',
    phase_progress='Pan: nearest SO3 orientation; other motions: nearest position; diagnostic only, no retiming')


def mask_statistics(mask):
    if mask is None:
        return None
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any():
        return None
    y, x = np.nonzero(mask)
    h, w = mask.shape
    return dict(centroid=[float(np.mean(x+.5)/w), float(np.mean(y+.5)/h)],
                area_scale=float(mask.mean()), frame_edge=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))


def visual_errors(reference_mask, execution_mask):
    reference, actual = mask_statistics(reference_mask), mask_statistics(execution_mask)
    if reference is None or actual is None:
        return dict(valid=False, composition_error=None, subject_scale_error=None,
                    reference_missing=reference is None, execution_missing=actual is None)
    return dict(valid=True, composition_error=float(np.linalg.norm(np.asarray(actual['centroid'])-reference['centroid'])),
                subject_scale_error=abs(actual['area_scale']-reference['area_scale'])/reference['area_scale'],
                reference=reference, execution=actual)


def observation_angle(reference_camera, actual_camera, reference_subject, actual_subject=None):
    actual_subject = reference_subject if actual_subject is None else actual_subject
    for subject in (reference_subject, actual_subject):
        if not subject.get('valid') or not subject.get('orientation_valid'):
            return None
    directions = []
    for camera, subject in ((reference_camera, reference_subject), (actual_camera, actual_subject)):
        direction = np.asarray(subject['rotation']).T@(np.asarray(camera)-subject['position'])
        if np.linalg.norm(direction) < 1e-8:
            return None
        directions.append(direction/np.linalg.norm(direction))
    return float(np.arccos(np.clip(directions[0]@directions[1], -1., 1.)))


def smoothness(times, positions, quaternions, event_times=()):
    from scipy.spatial.transform import Rotation
    times, positions = np.asarray(times), np.asarray(positions)
    if len(times) < 7 or np.any(np.diff(times) <= 0):
        return dict(translation_jerk_rms_m_s3=None, angular_acceleration_rms_rad_s2=None, valid=False)
    h = np.diff(times)
    if np.ptp(h) > 1e-5:
        raise ValueError('smoothness requires uniformly sampled physical time; resample explicitly first')
    dt = float(np.mean(h))
    jerk = np.diff(positions, n=3, axis=0)/dt**3
    rotations = Rotation.from_quat(quaternions)
    # World-frame increments (R_next R_previous^-1) are comparable between frames.
    omega = (rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt
    acceleration = np.diff(omega, axis=0)/dt
    result = dict(valid=True, translation_jerk_rms_m_s3=float(np.sqrt(np.mean(np.sum(jerk**2, axis=1)))),
                  angular_acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.sum(acceleration**2, axis=1)))))
    result['events'] = []
    for t in event_times:
        jm, am = abs(times[2:-1]-t) <= .3, abs(times[1:-1]-t) <= .3
        result['events'].append(dict(time_s=t, window_s=.3,
            translation_jerk_rms_m_s3=None if not jm.any() else float(np.sqrt(np.mean(np.sum(jerk[jm]**2, axis=1)))),
            angular_acceleration_rms_rad_s2=None if not am.any() else float(np.sqrt(np.mean(np.sum(acceleration[am]**2, axis=1))))))
    return result


def trajectory_summary(rows, original, engine_summary, failure='', physical_audit=None):
    from scipy.spatial.transform import Rotation
    if not rows:
        return dict(completed=False, failure=failure or 'no executed samples', **engine_summary)
    times = np.array([r['t'] for r in rows])
    actual = np.array([r['camera_position'] for r in rows])
    rotations = np.array([r['camera_quaternion_xyzw'] for r in rows])
    poses = [original.sample(t) for t in times]
    nominal = np.array([p[0] for p in poses])
    errors = np.linalg.norm(actual-nominal, axis=1)
    orientation_errors = np.array([np.linalg.norm(Rotation.from_matrix(pose[1]@Rotation.from_quat(q).as_matrix().T).as_rotvec())
                                   for pose, q in zip(poses, rotations)])
    clearances = np.array([r['truth_clearance_m'] for r in rows])
    joint_margins = np.array([r['info']['joint_margin_rad'] for r in rows])
    command_residual = max(r['info'].get('qp_violation', 0.) for r in rows)
    phases = []
    for phase in original.task['shot_phase']:
        ids = np.flatnonzero((times >= phase['start_s']-1e-8) & (times <= phase['end_s']+1e-8))
        if not len(ids):
            phases.append(dict(name=phase['name'], time_coverage=0., geometric_progress=0., completed=False))
            continue
        # Progress diagnostic only: never used to rematch/reduce composition/scale/angle errors.
        sample_times = np.linspace(phase['start_s'], phase['end_s'], 101)
        if phase['motion'].lower() == 'pan':
            curve = Rotation.from_matrix(np.array([original.sample(t)[1] for t in sample_times]))
            distances = (Rotation.from_quat(rotations[ids[-1]])*curve.inv()).magnitude()
            progress_metric = 'SO3_orientation_nearest_reference_sample'
        else:
            curve = np.array([original.sample(t)[0] for t in sample_times])
            distances = np.linalg.norm(curve-actual[ids[-1]], axis=1)
            progress_metric = 'position_nearest_reference_sample'
        nearest = int(np.argmin(distances))
        coverage = float(np.clip((times[ids[-1]]-phase['start_s'])/(phase['end_s']-phase['start_s']), 0., 1.))
        phases.append(dict(name=phase['name'], time_coverage=coverage, geometric_progress=nearest/100.,
            progress_metric=progress_metric,
            delay_s=max(0., times[ids[-1]]-sample_times[nearest]),
            end_position_error_m=float(errors[ids[-1]]), end_orientation_error_rad=float(orientation_errors[ids[-1]]),
            completed=bool(coverage >= .99 and
                errors[ids[-1]] <= EVALUATION_PROTOCOL['position_completion_m'] and
                orientation_errors[ids[-1]] <= EVALUATION_PROTOCOL['orientation_completion_rad'])))
    finished = times[-1] >= original.duration-1e-6
    task_preserved = not any(r['info'].get('tpr_status') == 'degraded' for r in rows)
    safe = bool(np.all(clearances >= EVALUATION_PROTOCOL['collision_margin_m']) and command_residual <= 1e-6 and joint_margins.min() >= 0)
    if physical_audit is not None:
        safe &= bool(physical_audit.get('pass_constraints', False))
    event_times = [p['start_s'] for p in original.task['shot_phase'][1:]]
    for previous, current in zip(rows[:-1], rows[1:]):
        if (previous['info'].get('active_decision') != current['info'].get('active_decision') or
                bool(previous['info'].get('dynamic_constraint_count', 0)) != bool(current['info'].get('dynamic_constraint_count', 0))):
            event_times.append(current['t'])
    physical_camera_tracking = None
    if all('actual_camera_position' in row for row in rows):
        raw_positions = np.array([row['actual_camera_position'] for row in rows])
        raw_quaternions = np.array([row['actual_camera_quaternion_xyzw'] for row in rows])
        raw_angles = np.array([np.linalg.norm(Rotation.from_matrix(pose[1]@Rotation.from_quat(q).as_matrix().T).as_rotvec())
                               for pose, q in zip(poses, raw_quaternions)])
        physical_camera_tracking = dict(role='intermediate physical diagnostic',
            position_rms_m=float(np.sqrt(np.mean(np.sum((raw_positions-nominal)**2, axis=1)))),
            orientation_rms_rad=float(np.sqrt(np.mean(raw_angles**2))))
    return native(dict(**engine_summary, evaluation_protocol=EVALUATION_PROTOCOL,
        camera_output_source=rows[0].get('camera_output_source', 'execution'),
        physical_camera_tracking=physical_camera_tracking,
        evaluation_sha256=digest(EVALUATION_PROTOCOL), completed=bool(finished and not failure), failure=failure,
        within_budget_success=bool(finished and not failure and safe and task_preserved and all(p['completed'] for p in phases)),
        received_task_preserved=task_preserved,
        safety_pass=safe, collision_samples=int(np.sum(clearances < 0.)),
        clearance_violation_samples=int(np.sum(clearances < EVALUATION_PROTOCOL['collision_margin_m'])),
        min_truth_clearance_m=float(clearances.min()), min_joint_margin_rad=float(joint_margins.min()),
        max_command_residual=float(command_residual), elapsed_shot_s=float(times[-1]), time_overrun_s=max(0., times[-1]-original.duration),
        original_position_rms_m=float(np.sqrt(np.mean(errors**2))), original_position_max_m=float(errors.max()),
        original_orientation_rms_rad=float(np.sqrt(np.mean(orientation_errors**2))), phases=phases,
        visual_smoothness=smoothness(times, actual, rotations, sorted(set(event_times))), physical_audit=physical_audit,
        stage_completion_fraction=float(np.mean([p['completed'] for p in phases])),
        terminal_command=rows[-1]['command'],
        terminal_camera_speed_m_s=(float(np.linalg.norm(actual[-1]-actual[-2])/(times[-1]-times[-2])) if len(rows)>1 else None),
        terminal_angular_speed_rad_s=(float(np.linalg.norm((Rotation.from_quat(rotations[-1])*Rotation.from_quat(rotations[-2]).inv()).as_rotvec())/(times[-1]-times[-2])) if len(rows)>1 else None),
        unfinished_reference_budget_s=max(0., original.duration-times[-1]),
        certified_escape_samples=sum(bool(r['info'].get('certified_escape')) for r in rows),
        control_loop_wall_ms=dict(mean=1000*np.mean([r['info'].get('loop_wall_s', 0.) for r in rows]),
                                  p95=1000*np.percentile([r['info'].get('loop_wall_s', 0.) for r in rows], 95),
                                  p99=1000*np.percentile([r['info'].get('loop_wall_s', 0.) for r in rows], 99),
                                  max=1000*max(r['info'].get('loop_wall_s', 0.) for r in rows),
                                  over_20ms=sum(r['info'].get('loop_wall_s', 0.) > .02 for r in rows)),
        sam_visual_measurement='pending_offline_segmentation', observation_angle_measurement='requires_valid_subject_lidar_track'))


def save_json(path, value):
    Path(path).write_text(json.dumps(native(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
