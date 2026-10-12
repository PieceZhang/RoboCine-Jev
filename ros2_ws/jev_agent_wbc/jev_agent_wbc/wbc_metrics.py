"""Unfiltered command and physical-state telemetry; explicit acceptance gates."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .jvs_kinematics import JOINT_LOWER, JOINT_UPPER


def excess(values, limits):
    return float(np.max(np.maximum(np.abs(values) - limits, 0.)))


class Metrics:
    def __init__(self, folder, limits):
        self.folder = Path(folder).expanduser()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = (self.folder / 'telemetry.csv').open('w', newline='')
        self.writer = None
        self.limits = limits
        self.previous = None
        self.rows = []
        self.position_tolerance = .02
        self.orientation_tolerance = .035
        self.position_max_tolerance = np.inf
        self.trajectory = None

    def record(self, t, dt, state, command, measured, wheels, lateral, info, phase='tracking'):
        l = self.limits
        row = dict(time_s=float(t), phase=phase, dt_s=float(dt), **info)
        row['motion_active'] = int(phase == 'tracking' and (self.trajectory is None
            or bool(self.trajectory.motion_mask(t))))
        row.update({f'command_{n}': float(v) for n, v in zip(['v', 'omega'] + [f'dq{i}' for i in range(1, 7)], command)})
        row.update({f'measured_{n}': float(v) for n, v in zip(['v', 'omega'] + [f'dq{i}' for i in range(1, 7)], measured)})
        row.update({f'state_{n}': float(v) for n, v in zip(['x', 'y', 'yaw'] + [f'q{i}' for i in range(1, 7)], state)})
        row.update({f'wheel_{i}': float(v) for i, v in enumerate(wheels)})
        row.update({f'command_wheel_{i}': float(v) for i, v in enumerate(l.wheels @ command)})
        row['lateral_velocity'] = float(lateral)
        row['command_velocity_violation'] = excess(command, l.velocity)
        row['command_wheel_speed_violation'] = excess(l.wheels @ command, l.wheel_speed)
        row['measured_position_violation'] = float(max(0., np.max(JOINT_LOWER-state[3:]), np.max(state[3:]-JOINT_UPPER)))
        row['measured_velocity_violation'] = excess(measured, l.velocity)
        row['measured_wheel_speed_violation'] = excess(wheels, l.wheel_speed)
        row['measured_acceleration_violation'] = 0.
        row['measured_wheel_acceleration_violation'] = 0.
        row['command_acceleration_violation'] = 0.
        row['command_wheel_acceleration_violation'] = 0.
        row['command_step_max'] = 0.
        if self.previous is not None:
            old_command, old_measured, old_wheels = self.previous
            row['measured_acceleration_violation'] = excess((measured-old_measured)/dt, l.acceleration)
            row['measured_wheel_acceleration_violation'] = excess((wheels-old_wheels)/dt, l.wheel_acceleration)
            row['command_acceleration_violation'] = excess((command-old_command)/dt, l.acceleration)
            row['command_wheel_acceleration_violation'] = excess(l.wheels @ (command-old_command)/dt, l.wheel_acceleration)
            row['command_step_max'] = float(np.max(np.abs(command-old_command)))
        self.previous = command.copy(), measured.copy(), wheels.copy()
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows.append(row)
        if len(self.rows) % 50 == 0:
            self.file.flush()
        return row

    def finish(self, completed, failure=''):
        self.file.flush()
        rows = self.rows
        tracking = [r for r in rows if r['phase'] == 'tracking']
        active = (np.ones(len(tracking), dtype=bool) if self.trajectory is None else
                  self.trajectory.motion_mask([r['time_s'] for r in tracking]))
        action = [r for r, keep in zip(tracking, active) if keep]
        transitions = [r for r, keep in zip(tracking, active) if not keep]
        summary = dict(completed=bool(completed), failure=failure, samples=len(rows),
                       verification_scope='control_rate_samples_only',
                       accuracy_scope='shot_motion_intervals_only',
                       motion_samples=len(action), transition_samples=len(transitions))
        if rows:
            for key in rows[0]:
                if 'violation' in key or key in ('solve_s', 'command_step_max'):
                    summary['max_' + key] = max(r[key] for r in rows)
            summary['max_lateral_velocity'] = max(abs(r['lateral_velocity']) for r in rows)
        for prefix, selected in (('', action), ('full_execution_', tracking),
                                 ('transition_', transitions)):
            if not selected:
                continue
            for key in ('position_error', 'orientation_error'):
                values = np.array([r[key] for r in selected])
                summary[prefix + key + '_rms'] = float(np.sqrt(np.mean(values**2)))
                summary[prefix + key + '_max'] = float(np.max(values))
                summary[prefix + key + '_p95'] = float(np.quantile(values, .95))
                intervals = np.array([r['dt_s'] for r in selected])
                summary[prefix + key + '_time_weighted_rms'] = float(np.sqrt(np.average(values**2, weights=intervals)))
        if tracking:
            summary['base_travel_m'] = float(np.hypot(tracking[-1]['state_x']-tracking[0]['state_x'], tracking[-1]['state_y']-tracking[0]['state_y']))
            summary['arm_motion_rad'] = max(max(r[f'state_q{i}'] for r in tracking)-min(r[f'state_q{i}'] for r in tracking) for i in range(1, 7))
            base_path = np.array([[r['state_x'], r['state_y']] for r in tracking])
            summary['base_path_length_m'] = float(np.sum(np.linalg.norm(np.diff(base_path, axis=0), axis=1)))
            if 'arm_horizontal_reach_m' in tracking[0]:
                reach = [r['arm_horizontal_reach_m'] for r in tracking]
                reserve = [r['arm_joint_reserve_rad'] for r in tracking]
                conditioning = [r['arm_sigma_min'] for r in tracking]
                summary.update(arm_reach_max_m=float(np.max(reach)),
                    arm_reach_p95_m=float(np.quantile(reach, .95)),
                    arm_joint_reserve_min_rad=float(np.min(reserve)),
                    arm_near_joint_stop_fraction=float(np.mean(np.array(reserve) < .05)),
                    arm_sigma_min_p10=float(np.quantile(conditioning, .1)))
            if 'controller_s' in tracking[0]:
                summary['controller_p95_s'] = float(np.quantile([r['controller_s'] for r in tracking], .95))
        summary['thresholds'] = dict(position_rms_m=self.position_tolerance,
                                     orientation_rms_rad=self.orientation_tolerance,
                                     command_violation=1e-6, measured_violation=1e-3,
                                     lateral_velocity_m_s=.005)
        if np.isfinite(self.position_max_tolerance):
            summary['thresholds']['position_max_m'] = self.position_max_tolerance
        # Shot tracking does not require both base and arm to move: a correct pan
        # can be performed entirely by the arm while its camera centre stays fixed.
        summary['tracking_accuracy_pass'] = bool(completed and action and not failure
            and summary['position_error_rms'] <= self.position_tolerance
            and summary['position_error_max'] <= self.position_max_tolerance
            and summary['orientation_error_rms'] <= self.orientation_tolerance)
        summary['sampled_checks_pass'] = bool(summary['tracking_accuracy_pass']
            and summary['max_lateral_velocity'] <= .005
            and all(v <= (1e-3 if 'measured_' in k else 1e-6) for k, v in summary.items() if 'violation' in k))
        # A ROS-rate report alone cannot certify all intermediate physics steps.
        summary['stage1_pass'] = False
        summary['acceptance_note'] = 'Full physics-step audit required for Stage 1 acceptance'
        if tracking and self.trajectory is not None and 'camera_x' in tracking[0]:
            self.shot_report(summary, tracking)
        (self.folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary

    def shot_report(self, summary, rows):
        trajectory = self.trajectory
        times = np.array([r['time_s'] for r in rows])
        wall = np.array([r['wall_elapsed_s'] for r in rows])
        positions = np.array([[r[f'camera_{a}'] for a in 'xyz'] for r in rows])
        forward = np.array([[r[f'camera_forward_{a}'] for a in 'xyz'] for r in rows])
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        angles = np.arccos(np.clip(np.sum(forward[1:]*forward[:-1], axis=1), -1., 1.))
        summary.update(execution_duration_s=float(wall[-1]),
            camera_step_max_m=float(np.max(steps)) if len(steps) else 0.,
            camera_step_max_rad=float(np.max(angles)) if len(angles) else 0.,
            camera_speed_p99_m_s=float(np.quantile(steps/np.maximum(np.diff(wall), 1e-6), .99)) if len(steps) else 0.)
        if trajectory.subject is None:
            return
        target = trajectory.subject
        summary['subject_odom_m'] = target.tolist()
        to_target = target-positions
        to_target /= np.linalg.norm(to_target, axis=1)[:, None]
        focus = np.arccos(np.clip(np.sum(to_target*forward, axis=1), -1., 1.))
        active = trajectory.motion_mask(times)
        summary['subject_focus_rms_rad'] = float(np.sqrt(np.mean(focus[active]**2))) if np.any(active) else None
        shot_results = []
        for shot in trajectory.shots:
            start = shot.get('motion_start_s', shot['start_s'])
            end = shot.get('motion_end_s', shot['end_s'])
            # Completion, like accuracy, is measured at action endpoints. A
            # following hold cannot hide unfinished requested camera motion.
            first = int(np.argmin(abs(times-start)))
            last = int(np.argmin(abs(times-end)))
            desired_first = trajectory.sample(start)[0]-target
            desired_last = trajectory.sample(end)[0]-target
            actual_first, actual_last = positions[first]-target, positions[last]-target
            kind = shot['type']
            if kind in ('orbit', 'pan') and np.linalg.norm(desired_last-desired_first) > .02:
                def sweep(a, b):
                    return np.arctan2(a[0]*b[1]-a[1]*b[0], a[:2]@b[:2])
                expected = float(sweep(desired_first, desired_last))
                actual = float(sweep(actual_first, actual_last))
                unit = 'rad'
            else:
                expected = float(np.linalg.norm(desired_last[:2])-np.linalg.norm(desired_first[:2]))
                actual = float(np.linalg.norm(actual_last[:2])-np.linalg.norm(actual_first[:2]))
                unit = 'm'
            ratio = actual/expected if abs(expected) > 1e-4 else 1.
            selected = (times >= start)&(times <= end)
            motion_first = int(np.argmin(abs(times-shot.get('motion_start_s', start))))
            motion_last = int(np.argmin(abs(times-shot.get('motion_end_s', end))))
            # The explicit stationary windows are available for repositioning
            # between shots. Direction applies to the planned motion interval;
            # camera/command continuity is still checked across every window.
            coarse_time = np.arange(wall[motion_first], wall[motion_last], .5)
            coarse = np.column_stack([np.interp(coarse_time, wall, positions[:, j]) for j in range(3)])-target
            progress = (np.unwrap(np.arctan2(coarse[:, 1], coarse[:, 0])) if unit == 'rad'
                        else np.linalg.norm(coarse[:, :2], axis=1))
            # Ignore <=2 mm / 2 mrad per half-second feedback noise. A smooth
            # but repeated reversal is not the requested Orbit/Pan/MoveOut.
            reverse = float(np.sum(np.maximum(-np.sign(expected)*np.diff(progress)-.002, 0.)))
            reverse_limit = (.02 if unit == 'rad' else .05)+.25*abs(expected)
            shot_results.append(dict(type=kind, expected=expected, actual=actual, unit=unit,
                accuracy_start_s=start, accuracy_end_s=end,
                completion_ratio=ratio, motion_completed=bool(.70 <= ratio <= 1.30 and times[-1] >= end-.1),
                reverse_motion=reverse, reverse_motion_limit=reverse_limit,
                direction_check_start_s=shot.get('motion_start_s', start),
                direction_check_end_s=shot.get('motion_end_s', end),
                motion_direction_pass=bool(reverse <= reverse_limit),
                position_rms_m=float(np.sqrt(np.mean([r['position_error']**2 for r in np.array(rows, dtype=object)[selected]]))) if np.any(selected) else None,
                position_max_m=float(np.max([r['position_error'] for r in np.array(rows, dtype=object)[selected]])) if np.any(selected) else None,
                orientation_rms_rad=float(np.sqrt(np.mean([r['orientation_error']**2 for r in np.array(rows, dtype=object)[selected]]))) if np.any(selected) else None))
        summary['shots'] = shot_results
        summary['shot_motions_pass'] = all(s['motion_completed'] and s['motion_direction_pass'] for s in shot_results)
        summary['command_continuity_pass'] = all(summary.get('max_'+name, float('inf')) <= 1e-6
            for name in ('command_acceleration_violation', 'command_wheel_acceleration_violation',
                         'command_position_violation', 'command_velocity_violation', 'qp_violation'))
        summary['sampled_camera_continuity_pass'] = bool(summary['camera_step_max_m'] <= .10
                                                        and summary['camera_step_max_rad'] <= .20)
        summary['minimum_command_clearance_m'] = min(r.get('command_sweep_lower_m', float('inf')) for r in rows)
        summary['approximate_success'] = bool(summary['completed'] and not summary['failure']
            and summary['shot_motions_pass'] and summary['command_continuity_pass']
            and summary['sampled_camera_continuity_pass'] and summary['minimum_command_clearance_m'] >= .025
            and summary.get('max_measured_position_violation', float('inf')) <= .001
            and summary['subject_focus_rms_rad'] is not None and summary['subject_focus_rms_rad'] <= .15)

    def close(self):
        self.file.close()


class PhysicsAudit:
    """Audit *every* physics step, including contact transients between QP ticks."""
    def __init__(self, folder, limits, physics_dt):
        self.limits, self.dt = limits, physics_dt
        self.file = gzip.open(Path(folder) / 'physics.csv.gz', 'wt', newline='')
        self.writer = None
        self.previous = None
        self.maxima = {}
        self.samples = 0

    def record(self, t, q, measured, wheels, lateral):
        l = self.limits
        row = dict(time_s=t, position_violation=float(max(0., np.max(JOINT_LOWER-q), np.max(q-JOINT_UPPER))),
                   velocity_violation=excess(measured, l.velocity), wheel_speed_violation=excess(wheels, l.wheel_speed),
                   acceleration_violation=0., wheel_acceleration_violation=0., lateral_velocity=abs(float(lateral)))
        if self.previous is not None:
            old, old_wheels = self.previous
            row['acceleration_violation'] = excess((measured-old)/self.dt, l.acceleration)
            row['wheel_acceleration_violation'] = excess((wheels-old_wheels)/self.dt, l.wheel_acceleration)
        self.previous = measured.copy(), wheels.copy()
        for key, v in row.items():
            if key != 'time_s':
                self.maxima[key] = max(self.maxima.get(key, 0.), v)
        row.update({f'measured_{i}': float(v) for i, v in enumerate(measured)})
        row.update({f'wheel_{i}': float(v) for i, v in enumerate(wheels)})
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.samples += 1

    def finish(self):
        self.file.close()
        return dict(samples=self.samples, physics_dt=self.dt, maxima=self.maxima,
                    pass_constraints=all(v <= (.005 if key == 'lateral_velocity' else .001)
                                         for key, v in self.maxima.items()))
