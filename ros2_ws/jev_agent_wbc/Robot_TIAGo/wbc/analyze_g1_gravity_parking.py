#!/usr/bin/env python3
"""Read-only measured terminal tracking decomposition; never a physical replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def slope(t, x):
    t = t-t.mean()
    return float(t @ (x-x.mean())/(t @ t))


def audit(folder):
    rows = [json.loads(line) for line in (folder/'measurements.jsonl').open()]
    cases = []
    for case in dict.fromkeys(r['case'] for r in rows if r['case'] != 'settle'):
        stop = [r for r in rows if r['case'] == case and r['phase'] == 'stop']
        terminal = [r for r in stop if r['stamp_s'] >= stop[-1]['stamp_s']-.5-1e-8]
        names = terminal[0]['joint_names']
        times = np.array([r['stamp_s'] for r in terminal])
        q = np.array([r['joint_positions'] for r in terminal])
        ref = np.array([r['applied_joint_position_target'] for r in terminal])
        qd = np.array([r['measured_joint_velocity'] for r in terminal])
        joints = []
        for i, name in enumerate(names):
            native = [r['pink_upper_velocity'].get(name) for r in terminal]
            present = [x for x in native if x is not None]
            joints.append(dict(name=name, actual_span_rad=float(np.ptp(q[:, i])),
                actual_position_slope_rad_s=slope(times, q[:, i]),
                actual_mean_velocity_rad_s=float(qd[:, i].mean()),
                drive_target_span_rad=float(np.ptp(ref[:, i])),
                drive_target_slope_rad_s=slope(times, ref[:, i]),
                target_minus_actual_mean_rad=float((ref[:, i]-q[:, i]).mean()),
                target_minus_actual_peak_abs_rad=float(np.max(np.abs(ref[:, i]-q[:, i]))),
                native_mean_velocity_rad_s=float(np.mean(present)) if present else None,
                native_peak_abs_velocity_rad_s=float(np.max(np.abs(present))) if present else None,
                mean_feedforward_Nm=float(np.mean([r['upper_gravity_feedforward'][name]
                    for r in terminal if name in r['upper_gravity_feedforward']])) if present else None))
        left_wrist = {}
        for name in ('left_wrist_pitch_joint', 'left_wrist_roll_joint', 'left_wrist_yaw_joint'):
            i = names.index(name)
            valid = [r for r in rows if r['case'] == case and name in r['pink_upper_velocity']]
            leads = np.array([r['applied_joint_position_target'][i]-r['joint_positions'][i] for r in valid])
            left_wrist[name] = dict(whole_case_peak_abs_lead_rad=float(np.max(np.abs(leads))),
                whole_case_lead_49mrad_count=int(np.count_nonzero(np.abs(leads) >= .049)),
                whole_case_samples=len(valid))
        cases.append(dict(case=case, terminal_window_s=float(times[-1]-times[0]), samples=len(terminal),
            final_base_angular_velocity_rad_s=terminal[-1]['base_angular_velocity_world'],
            final_base_linear_velocity_m_s=terminal[-1]['base_linear_velocity_world'],
            largest_actual_spans=sorted(joints, key=lambda x: x['actual_span_rad'], reverse=True)[:6],
            upper_joint_decomposition=[j for j in joints if j['native_mean_velocity_rad_s'] is not None],
            left_wrist_reference_lead=left_wrist,
            terminal_antiwindup_count=sum(bool(r.get('upper_reference_report', {}).get('antiwindup_active'))
                for r in terminal if r.get('upper_reference_report') is not None),
            terminal_camera_position_error_peak_m=max(r['position_error_m'] for r in terminal),
            terminal_camera_orientation_error_peak_rad=max(r['orientation_error_rad'] for r in terminal)))
    paths = [folder/'measurements.jsonl', folder/'summary.json', folder/'sources/g1_backend.py',
             folder/'sources/adaptive_runner.py', folder/'sources/upper_reference.py']
    return dict(folder=str(folder), cases=cases,
        evidence_files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = dict(schema='readonly_g1_measured_gravity_parking/v1',
        scope='Recorded post-step physical states and prior-state asynchronous Pink packets. Correlations and regression slopes are descriptive; no contact wrench reconstruction or physical counterfactual.',
        original_window_s=.5, original_joint_span_limit_rad=.01,
        original_final_base_angular_speed_limit_rad_s=.03,
        runs=[audit(folder.resolve()) for folder in a.result],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
