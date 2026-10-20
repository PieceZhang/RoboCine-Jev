#!/usr/bin/env python3
"""Read-only associations in G1 yaw logs; no policy/physics intervention."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import pinocchio as pin


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 4 or min(a.std(), b.std()) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def relative_upper_momentum(model, data, row):
    """Source-inertia model H about pelvis from joint motion, base fixed.

    This excludes floating-base motion, gravity/contact wrenches, gyroscopic
    terms and armature. It is NOT an actual pelvis reaction-torque measurement.
    """
    values = dict(zip(row['joint_names'], row['joint_positions']))
    rates = dict(zip(row['joint_names'], row['measured_joint_velocity']))
    q, v = pin.neutral(model), np.zeros(model.nv)
    for name in list(model.names)[1:]:
        joint = model.joints[model.getJointId(name)]
        q[joint.idx_q] = values[name]
        v[joint.idx_v] = rates[name]
    pin.forwardKinematics(model, data, q, v)
    momentum = np.zeros(3)
    waist = model.getJointId('waist_yaw_joint')
    for index in range(1, model.njoints):
        ancestor = index
        while ancestor and ancestor != waist:
            ancestor = model.parents[ancestor]
        if not ancestor:
            continue
        inertia, placement, velocity = model.inertias[index], data.oMi[index], data.v[index]
        com = placement.translation+placement.rotation@inertia.lever
        linear = placement.rotation@(velocity.linear+np.cross(velocity.angular, inertia.lever))
        angular = placement.rotation@velocity.angular
        momentum += placement.rotation@inertia.inertia@placement.rotation.T@angular
        momentum += np.cross(com, inertia.mass*linear)
    return float((np.asarray(row['T_world_base'])[:3, :3]@momentum)[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports, paths = [], [Path(__file__).resolve()]
    traces = []
    for folder in args.runs:
        raw = folder/'measurements.jsonl'
        source = folder/'sources/g1_29dof.urdf'
        rows = [json.loads(x) for x in raw.open()]
        model = pin.buildModelFromUrdf(str(source))
        data = model.createData()
        actuator = json.loads((folder/'actuator_audit.json').read_text())
        caps = dict(zip(actuator['joint_names'], actuator['effort_limits']))
        paths += [raw, source, folder/'actuator_audit.json', folder/'velocity_servo_config.json',
                  folder/'sources/adaptive_runner.py', folder/'provenance.json']
        if (folder/'contact_materials.json').exists():
            paths.append(folder/'contact_materials.json')
        for case in ('yaw_left', 'yaw_right'):
            moving = [r for r in rows if r['case'] == case and r['phase'] == 'move']
            if not moving:
                continue
            start = moving[0]['stamp_s']
            case_traces = []
            for row in moving:
                ids = {n: i for i, n in enumerate(row['joint_names'])}
                upper = [n for n in ids if n == 'waist_yaw_joint' or any(t in n for t in ('shoulder', 'elbow', 'wrist'))]
                right = [n for n in upper if n.startswith('right_')]
                qd = lambda n: float(row['measured_joint_velocity'][ids[n]])
                tau = lambda n: float(row['applied_joint_effort_estimate'][ids[n]])
                item = dict(run=folder.name, case=case, stamp_s=row['stamp_s'], elapsed_s=row['stamp_s']-start,
                    request_yaw=row['requested_body_velocity'][2], native_yaw=row['applied_policy_velocity'][2],
                    measured_yaw_body=row['measured_body_velocity'][2],
                    measured_yaw_world=row['base_angular_velocity_world'][2],
                    upper_qdot_l2=float(np.linalg.norm([qd(n) for n in upper])),
                    right_qdot_l2=float(np.linalg.norm([qd(n) for n in right])),
                    upper_position_l2=float(np.linalg.norm([row['joint_positions'][ids[n]] for n in upper])),
                    waist_yaw_q=float(row['joint_positions'][ids['waist_yaw_joint']]),
                    waist_yaw_qdot=qd('waist_yaw_joint'), waist_yaw_applied_tau=tau('waist_yaw_joint'),
                    upper_applied_power_estimate_w=float(sum(tau(n)*qd(n) for n in upper)),
                    source_relative_upper_angular_momentum_world_z=relative_upper_momentum(model, data, row),
                    left_hip_yaw_tau=tau('left_hip_yaw_joint'), right_hip_yaw_tau=tau('right_hip_yaw_joint'),
                    maximum_all_effort_fraction=float(max(abs(tau(n))/caps[n] for n in ids)),
                    controller_fresh=row['controller_status'] == 'fresh_command',
                    native_standing_mode=row['native_standing_mode'],
                    base_height_m=row['base_height_m'], base_tilt_rad=row['base_tilt_rad'])
                case_traces.append(item)
                traces.append(item)
            windows = []
            for index in range(5):
                selected = [r for r in case_traces if index <= r['elapsed_s'] < index+1.]
                if not selected:
                    continue
                mean = lambda k: float(np.mean([r[k] for r in selected]))
                windows.append(dict(elapsed_window_s=[index, index+1], samples=len(selected),
                    native_yaw_mean=mean('native_yaw'), actual_body_yaw_mean=mean('measured_yaw_body'),
                    actual_world_yaw_mean=mean('measured_yaw_world'), fresh_fraction=mean('controller_fresh'),
                    maximum_effort_fraction=max(r['maximum_all_effort_fraction'] for r in selected),
                    right_qdot_l2_mean=mean('right_qdot_l2'), upper_qdot_l2_mean=mean('upper_qdot_l2'),
                    upper_position_l2_mean=mean('upper_position_l2'), waist_yaw_qdot_mean=mean('waist_yaw_qdot'),
                    waist_yaw_tau_mean=mean('waist_yaw_applied_tau'),
                    waist_yaw_tau_max_abs=max(abs(r['waist_yaw_applied_tau']) for r in selected),
                    upper_power_estimate_mean_w=mean('upper_applied_power_estimate_w'),
                    relative_upper_Hz_mean=mean('source_relative_upper_angular_momentum_world_z'),
                    left_hip_yaw_tau_mean=mean('left_hip_yaw_tau'),
                    right_hip_yaw_tau_mean=mean('right_hip_yaw_tau'),
                    base_height_mean_m=mean('base_height_m'), base_tilt_mean_rad=mean('base_tilt_rad')))
            bins = []
            for index in range(25):
                selected = [r for r in case_traces if index*.2 <= r['elapsed_s'] < (index+1)*.2]
                if len(selected) < 2:
                    continue
                duration = selected[-1]['stamp_s']-selected[0]['stamp_s']
                bins.append(dict(
                    native_yaw=float(np.mean([r['native_yaw'] for r in selected])),
                    actual_yaw=float(np.mean([r['measured_yaw_world'] for r in selected])),
                    yaw_acceleration=(selected[-1]['measured_yaw_world']-selected[0]['measured_yaw_world'])/duration,
                    relative_upper_Hz_rate=(selected[-1]['source_relative_upper_angular_momentum_world_z']-
                        selected[0]['source_relative_upper_angular_momentum_world_z'])/duration,
                    waist_tau=float(np.mean([r['waist_yaw_applied_tau'] for r in selected]))))
            reports.append(dict(run=folder.name, case=case, samples=len(moving), windows=windows,
                native_standing_samples=sum(r['native_standing_mode'] for r in case_traces),
                exploratory_200ms_bin_correlations=dict(
                    waist_command_tau_vs_yaw_acceleration=correlation([r['waist_tau'] for r in bins], [r['yaw_acceleration'] for r in bins]),
                    relative_upper_Hz_rate_vs_yaw_acceleration=correlation([r['relative_upper_Hz_rate'] for r in bins], [r['yaw_acceleration'] for r in bins]),
                    native_input_vs_measured_rate=correlation([r['native_yaw'] for r in bins], [r['actual_yaw'] for r in bins])),
                correlation_scope='Within-case descriptive associations only; feedback/phase confounding and contact torques prevent causal attribution or feedforward fitting.'))
    with (args.output/'trace.jsonl').open('x') as stream:
        for row in traces:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    result = dict(schema='readonly_g1_yaw_coupling/v1',
        scope='Measured-state analysis only, no physics/policy change or feedforward fit. Applied joint torque is an actuator-command/model estimate, not a sensor.',
        momentum_scope=relative_upper_momentum.__doc__,
        evidence_files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        runs=reports)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
