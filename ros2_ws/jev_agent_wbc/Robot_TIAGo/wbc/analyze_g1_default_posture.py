#!/usr/bin/env python3
"""Genuine Pink walking-only posture candidates against recorded G1 yaw tasks.

The recorded base/lower-body trajectory is immutable. Upper joints integrate
native bounded Pink velocities under an ideal kinematic model; this is not a
physical counterfactual and cannot prove improved AGILE response or parking.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual, target):
    return [float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
            float(Rotation.from_matrix(actual[:3, :3]@target[:3, :3].T).magnitude())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g1-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    folder = args.g1_result.resolve()
    source = folder/'sources'
    backend = load('g1_frozen_posture', source/'g1_backend.py')
    collision = load('g1_frozen_collision', source/'pink_collision.py')
    rows = [json.loads(x) for x in (folder/'measurements.jsonl').open()]
    sensor = json.loads((folder/'sensor_mounts.json').read_text())
    offsets = json.loads((folder/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    metadata = json.loads((folder/'fine_collision_geometry.json').read_text())
    import yaml
    io = yaml.safe_load((source/'height_policy_io.yaml').read_text())['articulations']['robot']
    home = dict(zip(io['joint_names'], io['default_joint_pos']))
    names = rows[0]['joint_names']
    legs = [n for n in names if any(t in n for t in ('hip', 'knee', 'ankle'))]
    dt = .02
    reports = []
    with (args.output/'trace.jsonl').open('x') as stream:
        for cost in (.03, .1, .15):
            for case in ('yaw_left', 'yaw_right'):
                recorded = [r for r in rows if r['case'] == case and r['phase'] == 'move']
                solver = backend.PinkUpperBody(legs, names,
                    np.array(sensor['camera']['T_parent_optical']), source/'g1_29dof.urdf',
                    posture_home=home, posture_cost=cost, damping_cost=.02, waist_posture_cost=.15)
                solver.collision_helper = collision
                solver.collision_offsets = offsets
                solver.collision_geometry = collision.geometry_model(solver.model, metadata, offsets)
                pin = solver.pin
                data, geometry_data = solver.model.createData(), solver.collision_geometry.createData()
                upper_names = solver.owners['pink']
                upper_q = {n: recorded[0]['joint_positions'][names.index(n)] for n in upper_names}
                # At a task transition the last accepted target=None packet can
                # contain no motion entries; the native worker starts these at
                # zero rather than deleting its managed derivative coordinates.
                solver.previous = {n: recorded[0]['pink_upper_velocity'].get(n, 0.) for n in upper_names}
                started, samples = time.monotonic(), []
                for index, row in enumerate(recorded):
                    delay = started+index*dt-time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    assert np.linalg.norm(row['requested_body_velocity']) > 0
                    measured = dict(zip(names, row['joint_positions']))
                    measured.update(upper_q)
                    base = np.asarray(row['T_world_base'])
                    target = np.asarray(row['T_world_reference_optical'])
                    camera = solver.predicted_camera(measured, base)
                    previous = solver.previous.copy()
                    velocity = solver.velocity(measured, base, target, dt)
                    q = collision.full_configuration(solver.model, measured)
                    pin.forwardKinematics(solver.model, data, q)
                    pin.updateGeometryPlacements(solver.model, data, solver.collision_geometry, geometry_data, q)
                    pin.computeDistances(solver.model, data, solver.collision_geometry, geometry_data, q)
                    distance = float(min(x.min_distance for x in geometry_data.distanceResults))
                    margin = float(min(np.min(q-solver.model.lowerPositionLimit),
                                       np.min(solver.model.upperPositionLimit-q)))
                    item = dict(posture_cost=cost, case=case, stamp_s=row['stamp_s'],
                        camera_error=error(camera, target), native_velocity=velocity,
                        modeled_upper_joint_positions=upper_q.copy(),
                        modeled_upper_position_l2=float(np.linalg.norm(list(upper_q.values()))),
                        original_position_margin_rad=margin, all548_min_distance_m=distance,
                        peak_issued_speed_rad_s=float(max(map(abs, velocity.values()))),
                        peak_issued_acceleration_rad_s2=float(max(abs(velocity[n]-previous[n])/dt for n in velocity)),
                        native_upper_barrier=solver.last_collision_report)
                    samples.append(item)
                    stream.write(json.dumps(item, allow_nan=False)+'\n')
                    upper_q = {n: upper_q[n]+dt*velocity[n] for n in upper_names}
                elapsed = time.monotonic()-started
                e = np.array([r['camera_error'] for r in samples])
                rmse = np.sqrt(np.mean(e*e, axis=0))
                peak_speed = max(r['peak_issued_speed_rad_s'] for r in samples)
                peak_acc = max(r['peak_issued_acceleration_rad_s2'] for r in samples)
                minimum_distance = min(r['all548_min_distance_m'] for r in samples)
                report = dict(case=case, walking_posture_cost=cost, samples=len(samples),
                    original_absolute_start_s=recorded[0]['stamp_s'], original_absolute_end_s=recorded[-1]['stamp_s'],
                    modeled_duration_s=len(recorded)*dt, wall_seconds=elapsed,
                    timing_pass=elapsed <= len(recorded)*dt*1.05,
                    camera_position_rmse_m=float(rmse[0]), camera_orientation_rmse_rad=float(rmse[1]),
                    camera_position_max_m=float(e[:, 0].max()), camera_orientation_max_rad=float(e[:, 1].max()),
                    camera_terminal_error=e[-1].tolist(),
                    original_camera_budget_pass=bool(np.all(rmse <= [.05, .15]) and np.all(e[-1] <= [.05, .15])),
                    mean_modeled_upper_position_l2=float(np.mean([r['modeled_upper_position_l2'] for r in samples])),
                    final_modeled_upper_joint_positions=samples[-1]['modeled_upper_joint_positions'],
                    minimum_all548_distance_m=minimum_distance,
                    original_limits_and_slew_pass=bool(min(r['original_position_margin_rad'] for r in samples) >= -1e-8 and peak_speed <= .500001 and peak_acc <= 1.000001),
                    discrete_clearance_pass=minimum_distance >= .0065-1e-6)
                reports.append(report)
                print(json.dumps(report), flush=True)
    inputs = [source/'g1_backend.py', source/'pink_collision.py', source/'g1_29dof.urdf',
              source/'height_policy_io.yaml', folder/'measurements.jsonl', folder/'sensor_mounts.json',
              folder/'pink_collision_initialization.json', folder/'fine_collision_geometry.json', Path(__file__).resolve()]
    result = dict(schema='readonly_g1_walking_default_posture/v1', pace=1, physical_validation=False,
        scope=__doc__,
        unchanged=dict(all_upper_dofs=15, frame_position_cost=1., frame_orientation_cost=.3,
                       frame_gain=.1, frame_lm_damping=.001, damping_cost=.02, waist_posture_cost=.15,
                       dt_s=dt, speed_rad_s=.5, issued_acceleration_rad_s2=1.,
                       barrier_pairs=473, audited_all_pairs=548, barrier_distance_m=.0065,
                       world_camera_target_and_timing='original recorded optical target at original absolute stamp',
                       lower_body_and_base='original measured trajectory, cannot respond to changed upper model',
                       soft_posture_target='original source default; changed cost only during nonzero preferred body velocity'),
        remaining_scope='No freeze of upper DOFs, no physical servo/ground/learned-policy feedback model. Source-model feasibility only; does not validate 9cases,6Dcamera,stop or improve actual yaw.',
        evidence_files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        cases=reports)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')


if __name__ == '__main__':
    main()
