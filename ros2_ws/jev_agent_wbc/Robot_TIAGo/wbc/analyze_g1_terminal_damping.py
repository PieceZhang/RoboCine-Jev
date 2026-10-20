#!/usr/bin/env python3
"""Read-only genuine Pink counterfactuals on recorded G1 terminal states.

This does not integrate a physical trajectory or certify parking. Each solve
starts from a recorded measured state and predicts one original 20ms step.
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


def pose_error(actual, desired):
    return [float(np.linalg.norm(actual[:3, 3]-desired[:3, 3])),
            float(Rotation.from_matrix(actual[:3, :3]@desired[:3, :3].T).magnitude())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g1-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    folder = args.g1_result.resolve()
    source = folder/'sources'
    backend = load('recorded_g1_backend', source/'g1_backend.py')
    collision = load('recorded_g1_collision', source/'pink_collision.py')
    rows = [json.loads(line) for line in (folder/'measurements.jsonl').open()]
    metadata = json.loads((folder/'fine_collision_geometry.json').read_text())
    initialization = json.loads((folder/'pink_collision_initialization.json').read_text())
    camera = np.array(json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    names = rows[0]['joint_names']
    legs = [n for n in names if any(t in n for t in ('hip', 'knee', 'ankle'))]
    import yaml
    io = yaml.safe_load((source/'height_policy_io.yaml').read_text())['articulations']['robot']
    home = dict(zip(io['joint_names'], io['default_joint_pos']))
    parameters = dict(posture_home=home, posture_cost=.03, waist_posture_cost=.15)
    dt = .02
    cases = ('left', 'camera_shot', 'yaw_left', 'yaw_right')
    costs = (.02, .1, .2, .3)
    reports = []
    with (args.output/'trace.jsonl').open('x') as stream:
        for cost in costs:
            solver = backend.PinkUpperBody(legs, names, camera, source/'g1_29dof.urdf',
                                           damping_cost=cost, **parameters)
            solver.collision_helper = collision
            solver.collision_offsets = initialization['physical_body_frame_offsets']
            solver.collision_geometry = collision.geometry_model(
                solver.model, metadata, solver.collision_offsets)
            pin = solver.pin
            geometry_data = solver.collision_geometry.createData()
            full_data = solver.model.createData()

            def full_distance(joints):
                q = collision.full_configuration(solver.model, joints)
                pin.forwardKinematics(solver.model, full_data, q)
                pin.updateGeometryPlacements(solver.model, full_data,
                                             solver.collision_geometry, geometry_data, q)
                pin.computeDistances(solver.model, full_data, solver.collision_geometry, geometry_data, q)
                return float(min(x.min_distance for x in geometry_data.distanceResults))

            for case in cases:
                recorded = [r for r in rows if r['case'] == case and r['phase'] == 'stop']
                end = recorded[-1]['stamp_s']
                recorded = [r for r in recorded if r['stamp_s'] >= end-.5-1e-8]
                started = time.monotonic()
                samples = []
                for index, row in enumerate(recorded):
                    delay = started+index*dt-time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    measured = dict(zip(names, row['joint_positions']))
                    base = np.asarray(row['T_world_base'])
                    target = np.asarray(row['T_world_reference_optical'])
                    # The log records post-step state and a fresh prior-state
                    # packet. Center every comparison on the same most recent
                    # available native command; do not claim packet-exact replay.
                    previous = row['pink_upper_velocity'].copy()
                    solver.previous = previous.copy()
                    before = solver.predicted_camera(measured, base)
                    before_distance = full_distance(measured)
                    velocity = solver.velocity(measured, base, target, dt,
                        posture_reference=row['pink_parking_posture']['targets'])
                    predicted = measured.copy()
                    for name, value in velocity.items():
                        predicted[name] += dt*value
                    after = solver.predicted_camera(predicted, base)
                    after_distance = full_distance(predicted)
                    q = collision.full_configuration(solver.model, predicted)
                    margin = float(min(np.min(q-solver.model.lowerPositionLimit),
                                       np.min(solver.model.upperPositionLimit-q)))
                    peak_speed = max(map(abs, velocity.values()))
                    peak_acceleration = max(abs(velocity[n]-previous[n])/dt for n in velocity)
                    item = dict(case=case, damping_cost=cost, stamp_s=row['stamp_s'], dt_s=dt,
                        measured_camera_error=pose_error(np.asarray(row['T_world_camera_optical']), target),
                        source_fk_error_before=pose_error(before, target),
                        source_fk_error_after_one_step=pose_error(after, target),
                        camera_step_position_m=float(np.linalg.norm(after[:3, 3]-before[:3, 3])),
                        camera_step_angle_rad=float(Rotation.from_matrix(after[:3, :3]@before[:3, :3].T).magnitude()),
                        native_velocity=velocity, native_velocity_l2=float(np.linalg.norm(list(velocity.values()))),
                        peak_native_speed_rad_s=peak_speed, peak_issued_acceleration_rad_s2=peak_acceleration,
                        all548_distance_before_m=before_distance,
                        all548_distance_after_one_step_m=after_distance,
                        original_model_position_margin_rad=margin,
                        upper_barrier=solver.last_collision_report,
                        bounds_pass=bool(margin >= -1e-8 and peak_speed <= .500001 and peak_acceleration <= 1.000001),
                        next_discrete_clearance_pass=bool(after_distance >= .0065-1e-6))
                    samples.append(item)
                    stream.write(json.dumps(item, allow_nan=False)+'\n')
                elapsed = time.monotonic()-started
                reports.append(dict(case=case, damping_cost=cost, samples=len(samples),
                    source_span_s=recorded[-1]['stamp_s']-recorded[0]['stamp_s'],
                    scheduled_duration_s=len(recorded)*dt, wall_seconds=elapsed,
                    timing_pass=elapsed <= len(recorded)*dt*1.05,
                    maximum_native_speed_rad_s=max(x['peak_native_speed_rad_s'] for x in samples),
                    mean_native_velocity_l2=float(np.mean([x['native_velocity_l2'] for x in samples])),
                    maximum_after_position_error_m=max(x['source_fk_error_after_one_step'][0] for x in samples),
                    maximum_after_orientation_error_rad=max(x['source_fk_error_after_one_step'][1] for x in samples),
                    mean_camera_step_position_m=float(np.mean([x['camera_step_position_m'] for x in samples])),
                    minimum_after_all548_distance_m=min(x['all548_distance_after_one_step_m'] for x in samples),
                    original_limits_and_slew_pass=all(x['bounds_pass'] for x in samples),
                    all_predicted_discrete_clearances_pass=all(x['next_discrete_clearance_pass'] for x in samples)))
                print(json.dumps(reports[-1]), flush=True)
    paths = [source/'g1_backend.py', source/'pink_collision.py', source/'g1_29dof.urdf',
             source/'height_policy_io.yaml', folder/'measurements.jsonl',
             folder/'sensor_mounts.json', folder/'fine_collision_geometry.json',
             folder/'pink_collision_initialization.json', Path(__file__).resolve()]
    report = dict(schema='readonly_g1_terminal_pink_damping/v1',
        scope='Independent one-step genuine frozen Pink counterfactuals at measured q/base and unchanged world optical target. No physical rollout or strict-stop acceptance. Historical asynchronous acquisition state is not logged, so this is not packet-exact reproduction.',
        fixed=dict(frame_position_cost=1., frame_orientation_cost=.3, frame_gain=.1,
                   frame_lm_damping=.001, posture_cost=.03, waist_posture_cost=.15,
                   upper_dofs=15, speed_rad_s=.5, issued_acceleration_rad_s2=1.,
                   pink_barrier_pairs=473, audited_full_pairs=548, minimum_distance_m=.0065,
                   maximum_checked_pairs=20, dt_s=dt, original_joint_limits_preserved=True),
        pace=1, physical_validation=False, environment_verified=False,
        evidence_files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        cases=reports)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')


if __name__ == '__main__':
    main()
