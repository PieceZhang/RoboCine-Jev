#!/usr/bin/env python3
"""Render combo4 from unfiltered per-step physics poses and control annotations.

The shared renderer supplies the four-camera layout. Pose interpolation uses
the original PhysX measurements; no image or camera-pose stabilization is
applied during replay. Controller telemetry supplies phase/error annotations.
"""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import record_dynamic_dual_view as renderer


def physical_replay(run, folder):
    with (folder/'camera_physics.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    rows = [row for row in rows if np.isfinite(float(row['scene_time_s']))]
    time = np.array([float(row['scene_time_s']) for row in rows])
    if len(rows) < 20 or np.any(np.diff(time) <= 0):
        raise ValueError('strictly increasing physical camera measurements required')
    column = lambda names: np.array([[float(row[name]) for name in names] for row in rows])
    position = column(['base_'+axis for axis in 'xyz'])
    quaternion = column(['base_'+axis for axis in ['qx', 'qy', 'qz', 'qw']])
    base_position = run.initial_rotation.inv().apply(position-run.origin)
    base_rotation = run.initial_rotation.inv()*Rotation.from_quat(quaternion)
    arm = column(['position_joint'+str(i) for i in range(1, 7)])
    wheels = column(['position_left_wheel_joint', 'position_right_wheel_joint'])
    camera = column(['camera_'+axis for axis in 'xyz'])
    if not all(np.all(np.isfinite(values)) for values in (position, quaternion, arm, wheels, camera)):
        raise ValueError('physical replay requires finite measured poses and joint positions')
    if time[0] > 0. or time[-1] < run.duration:
        raise ValueError('physical measurements must cover the full recording duration')
    run.control_elapsed = run.elapsed.copy()
    run.elapsed = time
    run.states = np.column_stack([base_position, arm, wheels])
    run.slerp = Slerp(time, base_rotation)
    run.camera_positions = run.initial_rotation.inv().apply(camera-run.origin)
    run.physics_file = folder/'camera_physics.csv'
    run.physics_sha256 = renderer.digest(run.physics_file)
    return run


def complete_record(record, run, *, render_script_sha256=None):
    """Save provenance before SimulationApp.close can terminate the process."""
    if renderer.digest(run.physics_file) != run.physics_sha256:
        raise RuntimeError('physical measurements changed during recording')
    record.update(pose_source='unfiltered every-step PhysX base and joint measurements',
        pose_source_file=str(run.physics_file.resolve()), pose_source_sha256=run.physics_sha256,
        physics_recording_script_sha256=render_script_sha256 or renderer.digest(__file__),
        recording_metadata_script_sha256=renderer.digest(__file__),
        camera_stability_verified=True, image_stabilization=False,
        source_acceptance_mode='camera_stability_verified; other strict audits reported separately',
        annotation_source='last control-cycle telemetry sample at or before the physical timestamp',
        interpolation='linear measured physical positions/joints; SLERP measured physical base rotation')
    record['source_sha256']['camera_physics.csv'] = run.physics_sha256
    record['source_sha256']['camera_stability_physical.json'] = renderer.digest(
        run.physics_file.parent/'camera_stability_physical.json')
    return record


def main():
    if '--demo' not in sys.argv or sys.argv[sys.argv.index('--demo')+1] != 'two_movers':
        raise ValueError('select --demo two_movers for this combo4 recording entry point')
    original_load, original_annotation = renderer.load_run, renderer.frame_annotation
    original_render = renderer.render_one
    used = []

    def load(folder):
        audit = json.loads((folder/'camera_stability_physical.json').read_text())
        if not audit['camera_stability_pass']:
            raise ValueError('physical camera stability audit must pass before recording')
        if any(renderer.digest(folder/name) != expected
               for name, expected in audit['source_sha256'].items()):
            raise ValueError('physical measurements differ from the passing stability audit')
        run = physical_replay(original_load(folder), folder)
        used.append(run)
        return run

    def annotation(run, t):
        control = SimpleNamespace(elapsed=run.control_elapsed, rows=run.rows, reference=run.reference)
        return original_annotation(control, t)

    def render(args, app, name, urdf):
        record = complete_record(original_render(args, app, name, urdf), used[-1])
        (args.output/(name+'_recording.json')).write_text(
            json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        return record

    renderer.load_run, renderer.frame_annotation = load, annotation
    renderer.render_one = render
    renderer.main()


if __name__=='__main__':
    main()
