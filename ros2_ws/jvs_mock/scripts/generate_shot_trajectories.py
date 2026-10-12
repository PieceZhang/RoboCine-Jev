#!/usr/bin/env python3
"""Ordinary camera-space shots and a scene-aware Orbit -> Pan -> MoveOut sequence."""
import json
import math
from copy import deepcopy
from pathlib import Path
import re
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jvs_mock.jvs_kinematics import forward_kinematics
from retime_dynamic_combos import COMBO_TIMING, retime_sampled


SCENE_RELATIVE_PATH = Path(
    'cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda')
SUBJECT_ID = 'Kid'
DEFAULT_SPAWN = dict(x=5.0, z=1.0, yaw=0.0, height=0.002)
FAST_DYNAMIC_COMBOS = (1, 3, 4)


def _ease(value):
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10.0 - 15.0*value + 6.0*value*value)


def _look_at(position, target):
    """Camera basis in the Y-up scene: +X view, +Y left, +Z up."""
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    scene_up = np.array([0.0, 1.0, 0.0])
    left = np.cross(scene_up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    return np.column_stack((forward, left, up))


def _scene_target(repo_root, stage):
    """Read Kid's planner head target and verify it belongs to this USDA."""
    text = stage.read_text(encoding='utf-8')
    match = re.search(
        rf'def Xform "{re.escape(SUBJECT_ID)}"\s*\{{\s*'
        r'float3 xformOp:translate = \(([^)]+)\)', text)
    if not match:
        raise ValueError(f'{stage}: cannot find cast member {SUBJECT_ID!r}')
    cast_position = np.fromstring(match.group(1), sep=',')

    planned = repo_root/'cinema_traj_out'/'kujiale_0021_kid_combo'/'traj_final.json'
    document = json.loads(planned.read_text(encoding='utf-8'))
    centers = [segment.get('fix', {}).get('center')
               for segment in document.get('segments', [])
               if str(segment.get('target_id', '')).lower() == SUBJECT_ID.lower()
               and segment.get('fix', {}).get('center') is not None]
    if not centers:
        raise ValueError(f'{planned}: no {SUBJECT_ID} look target found')
    target = np.asarray(centers[0], float)
    if not np.allclose(target[[0, 2]], cast_position[[0, 2]], atol=1e-4):
        raise ValueError('planner target and USDA cast position disagree')
    return target


def _initial_camera_world(spawn=None):
    """Default stage-3 spawn and q=0 camera pose, expressed in scene axes."""
    camera_in_base, camera_rotation_in_base = forward_kinematics(np.zeros(6))
    spawn = DEFAULT_SPAWN if spawn is None else spawn
    yaw = spawn['yaw']
    base_rotation = np.array([
        [math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
        [math.cos(yaw), -math.sin(yaw), 0.0],
    ])
    base_position = np.array([
        spawn['x'], spawn['height'], spawn['z']])
    return (base_position + base_rotation @ camera_in_base,
            base_rotation @ camera_rotation_in_base)


def _generate_person_centered_trajectory(repo_root, segments, spawn=None):
    """Generate a continuous target-locked trajectory around Kid."""
    stage = repo_root/SCENE_RELATIVE_PATH
    target = _scene_target(repo_root, stage)
    spawn = dict(DEFAULT_SPAWN if spawn is None else spawn)
    initial_position, initial_rotation = _initial_camera_world(spawn)
    acquired_rotation = _look_at(initial_position, target)
    acquire = Slerp([0.0, 1.0], Rotation.from_matrix(
        np.stack((initial_rotation, acquired_rotation))))

    radial0 = initial_position - target
    planar_radius = float(np.linalg.norm(radial0[[0, 2]]))
    duration_s = sum(duration for _, duration, _ in segments)

    timeline = []
    elapsed = 0.0
    for kind, duration, params in segments:
        timeline.append((kind, elapsed, elapsed+duration, params))
        elapsed += duration

    def pose_at(t):
        # The first two seconds acquire Kid from the documented default spawn.
        if t < 2.0:
            position = initial_position
            rotation = acquire([_ease(t/2.0)]).as_matrix()[0]
        else:
            theta = 0.0
            extra_radius = 0.0
            for kind, start, end, params in timeline:
                fraction = _ease((t-(start+2.0))/(end-start-4.0))
                if kind in ('orbit', 'pan'):
                    theta += math.radians(params['angle_deg']) * fraction
                elif kind == 'moveout':
                    extra_radius += params['distance'] * fraction
                if t < end:
                    break

            c, s = math.cos(theta), math.sin(theta)
            radial = radial0.copy()
            radial[0] = c*radial0[0] + s*radial0[2]
            radial[2] = -s*radial0[0] + c*radial0[2]
            radial[[0, 2]] *= (planar_radius + extra_radius) / planar_radius
            position = target + radial
            rotation = _look_at(position, target)

        relative_position = initial_rotation.T @ (position-initial_position)
        relative_rotation = initial_rotation.T @ rotation
        rpy = Rotation.from_matrix(relative_rotation).as_euler('xyz')
        return [*relative_position.tolist(), *rpy.tolist()]

    sample_count = round(duration_s*20.0)+1
    samples = [dict(time_from_start=float(t), pose=pose_at(float(t)))
               for t in np.linspace(0.0, duration_s, sample_count)]
    # Preserve the exact relative-trajectory origin contract.
    samples[0]['pose'] = [0.0]*6
    target_metadata = dict(subject=SUBJECT_ID, mode='subject_head',
                           world_y_up_m=target.tolist(), target_lock=True)
    shots = []
    current_radius = planar_radius
    for index, (kind, start, end, params) in enumerate(timeline):
        shot = dict(type=kind, start_s=start, end_s=end,
                    motion_start_s=start+2.0, motion_end_s=end-2.0,
                    target=target_metadata)
        if index == 0:
            shot.update(focus_acquire_start_s=0.0, focus_acquire_end_s=2.0)
        if kind in ('orbit', 'pan'):
            shot.update(angle_deg=params['angle_deg'], radius=current_radius)
            if 'radius' in params:
                shot['requested_radius'] = params['radius']
        if kind == 'pan':
            shot['motion_model'] = 'target_locked_arc'
        if kind == 'moveout':
            shot['distance'] = params['distance']
            current_radius += params['distance']
        shots.append(shot)
    return dict(
        format='jvs_relative_task_space_trajectory/v1',
        relative_to='current_camera_link', frame_id='initial_camera_link',
        dof_order=['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        units=dict(translation='m', rotation='rad'),
        sample_rate_hz=20.0, duration_s=duration_s,
        initial_base_pose=dict(frame='scene_y_up', **spawn),
        scene_context=dict(
            stage=str(stage), target=target_metadata,
            assumed_spawn=spawn, assumed_initial_q=[0.0]*6),
        shots=shots, samples=samples)


def motion_only_document(document):
    """Keep identical action samples, without acquisition/holds/connections.

    Local timestamps belong only to each action. The executor, not this file,
    decides the transition duration and builds the execution timeline.
    """
    result = {key: deepcopy(value) for key, value in document.items()
              if key not in ('samples', 'shots', 'duration_s')}
    result.update(format='jvs_relative_shot_sequence/v1',
                  transition_policy='automatic_minimum_jerk', shots=[])
    for shot in document['shots']:
        start = shot.get('motion_start_s', shot['start_s'])
        end = shot.get('motion_end_s', shot['end_s'])
        item = {key: deepcopy(value) for key, value in shot.items()
                if not key.endswith('_s')}
        item['duration_s'] = end-start
        item['samples'] = [dict(time_from_start=round(sample['time_from_start']-start, 10),
                                pose=deepcopy(sample['pose']))
                           for sample in document['samples']
                           if start-1e-9 <= sample['time_from_start'] <= end+1e-9]
        if (not item['samples'] or abs(item['samples'][0]['time_from_start']) > 1e-8
                or abs(item['samples'][-1]['time_from_start']-item['duration_s']) > 1e-8):
            raise ValueError('motion endpoints must be sampled in the full trajectory')
        result['shots'].append(item)
    result['duration_s'] = sum(shot['duration_s'] for shot in result['shots'])
    return result


def generate(combo_mode='full'):
    root = Path(__file__).resolve().parents[1]/'trajectories'
    repo_root = Path(__file__).resolve().parents[3]
    cases = {
        'orbit': [('orbit', 26., dict(angle_deg=25., radius=.8))],
        'pan': [('pan', 18., dict(angle_deg=45.))],
        'pan_limit': [('pan', 18., dict(angle_deg=35.))],
        'moveout': [('moveout', 18., dict(distance=.6))],
        'moveout_forward': [('moveout', 18., dict(distance=.6))],
    }
    initial = {name: [0.]*6 for name in cases}
    manifest = []
    for name, segments in cases.items():
        doc = _generate_person_centered_trajectory(repo_root, segments)
        (root/f'shot_{name}.json').write_text(json.dumps(doc, indent=2)+'\n')
        manifest.append(dict(name=name, trajectory=f'shot_{name}.json', initial_q=initial[name]))
    manifest.extend(generate_validation_combos(mode=combo_mode))
    (root/'shot_stage2_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def generate_validation_combos(output_directory=None, indices=None, *, mode='full'):
    """Generate retained geometry, preserving the current dynamic-demo rhythm.

    The historical pair catalogue records the earlier profile's validation;
    regenerating geometry does not create new execution evidence.
    """
    root = (Path(__file__).resolve().parents[1]/'trajectories' if output_directory is None
            else Path(output_directory))
    root.mkdir(parents=True, exist_ok=True)
    if mode not in ('full', 'actions', 'both'):
        raise ValueError('combo mode must be full, actions or both')
    repo = Path(__file__).resolve().parents[3]
    cases = [
        (dict(DEFAULT_SPAWN),
         [('orbit', 22., dict(angle_deg=20.)), ('pan', 18., dict(angle_deg=-30.)),
          ('moveout', 18., dict(distance=.7))]),
        (dict(x=4.0, z=1.2, yaw=.28, height=.002),
         [('orbit', 22., dict(angle_deg=15.)), ('pan', 22., dict(angle_deg=-20.)),
          ('moveout', 18., dict(distance=.45))]),
        (dict(x=3.5, z=2.0, yaw=.62, height=.002),
         [('pan', 22., dict(angle_deg=-15.)), ('moveout', 18., dict(distance=.4)),
          ('orbit', 22., dict(angle_deg=20.))]),
        (dict(x=5.8, z=.5, yaw=-.35, height=.002),
         [('moveout', 18., dict(distance=.3)), ('orbit', 22., dict(angle_deg=12.)),
          ('pan', 22., dict(angle_deg=-10.))]),
    ]
    manifest = []
    for index, (spawn, segments) in enumerate(cases, 1):
        if indices is not None and index not in indices:
            continue
        document = _generate_person_centered_trajectory(repo, segments, spawn)
        if index == 1:
            document['design_note'] = ('Same default spawn and subject; reduce the bed-side Pan arc '
                                       'while preserving Orbit +20 deg and MoveOut 0.7 m. '
                                       'See combo_optimization_20261001 for validation.')
        if index in FAST_DYNAMIC_COMBOS:
            document = retime_sampled(document, **COMBO_TIMING[f'shot_combo{index}'])
        if index == 4:
            document['dynamic_camera_tracking'] = dict(orientation_weight_scale=100.,
                position_gain=8., position_recovery_speed_m_s=.4,
                joint_reserve_rad=.15, joint_reserve_weight=10000.)
        variants = ([] if mode == 'actions' else [('', document)])
        if mode != 'full':
            variants.append(('_actions', motion_only_document(document)))
        for suffix, variant in variants:
            name = f'combo{index}{suffix}'
            (root/f'shot_{name}.json').write_text(json.dumps(variant, indent=2)+'\n')
            manifest.append(dict(name=name, trajectory=f'shot_{name}.json', initial_q=[0.]*6))
    return manifest


def generate_reachable_combo(mode='full'):
    """Compatibility entry point: the reachable default-spawn shot is combo1."""
    return generate_validation_combos(indices=(1,), mode=mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-combos-only', action='store_true')
    parser.add_argument('--reachable-combo-only', action='store_true',
                        help='Generate shot_combo1 only (legacy flag; no unnumbered file is created)')
    parser.add_argument('--combo-mode', choices=('full', 'actions', 'both'), default='full',
                        help='Keep full format, generate action-only sequences, or generate both')
    options = parser.parse_args()
    if options.reachable_combo_only:
        generate_reachable_combo(options.combo_mode)
    elif options.validation_combos_only:
        generate_validation_combos(mode=options.combo_mode)
    else:
        generate(options.combo_mode)
