"""Rigid placement and height calibration of the authored orbit/movein shots."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .representation import Pose, fingerprint
from .reference import OriginalReference
from .wbc_protocol import CameraTrajectory, TimedTask

PACKAGE = Path(__file__).resolve().parents[1]


def bind_scenario(case, observation, scenario_root=None, *, obstacle_height_offset_m=0.):
    if case not in ('orbit', 'movein'):
        raise ValueError('this suite contains exactly orbit and movein')
    if not np.isfinite(obstacle_height_offset_m) or abs(obstacle_height_offset_m)>.15:
        raise ValueError('obstacle height micro-adjustment must be within 0.15 m')
    root = Path(scenario_root or PACKAGE/'scenarios')
    task = json.loads((root/case/'task.json').read_text())
    scene = json.loads((root/case/'runtime_scene.json').read_text())
    source_task, source_scene = deepcopy(task), deepcopy(scene)
    first = task['reference_camera_trajectory']['samples'][0]
    original_rotation = Rotation.from_quat(first['quaternion_xyzw']).as_matrix()
    actual = observation.world_from_camera
    delta = (np.arctan2(actual.rotation[1, 0], actual.rotation[0, 0])
             - np.arctan2(original_rotation[1, 0], original_rotation[0, 0]))
    rotation = Rotation.from_euler('z', delta).as_matrix()
    offset = np.asarray(actual.translation_m)-rotation@first['position']
    for sample in task['reference_camera_trajectory']['samples']:
        sample['position'] = (rotation@sample['position']+offset).tolist()
        sample['quaternion_xyzw'] = Rotation.from_matrix(
            rotation@Rotation.from_quat(sample['quaternion_xyzw']).as_matrix()).as_quat().tolist()
    scene['subject']['position'] = (rotation@scene['subject']['position']+offset).tolist()
    scene['subject']['orientation_xyzw'] = Rotation.from_matrix(
        rotation@Rotation.from_quat(scene['subject']['orientation_xyzw']).as_matrix()).as_quat().tolist()
    for obj in scene['obstacles']:
        for key in obj['keyframes']:
            key['position'] = (rotation@key['position']+offset).tolist()
            key['position'][2]+=float(obstacle_height_offset_m)
            key['velocity'] = (rotation@key['velocity']).tolist()
    # This is scene placement, not a robot reset or a geometry input to Agent.
    scene.pop('initial_state', None)
    scene.pop('base_z', None)
    provenance = dict(schema='multi_robot_scenario_binding/v1', case=case,
        source_task_sha256=fingerprint(source_task), source_scene_sha256=fingerprint(source_scene),
        initial_observation_id=observation.id, yaw_alignment_rad=float(delta),
        translation_m=offset.tolist(), obstacle_height_adjustment_m=float(offset[2]),
        additional_obstacle_height_offset_m=float(obstacle_height_offset_m),
        duration_s=8., timing_scale=1., obstacle_size_scale=1.,
        placement='common gravity-preserving yaw and translation; explicit obstacle-only height micro-adjustment',
        task_sha256=fingerprint(task), scene_sha256=fingerprint(scene))
    return OriginalReference(task, scene['subject']['position']), scene, provenance


def timed_path(path, observation, start_s, *, relaxed=False, task_id='avoidance'):
    times = np.linspace(0., path.duration, round(path.duration/.04)+1)
    from .intent_paths import poses_at
    positions,rotations=poses_at(path,times)
    poses=[Pose.from_rt(r,p) for p,r in zip(positions,rotations)]
    return TimedTask(task_id, observation.stamp_s, observation.id,
        CameraTrajectory(start_s, tuple(times), tuple(poses)),
        .08 if relaxed else .03, .20 if relaxed else .08)
