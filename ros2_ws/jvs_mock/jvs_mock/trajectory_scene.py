"""Shared, validated Y-up scene spawn contract for the two KC launches."""
import json
import math
from pathlib import Path


def trajectory_spawn(path, fallback):
    """Explicit initial_base_pose overrides launch defaults; old JSON is unchanged."""
    if not path:
        return dict(fallback)
    document = json.loads(Path(path).expanduser().read_text())
    pose = document.get('initial_base_pose')
    if pose is None:
        return dict(fallback)
    if pose.get('frame') != 'scene_y_up':
        raise ValueError('initial_base_pose.frame must be scene_y_up')
    result = {key: float(pose[key]) for key in ('x', 'z', 'yaw', 'height')}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('initial_base_pose must be finite')
    return result


def reanchor_subject_poses(document, times, poses, *, acquire_focus=True):
    """Changing JSON spawn retains Kid, signed orbit angles and MoveOut distances.

    Only generated, target-locked scene shots have the necessary world anchor.
    Ordinary relative trajectories retain their existing contract.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from .jvs_kinematics import forward_kinematics

    scene = document.get('scene_context', {})
    old = scene.get('assumed_spawn')
    new = document.get('initial_base_pose')
    if not old or not new or all(abs(float(old[k])-float(new[k])) < 1e-9
                                 for k in ('x', 'z', 'yaw', 'height')):
        return poses, old
    if (document.get('frame_id') != 'initial_camera_link'
            or not scene.get('target', {}).get('target_lock')):
        raise ValueError('changing the spawn of this relative path requires regenerating its samples')

    def camera(spawn):
        yaw = float(spawn['yaw'])
        basis = np.array([[np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.],
                          [np.cos(yaw), -np.sin(yaw), 0.]])
        p, r = forward_kinematics(np.zeros(6))
        return np.array([spawn['x'], spawn['height'], spawn['z']])+basis@p, basis@r

    target = np.array(scene['target']['world_y_up_m'])
    old_p, old_r = camera(old)
    new_p, new_r = camera(new)
    world = old_p+poses[:, :3]@old_r.T
    radial = world[:, [0, 2]]-target[[0, 2]]
    old_radius = np.linalg.norm(old_p[[0, 2]]-target[[0, 2]])
    new_radius = np.linalg.norm(new_p[[0, 2]]-target[[0, 2]])
    angles = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
    angles += np.arctan2(new_p[2]-target[2], new_p[0]-target[0])-angles[0]
    radii = new_radius+np.linalg.norm(radial, axis=1)-old_radius
    if np.any(radii <= .1):
        raise ValueError('new initial position places the camera on or across the subject')
    world[:, 0] = target[0]+radii*np.cos(angles)
    world[:, 2] = target[2]+radii*np.sin(angles)
    world[:, 1] += new_p[1]-old_p[1]

    def look_at(position):
        forward = target-position
        forward /= np.linalg.norm(forward)
        left = np.cross([0., 1., 0.], forward)
        left /= np.linalg.norm(left)
        return np.column_stack((forward, left, np.cross(forward, left)))

    acquire_end = float(document.get('shots', [{}])[0].get('focus_acquire_end_s', 2.))
    acquire = Slerp([0., 1.], Rotation.from_matrix([new_r, look_at(new_p)]))
    rotations = []
    for t, position in zip(times, world):
        fraction = float(np.clip(t/max(acquire_end, 1e-6), 0., 1.))
        fraction = fraction**3*(10.-15.*fraction+6.*fraction**2)
        rotation = acquire([fraction]).as_matrix()[0] if acquire_focus and t < acquire_end else look_at(position)
        rotations.append(Rotation.from_matrix(new_r.T@rotation).as_euler('xyz'))
    result = np.c_[(world-new_p)@new_r, rotations]
    if acquire_focus:
        result[0] = 0.
    return result, new
