"""Private native posture hints from an already admitted camera-path witness.

These hints never replace the exact camera trajectory, command leases, native
joint bounds, or independent collision checks. They contain no new observation.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def witness_posture(report,shot_s):
    if not report or not report.get('physical_valid'):
        return None
    times=np.asarray(report['times_s'],float);states=np.asarray(report['states'],float)
    if shot_s<times[0] or shot_s>times[-1]:return None
    model=report['native_model_document']
    if states.shape!=(len(times),3+len(model['joints'])) or np.any(np.diff(times)<=0):
        raise ValueError('invalid native allocation witness')
    state=np.array([np.interp(shot_s,times,states[:,i]) for i in range(states.shape[1])])
    if not np.isfinite(state).all():raise ValueError('nonfinite native allocation witness')
    base=np.asarray(model['base'],float).copy()
    base[:3,:3]=Rotation.from_euler('z',state[2]).as_matrix()@base[:3,:3]
    base[:2,3]+=state[:2]
    joints={joint['name']:float(joint['q0']+value) for joint,value in zip(model['joints'],state[3:])}
    if any(not joint['lower']<=joints[joint['name']]<=joint['upper'] for joint in model['joints']):
        raise ValueError('native posture witness exceeds original joint bounds')
    return dict(scope='private_selected_kinematic_witness_soft_preference_only',
        camera_path_id=report['camera_path_id'],shot_s=float(shot_s),
        joint_positions=joints,world_from_base=base.tolist(),
        base_heading_rad=float(np.arctan2(base[1,0],base[0,0])))
