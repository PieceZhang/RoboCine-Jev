"""Compile motion-only shot samples into a C2 execution reference at runtime.

The file contains no transition samples or transition timestamps. Connections
use the minimum-jerk SE(3) endpoint curve; their durations respect conservative
speed/acceleration bounds and leave the common WBC preview time for preparation.
This is a reference, not a collision certificate: WBC still checks every command.
"""
from copy import deepcopy

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

from .jvs_kinematics import rpy_matrix
from .trajectory_scene import reanchor_subject_poses


FORMAT = 'jvs_relative_shot_sequence/v1'


def quintic(s):
    s = np.clip(s, 0., 1.)
    return s**3*(10.-15.*s+6.*s*s)


class PoseSpline:
    """Interpolating quintic position/quaternion splines, clamped C2 endpoints."""
    def __init__(self, times, positions, rotations):
        def spline(values):
            zero = np.zeros(values.shape[1])
            return make_interp_spline(times, values, k=5,
                bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]))
        self.position = spline(np.asarray(positions))
        quaternions = Rotation.from_matrix(rotations).as_quat()
        for i in range(1, len(quaternions)):
            if quaternions[i]@quaternions[i-1] < 0.:
                quaternions[i] *= -1.
        self.quaternion = spline(quaternions)
        self.duration = float(times[-1])

    def sample(self, t):
        t = float(np.clip(t, 0., self.duration))
        raw, derivative = self.quaternion(t), self.quaternion(t, 1)
        norm = np.linalg.norm(raw)
        if norm < .1:
            raise ValueError('degenerate quaternion interpolation in motion samples')
        q = raw/norm
        dq = (derivative-q*(q@derivative))/norm
        omega = 2.*(q[3]*dq[:3]-dq[3]*q[:3]+np.cross(q[:3], dq[:3]))
        return self.position(t), Rotation.from_quat(q).as_matrix(), np.r_[self.position(t, 1), omega]


class MinimumJerkBridge:
    def __init__(self, first, last, duration):
        self.p0, self.r0 = first[:2]
        self.delta = last[0]-self.p0
        self.rotvec = Rotation.from_matrix(last[1]@self.r0.T).as_rotvec()
        self.duration = float(duration)

    def sample(self, t):
        s = float(np.clip(t/self.duration, 0., 1.))
        rate = 30.*s*s*(1.-s)**2/self.duration
        alpha = quintic(s)
        r = Rotation.from_rotvec(alpha*self.rotvec).as_matrix()@self.r0
        return self.p0+alpha*self.delta, r, np.r_[rate*self.delta, rate*self.rotvec]


def bridge_duration(first, last, turn=0., *, minimum=4., speed=.2,
                    acceleration=.25, angular_speed=.6, angular_acceleration=1.):
    # max |s'| = 15/8; max |s''| = 10*sqrt(3)/3 for quintic smoothstep.
    distance = float(np.linalg.norm(last[0]-first[0]))
    angle = max(float(np.linalg.norm(Rotation.from_matrix(last[1]@first[1].T).as_rotvec())), abs(turn))
    return max(minimum, 1.875*distance/speed, np.sqrt(5.773502692*distance/acceleration),
               1.875*angle/angular_speed, np.sqrt(5.773502692*angle/angular_acceleration))


def compile_sequence(document, initial_p, initial_r, initial_base_r, options=None):
    """Validate independent motion intervals and generate an execution timeline."""
    options = dict(options or {})
    defaults = dict(minimum=4., speed=.2, acceleration=.25, angular_speed=.6, angular_acceleration=1.)
    defaults.update(options)
    if any(not np.isfinite(v) or v <= 0. for v in defaults.values()):
        raise ValueError('automatic transition limits must be positive and finite')
    if (document.get('relative_to') != 'current_camera_link'
            or document.get('frame_id') not in ('base_link', 'initial_camera_link')
            or document.get('dof_order') != ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
            or document.get('units') != dict(translation='m', rotation='rad')):
        raise ValueError('invalid relative shot-sequence frame or units')
    rate = float(document.get('sample_rate_hz', 0.))
    if not np.isfinite(rate) or rate <= 0.:
        raise ValueError('sample_rate_hz must be positive and finite')
    shots = document.get('shots')
    if not isinstance(shots, list) or not shots:
        raise ValueError('motion-only sequence must contain shots')
    if 'samples' in document or any('motion_start_s' in s or 'start_s' in s for s in shots):
        raise ValueError('motion-only files cannot include transition samples or execution timestamps')
    times_list, poses_list = [], []
    for shot in shots:
        if shot.get('type') not in ('orbit', 'pan', 'moveout', 'neutral'):
            raise ValueError('unknown motion-only shot type')
        samples = shot.get('samples', [])
        if not isinstance(samples, list) or len(samples) < 2:
            raise ValueError('each motion must contain at least two samples')
        times = np.array([float(s['time_from_start']) for s in samples])
        poses = np.array([s['pose'] for s in samples], float)
        duration = float(shot.get('duration_s', 0.))
        if (not np.all(np.isfinite(times)) or not np.all(np.isfinite(poses))
                or poses.shape != (len(times), 6) or not np.isfinite(duration) or duration <= 0.
                or abs(times[0]) > 1e-9 or not np.all(np.diff(times) > 0.)
                or abs(times[-1]-duration) > 1e-6):
            raise ValueError('invalid motion samples, duration or local timestamps')
        times_list.append(times)
        poses_list.append(poses)
    action_duration = sum(t[-1] for t in times_list)
    if not np.isfinite(float(document.get('duration_s', action_duration))) or abs(
            float(document.get('duration_s', action_duration))-action_duration) > 1e-6:
        raise ValueError('motion-only duration_s must equal total motion duration')
    counts = [len(p) for p in poses_list]
    poses, spawn = reanchor_subject_poses(document, np.concatenate(times_list),
        np.concatenate(poses_list), acquire_focus=False)
    poses_list = np.split(poses, np.cumsum(counts)[:-1])
    frame = initial_base_r if document['frame_id'] == 'base_link' else initial_r
    motions = []
    for times, poses in zip(times_list, poses_list):
        positions = initial_p+poses[:, :3]@frame.T
        rotations = [frame@rpy_matrix(*p[3:])@frame.T@initial_r for p in poses]
        motions.append(PoseSpline(times, positions, rotations))
    first_pose = (np.asarray(initial_p), np.asarray(initial_r), np.zeros(6))
    acquire = bridge_duration(first_pose, motions[0].sample(0.), **dict(defaults, minimum=2.))
    pieces = [(0., acquire, MinimumJerkBridge(first_pose, motions[0].sample(0.), acquire))]
    transitions, execution_shots = [], []
    cursor, previous_end = acquire, 0.
    for index, (shot, motion) in enumerate(zip(shots, motions)):
        start, end = cursor, cursor+motion.duration
        pieces.append((start, end, motion))
        final = index == len(motions)-1
        duration = 2.
        if not final:
            following = motions[index+1]
            a = motion.sample(max(0., motion.duration-.5))[2][:2]
            b = following.sample(min(.5, following.duration))[2][:2]
            turn = float(abs(np.arcsin(np.sin(np.arctan2(b[1], b[0])-np.arctan2(a[1], a[0])))))
            if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-8:
                turn = 0.
            duration = bridge_duration(motion.sample(motion.duration), following.sample(0.), turn, **defaults)
            pieces.append((end, end+duration, MinimumJerkBridge(
                motion.sample(motion.duration), following.sample(0.), duration)))
            transitions.append(dict(start_s=end, end_s=end+duration, duration_s=duration,
                                    method='minimum_jerk_se3', tangent_turn_rad=turn))
        else:
            pieces.append((end, end+duration, MinimumJerkBridge(
                motion.sample(motion.duration), motion.sample(motion.duration), duration)))
        item = {k: deepcopy(v) for k, v in shot.items() if k not in ('samples', 'duration_s')}
        item.update(start_s=previous_end, end_s=end+duration if final else end+duration/2,
                    motion_start_s=start, motion_end_s=end)
        if index == 0:
            item.update(focus_acquire_start_s=0., focus_acquire_end_s=acquire)
        execution_shots.append(item)
        previous_end = item['end_s']
        cursor = end+duration
    return pieces, execution_shots, transitions, spawn
