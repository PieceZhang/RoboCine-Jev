"""C2 translation / SO(3) interpolation of the existing relative 6D JSON format.

For WBC, base_link means the *captured initial base* axes, fixed in odom.
The reference never moves with the live base. End poses are held indefinitely.
"""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from .jvs_kinematics import load_relative_trajectory, rpy_matrix


class CameraTrajectory:
    def __init__(self, path, initial_p, initial_r, initial_base_r, *, transition_options=None):
        from .trajectory_scene import reanchor_subject_poses, trajectory_spawn
        document = json.loads(Path(path).read_text())
        trajectory_spawn(path, {})  # validate an explicitly supplied scene pose
        self.pieces = None
        self.automatic_transitions = []
        if document.get('format') == 'jvs_relative_shot_sequence/v1':
            from .motion_sequence import compile_sequence
            self.pieces, self.shots, self.automatic_transitions, effective_spawn = compile_sequence(
                document, initial_p, initial_r, initial_base_r, transition_options)
            self.duration = float(self.pieces[-1][1])
            self.piece_ends = np.array([p[1] for p in self.pieces])
        else:
            trajectory = load_relative_trajectory(path)
            times = np.array([s.time_from_start for s in trajectory.samples])
            poses = np.array([s.pose for s in trajectory.samples])
            poses, effective_spawn = reanchor_subject_poses(document, times, poses)
            frame = initial_base_r if trajectory.frame_id == 'base_link' else initial_r
            positions = initial_p + poses[:, :3] @ frame.T
            rotations = [frame @ rpy_matrix(*pose[3:]) @ frame.T @ initial_r for pose in poses]
            self.translation = CubicSpline(times, positions, bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
            self.rotation = RotationSpline(times, Rotation.from_matrix(rotations))
            self.duration = float(times[-1])
            self.shots = document.get('shots', [dict(start_s=0., end_s=self.duration,
                type=document.get('shot_metadata', {}).get('shot', 'neutral'))])
        self.document = document
        self.subject = None
        scene = document.get('scene_context', {})
        target = scene.get('target', {}).get('world_y_up_m')
        spawn = effective_spawn
        if target is not None and spawn:
            from .jvs_kinematics import forward_kinematics
            yaw = float(spawn['yaw'])
            scene_rotation = np.array([[np.sin(yaw), np.cos(yaw), 0.],
                                       [0., 0., 1.], [np.cos(yaw), -np.sin(yaw), 0.]])
            local_p, local_r = forward_kinematics(np.zeros(6))
            world_p = np.array([spawn['x'], spawn['height'], spawn['z']])+scene_rotation@local_p
            self.subject = initial_p+initial_r@(scene_rotation@local_r).T@(np.array(target)-world_p)
        previous_end = 0.
        for shot in self.shots:
            if (shot['type'] not in ('neutral', 'orbit', 'pan', 'moveout')
                    or not np.all(np.isfinite([float(shot['start_s']), float(shot['end_s'])]))
                    or abs(float(shot['start_s'])-previous_end) > 1e-6
                    or not float(shot['end_s']) > previous_end):
                raise ValueError('shots must be contiguous positive intervals of known type')
            previous_end = float(shot['end_s'])
        if abs(previous_end-self.duration) > 1e-6:
            raise ValueError('shots must cover the full trajectory')
        for shot in self.shots:
            a, b = shot.get('motion_start_s', shot['start_s']), shot.get('motion_end_s', shot['end_s'])
            if not shot['start_s'] <= a < b <= shot['end_s']:
                raise ValueError('motion interval must lie inside its shot')

    def transition(self, t):
        """A stationary inter-shot window, including both sides of the junction."""
        for previous, following in zip(self.shots, self.shots[1:]):
            start = previous.get('motion_end_s', previous['end_s'])
            end = following.get('motion_start_s', following['start_s'])
            if start <= t < end and end > start:
                return dict(start_s=start, end_s=end, reference_s=end+.5)
        return None

    def motion_mask(self, times):
        """Accuracy applies only to planned action intervals, not connectors/holds."""
        times = np.asarray(times, float)
        active = np.zeros(times.shape, dtype=bool)
        for shot in self.shots:
            active |= (times >= shot.get('motion_start_s', shot['start_s'])) & (
                times <= shot.get('motion_end_s', shot['end_s']))
        return active

    def shot_context(self, t):
        index = next((i for i, s in enumerate(self.shots) if t < s['end_s']), len(self.shots)-1)
        shot = self.shots[index]
        return dict(shot_type=shot['type'], shot_phase=float(np.clip(
            (t-shot['start_s'])/(shot['end_s']-shot['start_s']), 0., 1.)),
            next_shot=self.shots[index+1]['type'] if index+1 < len(self.shots) else 'neutral',
            segment_index=index, start_s=shot['start_s'], end_s=shot['end_s'])

    def shot_weights(self, t, blend_s=2.):
        # Cubic partition of unity centred at every junction. No discrete change
        # of objective at a segment boundary, including the no-preview ablation.
        names = ('neutral', 'orbit', 'pan', 'moveout')
        weights = np.eye(4)[names.index(self.shots[0]['type'])].copy()
        for previous, shot in zip(self.shots, self.shots[1:]):
            s = np.clip((t-shot['start_s'])/blend_s+.5, 0., 1.)
            alpha = s*s*(3.-2.*s)
            weights += alpha*(np.eye(4)[names.index(shot['type'])]
                              - np.eye(4)[names.index(previous['type'])])
        return dict(zip(names, weights))

    def tangent(self, t):
        context = self.shot_context(t)
        # At a stopped endpoint, use the nearest nonzero tangent in this shot.
        probes = [t] + list(np.linspace(context['start_s'], context['end_s'], 31)[
            np.argsort(np.abs(np.linspace(context['start_s'], context['end_s'], 31)-t))])
        for probe in probes:
            velocity = self.sample(probe)[2][:2]
            if np.linalg.norm(velocity) > 1e-5:
                return float(np.arctan2(velocity[1], velocity[0]))
        return None

    def sample(self, t):
        bounded = float(np.clip(t, 0., self.duration))
        if self.pieces is not None:
            index = min(int(np.searchsorted(self.piece_ends, bounded, side='right')), len(self.pieces)-1)
            start, _, curve = self.pieces[index]
            p, r, twist = curve.sample(bounded-start)
            return p, r, twist if 0. < t < self.duration else np.zeros(6)
        p = self.translation(bounded)
        r = self.rotation(bounded).as_matrix()
        twist = np.zeros(6)
        if 0. < t < self.duration:
            twist[:3] = self.translation(bounded, 1)
            # RotationSpline differentiates in the local rotating frame.
            twist[3:] = r @ self.rotation(bounded, 1)
        return p, r, twist
