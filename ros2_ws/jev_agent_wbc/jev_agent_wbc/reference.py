"""Immutable original path and smooth, absolute-to-original task deformations."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from .contracts import PARAMETERS, digest, validate_task


def smoothstep(s):
    x = np.clip(s, 0., 1.)
    return x*x*x*(10.+x*(-15.+6.*x))


def look_at(position, subject):
    forward = np.asarray(subject)-np.asarray(position)
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        raise ValueError('camera coincides with subject')
    forward /= norm
    left = np.cross([0., 0., 1.], forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0., 1., 0.])
    left /= np.linalg.norm(left)
    return np.column_stack([forward, left, np.cross(forward, left)])


def bound_effect_estimate(evidence, parameter, amount):
    """Geometry at a numerical bound, not a semantic threshold or SAM metric."""
    if not parameter.endswith('_m'):
        return dict(local_rotation_rad=amount,
                    optical_axis_rotation_rad=0. if parameter == 'roll_rad' else abs(amount))
    subject = evidence.get('runtime_subject_position')
    if subject is None:
        return dict(available=False)
    subject = np.asarray(subject, float)
    ratios, angles = [], []
    for sample in evidence.get('reference_samples', []):
        p = np.asarray(sample['position'], float)
        axis = np.array([0., 0., 1.])
        if parameter == 'radial_m':
            axis = p-subject
            axis[2] = 0.
            axis /= max(np.linalg.norm(axis), 1e-9)
        elif parameter == 'lateral_m':
            if 'quaternion_xyzw' not in sample:
                continue
            axis = Rotation.from_quat(sample['quaternion_xyzw']).as_matrix()[:, 1]
        before, after = p-subject, p+amount*axis-subject
        a, b = np.linalg.norm(before), np.linalg.norm(after)
        if min(a, b) < 1e-6:
            return dict(available=False, warning='bound reaches subject position')
        ratios.append(float((a/b)**2))
        angles.append(float(np.arccos(np.clip(before@after/(a*b), -1., 1.))))
    return dict(available=bool(ratios), inverse_square_area_ratio_range=[min(ratios), max(ratios)] if ratios else None,
                max_view_angle_change_rad=max(angles) if angles else None,
                source='3D range/view geometry at bound; not masks or semantic acceptance')


class OriginalReference:
    def __init__(self, task, subject_position):
        if isinstance(task, (str, Path)):
            task = json.loads(Path(task).read_text())
        self._task = validate_task(task)
        self.sha256 = digest(self._task)
        samples = self._task['reference_camera_trajectory']['samples']
        times = np.array([row['t'] for row in samples])
        self.duration = float(times[-1])
        self.translation = CubicSpline(times, np.array([row['position'] for row in samples]),
                                       bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
        self.rotation = RotationSpline(times, Rotation.from_quat([row['quaternion_xyzw'] for row in samples]))
        self.subject = np.array(subject_position, float)
        if self.subject.shape != (3,) or not np.isfinite(self.subject).all():
            raise ValueError('runtime subject position must be finite 3D')
        self.subject.setflags(write=False)
        self.shots = [dict(type=self._allocation_type(p['motion']), start_s=p['start_s'], end_s=p['end_s'])
                      for p in self._task['shot_phase']]
        self.document = {}  # no authored tracking tolerances or changed scoring gates
        self.automatic_transitions = []
        self._sample_cache = {}

    @property
    def task(self):
        return deepcopy(self._task)

    @staticmethod
    def _allocation_type(motion):
        # The inherited MoveOut allocator strongly pulls the base along a planar
        # guide. Lateral/vertical camera translations must retain free arm/base
        # allocation rather than inheriting that longitudinal driving objective.
        return {'orbit': 'orbit', 'pan': 'pan', 'movein': 'moveout', 'moveout': 'moveout'}.get(motion, 'neutral')

    def sample(self, t):
        key = float(t)
        cached = self._sample_cache.get(key)
        if cached is not None:
            return tuple(value.copy() for value in cached)
        s = float(np.clip(t, 0., self.duration))
        r = self.rotation(s).as_matrix()
        twist = np.r_[self.translation(s, 1), r@self.rotation(s, 1)]
        if t < 0. or t >= self.duration:
            twist[:] = 0.
        result = (self.translation(s).copy(), r, twist)
        if len(self._sample_cache) >= 512:
            self._sample_cache.pop(next(iter(self._sample_cache)))
        self._sample_cache[key] = tuple(value.copy() for value in result)
        return result

    def phase(self, t):
        return deepcopy(next((p for p in self._task['shot_phase'] if t < p['end_s']), self._task['shot_phase'][-1]))

    def shot_context(self, t):
        i = next((i for i, p in enumerate(self.shots) if t < p['end_s']), len(self.shots)-1)
        p = self.shots[i]
        return dict(shot_type=p['type'], shot_phase=float(np.clip((t-p['start_s'])/(p['end_s']-p['start_s']), 0., 1.)),
                    next_shot=self.shots[i+1]['type'] if i+1 < len(self.shots) else 'neutral',
                    segment_index=i, start_s=p['start_s'], end_s=p['end_s'])

    def shot_weights(self, t, blend_s=2.):
        return {name: float(name == self.shot_context(t)['shot_type']) for name in ('neutral', 'orbit', 'pan', 'moveout')}

    def tangent(self, t):
        phase = self.phase(t)
        probes = sorted(np.linspace(phase['start_s'], phase['end_s'], 17), key=lambda s: abs(s-t))
        for probe in [t, *probes]:
            velocity = self.sample(probe)[2][:2]
            if np.linalg.norm(velocity) > 1e-5:
                return float(np.arctan2(velocity[1], velocity[0]))
        return None

    def transition(self, t):
        return None  # single-shot evaluations add no time or automatic connector

    def motion_mask(self, times):
        return (np.asarray(times) >= 0.) & (np.asarray(times) <= self.duration)


class DeformedReference:
    """A C2 blend of parameter values; poses are ALWAYS constructed from original.

    Previous plans only supply the blend's starting value/derivatives. They never
    become the new geometric reference, so accumulated drift is impossible.
    """
    def __init__(self, original, envelope, values, previous=None, activation_s=None, return_s=None):
        self.original, self.envelope = original, envelope
        self.values = np.asarray(values, float).copy()
        self.previous = previous
        self.activation_s = envelope.start_s if activation_s is None else float(activation_s)
        self.return_s = return_s
        if return_s is not None and not self.activation_s <= return_s <= envelope.end_s-envelope.blend_s+1e-9:
            raise ValueError('optimizer recovery must fit inside the authorized window')
        if not envelope.start_s <= self.activation_s < envelope.end_s:
            raise ValueError('activation must be inside the Jev authorized window')
        if self.activation_s+envelope.blend_s > envelope.end_s+1e-9:
            raise ValueError('insufficient time for authorized C2 blend')
        if self.values.shape != (len(PARAMETERS),) or not np.isfinite(self.values).all() or not envelope.contains(self.values, settled=True):
            raise ValueError('optimizer values are outside Jev permission')
        for name in ('duration', 'subject', 'shots', 'document', 'automatic_transitions', 'sha256'):
            setattr(self, name, getattr(original, name))

    def parameters(self, t):
        e = self.envelope
        old = np.zeros(len(PARAMETERS)) if self.previous is None else self.previous.parameters(t)
        if t <= self.activation_s:
            return old
        a = smoothstep((t-self.activation_s)/e.blend_s)
        values = old*(1.-a)+self.values*a
        if e.terminal == 'recover' or self.return_s is not None:
            onset = e.end_s-e.blend_s if self.return_s is None else self.return_s
            b = smoothstep((t-onset)/e.blend_s)
            values *= 1.-b
        elif t > e.end_s:
            # Retain is legal only up to the original budget; no indefinite permission.
            return np.zeros(len(PARAMETERS))
        return values

    def pose(self, t):
        p, r, _ = self.original.sample(t)
        values = self.parameters(t)
        radial = p-self.subject
        radial[2] = 0.
        if np.linalg.norm(radial) < 1e-8:
            radial = -r[:, 0].copy()
            radial[2] = 0.
        radial /= max(np.linalg.norm(radial), 1e-8)
        p += values[0]*radial+values[1]*r[:, 1]+np.array([0., 0., values[2]])
        aim = self.aim_weight(t)
        if aim > 0:
            # Preserve the original framing offset from a centered look-at orientation.
            p0, r0, _ = self.original.sample(t)
            correction = look_at(p, self.subject)@look_at(p0, self.subject).T
            r = Rotation.from_rotvec(aim*Rotation.from_matrix(correction).as_rotvec()).as_matrix()@r0
        r = r@Rotation.from_euler('ZYX', values[3:]).as_matrix()
        return p, r

    def aim_weight(self, t):
        old = 0. if self.previous is None else self.previous.aim_weight(t)
        fraction = smoothstep((t-self.activation_s)/self.envelope.blend_s)
        return old*(1.-fraction)+float('subject_aim' in self.envelope.protected)*fraction

    def sample(self, t):
        s = float(np.clip(t, 0., self.duration))
        p, r = self.pose(s)
        h = 1e-4
        a, b = max(0., s-h), min(self.duration, s+h)
        pa, ra = self.pose(a)
        pb, rb = self.pose(b)
        twist = np.r_[(pb-pa)/(b-a), Rotation.from_matrix(rb@ra.T).as_rotvec()/(b-a)]
        if t < 0. or t >= self.duration:
            twist[:] = 0.
        return p, r, twist

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'original'), name)
