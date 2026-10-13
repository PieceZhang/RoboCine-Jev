"""Strict task/envelope contracts. Semantics cannot change physical limits or metrics."""
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import hashlib
import json
import math

import numpy as np


PARAMETERS = ('radial_m', 'lateral_m', 'height_m', 'yaw_rad', 'pitch_rad', 'roll_rad')


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(native(value), sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def validate_task(document):
    """The four upstream fields are the complete task API; robot/scene are runtime inputs."""
    expected = {'reference_camera_trajectory', 'shot_phase', 'subject', 'intent'}
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError(f'task requires exactly {sorted(expected)}')
    data = deepcopy(document)
    path = data['reference_camera_trajectory']
    if not isinstance(path, dict) or set(path) != {'frame', 'samples'} or path['frame'] != 'world_z_up_camera_link':
        raise ValueError('reference frame must be world_z_up_camera_link (x forward, y left, z up)')
    samples = path['samples']
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError('reference requires at least three timestamped poses')
    last = -1.
    for sample in samples:
        if set(sample) != {'t', 'position', 'quaternion_xyzw'}:
            raise ValueError('reference sample requires t, position, quaternion_xyzw')
        t = number(sample['t'], 'sample.t')
        p, q = np.asarray(sample['position'], float), np.asarray(sample['quaternion_xyzw'], float)
        if t <= last or p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
            raise ValueError('reference times must increase and poses must be finite')
        if abs(np.linalg.norm(q)-1.) > 1e-5:
            raise ValueError('reference quaternion must be normalized')
        last = t
    if samples[0]['t'] != 0.:
        raise ValueError('reference must start at t=0')
    phases = data['shot_phase']
    end = 0.
    if not isinstance(phases, list) or not phases:
        raise ValueError('shot_phase must contain contiguous named intervals')
    for phase in phases:
        if set(phase) != {'name', 'motion', 'start_s', 'end_s'}:
            raise ValueError('phase requires name, motion, start_s, end_s')
        a, b = number(phase['start_s'], 'phase.start'), number(phase['end_s'], 'phase.end')
        if not isinstance(phase['name'], str) or not phase['name'] or not isinstance(phase['motion'], str):
            raise ValueError('phase names/motions must be strings')
        if abs(a-end) > 1e-8 or b <= a:
            raise ValueError('phases must be contiguous and positive')
        end = b
    if abs(end-last) > 1e-8:
        raise ValueError('phases must cover the original time budget')
    if not isinstance(data['subject'], dict) or set(data['subject']) != {'id', 'description'}:
        raise ValueError('subject requires only id and description; 3D tracking is runtime data')
    if any(not isinstance(v, str) or not v.strip() for v in data['subject'].values()):
        raise ValueError('subject identity/description must be nonempty')
    if not isinstance(data['intent'], str) or not data['intent'].strip():
        raise ValueError('intent must be nonempty text')
    return data


@dataclass(frozen=True)
class Envelope:
    reference_sha256: str
    decision_id: str
    event: str
    issued_at_s: float
    start_s: float
    end_s: float
    blend_s: float
    bounds: dict
    protected: tuple = ()
    terminal: str = 'retain'
    rationale: str = ''
    evidence_revision: int = 0
    parent_id: str = ''
    settled_bounds: dict = field(default_factory=dict)

    def validate(self, reference_sha256, duration, physical_caps):
        if self.reference_sha256 != reference_sha256:
            raise ValueError('envelope references a different original trajectory')
        for name in ('issued_at_s', 'start_s', 'end_s', 'blend_s'):
            number(getattr(self, name), name)
        if not 0 <= self.issued_at_s <= self.start_s < self.end_s <= duration+1e-8:
            raise ValueError('envelope interval violates the original time budget')
        if not 0 < self.blend_s <= self.end_s-self.start_s:
            raise ValueError('invalid envelope transition duration')
        if self.terminal not in ('retain', 'recover') or (set(self.bounds)|set(self.settled_bounds))-set(PARAMETERS):
            raise ValueError('unknown terminal policy or task parameter')
        for name, interval in [*self.bounds.items(), *self.settled_bounds.items()]:
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                raise ValueError('every bound must be a numeric [lower, upper] interval')
            lo, hi = [number(x, name) for x in interval]
            cap = physical_caps[name]
            if not -cap-1e-8 <= lo <= 0. <= hi <= cap+1e-8:
                raise ValueError(f'{name}: bound exceeds physical search domain or excludes original')
        for name, (lo, hi) in self.settled_bounds.items():
            outer = self.bounds.get(name, [0., 0.])
            if lo < outer[0]-1e-9 or hi > outer[1]+1e-9:
                raise ValueError('settled bounds must be inside the explicitly authorized transition union')
        allowed = {'subject_aim', 'motion_direction', 'horizon', 'fixed_progress'}
        if set(self.protected)-allowed:
            raise ValueError('unknown protected attribute')
        settled = self.settled_bounds or self.bounds
        if 'horizon' in self.protected and any(abs(v) > 1e-12 for v in settled.get('roll_rad', [0., 0.])):
            raise ValueError('horizon protection conflicts with roll permission')
        if 'subject_aim' in self.protected and any(abs(v) > 1e-12 for k in ('yaw_rad', 'pitch_rad') for v in settled.get(k, [0., 0.])):
            raise ValueError('subject aim protection conflicts with independent yaw/pitch permission')
        return self

    def contains(self, parameters, atol=1e-8, settled=False):
        bounds = (self.settled_bounds or self.bounds) if settled else self.bounds
        return all(bounds.get(k, [0., 0.])[0]-atol <= float(v) <=
                   bounds.get(k, [0., 0.])[1]+atol for k, v in zip(PARAMETERS, parameters))

    def to_dict(self):
        return native(asdict(self))


@dataclass
class Validation:
    feasible: bool
    reason: str
    min_clearance_m: float = math.inf
    joint_margin_rad: float = math.inf
    max_velocity_ratio: float = 0.
    max_acceleration_ratio: float = 0.
    max_tracking_error_m: float = 0.
    max_orientation_error_rad: float = 0.
    cost: float = math.inf
    terminal_state: list = field(default_factory=list)
    samples: int = 0
    next_phase_checked: bool = False
    original_loss_integral: float = 0.
    original_objective: float = 0.
    target_tracking_loss_integral: float = 0.
    task_degraded_samples: int = 0
    last_task_degradation_s: float | None = None
    # Recorded from FK of the predicted execution, not from the modified target.
    camera_trace: list = field(default_factory=list)

    def to_dict(self):
        return native(asdict(self))
