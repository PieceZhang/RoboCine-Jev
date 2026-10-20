"""Timed camera tasks and honest capability negotiation for physical backends.

Reuses jev_geometry/v1; joints and solver internals remain behind this boundary.
This module validates messages, not collisions, balance or trajectory feasibility.
All timestamps use the simulator's continuously advancing clock, in seconds.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .representation import (PARTS, Box, Observation, Pose, Request, fields,
                             fingerprint, number)

VERSION = 'jev_wbc/v1'
AXES = 'right_handed_camera_x_forward_y_left_z_up'
# p_optical = R_optical_semantic @ p_semantic. Both frames have the same origin.
# Physical cameras: +X right, +Y down, +Z forward. Representation: +X forward.
OPTICAL_FROM_SEMANTIC = Pose.from_rt(
    np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]), np.zeros(3))
FEATURES = frozenset((
    'timed_camera_se3', 'platform_pose', 'manipulator_pose',
    'platform_avoid_region', 'platform_stay_in_region',
    'manipulator_avoid_region', 'manipulator_stay_in_region',
))
REASONS = frozenset((
    'not_evaluated', 'tracking_limit', 'self_collision', 'environment_collision',
    'contact_or_balance', 'out_of_policy_domain', 'solver_infeasible', 'solver_failure',
    'observation_stale', 'unsupported_request', 'configuration_limit', 'actuation_limit',
    'camera_view_blocked', 'sensor_evidence_unknown', 'command_stale', 'deadline_exceeded',
    'accepted',
))


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f'{name}: nonempty bounded string required')


def semantic_camera(world_from_optical: Pose) -> Pose:
    return world_from_optical.compose(OPTICAL_FROM_SEMANTIC)


def optical_camera(world_from_semantic: Pose) -> Pose:
    return world_from_semantic.compose(OPTICAL_FROM_SEMANTIC.inverse())


@dataclass(frozen=True)
class CameraTrajectory:
    """Immutable world poses at absolute simulation times; no extrapolation.

    Translation is piecewise linear and orientation follows the shortest SO(3)
    arc. Discontinuous sample velocities are NOT dynamically certified here.
    The numerical backend must check them against its own physical limits.
    """
    start_s: float
    times_s: tuple
    poses: tuple
    _id: str = field(init=False, repr=False)
    _interpolator: object = field(init=False, repr=False, compare=False)
    _times_array: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        number(self.start_s, 'start_s', 0.)
        times = tuple(number(t, 'sample.t_s', 0.) for t in self.times_s)
        if (len(times) < 2 or len(times) > 100000 or times[0] != 0.
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError('at least two increasing times starting at zero required')
        if len(self.poses) != len(times) or any(not isinstance(p, Pose) for p in self.poses):
            raise ValueError('one validated pose per timestamp required')
        if not np.isfinite(self.start_s + times[-1]):
            raise ValueError('trajectory end must be finite')
        object.__setattr__(self, 'times_s', times)
        object.__setattr__(self, 'poses', tuple(self.poses))
        object.__setattr__(self, '_id', fingerprint(self.to_dict()))
        time_array = np.asarray(times)
        time_array.setflags(write=False)
        object.__setattr__(self, '_times_array', time_array)
        object.__setattr__(self, '_interpolator', Slerp(times,
            Rotation.from_quat([p.quaternion_xyzw for p in self.poses])))

    @property
    def end_s(self):
        return self.start_s + self.times_s[-1]

    @property
    def id(self):
        return self._id

    def to_dict(self):
        return dict(schema=VERSION, kind='camera_trajectory', frame='world', axes=AXES,
                    start_s=float(self.start_s), interpolation='linear_translation_shortest_so3',
                    samples=[dict(t_s=t, pose=p.to_dict()) for t, p in zip(self.times_s, self.poses)])

    @classmethod
    def from_dict(cls, value):
        fields(value, ('schema', 'kind', 'frame', 'axes', 'start_s', 'interpolation', 'samples'))
        if any(value[k] != v for k, v in dict(schema=VERSION, kind='camera_trajectory',
                frame='world', axes=AXES, interpolation='linear_translation_shortest_so3').items()):
            raise ValueError('unsupported camera trajectory convention')
        if not isinstance(value['samples'], list):
            raise ValueError('samples must be a list')
        for item in value['samples']:
            fields(item, ('t_s', 'pose'))
        return cls(value['start_s'], tuple(x['t_s'] for x in value['samples']),
                   tuple(Pose.from_dict(x['pose']) for x in value['samples']))

    def sample(self, stamp_s):
        stamp_s = number(stamp_s, 'stamp_s', self.start_s)
        if stamp_s > self.end_s:
            raise ValueError('trajectory expired; extrapolation is forbidden')
        # Absolute time subtraction can round just beyond a valid final knot.
        t = float(np.clip(stamp_s - self.start_s, 0., self.times_s[-1]))
        i = int(np.clip(np.searchsorted(self._times_array, t, side='right')-1, 0, len(self.poses)-2))
        a, b = self.times_s[i:i+2]
        fraction = (t-a)/(b-a)
        p = (1-fraction)*np.asarray(self.poses[i].translation_m) + fraction*np.asarray(self.poses[i+1].translation_m)
        rotation = self._interpolator(t)
        return Pose(tuple(p), tuple(rotation.as_quat()))

    def sample_with_twist(self, stamp_s):
        """Pose and [world linear, world angular] feedforward, per second.

        At an interior knot the right-hand segment is used; at the final knot
        the left-hand segment is used. Native limits must check knot changes.
        """
        pose = self.sample(stamp_s)
        t = float(np.clip(stamp_s-self.start_s, 0., self.times_s[-1]))
        i = int(np.clip(np.searchsorted(self._times_array, t, side='right')-1, 0, len(self.poses)-2))
        dt = self.times_s[i+1]-self.times_s[i]
        a, b = self.poses[i:i+2]
        linear = (np.asarray(b.translation_m)-a.translation_m)/dt
        angular = Rotation.from_matrix(b.rotation@a.rotation.T).as_rotvec()/dt
        return pose, np.r_[linear, angular]

    def optical_samples(self):
        """Explicit conversion for existing native runners; relative t_s retained.

        This export does not transfer the absolute clock or authorize execution.
        A live adapter must preserve start_s and freshness from TimedTask.
        """
        return [dict(t_s=t, position_m=list(p.translation_m), quaternion_xyzw=list(p.quaternion_xyzw))
                for t, source in zip(self.times_s, self.poses) for p in [optical_camera(source)]]


@dataclass(frozen=True)
class TimedTask:
    task_id: str
    issued_at_s: float
    observation_id: str
    trajectory: CameraTrajectory
    position_tolerance_m: float
    orientation_tolerance_rad: float
    geometry_request: dict | None = None

    def __post_init__(self):
        _identifier(self.task_id, 'task_id')
        _identifier(self.observation_id, 'observation_id')
        number(self.issued_at_s, 'issued_at_s', 0.)
        number(self.position_tolerance_m, 'position_tolerance_m', 0.)
        number(self.orientation_tolerance_rad, 'orientation_tolerance_rad', 0.)
        if not isinstance(self.trajectory, CameraTrajectory):
            raise ValueError('validated CameraTrajectory required')
        if self.issued_at_s > self.trajectory.end_s:
            raise ValueError('task issued after its trajectory expired')
        if self.geometry_request is not None:
            request = Request.from_dict(self.geometry_request)
            if request.observation_id != self.observation_id or request.camera_path_id != self.trajectory.id:
                raise ValueError('geometry request binding mismatch')
            if request.start_s < self.trajectory.start_s or request.end_s > self.trajectory.end_s:
                raise ValueError('geometry interval exceeds camera trajectory')
            # Store canonical bytes; nested caller mutation cannot alter an accepted task.
            object.__setattr__(self, 'geometry_request', request.to_dict())

    def to_dict(self):
        return dict(schema=VERSION, kind='task', task_id=self.task_id,
                    issued_at_s=self.issued_at_s, observation_id=self.observation_id,
                    camera_trajectory=self.trajectory.to_dict(),
                    position_tolerance_m=self.position_tolerance_m,
                    orientation_tolerance_rad=self.orientation_tolerance_rad,
                    geometry_request=deepcopy(self.geometry_request),
                    camera_modification='requires_new_agent_authorized_task')

    @classmethod
    def from_dict(cls, value):
        fields(value, ('schema', 'kind', 'task_id', 'issued_at_s', 'observation_id',
                       'camera_trajectory', 'position_tolerance_m', 'orientation_tolerance_rad',
                       'geometry_request', 'camera_modification'))
        if (value['schema'] != VERSION or value['kind'] != 'task'
                or value['camera_modification'] != 'requires_new_agent_authorized_task'):
            raise ValueError('unsupported task convention')
        return cls(value['task_id'], value['issued_at_s'], value['observation_id'],
                   CameraTrajectory.from_dict(value['camera_trajectory']),
                   value['position_tolerance_m'], value['orientation_tolerance_rad'], value['geometry_request'])

    @property
    def required_features(self):
        result = {'timed_camera_se3'}
        if self.geometry_request is not None:
            request = Request.from_dict(self.geometry_request)
            for p in request.pose_preferences:
                result.add(('platform' if p['part'] == PARTS[0] else 'manipulator')+'_pose')
            for p in request.occupancy_preferences:
                result.add(('platform' if p['part'] == PARTS[0] else 'manipulator')+'_'+p['relation'])
        return frozenset(result)


@dataclass(frozen=True)
class Capabilities:
    backend: str
    runtime_ready: bool
    features: frozenset = frozenset()

    def __post_init__(self):
        _identifier(self.backend, 'backend')
        if not isinstance(self.runtime_ready, bool):
            raise ValueError('runtime_ready must be boolean')
        if isinstance(self.features, str) or set(self.features)-FEATURES:
            raise ValueError('unknown capability feature')
        object.__setattr__(self, 'features', frozenset(self.features))

    def to_dict(self):
        return dict(backend=self.backend, runtime_ready=self.runtime_ready, features=sorted(self.features))


def negotiate(task, observation, capabilities, now_s, max_observation_age_s=.25):
    """Support is different from numerical feasibility; this never reports feasible.

    Call at ingress before submitting to the native solver. Snapshot geometry
    stays bound to the original observation. Execution needs current measured
    feedback and the backend's independent physical guards as well.
    """
    task = TimedTask.from_dict(task.to_dict())
    observation = Observation.from_dict(observation.to_dict())
    now_s = number(now_s, 'now_s', 0.)
    max_observation_age_s = number(max_observation_age_s, 'max_observation_age_s', 0.)
    reasons = []
    if task.observation_id != observation.id:
        reasons.append('observation_binding_mismatch')
    if observation.source != 'measured_fk':
        reasons.append('current_physical_measurement_required')
    if observation.stamp_s > now_s or now_s-observation.stamp_s > max_observation_age_s:
        reasons.append('observation_stale_or_future')
    if task.issued_at_s > now_s or task.issued_at_s < observation.stamp_s:
        reasons.append('task_clock_mismatch')
    if now_s > task.trajectory.end_s:
        reasons.append('trajectory_expired')
    if not capabilities.runtime_ready:
        reasons.append('backend_unavailable')
    missing = sorted(task.required_features-capabilities.features)
    if missing:
        reasons.append('unsupported_features')
    return dict(supported=not reasons, feasible=None, reasons=reasons,
                unsupported_features=missing, task_id=task.task_id,
                camera_path_id=task.trajectory.id, backend=capabilities.backend,
                camera_reference_changed=False)


def measured_feedback(task, observation, *, feasible=None, reason='not_evaluated',
                      remaining_adjustment=None):
    """Report measurement-time camera error, geometry and native-solver status.

    feasible=None means unknown. Error within tolerance does not imply collision
    safety, future feasibility or policy training coverage.
    """
    if not isinstance(task, TimedTask):
        raise ValueError('validated TimedTask required')
    observation = Observation.from_dict(observation.to_dict())
    if observation.source != 'measured_fk':
        raise ValueError('feedback requires measured physical geometry')
    if feasible is not None and not isinstance(feasible, bool):
        raise ValueError('feasible must be boolean or null')
    if reason not in REASONS:
        raise ValueError('use a public failure code; numerical details stay in the backend journal')
    target = task.trajectory.sample(observation.stamp_s)
    actual = observation.world_from_camera
    position_error = float(np.linalg.norm(np.asarray(actual.translation_m)-target.translation_m))
    orientation_error = float(Rotation.from_matrix(target.rotation@actual.rotation.T).magnitude())
    if remaining_adjustment is None:
        remaining_adjustment = {part: None for part in PARTS}
    fields(remaining_adjustment, PARTS)
    for value in remaining_adjustment.values():
        if value is not None and not 0. <= number(value, 'remaining_adjustment', 0.) <= 1.:
            raise ValueError('remaining adjustment must be a fraction in [0,1] or null')
    return dict(schema=VERSION, kind='feedback', task_id=task.task_id,
                camera_path_id=task.trajectory.id, stamp_s=observation.stamp_s,
                feasible=feasible, reason=reason, evidence_kind='measured',
                actual_camera=actual.to_dict(), position_error_m=position_error,
                orientation_error_rad=orientation_error,
                camera_within_tolerance=bool(position_error <= task.position_tolerance_m
                    and orientation_error <= task.orientation_tolerance_rad),
                geometry=observation.summary(), remaining_adjustment=deepcopy(remaining_adjustment),
                camera_reference_changed=False, safety_certificate=False)


def observation_from_measured_bounds(stamp_s, world_from_mount, world_from_optical,
                                     world_bounds, *, box_budget=3):
    """Conservative Representation from acquisition-time physical collider AABBs.

    The backend must assign every relevant robot collider to one of the two
    functional parts. This function checks containment, not that assignment's
    completeness. Do not substitute visual bounds or predicted joint state.
    """
    fields(world_bounds, PARTS)
    occupancy = {}
    mount_from_world = world_from_mount.inverse()
    for part in PARTS:
        boxes = []
        for low, high in world_bounds[part]:
            low, high = np.asarray(low, float), np.asarray(high, float)
            if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
                raise ValueError('finite nonempty collider bounds required')
            box = Box(tuple((low+high)/2), tuple((high-low)/2))
            boxes.append(Box.enclosing(mount_from_world.apply(box.vertices)))
        occupancy[part] = tuple(boxes)
    return Observation.from_fk(stamp_s, world_from_mount, semantic_camera(world_from_optical),
                               occupancy, box_budget=box_budget, source='measured_fk')
