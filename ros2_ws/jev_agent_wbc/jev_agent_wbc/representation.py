"""Robot-independent, JSON-native geometry and task contracts.

Only two functional parts, three semantic frames, metres/seconds and XYZW
quaternions cross this boundary. No robot model or solver is imported here.
Boxes are conservative *proxies*, never a collision certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from hashlib import sha256
import json
from typing import Protocol
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 'jev_geometry/v1'
PARTS = ('mobile_platform', 'camera_manipulator')
FRAMES = ('world', 'mount_snapshot', 'camera_snapshot', 'mount', 'camera')
SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def vector(value, n, name):
    if (isinstance(value,(list,tuple)) and any(isinstance(v,(bool,np.bool_)) for v in value)
            or np.asarray(value).dtype.kind not in 'iuf'):
        raise ValueError(f'{name}: expected numeric values, not strings or booleans')
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected {n} finite numbers')
    return a


def number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name}: expected a finite number')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name}: must be >= {minimum}')
    return float(value)


def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value)-set(required)-set(optional) or set(required)-set(value):
        raise ValueError(f'expected fields {required}; optional {optional}')


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Pose:
    """T_parent_child: p_parent = R @ p_child + translation_m."""
    translation_m: tuple
    quaternion_xyzw: tuple = (0., 0., 0., 1.)

    def __post_init__(self):
        p = vector(self.translation_m, 3, 'translation_m')
        q = vector(self.quaternion_xyzw, 4, 'quaternion_xyzw')
        if abs(np.linalg.norm(q)-1.) > 1e-6:
            raise ValueError('quaternion must be unit length')
        # q and -q have the same serialization, including rotations by pi.
        pivot = next((v for v in q[::-1] if abs(v) > 1e-12), 1.)
        q = q * (1 if pivot > 0 else -1)
        q[q == 0.] = 0.
        object.__setattr__(self, 'translation_m', tuple(float(v) for v in p))
        object.__setattr__(self, 'quaternion_xyzw', tuple(float(v) for v in q))

    @property
    def rotation(self):
        return Rotation.from_quat(self.quaternion_xyzw).as_matrix()

    def apply(self, points):
        return np.asarray(points) @ self.rotation.T + self.translation_m

    def inverse(self):
        r = self.rotation.T
        return Pose.from_rt(r, -r @ self.translation_m)

    def compose(self, other):
        return Pose.from_rt(self.rotation @ other.rotation, self.apply(other.translation_m))

    @classmethod
    def from_rt(cls, rotation, translation):
        r = np.asarray(rotation, float)
        if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1.):
            raise ValueError('expected a proper rotation matrix')
        return cls(tuple(translation), tuple(Rotation.from_matrix(r).as_quat()))

    @classmethod
    def from_dict(cls, value):
        fields(value, ('translation_m', 'quaternion_xyzw'))
        return cls(**value)

    def to_dict(self):
        return dict(translation_m=list(self.translation_m), quaternion_xyzw=list(self.quaternion_xyzw))


@dataclass(frozen=True)
class Box:
    """Axis-aligned closed box in its owner's explicitly named frame."""
    center_m: tuple
    half_extents_m: tuple

    def __post_init__(self):
        c = vector(self.center_m, 3, 'center_m')
        h = vector(self.half_extents_m, 3, 'half_extents_m')
        if np.any(h <= 0):
            raise ValueError('box half extents must be positive')
        object.__setattr__(self, 'center_m', tuple(float(v) for v in c))
        object.__setattr__(self, 'half_extents_m', tuple(float(v) for v in h))

    @property
    def vertices(self):
        return np.asarray(self.center_m) + SIGNS * self.half_extents_m

    def distance(self, points):
        d = np.abs(np.asarray(points)-self.center_m)-self.half_extents_m
        return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)

    @classmethod
    def enclosing(cls, points):
        p = np.asarray(points, float)
        low, high = p.min(axis=0), p.max(axis=0)
        return cls(tuple((low+high)/2), tuple(np.maximum((high-low)/2, 1e-9)))

    @classmethod
    def from_dict(cls, value):
        fields(value, ('center_m', 'half_extents_m'))
        return cls(**value)

    def to_dict(self):
        return dict(center_m=list(self.center_m), half_extents_m=list(self.half_extents_m))


def compress_boxes(boxes, budget):
    """Greedy spatial merging, independent of topology; preserves containment.

    Components have no stable identity and must never be interpreted as links.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 8:
        raise ValueError('box budget must be an integer in [1, 8]')
    boxes = list(boxes)
    if not boxes:
        return ()
    low = np.array([np.array(b.center_m)-b.half_extents_m for b in boxes])
    high = np.array([np.array(b.center_m)+b.half_extents_m for b in boxes])
    low,high=compress_bounds(low,high,budget)
    boxes = [Box(tuple((a+b)/2),tuple((b-a)/2)) for a,b in zip(low,high)]
    return tuple(sorted(boxes,key=lambda b:b.center_m))


@lru_cache(maxsize=32)
def _merge_pairs(count):
    return np.triu_indices(count,1)


def compress_bounds(low,high,budget):
    """Internal array version of the identical conservative greedy merging."""
    while len(low) > budget:
        i,j = _merge_pairs(len(low))
        merged_low,merged_high = np.minimum(low[i],low[j]),np.maximum(high[i],high[j])
        volumes = np.prod(high-low,axis=1)
        growth = np.prod(merged_high-merged_low,axis=1)-volumes[i]-volumes[j]
        best = int(np.argmin(growth))
        keep = np.ones(len(low),bool);keep[[i[best],j[best]]] = False
        low = np.vstack([low[keep],merged_low[best]])
        high = np.vstack([high[keep],merged_high[best]])
    centers=(low+high)/2
    order=np.lexsort((centers[:,2],centers[:,1],centers[:,0]))
    return low[order],high[order]


@dataclass(frozen=True)
class Observation:
    stamp_s: float
    world_from_mount: Pose
    mount_from_camera: Pose
    occupancy: dict
    source: str = 'measured_fk'

    def __post_init__(self):
        number(self.stamp_s, 'stamp_s', 0.)
        if set(self.occupancy) != set(PARTS):
            raise ValueError('exactly the two functional parts are required')
        if self.source not in ('measured_fk', 'predicted_fk', 'synthetic_fk'):
            raise ValueError('invalid observation source')
        for boxes in self.occupancy.values():
            # An empty manipulator is valid for a rigidly mounted camera.
            if len(boxes) > 8 or any(not isinstance(b, Box) for b in boxes):
                raise ValueError('occupancy must contain at most eight boxes per part')
        if not self.occupancy['mobile_platform']:
            raise ValueError('mobile_platform must have nonempty occupancy')

    @property
    def world_from_camera(self):
        return self.world_from_mount.compose(self.mount_from_camera)

    def to_dict(self):
        return dict(schema=VERSION, kind='observation', stamp_s=float(self.stamp_s), source=self.source,
            units='m_s_rad', axes='right_handed_camera_x_forward_y_left_z_up',
            world_from_mount=self.world_from_mount.to_dict(), mount_from_camera=self.mount_from_camera.to_dict(),
            occupancy_frame='mount', occupancy={k: [b.to_dict() for b in self.occupancy[k]] for k in PARTS},
            geometry_scope='conservative_proxy_of_backend_collision_model_not_safety_certificate')

    @property
    def id(self):
        return fingerprint(self.to_dict())

    @classmethod
    def from_fk(cls, stamp_s, world_from_mount, world_from_camera, occupancy_mount,
                *, box_budget=3, source='measured_fk'):
        """Common ingress for any backend's FK and conservative part geometry."""
        return cls(stamp_s,world_from_mount,world_from_mount.inverse().compose(world_from_camera),
                   {part:compress_boxes(boxes,box_budget) for part,boxes in occupancy_mount.items()},source)

    @classmethod
    def from_dict(cls, value):
        fields(value, ('schema','kind','stamp_s','source','units','axes','world_from_mount',
                       'mount_from_camera','occupancy_frame','occupancy','geometry_scope'))
        if (value['schema'] != VERSION or value['kind'] != 'observation' or value['occupancy_frame'] != 'mount'
                or value['units'] != 'm_s_rad' or value['axes'] != 'right_handed_camera_x_forward_y_left_z_up'
                or value['geometry_scope'] != 'conservative_proxy_of_backend_collision_model_not_safety_certificate'):
            raise ValueError('unsupported observation convention')
        return cls(value['stamp_s'], Pose.from_dict(value['world_from_mount']),
                   Pose.from_dict(value['mount_from_camera']),
                   {k: tuple(Box.from_dict(b) for b in v) for k,v in value['occupancy'].items()}, value['source'])

    def summary(self, obstacles=None):
        """Compact text-ready geometry; full numeric evidence stays in the journal."""
        parts = {}
        for name, boxes in self.occupancy.items():
            entry = dict(components=len(boxes), bounds=None, regions=[b.to_dict() for b in boxes])
            if boxes:
                entry['bounds'] = Box.enclosing(np.vstack([b.vertices for b in boxes])).to_dict()
            parts[name] = entry
        result = dict(schema=VERSION, observation_id=self.id, stamp_s=self.stamp_s, source=self.source,
            frame='mount_snapshot', world_from_mount=self.world_from_mount.to_dict(),
            mount_from_camera=self.mount_from_camera.to_dict(), parts=parts,
            geometry_scope='conservative_proxy_not_safety_certificate')
        if obstacles is not None:
            result['obstacles'] = obstacles
        return result


@dataclass(frozen=True)
class Request:
    """Exact camera path reference plus subordinate geometric preferences.

    Snapshot frames are frozen at observation_id; mount/camera follow current FK.
    All preferences apply over [start_s, end_s]; none relax backend hard limits.
    """
    observation_id: str
    camera_path_id: str
    start_s: float
    end_s: float
    pose_preferences: tuple = ()
    occupancy_preferences: tuple = ()

    def __post_init__(self):
        for value in (self.observation_id, self.camera_path_id):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError('nonempty bounded reference IDs required')
        number(self.start_s, 'start_s', 0.)
        number(self.end_s, 'end_s', self.start_s)
        if self.end_s <= self.start_s:
            raise ValueError('preference interval must have positive duration')
        if len(self.pose_preferences) > 4 or len(self.occupancy_preferences) > 4:
            raise ValueError('at most four preferences of each type')
        for p in self.pose_preferences:
            fields(p, ('part','frame','target','position_tolerance_m','orientation_tolerance_rad','weight'))
            self._common(p)
            Pose.from_dict(p['target'])
            if np.any(vector(p['position_tolerance_m'], 3, 'position_tolerance_m') < 0):
                raise ValueError('negative tolerance')
            number(p['orientation_tolerance_rad'], 'orientation_tolerance_rad', 0.)
        for p in self.occupancy_preferences:
            fields(p, ('part','frame','relation','region','margin_m','weight'))
            self._common(p)
            if p['relation'] not in ('avoid_region', 'stay_in_region'):
                raise ValueError('unsupported occupancy relation')
            Box.from_dict(p['region'])
            number(p['margin_m'], 'margin_m', 0.)
            if p['relation'] == 'stay_in_region' and min(p['region']['half_extents_m']) <= p['margin_m']:
                raise ValueError('margin empties preferred region')

    @staticmethod
    def _common(p):
        if p['part'] not in PARTS or p['frame'] not in FRAMES:
            raise ValueError('unknown functional part or snapshot frame')
        weight = number(p['weight'], 'weight', 0.)
        if weight > 10.:
            raise ValueError('preference weight must be <= 10')

    def to_dict(self):
        return dict(schema=VERSION, kind='request', observation_id=self.observation_id,
            camera_path_id=self.camera_path_id, start_s=self.start_s, end_s=self.end_s,
            pose_preferences=deepcopy(list(self.pose_preferences)), occupancy_preferences=deepcopy(list(self.occupancy_preferences)),
            semantics='camera_primary_geometry_soft_backend_safety_independent')

    @property
    def id(self):
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        fields(value, ('schema','kind','observation_id','camera_path_id','start_s','end_s',
                       'pose_preferences','occupancy_preferences','semantics'))
        if value['schema'] != VERSION or value['kind'] != 'request' or value['semantics'] != 'camera_primary_geometry_soft_backend_safety_independent':
            raise ValueError('unsupported request convention')
        return cls(value['observation_id'], value['camera_path_id'], value['start_s'], value['end_s'],
                   tuple(value['pose_preferences']), tuple(value['occupancy_preferences']))


class GeometryBackend(Protocol):
    """Backend owns state type, FK, collision model and all motion solving."""
    def observe(self, state, stamp_s: float, **kwargs) -> Observation: ...
    def compile(self, request: Request, observation: Observation, camera_path): ...


@dataclass(frozen=True)
class TaskContext:
    """Task timing is distinct from geometric occupancy and backend feasibility."""
    reference_id: str
    stamp_s: float
    deadline_s: float
    earliest_change_s: float
    first_predicted_failure_s: object = None
    recovery_required: bool = False

    def __post_init__(self):
        if not isinstance(self.reference_id,str) or not 0<len(self.reference_id)<=256:
            raise ValueError('bounded immutable reference ID required')
        number(self.stamp_s,'stamp_s',0.)
        number(self.deadline_s,'deadline_s',0.)
        number(self.earliest_change_s,'earliest_change_s',self.stamp_s)
        if self.first_predicted_failure_s is not None:
            number(self.first_predicted_failure_s,'first_predicted_failure_s',0.)
        if not isinstance(self.recovery_required,bool):
            raise ValueError('recovery_required must be boolean')

    def to_dict(self):
        return dict(schema=VERSION,kind='task_context',reference_id=self.reference_id,
            stamp_s=self.stamp_s,deadline_s=self.deadline_s,earliest_change_s=self.earliest_change_s,
            first_predicted_failure_s=self.first_predicted_failure_s,recovery_required=self.recovery_required,
            unchanged_prefix_at_risk=(None if self.first_predicted_failure_s is None else
                                     self.first_predicted_failure_s<self.earliest_change_s),
            target='complete_original_visible_intent_and_terminal_frame',
            past_coverage='immutable_measured_history')

    @classmethod
    def from_dict(cls,value):
        names=('reference_id','stamp_s','deadline_s','earliest_change_s',
               'first_predicted_failure_s','recovery_required')
        fields(value,('schema','kind',*names,'unchanged_prefix_at_risk','target','past_coverage'))
        result=cls(**{k:value[k] for k in names})
        if result.to_dict()!=value:
            raise ValueError('inconsistent task context')
        return result


class CompiledGuidance:
    """Task-space residuals usable by IK/WBC; contains no actuator prescription."""
    def __init__(self, request, observation, camera_path):
        # Revalidate even objects whose nested payload was mutated after creation.
        self.request = Request.from_dict(request.to_dict())
        self.observation = Observation.from_dict(observation.to_dict())
        if request.observation_id != observation.id:
            raise ValueError('observation binding mismatch')
        self.camera_path = camera_path
        self.frames = dict(world=Pose((0.,0.,0.)), mount_snapshot=observation.world_from_mount,
                           camera_snapshot=observation.world_from_camera)

    def residual(self, current):
        values = []
        frames = dict(self.frames, mount=current.world_from_mount, camera=current.world_from_camera)
        for p in self.request.pose_preferences:
            target = frames[p['frame']].compose(Pose.from_dict(p['target']))
            actual = current.world_from_mount if p['part'] == 'mobile_platform' else current.world_from_camera
            delta = target.rotation.T @ (np.array(actual.translation_m)-target.translation_m)
            excess = np.sign(delta)*np.maximum(np.abs(delta)-p['position_tolerance_m'], 0.)
            angle = Rotation.from_matrix(target.rotation.T@actual.rotation).as_rotvec()
            length = np.linalg.norm(angle)
            angle *= max(0., length-p['orientation_tolerance_rad'])/max(length, 1e-12)
            values.extend(np.sqrt(p['weight'])*np.r_[excess, .35*angle])
        for p in self.request.occupancy_preferences:
            region = Box.from_dict(p['region'])
            relative = frames[p['frame']].inverse().compose(current.world_from_mount)
            boxes = current.occupancy[p['part']]
            if not boxes:
                values.append(0.)
                continue
            # AABB of each transformed proxy remains conservative in region frame.
            occupied = [Box.enclosing(relative.apply(b.vertices)) for b in boxes]
            if p['relation'] == 'stay_in_region':
                vertices = np.vstack([b.vertices for b in occupied])
                violation = np.maximum(np.max(np.abs(vertices-region.center_m), axis=0)
                                       - region.half_extents_m + p['margin_m'], 0.)
            else:
                # Signed box separation; penetration is a soft overlap heuristic.
                separations = [np.abs(np.array(b.center_m)-region.center_m)
                               - np.array(b.half_extents_m)-region.half_extents_m for b in occupied]
                distances = [np.linalg.norm(np.maximum(d,0))+min(max(d),0) for d in separations]
                violation = [max(0.,p['margin_m']-min(distances))]
            values.extend(np.sqrt(p['weight'])*np.asarray(violation))
        return np.asarray(values, float)

    def residual_arrays(self,mount,camera,occupancy):
        """Same residual on backend FK arrays, after contract compilation.

        The public ingress remains validated; internal finite differences do
        not construct and validate the same Pose/Box contract 16 times.
        occupancy contains (centers, half_extents) in the mount frame.
        """
        values=[]
        frames={k:(v.rotation,np.asarray(v.translation_m)) for k,v in self.frames.items()}
        frames.update(mount=mount,camera=camera)
        for p in self.request.pose_preferences:
            fr,fp=frames[p['frame']]
            target=p['target'];tr=fr@Rotation.from_quat(target['quaternion_xyzw']).as_matrix()
            tp=fp+fr@np.asarray(target['translation_m'])
            ar,ap=mount if p['part']=='mobile_platform' else camera
            delta=tr.T@(ap-tp)
            excess=np.sign(delta)*np.maximum(np.abs(delta)-p['position_tolerance_m'],0.)
            angle=Rotation.from_matrix(tr.T@ar).as_rotvec();length=np.linalg.norm(angle)
            angle*=max(0.,length-p['orientation_tolerance_rad'])/max(length,1e-12)
            values.extend(np.sqrt(p['weight'])*np.r_[excess,.35*angle])
        for p in self.request.occupancy_preferences:
            centers,half=occupancy[p['part']]
            if not len(centers):values.append(0.);continue
            fr,fp=frames[p['frame']];mr,mp=mount
            relative=fr.T@mr;translation=fr.T@(mp-fp)
            occupied=centers@relative.T+translation
            extent=half@np.abs(relative).T
            region=p['region'];delta=np.abs(occupied-region['center_m'])
            if p['relation']=='stay_in_region':
                violation=np.maximum(np.max(delta+extent,axis=0)-region['half_extents_m']+p['margin_m'],0.)
            else:
                d=delta-extent-region['half_extents_m']
                distance=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(np.max(d,axis=1),0)
                violation=[max(0.,p['margin_m']-min(distance))]
            values.extend(np.sqrt(p['weight'])*np.asarray(violation))
        return np.asarray(values,float)

    def feedback(self, current, kind='predicted', backend_status='unknown'):
        if kind not in ('predicted','measured') or backend_status not in ('pass','fail','unknown'):
            raise ValueError('invalid result provenance/status')
        mount,camera=current.world_from_mount,current.world_from_camera
        occupancy={part:(np.array([b.center_m for b in boxes]).reshape(-1,3),
                         np.array([b.half_extents_m for b in boxes]).reshape(-1,3)) for part,boxes in current.occupancy.items()}
        residual=self.residual_arrays((mount.rotation,np.asarray(mount.translation_m)),
            (camera.rotation,np.asarray(camera.translation_m)),occupancy)
        return dict(schema=VERSION, kind='result', evidence_kind=kind, request_id=self.request.id, camera_path_id=self.request.camera_path_id,
            observation_id=current.id, stamp_s=current.stamp_s, backend_status=backend_status,
            preference_residual_norm=float(np.linalg.norm(residual)), geometry=current.summary(),
            safety_certificate=False)
