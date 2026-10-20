"""Shared measured geometry, exact task sessions and strict stop measurements.

USD is read only during geometry preparation. Runtime transforms must come from
PhysX acquisition-time state, including base height, roll and pitch. Conservative
boxes are representation/guard proxies, not exact contact or distance certificates.
"""
from copy import deepcopy
import json

import numpy as np

from .representation import PARTS, Box, Observation, Pose, number, compress_bounds
from .wbc_protocol import (Capabilities, TimedTask, measured_feedback, negotiate,
                           optical_camera, observation_from_measured_bounds)

CONSTRAINTS = ('configuration', 'actuation', 'self_collision', 'environment', 'policy_domain')
CONSTRAINT_SCOPES = frozenset(('measured_native_limits', 'measured_collision_model',
                             'policy_config_outer_domain', 'fresh_environment_map',
                             'not_available', 'not_applicable'))


def constraint_feedback(evidence, now_s, max_age_s=.1):
    """Public constraint status; stale/absent evidence is explicitly unknown.

    Callers provide model-derived measurements, not optimistic defaults. Passing
    this validator does not turn a collision proxy into an exact certificate.
    Numerical rows, joint identities and individual actuator limits stay private.
    """
    now_s = number(now_s, 'now_s', 0.)
    max_age_s = number(max_age_s, 'max_age_s', 0.)
    evidence = {} if evidence is None else evidence
    if not isinstance(evidence, dict) or set(evidence)-set(CONSTRAINTS):
        raise ValueError('unknown public constraint family')
    result = {}
    for name in CONSTRAINTS:
        raw = evidence.get(name)
        if raw is None:
            result[name] = dict(status='unknown', stamp_s=None, scope='not_available', reason='missing_evidence')
            continue
        if not isinstance(raw, dict) or set(raw)-{'status', 'stamp_s', 'scope', 'reserve_fraction', 'clearance_m'}:
            raise ValueError('unsupported constraint evidence fields')
        status, scope = raw.get('status'), raw.get('scope')
        if status not in ('pass', 'fail', 'unknown', 'not_applicable') or scope not in CONSTRAINT_SCOPES:
            raise ValueError('explicit constraint status and scope required')
        row = deepcopy(raw)
        if 'reserve_fraction' in row and row['reserve_fraction'] is not None:
            if not 0 <= number(row['reserve_fraction'], 'reserve_fraction', 0.) <= 1:
                raise ValueError('reserve must be a fraction or null')
        if 'clearance_m' in row and row['clearance_m'] is not None:
            clearance = number(row['clearance_m'], 'clearance_m')
            if status == 'pass' and clearance < 0:
                raise ValueError('negative clearance cannot be reported as passing')
        stamp = row.get('stamp_s')
        if status == 'not_applicable':
            if scope != 'not_applicable' or stamp is not None:
                raise ValueError('not-applicable constraints require explicit empty provenance')
        elif stamp is None:
            if status != 'unknown':
                raise ValueError('measured constraint status requires a timestamp')
        else:
            stamp = number(stamp, 'constraint stamp_s', 0.)
            if stamp > now_s or now_s-stamp > max_age_s:
                row.update(status='unknown', reason='stale_or_future_evidence')
        result[name] = row
    return result


def rigid(value):
    if isinstance(value, Pose):
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = value.rotation, value.translation_m
        return result
    result = np.asarray(value, dtype=float)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(result[:3, :3].T@result[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(result[:3, :3])-1) > 1e-5):
        raise ValueError('finite rigid physical transform required')
    return result


def pose(value):
    matrix = rigid(value)
    return Pose.from_rt(matrix[:3, :3], matrix[:3, 3])


class MeasuredColliderGeometry:
    def __init__(self, shapes):
        if not shapes:
            raise ValueError('no physical collider geometry')
        self.shapes = deepcopy(list(shapes))
        names = set()
        for shape in self.shapes:
            if not isinstance(shape['name'], str) or shape['name'] in names:
                raise ValueError('unique collider names required')
            names.add(shape['name'])
            if not isinstance(shape['body'], str) or not shape['body'] or shape['part'] not in PARTS:
                raise ValueError('every collider needs a measured body and functional part')
            Box(tuple(shape['center_body']), tuple(shape['half_extents_body']))
        if {shape['part'] for shape in self.shapes} != set(PARTS):
            raise ValueError('both functional parts require physical collider coverage')
        self.body_names = tuple(sorted({shape['body'] for shape in self.shapes}))
        self._indices = np.array([self.body_names.index(s['body']) for s in self.shapes])
        self._centers = np.array([s['center_body'] for s in self.shapes])
        self._half = np.array([s['half_extents_body'] for s in self.shapes])
        self._part_indices = {part: np.array([i for i, s in enumerate(self.shapes) if s['part'] == part]) for part in PARTS}

    @classmethod
    def from_stage(cls, stage, root, manipulator_bodies, body_names=None):
        """Cache actual enabled collision GPrims, including guide-purpose meshes.

        Body-local bounds are in physical metres, with mesh/ancestor scale baked
        in. Runtime USD transforms are never substituted for measured body poses.
        Fixed merged bodies must be resolved explicitly by the backend rather
        than silently omitted from representation.
        """
        from pxr import Usd, UsdGeom, UsdPhysics
        root_prim = stage.GetPrimAtPath(str(root))
        if not root_prim:
            raise ValueError('robot USD root missing')
        manipulator_bodies = set(manipulator_bodies)
        if not manipulator_bodies:
            raise ValueError('explicit camera-manipulator body assignment required')
        scale = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('invalid USD stage units')
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'],
                                 useExtentsHint=False, ignoreVisibility=True)
        xforms = UsdGeom.XformCache()
        shapes, owners = [], {}
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            ancestor, collision = prim, None
            while ancestor and ancestor.GetPath().HasPrefix(root_prim.GetPath()):
                if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                    collision = UsdPhysics.CollisionAPI(ancestor)
                    break
                ancestor = ancestor.GetParent()
            if collision is None or collision.GetCollisionEnabledAttr().Get() is False:
                continue
            owner = prim
            while owner and owner.GetPath().HasPrefix(root_prim.GetPath()):
                if owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    break
                owner = owner.GetParent()
            if not owner or not owner.GetPath().HasPrefix(root_prim.GetPath()) or not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f'collider lacks an explicit rigid-body owner: {prim.GetPath()}')
            body, path = owner.GetName(), str(owner.GetPath())
            if body in owners and owners[body] != path:
                raise ValueError('duplicate rigid-body names cannot identify measured geometry')
            owners[body] = path
            if prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                    raise ValueError(f'invalid physical collision mesh points: {prim.GetPath()}')
            else:
                bound = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
                if bound.IsEmpty():
                    raise ValueError(f'empty collision bounds: {prim.GetPath()}')
                low, high = np.asarray(bound.GetMin()), np.asarray(bound.GetMax())
                points = Box(tuple((low+high)/2), tuple(np.maximum((high-low)/2, 1e-9))).vertices
            prim_world = np.asarray(xforms.GetLocalToWorldTransform(prim)).T
            points_world = (points@prim_world[:3, :3].T+prim_world[:3, 3])*scale
            authored = np.asarray(xforms.GetLocalToWorldTransform(owner)).T.copy()
            authored[:3, 3] *= scale
            # Bake authored scale into shape geometry, not the measured SE(3).
            u, _, vt = np.linalg.svd(authored[:3, :3])
            rotation = u@np.diag([1., 1., np.linalg.det(u@vt)])@vt
            # Align only once in the body's frame. World alignment followed by
            # inverse rotation would artificially inflate angled arm geometry.
            local = (points_world-authored[:3, 3])@rotation
            enclosed = Box.enclosing(local)
            shapes.append(dict(name=str(prim.GetPath()), body=body, part=PARTS[body in manipulator_bodies],
                               center_body=list(enclosed.center_m),
                               half_extents_body=(np.asarray(enclosed.half_extents_m)+1e-6).tolist()))
        result = cls(shapes)
        if body_names is not None and set(result.body_names)-set(body_names):
            raise ValueError('some collider owners have no measured-state source: '+
                             ', '.join(sorted(set(result.body_names)-set(body_names))))
        return result

    def _transforms(self, body_poses):
        missing = set(self.body_names)-set(body_poses)
        if missing:
            raise ValueError('missing acquisition-time physical bodies: '+', '.join(sorted(missing)))
        transforms = np.asarray([rigid(body_poses[n]) if isinstance(body_poses[n], Pose)
                                 else body_poses[n] for n in self.body_names], dtype=float)
        if (transforms.shape != (len(self.body_names), 4, 4) or not np.isfinite(transforms).all()
                or not np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(np.swapaxes(transforms[:, :3, :3], 1, 2)@transforms[:, :3, :3], np.eye(3), atol=1e-5, rtol=0)
                or np.any(np.abs(np.linalg.det(transforms[:, :3, :3])-1) > 1e-5)):
            raise ValueError('finite rigid physical body transforms required')
        return transforms

    def shape_world_bounds(self, body_poses):
        transforms = self._transforms(body_poses)[self._indices]
        centers = np.einsum('nij,nj->ni', transforms[:, :3, :3], self._centers)+transforms[:, :3, 3]
        extents = np.einsum('nij,nj->ni', np.abs(transforms[:, :3, :3]), self._half)
        result = []
        for i, shape in enumerate(self.shapes):
            matrix, center, extent = transforms[i], centers[i], extents[i]
            result.append(dict(name=shape['name'], body=shape['body'], part=shape['part'],
                               low=center-extent, high=center+extent,
                               center_world=center, rotation_world=matrix[:3, :3].copy(),
                               half_extents_body=np.asarray(shape['half_extents_body'])))
        return result

    def world_bounds(self, body_poses):
        result = {part: [] for part in PARTS}
        for shape in self.shape_world_bounds(body_poses):
            result[shape['part']].append((shape['low'], shape['high']))
        return result

    def observe(self, stamp_s, T_world_mount, T_world_camera_optical, body_poses, box_budget=3):
        if isinstance(box_budget, bool) or not isinstance(box_budget, int) or not 1 <= box_budget <= 8:
            raise ValueError('box_budget must be in [1,8]')
        from .wbc_protocol import semantic_camera
        mount = pose(T_world_mount)
        matrix = rigid(T_world_mount)
        transforms = self._transforms(body_poses)[self._indices]
        rotation = np.einsum('ij,njk->nik', matrix[:3, :3].T, transforms[:, :3, :3])
        translation = (transforms[:, :3, 3]-matrix[:3, 3])@matrix[:3, :3]
        centers = np.einsum('nij,nj->ni', rotation, self._centers)+translation
        extents = np.einsum('nij,nj->ni', np.abs(rotation), self._half)
        occupancy = {}
        for part, selected in self._part_indices.items():
            low, high = compress_bounds(centers[selected]-extents[selected], centers[selected]+extents[selected], box_budget)
            occupancy[part] = tuple(Box(tuple((a+b)/2), tuple(np.maximum((b-a)/2, 1e-9))) for a,b in zip(low,high))
        camera = semantic_camera(pose(T_world_camera_optical))
        return Observation(stamp_s, mount, mount.inverse().compose(camera), occupancy, source='measured_fk')

    def manifest(self):
        return dict(schema='measured_collider_geometry/v1', shapes=deepcopy(self.shapes),
                    body_names=list(self.body_names), runtime_transform_source='measured_physx_bodies',
                    geometry_scope='conservative_collision_aabbs_not_exact_contact_certificate')


class LiveTaskSession:
    """Immutable admitted task and exact reference/measurement boundary.

    Actuation, numerical feasibility, contact guards and command leases remain
    the native backend's responsibility. This class never advances physics.
    """
    def __init__(self, capabilities, max_observation_age_s=.1):
        if not isinstance(capabilities, Capabilities):
            raise ValueError('explicit native capabilities required')
        self.capabilities = capabilities
        self.max_observation_age_s = number(max_observation_age_s, 'max_observation_age_s', 0.)
        self._task_json = None
        self._task = None
        self._initial = None
        self._last_feedback_stamp = -1.

    @property
    def task(self):
        return TimedTask.from_dict(json.loads(self._task_json)) if self._task_json else None

    @property
    def start_s(self):
        return self._task.trajectory.start_s if self._task else None

    @property
    def end_s(self):
        return self._task.trajectory.end_s if self._task else None

    @property
    def task_id(self):
        return self._task.task_id if self._task else None

    @property
    def camera_path_id(self):
        return self._task.trajectory.id if self._task else None

    def accept(self, task, observation, now_s):
        candidate = TimedTask.from_dict(task.to_dict())
        observation = Observation.from_dict(observation.to_dict())
        result = negotiate(candidate, observation, self.capabilities, now_s, self.max_observation_age_s)
        if result['supported']:
            self._task_json = json.dumps(candidate.to_dict(), sort_keys=True, allow_nan=False)
            self._task = candidate
            self._initial = observation
            self._last_feedback_stamp = observation.stamp_s
        return result

    def reference(self, stamp_s):
        if self._task is None:
            raise ValueError('no admitted task')
        semantic, twist = self._task.trajectory.sample_with_twist(stamp_s)
        return rigid(optical_camera(semantic)), twist.copy()

    def feedback(self, observation, *, feasible=None, reason='not_evaluated',
                 remaining_adjustment=None, hold_terminal=False, constraints=None, now_s=None):
        if self._task is None:
            raise ValueError('no admitted task')
        if observation.stamp_s <= self._last_feedback_stamp:
            raise ValueError('physical feedback timestamps must strictly increase')
        now_s = observation.stamp_s if now_s is None else number(now_s, 'now_s', 0.)
        if observation.stamp_s > now_s or now_s-observation.stamp_s > self.max_observation_age_s:
            raise ValueError('physical feedback is stale or from a future clock')
        result = measured_feedback(self._task, observation, feasible=feasible, reason=reason,
                                   remaining_adjustment=remaining_adjustment, hold_terminal=hold_terminal)
        result['constraints'] = constraint_feedback(constraints, now_s, self.max_observation_age_s)
        self._last_feedback_stamp = observation.stamp_s
        return result


def rotation_span_rad(rotations):
    rotations = np.asarray(rotations, float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or not len(rotations) or not np.isfinite(rotations).all():
        raise ValueError('finite recorded orientation sequence required')
    cosine = (np.einsum('aij,bij->ab', rotations, rotations)-1.)/2.
    return float(np.arccos(np.clip(cosine, -1., 1.)).max())


def audit_stop(stamps_s, base_transforms, camera_transforms, joints, joint_units,
               final_base_linear_velocity, final_base_angular_velocity, window_s=.5):
    """Strict shared final-half-second test on measured state; no thresholds tuned."""
    window_s = number(window_s, 'window_s', .5)
    stamps = np.asarray(stamps_s, float)
    bases, cameras, joints = np.asarray(base_transforms, float), np.asarray(camera_transforms, float), np.asarray(joints, float)
    units = tuple(joint_units)
    linear, angular = np.asarray(final_base_linear_velocity, float), np.asarray(final_base_angular_velocity, float)
    if (stamps.ndim != 1 or len(stamps) < 2 or not np.isfinite(stamps).all()
            or np.any(np.diff(stamps) <= 0) or stamps[-1]-stamps[0] < window_s-1e-8
            or bases.shape != (len(stamps), 4, 4) or cameras.shape != bases.shape
            or joints.shape != (len(stamps), len(units)) or not np.isfinite(joints).all()
            or any(unit not in ('m', 'rad') for unit in units)
            or linear.shape != (3,) or angular.shape != (3,) or not np.isfinite([linear, angular]).all()):
        raise ValueError('complete finite measured stop window required')
    for matrix in (*bases, *cameras):
        rigid(matrix)
    # Include the sample at/before the boundary; never evaluate a shorter tail.
    first = max(0, int(np.searchsorted(stamps, stamps[-1]-window_s, side='right'))-1)
    b, c, q = bases[first:], cameras[first:], joints[first:]
    b_span, c_span = float(np.linalg.norm(np.ptp(b[:, :3, 3], axis=0))), float(np.linalg.norm(np.ptp(c[:, :3, 3], axis=0)))
    br, cr = rotation_span_rad(b[:, :3, :3]), rotation_span_rad(c[:, :3, :3])
    q_span = np.ptp(q, axis=0)
    tolerance = np.array([.002 if unit == 'm' else .01 for unit in units])
    speed, angular_speed = float(np.linalg.norm(linear)), float(np.linalg.norm(angular))
    return dict(schema='physical_wbc_stop/v1', measured_window_s=float(stamps[-1]-stamps[first]),
                base_position_span_m=b_span, camera_position_span_m=c_span,
                base_rotation_span_rad=br, camera_rotation_span_rad=cr,
                joint_span=q_span.tolist(), joint_tolerance=tolerance.tolist(),
                final_base_linear_speed_mps=speed, final_base_angular_speed_radps=angular_speed,
                passed=bool(b_span < .005 and c_span < .005 and br < .01 and cr < .01
                            and np.all(q_span < tolerance) and speed < .02 and angular_speed < .03))
