"""Camera-only completion specifications and geometric visibility measurements.

Reference time indexes intent, never a tracking objective. Scene timestamps remain
physical time. The calibrated cylinder and sphere scene are explicit CPU fixture
models; their measurements are not SAM masks or a continuous safety certificate.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .contracts import native


@dataclass(frozen=True)
class IntentPolicy:
    version: str = 'camera_intent/v1'
    collision_margin_m: float = .025
    visibility_floor: float = .85
    effective_coverage_min: float = .95
    endpoint_fraction: float = .035
    reversal_fraction_max: float = .025
    max_occlusion_s: float = .25
    coverage_bins: int = 32
    minimum_bin_dwell_s: float = .025
    max_duration_factor: float = 1.75
    max_speed_m_s: float = 1.5
    max_acceleration_m_s2: float = 4.
    max_angular_speed_rad_s: float = 3.


POLICY = IntentPolicy()


@lru_cache(maxsize=64)
def _cylinder_points(radius, height, position, orientation):
    angles = np.linspace(0., 2*np.pi, 32, endpoint=False)
    ring = np.column_stack([radius*np.cos(angles), radius*np.sin(angles), np.zeros(32)])
    local = np.vstack([ring-[0,0,height/2], ring+[0,0,height/2]])
    r = Rotation.from_quat(orientation).as_matrix()
    points = local@r.T+np.asarray(position)
    points.setflags(write=False)
    return points


def cylinder_points(model):
    return _cylinder_points(float(model['radius']), float(model['height']), tuple(model['position']),
                            tuple(model.get('orientation_xyzw', [0., 0., 0., 1.])))


def project(points, position, rotation, intrinsics):
    camera = (np.asarray(points)-position)@rotation
    fx, fy, cx, cy = intrinsics
    depth = camera[..., 0]
    uv = np.stack([cx-fx*camera[..., 1]/np.maximum(depth, 1e-8),
                   cy-fy*camera[..., 2]/np.maximum(depth, 1e-8)], axis=-1)
    return uv, depth


def projected_area(position, rotation, model, intrinsics):
    if model is None:
        return None
    uv, depth = project(cylinder_points(model), position, rotation, intrinsics)
    if np.any(depth <= 1e-5):
        return None
    return float(ConvexHull(uv).volume)  # Unclipped, unoccluded silhouette area.


def obstacle_state(scene, t):
    centers = np.asarray(scene.get('centers', []), float).reshape(-1, 3)
    velocities = np.asarray(scene.get('velocities', np.zeros_like(centers)), float)
    age = t-float(scene.get('observation_stamp', t))
    radii = np.asarray(scene.get('radii', []), float)
    radii = radii+max(0., age)*np.asarray(scene.get('radius_rates', np.zeros(len(radii))))
    return centers+age*velocities, radii


def subject_clearances(positions, model, camera_radius):
    """Signed distance to the calibrated solid cylinder, minus camera radius."""
    positions = np.asarray(positions)
    if model is None:
        return np.full(len(positions), np.inf)
    rotation = Rotation.from_quat(model.get('orientation_xyzw', [0.,0.,0.,1.])).as_matrix()
    local = (positions-np.asarray(model['position']))@rotation
    q = np.column_stack([np.linalg.norm(local[:,:2],axis=1)-model['radius'],
                          np.abs(local[:,2])-model['height']/2])
    return np.minimum(np.max(q,axis=1),0.)+np.linalg.norm(np.maximum(q,0.),axis=1)-camera_radius


def visibility(position, rotation, scene, t, model, intrinsics):
    """Area-weighted multi-ray silhouette grid, with analytic cylinder intersection.

    Rays cover the entire projected bounding box (13 x 17), then intersect the
    *front* cylinder surface. Background pixels and self-hidden points don't
    dilute occlusion. Frame containment is separate from external visibility.
    """
    if model is None or not scene.get('valid'):
        return dict(visibility=None, in_frame=None, projected_area=None,
                    source='unknown_subject_geometry_or_obstacles')
    points = cylinder_points(model)
    uv, depth = project(points, position, rotation, intrinsics)
    if np.any(depth <= 1e-5):
        return dict(visibility=0., in_frame=0., projected_area=None, source='behind_camera')
    area = float(ConvexHull(uv).volume)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    x, y = np.meshgrid(lo[0]+(np.arange(13)+.5)/13*(hi[0]-lo[0]),
                       lo[1]+(np.arange(17)+.5)/17*(hi[1]-lo[1]))
    pixels = np.column_stack([x.ravel(), y.ravel()])
    fx, fy, cx, cy = intrinsics
    rays = np.column_stack([np.ones(len(pixels)), (cx-pixels[:, 0])/fx, (cy-pixels[:, 1])/fy])@rotation.T
    target_rotation = Rotation.from_quat(model.get('orientation_xyzw', [0., 0., 0., 1.])).as_matrix()
    origin = (position-np.asarray(model['position']))@target_rotation
    local = rays@target_rotation
    a = (local[:, :2]**2).sum(axis=1)
    b = 2*(local[:, :2]*origin[:2]).sum(axis=1)
    c = (origin[:2]**2).sum()-model['radius']**2
    discriminant = b*b-4*a*c
    hit = np.full(len(rays), np.inf)
    for sign in (-1., 1.):
        root = (-b+sign*np.sqrt(np.maximum(0., discriminant)))/np.maximum(2*a, 1e-12)
        z = origin[2]+root*local[:, 2]
        valid = (discriminant >= 0)&(root > 0)&(np.abs(z) <= model['height']/2)
        hit = np.minimum(hit, np.where(valid, root, np.inf))
    for z in (-model['height']/2, model['height']/2):
        root = (z-origin[2])/np.where(np.abs(local[:, 2]) > 1e-12, local[:, 2], 1e-12)
        xy = origin[:2]+root[:, None]*local[:, :2]
        valid = (root > 0)&((xy**2).sum(axis=1) <= model['radius']**2)
        hit = np.minimum(hit, np.where(valid, root, np.inf))
    valid = np.isfinite(hit)
    if not valid.any():
        return dict(visibility=None, in_frame=None, projected_area=area, source='no_surface_rays')
    targets = position+rays[valid]*hit[valid, None]
    delta = targets-position
    blocked = np.zeros(len(targets), bool)
    centers, radii = obstacle_state(scene, t)
    for center, radius in zip(centers, radii):
        u = np.clip((delta@(center-position))/np.maximum((delta*delta).sum(axis=1), 1e-12), 0., 1.)
        blocked |= np.linalg.norm(position+u[:, None]*delta-center, axis=1) < radius
    pixels = pixels[valid]
    contained = np.all((pixels >= 0.)&(pixels <= 1.), axis=1)
    return dict(visibility=float(np.mean(~blocked)), in_frame=float(np.mean(contained)),
                projected_area=area, surface_rays=int(valid.sum()),
                source='calibrated_cylinder_silhouette_grid_sphere_occlusion')


def default_intrinsics():
    fy = .5/np.tan(np.deg2rad(27.5))
    return [fy*288/384, fy, .5, .5]


class IntentSpec:
    def __init__(self, original, model=None, intrinsics=None, policy=POLICY):
        self.original, self.model, self.policy = original, model, policy
        self.intrinsics = default_intrinsics() if intrinsics is None else intrinsics
        if model is not None:
            if model.get('id', original.task['subject']['id']) != original.task['subject']['id']:
                raise ValueError('calibrated subject identity does not match task')
            if not np.allclose(model['position'], original.subject, atol=1e-8, rtol=0.):
                raise ValueError('this intent model requires a stationary calibrated subject')
            if not all(np.isfinite(model[k]) and model[k] > 0 for k in ('radius','height')):
                raise ValueError('invalid calibrated subject dimensions')
        self.segments = []
        # Establish/develop/resolve landmarks belong to one monotone movement;
        # distinct motion types remain separate, ordered completion obligations.
        for phase in original.task['shot_phase']:
            motion = phase['motion'].lower()
            if self.segments and self.segments[-1]['motion'] == motion:
                self.segments[-1]['end_s'] = phase['end_s']
                self.segments[-1]['landmarks'].append(dict(phase))
            else:
                self.segments.append(dict(motion=motion, start_s=phase['start_s'],
                    end_s=phase['end_s'], landmarks=[dict(phase)]))
        for segment in self.segments:
            p, r, _ = original.sample(segment['start_s'])
            segment['axis'] = r[:, 1].tolist() if 'truck' in segment['motion'] else [0., 0., 1.]
            segment['angle_anchor'] = float(np.arctan2(*(p-original.subject)[[1, 0]]))
            times = np.linspace(segment['start_s'], segment['end_s'], 257)
            poses = [original.sample(float(t)) for t in times]
            values = self.features(segment, [p[0] for p in poses], [p[1] for p in poses])
            segment['start_value'], segment['end_value'] = float(values[0]), float(values[-1])
            segment['reference_range'] = [float(np.min(values)), float(np.max(values))]
            # Reject a non-monotone specification rather than silently replacing
            # a compound movement by its endpoint difference.
            d = values[-1]-values[0]
            segment['supported'] = bool(np.all(np.isfinite(values)) and abs(d) > 1e-7
                and np.sum(np.maximum(0., -np.diff(values)*np.sign(d))) <= .01*abs(d))
            segment['feature'] = ('log_unoccluded_projected_area' if 'move' in segment['motion'] else
                'unwrapped_subject_azimuth' if segment['motion'] == 'orbit' else
                'fixed_start_frame_lateral_displacement' if 'truck' in segment['motion'] else
                'world_vertical_displacement' if 'pedestal' in segment['motion'] else 'unsupported')
            segment['supported'] &= segment['feature'] != 'unsupported'

    def features(self, segment, positions, rotations):
        positions = np.asarray(positions)
        motion = segment['motion']
        if motion in ('movein', 'moveout'):
            return np.array([np.log(max(projected_area(p, r, self.model, self.intrinsics) or np.nan, 1e-12))
                             for p, r in zip(positions, rotations)])
        relative = positions-self.original.subject
        if motion == 'orbit':
            theta = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
            return theta+2*np.pi*round((segment['angle_anchor']-theta[0])/(2*np.pi))
        return relative@np.asarray(segment['axis'])

    def to_dict(self):
        return native(dict(reference_sha256=self.original.sha256, policy=asdict(self.policy),
            segments=self.segments, subject_model=self.model, intrinsics=self.intrinsics,
            reference_role='directed_ranges_and_order_only', motion_clock='independent_of_scene_time'))

    def measure(self, times, positions, rotations, scenes, progress_times):
        """Evaluate an actual prefix plus proposed suffix without resetting credit."""
        times, positions = np.asarray(times), np.asarray(positions)
        progress_times = np.asarray(progress_times)
        visible = [visibility(p, r, scene, float(t), self.model, self.intrinsics)
                   for t, p, r, scene in zip(times, positions, rotations, scenes)]
        return self.score(times, positions, rotations, visible, progress_times)

    def score(self, times, positions, rotations, measurements, progress_times):
        times, progress_times = np.asarray(times), np.asarray(progress_times)
        p = self.policy
        known = np.array([v['visibility'] is not None and v['in_frame'] is not None for v in measurements])
        vis = np.array([v['visibility'] if v['visibility'] is not None else 0. for v in measurements])
        frame = np.array([v['in_frame'] if v['in_frame'] is not None else 0. for v in measurements])
        good = known&(vis >= p.visibility_floor)&(frame >= .98)
        dt = np.diff(times)
        longest = run = 0.
        for i, h in enumerate(dt):
            run = 0. if good[i] and good[i+1] else run+h
            longest = max(longest, run)
        segments = []
        for index, segment in enumerate(self.segments):
            ids = np.flatnonzero((progress_times >= segment['start_s']-1e-7)&
                                 (progress_times <= segment['end_s']+1e-7))
            if len(ids) < 2 or not segment['supported']:
                segments.append(dict(index=index, motion=segment['motion'], complete=False,
                    effective_coverage=0., reason='missing_or_unsupported_intent_segment'))
                continue
            if segment['motion'] in ('movein','moveout') and all(measurements[i].get('projected_area') for i in ids):
                values = np.log([measurements[i]['projected_area'] for i in ids])
            else:
                values = self.features(segment, np.asarray(positions)[ids], np.asarray(rotations)[ids])
            delta = segment['end_value']-segment['start_value']
            rho = (values-segment['start_value'])/delta
            reversal = float(np.sum(np.maximum(0., -np.diff(rho))))
            dwell = np.zeros(p.coverage_bins)
            highwater = -np.inf
            for j in range(len(ids)-1):
                a, b = rho[j:j+2]
                # Ordered *new* progress, never absolute distance or repeated
                # visits. No interpolation credit across a visibility gap.
                left, right = max(a, highwater, 0.), min(b, 1.)
                highwater = max(highwater, a)
                if right <= left or not (good[ids[j]] and good[ids[j+1]]):
                    continue
                width = max(b-a, 1e-12)
                for k in range(p.coverage_bins):
                    overlap = max(0., min(right, (k+1)/p.coverage_bins)-max(left, k/p.coverage_bins))
                    dwell[k] += (times[ids[j+1]]-times[ids[j]])*overlap/width
                highwater = max(highwater, b)
            coverage = float(np.mean(dwell >= p.minimum_bin_dwell_s))
            start_error, end_error = abs(float(rho[0])), abs(float(rho[-1])-1.)
            geometric = bool(start_error <= p.endpoint_fraction and end_error <= p.endpoint_fraction
                and reversal <= p.reversal_fraction_max and np.min(rho) >= -p.endpoint_fraction
                and np.max(rho) <= 1.+p.endpoint_fraction)
            segments.append(dict(index=index, motion=segment['motion'], feature=segment['feature'],
                requested_range=[segment['start_value'], segment['end_value']],
                actual_range=[float(values.min()), float(values.max())],
                start_error_fraction=start_error, end_error_fraction=end_error,
                reverse_travel_fraction=reversal, effective_coverage=coverage,
                geometric_complete=geometric, complete=geometric and coverage >= p.effective_coverage_min,
                coverage_bin_dwell_s=dwell.tolist()))
        geometric = bool(segments and all(s.get('geometric_complete', False) for s in segments))
        ordered = bool(np.all(np.diff(progress_times) >= -1e-7) and progress_times[0] <= 1e-6
                       and progress_times[-1] >= self.original.duration-1e-6)
        complete = bool(segments and all(s['complete'] for s in segments) and longest <= p.max_occlusion_s and ordered)
        return native(dict(intent_complete=complete, geometric_complete=geometric, segments=segments,
            effective_coverage=min((s['effective_coverage'] for s in segments), default=0.),
            visibility_mean=float(np.mean(vis)), visibility_p05=float(np.percentile(vis, 5)),
            visibility_min=float(vis.min()), longest_occlusion_s=longest,
            in_frame_min=float(frame.min()), evidence_known=bool(known.all()),
            duration_s=float(times[-1]), progress_s=float(progress_times[-1]), phase_order_valid=ordered,
            source='calibrated_geometry_not_RGB_or_SAM', tracking_error_used_for_selection=False))


def progress_at(path, t):
    return path.progress_time(t) if hasattr(path, 'progress_time') else min(float(t), path.duration)


def selection_key(metrics, smoothness_cost=0.):
    """Lexicographic order: no reference loss can compensate for unfinished work."""
    return (not metrics['intent_complete'], not metrics['geometric_complete'],
            -metrics['effective_coverage'], metrics['longest_occlusion_s'],
            -metrics['visibility_p05'], -metrics['visibility_mean'],
            smoothness_cost, metrics['duration_s'])
