"""Camera-only completion specifications and geometric visibility measurements.

Reference time indexes intent, never a tracking objective. Scene timestamps remain
physical time. The calibrated cylinder and sphere scene are explicit CPU fixture
models; their measurements are not SAM masks or a continuous safety certificate.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .contracts import native


@dataclass(frozen=True)
class IntentPolicy:
    version: str = 'camera_intent/v2_subject_frame_trend'
    collision_margin_m: float = .025
    visibility_floor: float = .85
    effective_coverage_min: float = .95
    endpoint_fraction: float = .035
    reversal_fraction_max: float = .025
    max_occlusion_s: float = .25
    coverage_bins: int = 32
    minimum_bin_dwell_s: float = .025
    max_duration_factor: float = 1.75
    min_duration_factor: float = .65
    max_speed_m_s: float = 1.5
    max_acceleration_m_s2: float = 4.
    max_angular_speed_rad_s: float = 3.
    terminal_centroid_tolerance: float = .02
    terminal_log_area_tolerance: float = .05
    terminal_bbox_log_tolerance: float = .06
    terminal_view_tolerance_rad: float = np.deg2rad(3.)
    terminal_silhouette_axis_tolerance_rad: float = np.deg2rad(3.)
    terminal_hold_s: float = .4
    terminal_speed_m_s: float = .03
    companion_displacement_max_m: float = .24
    companion_travel_max_m: float = .52
    process_view_error_max_rad: float = np.deg2rad(12.)
    process_log_area_error_max: float = .28


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
    velocities = np.asarray(scene.get('velocities', np.zeros_like(centers)), float).reshape(-1,3)
    age = t-float(scene.get('observation_stamp', t))
    radii = np.asarray(scene.get('radii', []), float)
    radii = radii+max(0., age)*np.asarray(scene.get('radius_rates', np.zeros(len(radii))))
    return centers+age*velocities, radii


def nominal_visibility_scene(scene):
    """Forecast opacity from observed proxies, not future collision uncertainty.

    The caller must label this as a nominal forecast. The original scene and
    its conservative collision envelope remain intact for physical checking.
    """
    return dict(scene,radius_rates=np.zeros(len(scene.get('radii',[]))))


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
    # A sphere outside the ray segment AABB cannot occlude any target ray.
    # Most observed wall surfels lie beyond the subject; culling them is exact.
    ray_low = np.minimum(position,targets.min(axis=0))
    ray_high = np.maximum(position,targets.max(axis=0))
    possible = np.all(centers+radii[:,None] >= ray_low,axis=1) & np.all(centers-radii[:,None] <= ray_high,axis=1)
    for center, radius in zip(centers[possible], radii[possible]):
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
        self.original, self.model, self.policy = original, deepcopy(model), policy
        self.intrinsics = list(default_intrinsics() if intrinsics is None else intrinsics)
        if len(self.intrinsics)!=4 or not np.isfinite(self.intrinsics).all() or min(self.intrinsics[:2])<=0:
            raise ValueError('finite positive normalized focal lengths are required')
        if model is not None:
            if model.get('id', original.task['subject']['id']) != original.task['subject']['id']:
                raise ValueError('calibrated subject identity does not match task')
            if not np.allclose(model['position'], original.subject, atol=1e-8, rtol=0.):
                raise ValueError('this intent model requires a stationary calibrated subject')
            if not all(np.isfinite(model[k]) and model[k] > 0 for k in ('radius','height')):
                raise ValueError('invalid calibrated subject dimensions')
        self._frame_cache = {}
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
            values = self.features(segment,original.translation(times),original.rotation(times).as_matrix())
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
            reference_role='directed_ranges_order_trend_and_immutable_terminal_frame',
            terminal_target=self.reference_frame(self.original.duration), motion_clock='independent_of_scene_time'))

    def reference_frame(self, progress):
        from .subject_state import frame_state
        key = float(np.clip(progress, 0., self.original.duration))
        if key not in self._frame_cache:
            p, r = self.original.translation(key), self.original.rotation(key).as_matrix()
            self._frame_cache[key] = frame_state(p, r, self.model, self.intrinsics)
        return self._frame_cache[key]

    def frame_and_trend(self, times, positions, rotations, progress_times):
        from .subject_state import frame_states, frame_error
        states = frame_states(positions,rotations,self.model,self.intrinsics)
        missing=np.array(sorted(set(float(np.clip(t,0.,self.original.duration)) for t in progress_times)-self._frame_cache.keys()))
        if len(missing):
            frames=frame_states(self.original.translation(missing),self.original.rotation(missing).as_matrix(),self.model,self.intrinsics)
            self._frame_cache.update(zip(missing,frames))
        references = [self.reference_frame(t) for t in progress_times]
        errors = [frame_error(a,b) for a,b in zip(states,references)]
        if not errors or not all(e['valid'] for e in errors):
            return dict(terminal=dict(pass_check=False,reason='unknown_subject_frame'),
                        trend=dict(pass_check=False,reason='unknown_subject_frame'))
        target = self.reference_frame(self.original.duration)
        window = np.flatnonzero(times >= times[-1]-self.policy.terminal_hold_s-1e-7)
        terminal_errors = [frame_error(states[i],target) for i in window]
        terminal = {k:(max(e[k] for e in terminal_errors) if all(e[k] is not None for e in terminal_errors) else None)
                    for k in errors[0] if k!='valid'}
        speed = np.linalg.norm(np.diff(np.asarray(positions)[window],axis=0),axis=1)/np.maximum(np.diff(times[window]),1e-8)
        terminal.update(confirmed_window_s=float(times[window[-1]]-times[window[0]]),
            max_speed_m_s=float(max(speed,default=0.)),target=target,predicted_or_actual=states[-1],
            source='pose_and_calibrated_shape_requires_independent_observation_confirmation')
        p = self.policy
        terminal['pass_check'] = bool(terminal['centroid_error']<=p.terminal_centroid_tolerance and
            terminal['log_area_error']<=p.terminal_log_area_tolerance and
            terminal['bbox_log_error']<=p.terminal_bbox_log_tolerance and
            terminal['view_angle_error_rad']<=p.terminal_view_tolerance_rad and
            (target.get('silhouette_axis_rad') is None or terminal['silhouette_axis_error_rad'] is not None
             and terminal['silhouette_axis_error_rad']<=p.terminal_silhouette_axis_tolerance_rad) and
            terminal['max_speed_m_s']<=p.terminal_speed_m_s and
            terminal['confirmed_window_s']>=p.terminal_hold_s-.051)
        nominal = self.original.translation(np.clip(progress_times,0.,self.original.duration))
        residual = np.asarray(positions)-nominal
        # Progress-aligned subject-relative companion displacement. Reference's
        # own dolly/rise/orbit trends are retained; no equal-time tracking cost.
        displacement = float(np.linalg.norm(residual,axis=1).max())
        travel = float(np.linalg.norm(np.diff(residual,axis=0),axis=1).sum())
        trend = dict(companion_displacement_max_m=displacement,companion_travel_m=travel,
            view_angle_error_max_rad=max(e['view_angle_error_rad'] for e in errors),
            log_area_error_max=max(e['log_area_error'] for e in errors),
            centroid_error_max=max(e['centroid_error'] for e in errors),
            comparison_clock='motion_progress_not_scene_time')
        trend['pass_check'] = bool(displacement<=p.companion_displacement_max_m and
            travel<=p.companion_travel_max_m and trend['view_angle_error_max_rad']<=p.process_view_error_max_rad and
            trend['log_area_error_max']<=p.process_log_area_error_max)
        return native(dict(terminal=terminal,trend=trend))

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
        framing = self.frame_and_trend(times, positions, rotations, progress_times)
        range_complete = complete
        complete = complete and framing['terminal']['pass_check'] and framing['trend']['pass_check']
        violations = []
        if not range_complete:
            violations.append(dict(kind='incomplete_visible_intent',
                effective_coverage=min((s['effective_coverage'] for s in segments), default=0.),
                required_coverage=p.effective_coverage_min, longest_occlusion_s=longest,
                interval_s=[float(times[0]),float(times[-1])],
                consequence='missing filmed coverage; previous occluded travel cannot be credited',
                recovery='not_verified'))
        for name, value in (('terminal_frame_mismatch',framing['terminal']),
                            ('motion_trend_violation',framing['trend'])):
            if not value['pass_check']:
                violations.append(dict(kind=name, measurements=value,
                    interval_s=([float(times[-1]-p.terminal_hold_s),float(times[-1])]
                        if name=='terminal_frame_mismatch' else [float(times[0]),float(times[-1])]),
                    consequence='terminal composition mismatch' if name=='terminal_frame_mismatch' else 'process deviates from original shot',
                    recovery='not_verified'))
        return native(dict(intent_complete=complete, range_complete=range_complete,
            intent_violations=violations,
            terminal_frame=framing['terminal'], motion_trend=framing['trend'],
            geometric_complete=geometric, segments=segments,
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
    quality = metrics.get('curve_quality', {})
    terminal = metrics.get('terminal_frame', {})
    return (not quality.get('direction_pass',True), not quality.get('regular_curve_pass',True),
            not metrics['intent_complete'], not metrics['geometric_complete'],
            -metrics['effective_coverage'], metrics['longest_occlusion_s'],
            -metrics['visibility_p05'], -metrics['visibility_mean'],
            round(metrics.get('motion_trend',{}).get('companion_travel_m',0.),3),
            0. if terminal.get('pass_check') else round(terminal.get('log_area_error',0.),4),
            metrics['duration_s'], smoothness_cost)
