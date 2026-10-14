"""Small, timestamped subject evidence. No neural inference in the WBC process.

Camera-link convention: +X forward, +Y left, +Z up. Intrinsics are normalized
fx/width, fy/height, cx/width, cy/height. Unknown visibility stays unknown.
"""
from copy import deepcopy
from collections import deque

import numpy as np

from .contracts import native, number


def camera_lidar_transform(world_camera, world_lidar):
    """T_CL = inv(T_WC) T_WL; both transforms MUST be at acquisition time."""
    wc, wl = np.asarray(world_camera, float), np.asarray(world_lidar, float)
    for transform in (wc, wl):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError('expected finite acquisition-time SE(3) transforms')
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError('invalid homogeneous transform')
        r = transform[:3, :3]
        if not np.allclose(r.T@r, np.eye(3), atol=1e-5) or np.linalg.det(r) < .999:
            raise ValueError('invalid rotation')
    return np.linalg.solve(wc, wl)


def mask_evidence(mask, subject_id, stamp_s, *, identity_valid=False,
                  confidence=None, occluded=None, source='sam2.1_rgb'):
    """Reduce a SAM mask outside control; mask area is VISIBLE area, not size."""
    result = dict(subject_id=subject_id, stamp_s=float(stamp_s), source=source,
                  identity_valid=bool(identity_valid), confidence=confidence,
                  occluded=occluded, valid=False, reason='missing_mask')
    if mask is None:
        return result
    values = np.asarray(mask)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        return {**result, 'reason': 'invalid_mask'}
    values = values.astype(bool)
    y, x = np.nonzero(values)
    if not len(x):
        return {**result, 'reason': 'empty_mask'}
    h, w = values.shape
    edge = min(x.min()/w, (w-1-x.max())/w, y.min()/h, (h-1-y.max())/h)
    coordinates = np.column_stack([(x+.5)/w, (y+.5)/h])
    covariance = np.cov(coordinates, rowvar=False, bias=True) if len(x)>1 else np.zeros((2,2))
    eigenvalues, axes = np.linalg.eigh(covariance)
    principal = axes[:, -1]
    anisotropy = float((eigenvalues[-1]-eigenvalues[0])/max(eigenvalues.sum(),1e-12))
    return {**result, 'valid': True, 'reason': '',
            'bbox': [float(x.min()/w),float(y.min()/h),float((x.max()+1)/w),float((y.max()+1)/h)],
            'bbox_size': [float((x.max()-x.min()+1)/w),float((y.max()-y.min()+1)/h)],
            'aspect_ratio': float(((x.max()-x.min()+1)/w)/((y.max()-y.min()+1)/h)),
            'silhouette_axis_rad': float(np.arctan2(principal[1],principal[0])%np.pi) if anisotropy>.1 else None,
            'silhouette_anisotropy': anisotropy, 'body_orientation': 'unknown',
            'unoccluded_area': None, 'external_visibility_fraction': None,
            'scale_measurement_valid': bool(occluded is False and edge>0),
            'centroid': [float(np.mean(x+.5)/w), float(np.mean(y+.5)/h)],
            'visible_area': float(values.mean()), 'edge_margin': float(edge),
            'truncated': bool(edge == 0.)}


def project_subject(position, rotation, subject, intrinsics):
    relative = np.asarray(rotation).T@(np.asarray(subject)-position)
    depth = float(relative[0])
    if depth <= 1e-5:
        return None
    fx, fy, cx, cy = intrinsics
    return np.array([cx-fx*relative[1]/depth, cy-fy*relative[2]/depth])


def phase_progress(original, t, position, rotation, subject):
    """Signed phase-specific progress; pure Pan never uses translation progress."""
    from scipy.spatial.transform import Rotation
    phase = original.phase(t)
    start, r0, _ = original.sample(phase['start_s'])
    end, r1, _ = original.sample(phase['end_s'])
    nominal, rn, _ = original.sample(t)
    motion = phase['motion'].lower()
    if motion == 'pan':
        axis = Rotation.from_matrix(r1@r0.T).as_rotvec()
        width = float(axis@axis)
        if width < 1e-8:
            return None, None
        return (float(Rotation.from_matrix(rotation@r0.T).as_rotvec()@axis/width),
                float(Rotation.from_matrix(rn@r0.T).as_rotvec()@axis/width))
    if motion == 'orbit':
        a = start-original.subject
        def angle(p, center):
            b = p-center
            return float(np.arctan2(a[0]*b[1]-a[1]*b[0], a[:2]@b[:2]))
        span = angle(end, original.subject)
        if abs(span) < 1e-5:
            return None, None
        return angle(position, subject)/span, angle(nominal, original.subject)/span
    if motion in ('movein', 'moveout'):
        d0 = np.linalg.norm(start-original.subject)
        span = np.linalg.norm(end-original.subject)-d0
        if abs(span) < 1e-5:
            return None, None
        return (float((np.linalg.norm(position-subject)-d0)/span),
                float((np.linalg.norm(nominal-original.subject)-d0)/span))
    axis = end-start
    if axis@axis < 1e-8:
        return None, None
    return float((position-start)@axis/(axis@axis)), float((nominal-start)@axis/(axis@axis))


class SubjectEvidence:
    """Latest evidence plus a bounded geometry history for delayed RGB results.

    Updates are supplied by the sensor adapters on the SAME shot clock as WBC.
    Invalid newer samples replace current evidence; they cannot resurrect history.
    """
    def __init__(self, subject_id, max_age_s=.6, max_skew_s=.12):
        self.subject_id, self.max_age_s, self.max_skew_s = subject_id, max_age_s, max_skew_s
        self.visual = self.geometry = self.last_trusted_visual = None
        self.geometry_history = deque(maxlen=64)
        self.visual_history = deque(maxlen=32)
        self.revision = 0

    def update_visual(self, record):
        data = deepcopy(record)
        number(data['stamp_s'], 'visual stamp')
        if data.get('subject_id') != self.subject_id:
            raise ValueError('visual subject identity mismatch')
        if data.get('valid'):
            xy = np.asarray(data.get('centroid'), float)
            if xy.shape != (2,) or not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
                raise ValueError('invalid normalized centroid')
            if not 0 < number(data['visible_area'], 'visible area') <= 1:
                raise ValueError('invalid visible area')
            if not 0 <= number(data['edge_margin'], 'edge margin') <= .5:
                raise ValueError('invalid edge margin')
        confidence = data.get('confidence')
        if confidence is not None and not 0 <= number(confidence, 'confidence') <= 1:
            raise ValueError('invalid segmentation confidence')
        if 'camera_position' in data:
            p = np.asarray(data['camera_position'], float)
            q = np.asarray(data.get('camera_quaternion_xyzw'), float)
            k = np.asarray(data.get('intrinsics'), float)
            if (p.shape != (3,) or q.shape != (4,) or k.shape != (4,) or
                    not np.isfinite(np.r_[p, q, k]).all() or abs(np.linalg.norm(q)-1.) > 1e-5 or np.any(k[:2] <= 0)):
                raise ValueError('invalid acquisition-time camera geometry')
        if self.visual and data['stamp_s'] <= self.visual['stamp_s']:
            return False
        self.visual = data
        self.visual_history.append(data)
        if data.get('valid') and data.get('identity_valid'):
            self.last_trusted_visual = deepcopy(data)
        self.revision += 1
        return True

    def update_geometry(self, record):
        data = deepcopy(record)
        number(data['stamp_s'], 'geometry stamp')
        if data.get('subject_id', self.subject_id) != self.subject_id:
            raise ValueError('geometry subject identity mismatch')
        if data.get('valid'):
            p = np.asarray(data['position'], float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError('invalid subject position')
        if self.geometry and data['stamp_s'] <= self.geometry['stamp_s']:
            return False
        self.geometry = data
        self.geometry_history.append(data)
        self.revision += 1
        return True

    def snapshot(self, now):
        def checked(record):
            if record is None:
                return dict(status='unknown', reason='missing')
            age = now-record['stamp_s']
            valid = record.get('valid') and -1e-8 <= age <= self.max_age_s
            return {**deepcopy(record), 'status': 'valid' if valid else 'unknown',
                    'age_s': age, 'fresh': -1e-8 <= age <= self.max_age_s}
        visual, geometry = checked(self.visual), checked(self.geometry)
        if visual.get('identity_valid') is not True:
            visual['status'] = 'unknown'
        # SAM returns after capture. Compare with geometry from that capture,
        # retaining the latest geometry separately for current planning. A new
        # invalid geometry sample must still invalidate fusion, not resurrect it.
        aligned_geometry = dict(status='unknown', reason='no_capture_time_geometry')
        if visual['status'] == geometry['status'] == 'valid' and self.geometry_history:
            closest = min(self.geometry_history, key=lambda row: abs(row['stamp_s']-visual['stamp_s']))
            if abs(closest['stamp_s']-visual['stamp_s']) <= self.max_skew_s:
                aligned_geometry = checked(closest)
        aligned = aligned_geometry['status'] == 'valid'
        consistent = None
        if aligned and 'camera_position' in visual:
            from scipy.spatial.transform import Rotation
            center = project_subject(np.asarray(visual['camera_position']),
                Rotation.from_quat(visual['camera_quaternion_xyzw']).as_matrix(),
                aligned_geometry['position'], visual['intrinsics'])
            consistent = bool(center is not None and np.linalg.norm(center-visual['centroid']) <= .08)
        scale_valid = (aligned and visual.get('occluded') is False and
                       visual.get('truncated') is False and aligned_geometry.get('stationary') is True and consistent is True)
        diagnosis = 'insufficient_evidence'
        if visual['status'] == 'valid':
            diagnosis = ('truncation' if visual.get('truncated') else
                         'occlusion' if visual.get('occluded') else
                         'visibility_unknown' if visual.get('occluded') is None else 'observed')
        if consistent is False:
            diagnosis = 'projection_association_mismatch'
        recent=[v for v in self.visual_history if now-v['stamp_s']<=1.2 and v.get('valid')
                and v.get('identity_valid') and v.get('occluded') is False and v.get('truncated') is False]
        slope=None
        if scale_valid and len(recent)>=3 and recent[-1]['stamp_s']-recent[0]['stamp_s']>.2:
            slope=float(np.polyfit([v['stamp_s']-recent[0]['stamp_s'] for v in recent],
                np.log([v['visible_area'] for v in recent]),1)[0])
        return native(dict(subject_id=self.subject_id, revision=self.revision, time_s=now,
            measured_scale_trend_log_per_s=slope,
            visual=visual, geometry=geometry, aligned_geometry=aligned_geometry, time_aligned=aligned,
            fusion_consistent=consistent,
            scale_measurement_valid=scale_valid, diagnosis=diagnosis,
            last_trusted_visual=deepcopy(self.last_trusted_visual),
            note='Historical evidence is not current. Area loss alone never requests radial compensation.'))


def visual_consequences(original, trace, evidence=None, intrinsics=None):
    """O(samples) projection of predicted EXECUTED poses onto original-time targets.

    No rendering, future simulator truth, or SAM counterfactuals. The area ratio is
    explicitly a range proxy for a stationary subject, not a predicted mask area.
    """
    from scipy.spatial.transform import Rotation
    evidence = evidence or {}
    geometry = evidence.get('geometry', {})
    subject = np.asarray(geometry['position'] if geometry.get('status') == 'valid'
                         else original.subject, float)
    geometry_source = geometry.get('source', 'runtime_initial_calibration') if geometry.get('status') == 'valid' else 'runtime_initial_calibration'
    rows = []
    for sample in trace:
        t = sample['t']
        p = np.asarray(sample['position'])
        r = Rotation.from_quat(sample['quaternion_xyzw']).as_matrix()
        op, ori, _ = original.sample(t)
        a, b = op-original.subject, p-subject
        da, db = np.linalg.norm(a), np.linalg.norm(b)
        uv = None if intrinsics is None else project_subject(p, r, subject, intrinsics)
        ref_uv = None if intrinsics is None else project_subject(op, ori, original.subject, intrinsics)
        angle = None if min(da, db) < 1e-6 else float(np.arccos(np.clip(a@b/(da*db), -1., 1.)))
        scale = None if min(da, db) < 1e-6 else float((da/db)**2)
        progress, nominal_progress = phase_progress(original, t, p, r, subject)
        rows.append(dict(t=float(t), phase=original.phase(t)['name'],
            phase_progress=progress, reference_phase_progress=nominal_progress,
            phase_progress_error=None if progress is None else abs(progress-nominal_progress),
            composition_error=None if uv is None or ref_uv is None else float(np.linalg.norm(uv-ref_uv)),
            range_scale_error=None if scale is None else abs(scale-1.),
            view_angle_error_rad=angle,
            orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(r@ori.T).as_rotvec())),
            projected_center=None if uv is None else uv.tolist(),
            center_in_frame=None if intrinsics is None else bool(uv is not None and np.all((uv >= 0) & (uv <= 1))),
            joint_margin_rad=sample['joint_margin_rad']))
    keys = ('composition_error', 'range_scale_error', 'view_angle_error_rad', 'orientation_error_rad', 'phase_progress_error')
    def summary(selected):
        out = {}
        for key in keys:
            valid = [row for row in selected if row[key] is not None]
            worst = max(valid, key=lambda row: row[key]) if valid else None
            out[key] = None if worst is None else float(worst[key])
            out[key+'_worst_s'] = None if worst is None else worst['t']
        out['center_out_of_frame_samples'] = sum(r['center_in_frame'] is False for r in selected)
        # Counts depend on rollout resolution (.02s near the install point,
        # .1s farther away); compare elapsed duration instead of sample counts.
        out['center_out_of_frame_duration_s'] = (None if intrinsics is None else float(sum(
            (b['t']-a['t'])*.5*(int(a['center_in_frame'] is False)+int(b['center_in_frame'] is False))
            for a, b in zip(selected[:-1], selected[1:]))))
        if selected and len({row['phase'] for row in selected}) == 1:
            out['executed_phase_progress_start'] = selected[0]['phase_progress']
            out['executed_phase_progress_end'] = selected[-1]['phase_progress']
            out['reference_phase_progress_start'] = selected[0]['reference_phase_progress']
            out['reference_phase_progress_end'] = selected[-1]['reference_phase_progress']
        out['min_joint_margin_rad'] = min((r['joint_margin_rad'] for r in selected if r['joint_margin_rad'] is not None), default=None)
        return out
    phases = {phase['name']: summary([r for r in rows if r['phase'] == phase['name']])
              for phase in original.task['shot_phase'] if any(r['phase'] == phase['name'] for r in rows)}
    return native(dict(source='predicted_execution_FK_projection', subject_source=geometry_source,
        prediction_interval_s=None if not rows else [rows[0]['t'], rows[-1]['t']],
        prediction_kind='conditional_geometry_proxy', occlusion='unknown',
        projected_scale='unknown', range_scale_assumption='stationary subject; inverse square range only',
        measured_scale_usable=evidence.get('scale_measurement_valid', False),
        summary=summary(rows), phases=phases, samples=rows))
