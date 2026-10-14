"""Text-only subject measurements and reference/candidate/actual comparisons.

A cylinder fit is a calibrated shape estimate, not semantic body-pose inference.
The CPU adapter raycasts the current scene; only hit points enter the estimator.
"""
from collections import deque

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .camera_intent import cylinder_points, project
from .contracts import native


def frame_state(position, rotation, model, intrinsics):
    if model is None:
        return dict(valid=False, reason='missing_subject_geometry', body_orientation='unknown')
    uv, depth = project(cylinder_points(model), np.asarray(position), rotation, intrinsics)
    if np.any(depth <= 1e-5):
        return dict(valid=False, reason='subject_behind_camera', body_orientation='unknown')
    polygon = uv[ConvexHull(uv).vertices]
    a, b = polygon, np.roll(polygon, -1, axis=0)
    cross = a[:,0]*b[:,1]-b[:,0]*a[:,1]
    area = .5*cross.sum()
    centroid = ((a+b)*cross[:,None]).sum(axis=0)/(6*area)
    xx = np.sum((a[:,0]**2+a[:,0]*b[:,0]+b[:,0]**2)*cross)/(12*area)
    yy = np.sum((a[:,1]**2+a[:,1]*b[:,1]+b[:,1]**2)*cross)/(12*area)
    xy = np.sum((2*a[:,0]*a[:,1]+a[:,0]*b[:,1]+b[:,0]*a[:,1]+2*b[:,0]*b[:,1])*cross)/(24*area)
    covariance=np.array([[xx,xy],[xy,yy]])-np.outer(centroid,centroid)
    eigenvalues,axes=np.linalg.eigh(covariance)
    anisotropy=float((eigenvalues[-1]-eigenvalues[0])/max(eigenvalues.sum(),1e-12))
    principal=axes[:,-1]
    silhouette_axis=float(np.arctan2(principal[1],principal[0])%np.pi) if anisotropy>.1 else None
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    relative = np.asarray(position)-np.asarray(model['position'])
    distance = np.linalg.norm(relative)
    return native(dict(valid=True, centroid=centroid, unoccluded_area=abs(float(area)),
        bbox=np.r_[lo, hi], bbox_size=hi-lo, aspect_ratio=float((hi[0]-lo[0])/(hi[1]-lo[1])),
        view_direction=relative/max(distance,1e-8), distance_m=distance,
        elevation_rad=float(np.arctan2(relative[2], np.linalg.norm(relative[:2]))),
        silhouette_axis_rad=silhouette_axis,silhouette_anisotropy=anisotropy,
        body_orientation='unknown', shape_model='calibrated_cylinder',
        source='geometry_projection_not_observed_mask'))


def frame_error(actual, target):
    if not actual.get('valid') or not target.get('valid'):
        return dict(valid=False, reason='missing_frame_state')
    aa,ta=actual.get('silhouette_axis_rad'),target.get('silhouette_axis_rad')
    axis_error=None if aa is None or ta is None else float(abs((aa-ta+np.pi/2)%np.pi-np.pi/2))
    return dict(valid=True,silhouette_axis_error_rad=axis_error,
        centroid_error=float(np.linalg.norm(np.asarray(actual['centroid'])-target['centroid'])),
        log_area_error=abs(float(np.log(actual['unoccluded_area']/target['unoccluded_area']))),
        bbox_log_error=float(np.max(np.abs(np.log(np.asarray(actual['bbox_size'])/target['bbox_size'])))),
        view_angle_error_rad=float(np.arccos(np.clip(np.dot(actual['view_direction'],target['view_direction']),-1.,1.))))


def cylinder_ray_distances(origin, rays, model):
    """First hit, supports a calibrated oriented solid cylinder."""
    rotation = Rotation.from_quat(model.get('orientation_xyzw',[0.,0.,0.,1.])).as_matrix()
    o = (np.asarray(origin)-model['position'])@rotation
    d = np.asarray(rays)@rotation
    a = (d[:,:2]**2).sum(axis=1); b = 2*(d[:,:2]*o[:2]).sum(axis=1)
    disc = b*b-4*a*((o[:2]**2).sum()-model['radius']**2)
    hit = np.full(len(d),np.inf)
    for sign in (-1.,1.):
        root = (-b+sign*np.sqrt(np.maximum(0.,disc)))/np.maximum(2*a,1e-12)
        valid = (disc>=0)&(root>.05)&(np.abs(o[2]+root*d[:,2])<=model['height']/2)
        hit = np.minimum(hit,np.where(valid,root,np.inf))
    for z in (-model['height']/2,model['height']/2):
        root = (z-o[2])/np.where(np.abs(d[:,2])>1e-12,d[:,2],1e-12)
        xy = o[:2]+root[:,None]*d[:,:2]
        valid = (root>.05)&((xy**2).sum(axis=1)<=model['radius']**2)
        hit = np.minimum(hit,np.where(valid,root,np.inf))
    return hit


def simulated_subject_returns(origin, rays, model, spheres):
    """Nearest returns from subject AND obstacles, without semantic object IDs.

    The fitter performs the initial-calibration ROI association itself.
    """
    distance = cylinder_ray_distances(origin,rays,model)
    for sphere in spheres:
        relative=np.asarray(origin)-sphere['center'];along=rays@relative
        disc=along**2-(relative@relative-sphere['radius']**2)
        near=-along-np.sqrt(np.maximum(0.,disc))
        distance=np.where((disc>=0)&(near>.05)&(near<distance),near,distance)
    valid=np.isfinite(distance)&(distance<4.5)
    return np.asarray(origin)+rays[valid]*distance[valid,None]


def fit_subject_points(points, calibration, stamp):
    """Fit XY from side returns; Z needs a visible cap, otherwise calibrated prior.

    The initial association and dimensions are calibration, explicitly reported.
    No point centroid is mistaken for the center of a partially observed object.
    """
    points=np.asarray(points,float).reshape(-1,3)
    record=dict(subject_id=calibration['id'],stamp_s=float(stamp),source='lidar_cylinder_surface_fit',
        measured=True,valid=False,returns=len(points),body_orientation='unknown',orientation_valid=False)
    if len(points)<12:
        return dict(record,reason='insufficient_subject_returns')
    if not np.allclose(calibration.get('orientation_xyzw',[0,0,0,1]),[0,0,0,1]):
        return dict(record,reason='upright_cylinder_fit_only')
    center=np.array(calibration['position'],float)
    associated=points[np.linalg.norm(points-center,axis=1)<calibration['height']/2+calibration['radius']+.08]
    side=associated[np.abs(associated[:,2]-center[2])<calibration['height']/2-.008]
    if len(side)<12:
        return dict(record,reason='insufficient_side_surface_returns')
    fit,_,rank,_=np.linalg.lstsq(np.column_stack([2*side[:,:2],np.ones(len(side))]),(side[:,:2]**2).sum(axis=1),rcond=None)
    radius2=fit[2]+fit[:2]@fit[:2]
    if rank<3 or radius2<=0 or abs(np.sqrt(radius2)-calibration['radius'])>.015:
        return dict(record,reason='shape_or_association_mismatch')
    center[:2]=fit[:2]
    # Interior cap returns identify top/bottom unambiguously with known height.
    cap=associated[np.linalg.norm(associated[:,:2]-center[:2],axis=1)<calibration['radius']-.008]
    measured_z=False
    if len(cap)>=8:
        z=float(np.median(cap[:,2])); sign=np.sign(z-center[2])
        if abs(z-center[2])>calibration['height']*.3 and np.std(cap[:,2])<.003:
            center[2]=z-sign*calibration['height']/2;measured_z=True
    residual=float(np.sqrt(np.mean((np.linalg.norm(side[:,:2]-center[:2],axis=1)-calibration['radius'])**2)))
    return dict(record,valid=True,reason='',position=center.tolist(),radius_estimate_m=float(np.sqrt(radius2)),
        fit_residual_m=residual,selected_returns=len(side),position_components=['measured','measured','measured_cap' if measured_z else 'calibrated_prior'],
        position_uncertainty_m=[max(.002,residual),max(.002,residual),.003 if measured_z else .02],
        stationary=True,dimensions_source='radius_fitted_height_initial_calibration',identity_source='initial_geometric_association')


class SubjectStateObserver:
    def __init__(self, calibration, intrinsics):
        self.calibration=dict(calibration);self.intrinsics=intrinsics
        self.history=deque(maxlen=12);self.latest=None

    def observe(self, points, position, rotation, stamp):
        geometry=fit_subject_points(points,self.calibration,stamp)
        state=frame_state(position,rotation,dict(self.calibration,position=geometry['position'],radius=geometry['radius_estimate_m']),self.intrinsics) if geometry['valid'] else dict(valid=False)
        self.latest=dict(subject_id=self.calibration['id'],stamp_s=float(stamp),geometry=geometry,
            state=state,visual=dict(status='unknown',reason='RGB_SAM_not_enabled'),
            source='current_simulated_lidar_returns_with_calibrated_shape',measured=True)
        if geometry['valid'] and state['valid']:
            self.history.append((float(stamp),state['unoccluded_area']))
        return self.snapshot(stamp)

    def snapshot(self, now):
        if self.latest is None:return dict(status='unknown',reason='missing_subject_scan')
        age=float(now-self.latest['stamp_s'])
        valid=self.latest['geometry']['valid'] and -1e-8<=age<=.3
        h=list(self.history)
        trend=None
        if len(h)>=3 and h[-1][0]-h[0][0]>.2:
            trend=float(np.polyfit([v[0]-h[0][0] for v in h],np.log([v[1] for v in h]),1)[0])
        return native(dict(self.latest,status='valid' if valid else 'unknown',age_s=age,
            scale_trend_log_per_s=trend,scale_trend=('unknown' if trend is None else
                'increasing' if trend>.005 else 'decreasing' if trend<-.005 else 'steady'),body_orientation='unknown',
            reliability='shape-conditioned LiDAR estimate; not RGB appearance or semantic body pose'))


def compare_subject_evidence(original, active, snapshot, model, intrinsics, now, camera_feedback=None):
    """Three-way capture-time comparison, carrying missing fields as unknown.

    SAM features compare against projected shape only when usable. A segmentation
    confidence never supplies external visibility or body orientation.
    """
    from .camera_intent import progress_at
    visual=snapshot.get('visual',{})
    geometry=snapshot.get('aligned_geometry',{}) if visual.get('status')=='valid' else snapshot.get('geometry',{})
    stamp=float(visual.get('stamp_s',now)) if visual.get('status')=='valid' else float(now)
    progress=progress_at(active,stamp)
    op,ori,_=original.sample(progress); cp,cr,_=active.sample(stamp)
    reference=frame_state(op,ori,model,intrinsics) if intrinsics else dict(valid=False)
    candidate=frame_state(cp,cr,model,intrinsics) if intrinsics else dict(valid=False)
    terminal=frame_state(*original.sample(original.duration)[:2],model,intrinsics) if intrinsics else dict(valid=False)
    actual=dict(valid=False,source='unknown',body_orientation='unknown')
    observed_errors={}
    pose_record=visual if 'camera_position' in visual else (camera_feedback or {})
    pose_stamp=float(pose_record.get('stamp_s',pose_record.get('time_s',-1e9)))
    if (geometry.get('status')=='valid' and model is not None and intrinsics
            and 'camera_position' in pose_record and abs(pose_stamp-stamp)<=.12):
        measured_model=dict(model,position=geometry['position'])
        measured_rotation=Rotation.from_quat(pose_record['camera_quaternion_xyzw']).as_matrix()
        actual=frame_state(pose_record['camera_position'],measured_rotation,measured_model,intrinsics)
        actual.update(source='measured_subject_geometry_and_capture_pose_shape_projection',
            measured_geometry_source=geometry.get('source'),body_orientation='unknown')
        observed_errors={name:frame_error(actual,target) for name,target in
            (('reference',reference),('candidate',candidate),('terminal',terminal))}
    if visual.get('status')=='valid':
        actual=dict(visual,source=visual.get('source','SAM_mask'),valid=True,
            unoccluded_area=visual['visible_area'] if snapshot.get('scale_measurement_valid') else None,
            view_direction=None,body_orientation='unknown')
        for name,target in (('reference',reference),('candidate',candidate),('terminal',terminal)):
            error=dict(centroid_error=None,log_area_error=None,view_angle_error_rad=None,bbox_log_error=None)
            if target.get('valid'):
                error['centroid_error']=float(np.linalg.norm(np.asarray(visual['centroid'])-target['centroid']))
                if snapshot.get('scale_measurement_valid'):
                    error['log_area_error']=abs(float(np.log(visual['visible_area']/target['unoccluded_area'])))
                    if visual.get('bbox_size'):
                        error['bbox_log_error']=float(np.max(np.abs(np.log(np.asarray(visual['bbox_size'])/target['bbox_size']))))
                if geometry.get('status')=='valid' and 'camera_position' in visual:
                    v=np.asarray(visual['camera_position'])-geometry['position'];v/=max(np.linalg.norm(v),1e-8)
                    error['view_angle_error_rad']=float(np.arccos(np.clip(v@target['view_direction'],-1.,1.)))
            observed_errors[name]=error
    return native(dict(reference_sha256=original.sha256,reference_at_capture_progress=reference,
        candidate_prediction_at_capture=candidate,actual_measurement=actual,
        immutable_terminal_target=terminal,errors=observed_errors,
        capture_stamp_s=stamp,progress_s=progress,measured_scale_valid=snapshot.get('scale_measurement_valid',False),
        observation_status=visual.get('status','unknown'),geometry_status=geometry.get('status','unknown'),
        body_orientation='unknown',terminal_verified=False))


def subject_observation_error(observation, original, now):
    """Latest measured identity, timestamp and calibration must remain consistent.

    None is allowed for offline geometry-only tools; run_case always supplies
    actual sensor evidence and independently requires terminal confirmation.
    """
    if observation is None:return ''
    geometry=observation.get('geometry',{})
    if observation.get('status')!='valid' or not geometry.get('valid'):
        return 'subject_measurement_unknown_or_stale'
    stamp=observation.get('stamp_s',geometry.get('stamp_s'))
    if stamp is None or not np.isfinite(stamp) or not -1e-8<=now-stamp<=.3:
        return 'subject_measurement_unknown_or_stale'
    if observation.get('subject_id',geometry.get('subject_id'))!=original.task['subject']['id']:
        return 'subject_measurement_identity_mismatch'
    position=np.asarray(geometry.get('position'),float)
    if position.shape!=(3,) or not np.isfinite(position).all():return 'subject_measurement_invalid_geometry'
    if np.linalg.norm(position-original.subject)>.03:
        return 'subject_calibration_mismatch_requires_reconstruction'
    return ''
