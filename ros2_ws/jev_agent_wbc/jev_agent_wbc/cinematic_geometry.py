"""Low-dimensional camera curves and numerically bounded final-frame freedom.

The terminal region is sampled along rays, not an axis-aligned box certificate.
Each selected endpoint is checked again against the unchanged image tolerances.
"""
import itertools
import numpy as np
from .camera_intent import POLICY, default_intrinsics
from .reference import look_at,look_at_many
from .subject_state import frame_state,frame_states,frame_error


def terminal_error(original, position, model, intrinsics):
    p, r, _ = original.sample(original.duration)
    rotation = look_at(position, original.subject) @ look_at(p, original.subject).T @ r
    key=repr((model,intrinsics))
    cache=original.__dict__.setdefault('_terminal_frame_cache',{})
    if key not in cache:cache[key]=frame_state(p,r,model,intrinsics)
    target=cache[key]
    error=frame_error(frame_state(position,rotation,model,intrinsics),target)
    error['silhouette_axis_required']=target.get('silhouette_axis_rad') is not None
    return error


def inside_terminal_band(error, reserve=1.):
    limits = dict(centroid_error=POLICY.terminal_centroid_tolerance,
                  log_area_error=POLICY.terminal_log_area_tolerance,
                  bbox_log_error=POLICY.terminal_bbox_log_tolerance,
                  view_angle_error_rad=POLICY.terminal_view_tolerance_rad,
                  silhouette_axis_error_rad=POLICY.terminal_silhouette_axis_tolerance_rad)
    return bool(error.get('valid') and
        (not error.get('silhouette_axis_required') or error.get('silhouette_axis_error_rad') is not None) and
        all(error.get(k) is not None and error[k] <= reserve*v
        for k,v in limits.items() if k != 'silhouette_axis_error_rad' or error.get(k) is not None))


def terminal_band(original, model, intrinsics=None, axis_only=False):
    intrinsics = default_intrinsics() if intrinsics is None else intrinsics
    key = repr((model, intrinsics,axis_only))
    cache = original.__dict__.setdefault('_terminal_band_cache', {})
    if key in cache:
        return cache[key]
    p = original.sample(original.duration)[0]
    offsets = [np.zeros(3)]
    if model is not None and axis_only:
        directions=np.array([xyz for xyz in itertools.product((-1.,0.,1.),repeat=3) if sum(v!=0 for v in xyz)==1])
        original_rotation=original.sample(original.duration)[1]
        residual=look_at(p,original.subject).T@original_rotation
        target=frame_state(p,original_rotation,model,intrinsics)
        lo=np.zeros(6);hi=np.full(6,.35)
        for _ in range(14):
            mid=(lo+hi)/2;positions=p+mid[:,None]*directions
            frames=frame_states(positions,look_at_many(positions,original.subject)@residual,model,intrinsics)
            errors=[dict(frame_error(frame,target),silhouette_axis_required=target.get('silhouette_axis_rad') is not None) for frame in frames]
            valid=np.array([inside_terminal_band(error,.8) for error in errors])
            lo=np.where(valid,mid,lo);hi=np.where(valid,hi,mid)
        offsets.extend(lo[:,None]*directions)
    elif model is not None:
        for xyz in itertools.product((-1., 0., 1.), repeat=3):
            if axis_only and sum(v!=0 for v in xyz)!=1:continue
            d = np.array(xyz)
            if not np.any(d):
                continue
            d /= np.linalg.norm(d)
            lo, hi = 0., .35
            for _ in range(14):
                mid = (lo+hi)/2
                if inside_terminal_band(terminal_error(original, p+mid*d, model, intrinsics), .8):
                    lo = mid
                else:
                    hi = mid
            offsets.append(lo*d)
    value = dict(method='6_axis_bisection_rechecked_endpoints' if axis_only else '26_ray_bisection_rechecked_endpoints', tolerance_reserve=.8,
                 offsets_m=[v.tolist() for v in offsets], reference_position=p.tolist())
    cache[key] = value
    return value


def curve_quality(path, start=None, samples=121, end=None, phase=None, _single=False):
    """Audit only the proposed suffix; past reversals cannot be repaired retroactively."""
    start = getattr(path, 'activation_s', 0.) if start is None else start
    original = getattr(path, 'original', path)
    from .combo import is_combo, phase_quality, phase_direction
    if not _single and is_combo(original):
        return phase_quality(path, start, end)
    times = np.linspace(start, path.duration if end is None else end, samples)
    from .intent_paths import positions_at
    points=positions_at(path,times)
    original = getattr(path, 'original', path)
    motion = original.phase(start)['motion'].lower()
    delta = np.diff(points, axis=0)
    if motion in ('movein', 'moveout'):
        feature = np.linalg.norm(points-original.subject, axis=1)
        sign = -1. if motion == 'movein' else 1.
    elif motion == 'orbit':
        feature = np.unwrap(np.arctan2(points[:,1]-original.subject[1],points[:,0]-original.subject[0]))
        ref = positions_at(original,times)
        angles = np.unwrap(np.arctan2(ref[:,1]-original.subject[1],ref[:,0]-original.subject[0]))
        sign = np.sign(angles[-1]-angles[0])
    else:
        axis = original.sample(0.)[1][:,1] if 'truck' in motion else np.array([0.,0.,1.])
        feature = points@axis
        sign = np.sign((original.sample(original.duration)[0]-original.sample(0.)[0])@axis)
    reverse = float(np.maximum(0., -sign*np.diff(feature)).sum())
    unit='rad' if motion=='orbit' else 'm'
    if phase is not None:
        directed, unit = phase_direction(original, phase, points)
        reverse=float(np.maximum(0.,-directed).sum())
    lengths = np.linalg.norm(delta, axis=1)
    # Curvature must not depend on speed or dwell time. Evaluate equal-distance
    # chords; the endpoint's near-zero velocities otherwise amplify harmless
    # submillimetre numerical noise into arbitrarily large turning angles.
    arc=np.r_[0.,np.cumsum(lengths)]
    keep=np.r_[True,np.diff(arc)>1e-9]
    uniform=np.linspace(0.,arc[-1],81)
    regular=np.column_stack([np.interp(uniform,arc[keep],points[keep,j]) for j in range(3)])
    chords=np.diff(regular,axis=0);chord_lengths=np.linalg.norm(chords,axis=1)
    moving=chord_lengths>1e-5
    tangents=chords[moving]/chord_lengths[moving,None]
    turns = np.arccos(np.clip(np.sum(tangents[1:]*tangents[:-1],axis=1),-1,1))
    # A straight or single broad arc has low excess turning. Slow stop samples
    # are omitted so numerical tangents at zero speed cannot dominate.
    total_turn = float(turns.sum())
    chord = float(np.linalg.norm(points[-1]-points[0]))
    # A measured nonzero incoming tangent requires a short C2 transition into
    # a new curve. Retain raw turning diagnostics; bound the excluded join to
    # 20 mm of travel and 0.9 rad, never omit a mid-shot bend or any reversal.
    tangent_distance=((uniform[:-1]+uniform[1:])/2)[moving]
    turn_distance=(tangent_distance[:-1]+tangent_distance[1:])/2
    join=turn_distance<.02
    core=turns[~join];join_turn=float(turns[join].sum())
    regular=bool(float(core.sum())<3.5 and float(max(core,default=0.))<.45 and join_turn<.9)
    return dict(reverse_motion=reverse, reverse_unit=unit,
                direction_pass=reverse <= (.003 if unit=='rad' else .004),
                length_m=float(lengths.sum()), chord_m=chord,
                total_turn_rad=total_turn, max_step_turn_rad=float(max(turns,default=0.)),
                regular_curve_pass=regular, core_turn_rad=float(core.sum()),core_max_step_turn_rad=float(max(core,default=0.)),
                initial_join_turn_rad=join_turn,initial_join_limit_m=.02,
                strict_full_curve_pass=total_turn < 3.5 and float(max(turns,default=0.)) < .45)


def regularize_lift(seed, report, scene):
    """Fit a robot proposal with two global shape modes, then require a new lift.

    This never certifies a smoothed trajectory by inheriting its source's
    collision result. The caller must run the complete strict numerical gate.
    """
    from scipy.optimize import minimize
    from .intent_paths import IntentPath, positions_at
    from .camera_intent import obstacle_state
    trace=report['trace']
    source_times=np.array([row['t'] for row in trace])
    source_positions=np.array([row['position'] for row in trace])
    band=terminal_band(seed.original,seed.model,seed.intrinsics)
    terminal=np.array(band['reference_position'])
    actual_end=np.array([np.interp(seed.duration,source_times,source_positions[:,j]) for j in range(3)])
    offsets=np.array(band['offsets_m'])
    endpoint=offsets[np.argmin(np.linalg.norm(terminal+offsets-actual_end,axis=1))]
    program=dict(seed.program,route=seed.program.get('seed_route',seed.program['route']),
                 regular_curve=True,endpoint_offset_m=endpoint.tolist(),
                 projection_fit='two_global_modes_new_strict_lift_required')
    if program['route']=='continuation':program['route']='timing'
    base=IntentPath(seed.original,seed.previous,seed.activation_s,program,seed.model,seed.intrinsics)
    times=np.linspace(seed.activation_s,seed.duration,81)
    u=(times-seed.activation_s)/(seed.duration-seed.activation_s)
    b=64*u**3*(1-u)**3
    basis=np.column_stack([b,b*(2*u-1)])
    nominal=positions_at(base,times)
    target=np.column_stack([np.interp(times,source_times,source_positions[:,j]) for j in range(3)])
    coefficients=np.linalg.lstsq(basis,target-nominal,rcond=None)[0]
    centers,radii=[],[]
    for t in times:
        c,r=obstacle_state(scene,t);centers.append(c);radii.append(r)
    centers=np.asarray(centers);radii=np.asarray(radii)
    # Only nearby observed proxies enter this bounded fit. The complete scene
    # still enters the subsequent collision and whole-body verification.
    near=np.min(np.linalg.norm(nominal[:,None]-centers,axis=2)-radii,axis=0)<1.
    centers=centers[:,near];radii=radii[:,near]
    motion=seed.original.phase(seed.activation_s)['motion'].lower()
    if motion=='orbit':
        tangent=nominal-seed.subject
        tangent=np.column_stack([-tangent[:,1],tangent[:,0],np.zeros(len(times))])
        tangent*=np.sign(np.sum(np.diff(nominal,axis=0)*tangent[:-1]))
    elif motion in ('movein','moveout'):
        tangent=(seed.subject-nominal)*(1 if motion=='movein' else -1)
    else:
        axis=seed.original.sample(0)[1][:,1] if 'truck' in motion else np.array([0.,0.,1.])
        axis*=np.sign((terminal-seed.original.sample(0)[0])@axis)
        tangent=np.tile(axis,(len(times),1))
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-8)
    def objective(x):
        points=nominal+basis@x.reshape(2,3)
        return float(np.mean((points-target)**2)+.03*np.sum(x*x)+
                     5.*np.mean(np.diff(points,n=2,axis=0)**2))
    def constraints(x):
        points=nominal+basis@x.reshape(2,3)
        direction=np.sum(np.diff(points,axis=0)*tangent[:-1],axis=1)+1e-6
        clear=np.linalg.norm(points[:,None]-centers,axis=2)-radii-.095
        return np.r_[direction,clear.ravel()]
    fit=minimize(objective,np.clip(coefficients.ravel(),-.6,.6),method='SLSQP',
                 bounds=[(-.6,.6)]*6,constraints={'type':'ineq','fun':constraints},
                 options={'maxiter':50,'ftol':1e-8})
    if not fit.success or np.min(constraints(fit.x)) < -1e-5:
        return None
    positions=nominal+basis@fit.x.reshape(2,3)
    geometry=dict(times=times.tolist(),positions=positions.tolist(),
                  start_velocity=base.geometry['start_velocity'],
                  start_acceleration=base.geometry['start_acceleration'],
                  joint_seed=trace, joint_seed_preference_only=bool(np.max(np.linalg.norm(positions-target,axis=1))>.008))
    return IntentPath(seed.original,seed.previous,seed.activation_s,program,
                      seed.model,seed.intrinsics,geometry=geometry)


def smooth_lift(seed, report):
    """Remove small solver wiggles, then require a new strict body lift.

    Bound the fitting displacement to 10 mm; this is a proposal construction
    bound, never a replacement for camera, actuator or collision verification.
    """
    from scipy.interpolate import UnivariateSpline
    from .intent_paths import IntentPath
    trace=report['trace'];ts=np.array([r['t'] for r in trace])
    xyz=np.array([r['position'] for r in trace])
    times=np.linspace(seed.activation_s,seed.duration,121)
    target=np.column_stack([np.interp(times,ts,xyz[:,j]) for j in range(3)])
    band=terminal_band(seed.original,seed.model,seed.intrinsics)
    end=np.array(band['reference_position']);offsets=np.array(band['offsets_m'])
    endpoint=end+offsets[np.argmin(np.linalg.norm(end+offsets-target[-1],axis=1))]
    if np.linalg.norm(endpoint-target[-1])>.01:return None
    p0,_,v0=seed.previous.sample(seed.activation_s);h=.001
    a0=(seed.previous.sample(seed.activation_s+h)[2][:3]-seed.previous.sample(seed.activation_s-h)[2][:3])/(2*h)
    span=seed.duration-seed.activation_s;u=(times-seed.activation_s)/span
    s=10*u**3-15*u**4+6*u**5
    join_span=min(.6,span/3)
    w=np.clip((times-seed.activation_s)/join_span,0.,1.)
    join=10*w**3-15*w**4+6*w**5
    for tolerance in (.00025,.001,.002,.004,.006):
        splines=[UnivariateSpline(times,target[:,j],k=3,s=len(times)*tolerance**2) for j in range(3)]
        points=np.column_stack([f(times) for f in splines])
        d=p0-points[0];v=join_span*(v0[:3]-np.array([f(times[0],1) for f in splines]))
        a=join_span**2*(a0-np.array([f(times[0],2) for f in splines]))
        points+=(1-join)[:,None]*d+(w-6*w**3+8*w**4-3*w**5)[:,None]*v
        points+=.5*(w**2-3*w**3+3*w**4-w**5)[:,None]*a+s[:,None]*(endpoint-points[-1])
        if np.max(np.linalg.norm(points-target,axis=1))>.01:continue
        program=dict(seed.program,route=seed.program.get('seed_route',seed.program['route']),
            regular_curve=True,endpoint_offset_m=(endpoint-end).tolist(),projection_fit='bounded_smoothing_new_strict_lift',
            smoothing_tolerance_m=tolerance)
        if program['route']=='continuation':program['route']='timing'
        geometry=dict(times=times.tolist(),positions=points.tolist(),start_velocity=v0[:3].tolist(),
            start_acceleration=a0.tolist(),joint_seed=trace,joint_seed_preference_only=False)
        path=IntentPath(seed.original,seed.previous,seed.activation_s,program,seed.model,seed.intrinsics,geometry=geometry)
        quality=curve_quality(path)
        if quality['direction_pass'] and quality['regular_curve_pass']:return path
    return None
