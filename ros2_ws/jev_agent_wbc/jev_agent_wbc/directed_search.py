"""Jev chooses a construction contract BEFORE any candidate is generated.

The first answer restricts the generator; the second fixes its only seed and
allocation. Scalar geometry probes describe observed space, not screened paths.
There is no numerical ranking, alternate seed or retry on a planning failure.
"""
import numpy as np

FAMILIES = {
    'joint_curve': 'One broad horizontal detour, jointly optimize camera and both robot parts. Planar displacement can be supplied by the mobile platform without spending the manipulator vertical workspace. Prefer this for an open lateral or radial bypass.',
    'joint_vertical': 'One vertical detour, jointly optimize camera and both robot parts. Use when planar bypass space is obstructed and vertical displacement is justified despite the limited manipulator height and wrist reserve.',
    'timed_curve': 'One fixed line/arch with a monotone obstacle-aware clock. Choose when the spatial route is reachable and passing time is the main problem.',
    'regular_arch': 'One small early spatial arch with slower progress and an endpoint in the original image tolerance band. Choose when a compact lateral repair has enough clearance.',
}


def axes(original, active, activation):
    p = active.sample(activation)[0]
    radial = p-original.subject
    radial[2] = 0.
    radial /= max(np.linalg.norm(radial), 1e-8)
    left = np.cross(radial, [0., 0., 1.])
    return dict(adaptive=np.zeros(3), left=left, right=-left,
                above=np.array([0.,0.,1.]), below=np.array([0.,0.,-1.]),
                outer=radial, inner=-radial)


def construction_evidence(original, active, scene, t, activation,mount_height_m=None):
    """Only the immutable reference and timestamped observed map; no truth."""
    from .task_space_planning import camera_scene
    from .camera_intent import obstacle_state
    observed = camera_scene(scene,t,active.duration)
    times = np.linspace(activation, active.duration, 9)
    rows=[]
    directions=axes(original,active,activation)
    chord=original.sample(original.duration)[0]-active.sample(activation)[0]
    chord/=max(np.linalg.norm(chord),1e-8)
    motion=original.phase(activation)['motion'].lower()
    primary=np.array([0.,0.,1.]) if 'pedestal' in motion else chord
    allowed=[name for name,direction in directions.items()
             if abs(direction@chord)<.8 and abs(direction@primary)<.8]
    from .combo import is_combo
    if is_combo(original):
        phase=original.phase(activation)
        if motion in ('movein','moveout'):
            primary=active.sample(activation)[0]-original.subject
        elif 'truck' in motion:
            primary=original.sample(phase['start_s'])[1][:,1]
        elif motion=='orbit':
            radial=active.sample(activation)[0]-original.subject
            primary=np.array([-radial[1],radial[0],0.])
        elif motion=='transition':
            primary=original.sample(phase['end_s'])[0]-original.sample(phase['start_s'])[0]
        primary=primary/max(np.linalg.norm(primary),1e-8)
        allowed=[name for name,direction in directions.items() if abs(direction@primary)<.8]
    probes={k:[] for k in directions if k!='adaptive'}
    broad_probes={k:[] for k in directions if k!='adaptive'}
    for at in times:
        p=active.sample(float(at))[0]
        centers,radii=obstacle_state(observed,float(at))
        rows.append(dict(t=float(at), camera_position=p.tolist(),
            nearest_surface_clearance_m=float(np.min(np.linalg.norm(centers-p,axis=1)-radii)) if len(centers) else None))
        for name,direction in directions.items():
            if name!='adaptive' and len(centers):
                probes[name].append(float(np.min(np.linalg.norm(centers-(p+.25*direction),axis=1)-radii)))
                broad_probes[name].append(float(np.min(np.linalg.norm(centers-(p+.5*direction),axis=1)-radii)))
    if any(row['nearest_surface_clearance_m'] is not None and row['nearest_surface_clearance_m']<.14 for row in rows):
        allowed=[name for name in allowed if name!='adaptive']
    workspace={}
    if mount_height_m is not None:
        from .jvs_kinematics import JOINT_ORIGINS,CAMERA_OFFSET
        reach=float(np.linalg.norm(JOINT_ORIGINS[1:],axis=1).sum()+np.linalg.norm(CAMERA_OFFSET))
        upper=float(mount_height_m+reach)
        peak=active.sample(float((activation+active.duration)/2))[0][2]+.3
        workspace=dict(camera_height_upper_bound_m=upper,vertical_seed_midpoint_height_m=float(peak),
            upper_bound_source='measured mount height + URDF chain-length bound; joint limits and orientation can only reduce reach',
            scope='necessary geometric seed bound; not an IK or collision certificate')
        if peak>upper:allowed=[name for name in allowed if name!='above']
    return dict(protocol='single_candidate/v1', activation_s=activation,
        combo_fixed_phase_clock=is_combo(original),shot_phase=original.task['shot_phase'],
        original_clock_end_s=original.duration, reference_samples=rows,
        axes_world={k:v.tolist() for k,v in directions.items()},
        allowed_directions=allowed,
        manipulator_workspace=workspace,
        direction_domain=('Exclude directions predominantly parallel to the current main action; preserve the entire phase sequence.'
            if is_combo(original) else 'Exclude detours predominantly parallel to the primary or companion motion; no candidate was generated to make this restriction.'),
        displaced_point_probes={k:dict(min_surface_distance_m=min(v) if v else None,
            displacement_m=.25,scope='isolated offset points; not a generated path or body certificate') for k,v in probes.items()},
        broad_displaced_point_probes={k:dict(min_surface_distance_m=min(v) if v else None,
            displacement_m=.5,scope='isolated offset points; not a path, timing or reachability certificate') for k,v in broad_probes.items()},
        scope='observed linear obstacle forecast; no candidate generated or screened',
        required='exactly one construction, one continuous allocation check; no fallback candidates')


def options(family,combo=False):
    if family not in FAMILIES:
        raise ValueError('unknown directed generator')
    directions = (('adaptive','left','right','outer') if family=='joint_curve' else
                  ('above','below') if family=='joint_vertical' else ('left','right','above','below','adaptive'))
    return {name: dict(family=family,direction=name,amplitude_m=0. if name=='adaptive' else (.3 if family.startswith('joint_') else .2),
                      plane_tolerance_m=.025 if family.startswith('joint_') else 0.,
                      # Delay the middle of the motion enough to preserve
                      # acceleration reserve while the redundant posture opens.
                      tempo=1. if combo else 4.,allocation='anticipatory',search_config='base') for name in directions}


def generate(original, active, scene, activation, model, intrinsics, budget, plan,initial_boundary=None):
    from .combo import is_combo,combo_curves
    family=plan['family'];direction=plan['direction']
    allowed=options(family,combo=is_combo(original))
    if direction not in allowed or plan != allowed[direction]:
        raise ValueError('construction contract outside typed choices')
    d=axes(original,active,activation)[direction]
    if is_combo(original):
        if family not in ('joint_curve','joint_vertical'):
            raise ValueError('combo currently requires phase-preserving joint construction')
        paths=combo_curves(original,active,activation,model,intrinsics,1,d,
            plan['amplitude_m'],initial_boundary,plan['tempo'])
        for path in paths:
            path.program['construction_contract']=dict(plan)
            if np.any(d) and family.startswith('joint_'):
                chord=original.sample(original.duration)[0]-active.sample(activation)[0]
                normal=np.array([0.,0.,1.]) if family=='joint_curve' else np.cross(d,chord)
                normal/=max(np.linalg.norm(normal),1e-8)
                path.program.update(construction_axis_world=d.tolist(),
                    construction_plane_normal_world=normal.tolist(),construction_plane_tolerance_m=plan['plane_tolerance_m'])
        return paths
    if initial_boundary is not None and activation!=0.:
        raise ValueError('initial boundary requires the unstarted shot')
    if family in ('joint_curve','joint_vertical'):
        from .regular_curve_search import global_curves
        construction_axis=None
        if np.any(d):
            chord=original.sample(original.duration)[0]-active.sample(activation)[0]
            if family=='joint_curve':chord[2]=0.
            chord/=max(np.linalg.norm(chord),1e-8)
            construction_axis=d-chord*(d@chord)
            construction_axis/=max(np.linalg.norm(construction_axis),1e-8)
        paths=global_curves(original,active,scene,activation,model,intrinsics,count=1,
            seconds=min(2.,budget.candidate_wall_budget_s*.18),tempo=plan['tempo'],seed_direction=d,
            seed_amplitude=plan['amplitude_m'],initial_boundary=initial_boundary,construction_axis=construction_axis,
            plane_tolerance=plan['plane_tolerance_m'],fit_seed=False)
        for path in paths:
            if construction_axis is not None:
                chord=original.sample(original.duration)[0]-np.asarray(path.geometry['positions'][0])
                normal=np.cross(construction_axis,chord);normal/=max(np.linalg.norm(normal),1e-8)
                path.program['construction_axis_world']=construction_axis.tolist()
                path.program['construction_plane_normal_world']=normal.tolist()
                path.program['construction_plane_tolerance_m']=plan['plane_tolerance_m']
    elif family=='timed_curve':
        from .curve_timing import timed_curves
        paths=timed_curves(original,active,scene,activation,model,intrinsics,count=1,
            displacement=plan['amplitude_m']*d,initial_boundary=initial_boundary)
    else:
        from .intent_paths import IntentPath
        from .cinematic_geometry import terminal_band
        offsets=np.asarray(terminal_band(original,model,intrinsics)['offsets_m'])
        endpoint=offsets[np.argmax(offsets@d)] if np.any(d) else np.zeros(3)
        program=dict(route=direction if direction!='adaptive' else 'timing',
            amplitude_m=plan['amplitude_m'],extension_s=0.,profile='arch',phase_rate=-.7,
            regular_curve=True,shape_skew=.2,endpoint_offset_m=endpoint.tolist(),
            endpoint_region='26_ray_bisection_rechecked_endpoints',search_role='coordinated_clearance')
        paths=[IntentPath(original,active,activation,program,model,intrinsics)]
        if initial_boundary is not None:
            from .reference import smoothstep
            path=paths[0];geometry=dict(path.geometry)
            times=np.asarray(geometry['times'])
            offset=np.asarray(initial_boundary['position'])-original.sample(0.)[0]
            geometry['positions']=(np.asarray(geometry['positions'])+(1.-smoothstep(times/path.duration))[:,None]*offset).tolist()
            geometry['start_velocity']=initial_boundary['twist'][:3]
            geometry['start_acceleration']=[0.,0.,0.]
            geometry['initial_rotation_xyzw']=initial_boundary['quaternion_xyzw']
            paths=[IntentPath(original,active,0.,dict(program,measured_initial_boundary=True),model,intrinsics,geometry=geometry)]
    for path in paths:
        path.program['construction_contract']=dict(plan)
    return paths
