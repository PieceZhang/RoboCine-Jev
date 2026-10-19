"""Typed Jev programs for ordered, stopped camera actions.

The source task, stage order and deadlines are immutable. Optional single broad
bows and timing profiles require numerical screening; optimized endpoints stay
inside the existing final-frame band. Early completion holds the endpoint;
obstacle time never pauses. A proposal is not an execution certificate.
"""
import numpy as np

from .contracts import digest
from .intent_paths import IntentPath,positions_at
from .jev import choice


RATES={'original':1.,'moderate_early':1.6,'early':2.5}
SHAPES=('original','positive_small','negative_small','above_small','below_small',
        'positive_broad','negative_broad','above_broad','left_small','right_small','left_broad','right_broad',
        'crossing_left','crossing_right')
# Retained for controlled geometry diagnostics; tight semicircles have not
# demonstrated stable mixed-scene execution and are not offered to the Agent.
DIAGNOSTIC_SHAPES=('positive_round','negative_round')


def camera_corridor_space(path,scene):
    """Observed approach-line separation, a preference rather than a forecast.

    Static/unconfirmed objects get an explicitly hypothetical direct approach
    probe. No velocity is inferred or published as a measured fact.
    """
    first=path.sample(0.)[0]
    points=positions_at(path,np.linspace(.3*path.duration,.7*path.duration,17))
    rows=[]
    for obj in scene.get('object_observations',[]):
        center=np.asarray(obj['center']);extent=np.asarray(obj['half_extent'])
        if not obj.get('moving_confirmed') and any(other.get('moving_confirmed') and
            np.linalg.norm(np.asarray(other['center'])-center)<.05 and
            np.linalg.norm(np.asarray(other['half_extent'])-extent)<.03
            for other in scene.get('object_observations',[])):
            # Do not count an older unconfirmed segmentation of the same
            # occupancy as a second hypothetical mover. Raw map remains intact.
            continue
        if center[2]+extent[2]<min(first[2],points[:,2].min())-.10 or center[2]-extent[2]>max(first[2],points[:,2].max())+.10:continue
        toward=first-center
        if np.linalg.norm(toward)>2.:continue
        hint=obj.get('motion_hint')
        velocity=np.asarray(hint['velocity_estimate'] if hint else obj['velocity'])
        known=bool((obj.get('moving_confirmed') or hint) and np.linalg.norm(velocity)>.04)
        if known and velocity@toward<=0.:continue
        direction=velocity if known else toward
        if np.linalg.norm(direction)<1e-6:continue
        direction=direction/np.linalg.norm(direction)
        relative=points-center
        transverse=relative-(relative@direction)[:,None]*direction
        clearance=np.linalg.norm(transverse,axis=1)-max(extent)-.06
        rows.append(dict(object_id=obj.get('id'),
            source=('observed_motion_line' if obj.get('moving_confirmed') else 'unconfirmed_measured_displacement_line') if known else 'hypothetical_direct_approach_not_observed_motion',
            middle_phase_clearance_lower_m=round(float(np.quantile(clearance,.1)),3)))
    return dict(probes=rows,scope='Transverse geometric yielding space only; no arrival time or future collision certificate')


def configuration_space(report,base_z):
    """Static camera-holding posture probes; not nonholonomic path proofs."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from .whole_body_qp import whole_body_kinematics
    from .jvs_kinematics import JOINT_LOWER,JOINT_UPPER
    x=np.asarray(report['trace'][len(report['trace'])//2]['state'])
    target,rotation,_=whole_body_kinematics(x,base_z)
    rows=[]
    for offset in ([.2,0.],[-.2,0.],[0.,.2],[0.,-.2]):
        state=x.copy();state[:2]+=offset
        def residual(q):
            state[3:]=q;p,r,_=whole_body_kinematics(state,base_z)
            return np.r_[p-target,.35*Rotation.from_matrix(r@rotation.T).as_rotvec()]
        solved=least_squares(residual,np.clip(x[3:],JOINT_LOWER+.02,JOINT_UPPER-.02),
            bounds=(JOINT_LOWER+.02,JOINT_UPPER-.02),max_nfev=40,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        error=residual(solved.x);pe=float(np.linalg.norm(error[:3]));re=float(np.linalg.norm(error[3:])/.35)
        rows.append(dict(platform_offset_world_m=offset,position_error_m=round(pe,4),
            orientation_error_rad=round(re,4),camera_hold_found=bool(pe<.015 and re<.035)))
    return dict(probes=rows,scope='Mid-action static arm compensation only; failed local IK is not global infeasibility. No drive, actuator-time or collision certificate.')


def robust_camera_profiles(cards):
    """Retain comparable useful crossing reserves after full numerical gates."""
    scores={}
    for key,card in cards.items():
        rows=(card.get('camera_crossing_space') or {}).get('probes',[])
        measured=[r for r in rows if r['source'] in ('observed_motion_line','unconfirmed_measured_displacement_line')]
        relevant=measured or rows
        if relevant and card['kinematics']['min_joint_margin_rad']>=.10:
            scores[key]=min(r['middle_phase_clearance_lower_m'] for r in relevant)
    if not scores or max(scores.values())<=.025:return cards,None
    best=max(scores.values());keep={key:value for key,value in cards.items() if scores.get(key,-np.inf)>=best-.03}
    return keep,dict(best_transverse_reserve_m=best,comparison_band_m=.03,
        offered=list(keep),excluded=[key for key in cards if key not in keep],
        scope='Geometric robustness preference only, not a future safety certificate; all execution checks remain mandatory')


def rounded_pedestal(first,last,rotation,u,sign):
    """Two tangent legs and a rounded turn with monotone vertical revelation."""
    delta=last-first;rise=abs(delta[2])
    if rise<.04:return None
    axis=delta.copy();axis[2]=0.
    if np.linalg.norm(axis)<.01:axis=rotation[:,0].copy();axis[2]=0.
    axis/=np.linalg.norm(axis)
    displacement=delta@axis;radius=rise/2
    center=(max(0.,displacement)+.30-radius if sign>0 else min(0.,displacement)-.30+radius)
    first_length=abs(center);last_length=abs(displacement-center)
    arc=np.pi*radius;length=first_length+arc+last_length
    grid=np.linspace(0.,length,4001)
    # Spend time through the turn instead of pinching it into a few samples.
    weight=1.+3.*np.exp(-((grid-first_length-arc/2)/(arc*.8))**4)
    clock=np.r_[0.,np.cumsum(.5*(weight[:-1]+weight[1:])*np.diff(grid))];clock/=clock[-1]
    smooth=10*u**3-15*u**4+6*u**5
    distance=np.interp(smooth,clock,grid)
    horizontal=np.empty(len(u));vertical=np.empty(len(u))
    incoming=distance<first_length;outgoing=distance>first_length+arc;turn=~(incoming|outgoing)
    horizontal[incoming]=sign*distance[incoming];vertical[incoming]=0.
    angle=(distance[turn]-first_length)/radius
    horizontal[turn]=center+sign*radius*np.sin(angle)
    vertical[turn]=radius*(1.-np.cos(angle))
    horizontal[outgoing]=center-sign*(distance[outgoing]-first_length-arc);vertical[outgoing]=rise
    return first+horizontal[:,None]*axis+np.c_[np.zeros((len(u),2)),np.sign(delta[2])*vertical]


def build(original, selections, model=None, intrinsics=None, shapes=None, directions=None, amplitudes=None):
    phases=original.task['shot_phase']
    if any(value not in RATES for value in selections.values()) or any(value not in SHAPES+DIAGNOSTIC_SHAPES for value in (shapes or {}).values()):
        raise ValueError('unsupported phase program')
    times=np.unique(np.round(np.r_[np.arange(0.,original.duration,.025),original.duration,
        [p[k] for p in phases for k in ('start_s','end_s')],
        [p['start_s']+(p['end_s']-p['start_s'])/RATES[selections[p['name']]]
         for p in phases if p['motion'].lower()!='transition']],10))
    progress=times.copy()
    for phase in phases:
        if phase['motion'].lower()=='transition':continue
        a,b=phase['start_s'],phase['end_s'];rate=RATES[selections[phase['name']]]
        mask=(times>=a)&(times<=b)
        progress[mask]=np.minimum(b,a+rate*(times[mask]-a))
    positions=positions_at(original,progress)
    shapes=shapes or {}
    for phase in phases:
        selected=shapes.get(phase['name'],'original')
        if selected=='original':continue
        a,b=phase['start_s'],phase['end_s'];mask=(times>=a)&(times<=b)
        first,rotation,_=original.sample(a);last=original.sample(b)[0]
        if selected.endswith('round'):
            if 'pedestal' not in phase['motion'].lower():raise ValueError('rounded turn requires a pedestal action')
            u=np.clip((progress[mask]-a)/(b-a),0.,1.)
            rounded=rounded_pedestal(first,last,rotation,u,1. if selected.startswith('positive') else -1.)
            if rounded is None:raise ValueError('insufficient vertical room for a rounded turn')
            positions[mask]=rounded
            continue
        chord=last-first;axis=chord/max(np.linalg.norm(chord),1e-9)
        direction=rotation[:,0]-axis*(axis@rotation[:,0])
        if selected.startswith(('left','right')):
            direction=rotation[:,1]-axis*(axis@rotation[:,1])
        if 'pedestal' in phase['motion'].lower() and not selected.startswith(('above','below')):
            # A pedestal action may include a forward companion displacement.
            # Projecting its bow off that slanted chord adds a vertical reversal.
            # Use a horizontal bow so its directed vertical revelation survives.
            direction=rotation[:,1].copy() if selected.startswith(('left','right')) else rotation[:,0].copy()
            direction[2]=0.
        if selected.startswith(('above','below')):
            direction=np.array([0.,0.,1.])-axis*axis[2]
        if np.linalg.norm(direction)<.1:
            candidates=np.vstack([rotation[:,1],np.eye(3)])
            perpendicular=candidates-(candidates@axis)[:,None]*axis
            direction=perpendicular[np.argmax(np.linalg.norm(perpendicular,axis=1))]
        direction/=np.linalg.norm(direction)
        if selected.startswith(('negative','below','right')):direction=-direction
        if selected.startswith('crossing_'):
            direction=np.asarray((directions or {})[phase['name']],float)
            if direction.shape!=(3,) or not np.isfinite(direction).all() or abs(direction[2])>1e-8 or np.linalg.norm(direction)<1e-6:
                raise ValueError('crossing bow requires a finite horizontal observed direction')
            direction=direction/np.linalg.norm(direction)
            if selected=='crossing_right':direction=-direction
        amplitude=.15 if selected.endswith('small') else .30
        if selected.startswith('crossing_'):
            amplitude=float((amplitudes or {}).get(phase['name'],.30))
            if not .05<=amplitude<=.40:raise ValueError('invalid observed crossing amplitude')
        u=np.clip((progress[mask]-a)/(b-a),0.,1.)
        positions[mask]+=amplitude*(64*u**3*(1-u)**3)[:,None]*direction
    geometry=dict(times=times.tolist(),positions=positions.tolist(),
        progress_times=progress.tolist(),start_velocity=[0.,0.,0.],start_acceleration=[0.,0.,0.])
    if len(phases)==1:
        phase=phases[0]
        u=np.clip((progress-phase['start_s'])/(phase['end_s']-phase['start_s']),0.,1.)
        b=64*u**3*(1-u)**3;s=10*u**3-15*u**4+6*u**5
        # Global endpoint refinement must finish with the action's progress.
        # Using elapsed time would create a second drift after an early stop.
        geometry['shape_basis']=np.column_stack([b,b*(2*u-1),s]).tolist()
    selected_directions={name:np.asarray(value,float).tolist() for name,value in (directions or {}).items()
        if shapes.get(name,'').startswith('crossing_')}
    return IntentPath(original,original,0.,dict(route='timing',amplitude_m=0.,extension_s=0.,
        regular_curve=True,curve_family='phase_program',phase_timing=dict(selections),phase_shapes=shapes,
        observed_shape_directions=selected_directions,observed_shape_amplitudes=dict(amplitudes or {})),
        model,intrinsics,geometry=geometry)


def prepare(engine, scene, state, base_z,initial_boundary=None):
    import time
    started=time.monotonic()
    if engine.mode!='jev' or engine.jev.decider.client.url!='https://api.typesafe.ai/v1/systemone':
        raise ValueError('phase programs require the official Jev API')
    original=engine.original
    def make_candidate(*args,**kwargs):
        path=build(*args,**kwargs)
        if initial_boundary is None:return path
        from .reference import smoothstep
        from scipy.spatial.transform import Rotation
        from .whole_body_qp import whole_body_kinematics
        p,r,_=whole_body_kinematics(initial_boundary,base_z)
        geometry=dict(path.geometry);times=np.asarray(geometry['times']);u=times/path.duration
        positions=np.asarray(geometry['positions'])+(1.-smoothstep(u))[:,None]*(p-path.sample(0.)[0])
        geometry.update(positions=positions.tolist(),initial_rotation_xyzw=Rotation.from_matrix(r).as_quat().tolist())
        return IntentPath(path.original,path.previous,0.,dict(path.program,measured_initial_boundary=True),
            path.model,path.intrinsics,geometry=geometry)
    phases=[p for p in original.task['shot_phase'] if p['motion'].lower()!='transition']
    observation=engine.geometry_backend.observe(state,0.,base_z=base_z)
    evidence=dict(stage='body_aware_phase_timing',intent=original.task['intent'],
        robot_geometry=observation.summary(engine.geometry_backend.obstacle_relations(observation,scene)),
        observed_tracks=scene.get('tracks',[]),observed_objects=scene.get('object_observations',[]),observation_stamp=scene['observation_stamp'],
        phases=[dict(p,start_position=original.sample(p['start_s'])[0].tolist(),
            end_position=original.sample(p['end_s'])[0].tolist()) for p in phases],
        contract='Preserve every directed action, endpoint, phase order and deadline. Advance within a phase and then hold. Low obstacles require platform clearance, middle obstacles whole-arm clearance, high obstacles camera clearance. No future obstacle motion is known. Select early passage when the slow start would spend chassis escape or arm compensation reserve. A straight transverse camera move can exhaust arm folding when the chassis moves forward to escape: a single broad bow permits chassis travel while returning to the original endpoint. Physical checks remain mandatory.')
    questions={p['name']:choice('Choose the within-action timing using the observed whole-body geometry and action direction.',{
        'original':'Retain authored slow start and timing.',
        'moderate_early':'Complete the same spatial action in 62.5% of its interval, then hold.',
        'early':'Complete the same spatial action in 40% of its interval, then hold; prepare platform clearance early.'}) for p in phases}
    for p in phases:
        first,rotation,_=original.sample(p['start_s']);last=original.sample(p['end_s'])[0]
        axis=last-first;axis/=max(np.linalg.norm(axis),1e-9)
        direction=rotation[:,0]-axis*(axis@rotation[:,0])
        if np.linalg.norm(direction)<.1:direction=rotation[:,1]-axis*(axis@rotation[:,1])
        direction/=np.linalg.norm(direction)
        questions[p['name']+'_shape']=choice('Choose one smooth spatial bow for whole-body clearance. Positive world direction is '+str(direction.tolist())+'. Endpoints stay unchanged.',{
            'original':'Keep the original spatial path when body posture and obstacle clearance permit.',
            'positive_small':'One smooth 0.15 m bow in the positive direction.',
            'positive_broad':'One smooth 0.30 m bow in the positive direction for platform and arm clearance.',
            'negative_small':'One smooth 0.15 m bow opposite the positive direction.',
            'negative_broad':'One smooth 0.30 m bow opposite the positive direction.'})
    verified={}
    if len(phases)==1:
        from .kinematic_feasibility import KinematicSearch, summary
        from .cinematic_geometry import curve_quality
        search=KinematicSearch(state,np.zeros(8),0.,base_z,engine.wbc.limits,scene=scene)
        geometric_search=KinematicSearch(state,np.zeros(8),0.,base_z,engine.wbc.limits)
        phase=phases[0];name=phase['name'];cards={}
        moving_observed=any(np.linalg.norm(row['velocity'])>.04 for row in scene.get('tracks',[]))
        rates=RATES if moving_observed else {'original':1.}
        from .dynamic_collision import DynamicCollision
        frozen_occupancy=DynamicCollision(scene,scene['observation_stamp'])
        crossing_direction=None
        first=original.sample(phase['start_s'])[0]
        high=[row for row in scene.get('object_observations',[])
              if abs(row['center'][2]-first[2])<row['half_extent'][2]+.15]
        if high:
            nearest=min(high,key=lambda row:(not row['moving_confirmed'] and not row.get('motion_hint'),np.linalg.norm(np.asarray(row['center'])-first)))
            axis=np.asarray(nearest['velocity'] if nearest['moving_confirmed'] else
                nearest['motion_hint']['velocity_estimate'] if nearest.get('motion_hint') else np.asarray(nearest['center'])-first)
            if np.linalg.norm(axis[:2])>.04:
                crossing_direction=np.array([-axis[1],axis[0],0.]);crossing_direction/=np.linalg.norm(crossing_direction)
        for rate in rates:
            for shape in SHAPES:
                if shape.endswith('round') and 'pedestal' not in phase['motion'].lower():continue
                if shape.startswith('crossing_') and (crossing_direction is None or
                        ('pedestal' not in phase['motion'].lower() and phase['motion'].lower()!='orbit')):continue
                candidate=make_candidate(original,{name:rate},engine.subject_model,engine.intrinsics,{name:shape},
                    directions={name:crossing_direction} if crossing_direction is not None else None)
                quality=curve_quality(candidate)
                if shape.startswith('crossing_') and not quality['strict_full_curve_pass']:
                    for amplitude in (.27,.24,.21,.18):
                        candidate=make_candidate(original,{name:rate},engine.subject_model,engine.intrinsics,{name:shape},
                            directions={name:crossing_direction},amplitudes={name:amplitude})
                        quality=curve_quality(candidate)
                        if quality['direction_pass'] and quality['strict_full_curve_pass']:break
                if not quality['direction_pass'] or not quality['strict_full_curve_pass']:continue
                report=geometric_search.lift(candidate,candidate.duration)
                if not report['feasible'] or (shape!='original' and report['min_joint_margin_rad']<.15):
                    from .joint_curve_optimizer import refine
                    candidate.program.update(fixed_spatial_shape=True,joint_reserve_rad=.12)
                    if 'pedestal' in phase['motion'].lower():candidate.program['fixed_camera_path']=True
                    pedestal='pedestal' in phase['motion'].lower()
                    fitted,refinement=refine(candidate,geometric_search,seconds=7. if pedestal else 3.,
                        max_iterations=40 if pedestal else 20,knots=49 if pedestal else 33)
                    if fitted is not None:
                        candidate=fitted
                        report=geometric_search.lift(candidate,candidate.duration)
                        report['global_configuration_refinement']=refinement
                if not report['feasible']:continue
                quality=curve_quality(candidate)
                if not quality['direction_pass'] or not quality['strict_full_curve_pass']:continue
                from .floating_camera import camera_motion_limits
                from .intent_paths import poses_at
                check_times=np.linspace(0.,candidate.duration,max(401,int(candidate.duration/.02)+1))
                cp,cr=poses_at(candidate,check_times)
                if not camera_motion_limits(list(zip(cp,cr)),check_times)['camera_limits_pass']:continue
                # Initial observations cannot certify an entire future shot.
                # Keep full FK/actuator screening, but give dynamic geometry a
                # bounded lease and require fresh execution certificates.
                safety=search.lift(candidate,min(candidate.duration,1.5))
                if not safety['feasible']:continue
                report['observed_safety_lease']=summary(safety)
                report['future_collision_safety']='unverified after the 1.5 s observation lease; mandatory fresh execution sweeps'
                key=rate+'__'+shape
                verified[key]=(candidate,report,rate,shape)
                probes=np.linspace(0.,candidate.duration,5)
                trace_times=np.array([row['t'] for row in report['trace']])
                trace_states=np.array([row['state'] for row in report['trace']])
                occupied_clearance=np.array([float(frozen_occupancy.clearance(trace_states[j],base_z).min())
                    for j in np.linspace(0,len(trace_states)-1,17).astype(int)])
                cards[key]=dict(timing=rate,shape=shape,
                    camera_crossing_space=(camera_corridor_space(candidate,scene)
                        if 'pedestal' in phase['motion'].lower() or phase['motion'].lower()=='orbit' else None),
                    platform_yielding_space=configuration_space(report,base_z),
                    camera_positions_world=positions_at(candidate,probes).round(3).tolist(),
                    platform_positions_world=np.column_stack([np.interp(probes,trace_times,trace_states[:,j]) for j in range(2)]).round(3).tolist(),
                    frozen_observed_occupancy=dict(min_clearance_m=round(float(occupied_clearance.min()),3),
                        lower_quartile_clearance_m=round(float(np.quantile(occupied_clearance,.25)),3),
                        mean_clearance_m=round(float(occupied_clearance.mean()),3),
                        scope='Current observed occupancy frozen in place; geometric space for yielding, not a future collision prediction'),
                    kinematics=dict(min_joint_margin_rad=round(report['min_joint_margin_rad'],2),
                        full_shot_fk_actuators_verified=True,safety_lease_s=safety['end_s'],
                        observed_clearance_lower_m=round(safety['min_body_clearance_m'],3)),curve_quality=quality)
        if not verified:raise RuntimeError('no_verified_phase_program')
        if 'pedestal' in phase['motion'].lower() or phase['motion'].lower()=='orbit':
            cards,robustness=robust_camera_profiles(cards)
            evidence['camera_robustness_screen']=robustness
        evidence['verified_profiles']=cards
        evidence['timing_evidence']='Early timing is offered only with confirmed motion evidence; otherwise keep the authored timing and reserve space for later observed hazards.'
        evidence['profile_scope']='Full-shot FK and actuator screening. Observed collision prediction only for the first 1.5 s, never a full-shot safety promise. Execution must continue sensing and certifying.'
        prompt=('Choose a continuous whole-body camera profile using actual world coordinates and observed obstacle heights. '
            'For low and middle obstacles, compare the static platform yielding probes: a large current clearance is insufficient if the arm cannot compensate platform displacement. '
            'A horizontal bow can let the chassis travel while retaining arm reach; raising the camera does not clear a low chassis obstacle. '
            'Prefer useful reserve in the likely escape direction, adequate joint margin, and the simplest compatible spatial path. Margins above 0.10 rad are adequate. '
            'Probes are search evidence, not certificates. Frozen occupancy is not future-motion evidence.')
        prompt+=' For camera-height objects in every action, prefer positive transverse crossing space: retreating along an approach line leaves the camera in its corridor. Unconfirmed approach probes are robustness hypotheses, not measured motion.'
        questions={'profile':choice(prompt,{key:str(dict(timing=v['timing'],shape=v['shape'],
            joint_margin_rad=v['kinematics']['min_joint_margin_rad'],camera_crossing_space=v['camera_crossing_space'],
            platform_yielding_space=v['platform_yielding_space'])) for key,v in cards.items()})}
    import threading
    client=engine.jev.decider.client
    numerical_wall=time.monotonic()-started
    api_started=time.monotonic()
    timer=threading.Timer(90.,lambda:setattr(client,'closed',True));timer.start()
    try:answers=client.ask(evidence,questions)
    finally:timer.cancel()
    api_wall=time.monotonic()-api_started
    if verified:
        selected,proof,rate,shape=verified[answers['profile']['choice']]
        selections={phases[0]['name']:rate};shapes={phases[0]['name']:shape}
    else:
        selections={p['name']:answers[p['name']]['choice'] for p in phases}
        shapes={p['name']:answers[p['name']+'_shape']['choice'] for p in phases}
    if any(s not in RATES for s in selections.values()):raise ValueError('invalid phase timing choice')
    path=selected if verified else make_candidate(original,selections,engine.subject_model,engine.intrinsics,shapes)
    from .cinematic_geometry import curve_quality
    from .floating_camera import camera_motion_limits
    from .intent_paths import poses_at
    times=np.linspace(0.,path.duration,max(401,int(path.duration/.02)+1))
    positions,rotations=poses_at(path,times)
    limits=camera_motion_limits(list(zip(positions,rotations)),times)
    quality=curve_quality(path)
    if not limits['camera_limits_pass'] or not quality['strict_full_curve_pass'] or not quality['direction_pass']:
        raise RuntimeError('phase_program_camera_geometry_rejected')
    engine.active=path;engine.wbc.set_trajectory(path)
    if verified:engine.wbc.kinematic_reference=proof
    engine.wbc.body_aware=True
    record=dict(selections=selections,shapes=shapes,selection_sha256=digest(dict(timing=selections,shapes=shapes)),geometry=path.geometry,
        startup_wall_s=time.monotonic()-started,numerical_screen_wall_s=numerical_wall,api_wall_s=api_wall,
        program=path.program,quality=quality,camera_limits=limits,
        kinematic_verification=None if not verified else summary(proof),
        scope='Jev-selected phase program; observed-geometry FK screening when available; every executed command requires fresh physical checks')
    engine._write('phase_program_selected',0.,**record)
    return record
