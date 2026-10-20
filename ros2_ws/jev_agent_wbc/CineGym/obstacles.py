"""Independent Hermite lanes fitted to simultaneous full-body nominal geometry.

The nominal rollout is an offline authoring tool. These trajectories and their
future positions are never included in an Agent observation.
"""
import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, BPoly
from scipy.optimize import brentq

from jev_agent_wbc.kc_svsdf import CollisionGeometry, primitive_sdf
from jev_agent_wbc.reference import smoothstep
from .config import LEVELS, MOTIONS, PROFILES


def lane(rng, duration, event, anchor, tangent, normal, speed, mode):
    times = np.unique(np.r_[np.linspace(0., duration, int(np.ceil(duration/.15))+1), event])
    u = times/duration
    f = smoothstep(u)
    df = 30*u*u*(1-u)**2/duration
    fe = smoothstep(event/duration)
    delta = f-fe
    travel = speed*duration/1.875
    along, along_v = travel*delta, travel*df
    side, side_v = np.zeros_like(times), np.zeros_like(times)
    vertical, vertical_v = np.zeros_like(times), np.zeros_like(times)
    amplitude = float(rng.uniform(.06, .16))
    omega = float(rng.uniform(4., 7.))
    if mode == 'curved':
        side = amplitude*8*delta**2
        side_v = amplitude*16*delta*df
    elif mode == 'stop_go':
        # An actual stationary interval at the crossing, with C1 arrival/departure.
        hold = .055
        a = np.maximum(np.abs(delta)-hold, 0.)
        along = travel*np.sign(delta)*a*a/(a+.04)
        along_v = travel*a*(a+.08)/(a+.04)**2*df
    elif mode == 'reverse':
        along = travel*np.sin(omega*delta)/omega
        along_v = travel*np.cos(omega*delta)*df
    elif mode == 'wave':
        side = amplitude*np.sin(omega*delta)
        side_v = amplitude*omega*np.cos(omega*delta)*df
    elif mode == 'diagonal3d':
        vertical = .035*np.sin(omega*delta)
        vertical_v = .035*omega*np.cos(omega*delta)*df
    positions = anchor+along[:, None]*tangent+side[:, None]*normal
    velocities = along_v[:, None]*tangent+side_v[:, None]*normal
    positions[:, 2] += vertical
    velocities[:, 2] += vertical_v
    return times, positions, velocities


def natural_lane(rng, duration, event, anchor, tangent, normal, speed, mode):
    """Independent C2 lanes with local motion changes and moving boundaries.

    A natural cubic spline supplies knot velocities: the runtime's Hermite
    representation is then exactly the same C2 curve. No global shot clock
    eases all actors to rest simultaneously. Zero boundary acceleration also
    joins the sensor pre-roll's constant velocity continuously.
    """
    times=np.unique(np.r_[np.linspace(0.,duration,int(np.ceil(duration/.08))+1),event])
    tau=times-event
    width=float(rng.uniform(.65,1.1))
    def primitive(t, start, span):
        u=np.clip((t-start)/span,0.,1.)
        return span*(2.5*u**4-3*u**5+u**6)+np.maximum(t-start-span,0.)
    def window_integral(t):
        return primitive(t,-1.3*width,.8*width)-primitive(t,.4*width,.8*width)
    along=speed*tau
    if mode in ('stop_go','reverse'):
        factor=1. if mode=='stop_go' else 2.
        along-=factor*speed*(window_integral(tau)-window_integral(0.))
    side=np.zeros_like(times);vertical=np.zeros_like(times)
    amplitude=float(rng.uniform(.05,.12))
    if mode=='curved':
        side=amplitude*(np.tanh(tau/(1.5*width)))
    elif mode=='wave':
        side=amplitude*np.sin(tau/width)*np.exp(-.5*(tau/(2.5*width))**2)
    elif mode=='diagonal3d':
        vertical=.035*np.tanh(tau/(1.5*width))
    positions=anchor+along[:,None]*tangent+side[:,None]*normal
    positions[:,2]+=vertical
    curve=CubicSpline(times,positions,bc_type='natural')
    return times,positions,curve(times,1)


def motion_bounds(curve):
    """Conservative continuous bounds from derivative Bernstein hulls."""
    def bound(order):
        controls=BPoly.from_power_basis(curve.derivative(order)).c
        return float(np.linalg.norm(controls,axis=-1).max())
    return dict(speed_m_s=bound(1),acceleration_m_s2=bound(2),jerk_m_s3=bound(3))


def fit_obstacles(rng, scene, times, states, config, timing_pool=None,phases=None):
    geometry = CollisionGeometry()
    frame_p, frame_r = geometry.frame_arrays(states, scene['base_z'])
    profile = PROFILES[config.difficulty]
    count = config.obstacle_count or int(rng.integers(profile['count'][0], profile['count'][1]+1))
    duration = float(times[-1])
    curves, objects, reports = [], [], []
    survivors=list(range(len(timing_pool))) if timing_pool is not None else None
    # A shuffled cycle covers behaviors without collapsing each episode to one lane type.
    modes = list(rng.permutation(MOTIONS))
    levels = list(LEVELS)+list(rng.choice(LEVELS, size=count-3))
    order = rng.permutation(count)
    event_fractions = np.linspace(.25, .75, count)[order]
    if config.difficulty == 'hard':
        event_fractions = rng.uniform(.35, .65, count)
    natural=config.generation_style=='natural'
    hazards=set(rng.choice(count,size=min(count,{'easy':1,'medium':2,'hard':3}[config.difficulty]),replace=False)) if natural else set(range(count))
    if natural:
        # Difficulty includes actual time to react, not just penetration.
        # Leave room to perceive, choose and recover within the authored shot.
        for i in hazards:event_fractions[i]=rng.uniform(.40,.65)
        if phases is not None and len(phases)>1:
            # Spread independent encounters over action interiors. The middle
            # of the entire combo is usually a stopped phase boundary, where
            # a timing repair has no freedom and all hazards used to bunch up.
            phase_order=rng.permutation(len(phases))
            for index,i in enumerate(sorted(hazards)):
                phase=phases[int(phase_order[index%len(phases)])]
                event_fractions[i]=(phase['start_s']+rng.uniform(.4,.65)*
                    (phase['end_s']-phase['start_s']))/duration
    for j, level in enumerate(levels):
        conflict=j in hazards
        mode = str(modes[j % len(modes)])
        allowed = {'low': (0,), 'middle': (1, 2, 3, 4), 'high': (5, 6, 7)}[level]
        height_bounds = {'low': (.13, .29), 'middle': (.34, .62), 'high': (.67, 1.14)}[level]
        accepted = None
        for trial in range(180):
            fraction = float(np.clip(event_fractions[j]+rng.uniform(-.055, .055), .1 if natural else .2, .9 if natural else .8))
            k = int(np.argmin(abs(times-duration*fraction)))
            event = float(times[k])
            centers = frame_p[k, geometry.shape_bodies]+np.einsum('pij,pj->pi', frame_r[k, geometry.shape_bodies], geometry.shape_offsets)
            candidates = [i for i, shape in enumerate(geometry.shapes)
                          if shape[0] in allowed and height_bounds[0] <= centers[i, 2] <= height_bounds[1]]
            if not candidates:
                continue
            anchor = centers[int(rng.choice(candidates))].copy()
            radius = float(rng.uniform(*config.radius_m))
            anchor[2] = float(np.clip(anchor[2]+rng.uniform(-.025, .025), max(height_bounds[0], radius+.045), height_bounds[1]))
            angle = float(rng.uniform(-np.pi, np.pi))
            normal = np.array([np.cos(angle), np.sin(angle), 0.])
            tangent = np.cross([0., 0., float(rng.choice([-1., 1.]))], normal)
            penetration = (min(float(rng.uniform(*profile['penetration'])), radius*.8)
                           if conflict else -float(rng.uniform(.18,.32)))
            def clearance(offset):
                return float(geometry._primitive_arrays(frame_p[k], frame_r[k], (anchor+offset*normal)[None]).min()-radius)
            if clearance(0.) > -penetration:
                continue
            try:
                offset = brentq(lambda x: clearance(x)+penetration, 0., 1.5, xtol=1e-7)
            except ValueError:
                continue
            anchor += offset*normal
            speed = float(rng.uniform(*profile['speed']))
            kt, kp, kv = (natural_lane if natural else lane)(rng, duration, event, anchor, tangent, normal, speed, mode)
            curve = CubicHermiteSpline(kt, kp, kv)
            points = curve(times)
            speeds = np.linalg.norm(curve(times, 1), axis=1)
            accels = np.linalg.norm(curve(times, 2), axis=1)
            bounds=motion_bounds(curve)
            speed_cap=min(config.max_obstacle_speed_m_s,profile['speed'][1]*1.4) if natural else config.max_obstacle_speed_m_s
            acceleration_cap=min(config.max_obstacle_acceleration_m_s2,{'easy':.8,'medium':1.2,'hard':2.}[config.difficulty]) if natural else config.max_obstacle_acceleration_m_s2
            if bounds['speed_m_s'] > speed_cap or bounds['acceleration_m_s2'] > acceleration_cap:
                continue
            if natural and bounds['jerk_m_s3']>{'easy':3.,'medium':6.,'hard':12.}[config.difficulty]:
                continue
            if points[:, 2].min()-radius < .005:
                continue
            ds = geometry._primitive_arrays(frame_p, frame_r, points[:, None, :]).min(axis=-1)[:, 0]-radius
            if (natural or config.enforce_global_penetration) and ds.min() < -profile['penetration'][1]-1e-7:
                continue
            if natural:
                if not conflict and ds.min()<.12:continue
                contact=times[ds<0.]
                warning=times[ds<.10]
                if conflict and (not len(contact) or contact[0]<{'easy':3.,'medium':2.5,'hard':1.2}[config.difficulty] or
                    np.count_nonzero(ds<0.)*float(np.max(np.diff(times)))>{'easy':1.4,'medium':2.,'hard':3.}[config.difficulty]):continue
                if len(warning) and warning[0]<.6:continue
            if min(ds[0], ds[-1]) < config.initial_clearance_m:
                continue
            if natural:
                past=kp[0]+np.linspace(-.8,0.,41)[:,None]*kv[0]
                if (geometry.distances(states[0],past,scene['base_z'])-radius).min()<.025:
                    continue
            pair_min = min((float((np.linalg.norm(points-c(times), axis=1)-radius-o['radius']).min())
                            for c, o in zip(curves, objects)), default=100.)
            if pair_min < .025:
                continue
            subject_min = float((primitive_sdf(points-np.array(scene['subject']['position']), 'cylinder',
                                 [scene['subject']['radius'], scene['subject']['height']])-radius).min())
            # Include the yellow top sphere present in the physical stage.
            cap = np.array(scene['subject']['position'])+[0., 0., scene['subject']['height']/2]
            if min(subject_min, float((np.linalg.norm(points-cap, axis=1)-radius-.07).min())) < .06:
                continue
            at_hit = geometry.distances(states[k], [anchor], scene['base_z'])[0]-radius
            if conflict and min(at_hit[list(allowed)]) >= 0.:
                continue
            if survivors is not None:
                from .solvability import safe_clock_indices
                remaining=safe_clock_indices(timing_pool,survivors,curve,radius,bounds['speed_m_s'])
                if not remaining:continue
            witness = int(np.argmin(ds))
            obj_id = f'{level}_{j+1:02d}_{mode}'
            obj = dict(id=obj_id, radius=radius, color={'low': [.9, .35, .1], 'middle': [.9, .65, .1], 'high': [.65, .2, .8]}[level],
                       keyframes=[dict(t=float(t), position=p.tolist(), velocity=v.tolist()) for t, p, v in zip(kt, kp, kv)])
            report = dict(obstacle_id=obj_id, height_level=level, motion=mode, radius_m=radius,
                interaction_role='conflict' if conflict else 'passing',
                motion_model='local_C2_independent_lane' if natural else 'global_shot_smoothstep',
                continuous_motion_bounds=bounds,
                center_height_range_m=[float(points[:, 2].min()), float(points[:, 2].max())],
                intended_crossing_s=event, intended_body_group=list(allowed),
                target_penetration_m=penetration, min_surface_clearance_m=float(ds.min()),
                global_penetration_cap_m=profile['penetration'][1] if natural or config.enforce_global_penetration else None,
                witness_time_s=float(times[witness]), witness_body_index=int(np.argmin(geometry.distances(states[witness], [points[witness]], scene['base_z'])[0])),
                interference_sample_count=int((ds < 0.).sum()),
                initial_clearance_m=float(ds[0]), final_clearance_m=float(ds[-1]),
                max_speed_m_s=float(speeds.max()), max_acceleration_m_s2=float(accels.max()))
            report['conflict_duration_s']=float(np.count_nonzero(ds<0.)*np.max(np.diff(times)))
            report['first_conflict_s']=float(times[ds<0.][0]) if np.any(ds<0.) else None
            report['initial_reaction_window_s']=report['first_conflict_s']
            accepted = obj, report, curve
            if survivors is not None:survivors=remaining
            break
        if accepted is None:
            raise ValueError(f'could not fit {level}/{mode} with clear endpoints and independent lanes')
        obj, report, curve = accepted
        objects.append(obj); reports.append(report); curves.append(curve)
    all_ds = np.stack([geometry._primitive_arrays(frame_p, frame_r, c(times)[:, None, :]).min(axis=-1)[:, 0]-o['radius']
                       for c, o in zip(curves, objects)], axis=1)
    return objects, dict(obstacles=reports, audit_step_s=float(np.max(np.diff(times))),
        generation_style=config.generation_style,conflict_actors=len(hazards),
        timing_witness_indices=survivors,
        max_simultaneous_interferences=int(((all_ds < 0.).sum(axis=1)).max()),
        min_surface_clearance_m=float(all_ds.min()),
        obstacle_pair_min_clearance_m=min(float((np.linalg.norm(a(times)-b(times), axis=1)-objects[i]['radius']-objects[j]['radius']).min())
            for i, a in enumerate(curves) for j, b in enumerate(curves) if j > i),
        scope='sampled full 8-body obstacle-free nominal geometry; not continuous safety or avoidance solvability certification')
