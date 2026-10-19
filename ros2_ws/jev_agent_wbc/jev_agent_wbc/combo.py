"""Explicit combo completion and phase-aware geometry; no hidden gap interpolation.

Authored main actions and their clock are immutable. Jev chooses each connector
before the shot starts; this is a geometric proposal, not a safety certificate.
The execution backend and candidate gates still own physical feasibility.
"""
from copy import deepcopy
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

from .contracts import validate_task, digest


def is_combo(original):
    if hasattr(original,'is_combo'):
        return original.is_combo
    return len({p['motion'].lower() for p in getattr(original,'task',{}).get('shot_phase',[])}) > 1


def gaps(task):
    task = validate_task(task, allow_phase_gaps=True)
    return [(a, b) for a, b in zip(task['shot_phase'], task['shot_phase'][1:])
            if b['start_s'] > a['end_s']+1e-8]


def complete_task(task, subject, choose, observed=None):
    """Fill only missing transition samples with a typed, journaled choice.

    `choose(evidence, questions)` is the existing SystemOneClient.ask interface.
    A deterministic callable is injectable for unit tests; production uses Jev.
    Scene keyframes and nominal future robot states never enter this interface.
    """
    from .jev import choice
    from .reference import smoothstep
    source = validate_task(task, allow_phase_gaps=True)
    missing = gaps(source)
    if not missing:
        return source, dict(completed=False, source_sha256=digest(source), transitions=[])
    samples = source['reference_camera_trajectory']['samples']
    by_time = {float(row['t']): row for row in samples}
    evidence = dict(intent=source['intent'], subject=source['subject'],
        shot_phase=source['shot_phase'], observed_tracks=(observed or {}).get('tracks', []),
        observation_stamp=(observed or {}).get('observation_stamp'), transitions=[])
    questions = {}
    definitions = []
    for i, (before, after) in enumerate(missing):
        a, b = before['end_s'], after['start_s']
        if a not in by_time or b not in by_time:
            raise ValueError('combo main actions require explicit endpoint poses')
        if any(a < row['t'] < b for row in samples):
            raise ValueError('undeclared gap contains authored samples; declare its phase explicitly')
        left, right = by_time[a], by_time[b]
        p, q = np.array(left['position']), np.array(right['position'])
        chord = q-p; axis = chord/max(np.linalg.norm(chord), 1e-9)
        up = np.array([0., 0., 1.]); up -= axis*(up@axis)
        if np.linalg.norm(up) < 1e-6:
            up = np.array([1., 0., 0.]); up -= axis*(up@axis)
        up /= np.linalg.norm(up)
        side = np.cross(axis, up)
        if np.linalg.norm(side) < 1e-6:
            side = np.array([0., 1., 0.])
        amplitude = min(.06, .3*np.linalg.norm(chord))
        directions = dict(left=side, right=-side, above=up, below=-up)
        key = f'transition_{i+1}'
        evidence['transitions'].append(dict(id=key, start_s=a, end_s=b,
            from_motion=before['motion'], to_motion=after['motion'],
            from_position=p.tolist(), to_position=q.tolist(), subject_position=list(subject),
            arc_amplitude_m=amplitude, directions_world={k:v.tolist() for k,v in directions.items()},
            constraints='fixed endpoints and duration; C2 stopped joins; requires independent physical checks'))
        questions[key] = choice('Choose the spatial side of this short connecting arc using the two actions, subject and observed geometry.',
            {k:f'Bow toward world direction {v.tolist()}; keep authored endpoints and timing.' for k,v in directions.items()})
        definitions.append((key,a,b,left,right,directions,amplitude))
    answers = choose(evidence, questions)
    result = deepcopy(source); extra=[]; transitions=[]
    for key,a,b,left,right,directions,amplitude in definitions:
        selected=answers[key]['choice']
        if selected not in directions:
            raise ValueError('transition choice outside typed options')
        times=np.linspace(a,b,max(9,int(np.ceil((b-a)/.05))+1))
        u=(times-a)/(b-a);s=smoothstep(u)
        p=np.asarray(left['position']);q=np.asarray(right['position'])
        positions=p+s[:,None]*(q-p)+(64*u**3*(1-u)**3)[:,None]*amplitude*directions[selected]
        r0=Rotation.from_quat(left['quaternion_xyzw']);r1=Rotation.from_quat(right['quaternion_xyzw'])
        rotations=r0*Rotation.from_rotvec(s[:,None]*(r0.inv()*r1).as_rotvec())
        extra.extend(dict(t=float(t),position=point.tolist(),quaternion_xyzw=quat.tolist())
            for t,point,quat in zip(times[1:-1],positions[1:-1],rotations.as_quat()[1:-1]))
        result['shot_phase'].append(dict(name=key,motion='transition',start_s=a,end_s=b))
        transitions.append(dict(id=key,start_s=a,end_s=b,choice=selected,
            selected_probability=answers[key].get('probabilities',{}).get(selected),amplitude_m=amplitude))
    result['shot_phase'].sort(key=lambda p:p['start_s'])
    result['reference_camera_trajectory']['samples']=sorted(samples+extra,key=lambda row:row['t'])
    validate_task(result)
    return result,dict(completed=True,source_sha256=digest(source),completed_sha256=digest(result),
        transitions=transitions,physical_validation='pending execution/candidate checks')


class PhaseTranslation:
    """C2 stopped joins for authored stop-to-stop actions and generated gaps."""
    def __init__(self, task):
        rows=task['reference_camera_trajectory']['samples']
        self.ends=[];self.curves=[]
        for phase in task['shot_phase']:
            samples=[r for r in rows if phase['start_s']-1e-9<=r['t']<=phase['end_s']+1e-9]
            if len(samples)<3 or samples[0]['t']!=phase['start_s'] or samples[-1]['t']!=phase['end_s']:
                raise ValueError('each combo phase requires endpoint poses and at least three samples')
            self.ends.append(phase['end_s'])
            self.curves.append(make_interp_spline([r['t'] for r in samples],[r['position'] for r in samples],k=5,
                bc_type=([(1,np.zeros(3)),(2,np.zeros(3))],[(1,np.zeros(3)),(2,np.zeros(3))])))

    def __call__(self,t,nu=0):
        values=np.asarray(t,float);flat=values.reshape(-1)
        index=np.minimum(np.searchsorted(self.ends,flat,side='right'),len(self.ends)-1)
        out=np.empty((len(flat),3))
        for i,curve in enumerate(self.curves):
            mask=index==i
            if mask.any():out[mask]=curve(flat[mask],nu)
        return out.reshape(values.shape+(3,))


class PhaseRotation:
    """SO(3) interpolation with zero angular velocity/acceleration at joins."""
    def __init__(self,task):
        self.ends=[];self.curves=[];self.starts=[]
        rows=task['reference_camera_trajectory']['samples']
        for phase in task['shot_phase']:
            samples=[r for r in rows if phase['start_s']-1e-9<=r['t']<=phase['end_s']+1e-9]
            r=Rotation.from_quat([row['quaternion_xyzw'] for row in samples]);base=r[0]
            self.ends.append(phase['end_s']);self.starts.append(base)
            self.curves.append(make_interp_spline([row['t'] for row in samples],(base.inv()*r).as_rotvec(),k=5,
                bc_type=([(1,np.zeros(3)),(2,np.zeros(3))],[(1,np.zeros(3)),(2,np.zeros(3))])))

    def __call__(self,t,order=0):
        values=np.asarray(t,float);flat=values.reshape(-1)
        index=np.minimum(np.searchsorted(self.ends,flat,side='right'),len(self.ends)-1)
        matrices=np.empty((len(flat),3,3));velocity=np.empty((len(flat),3))
        for i,curve in enumerate(self.curves):
            mask=index==i
            if not mask.any():continue
            at=flat[mask];r=Rotation.from_rotvec(curve(at))
            if order:
                h=1e-5
                # Body-frame rate, matching RotationSpline's order=1 API.
                velocity[mask]=(Rotation.from_rotvec(curve(at-h)).inv()*Rotation.from_rotvec(curve(at+h))).as_rotvec()/(2*h)
            else:matrices[mask]=(self.starts[i]*r).as_matrix()
        if order:return velocity.reshape(values.shape+(3,))
        return Rotation.from_matrix(matrices[0] if values.ndim==0 else matrices)


def motion_phases(original):
    """Merge adjacent landmarks only within the same main motion."""
    merged=[]
    for phase in original.task['shot_phase']:
        if merged and phase['motion']==merged[-1]['motion'] and phase['motion']!='transition':
            merged[-1]['end_s']=phase['end_s']
        else:merged.append(dict(phase))
    return merged


def phase_direction(original, phase, points):
    motion=phase['motion'].lower();points=np.asarray(points)
    ref=np.array([original.sample(phase[k])[0] for k in ('start_s','end_s')])
    if motion=='transition':return np.zeros(max(0,len(points)-1)), 'm'
    if motion in ('movein','moveout'):
        return (-1 if motion=='movein' else 1)*np.diff(np.linalg.norm(points-original.subject,axis=1)), 'm'
    if motion=='orbit':
        r=ref-original.subject
        sign=np.sign(np.arctan2(np.cross(r[0,:2],r[1,:2]),r[0,:2]@r[1,:2]))
        p=points-original.subject
        return sign*np.diff(np.unwrap(np.arctan2(p[:,1],p[:,0]))), 'rad'
    axis=original.sample(phase['start_s'])[1][:,1] if 'truck' in motion else np.array([0.,0.,1.])
    return np.sign((ref[1]-ref[0])@axis)*np.diff(points@axis), 'm'


def phase_quality(path,start=0.,end=None,progress=None):
    """Existing shape/direction thresholds, applied to each declared interval."""
    from .cinematic_geometry import curve_quality
    from .camera_intent import progress_at
    original=getattr(path,'original',path);end=path.duration if end is None else end
    clock=np.linspace(start,end,max(401,int((end-start)/.02)+1))
    progress=np.array([progress_at(path,t) for t in clock]) if progress is None else np.asarray(progress(clock))
    results=[]
    for phase in motion_phases(original):
        lo=max(phase['start_s'],progress[0]);hi=min(phase['end_s'],progress[-1])
        if hi<=lo+1e-8:continue
        a,b=np.interp([lo,hi],progress,clock)
        quality=curve_quality(path,start=float(a),end=float(b),phase=phase,_single=True)
        results.append(dict(phase=phase,interval_s=[float(a),float(b)],**quality))
    return dict(scope='per_phase_including_transition_shape',segments=results,
        direction_pass=bool(results) and all(r['direction_pass'] for r in results),
        regular_curve_pass=bool(results) and all(r['regular_curve_pass'] for r in results),
        strict_full_curve_pass=bool(results) and all(r['strict_full_curve_pass'] for r in results),
        reverse_motion=sum(r['reverse_motion'] for r in results),reverse_unit='per_phase_m_or_rad',
        total_turn_rad=sum(r['total_turn_rad'] for r in results),
        max_step_turn_rad=max((r['max_step_turn_rad'] for r in results),default=0.),
        length_m=sum(r['length_m'] for r in results))


def combo_curves(original,active,activation,model,intrinsics,count=1,seed_direction=None,
                 seed_amplitude=.15,initial_boundary=None,tempo=1.):
    """Deform the complete piecewise reference without removing later actions.

    Spatial bumps and clock changes vanish at each phase boundary. Their
    coefficients are proposals; the ordinary independent gates certify them.
    """
    from .reference import smoothstep
    from .intent_paths import IntentPath,positions_at
    end=active.duration
    if end-activation<.3:return []
    phases=motion_phases(original)
    boundaries=[p[k] for p in phases for k in ('start_s','end_s') if activation<p[k]<end]
    times=np.unique(np.r_[np.linspace(activation,end,max(49,int(np.ceil((end-activation)/.08))+1)),boundaries])
    progress=times.copy();bump=np.zeros(len(times));skew=np.zeros(len(times));endpoint=np.zeros(len(times))
    for phase in phases:
        a,b=phase['start_s'],phase['end_s'];mask=(times>=a)&(times<=b)
        u=np.clip((times[mask]-max(a,activation))/max(b-max(a,activation),1e-8),0.,1.)
        shape=64*u**3*(1-u)**3
        if phase['motion']=='transition':
            distance=np.linalg.norm(original.sample(b)[0]-original.sample(a)[0])
            shape*=min(1.,distance/.8)  # Short connectors must not inherit a full main-action detour.
        bump[mask]=shape;skew[mask]=shape*(2*u-1)
        if b==original.duration:endpoint[mask]=smoothstep(u)
    # Preserve the authored phase clock: no global inversion of a mixed-motion feature.
    base=positions_at(original,progress)
    p0,_,v0=active.sample(activation);h=.001
    a0=(active.sample(activation+h)[2][:3]-active.sample(activation-h)[2][:3])/(2*h)
    if initial_boundary is not None:
        p0=np.asarray(initial_boundary['position']);v0=np.asarray(initial_boundary['twist']);a0=np.zeros(3)
    nr=original.sample(activation)
    na=(original.sample(activation+h)[2][:3]-original.sample(activation-h)[2][:3])/(2*h)
    join_end=next((p['end_s'] for p in phases if p['end_s']>activation+.2),end)
    span=join_end-activation;u=np.clip((times-activation)/span,0,1)
    base+=(1-smoothstep(u))[:,None]*(p0-nr[0])
    base+=(u-6*u**3+8*u**4-3*u**5)[:,None]*span*(v0[:3]-nr[2][:3])
    base+=.5*(u**2-3*u**3+3*u**4-u**5)[:,None]*span**2*(a0-na)
    if seed_direction is None:
        chord=original.sample(end)[0]-p0;side=np.cross(chord,[0.,0.,1.]);side/=max(np.linalg.norm(side),1e-8)
        directions=[np.zeros(3),side,-side,np.array([0.,0.,1.]),np.array([0.,0.,-1.])]
    else:directions=[np.asarray(seed_direction,float)]
    paths=[]
    for direction in directions[:count]:
        seed=seed_amplitude*direction
        geometry=dict(times=times.tolist(),positions=(base+bump[:,None]*seed).tolist(),
            progress_times=progress.tolist(),shape_basis=np.column_stack([bump,skew,endpoint]).tolist(),
            start_velocity=v0[:3].tolist(),start_acceleration=a0.tolist())
        program=dict(route='timing' if not np.any(seed) else 'left',amplitude_m=float(np.linalg.norm(seed)),
            extension_s=0.,regular_curve=True,curve_family='global_endpoint_polynomial',
            combo_phase_preserving=True,endpoint_offset_m=[0.,0.,0.],search_role='coordinated_clearance',
            unverified_seed_direction=seed.tolist(),shape_coefficients=np.r_[seed,np.zeros(3)].tolist())
        if initial_boundary is not None:
            program['measured_initial_boundary']=True
            geometry['initial_rotation_xyzw']=initial_boundary['quaternion_xyzw']
        paths.append(IntentPath(original,active,activation,program,model,intrinsics,geometry=geometry))
    return paths
