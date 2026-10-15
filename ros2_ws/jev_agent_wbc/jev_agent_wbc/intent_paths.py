"""Agent-selected motion programs instantiated as smooth camera-space paths."""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

from .camera_intent import POLICY, default_intrinsics, projected_area, progress_at
from .reference import look_at, smoothstep


STRATEGIES = {
    'framing': 'Compare small radius/height adjustments and time allocation; preserve subject scale and the immutable final frame.',
    'view': 'Compare small side/height adjustments and time allocation; preserve viewing direction, motion trend and final framing.',
    'joint': 'Compare bounded time allocation and small returning side/radius/height changes; preserve the final frame.',
    'timing': 'Keep the original shot trend; smoothly accelerate or slow passage while obstacles move, plus small local deformations.',
    'restore': 'Finish the remaining intent with a smooth continuation or a new visible route.',
}


def positions_at(path, times):
    """Vectorized geometry-only sampling; collision checks need no twist or FK."""
    from .reference import OriginalReference
    times = np.asarray(times)
    if isinstance(path, IntentPath):
        if path.program['route'] == 'continuation':
            return positions_at(path.previous,times)
        before = times <= path.activation_s
        positions = np.empty((len(times), 3))
        positions[before] = positions_at(path.previous, times[before])
        positions[~before] = path.translation(np.clip(times[~before], path.activation_s, path.duration))
        return positions
    if isinstance(path, OriginalReference):
        return path.translation(np.clip(times, 0., path.duration))
    return np.array([path.sample(float(t))[0] for t in times]).reshape(-1,3)


def pose_at(path, t):
    from .reference import OriginalReference
    if isinstance(path,IntentPath):return path.pose(t)
    if isinstance(path,OriginalReference):
        at=float(max(0.,min(t,path.duration)))
        cache=getattr(path,'_geometry_pose_cache',None)
        if cache is None:
            cache=path._geometry_pose_cache={}
        if at not in cache:
            if len(cache)>8192:
                cache.clear()
            cache[at]=(path.translation(at),path.rotation(at).as_matrix())
        return tuple(value.copy() for value in cache[at])
    return path.sample(t)[:2]


class IntentPath:
    """An immutable exact path with an unchanged executed/activation prefix.

    Timing uses monotone smooth speed pulses or a quintic warp of remaining motion. Spatial
    freedom lives on the shot's primary feature level sets. The nominal path is
    one seed, not the selection objective. No joint state or robot model is used.
    """
    def __init__(self, original, previous, activation_s, program, model=None,
                 intrinsics=None, geometry=None, seed_cache=None):
        self.original, self.previous = original, previous
        self.activation_s = float(activation_s)
        self.program = dict(program)
        self.seed_cache={} if seed_cache is None else seed_cache
        if self.program.get('timing_profile') in ('early_fast','early_slow'):
            fast=self.program['timing_profile']=='early_fast'
            self.program.setdefault('tempo_rate',2.5 if fast else .85)
            self.program.setdefault('tempo_ramp_s',.4 if fast else .6)
        self.model = model
        self.intrinsics = default_intrinsics() if intrinsics is None else intrinsics
        self.extension_s = float(program.get('extension_s', 0.))
        self.duration = previous.duration+self.extension_s
        if not np.isfinite(self.program.get('phase_rate',0.)) or abs(self.program.get('phase_rate',0.)) >= 1.:
            raise ValueError('fixed-deadline clock must remain monotone')
        if not 0.<=self.program.get('up_fraction',.5)<=1.:
            raise ValueError('invalid spatial direction mixture')
        self.subject = original.subject
        self.return_s = None if program.get('profile') == 'hold' else self.duration
        self.values = np.zeros(6)  # Legacy diagnostics only; program is authoritative.
        self.envelope = SimpleNamespace(end_s=self.duration, decision_id='intent-program',
            blend_s=self.duration-self.activation_s, terminal='complete_intent', bounds={})
        if self.duration <= self.activation_s or (self.extension_s < 0 and program.get('timing_profile') != 'early_fast'):
            raise ValueError('invalid remaining motion duration')
        if self.duration < POLICY.min_duration_factor*original.duration-1e-8:
            raise ValueError('intent duration is too compressed')
        if self.duration > POLICY.max_duration_factor*original.duration+1e-8:
            raise ValueError('intent duration limit exceeded')
        if program.get('timing_profile') != 'early_slow' and self.extension_s/(self.duration-self.activation_s) > .52:
            raise ValueError('time warp would reverse progress')
        if geometry is None:
            times = np.linspace(self.activation_s, self.duration, 49)
            positions = np.array([self._construct_position(float(t)) for t in times])
            h = 1e-3
            p0, r0, v0 = previous.sample(self.activation_s)
            a0 = (previous.sample(self.activation_s+h)[2][:3]-previous.sample(self.activation_s-h)[2][:3])/(2*h)
            self.geometry = dict(times=times.tolist(), positions=positions.tolist(),
                start_velocity=v0[:3].tolist(), start_acceleration=a0.tolist())
        else:
            self.geometry = geometry
        g = self.geometry
        self.translation = make_interp_spline(g['times'], g['positions'], k=5,
            bc_type=([(1, np.array(g['start_velocity'])), (2, np.array(g['start_acceleration']))],
                     [(1, np.zeros(3)), (2, np.zeros(3))]))
        self._cache = {}

    def parent_time(self, t):
        if t <= self.activation_s:
            return float(t)
        elapsed = min(float(t), self.duration)-self.activation_s
        u = elapsed/(self.duration-self.activation_s)
        if self.program.get('phase_rate'):
            half = (self.duration-self.activation_s)/2
            ramp = min(.5, half/4)
            def pulse_integral(x):
                x = max(0.,min(x,half))
                def si(v):
                    return 2.5*v**4-3*v**5+v**6
                if x < ramp:
                    return ramp*si(x/ramp)
                if x <= half-ramp:
                    return x-ramp/2
                v=(x-half+ramp)/ramp
                return half-1.5*ramp+ramp*(v-si(v))
            # Equal opposite speed pulses redistribute progress while keeping
            # the original endpoint time. Clock speed is in [1-|a|,1+|a|]
            # and clock acceleration vanishes at the unchanged prefix.
            return self.activation_s+elapsed+self.program['phase_rate']*(
                pulse_integral(elapsed)-pulse_integral(elapsed-half))
        if self.program.get('timing_profile') in ('early_slow','early_fast') and self.extension_s:
            # Smooth speed pulses slow or advance progress. Scene time never pauses.
            # Integral of quintic smoothstep makes clock velocity/acceleration
            # continuous; source progress remains strictly positive.
            rate = float(self.program.get('tempo_rate',2.5 if self.extension_s<0 else .85))
            ramp = min(float(self.program.get('tempo_ramp_s',.4 if self.extension_s<0 else .6)),abs(self.extension_s)/rate)
            plateau = max(0.,abs(self.extension_s)/rate-ramp)
            def integral(x):
                x=np.clip(x,0.,1.)
                return 2.5*x**4-3*x**5+x**6
            if elapsed<ramp:
                delay=rate*ramp*integral(elapsed/ramp)
            elif elapsed<ramp+plateau:
                delay=rate*(ramp/2+elapsed-ramp)
            elif elapsed<2*ramp+plateau:
                x=(elapsed-ramp-plateau)/ramp
                delay=rate*(ramp/2+plateau+ramp*(x-integral(x)))
            else:
                delay=abs(self.extension_s)
            return min(self.previous.duration,self.activation_s+elapsed-np.sign(self.extension_s)*delay)
        return self.activation_s+elapsed-self.extension_s*smoothstep(u)

    def progress_time(self, t):
        return progress_at(self.previous, self.parent_time(t))

    def _construct_position(self, t):
        source_t = self.parent_time(t)
        p, r = pose_at(self.previous,source_t)
        u = max(0.,min((t-self.activation_s)/(self.duration-self.activation_s), 1.))
        bump = (smoothstep((t-self.activation_s)/self.program['rise_s'])
                if self.program.get('profile') == 'hold' else 64*u**3*(1-u)**3)
        if 'peak_s' in self.program:
            peak=float(self.program['peak_s'])
            bump=(smoothstep((t-self.activation_s)/(peak-self.activation_s)) if t<=peak else
                  1.-smoothstep((t-peak)/(self.duration-peak)))
        amplitude = float(self.program.get('amplitude_m', 0.))*bump
        if not amplitude:
            return p
        ref_t = self.progress_time(t)
        phase = self.original.phase(ref_t)
        motion = phase['motion'].lower()
        nominal, nr = pose_at(self.original,ref_t)
        radial = nominal-self.subject
        radial[2] = 0.
        radial /= max(np.linalg.norm(radial), 1e-8)
        left = np.array([radial[1],-radial[0],0.])
        up = np.array([0., 0., 1.])
        route = self.program['route']
        offsets = dict(left=left, right=-left, above=up, below=-up,
            outer=radial, inner=-radial, outer_above=(radial+up)/np.sqrt(2),
            inner_above=(-radial+up)/np.sqrt(2), outer_below=(radial-up)/np.sqrt(2),
            inner_below=(-radial-up)/np.sqrt(2), left_outer=(left+radial)/np.sqrt(2),
            left_inner=(left-radial)/np.sqrt(2), right_outer=(-left+radial)/np.sqrt(2),
            right_inner=(-left-radial)/np.sqrt(2),
            left_above=(left+up)/np.sqrt(2), right_above=(-left+up)/np.sqrt(2),
            timing=np.zeros(3), continuation=np.zeros(3))
        offset = offsets[route].copy()
        if 'up_fraction' in self.program:
            vertical=float(self.program['up_fraction'])
            horizontal=offset.copy();horizontal[2]=0.
            horizontal/=max(np.linalg.norm(horizontal),1e-8)
            offset=np.sqrt(1-vertical**2)*horizontal+np.sign(offset[2])*vertical*up
        if motion == 'orbit':
            # Preserve the *same* azimuth sector, not just angular travel.
            offset = offset@radial*radial+offset[2]*up
        elif 'truck' in motion:
            axes=getattr(self.original,'_intent_truck_axes',None)
            if axes is None:
                axes=self.original._intent_truck_axes={}
            if motion not in axes:
                first=next(s for s in self.original.task['shot_phase'] if s['motion']==motion)
                axes[motion]=self.original.sample(first['start_s'])[1][:,1]
            axis=axes[motion]
            offset -= (offset@axis)*axis
        elif 'pedestal' in motion:
            offset[2] = 0.
        p = p+amplitude*offset
        if motion in ('movein', 'moveout') and self.model is not None:
            if ref_t not in self.seed_cache:
                self.seed_cache[ref_t]=(projected_area(nominal,nr,self.model,self.intrinsics),
                                        look_at(nominal,self.subject).T@nr)
            target,framing=self.seed_cache[ref_t]
            # Compensate changed view-dependent cylinder silhouette, not visible
            # mask area. Fixed intrinsics: zoom cannot masquerade as a dolly.
            for _ in range(3):
                pr = look_at(p, self.subject)@framing
                area = projected_area(p, pr, self.model, self.intrinsics)
                if area is None or target is None:
                    break
                if abs(area/target-1.) < 1e-6:
                    break
                p = self.subject+(p-self.subject)*np.sqrt(area/target)
        return p

    def pose(self, t):
        if t <= self.activation_s or self.program['route'] == 'continuation':
            return pose_at(self.previous,t)
        p = self.translation(np.clip(t, self.activation_s, self.duration))
        base_p, base_r = pose_at(self.previous,self.parent_time(t))
        r = look_at(p, self.subject)@look_at(base_p, self.subject).T@base_r
        return p, r

    def sample(self, t):
        if t <= self.activation_s or self.program['route'] == 'continuation':
            return self.previous.sample(t)
        key = float(t)
        if key in self._cache:
            return tuple(v.copy() for v in self._cache[key])
        p, r = self.pose(t)
        h = 1e-4
        a, b = max(self.activation_s, t-h), min(self.duration, t+h)
        twist = np.zeros(6)
        if t < self.duration and b > a:
            ra, rb = self.pose(a)[1], self.pose(b)[1]
            twist = np.r_[self.translation(t, 1), Rotation.from_matrix(rb@ra.T).as_rotvec()/(b-a)]
        result = p, r, twist
        if len(self._cache) > 2048:
            self._cache.clear()
        self._cache[key] = tuple(v.copy() for v in result)
        return result

    def parameters(self, t):
        return np.zeros(6)

    def to_spec(self):
        from .candidates import path_spec
        return dict(kind='intent_program/v1', original_sha256=self.original.sha256,
            activation_s=self.activation_s, duration_s=self.duration, program=self.program,
            subject_model=self.model, intrinsics=self.intrinsics, geometry=self.geometry,
            previous=path_spec(self.previous))

    @classmethod
    def from_spec(cls, original, spec):
        from .candidates import restore_path
        return cls(original, restore_path(original, spec['previous']), spec['activation_s'],
            spec['program'], spec['subject_model'], spec['intrinsics'], spec['geometry'])

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'original'), name)


def propose_intent_paths(original, active, strategy, t, budget, model=None, intrinsics=None,
                         allow_time_extension=True, scene=None, kinematic_search=None, search_config=None):
    from .search_configuration import profile
    config = profile(search_config)
    activation = t+budget.activation_lead_s
    if activation >= active.duration-.3:
        return []
    motion = original.phase(progress_at(active, activation))['motion'].lower()
    routes = (['outer', 'inner', 'above', 'below'] if motion == 'orbit' else
              ['left', 'right', 'outer', 'inner'] if 'pedestal' in motion else
              ['outer', 'inner', 'above', 'below'] if 'truck' in motion else
              ['left', 'right', 'above', 'below'])
    if not allow_time_extension:
        routes += (['left_outer','left_inner','right_outer','right_inner'] if 'pedestal' in motion else
                   ['outer_above','inner_above','outer_below','inner_below'] if motion == 'orbit' or 'truck' in motion else
                   ['left_above','right_above'])
    programs = [dict(route='continuation', amplitude_m=0., extension_s=0.)]
    if not allow_time_extension:
        # A fixed shot deadline forbids duration changes, not redistribution
        # of progress within the remaining interval.
        programs += [dict(route='timing', amplitude_m=0., extension_s=0.,
                          phase_rate=sign*.9) for sign in (1., -1.)]
    spare = POLICY.max_duration_factor*original.duration-active.duration if allow_time_extension else 0.
    # Time-first programs keep all subject-relative states on the original shot.
    # Small returning spatial programs coexist with obstacles; large held offsets
    # remain replayable historical programs but are not proposed as full success.
    if allow_time_extension:
        for advance in (1.6,2.4):
            if active.duration-advance>=POLICY.min_duration_factor*original.duration and advance/2.5+.4 < active.duration-advance-activation:
                programs.append(dict(route='timing',amplitude_m=0.,extension_s=-advance,timing_profile='early_fast'))
    for extension in (1.6,):
        if extension <= spare+1e-8 and extension/.85+.6 < active.duration+extension-activation:
            programs.append(dict(route='timing',amplitude_m=0.,extension_s=extension,timing_profile='early_slow'))
    amplitudes=(.12,.22) if allow_time_extension else (.12,.18,.20,.22)
    if config is not None:
        amplitudes=config['amplitudes']
    for amplitude in amplitudes:
        programs += [dict(route=route,amplitude_m=amplitude,extension_s=0.,profile='arch') for route in routes]
    fast=[v for v in programs if v.get('timing_profile')=='early_fast']
    if fast:
        advance=min(v['extension_s'] for v in fast)
        programs += [dict(route=route,amplitude_m=.08,extension_s=advance,profile='arch',timing_profile='early_fast') for route in routes]
    if not allow_time_extension:
        programs += [dict(route=route,amplitude_m=amplitude,extension_s=0.,profile='arch',
                          phase_rate=rate) for rate in (.45,-.45,.7,-.7,.9,-.9)
                     for amplitude in amplitudes for route in routes]
    if strategy in ('view','framing'):
        # The Agent's family affects the search order, never the numeric gates.
        preferred = ('above','below') if strategy=='framing' else ('inner','outer')
        programs.sort(key=lambda v:(v['route'] not in preferred,v['amplitude_m']))
    if not allow_time_extension:
        programs += [dict(program,up_fraction=fraction) for program in programs
                     if program['route'] in ('outer_above','inner_above','outer_below','inner_below','left_above','right_above')
                     for fraction in (.35,.9)]
    encounter=None
    if scene is not None and scene.get('valid') and len(scene.get('centers',[])):
        times=np.linspace(activation,active.duration,81)
        points=positions_at(active,times)
        centers=np.asarray(scene['centers'])[None]+(times-scene['observation_stamp'])[:,None,None]*np.asarray(scene['velocities'])[None]
        gap=np.min(np.linalg.norm(points[:,None]-centers,axis=2)-np.asarray(scene['radii'])[None],axis=1)
        encounter=float(times[np.argmin(gap)])
    result = []
    seed_cache={}
    radial = pose_at(active, activation)[0]-original.subject
    radial[2] = 0.
    radial /= max(np.linalg.norm(radial), 1e-8)
    seeds = programs[:budget.max_proposals] if allow_time_extension else programs
    for program in seeds:
        if kinematic_search is not None and program['amplitude_m']:
            program = dict(program, amplitude_m=kinematic_search.amplitude(
                program['route'], program['amplitude_m'], radial))
        if program.get('timing_profile') != 'early_slow' and program['extension_s']/(active.duration+program['extension_s']-activation) > .52:
            continue
        if encounter is not None and program['amplitude_m']:
            duration=active.duration+program['extension_s']
            program['peak_s']=float(np.clip(encounter,activation+min(.6,(duration-activation)/3),
                                           duration-min(.6,(duration-activation)/3)))
        result.append(IntentPath(original, active, activation, program, model, intrinsics,seed_cache=seed_cache))
    if not allow_time_extension and scene is not None and scene.get('valid'):
        # Cheap, bounded geometry seed search precedes expensive full intent
        # and FK checks. Small budgets must not silently omit the opposite
        # timing sign or all diagonal routes. These scores are NOT certificates.
        from .floating_camera import predicted_clearances
        from .camera_intent import visibility
        from .subject_state import frame_state,frame_error
        times=np.linspace(t,active.duration+POLICY.terminal_hold_s,49)
        def score(path):
            points=positions_at(path,times)
            clear=predicted_clearances(points,times,scene,.06)
            at_risk=np.argsort(clear)[:5]
            seen=[visibility(*pose_at(path,float(times[i])),scene,float(times[i]),model,
                             default_intrinsics() if intrinsics is None else intrinsics)['visibility'] for i in at_risk]
            visible=min(0. if v is None else v for v in seen)
            gap=float(clear.min())
            progress=np.array([progress_at(path,float(at)) for at in times])
            residual=points-original.translation(np.clip(progress,0.,original.duration))
            distances=np.linalg.norm(residual,axis=1)
            trend=(distances.max()<=POLICY.companion_displacement_max_m and
                   np.linalg.norm(np.diff(residual,axis=0),axis=1).sum()<=POLICY.companion_travel_max_m)
            for i in np.argsort(distances)[-3:]:
                actual=frame_state(*pose_at(path,float(times[i])),model,default_intrinsics() if intrinsics is None else intrinsics)
                target=frame_state(*pose_at(original,float(progress[i])),model,default_intrinsics() if intrinsics is None else intrinsics)
                error=frame_error(actual,target)
                trend=trend and error['valid'] and error['view_angle_error_rad']<=POLICY.process_view_error_max_rad and error['log_area_error']<=POLICY.process_log_area_error_max
            # Physical clearance first. Shot deviations are costs, never a
            # hidden generator-side feasibility gate.
            support_penalty=0.
            if kinematic_search is not None:
                displacement=points[np.argmax(distances)]-positions_at(active,[times[np.argmax(distances)]])[0]
                length=np.linalg.norm(displacement)
                if length>1e-6:
                    support=np.linalg.norm((displacement/length)@kinematic_search.j[:3])
                    support_penalty=length/max(support,.02)
            timing_preference=(path.program['route']!='timing') if search_config=='timing' else False
            return (gap<.028, timing_preference if gap>=.028 else -gap,
                    not trend,visible<POLICY.visibility_floor,-visible,
                    path.program['amplitude_m'] if gap>=.028 else -gap,
                    round(support_penalty,1),abs(path.program.get('phase_rate',0.)),-min(gap,.08))
        result.sort(key=score)
    result=result[:budget.max_proposals]
    for path in result:
        path.seed_cache={}  # construction-only cache must not cross workers
    return result
