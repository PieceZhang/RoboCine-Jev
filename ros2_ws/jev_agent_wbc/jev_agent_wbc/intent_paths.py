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
        at=float(np.clip(t,0.,path.duration))
        return path.translation(at),path.rotation(at).as_matrix()
    return path.sample(t)[:2]


class IntentPath:
    """An immutable exact path with an unchanged executed/activation prefix.

    Timing uses monotone smooth speed pulses or a quintic warp of remaining motion. Spatial
    freedom lives on the shot's primary feature level sets. The nominal path is
    one seed, not the selection objective. No joint state or robot model is used.
    """
    def __init__(self, original, previous, activation_s, program, model=None,
                 intrinsics=None, geometry=None):
        self.original, self.previous = original, previous
        self.activation_s = float(activation_s)
        self.program = dict(program)
        if self.program.get('timing_profile') in ('early_fast','early_slow'):
            fast=self.program['timing_profile']=='early_fast'
            self.program.setdefault('tempo_rate',2.5 if fast else .85)
            self.program.setdefault('tempo_ramp_s',.4 if fast else .6)
        self.model = model
        self.intrinsics = default_intrinsics() if intrinsics is None else intrinsics
        self.extension_s = float(program.get('extension_s', 0.))
        self.duration = previous.duration+self.extension_s
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
        u = np.clip((t-self.activation_s)/(self.duration-self.activation_s), 0., 1.)
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
        motion = self.original.phase(ref_t)['motion'].lower()
        nominal, nr = pose_at(self.original,ref_t)
        radial = nominal-self.subject
        radial[2] = 0.
        radial /= max(np.linalg.norm(radial), 1e-8)
        left = np.cross(radial, [0., 0., 1.])
        up = np.array([0., 0., 1.])
        route = self.program['route']
        offsets = dict(left=left, right=-left, above=up, below=-up,
            outer=radial, inner=-radial, outer_above=(radial+up)/np.sqrt(2),
            left_above=(left+up)/np.sqrt(2), right_above=(-left+up)/np.sqrt(2),
            timing=np.zeros(3), continuation=np.zeros(3))
        offset = offsets[route].copy()
        if motion == 'orbit':
            # Preserve the *same* azimuth sector, not just angular travel.
            offset = offset@radial*radial+offset[2]*up
        elif 'truck' in motion:
            phase = next(s for s in self.original.task['shot_phase'] if s['motion'] == motion)
            axis = self.original.sample(phase['start_s'])[1][:, 1]
            offset -= (offset@axis)*axis
        elif 'pedestal' in motion:
            offset[2] = 0.
        p = p+amplitude*offset
        if motion in ('movein', 'moveout') and self.model is not None:
            target = projected_area(nominal, nr, self.model, self.intrinsics)
            framing = look_at(nominal, self.subject).T@nr
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
                         allow_time_extension=True, scene=None):
    activation = t+budget.activation_lead_s
    if activation >= active.duration-.3:
        return []
    motion = original.phase(progress_at(active, activation))['motion'].lower()
    routes = (['outer', 'inner', 'above', 'below'] if motion == 'orbit' else
              ['left', 'right', 'outer', 'inner'] if 'pedestal' in motion else
              ['outer', 'inner', 'above', 'below'] if 'truck' in motion else
              ['left', 'right', 'above', 'below'])
    programs = [dict(route='continuation', amplitude_m=0., extension_s=0.)]
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
    for amplitude in (.12,.22):
        programs += [dict(route=route,amplitude_m=amplitude,extension_s=0.,profile='arch') for route in routes]
    fast=[v for v in programs if v.get('timing_profile')=='early_fast']
    if fast:
        advance=min(v['extension_s'] for v in fast)
        programs += [dict(route=route,amplitude_m=.08,extension_s=advance,profile='arch',timing_profile='early_fast') for route in routes]
    if strategy in ('view','framing'):
        # The Agent's family affects the search order, never the numeric gates.
        preferred = ('above','below') if strategy=='framing' else ('inner','outer')
        programs.sort(key=lambda v:(v['route'] not in preferred,v['amplitude_m']))
    encounter=None
    if scene is not None and scene.get('valid') and len(scene.get('centers',[])):
        times=np.linspace(activation,active.duration,81)
        points=positions_at(active,times)
        centers=np.asarray(scene['centers'])[None]+(times-scene['observation_stamp'])[:,None,None]*np.asarray(scene['velocities'])[None]
        gap=np.min(np.linalg.norm(points[:,None]-centers,axis=2)-np.asarray(scene['radii'])[None],axis=1)
        encounter=float(times[np.argmin(gap)])
    result = []
    for program in programs[:budget.max_proposals]:
        if program.get('timing_profile') != 'early_slow' and program['extension_s']/(active.duration+program['extension_s']-activation) > .52:
            continue
        if encounter is not None and program['amplitude_m']:
            duration=active.duration+program['extension_s']
            program['peak_s']=float(np.clip(encounter,activation+min(.6,(duration-activation)/3),
                                           duration-min(.6,(duration-activation)/3)))
        result.append(IntentPath(original, active, activation, program, model, intrinsics))
    return result
