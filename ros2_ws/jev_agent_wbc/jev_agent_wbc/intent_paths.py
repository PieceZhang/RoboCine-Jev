"""Agent-selected motion programs instantiated as smooth camera-space paths."""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

from .camera_intent import POLICY, default_intrinsics, projected_area, progress_at
from .reference import look_at, smoothstep


STRATEGIES = {
    'framing': 'Propose radial and height routes, preserve directed intent ranges; also test slower passage.',
    'view': 'Propose both sides and height routes that preserve subject-relative motion; also test slower passage.',
    'joint': 'Propose alternate sides, radii and heights together, with bounded time extensions.',
    'timing': 'Propose bounded slower passage through the moving obstacle, plus spatial escape alternatives.',
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
    return path.pose(t) if isinstance(path, IntentPath) else path.sample(t)[:2]


class IntentPath:
    """An immutable exact path with an unchanged executed/activation prefix.

    Timing is a monotone quintic warp of the parent's remaining motion. Spatial
    freedom lives on the shot's primary feature level sets. The nominal path is
    one seed, not the selection objective. No joint state or robot model is used.
    """
    def __init__(self, original, previous, activation_s, program, model=None,
                 intrinsics=None, geometry=None):
        self.original, self.previous = original, previous
        self.activation_s = float(activation_s)
        self.program = dict(program)
        self.model = model
        self.intrinsics = default_intrinsics() if intrinsics is None else intrinsics
        self.extension_s = float(program.get('extension_s', 0.))
        self.duration = previous.duration+self.extension_s
        self.subject = original.subject
        self.return_s = None if program.get('profile') == 'hold' else self.duration
        self.values = np.zeros(6)  # Legacy diagnostics only; program is authoritative.
        self.envelope = SimpleNamespace(end_s=self.duration, decision_id='intent-program',
            blend_s=self.duration-self.activation_s, terminal='complete_intent', bounds={})
        if self.duration <= self.activation_s or self.extension_s < 0:
            raise ValueError('invalid remaining motion duration')
        if self.duration > POLICY.max_duration_factor*original.duration+1e-8:
            raise ValueError('intent duration limit exceeded')
        if self.extension_s/(self.duration-self.activation_s) > .52:
            raise ValueError('time warp would reverse progress')
        if geometry is None:
            times = np.linspace(self.activation_s, self.duration, 81)
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
        return self.activation_s+elapsed-self.extension_s*smoothstep(u)

    def progress_time(self, t):
        return progress_at(self.previous, self.parent_time(t))

    def _construct_position(self, t):
        source_t = self.parent_time(t)
        p, r, _ = self.previous.sample(source_t)
        u = np.clip((t-self.activation_s)/(self.duration-self.activation_s), 0., 1.)
        bump = (smoothstep((t-self.activation_s)/self.program['rise_s'])
                if self.program.get('profile') == 'hold' else 64*u**3*(1-u)**3)
        amplitude = float(self.program.get('amplitude_m', 0.))*bump
        if not amplitude:
            return p
        ref_t = self.progress_time(t)
        motion = self.original.phase(ref_t)['motion'].lower()
        nominal, nr, _ = self.original.sample(ref_t)
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
            for _ in range(5):
                pr = look_at(p, self.subject)@framing
                area = projected_area(p, pr, self.model, self.intrinsics)
                if area is None or target is None:
                    break
                if abs(area/target-1.) < 1e-8:
                    break
                p = self.subject+(p-self.subject)*np.sqrt(area/target)
        return p

    def pose(self, t):
        if t <= self.activation_s or self.program['route'] == 'continuation':
            return self.previous.sample(t)[:2]
        p = self.translation(np.clip(t, self.activation_s, self.duration))
        base_p, base_r, _ = self.previous.sample(self.parent_time(t))
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
                         allow_time_extension=True):
    activation = t+budget.activation_lead_s
    if activation >= active.duration-.3:
        return []
    motion = original.phase(progress_at(active, activation))['motion'].lower()
    routes = (['outer', 'inner', 'above', 'below'] if motion == 'orbit' else
              ['left', 'right', 'outer', 'inner'] if 'pedestal' in motion else
              ['outer', 'inner', 'above', 'below'] if 'truck' in motion else
              ['left', 'right', 'above', 'below'])
    programs = [dict(route='continuation', amplitude_m=0., extension_s=0.)]
    # Distinct sides, radii/heights, and passage times. All complete the same
    # feature interval; the agent's search family is a program, not permission
    # to shrink the intended range.
    for amplitude in (.35, .65):
        programs += [dict(route=route, amplitude_m=amplitude, extension_s=0.,
            profile='hold' if amplitude == .35 else 'arch',
            rise_s=min(1.2,(active.duration-activation)/2.)) for route in routes]
    spare = POLICY.max_duration_factor*original.duration-active.duration if allow_time_extension else 0.
    for extension in (2., 4.):
        if extension <= spare+1e-8:
            programs.append(dict(route='timing', amplitude_m=0., extension_s=extension))
    if strategy == 'joint':
        # Coupled routes augment the cardinal alternatives; they must not remove
        # the inward route (especially when an outer detour hides the subject).
        coupled = ['outer_above', 'left_above'] if 'pedestal' not in motion else ['left_above', 'right_above']
        programs += [dict(route=route, amplitude_m=.5, extension_s=0.) for route in coupled]
    if spare >= 2.:
        programs += [dict(route=route, amplitude_m=.35, extension_s=2.) for route in routes]
    if strategy == 'timing':
        programs.sort(key=lambda v: (v['extension_s'] == 0., v['amplitude_m']))
    result = []
    for program in programs[:budget.max_proposals]:
        if program['extension_s']/(active.duration+program['extension_s']-activation) > .52:
            continue
        result.append(IntentPath(original, active, activation, program, model, intrinsics))
    return result
