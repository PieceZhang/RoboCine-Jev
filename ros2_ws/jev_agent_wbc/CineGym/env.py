"""Unified Gym-style CPU / PhysX interaction without requiring Gymnasium.

Policy input is measured state plus the existing sparse LiDAR observation. Truth
clearances live only in evaluation info. CPU integration is not a PhysX result.
"""
from copy import deepcopy
import time
import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.kc_svsdf import advance, CollisionGeometry
from jev_agent_wbc.sensors import RuntimeLidar
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
from .generator import generate_episode
from .actions import validate_action


class CPUEnv:
    """reset -> (observation, info); step -> (obs, reward, terminated, truncated, info).

    Action: eight physical velocities [base_v, base_yaw_rate, dq1..dq6]. Invalid
    commands are rejected, never silently clipped. Call run_closed_loop for the
    existing Agent or choose CineGymEnv(backend='physx') for physical interaction.
    """
    def __init__(self, config=None, *, dt=.02, collision_step_s=.005, terminate_on_collision=True):
        if not np.isfinite(dt) or not .005 <= dt <= .1:
            raise ValueError('dt must be within 0.005..0.1 seconds')
        if not np.isfinite(collision_step_s) or not 0 < collision_step_s <= dt:
            raise ValueError('collision_step_s must be positive and <= dt')
        self.config, self.dt = config, float(dt)
        self.collision_step_s = float(collision_step_s)
        self.terminate_on_collision = bool(terminate_on_collision)
        self.limits = Limits()
        self.geometry = CollisionGeometry()
        self.episode = None
        self.closed = False
        self.done = True

    def reset(self, *, seed=None, episode=None):
        if self.closed:
            raise RuntimeError('environment is closed')
        if episode is not None and seed is not None:
            raise ValueError('supply an episode or a seed, not both')
        if episode is None:
            if seed is None:
                seed = int(np.random.SeedSequence().generate_state(1)[0])
            episode = generate_episode(seed, self.config)
        self.episode = deepcopy(episode)
        self.reference, self.runtime = self.episode.reference, self.episode.runtime
        self.state = np.array(self.episode.scene['initial_state'], float)
        self.z = float(self.episode.scene['base_z'])
        self.t = 0.
        self.previous = np.zeros(8)
        self.lidar = RuntimeLidar()
        for stamp in np.linspace(-.8, 0., 9):
            self.observed = self.lidar.synthetic_scan(self.runtime, self.state, self.z, float(stamp))
        self.done = False
        self.collided = False
        self.errors, self.orientation_errors = [], []
        self.min_clearance = self._clearance(self.state, self.t)
        self.wall_start = time.monotonic()
        return self._observation(), dict(seed=self.episode.metadata['seed'],
            episode_sha256=self.episode.metadata['sha256'], backend='cpu', pace=1.)

    def _clearance(self, state, t):
        truth = self.runtime.truth_collision(t)
        ds = self.geometry.distances(state,truth['surfel_centers'],self.z)
        return float((ds-truth['surfel_radii'][:,None]).min())

    def _observation(self):
        p, r, _ = whole_body_kinematics(self.state, self.z)
        target, rotation, twist = self.reference.sample(self.t)
        return dict(t=self.t, dt=min(self.dt, max(0., self.reference.duration-self.t)),
            state=self.state.copy(), measured_control=self.previous.copy(), base_z=self.z,
            camera_position=p, camera_rotation=r,
            reference_position=target, reference_rotation=rotation, reference_twist=twist,
            # The upstream task is known; future obstacle keyframes are not.
            task=self.reference.task, lidar=deepcopy(self.observed))

    def step(self, action):
        if self.closed or self.done or self.episode is None:
            raise RuntimeError('reset an open environment before stepping')
        dt = min(self.dt, self.reference.duration-self.t)
        u = validate_action(action, self.state, self.previous, dt, self.limits)
        candidate = advance(self.state, u, dt)
        # Check within the control interval, so a fast crossing cannot jump
        # straight from one collision-free endpoint to another unnoticed.
        offsets = np.linspace(0., dt, int(np.ceil(dt/self.collision_step_s))+1)
        clearance = min(self._clearance(advance(self.state, u, s), self.t+s) for s in offsets)
        collision = clearance < 0.
        self.min_clearance = min(self.min_clearance, clearance)
        self.collided |= collision
        self.state, self.previous = candidate, u.copy()
        self.t = min(self.reference.duration, self.t+dt)
        if self.t-self.lidar.last >= .1-1e-9:
            self.observed = self.lidar.synthetic_scan(self.runtime, self.state, self.z, self.t)
        observation = self._observation()
        error = float(np.linalg.norm(observation['camera_position']-observation['reference_position']))
        angle = float(Rotation.from_matrix(observation['camera_rotation']@observation['reference_rotation'].T).magnitude())
        self.errors.append(error); self.orientation_errors.append(angle)
        finished = self.t >= self.reference.duration-1e-9
        terminated = bool(finished or (collision and self.terminate_on_collision))
        self.done = terminated
        remaining = self.t-(time.monotonic()-self.wall_start)
        if remaining > 0.:
            time.sleep(remaining)  # pace=1; never add deliberate slowdown
        info = dict(backend='cpu', truth_clearance_m=clearance, min_clearance_m=self.min_clearance,
            collision=collision, collided=self.collided, position_error_m=error, orientation_error_rad=angle,
            completed=finished, elapsed_wall_s=time.monotonic()-self.wall_start,
            collision_audit_step_s=self.collision_step_s,
            safety_scope='sampled kinematic full-body clearance against spheres and conservative cuboid volume covers; no physical contact measurement')
        if terminated:
            info.update(position_rms_m=float(np.sqrt(np.mean(np.square(self.errors)))),
                        orientation_rms_rad=float(np.sqrt(np.mean(np.square(self.orientation_errors)))))
        reward = -dt*(error+.2*angle)-float(collision)
        return observation, float(reward), terminated, False, info

    def close(self):
        self.closed = self.done = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CineGymEnv:
    """Unified reset/step API. PhysX advances independently of the caller.

    CPU options are passed to CPUEnv; PhysX options to PhysXEnv. Both expose
    measured observations and eight velocity actions in physical units.
    """
    def __init__(self, config=None, *, backend='cpu', **options):
        if backend == 'cpu':
            self._env = CPUEnv(config, **options)
        elif backend == 'physx':
            from .physx_env import PhysXEnv
            self._env = PhysXEnv(config, **options)
        else:
            raise ValueError('backend must be cpu or physx')
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *, seed=None, episode=None):
        return self._env.reset(seed=seed, episode=episode)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
