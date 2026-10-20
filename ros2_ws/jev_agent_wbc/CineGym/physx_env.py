"""External-policy Gym interface to the continuously running PhysX executor."""
from copy import deepcopy
import json
from multiprocessing.connection import Client
from pathlib import Path
import secrets
import subprocess
import tempfile
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.camera_intent import POLICY
from jev_agent_wbc.whole_body_qp import Limits
from .actions import validate_action
from .backends import start_closed_loop
from .episode import write_json
from .generator import generate_episode


class PhysXEnv:
    def __init__(self, config=None, *, output=None, isaac_python=None, python=None,
                 startup_timeout_s=90., observation_timeout_s=5., terminate_on_fault=True,
                 terminate_on_collision=True,
                 extra_args=(), dt=.02):
        if dt != .02:
            raise ValueError('PhysX control dt is .02 s; physics remains 100 Hz')
        if not all(np.isfinite(x) and x > 0 for x in (startup_timeout_s, observation_timeout_s)):
            raise ValueError('timeouts must be finite and positive')
        self.config, self.dt = config, dt
        self.output = Path(output) if output is not None else Path(tempfile.mkdtemp(prefix='cinegym-physx-'))
        self.isaac_python, self.python = isaac_python, python
        self.startup_timeout_s, self.observation_timeout_s = startup_timeout_s, observation_timeout_s
        self.terminate_on_fault, self.extra_args = bool(terminate_on_fault), tuple(extra_args)
        self.terminate_on_collision = bool(terminate_on_collision)
        self.limits = Limits()
        self.closed, self.done = False, True
        self.handle = self.connection = self.ipc = self.reader = None
        self.reader_stop = threading.Event()
        self.condition = threading.Condition()
        self.latest = self.current = None
        self.reader_error = None
        self.episode_number = 0

    def _release(self):
        self.reader_stop.set()
        if self.handle is not None:
            if self.reader_error or (self.current is not None and
                self.current['t'] >= self.reference.duration+POLICY.terminal_hold_s-1e-8):
                try:
                    self.handle.wait(timeout=15.)
                except subprocess.TimeoutExpired:
                    pass
            self.handle.close()
            self.handle = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.reader is not None:
            self.reader.join(timeout=1.)
            self.reader = None
        if self.ipc is not None:
            self.ipc.cleanup()
            self.ipc = None

    def reset(self, *, seed=None, episode=None):
        if self.closed:
            raise RuntimeError('environment is closed')
        if seed is not None and episode is not None:
            raise ValueError('supply an episode or a seed, not both')
        self._release()
        if episode is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0]) if seed is None else seed
            episode = generate_episode(seed, self.config)
        self.episode = deepcopy(episode)
        self.reference = self.episode.reference
        self.episode_number += 1
        folder = self.output/f'episode_{self.episode_number:04d}'
        case = self.episode.save(folder/'case')
        self.ipc = tempfile.TemporaryDirectory(prefix='cinegym-ipc-')
        descriptor = Path(self.ipc.name)/'connection.json'
        address, authkey = str(Path(self.ipc.name)/'socket'), secrets.token_bytes(32)
        write_json(descriptor, dict(address=address, authkey=authkey.hex()))
        descriptor.chmod(0o600)
        self.latest = self.current = None
        self.reader_error = None
        self.reader_stop.clear()
        started = time.monotonic()
        try:
            self.handle = start_closed_loop(case, folder/'run', backend='physx', mode='strict',
                isaac_python=self.isaac_python, python=self.python,
                extra_args=[*self.extra_args, '--cinegym-bridge', str(descriptor)])
            # The socket is established during equipment initialization, before
            # arming. No scene time is paused for a policy after that point.
            deadline = started+self.startup_timeout_s
            while self.connection is None:
                if self.handle.process.poll() is not None:
                    raise RuntimeError(f'PhysX startup failed; see {folder / "run/backend.log"}')
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'PhysX startup timed out; see {folder / "run/backend.log"}')
                if Path(address).exists():
                    try:
                        self.connection = Client(address, family='AF_UNIX', authkey=authkey)
                    except (FileNotFoundError, ConnectionRefusedError):
                        pass
                if self.connection is None:
                    time.sleep(.01)
            self.reader = threading.Thread(target=self._read, daemon=True, name='cinegym-observations')
            self.reader.start()
            self.current = self._next(-1, max(.01, deadline-time.monotonic()))
            self.limits = Limits(**self.current['action_limits'])
            self.done = False
            self.wall_start = time.monotonic()
            obs = self._observation()
            info = dict(backend='physx', seed=self.episode.metadata['seed'],
                episode_sha256=self.episode.metadata['sha256'], output=str(folder/'run'),
                physics_hz=100., pace=1., physics_waits_for_agent=False,
                control_status=deepcopy(self.current['control_status']))
            return obs, info
        except BaseException:
            self._release()
            self.done = True
            raise

    def _read(self):
        conn = self.connection
        try:
            while not self.reader_stop.is_set():
                if not conn.poll(.05):
                    continue
                packet = json.loads(conn.recv_bytes(16*1024*1024))
                if packet.get('type') == 'observation':
                    with self.condition:
                        self.latest = packet['observation']
                        self.latest['_evaluation'] = packet.get('evaluation')
                        self.condition.notify_all()
        except (OSError, EOFError, ValueError) as error:
            with self.condition:
                self.reader_error = f'{type(error).__name__}: {error}'
                self.condition.notify_all()

    def _next(self, sequence, timeout):
        deadline = time.monotonic()+timeout
        with self.condition:
            while self.latest is None or self.latest['sequence'] <= sequence:
                if self.reader_error or self.handle.process.poll() is not None:
                    raise RuntimeError('PhysX observation stream ended; inspect backend.log and summary.json')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('PhysX observation timeout; physics was not paused')
                self.condition.wait(min(.05, remaining))
            return deepcopy(self.latest)

    def _observation(self):
        obs = deepcopy(self.current)
        obs.pop('_evaluation', None)
        for name in ('state', 'measured_control', 'base_rotation', 'measured_base_twist', 'camera_position',
                     'camera_rotation', 'reference_position', 'reference_rotation', 'reference_twist'):
            if obs.get(name) is not None:
                obs[name] = np.asarray(obs[name], float)
        for name, value in obs['lidar'].items():
            if name.startswith('surfel_') and isinstance(value, list):
                obs['lidar'][name] = np.asarray(value)
        obs['task'] = self.reference.task
        return obs

    def step(self, action):
        if self.closed or self.done or self.current is None:
            raise RuntimeError('reset an open environment before stepping')
        u = validate_action(action, self.current['state'], self.current['measured_control'], self.dt, self.limits)
        source_sequence, old_time = self.current['sequence'], self.current['t']
        previous_audit = self.current.get('_evaluation') or {}
        stream_error = None
        try:
            self.connection.send_bytes(json.dumps(dict(type='action', sequence=source_sequence, action=u.tolist()),
                                                  allow_nan=False).encode())
            self.current = self._next(source_sequence, self.observation_timeout_s)
        except (OSError, EOFError, RuntimeError, TimeoutError) as error:
            stream_error = str(error)
            with self.condition:
                if self.latest is not None and self.latest['sequence'] > self.current['sequence']:
                    self.current = deepcopy(self.latest)
        obs = self._observation()
        status = obs['control_status']
        finished = obs['t'] >= self.reference.duration+POLICY.terminal_hold_s-1e-8
        fault = status.get('runtime_fault')
        audit = self.current.get('_evaluation') or {}
        collision = audit.get('collision_samples', 0) > previous_audit.get('collision_samples', 0)
        contact = audit.get('physical_contacts', 0) > previous_audit.get('physical_contacts', 0)
        terminated = bool(finished or ((collision or contact) and self.terminate_on_collision))
        truncated = bool(stream_error or (fault and self.terminate_on_fault)) and not terminated
        self.done = bool(terminated or truncated)
        position_error = float(np.linalg.norm(obs['camera_position']-obs['reference_position']))
        angle = float(Rotation.from_matrix(obs['camera_rotation']@obs['reference_rotation'].T).magnitude())
        elapsed = max(0., obs['t']-old_time)
        reward = -elapsed*(position_error+.2*angle)-float(collision or contact)
        info = dict(backend='physx', completed=finished, motion_complete=obs['t'] >= self.reference.duration,
            runtime_fault=fault, stream_error=stream_error, control_status=status,
            submitted_action_sequence=source_sequence,
            action_accepted=status.get('accepted_action_sequence', -1) >= source_sequence,
            physics_waits_for_agent=False, elapsed_sim_s=elapsed,
            position_error_m=position_error, orientation_error_rad=angle,
            elapsed_wall_s=time.monotonic()-self.wall_start, evaluation=audit or None,
            collision=None if not audit else bool(collision or contact),
            truth_clearance_m=audit.get('truth_clearance_m'), min_clearance_m=audit.get('min_clearance_m'),
            evaluation_delay_s=None if not audit else obs['t']-audit['stamp_s'],
            safety_scope='measured physical state and current-surface gates; full contact/safety results in backend summary')
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.closed = self.done = True
        self._release()
