"""Bounded nonblocking controller transport; physics never awaits a trajectory.

Only worker initialization happens before arming. No planning/semantic call is
made before the first live observation. Queues hold at most one snapshot each.
"""
from dataclasses import asdict
import json
import multiprocessing as mp
import os
import pickle
from queue import Empty, Full
import signal
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .async_journal import AsyncJournal
from .candidates import path_spec, restore_path
from .contracts import native
from .perception import SubjectEvidence
from .planning import joint_margin
from .realtime import CommandWatchdog


def offer(queue, value):
    """Drop old queued data, never wait for a slow producer/consumer."""
    try:
        queue.put_nowait(value)
        return True
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            return False
        try:
            queue.put_nowait(value)
            return True
        except Full:
            return False


def controller_process(inbox, outbox, stop, original, scene, output, options, physical_limits):
    # Own process group includes the numerical grandchildren. Bounded shutdown
    # can kill this group without touching another simulation or user's process.
    os.setsid()
    from physx_compat import install_worker_adapter
    install_worker_adapter()
    from .engine import CinematicEngine
    engine = None
    try:
        engine = CinematicEngine(original, scene, output, **options)
        engine.wbc.measured_limits = physical_limits
        engine.warm_workers()
        offer(outbox, pickle.dumps(dict(ready=True, summary=engine.summary(), limits=engine.wbc.limits)))
        while not stop.is_set():
            try:
                snapshot = pickle.loads(inbox.get(timeout=.05))
            except Empty:
                continue
            if stop.is_set():
                break
            camera = snapshot.pop('camera')
            engine.perception = snapshot.pop('perception')
            sequence = snapshot.pop('sequence')
            wall = snapshot.pop('observation_wall')
            engine.set_camera_output(**camera)
            # Anchor rate constraints to actual measured velocity, including
            # commands held/braked by the independent actuator supervisor.
            engine.wbc.previous = np.asarray(snapshot['measured_control']).copy()
            command, info = engine.step(**snapshot)
            engine._write('controller_step_completed', snapshot['t'], sequence=sequence,
                observation_wall=wall, completed_wall=time.monotonic(), loop_wall_s=info['loop_wall_s'])
            result = dict(t=snapshot['t'], observation_wall=wall, sequence=sequence,
                          command=command, info=info, active=path_spec(engine.active),
                          feedback=engine.actual_feedback, summary=engine.summary())
            offer(outbox, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as error:
        if engine:
            engine._write('controller_failed', 0., reason=f'{type(error).__name__}: {error}')
        offer(outbox, pickle.dumps(dict(error=f'{type(error).__name__}: {error}')))
    finally:
        if engine:
            engine.close()


class RealtimeEngine:
    def __init__(self, original, scene, output, *, physical_limits, runtime_policy, **options):
        self.original = self.active = original
        self.output = output
        self.policy = runtime_policy
        self.intrinsics = options.get('intrinsics')
        self.perception = SubjectEvidence(original.task['subject']['id'])
        self.wbc = SimpleNamespace(limits=options['limits'], previous=np.zeros(8))
        self.actual_feedback = {}
        self.installed_plans = 0
        self.jev = SimpleNamespace(decider=SimpleNamespace(client=SimpleNamespace(calls=0)))
        self.last_result = None
        self.camera = None
        self.sequence = 0
        self.watchdog = None
        self.closed = False
        self.worker_halted = False
        self.fault_logged = False
        self.journal = AsyncJournal(output/'runtime_events.jsonl')
        ctx = mp.get_context('spawn')
        self.inbox, self.outbox, self.stop = ctx.Queue(1), ctx.Queue(1), ctx.Event()
        self.process = ctx.Process(target=controller_process,
            args=(self.inbox, self.outbox, self.stop, original, scene, output,
                  dict(options, runtime_policy=runtime_policy), physical_limits), name='jev-live-controller')
        self.process.start()
        # Startup is equipment initialization only, bounded independently of shot.
        try:
            message = pickle.loads(self.outbox.get(timeout=45.))
            if not message.get('ready'):
                raise RuntimeError(message.get('error', 'controller did not initialize'))
            self._summary = message['summary']
            self.wbc.limits = message.get('limits', self.wbc.limits)
        except BaseException:
            self.close()
            raise
        self._write('controller_ready', 0., policy=asdict(runtime_policy), planning_before_arm=False)

    def _write(self, event, t, **details):
        self.journal.write(json.dumps(native(dict(event=event, time_s=t, **details)), allow_nan=False)+'\n')

    def set_camera_output(self, stamp_s, position, rotation, *, source, **kwargs):
        self.camera = dict(stamp_s=stamp_s, position=position, rotation=rotation, source=source, **kwargs)

    def arm(self, wall):
        if self.watchdog is not None:
            raise RuntimeError('a latched session cannot be rearmed; create a new session')
        self.watchdog = CommandWatchdog(self.policy, wall)

    def trip(self, reason, t):
        self.watchdog.trip(reason, t, time.monotonic())

    def escalate(self, reason, t):
        previous = self.watchdog.emergency_stop_required
        self.watchdog.escalate(reason, t, time.monotonic())
        if previous is None:
            self._write('emergency_stop_required', t, evidence=self.watchdog.emergency_stop_required,
                        automatic_rearm=False, physical_world_continues=True)

    def step(self, state, scene, t, dt, *, observation_wall=None, **kwargs):
        started = time.monotonic()
        if self.watchdog is None:
            self.arm(started)
        try:
            packet = pickle.loads(self.outbox.get_nowait())
        except Empty:
            packet = None
        if packet is not None:
            if 'error' in packet:
                self.trip('controller_error: '+packet['error'], t)
            else:
                self._summary = packet['summary']
                self.installed_plans = self._summary['installed_plans']
                self.jev.decider.client.calls = self._summary['jev_calls']
                if packet['info'].get('runtime_fault'):
                    self.trip(packet['info']['runtime_fault']['reason'], t)
                if self.watchdog.accept(packet, t, time.monotonic()):
                    self.last_result = packet
                    self.active = restore_path(self.original, packet['active'])
        command = self.watchdog.command(t, time.monotonic(), float(scene['observation_stamp']),
                                        alive=self.process.is_alive())
        self.actual_feedback = dict(camera_output_source=self.camera['source'], stamp_s=t,
            camera_position=np.asarray(self.camera['position']).tolist(),
            camera_quaternion_xyzw=Rotation.from_matrix(self.camera['rotation']).as_quat().tolist())
        submitted = False
        if self.watchdog.fault is None:
            self.sequence += 1
            snapshot = dict(state=state, scene=scene, t=t, dt=dt, **kwargs,
                camera=self.camera, perception=self.perception, sequence=self.sequence,
                observation_wall=started if observation_wall is None else observation_wall)
            # Freeze before Queue's asynchronous serialization thread can race
            # with mutable sensor buffers. No wait for an agent response.
            submitted = offer(self.inbox, pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL))
        else:
            self.stop.set()
            if not self.worker_halted:
                self.worker_halted = True
                # Nonblocking cancellation includes hung numerical/HTTP work;
                # never let abandoned planning compete with the drive loop.
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if not self.fault_logged and self.watchdog.fault is not None:
            self._write('actuator_fault_latched', t, fault=self.watchdog.fault)
            self.fault_logged = True
        info = dict(self.last_result['info']) if self.last_result else {}
        info.setdefault('target_position', self.active.sample(t)[0].tolist())
        info.setdefault('original_position', self.original.sample(t)[0].tolist())
        info.update(joint_margin_rad=joint_margin(np.asarray(state)),
            runtime_fault=self.watchdog.fault, safety_brake=self.watchdog.fault is not None,
            command_source_time_s=None if self.last_result is None else self.last_result['t'],
            command_source_wall_s=None if self.last_result is None else self.last_result['observation_wall'],
            command_source_sequence=None if self.last_result is None else self.last_result['sequence'],
            observation_enqueued=submitted, agent_camera_input_role='offered_to_async_transport',
            controller_feedback=None if self.last_result is None else self.last_result['feedback'],
            command_certificate_scope='worker snapshot only; current-state sweep required by actuator',
            actuator_poll_wall_s=time.monotonic()-started,
            loop_wall_s=None if self.last_result is None else self.last_result['info']['loop_wall_s'])
        # Missing worker timing is explicitly separate from a measured zero.
        if info['loop_wall_s'] is None:
            info['loop_wall_s'] = 0.
            info['controller_timing_missing'] = True
        self.wbc.previous = command.copy()
        return command, info

    def summary(self):
        return dict(self._summary, runtime_policy=asdict(self.policy),
                    actuator_fault=None if self.watchdog is None else self.watchdog.fault,
                    emergency_stop_required=None if self.watchdog is None else self.watchdog.emergency_stop_required,
                    rejected_command_packets=0 if self.watchdog is None else self.watchdog.rejected,
                    physics_waits_for_agent=False)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        # Outside the physical loop. Kill descendants even if their parent has
        # exited; the dedicated session id is the spawned controller's pid.
        self.process.join(timeout=1.)
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.process.join(timeout=.5)
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.process.is_alive():
            self.process.kill()
        self.process.join(timeout=.5)
        for queue in (self.inbox, self.outbox):
            queue.cancel_join_thread()
            queue.close()
        self.journal.close()
