"""Bounded local IPC for an external velocity policy. Never blocks physics.

Only the controller process touches sockets. Both directions retain one latest
message. Commands keep the exact source observation's sequence and timestamps;
the existing two-clock watchdog is the final authority for their validity.
"""
from collections import OrderedDict
from dataclasses import asdict
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import pickle
from queue import Empty, Queue
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.candidates import path_spec
from jev_agent_wbc.contracts import native
from jev_agent_wbc.planning import joint_margin
from jev_agent_wbc.realtime_engine import offer
from .actions import validate_action


def observation(snapshot, original, limits=None):
    camera = snapshot['camera']
    target, rotation, twist = original.sample(snapshot['t'])
    result = dict(t=snapshot['t'], dt=snapshot['dt'], sequence=snapshot['sequence'],
        observation_wall=snapshot['observation_wall'], state=snapshot['state'],
        measured_control=snapshot['measured_control'], base_z=snapshot['base_z'],
        base_rotation=snapshot.get('base_rotation'),
        measured_base_twist=snapshot.get('measured_base_twist'),
        camera_position=camera['position'], camera_rotation=camera['rotation'],
        camera_source=camera['source'], reference_position=target,
        reference_rotation=rotation, reference_twist=twist,
        lidar=snapshot['scene'], control_status=snapshot.get('control_status', {}))
    if limits is not None:
        result['action_limits'] = asdict(limits)
    return result


def action_packet(action, snapshot, original, limits, summary):
    """Build against the source observation, never rebase an old action to now."""
    started = time.monotonic()
    u = validate_action(action, snapshot['state'], snapshot['measured_control'], snapshot['dt'], limits)
    camera = snapshot['camera']
    p, r, _ = original.sample(snapshot['t'])
    position_error = float(np.linalg.norm(np.asarray(camera['position'])-p))
    angle = float(Rotation.from_matrix(np.asarray(camera['rotation'])@r.T).magnitude())
    info = dict(position_error=position_error, orientation_error=angle,
                joint_margin_rad=joint_margin(np.asarray(snapshot['state'])), qp_violation=0.,
                target_position=p.tolist(), original_position=p.tolist(),
                loop_wall_s=time.monotonic()-started, command_source='cinegym_external_velocity_policy')
    feedback = dict(camera_output_source=camera['source'], stamp_s=snapshot['t'],
        camera_position=np.asarray(camera['position']).tolist(),
        camera_quaternion_xyzw=Rotation.from_matrix(camera['rotation']).as_quat().tolist())
    return dict(t=snapshot['t'], observation_wall=snapshot['observation_wall'],
        sensor_stamp=float(snapshot['scene']['observation_stamp']),
        sequence=snapshot['sequence'], command=u, info=info, active=path_spec(original),
        feedback=feedback, summary=summary)


def controller_process(inbox, outbox, stop, original, scene, output, options, physical_limits):
    os.setsid()
    # Respect operating reserves as well as the actuator's physical limits.
    limits = options['limits']
    descriptor = json.loads(Path(options['bridge_descriptor']).read_text())
    listener = Listener(descriptor['address'], family='AF_UNIX', authkey=bytes.fromhex(descriptor['authkey']))
    os.chmod(descriptor['address'], 0o600)
    incoming, outgoing = Queue(1), Queue(1)
    disconnected = threading.Event()
    connection = []

    def transport():
        try:
            conn = listener.accept()
            connection.append(conn)
            while not stop.is_set():
                if conn.poll(.002):
                    request = json.loads(conn.recv_bytes(16384))
                    if request.get('type') == 'action':
                        offer(incoming, request)
                try:
                    payload = outgoing.get_nowait()
                except Empty:
                    continue
                conn.send_bytes(payload)
        except (EOFError, OSError, ValueError):
            disconnected.set()
        finally:
            for conn in connection:
                conn.close()

    thread = threading.Thread(target=transport, daemon=True, name='cinegym-socket')
    thread.start()
    summary = dict(mode='external', installed_plans=0, jev_calls=0, original_duration_s=original.duration,
                   controller='CineGym external velocity policy', physics_waits_for_agent=False)
    offer(outbox, pickle.dumps(dict(ready=True, summary=summary, limits=limits)))
    history = OrderedDict()
    last_action = -1
    fault_sent = False
    try:
        while not stop.is_set():
            try:
                snapshot = pickle.loads(inbox.get(timeout=.005))
                history[snapshot['sequence']] = snapshot
                if len(history) > 32:
                    history.popitem(last=False)
                offer(outgoing, json.dumps(native(dict(type='observation', observation=observation(snapshot, original, limits),
                                                       evaluation=snapshot.get('evaluation'))),
                                           allow_nan=False, separators=(',', ':')).encode())
            except Empty:
                pass
            if disconnected.is_set() and not fault_sent:
                offer(outbox, pickle.dumps(dict(error='external_policy_disconnected')))
                fault_sent = True
            try:
                request = incoming.get_nowait()
            except Empty:
                continue
            sequence = request.get('sequence')
            if type(sequence) is not int or sequence <= last_action or sequence not in history:
                continue
            source = history[sequence]
            # Telemetry can continue after a fault; external commands cannot.
            latest = next(reversed(history.values()))
            if latest.get('control_status', {}).get('runtime_fault'):
                continue
            last_action = sequence
            try:
                packet = action_packet(request['action'], source, original, limits, summary)
                offer(outbox, pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL))
            except (KeyError, ValueError, TypeError) as error:
                offer(outbox, pickle.dumps(dict(error='invalid_external_action: '+str(error))))
    except Exception as error:
        offer(outbox, pickle.dumps(dict(error=f'CineGym bridge: {type(error).__name__}: {error}')))
    finally:
        listener.close()
        for conn in connection:
            conn.close()
