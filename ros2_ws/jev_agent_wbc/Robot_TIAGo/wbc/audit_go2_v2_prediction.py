#!/usr/bin/env python3
"""Read-only v2 forecast audit using independent continuous ODE integration.

Synthetic operating-model snapshots only: no learned policy/physics rollout.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

DT = .02
SLEW = np.array([.5, .5, .8])
CAP = np.array([.2, .2, .2])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def independent_commands(first, future, duration, count, terminal):
    values = [np.asarray(first).copy()]
    for index in range(1, count):
        target = future if not terminal or (index+1)*DT <= duration+1e-9 else np.zeros(3)
        values.append(values[-1] + np.clip(target-values[-1], -SLEW*DT, SLEW*DT))
    return np.asarray(values)


def independent_continuous_model(body, link, measured, history, commands, horizon, queries):
    """Solve vx ODE + full tilted-body planar kinematics with actual held input.

    This does not import either candidate's trajectory implementation. Native
    delayed inputs and yaw integration are evaluated from first principles.
    """
    times = np.asarray([t for t, _ in history]); inputs = np.asarray([u for _, u in history])
    def delayed_input(t):
        t = t-.15
        if t < -1e-11:
            index = max(0, np.searchsorted(times, t+1e-11, side='right')-1)
            return inputs[index]
        index = min(len(commands)-1, max(0, int(np.floor((t+1e-11)/DT))))
        return commands[index]
    pending = delayed_input(0.)
    boundaries = np.unique(np.r_[0., horizon, queries, .15, times+.15,
        np.arange(len(commands))*DT+.15])
    boundaries = boundaries[(boundaries >= 0.) & (boundaries <= horizon+1e-12)]
    state = np.array([measured[0], 0., 0., 0.])
    snapshots = {0.: state.copy()}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end-start < 1e-13:
            snapshots[float(end)] = state.copy(); continue
        u = delayed_input((start+end)/2)
        steady = u[0]*(.55 if u[0] < 0 else 1.)
        def rhs(t, value):
            residual = max(0., 1.-t/.15)
            vy = u[1] + (measured[1]-pending[1])*residual
            wz = u[2] + (measured[2]-pending[2])*residual
            body_velocity = body[:3, :3] @ np.array([value[0], vy, 0.])
            c, s = np.cos(value[3]), np.sin(value[3])
            return [(steady-value[0])/.35,
                c*body_velocity[0]-s*body_velocity[1],
                s*body_velocity[0]+c*body_velocity[1], wz]
        solved = solve_ivp(rhs, (start, end), state, rtol=1e-11, atol=1e-13, max_step=.001)
        if not solved.success: raise RuntimeError(solved.message)
        state = solved.y[:, -1]; snapshots[float(end)] = state.copy()
    local = np.linalg.inv(body) @ link
    result = []
    for query in queries:
        key = min(snapshots, key=lambda t: abs(t-query))
        if abs(key-query) > 1e-9: raise RuntimeError('independent query not integrated')
        value = snapshots[key]; root = body.copy()
        root[:3, :3] = Rotation.from_euler('z', value[3]).as_matrix() @ body[:3, :3]
        root[:2, 3] += value[1:3]
        result.append(root @ local)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); robot = args.robot_dir.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False); sources = out/'sources'; sources.mkdir()
    files = ['camera_coordinator.py', 'bounded_body_prediction.py', 'terminal_yaw_allocator.py',
             'terminal_allocator.py', 'wbc_backend.py', 'adaptive_runner.py']
    for name in files: shutil.copy2(robot/name, sources/name)
    shutil.copy2(__file__, sources/Path(__file__).name)
    backend = load('go2_v2_independent_backend', sources/'wbc_backend.py')
    model = load('go2_v2_independent_forecast', sources/'bounded_body_prediction.py')
    terminal = load('go2_v2_independent_terminal', sources/'terminal_yaw_allocator.py')
    body = np.eye(4); body[:3, :3] = Rotation.from_euler('xyz', [.08, -.11, .43]).as_matrix()
    body[:3, 3] = [.2, -.3, .28]
    local = np.eye(4); local[:3, 3] = [.08, -.03, .09]
    local[:3, :3] = Rotation.from_euler('xyz', [.02, .04, -.07]).as_matrix()
    link = body @ local; stamp = 5.; queries = np.array([0., .2, .4, .6, .8, 1., 1.2])
    reports = []
    for sign in [-1., 1.]:
        history = [(stamp-k*DT, np.array([.04+.02*np.sin(k*.1), -.03+.01*np.sin(k*.2), sign*(.08+.02*np.sin(k*.15))]))
                   for k in range(32, 0, -1)]
        previous = history[-1][1]; measured = np.array([.026, -.017, sign*.047])
        first = previous + np.array([-.01, .01, -sign*.016]); future = np.array([-.15, .15, -sign*.15])
        moving = model.BoundedBodyPrediction(backend, stamp_s=stamp, body=body, link=link,
            measured_velocity=measured, previous_applied=previous, input_history=history)
        current = moving.trajectory(first, future)
        independent = independent_commands(first, future, 0., len(current['issued']), False)
        np.testing.assert_allclose(current['issued'], independent, atol=1e-14, rtol=0)
        expected = independent_continuous_model(body, link, measured,
            [(t-stamp, u) for t, u in history], independent, 1.2, queries)
        actual = [moving.predict_link(first, future, t) for t in queries]
        position_error = max(float(np.linalg.norm(a[:3, 3]-b[:3, 3])) for a, b in zip(actual, expected))
        rotation_error = max(float(Rotation.from_matrix(a[:3, :3]@b[:3, :3].T).magnitude()) for a, b in zip(actual, expected))
        reports.append(dict(kind='moving', sign=sign, independent_ODE_position_difference_m=position_error,
            independent_ODE_rotation_difference_rad=rotation_error,
            issued_commands_match_independent_schedule=True))
        endpoint = link.copy(); endpoint[:3, 3] += link[:3, :3] @ np.array([.6, .2, .2])
        endpoint[:3, :3] = link[:3, :3] @ Rotation.from_euler('z', sign*.3).as_matrix()
        stopped = terminal.TerminalContext(backend, stamp_s=stamp, task_end_s=stamp, stop_seconds=5.,
            endpoint_optical=endpoint, extrinsic=np.eye(4), body=body, link=link, measured_velocity=measured,
            previous_applied=previous, input_history=history)
        proposal = stopped.solve(); data = stopped._trajectory(proposal, stopped.desired)
        independent = independent_commands(proposal, stopped.desired, 1.6, len(data['issued']), True)
        np.testing.assert_allclose(data['issued'], independent, atol=1e-14, rtol=0)
        terminal_queries = np.r_[queries, 5.]
        expected = independent_continuous_model(body, link, measured,
            [(t-stamp, u) for t, u in history], independent, 5., terminal_queries)
        actual = [stopped._trajectory(proposal, stopped.desired)['links'][int(np.flatnonzero(np.isclose(stopped.events, t, atol=1e-10, rtol=0))[-1])]
                  for t in terminal_queries]
        position_error = max(float(np.linalg.norm(a[:3, 3]-b[:3, 3])) for a, b in zip(actual, expected))
        rotation_error = max(float(Rotation.from_matrix(a[:3, :3]@b[:3, :3].T).magnitude()) for a, b in zip(actual, expected))
        reports.append(dict(kind='terminal_yaw', sign=sign, independent_ODE_position_difference_m=position_error,
            independent_ODE_rotation_difference_rad=rotation_error,
            issued_commands_match_independent_schedule=True,
            endpoint_unchanged=bool(np.array_equal(endpoint, stopped.endpoint)),
            cap_slew_pass=bool(np.all(np.abs(independent) <= CAP+1e-12) and
                np.all(np.abs(np.diff(np.vstack([previous, independent]), axis=0)) <= SLEW*DT+1e-12)),
            literal_zero_from_2s=bool(np.all(independent[np.arange(len(independent))*DT >= 2.-1e-9] == 0.)),
            desired_yaw_has_correct_sign=bool(np.sign(stopped.desired[2]) == sign)))
    report = dict(schema='go2_v2_prediction_independent_ODE/v1', physical_execution=False,
        scope='Synthetic full-tilt snapshots, actual-history declaration, both yaw signs and vx sign crossing. Independent continuous ODE with explicit ZOH delayed input; compares candidate midpoint integration, not real learned dynamics.',
        position_numerical_comparison_tolerance_m=2e-5, rotation_numerical_comparison_tolerance_rad=1e-9,
        results=reports, passed=all(r['independent_ODE_position_difference_m'] < 2e-5 and
            r['independent_ODE_rotation_difference_rad'] < 1e-9 and r.get('cap_slew_pass', True) and
            r.get('literal_zero_from_2s', True) and r.get('endpoint_unchanged', True) and
            r.get('desired_yaw_has_correct_sign', True) for r in reports),
        evidence_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.iterdir() if p.is_file()})
    (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
