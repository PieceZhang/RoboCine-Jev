#!/usr/bin/env python3
"""Independent genuine-Popen task binding and public-feedback CPU black box.

No Kit or physical clock. Recorded q/base/camera seed the real Pink worker.
Unchanged recorded observations retain their original acquisition timestamps.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--recorded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package, out = args.package.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'sources'
    frozen.mkdir()
    for name in ('wbc_backend.py', 'solver_context.py', 'solver_feedback.py', 'adaptive_runner.py',
                 'gravity_model.py', 'pink_profile.py'):
        shutil.copy2(package/'Robot_G1'/name, frozen/name)
    shutil.copy2(package/'Robot_G1/kinematics/g1_29dof.urdf', frozen/'g1_29dof.urdf')
    shutil.copy2(Path(__file__), frozen/Path(__file__).name)
    sys.path.insert(0, str(package))
    from jev_agent_wbc.representation import Observation
    from jev_agent_wbc.wbc_protocol import Capabilities, CameraTrajectory, TimedTask
    from jev_agent_wbc.wbc_physical import LiveTaskSession
    backend = load('independent_context_backend', frozen/'wbc_backend.py')
    context = load('independent_context_helper', frozen/'solver_context.py')
    feedback = load('independent_solver_feedback', frozen/'solver_feedback.py')
    rows = [json.loads(line) for line in (args.recorded/'measurements.jsonl').open()]
    row = next(r for r in rows if r['case'] == 'left' and r['phase'] == 'move')
    names = row['joint_names']
    measured = dict(zip(names, row['joint_positions']))
    legs = [n for n in names if any(t in n for t in ('hip', 'knee', 'ankle'))]
    camera = np.array(json.loads((args.recorded/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    base, optical = np.array(row['T_world_base']), np.array(row['T_world_camera_optical'])
    session = LiveTaskSession(Capabilities('g1-transport-cpu-fixture', True, frozenset({'timed_camera_se3'})))
    observations = {round(r['stamp_s'], 8): r['measured_observation'] for r in rows}
    dt, stamp = .02, row['stamp_s']
    packets, checks, tasks = [], [], []

    def observed(at):
        return Observation.from_dict(deepcopy(observations[round(at, 8)]))

    def admit(task_id, *, template=None, changed=None):
        obs = observed(stamp)
        canonical_now = obs.stamp_s
        if template is None:
            task = TimedTask(task_id, canonical_now, obs.id,
                CameraTrajectory(canonical_now, (0., 2.), (obs.world_from_camera, obs.world_from_camera)), .05, .15)
        else:
            value = deepcopy(template.to_dict())
            if changed == 'id':
                value['task_id'] = task_id
            elif changed == 'tolerance':
                value['position_tolerance_m'] = .04
            elif changed == 'observation':
                value['observation_id'] = obs.id
                value['issued_at_s'] = canonical_now
            elif changed == 'issued':
                value['issued_at_s'] = canonical_now
            task = TimedTask.from_dict(value)
            # For unchanged original observation binding, retain its fresh
            # recorded fixture; full Task must pass real protocol admission.
            if task.observation_id != obs.id:
                obs = observed(template.issued_at_s)
        outcome = session.accept(task, obs, canonical_now)
        assert outcome['supported'], outcome
        tasks.append(task.to_dict())
        return task, context.task_context(task.to_dict())

    worker = backend.PinkProcess(legs, names, camera, frozen/'g1_29dof.urdf',
        source_path=frozen/'wbc_backend.py', stderr_path=out/'worker_stderr.log', max_age_s=.06)
    runtime = worker.runtime
    worker.warmup(measured)

    def submit(ctx):
        nonlocal stamp
        stamp = round(stamp+dt, 8)
        assert worker.submit(measured, base, optical, stamp, optical, dt, task_context=ctx)

    def poll(ctx, label):
        packet = worker.poll(stamp, current_context=ctx)
        packets.append(dict(label=label, cpu_stamp_s=stamp, packet=deepcopy(packet),
            failure_count=worker.failure_count, recovery_started=worker._recovery_started_wall_s,
            outstanding=int(worker.pending is not None)+int(worker.cancelled is not None)))
        assert packets[-1]['outstanding'] <= 2
        return packet

    def await_fresh(ctx, label):
        nonlocal stamp
        # Each synthetic 20ms clock tick consumes 20ms wall time. The actual
        # controller retains its original60ms lease/180ms recovery budget.
        for _ in range(3):
            time.sleep(dt)
            stamp = round(stamp+dt, 8)
            packet = poll(ctx, label)
            if packet['ready']:
                assert packet['status'] == 'fresh_command' and packet['ok']
                assert packet['task_context'] == ctx and packet['task_context_status'] == 'same_task'
                return packet
        raise AssertionError(f'no original-lease fresh result: {packets[-1]}')

    def public(packet, label, collision=None):
        nonlocal stamp
        stamp = round(stamp+dt, 8)
        support = feedback.public_support(packet, collision)
        current_observation = observed(stamp)
        result = session.feedback(current_observation, **support, now_s=current_observation.stamp_s)
        checks.append(dict(label=label, support=support, public_feedback=result))
        return support

    try:
        # Genuine pretask result, even if still fresh, cannot enter admitted A.
        submit(None)
        task, ctx = admit('context-A')
        dropped = poll(ctx, 'pretask_superseded')
        assert not dropped['ready'] and dropped['status'].startswith('superseded_task')
        assert worker.failure_count == 0 and worker._recovery_started_wall_s is None
        assert public(dropped, 'pretask_feedback')['reason'] == 'command_stale'
        submit(ctx)
        waiting = poll(ctx, 'new_task_immediate_wait')
        assert not waiting['ready'] or waiting['task_context'] == ctx
        if not waiting['ready']:
            assert public(waiting, 'missing_fresh_feedback')['reason'] != 'accepted'
            fresh = await_fresh(ctx, 'new_task_fresh')
        else:
            fresh = waiting
        assert public(fresh, 'fresh_feedback')['reason'] == 'accepted'
        checks.append(dict(label='pretask_to_A', passed=True, failure_count=worker.failure_count))

        for changed in ('id', 'tolerance', 'observation', 'issued'):
            # Use a new admitted A at the current time for each independent pair.
            original, old_ctx = admit('same-visible-id')
            submit(old_ctx)
            next_task, next_ctx = admit('next-id', template=original, changed=changed)
            # issued/observation change must differ, while original ID/path can
            # remain the same. Advance CPU clock before their admission.
            if next_ctx == old_ctx:
                stamp = round(stamp+dt, 8)
                next_task, next_ctx = admit('next-id', template=original, changed=changed)
            assert next_ctx != old_ctx
            count_before = worker.failure_count
            dropped = poll(next_ctx, f'{changed}_superseded')
            assert not dropped['ready'] and dropped['status'].startswith('superseded_task')
            assert worker.failure_count == count_before
            assert public(dropped, f'{changed}_feedback')['reason'] == 'command_stale'
            submit(next_ctx)
            fresh = await_fresh(next_ctx, f'{changed}_replacement_fresh')
            assert public(fresh, f'{changed}_fresh_feedback')['reason'] == 'accepted'
            checks.append(dict(label=f'full_task_change_{changed}', passed=True,
                old_context=old_ctx, new_context=next_ctx, failure_count=worker.failure_count))

        _, ctx = admit('snapshot-test')
        argument = deepcopy(ctx)
        submit(argument)
        argument['task_id'] = 'caller-mutated-after-submit'
        packet = await_fresh(ctx, 'snapshot_echo')
        assert packet['task_context'] == ctx
        checks.append(dict(label='submit_context_snapshot', passed=True))

        # A controller-poll stall is injected while the worker can finish. Old
        # buffered A data remains unusable after admitting B, including when
        # both its60ms lease and180ms recovery horizon have elapsed.
        _, old_ctx = admit('before-poll-stall')
        submit(old_ctx)
        stall_start = time.monotonic()
        time.sleep(.2)
        stall_wall = time.monotonic()-stall_start
        stamp = round(stamp+.2, 8)
        _, ctx = admit('after-poll-stall')
        dropped = poll(ctx, 'buffered_old_task_after_200ms_stall')
        assert not dropped['ready'] and dropped['status'].startswith('superseded_task')
        assert worker.failure_count == 0
        assert public(dropped, 'poststall_superseded_feedback')['reason'] == 'command_stale'
        submit(ctx)
        fresh = await_fresh(ctx, 'poststall_new_task_fresh')
        checks.append(dict(label='old_buffered_reply_after180ms_and_task_change', passed=True,
            injected_stall_wall_s=stall_wall, old_reply_applied=False,
            classification=dropped['status'], failure_count=worker.failure_count,
            note='Superseded classification precedes old TTL classification; only a new independently fresh task reply becomes ready.'))

        # Corrupt only the returned metadata after a real Popen worker solve.
        # This is an explicit transport fault injection, not an upstream error.
        for corruption in ('missing_echo', 'wrong_echo'):
            _, ctx = admit(corruption)
            submit(ctx)
            receive = worker._receive
            def injected(wait_s=0.):
                result = receive(wait_s)
                if result is not None:
                    result = deepcopy(result)
                    if corruption == 'missing_echo':
                        result.pop('task_context', None)
                    else:
                        result['task_context']['task_id'] += '-corrupted'
                return result
            worker._receive = injected
            time.sleep(dt)
            stamp = round(stamp+dt, 8)
            bad = poll(ctx, corruption)
            worker._receive = receive
            assert not bad['ready'] and bad['failure_kind'] == 'transport_failure'
            assert worker.failure_count == 1
            assert public(bad, corruption+'_feedback') == dict(reason='solver_failure', feasible=False)
            submit(ctx)
            fresh = await_fresh(ctx, corruption+'_recovery')
            assert worker.failure_count == 0
            checks.append(dict(label=corruption, passed=True, fault_injection='metadata after actual Popen reply'))

        for packet in (None, dict(ready=False,status='waiting_hold_upper_targets'),
                       dict(ready=False,status='stale_or_failed_result_dropped'),
                       dict(ready=True,ok=False,status='fresh_command'),
                       dict(ready=True,ok=True), dict(ready=False,ok=True,status='fresh_command')):
            support = public(packet, 'public_missing_or_invalid_fresh_proof')
            assert support['reason'] != 'accepted'
        assert public(fresh, 'collision_failure_precedes_fresh', {'status':'fail'}) == dict(reason='self_collision', feasible=False)
        assert public(dict(ready=False,status='worker_deadline_exceeded_hold_upper_targets'),
            'deadline_feedback') == dict(reason='deadline_exceeded', feasible=False)

        tree = ast.parse((frozen/'adaptive_runner.py').read_text())
        calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'public_support'
                 and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == 'packet']
        assert calls, 'real driver must call audited mapper with actual packet'
        checks.append(dict(label='actual_driver_calls_public_support_with_packet', passed=True, lines=calls))
    finally:
        worker.close()
        (out/'packets.json').write_text(json.dumps(packets, indent=2, allow_nan=False)+'\n')
        (out/'checks.json').write_text(json.dumps(checks, indent=2, allow_nan=False)+'\n')
        (out/'tasks.json').write_text(json.dumps(tasks, indent=2, allow_nan=False)+'\n')
    report = dict(schema='independent_g1_task_transport_popen/v1', passed=True,
        scope='Real isolated Pink Popen request/reply and admitted protocol CPU fixtures using unchanged historical observations/timestamps. Fixed seed joint state is transport-only. Metadata fault injections clearly separated. No physical execution, stop, contact or environment certification.',
        pace=1, synthetic_control_dt_s=dt, max_age_s=.06, recovery_deadline_s=.18,
        runtime=runtime, recorded_seed_sha256=sha(args.recorded/'measurements.jsonl'),
        checks=len(checks), packets=len(packets), physical_verified=False,
        executed_frozen_sha256={str(p.relative_to(out)):sha(p) for p in frozen.iterdir() if p.is_file()},
        source_unchanged_during_audit=all(sha(package/'Robot_G1'/p.name)==sha(p) for p in frozen.iterdir()
            if p.name in ('wbc_backend.py','solver_context.py','solver_feedback.py','adaptive_runner.py')))
    (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
