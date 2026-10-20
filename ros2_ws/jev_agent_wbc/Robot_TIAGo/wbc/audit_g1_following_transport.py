#!/usr/bin/env python3
"""True isolated Pink transport and canonical-order audit; no physics execution.

Recorded native tensors/35-shape geometry seed the worker. Write-ledger events
below are explicit synthetic CPU events, never claimed to be actual PhysX writes.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); package = args.package.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False); source = out/'sources'; source.mkdir()
    for name in ['wbc_backend.py', 'following_reserve.py', 'following_execution.py', 'solver_context.py',
                 'solver_feedback.py', 'pink_collision.py', 'gravity_model.py', 'adaptive_runner.py']:
        shutil.copy2(package/'Robot_G1'/name, source/name)
    shutil.copy2(package/'Robot_G1/kinematics/g1_29dof.urdf', source/'g1_29dof.urdf')
    shutil.copy2(__file__, source/Path(__file__).name)
    sys.path.insert(0, str(package))
    from jev_agent_wbc.representation import Pose
    from jev_agent_wbc.wbc_protocol import CameraTrajectory, TimedTask, OPTICAL_FROM_SEMANTIC
    backend = load('actual_following_transport_backend', source/'wbc_backend.py')
    context = load('actual_following_transport_context', source/'solver_context.py')
    feedback = load('actual_following_transport_feedback', source/'solver_feedback.py')
    following = load('actual_following_transport_reserve', source/'following_reserve.py')
    execution = load('actual_following_transport_execution', source/'following_execution.py')
    native_path = package/'Robot_G1/results/default_upper_yaw_01/startup_physx_mass_properties.json'
    native = json.loads(native_path.read_text()); names = native['joint_names']
    q = dict(zip(names, native['joint_positions']['values'][0]))
    actor = np.asarray(native['actor_world_poses_xyzw']['values'])[0]
    transforms = np.broadcast_to(np.eye(4), (len(actor), 4, 4)).copy()
    transforms[:, :3, :3] = Rotation.from_quat(actor[:, 3:]).as_matrix(); transforms[:, :3, 3] = actor[:, :3]
    bodies = dict(zip(native['body_names'], transforms)); pelvis = bodies['pelvis']
    history = package/'Robot_G1/results/adaptive_actual_gravity_phase_lease_01'
    geometry = json.loads((history/'fine_collision_geometry.json').read_text())
    extrinsic = json.loads((history/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    legs = [name for name in names if any(part in name for part in ('hip_', 'knee_', 'ankle_'))]
    actual_gravity = json.loads((package/'Robot_G1/config/actual_gravity_model.json').read_text())
    options = dict(posture_home=q, posture_cost=.03, waist_posture_cost=.15,
        damping_cost=.5, collision_gain=1., actual_gravity_metadata=actual_gravity)
    worker = backend.PinkProcess(legs, names, extrinsic, source/'g1_29dof.urdf',
        source_path=source/'wbc_backend.py', stderr_path=out/'worker_stderr.log', max_age_s=.06,
        solver_options=options)
    worker.warmup(q)
    worker.configure_collisions(geometry, q, pelvis, bodies)
    worker._send(dict(op='step', joints=q, T_world_pelvis=pelvis.tolist(), target=None,
        stamp_s=0., dt=.02, reset_command=True))
    initial = worker._receive(10.); worker.pending = None; worker.needs_reset = True
    camera = np.asarray(initial['predicted_camera']); target = camera.copy(); target[0, 3] += .005
    pose = Pose(tuple(target[:3, 3]), tuple(Rotation.from_matrix(target[:3, :3] @ OPTICAL_FROM_SEMANTIC.rotation).as_quat()))
    task = TimedTask('following-A', 1., 'cpu-recorded-fixture', CameraTrajectory(1., (0., 5.), (pose, pose)), .05, .15)
    ctx = context.task_context(task.to_dict())
    canonical = list(worker.optimized_joint_names); owner = list(worker.owners['pink'])
    physical_ids = [names.index(n) for n in canonical]
    permutation = [owner.index(n) for n in canonical]
    checks = []; packets = []; events = []; stamp = 1.; commit_seq = 0; commit_stamp = .995
    current_target = np.asarray([q[n] for n in canonical]).astype(np.float32).astype(float)
    before = current_target.copy()
    def snapshot(at, positions=None, task_context=None, sequence=None):
        selected = ctx if task_context is None else task_context
        return following.committed_reference_snapshot(task_id=selected['task_id'], task_sha256=selected['task_sha256'],
            joint_names=canonical, positions_rad=current_target if positions is None else positions,
            commit_seq=commit_seq if sequence is None else sequence, commit_stamp_s=commit_stamp, snapshot_stamp_s=at)
    def evidence(at):
        old = snapshot(at)
        return dict(schema='g1_following_evidence/v1', task_id=ctx['task_id'], task_sha256=ctx['task_sha256'],
            acquisition_stamp_s=at, joint_names=canonical, measured_q_rad=[q[n] for n in canonical],
            measured_qd_rad_s=[0.]*15, velocity_stamp_s=at, measurement_source='measured_physx',
            committed_reference=old, planned_advance_bound=following.planned_advance_bound(old,
                current_target, task_id=ctx['task_id'], request_stamp_s=at, write_dtype='float32'))
    def transact(label, transform=None, planned_during_solve=None, corrupt_reply=None):
        nonlocal stamp, current_target, commit_seq, commit_stamp
        stamp = round(stamp+.04, 8); record = evidence(stamp)
        if transform is not None: transform(record, stamp)
        if planned_during_solve is not None:
            record['planned_advance_bound'] = following.planned_advance_bound(record['committed_reference'],
                planned_during_solve, task_id=ctx['task_id'], request_stamp_s=stamp, write_dtype='float32')
        receiver = worker._receive
        def fault_injected_receive(timeout):
            result = receiver(timeout)
            if result is not None and result.get('ok') and result.get('following_reserve_report'):
                corrupt_reply(result)
            return result
        if corrupt_reply is not None: worker._receive = fault_injected_receive
        assert worker.submit(q, pelvis, target, stamp, camera, .02, task_context=ctx, following_evidence=record)
        request_stamp = stamp
        packet = None
        for _ in range(3):
            if planned_during_solve is not None:
                for substep in range(4):
                    at = stamp+substep*.005
                    current_target = np.asarray(planned_during_solve).copy()
                    commit_seq += 1; commit_stamp = at
                    events.append(dict(label=label, substep=substep, stamp_s=at,
                        synthetic_CPU_write=True, known_planned_target_committed=True))
                    time.sleep(.005)
                planned_during_solve = None
            else: time.sleep(.02)
            stamp = round(stamp+.02, 8)
            packet = worker.poll(stamp, current_context=ctx, execution_horizon_s=.02)
            if packet.get('ready') or packet.get('failure_kind') or packet.get('qp_failure') or packet.get('status') == 'solver_failure': break
            if worker.pending is None: break
        worker._receive = receiver
        packets.append(dict(label=label, request_stamp_s=request_stamp, poll_stamp_s=stamp,
            request_evidence=record, packet=deepcopy(packet), public_support=feedback.public_support(packet)))
        return packet, record
    try:
        owner_values = np.arange(15.)+.125
        physics = np.zeros(29)
        for n, value in zip(owner, owner_values): physics[names.index(n)] = value
        np.testing.assert_array_equal(physics[physical_ids], owner_values[permutation])
        assert canonical != owner and len(set(canonical)) == 15
        checks.append(dict(label='real_native_canonical15_vs_interleaved_PhysX29', passed=True,
            canonical=canonical, owner_order=owner, physical_ids=physical_ids, owner_permutation=permutation))
        packet, record = transact('normal_real_Popen')
        assert packet['ready'], packet
        report = packet['following_reserve_report']
        gate = following.validate_application_manifest(report, task_id=ctx['task_id'], task_sha256=ctx['task_sha256'],
            current_reference=snapshot(stamp), now_s=stamp, execution_horizon_s=.02)
        velocity = np.asarray([packet['velocity'][n] for n in canonical])
        proposal = before+velocity*.02; new = proposal.astype(np.float32).astype(float)
        authorization = execution.authorize_native_step(task_context=ctx, packet_seq=packet['seq'],
            acquisition_stamp_s=packet['stamp_s'], application_stamp_s=stamp, committed_reference=snapshot(stamp),
            integrator_before=before, velocity=velocity, proposed_reference=proposal,
            actual_float32_target=new, antiwindup_active=False)
        for substep in range(4):
            at = stamp+substep*.005
            checked = execution.validate_authorized_write(authorization, task_context=ctx,
                current_reference=snapshot(at), target_to_write=new, now_s=at, horizon_s=.005)
            # Synthetic CPU write ledger event, not PhysX. Commit after gate.
            current_target = new.copy(); commit_seq += 1; commit_stamp = at
            events.append(dict(substep=substep, stamp_s=at, write_check=checked,
                synthetic_CPU_write=True, committed_after_check=snapshot(at+.005)))
        before = proposal.copy()
        stamp = round(stamp+.02, 8)
        checks.append(dict(label='before_first_use_then_four_atomic_synthetic_writes', passed=True, gate=gate))
        for at, horizon, expired in [(packet['stamp_s']+.055, .005, False),
                                      (packet['stamp_s']+.055001, .005, True)]:
            try:
                execution.validate_authorized_write(authorization, task_context=ctx,
                    current_reference=snapshot(at), target_to_write=current_target, now_s=at, horizon_s=horizon)
                assert not expired
            except execution.FollowingExecutionRejected:
                assert expired
        held = execution.validate_expired_reference_hold(current_reference=snapshot(stamp+.1),
            target_to_write=current_target, velocity_to_write=[0.]*15, gravity_to_write=[0.]*15)
        assert held['fresh_native_motion_authorized'] is False
        try:
            execution.validate_authorized_write(authorization, task_context=ctx,
                current_reference=snapshot(stamp), target_to_write=authorization['payload']['start_reference']['positions_rad'],
                now_s=stamp, horizon_s=.005)
            raise AssertionError('reversion after first actual new target was accepted')
        except execution.FollowingExecutionRejected: pass
        checks.append(dict(label='four_substep60ms_boundary_and_expiry_hold_without_reference_reversion', passed=True))
        planned = current_target.copy(); planned[canonical.index('right_elbow_joint')] += .0002
        planned = planned.astype(np.float32).astype(float)
        advanced, _ = transact('normal_old_committed_then_known_plan_written_during_solve', planned_during_solve=planned)
        assert advanced['ready'], advanced
        following.validate_application_manifest(advanced['following_reserve_report'], task_id=ctx['task_id'],
            task_sha256=ctx['task_sha256'], current_reference=snapshot(stamp), now_s=stamp, execution_horizon_s=.02)
        out_of_box = current_target.copy(); out_of_box[0] += .001
        try:
            following.validate_application_manifest(advanced['following_reserve_report'], task_id=ctx['task_id'],
                task_sha256=ctx['task_sha256'], current_reference=snapshot(stamp, positions=out_of_box, sequence=commit_seq+1),
                now_s=stamp, execution_horizon_s=.02)
            raise AssertionError('additional unplanned reference advance accepted')
        except following.FollowingReserveRejected: pass
        checks.append(dict(label='true_Popen_old_committed_separate_planned_advance_and_extra_advance_rejection', passed=True))
        for label, transform in [
            ('wrong_canonical_order', lambda e, t: e.update(joint_names=owner)),
            ('wrong_velocity_stamp', lambda e, t: e.update(velocity_stamp_s=t-.02)),
            ('wrong_Task_SHA', lambda e, t: e.update(task_sha256='b'*64)),
        ]:
            bad, _ = transact(label, transform)
            checks.append(dict(label=label, passed=not bad['ready'], ready=bad['ready']))
            assert not bad['ready'], bad
            good, _ = transact(label+'_fresh_recovery')
            assert good['ready'], good
        def older_evidence(e, at):
            old = at-.02
            e['acquisition_stamp_s'] = old; e['velocity_stamp_s'] = old
            e['committed_reference'] = snapshot(old)
            e['planned_advance_bound'] = following.planned_advance_bound(e['committed_reference'], current_target,
                task_id=ctx['task_id'], request_stamp_s=old, write_dtype='float32')
        bad, _ = transact('outer_request_stamp_vs_evidence_minus20ms', older_evidence)
        before_use_accepted = False
        if bad['ready']:
            following.validate_application_manifest(bad['following_reserve_report'], task_id=ctx['task_id'],
                task_sha256=ctx['task_sha256'], current_reference=snapshot(stamp), now_s=stamp, execution_horizon_s=.02)
            before_use_accepted = True
        checks.append(dict(label='outer_request_stamp_equals_following_acquisition', passed=not before_use_accepted,
            worker_ready=bad['ready'], before_first_use_accepted=before_use_accepted,
            note='Fault-injected stale evidence retains matching measured positions in a static fixture; no physical motion.'))
        good, _ = transact('post_stamp_probe_fresh')
        assert good['ready'], good
        for key in ('seq', 'stamp_s', 'task_sha256', 'evidence_sha256'):
            def corrupt(result):
                binding = result['following_reserve_report']['worker_request_binding']
                binding[key] = binding[key]+1 if key in ('seq', 'stamp_s') else 'b'*64
            bad, _ = transact('real_reply_transport_corruption_'+key, corrupt_reply=corrupt)
            assert not bad['ready'] and bad['failure_kind'] == 'transport_failure', bad
            good, _ = transact('corruption_'+key+'_fresh_recovery')
            assert good['ready'], good
            checks.append(dict(label='real_reply_'+key+'_binding_and_fresh_recovery', passed=True))
        def wrong_binding_type(result): result['following_reserve_report']['worker_request_binding'] = []
        bad, _ = transact('real_reply_binding_wrong_type', corrupt_reply=wrong_binding_type)
        assert not bad['ready'] and bad['failure_kind'] == 'transport_failure', bad
        good, _ = transact('binding_wrong_type_fresh_recovery')
        assert good['ready'], good
        checks.append(dict(label='real_reply_binding_type_failure_and_fresh_recovery', passed=True))
        def closing(e, _):
            e['measured_qd_rad_s'][canonical.index('right_shoulder_roll_joint')] = .5
        failed, _ = transact('synthetic_closing_infeasible_QP', closing)
        failure = failed.get('qp_failure')
        assert not failed['ready'] and failure is not None, failed
        groups = failure['inequality_groups']; rows = len(failure['qp']['h'])
        count = sum(g['row_count'] for g in groups)
        reserve_group = next(g for g in groups if g['name'] == 'native_following_reserve')
        cbf_group = next(g for g in groups if g['name'] == 'native_pink_self_collision')
        assert rows == count and reserve_group['row_count'] == cbf_group['row_count'] == 20
        assert reserve_group['row_start']+20 == cbf_group['row_start']
        assert failure['qp_capture'] == 'native_pink_exception_frame'
        checks.append(dict(label='exact_QP_failure_retains_original_groups_and_20new_rows', passed=True,
            rows=rows, groups=groups, qp_capture=failure['qp_capture']))
        summary = dict(schema='g1_following_real_Popen_independent/v1', physical_execution=False,
            synthetic_write_ledger=True, runtime=worker.runtime, checks=checks, packets=packets,
            synthetic_execution_events=events, passed=all(c['passed'] for c in checks),
            scope='Real separate Pink process, real native tensor joint order and original35 collision shapes. States/drive-write ledger and fault injections are CPU fixtures, not physical closed-loop verification.',
            evidence_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()})
        (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(passed=summary['passed'], checks=[{k:v for k,v in c.items() if k in ('label','passed','worker_ready','before_first_use_accepted')} for c in checks])))
    finally: worker.close()


if __name__ == '__main__': main()
