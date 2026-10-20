"""Conditional native Limit algebra, provenance boundaries and genuine Pink QP."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = load('g1_following_test', PACKAGE/'Robot_G1/following_reserve.py')
TASK_SHA = 'a'*64


@pytest.fixture(scope='module')
def robot():
    folder = PACKAGE/'Robot_G1/results/adaptive_actual_gravity_phase_lease_01'
    backend = load('following_test_native_backend', folder/'sources/g1_backend.py')
    collision = load('following_test_native_collision', folder/'sources/pink_collision.py')
    first = json.loads(next((folder/'measurements.jsonl').open()))
    names = first['joint_names']
    legs = [name for name in names if any(key in name for key in ('hip', 'knee', 'ankle'))]
    extrinsic = json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    home = dict(zip(names, first['joint_positions']))
    solver = backend.PinkUpperBody(legs, names, extrinsic, folder/'sources/g1_29dof.urdf',
        posture_home=home, posture_cost=.03, waist_posture_cost=.15, damping_cost=.5,
        collision_gain=1.)
    solver.collision_helper = collision
    solver.collision_offsets = json.loads((folder/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    solver.collision_geometry = collision.geometry_model(solver.model,
        json.loads((folder/'fine_collision_geometry.json').read_text()), solver.collision_offsets)
    configuration, upper = solver.configuration(home)
    from pink.barriers import SelfCollisionBarrier
    barrier = SelfCollisionBarrier(20, gain=1., safe_displacement_gain=0., d_min=.0065)
    return solver, configuration, upper, barrier, first


def evidence(robot):
    _, cfg, names, _, _ = robot
    reference = F.committed_reference_snapshot(task_id='A', task_sha256=TASK_SHA, joint_names=names,
        positions_rad=cfg.q.tolist(), commit_seq=10, commit_stamp_s=1.98, snapshot_stamp_s=2.)
    return dict(schema='g1_following_evidence/v1', task_id='A', task_sha256=TASK_SHA, acquisition_stamp_s=2.,
        joint_names=list(names), measured_q_rad=cfg.q.tolist(), measured_qd_rad_s=[0.]*15,
        velocity_stamp_s=2., measurement_source='measured_physx', committed_reference=reference)


def limit(robot, record=None, **kwargs):
    _, cfg, _, barrier, _ = robot
    record = evidence(robot) if record is None else record
    return F.FollowingReserveLimit.from_native_barrier(cfg, barrier, record,
        task_id=kwargs.pop('task_id', 'A'), task_sha256=kwargs.pop('task_sha256', TASK_SHA),
        reference_token=kwargs.pop('reference_token', record['committed_reference']['token']),
        now_s=kwargs.pop('now_s', 2.), **kwargs)


def test_static_floor_is_not_globally_inflated():
    J = np.zeros((1, 15)); J[0, 0] = .2
    G, h, metrics = F._reserve_rows([.0076], J, [0.]*15, [0.]*15)
    np.testing.assert_array_equal(G, -J)
    np.testing.assert_allclose(h, [.0011], rtol=0, atol=1e-17)
    assert metrics['reserve_m'][0] == 0.


def test_closing_reserve_has_correct_units_and_values():
    J = np.zeros((1, 15)); J[0, 0] = .2
    qd = np.zeros(15); qd[0] = -.065
    _, h, metrics = F._reserve_rows([.010], J, qd, [0.]*15)
    assert metrics['actual_closing_m_s'][0] == pytest.approx(.013)
    assert metrics['assumed_braking_m_s2'][0] == pytest.approx(.1)
    assert metrics['reserve_m'][0] == pytest.approx(.12*.013 + .013**2/.2)
    assert h[0] == pytest.approx(.010-.0065-.12*.013-.013**2/.2)


def test_opening_reference_never_subsidizes_reserve():
    J = np.zeros((1, 15)); J[0, 0] = .2
    opening = np.zeros(15); opening[0] = .02
    _, h0, _ = F._reserve_rows([.010], J, [0.]*15, [0.]*15)
    _, hp, _ = F._reserve_rows([.010], J, [0.]*15, opening)
    _, hm, _ = F._reserve_rows([.010], J, [0.]*15, -opening)
    np.testing.assert_array_equal(h0, hp)
    np.testing.assert_allclose(h0-hm, [.004], rtol=0, atol=1e-17)


def test_planned_per_axis_box_has_no_cancelling_opening_credit():
    J = np.zeros((1, 15)); J[0, :2] = 1.
    plan = np.zeros(15); plan[:2] = [-.001, .001]
    _, _, metrics = F._reserve_rows([.010], J, [0.]*15, [0.]*15, planned_delta=plan)
    assert metrics['planned_closing_advance_m'][0] == pytest.approx(.001)
    assert J @ plan == pytest.approx([0.])


def test_zero_jacobian_static_is_finite_but_measured_closing_is_rejected():
    _, h, metrics = F._reserve_rows([.0076], np.zeros((1, 15)), [0.]*15, [0.]*15)
    assert np.isfinite(h).all() and metrics['assumed_braking_m_s2'][0] == 0.
    with pytest.raises(F.FollowingReserveRejected, match='zero_projected_braking'):
        F._reserve_rows([.0076], np.zeros((1, 15)), [0.]*15, [0.]*15, closing_speed_floor=[.001])


@pytest.mark.parametrize('bad', [True, np.bool_(True), '1', float('nan'), float('inf')])
def test_nonphysical_scalar_values_rejected(bad):
    with pytest.raises(F.FollowingReserveRejected):
        F._number(bad, 'stamp')


@pytest.mark.parametrize('bad', [[True]+[0.]*14, [np.bool_(False)]+[0.]*14,
    ['0']+[0.]*14, [float('nan')]+[0.]*14, [0.]*14, [[0.]*15]])
def test_bool_string_nonfinite_wrong_shape_vectors_rejected(bad):
    with pytest.raises(F.FollowingReserveRejected):
        F._vector(bad, 'measured')


def test_native_all20_and_caller_mutation_isolated(robot):
    data = evidence(robot)
    guarded = limit(robot, data)
    G, h = guarded.compute_qp_inequalities(robot[1].q, .02)
    assert G.shape == (20, 15) and h.shape == (20,)
    data['committed_reference']['positions_rad'][0] += .1
    data['measured_qd_rad_s'][0] = 2.
    g2, h2 = guarded.compute_qp_inequalities(robot[1].q, .02)
    np.testing.assert_array_equal(G, g2); np.testing.assert_array_equal(h, h2)
    G[:] = 1.; h[:] = 1.
    g3, h3 = guarded.compute_qp_inequalities(robot[1].q, .02)
    np.testing.assert_array_equal(g2, g3); np.testing.assert_array_equal(h2, h3)
    assert guarded.manifest()['actual_braking_certified'] is False


@pytest.mark.parametrize('changed', ['task', 'velocity_stamp', 'measurement_source', 'reference_snapshot',
    'reference_source', 'reference_task', 'reference_token', 'configuration', 'joint_order', 'names_string', 'startup'])
def test_request_cross_context_or_missing_measured_evidence_rejected(robot, changed):
    data = evidence(robot)
    if changed == 'task': data['task_id'] = 'B'
    elif changed == 'velocity_stamp': data['velocity_stamp_s'] += .001
    elif changed == 'measurement_source': data['measurement_source'] = 'issued_command_estimate'
    elif changed == 'reference_snapshot': data['committed_reference']['snapshot_stamp_s'] -= .02
    elif changed == 'reference_source': data['committed_reference']['source'] = 'planned_not_committed'
    elif changed == 'reference_task':
        ref = data['committed_reference']
        data['committed_reference'] = F.committed_reference_snapshot(task_id='B', task_sha256=TASK_SHA,
            **{k: ref[k] for k in ('joint_names', 'positions_rad', 'commit_seq', 'commit_stamp_s', 'snapshot_stamp_s')})
    elif changed == 'reference_token': data['committed_reference']['token'] = 'other'
    elif changed == 'configuration': data['measured_q_rad'][0] += 1e-8
    elif changed == 'joint_order': data['joint_names'].reverse()
    elif changed == 'names_string': data['joint_names'] = 'invalid'
    else:
        ref = data['committed_reference']
        data['committed_reference'] = F.committed_reference_snapshot(task_id=None,
            **{k: ref[k] for k in ('joint_names', 'positions_rad', 'commit_seq', 'commit_stamp_s', 'snapshot_stamp_s')})
    with pytest.raises(F.FollowingReserveRejected):
        limit(robot, data)


@pytest.mark.parametrize('now', [1.999, 2.061, True])
def test_future_or_old_measurement_rejected(robot, now):
    with pytest.raises(F.FollowingReserveRejected):
        limit(robot, now_s=now)


def test_native_recompute_changed_q_or_dt_rejected(robot):
    guarded = limit(robot)
    q = robot[1].q.copy(); q[0] += 1e-10
    with pytest.raises(F.FollowingReserveRejected): guarded.compute_qp_inequalities(q, .02)
    with pytest.raises(F.FollowingReserveRejected): guarded.compute_qp_inequalities(robot[1].q, .01)


def application_reference(data, positions, *, task_id='A', stamp=2.02, seq=14):
    return F.committed_reference_snapshot(task_id=task_id, task_sha256=TASK_SHA,
        joint_names=data['joint_names'], positions_rad=positions, commit_seq=seq,
        commit_stamp_s=stamp-.005, snapshot_stamp_s=stamp)


def test_same_value_repeat_physical_write_token_and_execution_horizon(robot):
    data = evidence(robot); guarded = limit(robot, data)
    current = application_reference(data, data['committed_reference']['positions_rad'], stamp=2.04)
    assert current['token'] == data['committed_reference']['token']
    assert guarded.validate_application(task_id='A', task_sha256=TASK_SHA, current_reference=current, now_s=2.04)['actual_braking_certified'] is False
    with pytest.raises(F.FollowingReserveRejected, match='expired'):
        guarded.validate_application(task_id='A', task_sha256=TASK_SHA, current_reference=current, now_s=2.04001)


def test_planned_future_write_separate_and_bound_checked_at_application(robot):
    data = evidence(robot); old = np.array(data['committed_reference']['positions_rad'])
    planned = old.copy(); planned[:2] += [.001, -.002]
    data['planned_advance_bound'] = F.planned_advance_bound(data['committed_reference'], planned,
        task_id='A', request_stamp_s=2.)
    guarded = limit(robot, data)
    assert guarded.manifest()['planned_advance_included']
    for position in [old, planned, .5*(old+planned)]:
        current = application_reference(data, position)
        assert guarded.validate_application(task_id='A', task_sha256=TASK_SHA, current_reference=current, now_s=2.02)['reference_rows_rebased'] is False
    beyond = planned.copy(); beyond[0] += .00001
    with pytest.raises(F.FollowingReserveRejected, match='outside_declared'):
        guarded.validate_application(task_id='A', task_sha256=TASK_SHA, current_reference=application_reference(data, beyond), now_s=2.02)
    with pytest.raises(F.FollowingReserveRejected, match='cross_task'):
        guarded.validate_application(task_id='B', task_sha256=TASK_SHA, current_reference=application_reference(data, planned, task_id='B'), now_s=2.02)


def test_no_plan_rejects_any_new_reference_value(robot):
    data = evidence(robot); guarded = limit(robot, data)
    new = np.array(data['committed_reference']['positions_rad']); new[0] += .00001
    with pytest.raises(F.FollowingReserveRejected, match='unplanned_reference_token'):
        guarded.validate_application(task_id='A', task_sha256=TASK_SHA, current_reference=application_reference(data, new), now_s=2.02)


def test_expected_physical_ledger_token_is_independent_input(robot):
    with pytest.raises(F.FollowingReserveRejected, match='current_committed_reference_token'):
        limit(robot, reference_token='unknown_actual_reference')


def test_changed_reference_needs_an_actual_new_commit_event(robot):
    data = evidence(robot); old = np.array(data['committed_reference']['positions_rad'])
    new = old.copy(); new[0] += .001
    data['planned_advance_bound'] = F.planned_advance_bound(data['committed_reference'], new,
        task_id='A', request_stamp_s=2.)
    guarded = limit(robot, data)
    current = F.committed_reference_snapshot(task_id='A', task_sha256=TASK_SHA,
        joint_names=data['joint_names'], positions_rad=new, commit_seq=10,
        commit_stamp_s=1.98, snapshot_stamp_s=2.02)
    with pytest.raises(F.FollowingReserveRejected, match='commit_event_did_not_advance'):
        guarded.validate_application(task_id='A', task_sha256=TASK_SHA, current_reference=current, now_s=2.02)


def test_native_h_j_pairing_mismatch_fails_closed(robot, monkeypatch):
    original = np.argpartition; calls = [0]
    def mismatched(*args, **kwargs):
        value = original(*args, **kwargs).copy(); calls[0] += 1
        if calls[0] == 2: value[-2:] = value[-2:][::-1]
        return value
    monkeypatch.setattr(F.np, 'argpartition', mismatched)
    with pytest.raises(F.FollowingReserveRejected, match='row_pairing_mismatch'):
        limit(robot)


def test_same_id_different_full_Task_sha_rejected(robot):
    data = evidence(robot)
    with pytest.raises(F.FollowingReserveRejected, match='cross_full_task'):
        limit(robot, data, task_sha256='b'*64)
    guarded = limit(robot, data)
    current = application_reference(data, data['committed_reference']['positions_rad'])
    with pytest.raises(F.FollowingReserveRejected, match='cross_task'):
        F.validate_application_manifest(guarded.manifest(), task_id='A', task_sha256='b'*64,
            current_reference=current, now_s=2.02)


def test_application_manifest_is_independent_of_Pink_Pin_process(robot, tmp_path):
    data = evidence(robot); guarded = limit(robot, data)
    path = tmp_path/'packet.json'
    path.write_text(json.dumps(dict(report=guarded.manifest(),
        current=application_reference(data, data['committed_reference']['positions_rad']))))
    code = """import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('isolated_following',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=json.load(open(sys.argv[2]));m.validate_application_manifest(x['report'],task_id='A',task_sha256='a'*64,current_reference=x['current'],now_s=2.02)
assert 'pink' not in sys.modules and 'pinocchio' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-I', '-c', code,
        str(PACKAGE/'Robot_G1/following_reserve.py'), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_manifest_copy_and_corruption_boundary(robot):
    data = evidence(robot); guarded = limit(robot, data)
    report = guarded.manifest(); report['application_binding']['committed_reference']['positions_rad'][0] += .01
    with pytest.raises(F.FollowingReserveRejected, match='binding_corrupted'):
        F.validate_application_manifest(report, task_id='A', task_sha256=TASK_SHA,
            current_reference=application_reference(data, data['committed_reference']['positions_rad']), now_s=2.02)
    # The caller mutation did not alter the native Limit's frozen reference.
    assert guarded.manifest()['application_binding']['committed_reference']['positions_rad'] == data['committed_reference']['positions_rad']


def test_explicit_float32_write_roundoff_is_bounded_and_charged():
    old = np.zeros(15); old[0] = float(np.float32(.2))
    planned = (old + np.r_[.01, np.zeros(14)]).astype(np.float32).astype(float)
    ref = F.committed_reference_snapshot(task_id='A', task_sha256=TASK_SHA,
        joint_names=[f'q{i}' for i in range(15)], positions_rad=old,
        commit_seq=1, commit_stamp_s=1.98, snapshot_stamp_s=2.)
    assert planned[0]-old[0] > .01
    with pytest.raises(F.FollowingReserveRejected, match='issued_speed_cap'):
        F.planned_advance_bound(ref, planned, task_id='A', request_stamp_s=2.)
    bound = F.planned_advance_bound(ref, planned, task_id='A', request_stamp_s=2., write_dtype='float32')
    allowance = np.asarray(bound['native_float32_roundoff_rad'])
    assert 0 < allowance[0] < 1e-6
    assert planned[0]-old[0] <= .01+allowance[0]
    J = np.zeros((1, 15)); J[0, 0] = -.2
    _, _, metrics = F._reserve_rows([.010], J, [0.]*15, [0.]*15,
        planned_delta=bound['delta_rad'], native_roundoff_rad=allowance)
    assert metrics['native_float32_roundoff_m'][0] == pytest.approx(.2*allowance[0])
    assert metrics['planned_closing_advance_m'][0] == pytest.approx(.2*(planned[0]-old[0]))
    too_far = planned.copy(); too_far[0] = float(np.nextafter(np.float32(planned[0]), np.float32(1.)))
    with pytest.raises(F.FollowingReserveRejected, match='issued_speed_cap'):
        F.planned_advance_bound(ref, too_far, task_id='A', request_stamp_s=2., write_dtype='float32')


def test_float32_planned_box_survives_native_manifest_transport(robot):
    data = evidence(robot); old = np.asarray(data['committed_reference']['positions_rad']).copy()
    # Synthetic committed target ledger with an exact known float32 boundary;
    # no claim this artificial target was physically applied to the robot.
    old[0] = float(np.float32(.2))
    data['committed_reference'] = F.committed_reference_snapshot(task_id='A', task_sha256=TASK_SHA,
        joint_names=data['joint_names'], positions_rad=old, commit_seq=10,
        commit_stamp_s=1.98, snapshot_stamp_s=2.)
    new = old.astype(np.float32); new[0] = np.float32(old[0]+.01); new = new.astype(float)
    data['planned_advance_bound'] = F.planned_advance_bound(data['committed_reference'], new,
        task_id='A', request_stamp_s=2., write_dtype='float32')
    guarded = limit(robot, data)
    report = json.loads(json.dumps(guarded.manifest()))
    assert report['application_binding']['planned_write_dtype'] == 'float32'
    current = application_reference(data, new)
    assert F.validate_application_manifest(report, task_id='A', task_sha256=TASK_SHA,
        current_reference=current, now_s=2.02)['actual_braking_certified'] is False


@pytest.mark.parametrize('change', ['task', 'stamp', 'token', 'source', 'overspeed'])
def test_planned_advance_cannot_borrow_context_or_enlarge_original_speed(robot, change):
    data = evidence(robot); planned = np.array(data['committed_reference']['positions_rad'])
    planned[0] += .001
    plan = F.planned_advance_bound(data['committed_reference'], planned, task_id='A', request_stamp_s=2.)
    if change == 'task': plan['task_id'] = 'B'
    elif change == 'stamp': plan['request_stamp_s'] += .001
    elif change == 'token': plan['from_reference_token'] = 'wrong'
    elif change == 'source': plan['source'] = 'physically_committed'
    else: plan['delta_rad'][0] = .0101
    data['planned_advance_bound'] = plan
    with pytest.raises(F.FollowingReserveRejected): limit(robot, data)


def test_original_genuine_QP_objective_and_all_old_constraints_preserved(robot):
    solver, cfg, names, barrier, first = robot
    data = evidence(robot); guarded = limit(robot, data)
    native = solver.pink.solve_ik; original_build = native.__globals__['build_ik']
    captured = {}
    def capture(configuration, tasks, dt, *args, **kwargs):
        captured['original'] = original_build(configuration, tasks, dt, *args, **kwargs)
        if 'limits' in kwargs:
            kwargs = dict(kwargs, limits=[*kwargs['limits'], guarded])
        else:
            args = (args[0], [*args[1], guarded], *args[2:])
        captured['guarded'] = original_build(configuration, tasks, dt, *args, **kwargs)
        return captured['guarded']
    native.__globals__['build_ik'] = capture
    previous = solver.previous.copy()
    try:
        measured = dict(zip(first['joint_names'], first['joint_positions']))
        target = solver.predicted_camera(measured, np.asarray(first['T_world_base']))
        velocity = solver.velocity(measured, np.asarray(first['T_world_base']), target, .02)
    finally:
        native.__globals__['build_ik'] = original_build
        solver.previous = previous
    original, augmented = captured['original'], captured['guarded']
    np.testing.assert_array_equal(original.P, augmented.P)
    np.testing.assert_array_equal(original.q, augmented.q)
    # Pink stacks native limits, added Limit, then all unchanged barriers.
    split = len(original.h)-20
    np.testing.assert_array_equal(original.G[:split], augmented.G[:split])
    np.testing.assert_array_equal(original.h[:split], augmented.h[:split])
    np.testing.assert_array_equal(original.G[split:], augmented.G[split+20:])
    np.testing.assert_array_equal(original.h[split:], augmented.h[split+20:])
    v = np.asarray([velocity[n] for n in names])
    assert np.min(augmented.h-augmented.G @ (.02*v)) >= -1e-9
    assert np.max(np.abs(v)) <= .5 and np.max(np.abs(v))/.02 <= 1.+1e-8
