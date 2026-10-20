#!/usr/bin/env python3
"""Native closest20 reserve CPU QP comparison; no trajectory/physics rollout."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linprog


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recorded', type=Path, required=True)
    parser.add_argument('--limit-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); folder = args.recorded.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False); sources = output/'sources'; sources.mkdir()
    for name in ('g1_backend.py', 'pink_collision.py', 'g1_29dof.urdf'):
        shutil.copy2(folder/'sources'/name, sources/name)
    shutil.copy2(args.limit_source, sources/'following_reserve.py')
    shutil.copy2(__file__, sources/Path(__file__).name)
    F = load('audited_following_limit', sources/'following_reserve.py')
    backend = load('audited_following_backend', sources/'g1_backend.py')
    collision = load('audited_following_collision', sources/'pink_collision.py')
    rows = [json.loads(line) for line in (folder/'measurements.jsonl').open()]
    by_stamp = {round(row['stamp_s'], 8): row for row in rows}
    by_sequence = {row.get('pink_runtime_recovery', {}).get('seq'): row for row in rows
                   if row.get('pink_runtime_recovery', {}).get('seq') is not None}
    by_acquisition = {round(row['pink_runtime_recovery']['acquisition_stamp_s'], 8): row for row in rows
                     if row.get('pink_runtime_recovery', {}).get('acquisition_stamp_s') is not None}
    names = rows[0]['joint_names']; legs = [n for n in names if any(t in n for t in ('hip', 'knee', 'ankle'))]
    extrinsic = json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    home = dict(zip(names, rows[0]['joint_positions']))
    solver = backend.PinkUpperBody(legs, names, extrinsic, sources/'g1_29dof.urdf',
        posture_home=home, posture_cost=.03, waist_posture_cost=.15, damping_cost=.02, collision_gain=1.)
    solver.collision_helper = collision
    solver.collision_offsets = json.loads((folder/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    solver.collision_geometry = collision.geometry_model(solver.model,
        json.loads((folder/'fine_collision_geometry.json').read_text()), solver.collision_offsets)
    native = solver.pink.solve_ik; old_build = native.__globals__['build_ik']; capture = {}; record = None
    def intercept(cfg, tasks, dt, *positional, **kwargs):
        capture['original'] = old_build(cfg, tasks, dt, *positional, **kwargs)
        barriers = kwargs.get('barriers', positional[2] if len(positional) > 2 else [])
        limit = F.FollowingReserveLimit.from_native_barrier(cfg, barriers[0], record,
            task_id=record['task_id'], task_sha256=record['task_sha256'],
            reference_token=record['committed_reference']['token'], now_s=record['acquisition_stamp_s'])
        capture['manifest'] = limit.manifest()
        if 'limits' in kwargs: kwargs = dict(kwargs, limits=[*kwargs['limits'], limit])
        else: positional = (positional[0], [*positional[1], limit], *positional[2:])
        capture['guarded'] = old_build(cfg, tasks, dt, *positional, **kwargs)
        return capture['guarded']
    results = []; native.__globals__['build_ik'] = intercept
    try:
        for stamp in [rows[0]['stamp_s'], 22., 24., 25., 25.44, 25.9, 25.92, 25.94, 26., 26.5, 26.98, 27.14]:
            row = by_stamp[round(stamp, 8)]; static = row is rows[0]
            measured = dict(zip(names, row['joint_positions'])); cfg, upper = solver.configuration(measured)
            if static:
                context = dict(task_id='synthetic_static_initial_fixture', task_sha256='0'*64)
                r = cfg.q.copy(); qd = np.zeros(15); solver.previous = dict.fromkeys(upper, 0.)
                target = solver.predicted_camera(measured, np.asarray(row['T_world_base']))
                arguments = dict(posture_reference=measured)
            else:
                context = row['current_solver_task_context']
                acquired = by_acquisition[round(stamp, 8)]
                previous = by_sequence.get(acquired['pink_runtime_recovery']['seq']-1)
                if previous is None:
                    results.append(dict(stamp_s=stamp, skipped=True,
                        reason='Preceding native successful solver state was not captured; do not fabricate previous velocity.'))
                    continue
                solver.previous = {n: previous['pink_upper_velocity'].get(n, 0.) for n in upper}
                r = np.asarray([row['applied_joint_position_target'][names.index(n)] for n in upper])
                qd = np.asarray([row['measured_joint_velocity'][names.index(n)] for n in upper])
                target = np.asarray(row['T_world_reference_optical'])
                request_log = by_stamp[round(stamp+.02, 8)]
                arguments = dict(posture_reference=request_log['pink_parking_posture']['targets'],
                    **request_log['pink_requested_soft_costs'])
            reference = F.committed_reference_snapshot(task_id=context['task_id'], task_sha256=context['task_sha256'],
                joint_names=upper, positions_rad=r.tolist(), commit_seq=round(stamp/.005),
                commit_stamp_s=max(0., stamp-.005), snapshot_stamp_s=stamp)
            record = dict(schema='g1_following_evidence/v1', task_id=context['task_id'], task_sha256=context['task_sha256'],
                acquisition_stamp_s=stamp, joint_names=upper, measured_q_rad=cfg.q.tolist(),
                measured_qd_rad_s=qd.tolist(), velocity_stamp_s=stamp, measurement_source='measured_physx',
                committed_reference=reference,
                audit_scope='Historical post-step r/q/qd and native previous/soft-task context reconstructed from recorded physical rows; reference commit event/token reconstructed from frozen4substep write schedule, not an originally captured request. No future application-time r used.')
            capture.clear(); velocity = None; error = None
            try: velocity = solver.velocity(measured, np.asarray(row['T_world_base']), target, .02, **arguments)
            except Exception as exc: error = repr(exc)
            if 'guarded' not in capture:
                results.append(dict(stamp_s=stamp, error=error, limit_constructed=False)); continue
            old, new = capture['original'], capture['guarded']
            def feasible(problem):
                return bool(linprog(np.zeros(15), A_ub=problem.G, b_ub=problem.h,
                    bounds=[(None, None)]*15, method='highs').success)
            count = len(capture['manifest']['pairs']); split = len(old.h)-count
            matrix_file = f'qp_{stamp:.2f}.npz'
            np.savez_compressed(output/matrix_file, P=new.P, q=new.q, G=new.G, h=new.h,
                original_P=old.P, original_q=old.q, original_G=old.G, original_h=old.h)
            unchanged = (np.array_equal(old.P, new.P) and np.array_equal(old.q, new.q)
                and np.array_equal(old.G[:split], new.G[:split]) and np.array_equal(old.h[:split], new.h[:split])
                and np.array_equal(old.G[split:], new.G[split+count:]) and np.array_equal(old.h[split:], new.h[split+count:]))
            residual = None if velocity is None else float(np.min(new.h-new.G@(.02*np.array([velocity[n] for n in upper]))))
            results.append(dict(stamp_s=stamp, synthetic_static_fixture=static,
                original_LP_feasible=feasible(old), guarded_LP_feasible=feasible(new),
                genuine_native_QP_solved=velocity is not None, error=error,
                original_objective_and_all_old_inequalities_exactly_preserved=unchanged,
                limit_constructed=True, native_rows_added=count, minimum_QP_residual=residual,
                manifest=capture['manifest'], matrix_file=matrix_file,
                evidence=record))
    finally: native.__globals__['build_ik'] = old_build
    report = dict(schema='g1_native20_following_limit_CPU/v1', physical_execution=False,
        runtime_wired=False, accepted_request_capture='reconstructed historical states; original failed/request packets are not claimed',
        fixed_native_tasks_and_limits_preserved=all(r.get('original_objective_and_all_old_inequalities_exactly_preserved') for r in results if not r.get('skipped')),
        all_original20_CBF_retained=True, parameters=dict(L_s=.06, T_guard_s=.06,
            assumed_braking_fraction=.5, original_d_min_m=.0065, native_qdot_cap_rad_s=.5,
            issued_qddot_cap_rad_s2=1.), results=results,
        interpretation='Actual qdot plus committed reference lead only; same-acquisition reference is never changed to future application r. Planned-write/application gate tested separately with synthetic write ledgers. All20 may be infeasible where pair377-only was feasible; no weaker constraints or previous resets.',
        actual_braking_and_servo_following_certified=False)
    paths = [p for p in sources.iterdir() if p.is_file()]+[folder/'measurements.jsonl',
        folder/'fine_collision_geometry.json', folder/'pink_collision_initialization.json']
    report['evidence_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(calls=len(results), solved=sum(r.get('genuine_native_QP_solved', False) for r in results),
        original_constraints_preserved=report['fixed_native_tasks_and_limits_preserved'])))


if __name__ == '__main__':
    main()
