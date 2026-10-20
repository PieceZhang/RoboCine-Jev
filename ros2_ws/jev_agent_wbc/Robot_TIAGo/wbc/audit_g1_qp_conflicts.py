#!/usr/bin/env python3
"""Genuine frozen-Pink QP/LP cross checks near recorded solver failures.

Historical exact failed requests/worker previous commands were not logged.
Every input below is explicitly a nearby measured-state counterfactual.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import linprog


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lp(G, h):
    return linprog(np.zeros(G.shape[1]), A_ub=G, b_ub=h,
        bounds=[(None, None)]*G.shape[1], method='highs',
        options={'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recorded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--barrier-gain', type=float, default=.1)
    parser.add_argument('--previous-modes', nargs='+', choices=('last_logged_native','zero_previous'),
                        default=['last_logged_native','zero_previous'])
    parser.add_argument('--braking-probe', action='store_true')
    args = parser.parse_args()
    folder, out = args.recorded.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'sources'
    frozen.mkdir()
    for name in ('g1_backend.py','pink_collision.py','g1_29dof.urdf','height_policy_io.yaml'):
        shutil.copy2(folder/'sources'/name, frozen/name)
    shutil.copy2(Path(__file__), frozen/Path(__file__).name)
    backend, collision = load('conflict_g1_backend', frozen/'g1_backend.py'), load('conflict_g1_collision', frozen/'pink_collision.py')
    rows = [json.loads(line) for line in (folder/'measurements.jsonl').open()]
    metadata = json.loads((folder/'fine_collision_geometry.json').read_text())
    initialization = json.loads((folder/'pink_collision_initialization.json').read_text())
    camera = np.array(json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    names = rows[0]['joint_names']
    legs = [n for n in names if any(t in n for t in ('hip','knee','ankle'))]
    import yaml
    io = yaml.safe_load((frozen/'height_policy_io.yaml').read_text())['articulations']['robot']
    home = dict(zip(io['joint_names'],io['default_joint_pos']))
    solver = backend.PinkUpperBody(legs, names, camera, frozen/'g1_29dof.urdf',
        posture_home=home,posture_cost=.03,waist_posture_cost=.15,damping_cost=.02)
    solver.collision_helper = collision
    solver.collision_offsets = initialization['physical_body_frame_offsets']
    solver.collision_geometry = collision.geometry_model(solver.model,metadata,solver.collision_offsets)
    original_solve = solver.pink.solve_ik
    capture = {}

    def audited_solve(configuration,tasks,dt,**kwargs):
        # CPU counterfactual only: instantiate the genuine upstream class with
        # a declared alternative class-K gain. No runtime files are changed.
        kwargs['barriers']=[type(b)(b.dim,gain=args.barrier_gain,
            safe_displacement_gain=b.safe_displacement_gain,d_min=b.d_min) for b in kwargs['barriers']]
        problem = solver.pink.build_ik(configuration,tasks,dt,
            limits=kwargs['limits'],barriers=kwargs['barriers'])
        group = []
        offset = 0
        for item in kwargs['limits']:
            result = item.compute_qp_inequalities(configuration.q,dt)
            if result is not None:
                G,h = result
                group.append(dict(kind=type(item).__name__,start=offset,end=offset+len(h)))
                offset += len(h)
        pairs = []
        for barrier in kwargs['barriers']:
            G,h = barrier.compute_qp_inequalities(configuration,dt)
            distances = np.array([d.min_distance for d in configuration.collision_data.distanceResults])
            chosen = np.argpartition(-(distances-barrier.d_min),-barrier.dim)[-barrier.dim:]
            for j,k in enumerate(chosen):
                pair = configuration.collision_model.collisionPairs[int(k)]
                objs = configuration.collision_model.geometryObjects
                pairs.append(dict(row=offset+j,pair_index=int(k),distance_m=float(distances[k]),
                    first=objs[pair.first].name,second=objs[pair.second].name,
                    gradient_norm=float(np.linalg.norm(G[j]*dt))))
            group.append(dict(kind=type(barrier).__name__,start=offset,end=offset+len(h)))
            offset += len(h)
        capture.update(problem=problem,groups=group,pairs=pairs,
            names=list(configuration.model.names)[1:],q=configuration.q.copy())
        return original_solve(configuration,tasks,dt,**kwargs)

    solver.pink.solve_ik = audited_solve
    failures = (14.96,15.5,16.14,16.66,17.88)
    reports = []
    full_data=solver.model.createData()
    full_geometry_data=solver.collision_geometry.createData()

    def distance_and_margin(joints):
        q=collision.full_configuration(solver.model,joints)
        solver.pin.forwardKinematics(solver.model,full_data,q)
        solver.pin.updateGeometryPlacements(solver.model,full_data,solver.collision_geometry,full_geometry_data,q)
        solver.pin.computeDistances(solver.model,full_data,solver.collision_geometry,full_geometry_data,q)
        return dict(minimum_all548_distance_m=float(min(x.min_distance for x in full_geometry_data.distanceResults)),
            position_margin_rad=float(min(np.min(q-solver.model.lowerPositionLimit),np.min(solver.model.upperPositionLimit-q))))

    def braking_probe(joints,velocity):
        """Finite20ms ideal upper-only braking, original1rad/s²; legs frozen."""
        q=joints.copy();v=velocity.copy();steps=[]
        start=time.monotonic();count=0
        while any(abs(x)>1e-12 for x in v.values()):
            remaining=start+count*.02-time.monotonic()
            if remaining>0:time.sleep(remaining)
            next_v={n:float(np.sign(x)*max(0.,abs(x)-.02)) for n,x in v.items()}
            for n in v:q[n]+=.01*(v[n]+next_v[n])
            sample=distance_and_margin(q);sample['stamp_relative_s']=(count+1)*.02
            steps.append(sample);v=next_v;count+=1
        return dict(upper_only_frozen_measured_legs=True,continuous_swept_certificate=False,
            original_acceleration_rad_s2=1.,sample_dt_s=.02,samples=steps,
            minimum_sampled_all548_distance_m=min(x['minimum_all548_distance_m'] for x in steps) if steps else distance_and_margin(q)['minimum_all548_distance_m'],
            minimum_sampled_joint_position_margin_rad=min(x['position_margin_rad'] for x in steps) if steps else distance_and_margin(q)['position_margin_rad'])
    scheduled_start = time.monotonic()
    try:
        for failure in failures:
            for delta in (-.04,-.02,0.):
                index = min(range(len(rows)),key=lambda i:abs(rows[i]['stamp_s']-(failure+delta)))
                row = rows[index]
                legal = next(r for r in reversed(rows[:index+1]) if r.get('pink_upper_velocity'))
                for mode in args.previous_modes:
                    elapsed_count = len(reports)
                    remaining = scheduled_start+elapsed_count*.02-time.monotonic()
                    if remaining>0:time.sleep(remaining)
                    previous = dict.fromkeys(solver.previous,0.)
                    if mode=='last_logged_native':previous.update(legal['pink_upper_velocity'])
                    solver.previous=previous.copy();capture.clear()
                    measured=dict(zip(names,row['joint_positions']))
                    error=None
                    try:
                        velocity=solver.velocity(measured,np.asarray(row['T_world_base']),
                            np.asarray(row['T_world_reference_optical']),.02,
                            posture_reference=row['pink_parking_posture']['targets'],
                            **row['pink_requested_soft_costs'])
                    except Exception as exc:
                        error=repr(exc);velocity=None
                    problem=capture['problem'];G,h=np.asarray(problem.G),np.asarray(problem.h)
                    feasible=lp(G,h)
                    groups=capture['groups'];command=next(x for x in groups if x['kind']=='CommandLimit')
                    n=G.shape[1];hi=h[command['start']:command['start']+n]
                    lo=-h[command['start']+n:command['end']]
                    conflicts=[]
                    for pair in capture['pairs']:
                        a=G[pair['row']]
                        minimum=float(np.where(a>=0,a*lo,a*hi).sum())
                        if minimum>h[pair['row']]+1e-10:
                            conflicts.append(dict(**pair,barrier_rhs=float(h[pair['row']]),
                                minimum_over_command_box=minimum,
                                infeasibility_gap_m_s=float(minimum-h[pair['row']]),
                                influential_joints=sorted([dict(name=name,coefficient=float(a[i]),
                                    delta_lo=float(lo[i]),delta_hi=float(hi[i]),
                                    minimum_contribution=float(a[i]*(lo[i] if a[i]>=0 else hi[i])))
                                    for i,name in enumerate(capture['names']) if abs(a[i])>1e-9],
                                    key=lambda x:abs(x['minimum_contribution']),reverse=True)))
                    subset=None
                    if feasible.status==2:
                        # Deletion filter returns an irreducible infeasible
                        # row set, not a minimum-cardinality global IIS.
                        active=list(range(len(h)))
                        for candidate in list(active):
                            trial=[i for i in active if i!=candidate]
                            if lp(G[trial],h[trial]).status==2:active=trial
                        subset=[dict(row=i,kind=next(x['kind'] for x in groups if x['start']<=i<x['end']),
                            rhs=float(h[i]),coefficients=G[i].tolist()) for i in active]
                    ident=f'{failure:.2f}_{delta:+.2f}_{mode}'
                    np.savez_compressed(out/(ident+'.npz'),P=problem.P,q=problem.q,G=G,h=h,
                        position=capture['q'],previous=np.array([previous[n] for n in capture['names']]))
                    before=distance_and_margin(measured)
                    after=None;braking=None
                    if velocity is not None:
                        next_joints=measured.copy()
                        for n,value in velocity.items():next_joints[n]+=.02*value
                        after=distance_and_margin(next_joints)
                        after.update(peak_speed_rad_s=max(map(abs,velocity.values())),
                            peak_change_over20ms_rad_s2=max(abs(velocity[n]-previous[n])/.02 for n in velocity),
                            minimum_barrier_velocity_residual_m_s=float(min(
                                h[x['row']]-G[x['row']]@np.array([velocity[n]*.02 for n in capture['names']]) for x in capture['pairs'])))
                        if args.braking_probe:braking=braking_probe(next_joints,velocity)
                    reports.append(dict(failure_reported_at_s=failure,nearby_measurement_s=row['stamp_s'],
                        phase=row['phase'],previous_mode=mode,logged_native_source_s=legal['stamp_s'],
                        qp_solved=velocity is not None,quadprog_error=error,
                        linear_feasible=bool(feasible.success),linear_status=int(feasible.status),
                        lp_message=feasible.message,lp_max_residual=None if feasible.x is None else float(np.max(G@feasible.x-h)),
                        minimum_H_eigenvalue=float(np.linalg.eigvalsh(problem.P).min()),
                        condition_H=float(np.linalg.cond(problem.P)),constraint_groups=groups,
                        minimum_reduced_pair_distance_m=min(x['distance_m'] for x in capture['pairs']),
                        single_barrier_command_box_conflicts=conflicts,
                        irreducible_infeasible_rows=subset,
                        fixed_measured_lower_pose_before=before,ideal_upper20ms_after=after,
                        bounded_braking_probe=braking,
                        saved_matrix=ident+'.npz'))
                    print(json.dumps({k:reports[-1][k] for k in ('failure_reported_at_s','nearby_measurement_s','previous_mode','qp_solved','linear_feasible')}),flush=True)
    finally:
        solver.pink.solve_ik=original_solve
    paths=[folder/'measurements.jsonl',folder/'fine_collision_geometry.json',folder/'pink_collision_initialization.json',
           folder/'sensor_mounts.json',folder/'pink.stderr.log']+[p for p in frozen.iterdir() if p.is_file()]
    result=dict(schema='independent_g1_near_failure_qp/v1',
        scope='Non-exact nearby measured-state counterfactuals. Failed worker input and its last internally solved velocity were absent from original log; no claimed exact reproduction or actual failure attribution.',
        original_speed_rad_s=.5,original_acceleration_rad_s2=1.,original_dt_s=.02,
        upper_pair_count=473,external_pair_count=548,barrier_closest_count=20,barrier_d_min_m=.0065,barrier_gain_per_s=args.barrier_gain,
        class_K_formula='J_distance(q) * qdot >= -alpha * (distance(q)-.0065); J in m/rad, qdot in rad/s, RHS m/s. Linear20ms distance-reserve factor=1-alpha*.02.',
        ideal_linear20ms_reserve_factor=1-args.barrier_gain*.02,
        cpu_pace=1,physical_execution=False,samples=reports,
        evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
