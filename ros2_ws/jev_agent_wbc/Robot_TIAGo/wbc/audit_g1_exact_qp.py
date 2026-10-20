#!/usr/bin/env python3
"""Original captured Pink Problem + same-request genuine class-K comparisons."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import linprog
import qpsolvers


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def feasible(G,h):
    return linprog(np.zeros(G.shape[1]),A_ub=G,b_ub=h,bounds=[(None,None)]*G.shape[1],method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recorded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gains',type=float,nargs='+',default=[.1,.5,1.])
    a=p.parse_args();folder=a.recorded.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False);frozen=out/'sources';frozen.mkdir()
    for name in ('g1_backend.py','pink_collision.py','g1_29dof.urdf','height_policy_io.yaml'):
        shutil.copy2(folder/'sources'/name,frozen/name)
    shutil.copy2(Path(__file__),frozen/Path(__file__).name)
    row=json.loads((folder/'exact_qp_failures.json').read_text())[0]
    runtime=row['pink_runtime_recovery'];diag=runtime['qp_failure'];request=runtime['failed_request']
    original={k:None if v is None else np.asarray(v,float) for k,v in diag['qp'].items()}
    baseline_gain=diag.get('collision_class_K_gain_per_s',.1)
    G,h=original['G'],original['h'];groups=diag['inequality_groups'];n=G.shape[1]
    assert diag['qp_capture']=='native_pink_exception_frame' and not diag['reconstructed_after_solver_failure']
    exact_lp=feasible(G,h);assert exact_lp.status==2
    active=list(range(len(h)))
    for i in list(active):
        trial=[j for j in active if j!=i]
        if feasible(G[trial],h[trial]).status==2:active=trial
    iis=[]
    for i in active:
        group=next(g for g in groups if g['row_start']<=i<g['row_start']+g['row_count'])
        item=dict(row=i,group=group['name'],rhs=float(h[i]),coefficients=G[i].tolist())
        local=i-group['row_start']
        if group['name']=='owned_velocity_acceleration':
            item.update(joint=diag['joint_names'][local % n],bound='upper' if local<n else 'lower')
        elif 'pairs' in group:item['pair']=group['pairs'][local]
        iis.append(item)
    barrier=next(g for g in groups if g['name']=='native_pink_self_collision')
    barrier_start=barrier['row_start']
    gamma_G=np.c_[G,np.r_[np.zeros(barrier_start),-h[barrier_start:]/baseline_gain]]
    gamma_h=np.r_[h[:barrier_start],np.zeros(len(h)-barrier_start)]
    minimum_gain=linprog(np.r_[np.zeros(n),1.],A_ub=gamma_G,b_ub=gamma_h,
        bounds=[(None,None)]*n+[(.1,None)],method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    backend=load('exact_frozen_g1',frozen/'g1_backend.py');collision=load('exact_frozen_collision',frozen/'pink_collision.py')
    import yaml
    io=yaml.safe_load((frozen/'height_policy_io.yaml').read_text())['articulations']['robot']
    home=dict(zip(io['joint_names'],io['default_joint_pos']))
    names=row['joint_names'];legs=[name for name in names if any(x in name for x in ('hip','knee','ankle'))]
    camera=json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    solver=backend.PinkUpperBody(legs,names,camera,frozen/'g1_29dof.urdf',
        posture_home=home,posture_cost=.03,damping_cost=.02,waist_posture_cost=.15)
    metadata=json.loads((folder/'fine_collision_geometry.json').read_text())
    solver.collision_helper=collision
    solver.collision_offsets=json.loads((folder/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    solver.collision_geometry=collision.geometry_model(solver.model,metadata,solver.collision_offsets)
    native_solve=solver.pink.solve_ik;captured={};gain=.1
    def wrapped(configuration,tasks,dt,**kwargs):
        kwargs['barriers']=[type(b)(b.dim,gain=gain,safe_displacement_gain=b.safe_displacement_gain,d_min=b.d_min) for b in kwargs['barriers']]
        problem=solver.pink.build_ik(configuration,tasks,dt,limits=kwargs['limits'],barriers=kwargs['barriers'])
        captured.update(problem=problem)
        return native_solve(configuration,tasks,dt,**kwargs)
    solver.pink.solve_ik=wrapped
    full_data=solver.model.createData();full_geometry=solver.collision_geometry.createData()
    def clearance(joints):
        q=collision.full_configuration(solver.model,joints)
        solver.pin.forwardKinematics(solver.model,full_data,q)
        solver.pin.updateGeometryPlacements(solver.model,full_data,solver.collision_geometry,full_geometry,q)
        solver.pin.computeDistances(solver.model,full_data,solver.collision_geometry,full_geometry,q)
        return dict(minimum_all548_distance_m=float(min(d.min_distance for d in full_geometry.distanceResults)),
            joint_position_margin_rad=float(min(np.min(q-solver.model.lowerPositionLimit),np.min(solver.model.upperPositionLimit-q))))
    comparisons=[]
    try:
        for gain in a.gains:
            solver.previous=dict(zip(diag['joint_names'],diag['previous_owned_velocity']))
            error=None;velocity=None
            try:
                velocity=solver.velocity(request['joints'],np.asarray(request['T_world_pelvis']),np.asarray(request['target']),request['dt'],
                    posture_reference=request.get('posture_reference'),posture_cost_override=request.get('posture_cost_override'),
                    damping_cost_override=request.get('damping_cost_override'))
            except Exception as exc:error=repr(exc)
            problem=captured['problem'];rebuild_lp=feasible(problem.G,problem.h)
            exact_h=h.copy();exact_h[barrier_start:]*=gain/baseline_gain
            exact_gain_lp=feasible(G,exact_h)
            exact_qp=qpsolvers.solve_problem(qpsolvers.Problem(original['P'],original['q'],G,exact_h),solver='quadprog')
            before=clearance(request['joints']);after=None;brake=[]
            if velocity is not None:
                next_joints=request['joints'].copy()
                for name,value in velocity.items():next_joints[name]+=request['dt']*value
                after=clearance(next_joints)
                after.update(peak_velocity_rad_s=max(map(abs,velocity.values())),
                    peak_acceleration_rad_s2=max(abs(velocity[name]-solver_previous)/request['dt']
                        for name,solver_previous in zip(diag['joint_names'],diag['previous_owned_velocity'])))
                v=velocity.copy();start=time.monotonic();count=0
                while any(abs(x)>1e-12 for x in v.values()):
                    wait=start+count*.02-time.monotonic()
                    if wait>0:time.sleep(wait)
                    next_v={name:float(np.sign(value)*max(0.,abs(value)-.02)) for name,value in v.items()}
                    for name in v:next_joints[name]+=.01*(v[name]+next_v[name])
                    sample=clearance(next_joints);sample['relative_s']=(count+1)*.02;brake.append(sample)
                    v=next_v;count+=1
            comparisons.append(dict(gain_per_s=gain,genuine_rebuilt_solved=velocity is not None,error=error,
                genuine_rebuilt_linear_feasible=bool(rebuild_lp.success),
                original_matrix_scaled_barrier_h_lp_feasible=bool(exact_gain_lp.success),
                original_matrix_scaled_barrier_h_quadprog_solved=exact_qp.x is not None,
                rebuilt_difference_from_original_at_original_gain={key:float(np.max(np.abs(np.asarray(getattr(problem,key))-original[key])))
                    for key in ('P','q','G','h')} if gain==baseline_gain else None,
                before=before,ideal_upper20ms_after=after,ideal_upper_brake20ms_samples=brake,
                minimum_ideal_brake_distance_m=min(x['minimum_all548_distance_m'] for x in brake) if brake else None,
                minimum_braking_joint_margin_rad=min(x['joint_position_margin_rad'] for x in brake) if brake else None,
                velocity=velocity))
            np.savez_compressed(out/f'gain_{gain}.npz',P=problem.P,q=problem.q,G=problem.G,h=problem.h)
    finally:solver.pink.solve_ik=native_solve
    paths=[folder/'exact_qp_failures.json',folder/'fine_collision_geometry.json',folder/'pink_collision_initialization.json',
        folder/'sensor_mounts.json']+[x for x in frozen.iterdir() if x.is_file()]
    result=dict(schema='independent_g1_original_failed_qp/v1',physical_execution=False,
        exact_original_capture=diag['qp_capture'],failed_request_stamp_s=request['stamp_s'],reported_post_stamp_s=row['stamp_s'],
        scope='Original captured P/q/G/h gives exact linear infeasibility/IIS. Gain comparisons preserve original request and previous velocities; genuine rebuilt model may differ by native FCL query roundoff. Only finite20ms ideal upper/body-distance probes with measured legs frozen; no physical or continuous-swept certification.',
        speed_cap_rad_s=.5,acceleration_cap_rad_s2=1.,dt_s=.02,d_min_m=.0065,external_FCL_threshold_m=.005,
        reduced_pairs=473,all_pairs=548,closest_selected=20,
        exact_lp_status=int(exact_lp.status),irreducible_original_rows=iis,
        minimum_feasible_class_K_gain_for_this_original_matrix=float(minimum_gain.x[-1]) if minimum_gain.success else None,
        original_class_K_gain_per_s=baseline_gain,
        ideal_discrete_distance_reserve_factor={str(g):1-g*.02 for g in a.gains},comparisons=comparisons,
        evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('exact_lp_status','minimum_feasible_class_K_gain_for_this_original_matrix')}))
    for x in comparisons:print(json.dumps(x))


if __name__=='__main__':main()
