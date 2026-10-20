#!/usr/bin/env python3
"""Independent measured-state diagnosis of G1 physical self-clearance abort.

No physical state is executed. Native command, measured qdot and distance
finite differences are kept distinct and use their actual acquisition stamps.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import hppfcl
import numpy as np
from scipy.optimize import linprog
from pink.barriers import SelfCollisionBarrier


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recorded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder=a.recorded.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'sources';source.mkdir()
    for name in ('g1_backend.py','pink_collision.py','g1_29dof.urdf','height_policy_io.yaml'):shutil.copy2(folder/'sources'/name,source/name)
    shutil.copy2(Path(__file__),source/Path(__file__).name)
    backend=load('abort_independent_backend',source/'g1_backend.py');collision=load('abort_independent_collision',source/'pink_collision.py')
    rows=[json.loads(line) for line in (folder/'measurements.jsonl').open()];by_stamp={round(r['stamp_s'],8):r for r in rows}
    by_seq={r.get('pink_runtime_recovery',{}).get('seq'):r for r in rows if r.get('pink_runtime_recovery',{}).get('seq') is not None}
    by_acquisition={round(r['pink_runtime_recovery']['acquisition_stamp_s'],8):r for r in rows if r.get('pink_runtime_recovery',{}).get('acquisition_stamp_s') is not None}
    names=rows[0]['joint_names'];legs=[n for n in names if any(t in n for t in ('hip','knee','ankle'))]
    camera=json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    import yaml
    io=yaml.safe_load((source/'height_policy_io.yaml').read_text())['articulations']['robot'];home=dict(zip(io['joint_names'],io['default_joint_pos']))
    solver=backend.PinkUpperBody(legs,names,camera,source/'g1_29dof.urdf',posture_home=home,posture_cost=.03,waist_posture_cost=.15,damping_cost=.02,collision_gain=1.)
    solver.collision_helper=collision
    solver.collision_offsets=json.loads((folder/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    solver.collision_geometry=collision.geometry_model(solver.model,json.loads((folder/'fine_collision_geometry.json').read_text()),solver.collision_offsets)
    def configuration(row):return solver.configuration(dict(zip(names,row['joint_positions'])))
    def pair_data(row):
        cfg,upper=configuration(row);distances=np.array([d.min_distance for d in cfg.collision_data.distanceResults]);critical=377
        b=SelfCollisionBarrier(20,gain=1.,safe_displacement_gain=0.,d_min=.0065)
        J=b.compute_jacobian(cfg);h=b.compute_barrier(cfg)
        ji=np.argpartition(-distances,-20)[-20:];hi=np.argpartition(-(distances-.0065),-20)[-20:]
        slot=int(np.flatnonzero(ji==critical)[0]);return cfg,upper,distances,J,h,ji,hi,slot
    def independent_distance(cfg,q,index):
        data=cfg.model.createData();gd=cfg.collision_model.createData();pin=solver.pin
        pin.forwardKinematics(cfg.model,data,q);pin.updateGeometryPlacements(cfg.model,data,cfg.collision_model,gd,q)
        pair=cfg.collision_model.collisionPairs[index];first=cfg.collision_model.geometryObjects[pair.first];second=cfg.collision_model.geometryObjects[pair.second]
        one,two=gd.oMg[pair.first],gd.oMg[pair.second]
        request=hppfcl.DistanceRequest();request.gjk_tolerance=1e-12;request.enable_cached_gjk_guess=False
        result=hppfcl.DistanceResult();hppfcl.distance(first.geometry,hppfcl.Transform3f(one.rotation,one.translation),second.geometry,hppfcl.Transform3f(two.rotation,two.translation),request,result)
        return float(result.min_distance)
    trace=[]
    for row in rows:
        if not 26.5<=row['stamp_s']<=27.18+1e-8:continue
        cfg,upper,d,J,h,ji,hi,slot=pair_data(row);critical_J=J[slot]
        acquired=by_acquisition.get(round(row['stamp_s'],8));command=np.array([acquired['pink_upper_velocity'][n] for n in upper]) if acquired else None
        qdot=np.array([row['measured_joint_velocity'][names.index(n)] for n in upper])
        next_row=by_stamp.get(round(row['stamp_s']+.02,8));next_distance=None
        if next_row is not None:
            next_cfg,_=configuration(next_row);next_distance=independent_distance(next_cfg,next_cfg.q,377)
        fd=None
        if round(row['stamp_s'],8) in (26.5,26.92,27.,27.08,27.14,27.16):
            fd=[]
            for i in range(len(upper)):
                plus=cfg.q.copy();minus=cfg.q.copy();plus[i]+=1e-6;minus[i]-=1e-6
                fd.append((independent_distance(cfg,plus,377)-independent_distance(cfg,minus,377))/2e-6)
        encoded_residual=None if command is None else J@command+h
        matched_residual=None if command is None else J@command+(d[ji]-.0065)
        source_geometry=row.get('fine_collision_evidence',{})
        comparisons=[x for x in rows if abs(x.get('fine_collision_evidence',{}).get('stamp_s',-100)-row['stamp_s'])<1e-8]
        actual_distance=None
        if comparisons:
            actual_distance=next(x['distance_m'] for x in comparisons[0]['fine_collision_evidence']['close_pairs'] if x['first']=='torso_link' and x['second']=='right_shoulder_yaw_link')
        lead={n:float(row['applied_joint_position_target'][names.index(n)]-row['joint_positions'][names.index(n)]) for n in upper}
        trace.append(dict(stamp_s=row['stamp_s'],critical_pair=377,distance_m=float(d[377]),same_stamp_external_FCL_m=actual_distance,
            h_J_row_mismatches=int(np.count_nonzero(ji!=hi)),critical_J_slot=slot,critical_h_pair=int(hi[slot]),
            critical_distance_gradient_m_per_rad=dict(zip(upper,map(float,critical_J))),
            independent_FD_max_error_m_per_rad=None if fd is None else float(np.max(np.abs(critical_J-fd))),
            authoritative_command_from_acquisition=row['stamp_s'] if acquired else None,
            application_interval_s=None if acquired is None else [acquired['stamp_s']-.02,acquired['stamp_s']],
            J_dot_native_command_m_s=None if command is None else float(critical_J@command),
            J_dot_measured_qdot_m_s=float(critical_J@qdot),
            correct_CBF_rhs_min_distance_rate_m_s=float(-1.*(d[377]-.0065)),
            minimum_all20_encoded_CBF_residual_m_s=None if encoded_residual is None else float(encoded_residual.min()),
            minimum_all20_matched_CBF_residual_m_s=None if matched_residual is None else float(matched_residual.min()),
            critical_CBF_command_residual_m_s=None if command is None else float(critical_J@command+d[377]-.0065),
            critical_CBF_measured_residual_m_s=float(critical_J@qdot+d[377]-.0065),
            next20ms_distance_finite_difference_m_s=None if next_distance is None else float((next_distance-independent_distance(cfg,cfg.q,377))/.02),
            measured_joint_velocity={n:float(qdot[i]) for i,n in enumerate(upper)},
            native_command=None if command is None else dict(zip(upper,map(float,command))),drive_minus_actual_rad=lead,
            actuator_effort_estimate_Nm={n:row['applied_joint_effort_estimate'][names.index(n)] for n in upper},
            maximum_upper_reference_lead_rad=max(map(abs,lead.values()))))

    # Capture real build_ik exactly once per solve, so benchmark does not count
    # a duplicate reconstruction. Both modes call native solve_ik/quadprog.
    native=solver.pink.solve_ik;old_build=native.__globals__['build_ik'];capture={};count=20
    def recorder(cfg,tasks,dt,*args,**kwargs):
        barriers=kwargs.get('barriers',args[2] if len(args)>2 else None)
        if barriers is None:
            # solve_ik passes damping,limits,barriers positionally.
            barriers=args[2]
        replacement=[type(b)(count,gain=b.gain[0],safe_displacement_gain=b.safe_displacement_gain,d_min=b.d_min) for b in barriers]
        if 'barriers' in kwargs:kwargs['barriers']=replacement
        else:args=tuple(list(args[:2])+[replacement]+list(args[3:]))
        started=time.perf_counter();problem=old_build(cfg,tasks,dt,*args,**kwargs)
        capture.update(problem=problem,build_s=time.perf_counter()-started);return problem
    native.__globals__['build_ik']=recorder
    benchmarks=[]
    exact=json.loads((folder/'exact_qp_failures.json').read_text())
    requests=[]
    for failure in exact:
        runtime=failure['pink_runtime_recovery'];diag=runtime['qp_failure']
        requests.append(dict(kind='exact_failed_request',request=runtime['failed_request'],previous=dict(zip(diag['joint_names'],diag['previous_owned_velocity'])),original=diag['qp']))
    for stamp in (23.,24.,25.,26.,26.5,26.92,27.,27.08,27.14):
        row=by_stamp.get(round(stamp,8));applied=by_acquisition.get(round(stamp,8));submitted=by_stamp.get(round(stamp+.02,8))
        if row is None or applied is None or submitted is None:continue
        previous_packet=by_seq.get(applied['pink_runtime_recovery']['seq']-1)
        if previous_packet is None:continue
        requests.append(dict(kind='reconstructed_nearby_accepted_request',request=dict(joints=dict(zip(names,row['joint_positions'])),
            T_world_pelvis=row['T_world_base'],target=row['T_world_reference_optical'],dt=.02,stamp_s=stamp,
            posture_reference=submitted['pink_parking_posture']['targets'],**submitted['pink_requested_soft_costs']),
            previous={n:previous_packet['pink_upper_velocity'].get(n,0.) for n in solver.previous},original=None,
            historically_issued_velocity=applied['pink_upper_velocity']))
    try:
        for item in requests:
            request=item['request'];outputs={}
            for count in (20,473):
                durations=[];builds=[];velocity=None;error=None
                for repeat in range(3):
                    solver.previous=item['previous'].copy();capture.clear();started=time.perf_counter()
                    try:velocity=solver.velocity(request['joints'],np.array(request['T_world_pelvis']),np.array(request['target']),request['dt'],posture_reference=request.get('posture_reference'),posture_cost_override=request.get('posture_cost_override'),damping_cost_override=request.get('damping_cost_override'))
                    except Exception as exc:error=repr(exc)
                    durations.append(time.perf_counter()-started);builds.append(capture.get('build_s'))
                problem=capture['problem'];feas=linprog(np.zeros(problem.G.shape[1]),A_ub=problem.G,b_ub=problem.h,bounds=[(None,None)]*problem.G.shape[1],method='highs')
                name=f'{request["stamp_s"]:.2f}_{count}.npz';np.savez_compressed(out/name,P=problem.P,q=problem.q,G=problem.G,h=problem.h)
                outputs[str(count)]=dict(qp_solved=velocity is not None,error=error,linear_feasible=bool(feas.success),
                    velocity=velocity,genuine_total_ms=[x*1000 for x in durations],genuine_build_ms=[x*1000 for x in builds],matrix_file=name,
                    original_captured_max_difference=None if item['original'] is None or count!=20 else {key:float(np.max(np.abs(np.asarray(getattr(problem,key))-np.asarray(item['original'][key])))) for key in ('P','q','G','h')})
            difference=None
            if outputs['20']['velocity'] is not None and outputs['473']['velocity'] is not None:
                difference=float(np.linalg.norm([outputs['20']['velocity'][n]-outputs['473']['velocity'][n] for n in solver.previous]))
            benchmarks.append(dict(stamp_s=request['stamp_s'],kind=item['kind'],outputs=outputs,all473_minus20_native_l2=difference))
    finally:native.__globals__['build_ik']=old_build
    paths=[folder/'measurements.jsonl',folder/'constraint_abort.json',folder/'exact_qp_failures.json',folder/'fine_collision_geometry.json',folder/'pink_collision_initialization.json']+[p for p in source.iterdir() if p.is_file()]
    report=dict(schema='independent_g1_physical_clearance_abort/v1',physical_execution=False,
        scope='Direct distance finite differences, genuine source barrier row association, acquisition-stamped issued command vs measured rates, and20/all473 native QP matrix/timing comparisons. Accepted-request replay inputs are reconstructed unless explicitly captured failed requests. Timing is CPU algorithm benchmark, not physics cadence.',
        trace=trace,benchmarks=benchmarks,evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(states=len(trace),row_mismatch_states=sum(r['h_J_row_mismatches']>0 for r in trace),max_FD_error=max(r['independent_FD_max_error_m_per_rad'] or 0 for r in trace),benchmark_states=len(benchmarks))))


if __name__=='__main__':main()
