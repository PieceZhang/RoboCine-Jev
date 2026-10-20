#!/usr/bin/env python3
"""Conditional reference/following reserve as a genuine extra Pink inequality.

This is a CPU design experiment, not a certificate of actuator deceleration.
The request-time actual committed drive target is never replaced with a future
application-time target. Later targets are used only in an explicit ex-post
consistency audit.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linprog
from pink.barriers import SelfCollisionBarrier


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def lp(G,h):
    return linprog(np.zeros(G.shape[1]),A_ub=G,b_ub=h,bounds=[(None,None)]*G.shape[1],method='highs')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recorded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-opening-credit',action='store_true')
    p.add_argument('--extra-lookahead',type=float,default=0.)
    a=p.parse_args();folder=a.recorded.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'sources';source.mkdir()
    for name in ('g1_backend.py','pink_collision.py','g1_29dof.urdf','height_policy_io.yaml'):shutil.copy2(folder/'sources'/name,source/name)
    shutil.copy2(Path(__file__),source/Path(__file__).name)
    backend=load('reserve_g1_backend',source/'g1_backend.py');collision=load('reserve_g1_collision',source/'pink_collision.py')
    rows=[json.loads(line) for line in (folder/'measurements.jsonl').open()];by_stamp={round(r['stamp_s'],8):r for r in rows}
    by_seq={r.get('pink_runtime_recovery',{}).get('seq'):r for r in rows if r.get('pink_runtime_recovery',{}).get('seq') is not None}
    by_acquisition={round(r['pink_runtime_recovery']['acquisition_stamp_s'],8):r for r in rows if r.get('pink_runtime_recovery',{}).get('acquisition_stamp_s') is not None}
    names=rows[0]['joint_names'];legs=[n for n in names if any(t in n for t in ('hip','knee','ankle'))]
    camera=json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    import yaml
    io=yaml.safe_load((source/'height_policy_io.yaml').read_text())['articulations']['robot'];home=dict(zip(io['joint_names'],io['default_joint_pos']))
    solver=backend.PinkUpperBody(legs,names,camera,source/'g1_29dof.urdf',posture_home=home,posture_cost=.03,waist_posture_cost=.15,damping_cost=.02,collision_gain=1.)
    solver.collision_helper=collision;solver.collision_offsets=json.loads((folder/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    solver.collision_geometry=collision.geometry_model(solver.model,json.loads((folder/'fine_collision_geometry.json').read_text()),solver.collision_offsets)
    states=[]
    for row in rows:
        if not 22.<=row['stamp_s']<=27.16+1e-8:continue
        measured=dict(zip(names,row['joint_positions']));cfg,upper=solver.configuration(measured)
        distances=np.array([x.min_distance for x in cfg.collision_data.distanceResults]);indices=np.argpartition(-distances,-20)[-20:]
        barrier=SelfCollisionBarrier(20,gain=1.,safe_displacement_gain=0.,d_min=.0065)
        Jall=barrier.compute_jacobian(cfg);slot=int(np.flatnonzero(indices==377)[0]);J=Jall[slot]
        # Both q and r come from the physical row at exactly the request stamp.
        r=np.array([row['applied_joint_position_target'][names.index(n)] for n in upper])
        qd=np.array([row['measured_joint_velocity'][names.index(n)] for n in upper])
        acquired=by_acquisition.get(round(row['stamp_s'],8));previous=None;original=None
        if acquired is not None:
            prevrow=by_seq.get(acquired['pink_runtime_recovery']['seq']-1)
            if prevrow is not None:previous={n:prevrow['pink_upper_velocity'].get(n,0.) for n in upper}
            original={n:acquired['pink_upper_velocity'][n] for n in upper}
        states.append(dict(row=row,q=cfg.q.copy(),upper=upper,J=J.copy(),distance=float(distances[377]),
            closing=max(0.,float(-J@qd)),reference_rate_projection=float(J@(r-cfg.q)),
            reference=r.copy(),projected_command_acceleration_max=float(np.sum(np.abs(J))),
            previous=previous,original=original))
    # Independent static boundary: same measured native initial shape, zero
    # actual velocity/reference lead. This is synthetic kinematic evidence.
    static=rows[0];staticq=dict(zip(names,static['joint_positions']));cfg,upper=solver.configuration(staticq)
    ds=np.array([x.min_distance for x in cfg.collision_data.distanceResults]);ids=np.argpartition(-ds,-20)[-20:]
    b=SelfCollisionBarrier(20,gain=1.,safe_displacement_gain=0.,d_min=.0065);jj=b.compute_jacobian(cfg)[int(np.flatnonzero(ids==377)[0])]
    static_state=dict(row=static,q=cfg.q.copy(),upper=upper,J=jj,distance=float(ds[377]),closing=0.,reference_rate_projection=0.,
        reference=cfg.q.copy(),projected_command_acceleration_max=float(np.sum(np.abs(jj))),previous=dict.fromkeys(upper,0.),original=dict.fromkeys(upper,0.))
    native=solver.pink.solve_ik;old_build=native.__globals__['build_ik'];capture={};active_state=None;assumption_factor=.5;lag=.06
    class ReferenceFollowingLimit:
        def compute_qp_inequalities(self,q,dt):
            state=active_state;accel=assumption_factor*state['projected_command_acceleration_max']
            reserve=(lag+a.extra_lookahead)*state['closing']+state['closing']**2/(2*max(accel,1e-12))
            # Delta_q is the genuine Pink displacement variable, in radians.
            # -J*Delta_q <= d-dmin-reserve + J*(r_request_actual-q).
            reference_projection=min(0.,state['reference_rate_projection']) if a.no_opening_credit else state['reference_rate_projection']
            return -state['J'][None,:],np.array([state['distance']-.0065-reserve+reference_projection])
    def recorder(cfg,tasks,dt,*args,**kwargs):
        capture['original_problem']=old_build(cfg,tasks,dt,*args,**kwargs)
        if 'limits' in kwargs:kwargs['limits']=list(kwargs['limits'])+[ReferenceFollowingLimit()]
        else:args=tuple([args[0],list(args[1])+[ReferenceFollowingLimit()]]+list(args[2:]))
        problem=old_build(cfg,tasks,dt,*args,**kwargs);capture['problem']=problem;return problem
    native.__globals__['build_ik']=recorder
    traces=[];solutions=[]
    try:
        for assumption_factor in (.25,.5,1.):
            for lag in (.04,.06):
                metrics=[]
                for state in states:
                    accel=assumption_factor*state['projected_command_acceleration_max'];c=state['closing']
                    reserve=c*(lag+a.extra_lookahead)+c*c/(2*max(accel,1e-12))
                    physical=state['distance']-.0065-(c*lag+c*c/(2*max(accel,1e-12)))
                    reference=state['distance']-.0065-reserve+(min(0.,state['reference_rate_projection']) if a.no_opening_credit else state['reference_rate_projection'])
                    projected=None if state['original'] is None else reference+.02*state['J']@np.array([state['original'][n] for n in state['upper']])
                    metrics.append(dict(stamp_s=state['row']['stamp_s'],distance_m=state['distance'],actual_closing_m_s=c,
                        assumed_relative_braking_lower_m_s2=accel,latency_s=lag,braking_reserve_m=reserve,extra_lookahead_s=a.extra_lookahead,
                        physical_stopping_margin_m=physical,committed_reference_linear_margin_m=reference,
                        original_command_extra_inequality_residual_m=projected,
                        closing_reference_lead_m=max(0.,-state['reference_rate_projection'])))
                traces.append(dict(assumed_deceleration_fraction=assumption_factor,latency_s=lag,states=metrics))
                first=next((i for i,r in enumerate(metrics) if r['original_command_extra_inequality_residual_m'] is not None and r['original_command_extra_inequality_residual_m']<0),0)
                selected=sorted(set([max(0,first-3),max(0,first-1),first,min(len(states)-1,first+1),min(len(states)-1,first+3)]+[min(range(len(states)),key=lambda i:abs(states[i]['row']['stamp_s']-t)) for t in (24.,25.,26.,26.5,26.9,27.08,27.14)]))
                for index in [-1]+selected:
                    active_state=static_state if index==-1 else states[index]
                    if active_state['previous'] is None:continue
                    row=active_state['row'];stamp=row['stamp_s'];submitted=by_stamp.get(round(stamp+.02,8),row)
                    measured=dict(zip(names,row['joint_positions']));reference=np.asarray(row['T_world_reference_optical'])
                    if index==-1:reference=solver.predicted_camera(measured,np.asarray(row['T_world_base']))
                    solver.previous=active_state['previous'].copy();capture.clear();error=None;velocity=None
                    try:
                        velocity=solver.velocity(measured,np.asarray(row['T_world_base']),reference,.02,
                            posture_reference=active_state['previous'] if index==-1 else submitted['pink_parking_posture']['targets'],
                            **({} if index==-1 else submitted['pink_requested_soft_costs']))
                    except Exception as exc:error=repr(exc)
                    original=capture['original_problem'];guarded=capture['problem'];original_lp=lp(original.G,original.h);guarded_lp=lp(guarded.G,guarded.h)
                    gr,hr=ReferenceFollowingLimit().compute_qp_inequalities(active_state['q'],.02)
                    reserve=lag*active_state['closing']+active_state['closing']**2/(2*max(assumption_factor*active_state['projected_command_acceleration_max'],1e-12))
                    extra_residual=None if velocity is None else float((hr-gr@(.02*np.array([velocity[n] for n in active_state['upper']])))[0])
                    ident=f'{assumption_factor}_{lag}_{stamp:.2f}_{index}'
                    np.savez_compressed(out/(ident+'.npz'),P=guarded.P,q=guarded.q,G=guarded.G,h=guarded.h)
                    solutions.append(dict(stamp_s=stamp,static_zero_rate_fixture=index==-1,
                        assumption_fraction=assumption_factor,latency_s=lag,
                        original_linear_feasible=bool(original_lp.success),guarded_linear_feasible=bool(guarded_lp.success),
                        genuine_QP_solved=velocity is not None,error=error,velocity=velocity,
                        extra_inequality_residual_m=extra_residual,
                        physical_stopping_margin_m=active_state['distance']-.0065-reserve,
                        already_outside_assumed_viability=bool(active_state['distance']-.0065-reserve<0),
                        reference_request_stamp_s=stamp,reference_source='actual already-committed post-step target at same historical request stamp; not the later reply application target',
                        original_P_q_preserved=bool(np.array_equal(original.P,guarded.P) and np.array_equal(original.q,guarded.q)),
                        matrix_file=ident+'.npz'))
    finally:native.__globals__['build_ik']=old_build
    paths=[folder/'measurements.jsonl',folder/'fine_collision_geometry.json',folder/'pink_collision_initialization.json']+[p for p in source.iterdir() if p.is_file()]
    report=dict(schema='conditional_g1_reference_following_reserve/v1',physical_execution=False,
        scope='Critical controlled torso/right_shoulder_yaw pair377 prototype only; full original native20-pair CBF and15DOF Frame/QP/configuration/speed/acceleration limits remain. Historical accepted-request state/previous/target reconstructed, not captured original requests. No solver output override, no previous reset, no gamma increase.',
        equation='-J_distance*Delta_q <= d-.0065-[(lag+lookahead)*c+c^2/(2*a_assumed)] + reference_projection; c=max(0,-J_distance*qd_actual). reference_projection=J*(r-q), or min(0,J*(r-q)) for no-opening-credit mode.',
        no_opening_reference_credit=a.no_opening_credit,extra_lookahead_s=a.extra_lookahead,
        units='J:m/rad; Delta_q,reference error:rad; distance,reserve:m; c:m/s; a_assumed:m/s^2.',
        assumptions='a_assumed=factor*sum(abs(J))*1rad/s^2 is ONLY an assumed actual relative deceleration floor. Issued acceleration cap1 is an upper limit, not a guarantee of any actual minimum braking. factor1 gives an optimistic necessary-distance diagnostic, not a sufficient safety proof. Smaller fractions are conditional conservative operating hypotheses requiring physical validation.',
        viability='Negative physical_stopping_margin means already outside the assumed stop-distance region. A feasible command cannot repair the past or certify physical clearance. Guard must start earlier/always on, and escalate infeasibility without loosening limits or resetting previous.',
        asynchronous_reference='A live worker must bind the actual request-time committed reference token and planned/actual commit separately. Reusing its result after the execution reference changes requires an independently verified reference-advance bound or matching actual commit token; never silently rebase the old QP on a later r. Application-time actual tracking/clearance checks and fresh retry remain necessary.',
        traces=traces,solutions=solutions,evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(states=len(states),solutions=len(solutions),static_pass=all(s['genuine_QP_solved'] for s in solutions if s['static_zero_rate_fixture']))))


if __name__=='__main__':main()
