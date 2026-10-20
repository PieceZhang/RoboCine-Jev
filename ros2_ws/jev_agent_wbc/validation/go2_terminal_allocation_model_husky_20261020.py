#!/usr/bin/env python3
"""CPU operating-model feasibility, not learned18/contact/parking certification.

The immutable WORLD end pose is held explicitly after original Task.end. A
bounded native body vx/vy input is allocated for1.6s, then original slew brakes
it; all three native body inputs are exactly zero by2s inside original5s stop.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_world_z_01'
OUTPUT=ROOT/'Robot_Go2/results/terminal_allocation_cpu_01'
sys.path.insert(0,str(ROOT))
from jev_agent_wbc.wbc_protocol import TimedTask


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def issued_schedule(desired_xy,previous,*,control_dt=.02):
    desired=np.r_[np.asarray(desired_xy),0.];previous=np.asarray(previous).copy()
    values=[];slew=np.array([.5,.5,.8]);times=np.arange(250)*control_dt
    for elapsed in times:
        target=desired if elapsed<1.6-1e-9 else np.zeros(3)
        candidate=np.clip(target,previous-slew*control_dt,previous+slew*control_dt)
        # The deadline is part of the commanded plan, not a downstream cut.
        bound=np.minimum(.2,slew*max(0.,2.-elapsed))
        candidate=np.clip(candidate,-bound,bound)
        values.append(candidate);previous=candidate
    return times,np.asarray(values)


def model_rollout(desired_xy,previous,initial_velocity,input_history,body,end_ee,xyz,euler):
    control_times,issued=issued_schedule(desired_xy,previous)
    h=.005;times=np.arange(1001)*h;t=times[:-1];delay=.15;tau=.35
    historical_times=np.asarray([s[0]for s in input_history]);historical_inputs=np.asarray([s[1]for s in input_history])
    all_times=np.r_[historical_times,control_times];all_inputs=np.vstack([historical_inputs,issued])
    indices=np.clip(np.searchsorted(all_times,t-delay,side='right')-1,0,len(all_times)-1)
    delayed=all_inputs[indices]
    backward=bool(desired_xy[0]<0 or initial_velocity[0]<-.01 or previous[0]<0)
    if backward:
        decay=np.exp(-h/tau);beta=.55*(1-decay)
        following,_=lfilter([beta],[1.,-decay],delayed[:,0],zi=[decay*initial_velocity[0]])
        before=np.r_[initial_velocity[0],following[:-1]]
        dx=before*tau*(1-decay)+.55*delayed[:,0]*(h-tau*(1-decay))
        vx=np.r_[initial_velocity[0],following]
    else:
        v=np.where(t<delay,initial_velocity[0],delayed[:,0]);dx=v*h;vx=np.r_[v,v[-1]]
    vy=np.where(t<delay,initial_velocity[1],delayed[:,1]);wz=np.where(t<delay,initial_velocity[2],delayed[:,2])
    yaw=np.r_[0.,np.cumsum(wz*h)];middle=yaw[:-1]+wz*h/2
    base_rotation=Rotation.from_euler('z',yaw).as_matrix()@body[:3,:3]
    local=np.c_[dx,vy*h,np.zeros_like(dx)]@body[:3,:3].T
    c,s=np.cos(middle),np.sin(middle)
    world=np.c_[c*local[:,0]-s*local[:,1],s*local[:,0]+c*local[:,1]]
    positions=np.tile(body[:3,3],(len(times),1));positions[1:,:2]+=np.cumsum(world,axis=0)
    # Root tilt/height are measured initial conditions, not controlled DOFs.
    link_positions=positions+np.einsum('nij,j->ni',base_rotation,[0,0,.07])
    relative=np.einsum('nji,nj->ni',base_rotation,end_ee[:3,3]-link_positions)
    native_xyz=np.c_[relative[:,:2],np.full(len(times),end_ee[2,3])]
    native_angles=Rotation.from_matrix(np.swapaxes(base_rotation,1,2)@end_ee[:3,:3]).as_euler('xyz')
    # Preserve original reserve ramp over.8s; sample every.1s plus both phase
    # boundaries. These are prediction constraints, not future FCL evidence.
    selected=np.unique(np.r_[np.arange(0,1001,20),320,400,1000]).astype(int)
    reserve=np.minimum(1.,times[selected]/.8)
    margins=np.c_[native_xyz[selected]-xyz[:,0]-reserve[:,None]*[.035,.035,.015],
        xyz[:,1]-native_xyz[selected]-reserve[:,None]*[.035,.035,.015],
        native_angles[selected]-euler[0]-.015*reserve[:,None],
        euler[1]-native_angles[selected]-.015*reserve[:,None]]
    return dict(times=times,issued_times=control_times,issued=issued,native_xyz=native_xyz,
        native_angles=native_angles,margins=margins,base_positions=positions,
        body_velocity=np.c_[vx,np.r_[vy,vy[-1]],np.r_[wz,wz[-1]]])


def main():
    started=time.monotonic();OUTPUT.mkdir(exist_ok=False)
    backend=load('terminal_model_backend',RUN/'sources/go2_backend.py');backend.UPSTREAM_MDP=RUN/'sources/go2_upstream'
    domain=backend.PolicyDomain();bounds=domain.check([.5,0,.5,1,0,0,0])
    xyz=np.asarray(bounds['xyz_outer_bounds']);euler=np.asarray(bounds['euler_outer_bounds_rad'])
    sampler_module=load('terminal_model_sampler',RUN/'sources/optical_sampler.py')
    tasks=[TimedTask.from_dict(task)for task in json.loads((RUN/'timed_tasks.json').read_text())]
    summary=json.loads((RUN/'summary.json').read_text());rows=[json.loads(line)for line in (RUN/'measurements.jsonl').read_text().splitlines()]
    extrinsic=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical']);results={}
    for task,result in zip(tasks,summary['cases']):
        stop_index=next(i for i,row in enumerate(rows)if row['case']==result['case']and row['phase']=='stop')
        previous_row=rows[stop_index-1];body=np.asarray(previous_row['T_world_base']);previous=np.asarray(previous_row['applied_policy_velocity'])
        measured=np.asarray(previous_row['measured_body_velocity']);stamp=task.trajectory.end_s
        endpoint=sampler_module.OpticalSampler(task).sample(stamp);end_ee=endpoint@np.linalg.inv(extrinsic)
        history=[(row['stamp_s']-.02-stamp,row['applied_policy_velocity'])for row in rows[max(0,stop_index-32):stop_index]]
        results[result['case']]={}
        for xy_target in [[.58,None],[.58,.15]]:
            cache={}
            def prediction(v):
                key=tuple(np.asarray(v));
                if key not in cache:cache[key]=model_rollout(v,previous,measured,history,body,end_ee,xyz,euler)
                return cache[key]
            def objective(v):
                p=prediction(v);target=np.array([xy_target[0],p['native_xyz'][0,1]if xy_target[1]is None else xy_target[1]])
                error=(p['native_xyz'][-1,:2]-target)/.1
                if xy_target[1]is None:error[1]=0
                return float(error@error+1e-5*np.sum((np.asarray(v)/.2)**2))
            constraint=lambda v:prediction(v)['margins'].reshape(-1)
            attempts=[]
            for seed in [np.zeros(2),[-.08,.08],[-.15,.15],[.08,-.08]]:
                solved=minimize(objective,seed,method='SLSQP',bounds=[(-.2,.2)]*2,
                    constraints=[dict(type='ineq',fun=constraint)],options=dict(maxiter=80,ftol=1e-11))
                attempts.append(solved)
            feasible=[attempt for attempt in attempts if prediction(attempt.x)['margins'].min()>=-1e-7]
            solved=min(feasible or attempts,key=lambda attempt:objective(attempt.x));p=prediction(solved.x)
            issued=p['issued'];slew=np.diff(np.vstack([previous,issued]),axis=0)/.02
            zero=issued[p['issued_times']>=2.-1e-9]
            name='x58'if xy_target[1]is None else'xy58_15'
            results[result['case']][name]=dict(original_stop_pass=result['strict_stop']['passed'],target_xy_link0_m=xy_target,
                solver_success=bool(solved.success),message=str(solved.message),desired_allocation_body_xy=solved.x.tolist(),
                nonlinear_min_reserved_domain_margin=float(p['margins'].min()),
                original_native_xyz=p['native_xyz'][0].tolist(),native_xyz_at_1p6_s=p['native_xyz'][320].tolist(),
                native_xyz_at_2_s=p['native_xyz'][400].tolist(),native_xyz_at_5_s=p['native_xyz'][-1].tolist(),
                predicted_body_velocity_at_2_s=p['body_velocity'][400].tolist(),predicted_body_velocity_at_5_s=p['body_velocity'][-1].tolist(),
                maximum_issued_abs=np.max(abs(issued),axis=0).tolist(),maximum_slew_abs=np.max(abs(slew),axis=0).tolist(),
                last3s_all_issued_exact_zero=bool(np.all(zero==0)),endpoint_optical_unchanged=True,
                joint_self_collision_future_status='unknown',learned18_parking_status='unverified',
                command_caps_and_slew_pass=bool(np.all(abs(issued)<=.2+1e-9)and np.all(abs(slew)<=[.5+1e-9,.5+1e-9,.8+1e-9])))
            np.savez_compressed(OUTPUT/(result['case']+'_'+name+'.npz'),**p)
    report=dict(schema='go2_terminal_allocation_operating_model/v1',results=results,
        scope='CPU kinematic operating-model optimization only. X uses measured backward gain.55/tau.35/delay.15; Y/yaw retain existing measured-inertia then delayed-input approximation. Frozen initial full tilt/height; no learned18, joints, FCL/contact or gait stability prediction.',
        phase=dict(allocation_max_s=1.6,original_slew_braking_max_s=.4,all_issued_zero_deadline_s=2,pure_zero_min_s=3,total_original_stop_s=5),
        reference='Exactly original absolute Task.end optical pose; explicit constant endpoint hold, no Trajectory extrapolation, retiming, knot editing or time-axis extension.',
        original_caps=[.2,.2,.2],original_slew=[.5,.5,.8],original_reserve=[.035,.035,.015],reserve_ramp_s=.8,
        limits='Satisfying this mathematical operating model is insufficient evidence for full physical parking; actual base rates and all18q must still pass original gates.',
        cpu_wall_s=time.monotonic()-started,physical_verified=False,
        files_sha256={str(path.relative_to(ROOT)):sha(path)for path in [Path(__file__),RUN/'measurements.jsonl',RUN/'timed_tasks.json',RUN/'sources/camera_coordinator.py',RUN/'sources/optical_sampler.py']})
    (OUTPUT/'model_results.json').write_text(json.dumps(report,indent=2)+'\n')
    for case,result in results.items():
        print(case,{name:dict(v=item['desired_allocation_body_xy'],margin=item['nonlinear_min_reserved_domain_margin'],
            xy2=item['native_xyz_at_2_s'][:2],xy5=item['native_xyz_at_5_s'][:2],zero=item['last3s_all_issued_exact_zero'],
            caps=item['command_caps_and_slew_pass'])for name,item in result.items()})


if __name__=='__main__':main()
