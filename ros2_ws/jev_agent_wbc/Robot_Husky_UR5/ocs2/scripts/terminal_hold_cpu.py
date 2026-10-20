#!/usr/bin/env python3
"""Genuine SLQ endpoint continuation replay on an ideal model, never PhysX."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROBOT=Path(__file__).resolve().parents[2]
PACKAGE=ROBOT.parent
sys.path.insert(0,str(PACKAGE/'scripts'))
from ocs2_wheeled_safety import HardCommandGuard,SkidSteerVelocityServo,terminal_reference_window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROBOT/'results/ocs2_acceptance_02')
    parser.add_argument('--seed-seconds-before-end',type=float,default=1.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sp=importlib.util.spec_from_file_location('husky_model',ROBOT/'ocs2/adapter.py')
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);adapter=module.WheeledBackendAdapter()
    task=json.loads((args.run/'task.json').read_text());trajectory=task['camera_trajectory']
    times=np.array([p['t_s'] for p in trajectory['samples']])+trajectory['start_s'];end=times[-1]
    rows=[json.loads(line) for line in (args.run/'samples.jsonl').open()]
    seed=min(rows,key=lambda r:abs(r['stamp_s']-(end-args.seed_seconds_before_end)))
    ids=[seed['joint_names'].index(n) for n in adapter.joint_names]
    base=np.array(seed['T_world_base']);yaw=float(np.arctan2(base[1,0],base[0,0]))
    tilt=Rotation.from_euler('z',-yaw).as_matrix()@base[:3,:3]
    x=np.r_[base[:2,3],yaw,np.array(seed['q'])[ids],seed['command']]
    # Public semantic camera uses +X forward/+Z up. Preserve its original pose;
    # explicitly transform once to the optical +Z forward/-Y up convention.
    sys.path.insert(0,str(PACKAGE))
    from jev_agent_wbc.wbc_protocol import TimedTask,optical_camera
    from jev_agent_wbc.wbc_physical import rigid
    parsed=TimedTask.from_dict(task)
    references=[rigid(optical_camera(p)).tolist() for p in parsed.trajectory.poses]
    model=args.run/'sources/live_model.json'
    binary=ROBOT/'ocs2/.cache/backend/camera_mpc_worker'
    process=subprocess.Popen([str(binary),str(ROBOT/'ocs2/backend.json'),str(model)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(args.output/'native.stderr.log','w'),text=True)
    lower=np.array([j['lower'] for j in adapter.model['joints']]);upper=np.array([j['upper'] for j in adapter.model['joints']])
    guard=HardCommandGuard(False,lower,upper);guard.previous=x[9:].copy()
    servo=SkidSteerVelocityServo(.1651,.5708);servo.previous=np.array([.3,-.3]);wheel_positions=np.zeros(2)
    stamp=seed['stamp_s'];states=[];policies=[];failures=[];start=time.monotonic()
    journal=(args.output/'model_samples.jsonl').open('w',buffering=1)
    try:
        while stamp<end+5.-1e-8:
            base[:3,:3]=Rotation.from_euler('z',x[2]).as_matrix()@tilt;base[:2,3]=x[:2]
            ts,poses,indices,terminal=terminal_reference_window(times,references,stamp,1.)
            terminal['chassis_brake_applied']=bool(stamp>=end-1e-8)
            terminal['all_actuators_brake_applied']=bool(stamp>=end+4.-1e-8)
            terminal['all_actuator_brake_offset_s']=4.
            request=dict(request_id=len(policies),state_stamp_s=stamp,horizon_s=1.,state=np.r_[x[:9],guard.previous].tolist(),rate_state_semantics='last_applied_velocity_command',T_world_base=base.tolist(),frame='world_optical',absolute_times_s=ts,camera_poses=poses,terminal_hold=terminal)
            process.stdin.write(json.dumps(request)+'\n');process.stdin.flush();result=json.loads(process.stdout.readline());policies.append(dict(request=request,result=result))
            if result.get('ok') is not True:failures.append(result);break
            for _ in range(min(10,round((end+5.-stamp)/.01))):
                velocity=np.array([np.interp(stamp,result['times_s'],np.array(result['states'])[:,9+k]) for k in range(8)])
                if stamp>=end-1e-8:velocity[:2]=0.
                command=guard.command(velocity,x[3:9],x[-6:],brake=stamp>=end+4.-1e-8);previous_wheels=servo.previous.copy();wheel_targets=servo.command(command[:2],x[10],brake=stamp>=end-1e-8);wheel_positions+=wheel_targets*.01
                if np.max(np.abs(wheel_targets-previous_wheels))>servo.wheel_slew*.01+1e-10:raise RuntimeError('wheel stop violated original slew')
                x[:9]+=.01*adapter.flow_map(x[:9],command);x[9:]=command;stamp=round(stamp+.01,8)
                base[:3,:3]=Rotation.from_euler('z',x[2]).as_matrix()@tilt;base[:2,3]=x[:2]
                actual=adapter.measured_camera_pose(base,x[3:9]);target=np.array(references[-1])
                row=dict(stamp_s=stamp,wheel_target_radps=wheel_targets.tolist(),wheel_positions_rad=wheel_positions.tolist(),internal_state=x.tolist(),T_world_camera_optical=actual.tolist(),position_to_original_endpoint_m=float(np.linalg.norm(actual[:3,3]-target[:3,3])),orientation_to_original_endpoint_rad=float(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()))
                states.append(row);journal.write(json.dumps(row)+'\n')
    finally:process.terminate();process.communicate(timeout=3.);journal.close()
    (args.output/'native_mpc.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in policies))
    tail=[r for r in states if r['stamp_s']>=end+4.5-1e-8]
    final=states[-1] if states else {};span=np.ptp([r['internal_state'] for r in tail],axis=0) if tail else np.full(17,np.inf)
    report=dict(scope='CPU ideal-model genuine OCS2 replay only; not PhysX, physical plant or readiness pass',seed_stamp_s=seed['stamp_s'],original_task_end_s=float(end),completed_seconds=stamp-seed['stamp_s'],failures=failures,model_endpoint_error_m=final.get('position_to_original_endpoint_m'),model_endpoint_orientation_rad=final.get('orientation_to_original_endpoint_rad'),model_last_half_second_configuration_span=span[:9].tolist() if tail else None,model_last_command=final.get('internal_state',[])[9:],solver_max_wall_s=max((p['result'].get('solver_wall_s',0) for p in policies),default=None),wall_s=time.monotonic()-start)
    report['before_all_actuator_brake']=min(states,key=lambda r:abs(r['stamp_s']-(end+4.))) if states else None
    report['final_wheel_targets_radps']=states[-1]['wheel_target_radps'] if states else None
    report['wheel_last_half_second_span_rad']=np.ptp([r['wheel_positions_rad'] for r in tail],axis=0).tolist() if tail else None
    report['initial_nonzero_base_command']=seed['command'][:2]
    report['model_pass']=bool(not failures and final and final['position_to_original_endpoint_m']<parsed.position_tolerance_m and final['orientation_to_original_endpoint_rad']<parsed.orientation_tolerance_rad and np.max(span[3:9])<.01 and np.max(span[:2])<.005 and span[2]<.01 and np.max(np.abs(states[-1]['wheel_target_radps']))==0.)
    files=[Path(__file__),PACKAGE/'scripts/ocs2_wheeled_safety.py',ROBOT/'ocs2/src/camera_mpc_worker.cpp',binary,model,args.run/'task.json',args.run/'samples.jsonl']
    report['source_files_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['model_pass'] else 2


if __name__=='__main__':raise SystemExit(main())
