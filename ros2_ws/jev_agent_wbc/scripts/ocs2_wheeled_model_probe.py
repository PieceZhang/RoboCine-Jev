#!/usr/bin/env python3
"""Offline ideal kinematic diagnostic. This never establishes PhysX readiness."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation
from ocs2_wheeled_safety import HardCommandGuard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm-offset',type=float,default=0.)
    parser.add_argument('--heading',type=float,default=0.)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    folder=args.robot_dir.resolve()/'ocs2'
    sp=importlib.util.spec_from_file_location('model_adapter',folder/'adapter.py');module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);adapter=module.WheeledBackendAdapter()
    nu=adapter.spec['input_dim'];x=np.r_[np.zeros(3),np.array(adapter.spec['initial_arm'])+args.arm_offset,np.zeros(nu)];x[2]=args.heading
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',x[2]).as_matrix();base[:2,3]=x[:2];base[2,3]=adapter.nominal_height
    center=adapter.measured_camera_pose(base,x[3:9]);times=np.arange(0.,21.0001,.05);poses=[]
    for t in times:
        phase=np.clip(t-1.,0.,20.);omega=2*np.pi/20;T=center.copy();T[:3,3]+=[.28*np.sin(omega*phase),.20*(1-np.cos(omega*phase)),.07*np.sin(2*omega*phase)];T[:3,:3]=Rotation.from_rotvec([.12*np.sin(omega*phase),.10*np.sin(2*omega*phase),.16*np.sin(omega*phase)]).as_matrix()@center[:3,:3];poses.append(T)
    process=subprocess.Popen([str(folder/'.cache/backend/camera_mpc_worker'),str(folder/'backend.json'),str(folder/'evidence/live_model_static.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(args.output/'native.stderr.log','w'),text=True)
    guard=HardCommandGuard(adapter.omni,np.array([j['lower'] for j in adapter.model['joints']]),np.array([j['upper'] for j in adapter.model['joints']]))
    position=[];orientation=[];failures=[];states=[];solvers=[];start=time.monotonic()
    journal=(args.output/'model_samples.jsonl').open('w',buffering=1)
    try:
        for i in range(210):
            t=i*.1;base[:3,:3]=Rotation.from_euler('z',x[2]).as_matrix();base[:2,3]=x[:2]
            request=dict(request_id=i,state_stamp_s=t,horizon_s=min(1.,21.-t),state=x.tolist(),T_world_base=base.tolist(),frame='world_optical',absolute_times_s=times.tolist(),camera_poses=[T.tolist() for T in poses])
            process.stdin.write(json.dumps(request)+'\n');process.stdin.flush();result=json.loads(process.stdout.readline())
            if result.get('ok') is not True:failures.append(dict(t=t,result=result));break
            solvers.append(result['solver_wall_s']);pt=np.array(result['times_s']);px=np.array(result['states']);u=np.array([np.interp(t+.1,pt,px[:,9+k]) for k in range(nu)])
            for _ in range(10):
                command=guard.command(u,x[3:9]);x[:9]+=.01*adapter.flow_map(x[:9],command);x[9:]=command
            actual=adapter.camera_pose(x[:9]);target=poses[(i+1)*2];position.append(float(np.linalg.norm(actual[:3,3]-target[:3,3])));orientation.append(float(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()));states.append(x.tolist());journal.write(json.dumps(dict(t_s=t+.1,internal_state=x.tolist(),position_error_m=position[-1],orientation_error_rad=orientation[-1],solver_wall_s=solvers[-1]))+'\n')
    finally:process.terminate();process.wait(timeout=2.);journal.close()
    files=[Path(__file__),folder/'adapter.py',folder/'backend.json',folder/'evidence/live_model_static.json',folder/'.cache/backend/camera_mpc_worker']
    report=dict(scope='offline ideal kinematic model feedback diagnostic, NOT PhysX and NOT readiness evidence',initial_arm_offset_rad=args.arm_offset,initial_heading_rad=args.heading,period_s=.1,completed_model_seconds=len(states)*.1,failures=failures,position_rmse_m=float(np.sqrt(np.mean(np.square(position)))) if position else None,position_max_m=max(position) if position else None,orientation_rmse_rad=float(np.sqrt(np.mean(np.square(orientation)))) if orientation else None,solver_max_wall_s=max(solvers) if solvers else None,wall_s=time.monotonic()-start,sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if not failures else 2


if __name__=='__main__':raise SystemExit(main())
