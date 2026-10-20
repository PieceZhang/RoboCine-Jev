#!/usr/bin/env python3
"""CPU source-FK fixture for full native Executor and unchanged timed SE3 curve."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'wbc')]
from async_client import AsyncGiskardClient
from source_state import source_state
from kinematics import TiagoKinematics
from physical_control import CameraController


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-heading-rad',type=float,default=0.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((ROOT/'robot.json').read_text());model=TiagoKinematics()
    base=np.eye(4);q=model.home.copy();previous=np.zeros(10)
    guard=CameraController(spec['joints'],spec['wheel_radius_m'],spec['wheel_separation_m'])
    base[:3,:3]=Rotation.from_euler('z',a.initial_heading_rad).as_matrix()
    initial,_=model.pose_and_jacobian(base,q)
    times=np.linspace(0.,16.,321);poses=[]
    for t in times:
        s=2*np.pi*t/16.;T=initial.copy()
        T[:3,3]+=[.22*(1-np.cos(s)),.10*np.sin(s),.06*np.sin(s)]
        T[:3,:3]=Rotation.from_rotvec([.08*np.sin(s),.10*np.sin(s),.15*np.sin(s)]).as_matrix()@initial[:3,:3]
        poses.append(T)
    poses=np.asarray(poses);rotations=Rotation.from_matrix(poses[:,:3,:3]);slerp=Slerp(times,rotations)
    client=AsyncGiskardClient(ROOT/'wbc/results/collision_geometry_01.json',a.output/'worker.log',backend='giskard-executor')
    rows=[]
    try:
        client.prewarm();start=time.monotonic()
        for i in range(1600):
            stamp=i*.01;future=(i+1)*.01
            reference=np.eye(4);reference[:3,3]=[np.interp(future,times,poses[:,k,3]) for k in range(3)]
            reference[:3,:3]=slerp(future).as_matrix()
            k=min(np.searchsorted(times,future,side='right')-1,len(times)-2)
            ff=np.r_[(poses[k+1,:3,3]-poses[k,:3,3])/(times[k+1]-times[k]),
                (rotations[k+1]*rotations[k].inv()).as_rotvec()/(times[k+1]-times[k])]
            camera,J=model.pose_and_jacobian(base,q);body_poses,body_js=source_state(spec,q)
            for name in body_poses:
                body_poses[name]=base@body_poses[name]
                body_js[name][:3]=base[:3,:3]@body_js[name][:3]
                body_js[name][3:]=base[:3,:3]@body_js[name][3:]
            error=np.r_[reference[:3,3]-camera[:3,3],Rotation.from_matrix(reference[:3,:3]@camera[:3,:3].T).as_rotvec()]
            request=dict(q=q,previous=previous,measured_rates=previous,poses=body_poses,body_jacobians=body_js,
                measured_camera=camera,mode='camera',reference=reference,feedforward=ff,jacobian=J,error=error,
                sample_s=stamp,phase='cpu_source_camera_fixture')
            command,status=client.command(request)
            command=guard.guard(q,command)
            guard.previous=command.copy()
            base,q=model.integrate(base,q,command,.01);previous=command
            camera,_=model.pose_and_jacobian(base,q)
            rows.append(dict(stamp_s=future,position_error_m=float(np.linalg.norm(reference[:3,3]-camera[:3,3])),
                orientation_error_rad=float(Rotation.from_matrix(reference[:3,:3]@camera[:3,:3].T).magnitude()),
                command=command.tolist(),native=status,q=q.tolist(),T_world_base=base.tolist()))
            remaining=start+future-time.monotonic()
            if remaining>0:time.sleep(remaining)
        elapsed=time.monotonic()-start
        result=dict(schema='tiago_full_executor_camera_cpu_fixture/v1',execution_kind='source-FK kinematic fixture; no measured PhysX or physical task-ingress claim',
            physical_tracking_verified=False,representation_ingress_verified=False,reference_interpolation='piecewise-linear translation + shortest-SO3; authored knots unchanged',
            initial_heading_rad=a.initial_heading_rad,
            dt_s=.01,simulated_seconds=16.,wall_seconds=elapsed,timing_pass=elapsed<=16.*1.05,
            position_rmse_m=float(np.sqrt(np.mean([r['position_error_m']**2 for r in rows]))),
            orientation_rmse_rad=float(np.sqrt(np.mean([r['orientation_error_rad']**2 for r in rows]))),
            native_fresh_fraction=sum(r['native']['fresh_native_command'] for r in rows)/len(rows),
            worker_initialization=client.initialization,events=client.events)
        result['passed']=bool(result['timing_pass'] and result['position_rmse_m']<.03
            and result['orientation_rmse_rad']<.08 and result['native_fresh_fraction']>.9 and not client.events)
        (a.output/'trace.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='worker_initialization'},indent=2))
    finally:client.close()


if __name__=='__main__':main()
