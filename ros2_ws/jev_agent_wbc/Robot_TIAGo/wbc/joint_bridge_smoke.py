#!/usr/bin/env python3
"""Paced CPU check of the frozen physical independent-joint fixture and native tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'wbc')]
from async_client import AsyncGiskardClient
from source_state import source_state
from kinematics import TiagoKinematics
from physical_control import CameraController
from giskard_runtime import independent_joint_fixture


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((ROOT/'robot.json').read_text());fixture=independent_joint_fixture(spec['joints'])
    (a.output/'authored_reference.json').write_text(json.dumps(fixture,indent=2)+'\n')
    model=TiagoKinematics();q=model.home.copy();base=np.eye(4);previous=np.zeros(10)
    guard=CameraController(spec['joints'],spec['wheel_radius_m'],spec['wheel_separation_m'])
    client=AsyncGiskardClient(ROOT/'wbc/results/collision_geometry_01.json',a.output/'worker.log',backend='giskard-executor')
    rows=[]
    try:
        client.prewarm();start=time.monotonic()
        for i in range(1800):
            motion=i<800
            home_return=800<=i<1300
            authored=fixture['samples'][i+1] if motion else dict(position=model.home,velocity=np.zeros(8))
            reference=np.asarray(authored['position'])
            desired=np.r_[0.,0.,np.clip(np.asarray(authored['velocity'])+2.*(reference-q),-guard.speed[2:],guard.speed[2:])]
            if not motion and not home_return:desired[:]=0.
            poses,js=source_state(spec,q)
            for name in poses:
                poses[name]=base@poses[name]
                js[name][:3]=base[:3,:3]@js[name][:3];js[name][3:]=base[:3,:3]@js[name][3:]
            request=dict(sample_s=i*.01,phase='cpu_independent_arm' if motion else ('cpu_fixed_home_return' if home_return else 'cpu_zero_command_stop'),
                mode='actuator',desired_rates=desired,q=q,previous=previous,measured_rates=previous,
                poses=poses,body_jacobians=js)
            command,status=client.command(request)
            command=guard.guard(q,command);guard.previous=command.copy()
            base,q=model.integrate(base,q,command,.01);previous=command
            rows.append(dict(stamp_s=(i+1)*.01,q=q.tolist(),command=command.tolist(),reference=reference.tolist(),
                base_position=base[:3,3].tolist(),native=status))
            remaining=start+(i+1)*.01-time.monotonic()
            if remaining>0:time.sleep(remaining)
        wall=time.monotonic()-start
        error=np.abs(q-model.home);tolerance=np.r_[.002,np.full(7,.01)]
        span=np.ptp([x['q'] for x in rows[:800]],axis=0)
        base_span=float(np.linalg.norm(np.ptp([x['base_position'] for x in rows],axis=0)))
        fresh=sum(x['native']['fresh_native_command'] for x in rows)/len(rows)
        passed=bool(np.all(error<tolerance) and np.all(span>np.asarray(fixture['amplitude'])*.5)
            and base_span<.03 and wall<=18.*1.05 and fresh>.9 and not client.events)
        summary=dict(schema='tiago_native_independent_joint_cpu_fixture/v1',
            execution_kind='CPU source-FK fixture; not measured PhysX or14DOFstop proof',
            passed=passed,physical_verified=False,simulated_seconds=18.,wall_seconds=wall,timing_pass=wall<=18.*1.05,
            authored_motion_s=8.,home_return_s=5.,zero_command_stop_s=5.,
            home_return_error=error.tolist(),home_return_tolerance=tolerance.tolist(),joint_span=span.tolist(),
            base_span_m=base_span,native_fresh_fraction=fresh,events=client.events,worker_initialization=client.initialization,
            source_files_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                [Path(__file__),ROOT/'giskard_runtime.py',ROOT/'wbc/full_framework.py',ROOT/'wbc/physical_control.py']})
        (a.output/'trace.json').write_text(json.dumps(rows,indent=2)+'\n')
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='worker_initialization'},indent=2))
        return 0 if passed else 2
    finally:client.close()


if __name__=='__main__':raise SystemExit(main())
