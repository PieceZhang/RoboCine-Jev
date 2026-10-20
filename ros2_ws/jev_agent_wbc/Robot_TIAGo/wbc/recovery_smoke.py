#!/usr/bin/env python3
"""Actual native worker failure/recovery, with a continuously paced CPU fixture."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'wbc')]
from async_client import AsyncGiskardClient
from source_state import source_state
from kinematics import TiagoKinematics
from physical_control import CameraController


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((ROOT/'robot.json').read_text());model=TiagoKinematics();q=model.home.copy();base=np.eye(4)
    guard=CameraController(spec['joints'],spec['wheel_radius_m'],spec['wheel_separation_m']);previous=np.zeros(10)
    client=AsyncGiskardClient(ROOT/'wbc/results/collision_geometry_01.json',a.output/'worker.log',backend='giskard-executor')
    rows=[];error=None;injected_at=None
    try:
        client.prewarm();start=time.monotonic()
        for i in range(1400):
            if i==50:
                client.process.terminate();injected_at=i*.01
            poses,js=source_state(spec,q)
            for name in poses:
                poses[name]=base@poses[name];js[name][:3]=base[:3,:3]@js[name][:3];js[name][3:]=base[:3,:3]@js[name][3:]
            request=dict(sample_s=i*.01,phase='cpu_recovery_move' if i<100 else 'cpu_recovery_stop',q=q,previous=previous,
                measured_rates=previous,poses=poses,body_jacobians=js,mode='actuator',desired_rates=np.r_[.1 if i<100 else 0.,0.,np.zeros(8)])
            command,status=client.command(request);command=guard.guard(q,command);guard.previous=command.copy()
            base,q=model.integrate(base,q,command,.01);previous=command
            rows.append(dict(stamp_s=(i+1)*.01,command=command.tolist(),native=status))
            remaining=start+(i+1)*.01-time.monotonic()
            if remaining>0:time.sleep(remaining)
        wall=time.monotonic()-start
    except BaseException:
        error=traceback.format_exc();wall=time.monotonic()-start if rows else None
    finally:client.close()
    fresh_after=[r for r in rows if r['stamp_s']>1. and r['native']['fresh_native_command']]
    stale=[r for r in rows if .5<r['stamp_s']<7. and not r['native']['fresh_native_command']]
    acceleration=np.max(np.abs(np.diff(np.vstack([np.zeros(10),[r['command'] for r in rows]]),axis=0)),axis=0)/.01 if rows else np.full(10,np.inf)
    passed=bool(error is None and len(rows)==1400 and fresh_after and stale and client.restarts==1
        and np.all(acceleration<=guard.accel+1e-7) and np.max(np.abs(rows[-1]['command']))<1e-6 and wall<=14.*1.05)
    result=dict(schema='tiago_actual_native_worker_recovery_cpu/v1',passed=passed,
        execution_kind='real isolated World/Executor process termination/restart with CPU source-FK fixture; no physical failure-injection claim',
        physical_failure_recovery_verified=False,steps=len(rows),injected_worker_termination_s=injected_at,
        simulated_seconds=len(rows)*.01,wall_seconds=wall,requested_pace=1.,restarts=client.restarts,
        stale_braking_samples=len(stale),first_fresh_after_failure_s=fresh_after[0]['stamp_s'] if fresh_after else None,
        maximum_issued_acceleration=acceleration.tolist(),events=client.events,error=error)
    (a.output/'trace.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
