#!/usr/bin/env python3
"""Cross-environment genuine framework worker benchmark; CPU fixture, not PhysX."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,default=ROOT/'wbc/results/collision_geometry_01.json')
    parser.add_argument('--backend',choices=['giskard-core','giskard-executor'],default='giskard-executor')
    parser.add_argument('--include-stop',action='store_true',help='One-second forward step followed by one-second zero request.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((ROOT/'robot.json').read_text())
    client=AsyncGiskardClient(args.metadata,args.output/'worker.log',backend=args.backend)
    started=time.monotonic();client.prewarm();warm=time.monotonic()-started
    model=TiagoKinematics();q=model.home.copy();base=np.eye(4);previous=np.zeros(10)
    start=time.monotonic();rows=[]
    try:
        steps=200 if args.include_stop else 100
        for i in range(steps):
            poses,jacobians=source_state(spec,q)
            for name in poses:
                poses[name]=base@poses[name]
                jacobians[name][:3]=base[:3,:3]@jacobians[name][:3]
                jacobians[name][3:]=base[:3,:3]@jacobians[name][3:]
            request=dict(q=q,previous=previous,measured_rates=previous,poses=poses,body_jacobians=jacobians,
                mode='actuator',desired_rates=np.r_[.10 if i<100 else 0.,0.,np.zeros(8)],sample_s=i*.01,
                phase='cpu_fixture_move' if i<100 else 'cpu_fixture_stop')
            command,status=client.command(request)
            # Same issued-command acceleration bounds, then kinematic fixture
            # integration; these states are never labelled measured physical FK.
            command=np.clip(command,previous-np.r_[.45,.65,np.ones(8)]*.01,previous+np.r_[.45,.65,np.ones(8)]*.01)
            base,q=model.integrate(base,q,command,.01);previous=command
            rows.append(dict(t_s=i*.01,command=command.tolist(),native=status))
            remaining=start+(i+1)*.01-time.monotonic()
            if remaining>0:time.sleep(remaining)
        wall=time.monotonic()-start
        fresh=sum(r['native']['fresh_native_command'] for r in rows)/len(rows)
        latency=[r['native']['worker_wall_s'] for r in rows if r['native']['worker_wall_s'] is not None]
        stop_pass=bool(np.max(np.abs(rows[-1]['command']))<1e-6) if args.include_stop else None
        command_acceleration=np.max(np.abs(np.diff(np.vstack([np.zeros(10),[r['command'] for r in rows]]),axis=0)),axis=0)/.01
        acceleration_pass=bool(np.all(command_acceleration<=np.r_[.45,.65,np.ones(8)]+1e-7))
        result=dict(schema='tiago_giskard_cross_env_bridge/v1',backend=args.backend,
            execution_kind='CPU source-URDF kinematic fixture, no Isaac/PhysX measurement',client_python=sys.version,
            client_numpy=np.__version__,worker_initialization=client.initialization,prewarm_wall_s=warm,
            logical_dt_s=.01,steps=steps,wall_s=wall,requested_pace=1.,timing_pass=wall<=steps*.01*1.05,
            final_zero_command_pass=stop_pass,maximum_issued_acceleration=command_acceleration.tolist(),issued_acceleration_pass=acceleration_pass,
            native_fresh_fraction=fresh,worker_median_wall_s=float(np.median(latency)) if latency else None,
            worker_p99_wall_s=float(np.quantile(latency,.99)) if latency else None,
            events=client.events,passed=bool(fresh>.9 and wall<=steps*.01*1.05 and not client.events and acceleration_pass and stop_pass is not False),
            physical_tracking_verified=False,representation_ingress_verified=False)
        (args.output/'trace.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='worker_initialization'},indent=2))
        return 0 if result['passed'] else 2
    finally:client.close()


if __name__=='__main__':raise SystemExit(main())
