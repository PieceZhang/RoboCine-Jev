#!/usr/bin/env python3
"""Replay measured states into two genuine upstream configurations, CPU only.

Collision-owner poses/Jacobians here come from source FK at the recorded state;
they are fixtures, not newly measured physical evidence or a counterfactual run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'wbc')]
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['XDG_CACHE_HOME']=str(ROOT/'wbc/cache')
os.environ['MPLCONFIGDIR']=str(ROOT/'wbc/cache/matplotlib')
import numpy as np
from full_framework import FullGiskardExecutor
from source_state import source_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['velocity','jerk_fd_diagnostic','jerk_zero_estimate','jerk_command_estimate','jerk_issued_state','jerk_issued_velocity'],required=True)
    parser.add_argument('--steps',type=int,default=60)
    parser.add_argument('--braking-time-s',type=float,default=.3)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((ROOT/'robot.json').read_text())
    config=json.loads((args.recording/'configuration.json').read_text())
    names=config['dof_names'];indices=[names.index(j['name']) for j in spec['joints']]
    metadata=json.loads((args.recording/'collision_geometry_prepared.json').read_text())
    solver=FullGiskardExecutor(spec,metadata,args.output/'native_meshes',derivative_mode=args.mode,braking_time_s=args.braking_time_s)
    recorded=[json.loads(line) for line in (args.recording/'samples.jsonl').read_text().splitlines()][:args.steps]
    started=time.monotonic();previous=np.zeros(10);rows=[]
    for i,row in enumerate(recorded):
        q=np.asarray(row['q'])[indices];base=np.asarray(row['T_world_base'])
        poses,jacobians=source_state(spec,q)
        for name in poses:
            poses[name]=base@poses[name]
            jacobians[name][:3]=base[:3,:3]@jacobians[name][:3]
            jacobians[name][3:]=base[:3,:3]@jacobians[name][3:]
        measured=np.r_[base[:3,0]@np.asarray(row['base_linear_velocity']),
            base[:3,2]@np.asarray(row['base_angular_velocity']),np.asarray(row['qd'])[indices]]
        request=dict(q=q,previous=previous,measured_rates=measured,poses=poses,body_jacobians=jacobians,
            mode='actuator',desired_rates=np.zeros(10),sample_s=row['t_s'],phase='recorded_zero_request_fixture')
        result=solver.solve(request)
        rows.append(dict(stamp_s=row['t_s'],measured_rates=measured.tolist(),
            upstream_proposal=np.asarray(result['upstream_proposal']).tolist(),safe_command=np.asarray(result['command']).tolist(),
            framework_tick_wall_s=result['framework_tick_wall_s'],projection_norm=result['projection_norm'],collision=result['collision']))
        previous=np.r_[row['command_vw'],row['command_joint_speed']]
        remaining=started+(i+1)*.01-time.monotonic()
        if remaining>0:time.sleep(remaining)
    wall=time.monotonic()-started
    summary=dict(schema='tiago_native_derivative_replay/v1',mode=args.mode,steps=len(rows),
        execution_kind='CPU genuine World/Executor measured-state replay; source-FK collision fixture; no counterfactual physical validation',
        original_recording=str(args.recording),original_trace_sha256=hashlib.sha256((args.recording/'samples.jsonl').read_bytes()).hexdigest(),
        model_manifest=solver.model_manifest,wall_s=wall,logical_s=.01*len(rows),requested_pace=1.,
        proposal_peak_abs=np.max(np.abs([r['upstream_proposal'] for r in rows]),axis=0).tolist(),
        final_proposal=rows[-1]['upstream_proposal'],zero_request=np.zeros(10).tolist(),
        physical_tracking_verified=False)
    (args.output/'trace.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='model_manifest'},indent=2))


if __name__=='__main__':main()
