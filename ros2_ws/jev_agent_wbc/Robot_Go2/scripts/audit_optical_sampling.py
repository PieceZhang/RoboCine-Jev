#!/usr/bin/env python3
"""CPU replay of the executed soft-pose coordinator with exact optical sampling."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE))
from jev_agent_wbc.wbc_protocol import TimedTask,optical_camera
from jev_agent_wbc.wbc_physical import rigid


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=PACKAGE/'Robot_Go2';run=root/'results/adaptive_backward_softpose_01';frozen=run/'sources'
    rows=[json.loads(line) for line in (run/'measurements.jsonl').open()]
    task=TimedTask.from_dict(json.loads((run/'timed_tasks.json').read_text())[0])
    sampler=load('audited_optical_sampler',root/'optical_sampler.py').OpticalSampler(task)
    backend=load('executed_sampling_backend',frozen/'go2_backend.py');backend.UPSTREAM_MDP=frozen/'go2_upstream'
    module=load('executed_soft_pose_coordinator',frozen/'camera_coordinator.py')
    extrinsic=np.array(json.loads((run/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    frames={round(r['fine_collision_evidence']['stamp_s']*50):r['fine_collision_evidence']['T_world_collision_bodies']
        for r in rows if r.get('fine_collision_evidence',{}).get('T_world_collision_bodies')}
    baseline=lambda t:rigid(optical_camera(task.trajectory.sample_with_twist(t)[0]))
    def replay(sample):
        controller=module.CameraCoordinator(extrinsic,backend=backend,response_profile='measured_backward_v1')
        output=[];wall=[]
        for i,row in enumerate(rows):
            stamp=row['stamp_s']-.02
            if 'camera_support' in row and i%5==0 and round(stamp*50) in frames:
                state=frames[round(stamp*50)];support=row['camera_support']
                future=np.array(row['requested_body_velocity'])*np.where(np.array(row['requested_body_velocity'])<0,[1/.55,1.,1.],1.)
                preference=support['native_posture_preference'];started=time.perf_counter()
                result=controller.command(stamp,lambda t:sample(min(t,task.trajectory.end_s)),
                    np.array(state['link0']),np.array(state['base']),support['preferred_base_command'],
                    rows[i-1]['measured_body_velocity'],max_lookahead_s=max(0.,task.trajectory.end_s-stamp),
                    future_policy_velocity=np.clip(future,-.2,.2),previous_applied=rows[i-1]['applied_policy_velocity'],
                    native_posture_x_m=preference['x_link0_m'],posture_weight=preference['weight'])
                wall.append(time.perf_counter()-started);output.append(result)
            controller.observe_applied_input(stamp,row['applied_policy_velocity'])
        return output,wall
    old,old_wall=replay(baseline);new,new_wall=replay(sampler.sample)
    matrix_error=max(float(np.max(abs(baseline(float(t))-sampler.sample(float(t)))))
        for t in np.linspace(task.trajectory.start_s,task.trajectory.end_s,501))
    command_error=max(float(np.max(abs(a['base_command']-b['base_command']))) for a,b in zip(old,new))
    margin_error=max(abs(a['predicted_min_domain_margin']-b['predicted_min_domain_margin']) for a,b in zip(old,new))
    statuses=all((a['feasible'],a['camera_supported'],a['reason'])==(b['feasible'],b['camera_supported'],b['reason']) for a,b in zip(old,new))
    assert matrix_error<1e-12 and command_error<1e-12 and margin_error<1e-12 and statuses
    result=dict(schema='go2_optical_sampling_replay/v1',physical_execution=False,pace=1,
        scope='Identical preserved measured collider/body snapshots, original world task, source coordinator and applied input history; only pose sampling differs.',
        calls=len(old),optical_matrix_max_abs_error=matrix_error,base_command_max_abs_error=command_error,
        domain_margin_max_abs_error=margin_error,statuses_equal=statuses,sampler=sampler.manifest(),
        wall_ms={name:dict(median=float(np.median(values)*1000),p95=float(np.quantile(values,.95)*1000))
            for name,values in [('public_pose_and_twist',old_wall),('optical_pose_only',new_wall)]},
        source_sha256={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),root/'optical_sampler.py',PACKAGE/'Robot_G1/adaptive_runner.py',
                run/'measurements.jsonl',run/'timed_tasks.json',frozen/'camera_coordinator.py',frozen/'go2_backend.py']})
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
