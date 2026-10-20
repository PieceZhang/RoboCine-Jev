#!/usr/bin/env python3
"""Attribute terminal Pink proposals by CPU-only soft-task ablations.

All original hard constraints are retained. Ablations are explanations, never
runtime proposals or accepted camera trajectories.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod);return mod


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recorded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);source=a.recorded/'sources'
    backend=load('objective_frozen_backend',source/'g1_backend.py');collision=load('objective_frozen_collision',source/'pink_collision.py')
    rows=[json.loads(l) for l in (a.recorded/'measurements.jsonl').open()]
    selected=[r for r in rows if r['case']=='camera_shot' and r['phase']=='stop']
    selected=[r for r in selected if r['stamp_s']>=selected[-1]['stamp_s']-.5-1e-8]
    names=selected[0]['joint_names'];legs=[n for n in names if any(t in n for t in ('hip','knee','ankle'))]
    import yaml
    io=yaml.safe_load((source/'height_policy_io.yaml').read_text())['articulations']['robot']
    home=dict(zip(io['joint_names'],io['default_joint_pos']))
    camera=json.loads((a.recorded/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    solver=backend.PinkUpperBody(legs,names,camera,source/'g1_29dof.urdf',posture_home=home,
        posture_cost=.03,waist_posture_cost=.15,damping_cost=.1)
    solver.collision_helper=collision
    solver.collision_offsets=json.loads((a.recorded/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    solver.collision_geometry=collision.geometry_model(solver.model,json.loads((a.recorded/'fine_collision_geometry.json').read_text()),solver.collision_offsets)
    original=solver.pink.solve_ik;variant='full'
    def instrumented(configuration,tasks,dt,**kwargs):
        if variant=='without_soft_posture':tasks=[t for t in tasks if type(t).__name__!='PostureTask']
        elif variant=='without_camera_frame':tasks=[t for t in tasks if type(t).__name__!='FrameTask']
        return original(configuration,tasks,dt,**kwargs)
    solver.pink.solve_ik=instrumented;result=[]
    try:
        for variant in ('full','without_soft_posture','without_camera_frame'):
            start=time.monotonic()
            for i,row in enumerate(selected):
                wait=start+i*.02-time.monotonic()
                if wait>0:time.sleep(wait)
                solver.previous={n:row['pink_upper_velocity'].get(n,0.) for n in solver.previous}
                measured=dict(zip(names,row['joint_positions']))
                velocity=solver.velocity(measured,np.asarray(row['T_world_base']),np.asarray(row['T_world_reference_optical']),.02,
                    posture_reference=row['pink_parking_posture']['targets'],**row['pink_requested_soft_costs'])
                result.append(dict(variant=variant,stamp_s=row['stamp_s'],velocity=velocity,
                    norm=float(np.linalg.norm(list(velocity.values()))),
                    preference_minus_actual={n:row['pink_parking_posture']['targets'][n]-measured[n] for n in velocity}))
    finally:solver.pink.solve_ik=original
    (a.output/'trace.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    summary=dict(schema='independent_g1_terminal_soft_objectives/v1',
        scope='Same measured terminal states, world camera target, previous command, and all native hard limits/barriers. Only named soft objectives are omitted as diagnostic CPU ablations; no motion is executed and omitted-camera case is not a valid task.',
        physical_execution=False,hard_constraints_unchanged=True,pace=1,case='camera_shot',samples_per_variant=len(selected),
        summaries=[dict(variant=v,mean_velocity_l2=float(np.mean([r['norm'] for r in result if r['variant']==v])),
            mean_owned_velocity={n:float(np.mean([r['velocity'][n] for r in result if r['variant']==v])) for n in solver.previous})
            for v in ('full','without_soft_posture','without_camera_frame')],
        evidence_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [source/'g1_backend.py',source/'pink_collision.py',a.recorded/'measurements.jsonl',Path(__file__)]})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary['summaries'],indent=2))


if __name__=='__main__':main()
