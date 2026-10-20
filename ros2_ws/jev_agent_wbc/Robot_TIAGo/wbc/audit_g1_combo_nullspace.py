#!/usr/bin/env python3
"""Independent frame-Jacobian null/range decomposition of genuine Pink output."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grid',type=Path,required=True);p.add_argument('--recorded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.grid/'sources'
    backend=load('null_frozen_backend',source/'g1_backend.py');collision=load('null_frozen_collision',source/'pink_collision.py')
    rows={round(r['stamp_s'],8):r for r in map(json.loads,(a.recorded/'measurements.jsonl').open())}
    trace=[r for r in map(json.loads,(a.grid/'trace.jsonl').open()) if r['mode']=='measured_q_recursive_previous' and r['posture_cost']==.03 and r['damping_cost'] in (.2,.5)]
    first=next(iter(rows.values()));names=first['joint_names'];legs=[n for n in names if any(t in n for t in ('hip','knee','ankle'))]
    camera=json.loads((a.recorded/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    solver=backend.PinkUpperBody(legs,names,camera,source/'g1_29dof.urdf')
    solver.collision_helper=collision;solver.collision_offsets=json.loads((a.recorded/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    solver.collision_geometry=collision.geometry_model(solver.model,json.loads((a.recorded/'fine_collision_geometry.json').read_text()),solver.collision_offsets)
    from pink.tasks import FrameTask
    import yaml
    io=yaml.safe_load((source/'height_policy_io.yaml').read_text())['articulations']['robot'];home=dict(zip(io['joint_names'],io['default_joint_pos']))
    result=[];start=time.monotonic()
    for i,item in enumerate(trace):
        delay=start+i*.02-time.monotonic()
        if delay>0:time.sleep(delay)
        row=rows[round(item['stamp_s'],8)];measured=dict(zip(names,row['joint_positions']))
        configuration,upper=solver.configuration(measured)
        frame=FrameTask('cine_camera_optical',position_cost=1.,orientation_cost=.3,lm_damping=1e-3,gain=.1)
        frame.set_target(solver.pin.SE3(np.linalg.inv(np.array(row['T_world_base']))@np.array(row['T_world_reference_optical'])))
        J=frame.compute_jacobian(configuration)
        _,singular,V=np.linalg.svd(J,full_matrices=True);rank=int(np.count_nonzero(singular>1e-8));N=V[rank:].T
        velocity=np.array([item['native_velocity'][n] for n in upper]);vnull=N@(N.T@velocity);vrange=velocity-vnull
        preferred=np.array([row['pink_parking_posture']['targets'][n] for n in upper]);preferred[upper.index('waist_yaw_joint')]=home['waist_yaw_joint']
        e_null=N.T@(preferred-configuration.q)
        C=np.diag([.15**2 if n=='waist_yaw_joint' else .03**2 for n in upper]);eigen=np.linalg.eigvalsh(N.T@C@N)
        damping=item['damping_cost'];tau=.02/.1*(1+damping*damping/eigen)
        wrist=upper.index('right_wrist_pitch_joint')
        result.append(dict(case=item['case'],stamp_s=item['stamp_s'],damping_cost=damping,
            frame_rank=rank,frame_nullity=N.shape[1],frame_smallest_singular_value=float(singular[-1]),
            native_norm=float(np.linalg.norm(velocity)),null_norm=float(np.linalg.norm(vnull)),range_norm=float(np.linalg.norm(vrange)),
            null_velocity_energy_fraction=float(vnull@vnull/(velocity@velocity)),
            right_wrist_pitch_native_rad_s=float(velocity[wrist]),right_wrist_pitch_null_rad_s=float(vnull[wrist]),right_wrist_pitch_range_rad_s=float(vrange[wrist]),
            null_preference_error_norm_rad=float(np.linalg.norm(e_null)),
            null_direction_dot_preference=float((N.T@velocity)@e_null),
            ideal_constant_nullspace_posture_decay_tau_range_s=[float(tau.min()),float(tau.max())],
            jacobian_null_residual=float(np.linalg.norm(J@vnull))))
    summaries=[]
    for case in ('left','camera_shot','yaw_left','yaw_right'):
        for damping in (.2,.5):
            selected=[r for r in result if r['case']==case and r['damping_cost']==damping]
            summaries.append(dict(case=case,damping_cost=damping,
                mean_null_velocity_energy_fraction=float(np.mean([r['null_velocity_energy_fraction'] for r in selected])),
                mean_null_norm=float(np.mean([r['null_norm'] for r in selected])),mean_range_norm=float(np.mean([r['range_norm'] for r in selected])),
                mean_wrist_native_rad_s=float(np.mean([r['right_wrist_pitch_native_rad_s'] for r in selected])),
                mean_wrist_null_rad_s=float(np.mean([r['right_wrist_pitch_null_rad_s'] for r in selected])),
                mean_wrist_range_rad_s=float(np.mean([r['right_wrist_pitch_range_rad_s'] for r in selected])),
                positive_null_preference_alignment_samples=sum(r['null_direction_dot_preference']>0 for r in selected),
                maximum_jacobian_null_residual=max(r['jacobian_null_residual'] for r in selected)))
    report=dict(schema='independent_g1_combo_nullspace/v1',physical_execution=False,pace=1,
        scope='SVD of actual genuine Pink FrameTask Jacobian at each recorded q/base and original target. Orthogonal row/null-space decomposition describes commanded joint velocity. Constant-nullspace decay estimates ignore Frame LM, changing geometry, active constraints, servo/contact and base motion; they are not physical settling-time estimates.',
        nullspace_formula='tau=(dt/posture_gain)*(1+DampingCost^2/lambda(N.T*diag(PostureCost^2)*N)); dt=.02, posture_gain=.1. Equal .03 cost would give9.089s atD=.2 and55.756s atD=.5.',
        summaries=summaries,samples=result,
        evidence_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.grid/'summary.json',a.grid/'trace.jsonl',a.recorded/'measurements.jsonl',Path(__file__)]})
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
