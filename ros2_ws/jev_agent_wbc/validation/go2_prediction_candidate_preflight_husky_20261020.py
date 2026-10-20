#!/usr/bin/env python3
"""Freeze and replay an optional input-prediction candidate; no Isaac/GPU."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_camera6d_softpose_01'
OUTPUT=ROOT/'Robot_Go2/results/prediction_candidate_preparation_01'
sys.path.insert(0,str(ROOT))
from jev_agent_wbc.wbc_protocol import TimedTask


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.monotonic();OUTPUT.mkdir(exist_ok=False)
    frozen=OUTPUT/'sources';frozen.mkdir()
    files=[ROOT/'Robot_Go2/camera_coordinator.py',ROOT/'Robot_Go2/adaptive_runner.py',
        ROOT/'Robot_Go2/optical_sampler.py',ROOT/'Robot_Go2/wbc_backend.py',
        ROOT/'Robot_G1/velocity_servo.py',ROOT/'Robot_G1/legged_collision.py',
        ROOT/'Robot_G1/contact_materials.py',ROOT/'config/robot_sensor_mounts.json',
        ROOT/'test/test_wbc_go2_prediction_profile.py',Path(__file__)]
    for p in files:shutil.copy2(p,frozen/p.name)
    shutil.copytree(ROOT/'jev_agent_wbc',frozen/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('camera_coordinator.py','adaptive_runner.py'):
        old=(RUN/'sources'/name).read_text().splitlines(keepends=True)
        new=(ROOT/'Robot_Go2'/name).read_text().splitlines(keepends=True)
        (OUTPUT/(name+'.diff')).write_text(''.join(difflib.unified_diff(old,new,fromfile='previous/'+name,tofile='candidate/'+name)))
    backend=load('preflight_backend',RUN/'sources/go2_backend.py');backend.UPSTREAM_MDP=RUN/'sources/go2_upstream'
    module=load('preflight_coordinator',frozen/'camera_coordinator.py')
    sampler_module=load('preflight_sampler',frozen/'optical_sampler.py')
    task=TimedTask.from_dict(json.loads((RUN/'timed_tasks.json').read_text())[0]);sampler=sampler_module.OpticalSampler(task)
    rows=[json.loads(line) for line in (RUN/'measurements.jsonl').read_text().splitlines()]
    extrinsic=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    coordinator=module.CameraCoordinator(extrinsic,backend=backend,response_profile='measured_backward_v1',
        prediction_profile='persistent_bias_12_v1')
    checked=[];captured={};code=type(coordinator).command.__code__
    def trace(frame,event,arg):
        if frame.f_code is code and event=='return':captured.clear();captured.update(frame.f_locals)
        return trace
    for i,row in enumerate(rows):
        stamp=row['stamp_s']-.02
        if row['case']!='settle':
            previous=rows[i-1];body=np.asarray(previous['T_world_base']);link=body.copy()
            link[:3,3]+=body[:3,:3]@np.array([0.,0.,.07])
            preferred=np.asarray(row['camera_support']['preferred_base_command'])
            future=np.asarray(row['requested_body_velocity'])*np.where(np.asarray(row['requested_body_velocity'])<0,[1/.55,1.,1.],1.)
            future=np.clip(future,-.2,.2);previous_input=np.asarray(previous['applied_policy_velocity'])
            sys.settrace(trace)
            result=coordinator.command(stamp,lambda t:sampler.sample(min(float(t),task.trajectory.end_s)),link,body,
                preferred,np.asarray(previous['measured_body_velocity']),max_lookahead_s=max(0.,task.trajectory.end_s-stamp),
                future_policy_velocity=future,previous_applied=previous_input,command_dt=.02,
                native_posture_x_m=.58,posture_weight=module.operating_posture_weight(stamp,task.trajectory.start_s,task.trajectory.end_s))
            sys.settrace(None)
            command=np.asarray(result['base_command']);margin=float(captured['margins'](command).min())
            checked.append(dict(row=i,stamp_s=stamp,original_feasible=row['camera_support']['feasible'],
                candidate_feasible=result['feasible'],candidate_reason=result['reason'],
                nonlinear_min_margin=margin,command=command.tolist(),
                command_caps_pass=bool(np.all(abs(command)<=.2+1e-9)),
                command_slew_pass=bool(np.all(abs(command-previous_input)<=np.array([.5,.5,.8])*.02+1e-9)),
                original_reference_changed=result['camera_reference_changed'],
                reserve_ramp_horizon_s=result['reserve_ramp_horizon_s'],
                prediction_intervals_s=result['prediction_intervals_s']))
        coordinator.observe_applied_input(stamp,np.asarray(row['applied_policy_velocity']))
    false=[r for r in checked if r['candidate_feasible'] is False]
    invalid=[r for r in checked if r['candidate_feasible'] is not False and r['nonlinear_min_margin']< -1e-7]
    caps=[r for r in checked if not (r['command_caps_pass'] and r['command_slew_pass'])]
    report=dict(schema='go2_prediction_candidate_cpu/v1',scope='Frozen actual-state replay only. Newly proposed inputs are never integrated into later preserved states. No CPU closed-loop or physical success claim.',
        samples=len(checked),original_false_count=sum(r['original_feasible'] is False for r in checked),
        candidate_false_count=len(false),recovered_original_false=sum(r['original_feasible'] is False and r['candidate_feasible'] is not False for r in checked),
        new_false_count=sum(r['original_feasible'] is not False and r['candidate_feasible'] is False for r in checked),
        accepted_nonlinear_violation_count=len(invalid),caps_or_slew_violation_count=len(caps),
        original_task_file_sha256=sha(RUN/'timed_tasks.json'),cpu_wall_s=time.monotonic()-started,
        cpu_gate_pass=not invalid and not caps and all(not r['original_reference_changed'] for r in checked))
    (OUTPUT/'cpu_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUTPUT/'cpu_rows.json').write_text(json.dumps(checked,indent=2)+'\n')
    command=['Robot_Go2/run_adaptive_wbc.sh','--cases','camera_6d','--move-seconds','16','--stop-seconds','5',
        '--fine-collision','--go2-response-profile','measured_backward_v1','--go2-native-posture','x58_all_v1',
        '--go2-prediction-profile','persistent_bias_12_v1','--camera-excursion','nontrivial',
        '--ground-material','source_multiply','--parking-anchor-reset','--rgb',
        '--output','Robot_Go2/results/adaptive_camera6d_prediction_01']
    declaration=dict(schema='go2_prediction_candidate_declaration/v1',pace=1,physics_hz=200,control_hz=50,
        physics_substeps_per_action=4,settle_s=2,original_task_duration_s=16,stop_s=5,total_sim_s=23,
        command=command,prediction_profile='persistent_bias_12_v1',native_posture_profile='x58_all_v1',
        prediction_nodes_full_s=[0,.2,.4,.6,.8,1.,1.2],reserve_ramp_s=.8,
        position_reserve_m=[.035,.035,.015],orientation_reserve_rad=.015,
        native_input_caps=[.2,.2,.2],native_input_slew=[.5,.5,.8],
        changes=['Optional persistent candidate-minus-preferred bias in supplied bounded future plan even with no posture weight.',
            '1.2s horizon retains original .8s five nodes and adds1.0/1.2; original .8s reserve ramp.',
            'Optional x58_all_v1 subordinate preference includes zero preferred velocity camera6d.'],
        unchanged=['Original WORLD Task linear/Slerp knots/absolute timestamps/tolerances/6D optical transform.',
            'All18 learned actions, original native motor caps/4substep writes/physics/contact/FCL/feedback.',
            'P-only camera correction norm25mm/50mrad; no world-Z integral.',
            'Original5s parking procedure; no new terminal hold/posture allocator.',
            'Original strictstop, camera, meaningful6D response, feedbackfalse and pace1.05 gates.'],
        cpu_gate_pass=report['cpu_gate_pass'],physical_verified=False,formal_readiness_changed=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        frozen_sha256={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()},
        preparation_evidence_sha256={p.name:sha(p) for p in OUTPUT.glob('*.json')})
    (OUTPUT/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('DECLARATION',sha(OUTPUT/'declaration.json'))


if __name__=='__main__':main()
