#!/usr/bin/env python3
"""Freeze the already-probed optional Z-input feedback for the original complete nine-case suite."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PROBE=ROOT/'Robot_Go2/results/adaptive_camera6d_world_z_01'
OUTPUT=ROOT/'Robot_Go2/results/world_z_full_preparation_01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=False);frozen=OUTPUT/'sources';frozen.mkdir()
    paths=[ROOT/'Robot_Go2/adaptive_runner.py',ROOT/'Robot_Go2/camera_coordinator.py',ROOT/'Robot_Go2/optical_sampler.py',
        ROOT/'Robot_Go2/world_z_feedback.py',ROOT/'Robot_Go2/wbc_backend.py',ROOT/'Robot_G1/velocity_servo.py',ROOT/'Robot_G1/legged_collision.py',
        ROOT/'Robot_G1/contact_materials.py',ROOT/'config/robot_sensor_mounts.json',Path(__file__)]
    for p in paths:shutil.copy2(p,frozen/p.name)
    shutil.copytree(ROOT/'jev_agent_wbc',frozen/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    source_match={name:sha(ROOT/'Robot_Go2'/name)==sha(PROBE/'sources'/name)
                  for name in ('adaptive_runner.py','camera_coordinator.py','optical_sampler.py','world_z_feedback.py')}
    assert all(source_match.values())
    cases='forward,backward,left,right,diagonal_left,diagonal_right,yaw_left,yaw_right,camera_shot'
    command=['Robot_Go2/run_adaptive_wbc.sh','--cases',cases,'--move-seconds','5','--stop-seconds','5',
        '--fine-collision','--go2-response-profile','measured_backward_v1','--go2-native-posture','x58_all_v1',
        '--go2-prediction-profile','persistent_bias_12_v1','--go2-world-z-feedback','bounded_integral_v1','--camera-excursion','diagnostic',
        '--ground-material','source_multiply','--parking-anchor-reset','--rgb',
        '--output','Robot_Go2/results/adaptive_suite_world_z_01']
    declaration=dict(schema='go2_z_feedback_full_declaration/v1',command=command,pace=1,physics_hz=200,control_hz=50,
        physics_substeps_per_action=4,settle_s=2,move_s=5,stop_s=5,cases=cases.split(','),total_sim_s=92,
        source_equal_to_camera6d_probe=source_match,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        frozen_sha256={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()},
        probe_evidence_sha256={str((PROBE/name).relative_to(ROOT)):sha(PROBE/name) for name in
            ['summary.json','provenance.json','timed_tasks.json','camera_render_check.json']},
        physical_validation_scope='New original complete9case suite with independently authored immutable WORLD TimedTask at each actual measured case boundary; no edits to old tasks and no within-task retargeting.',
        prediction_profile='persistent_bias_12_v1',native_posture_profile='x58_all_v1',world_z_feedback='bounded_integral_v1',ki_per_s=.2,
        prediction_nodes_full_s=[0,.2,.4,.6,.8,1,1.2],reserve_ramp_s=.8,position_reserve_m=[.035,.035,.015],
        native_input_caps=[.2,.2,.2],native_input_slew=[.5,.5,.8],
        unchanged=['Original9case speed=.1m/s, yaw=.12rad/s, all5s move+5s stop; no liveRGB.',
            'Original WORLD task construction/absolute clock/linear-Slerp/camera optical transform and50mm/.15radTask tolerance.',
            'All18 learned policy actions/native caps/physics/friction/source collisionpairs/FCL/4substep writes.',
            'Same optional Ki=.2 world-Z feedback as passed6D: original total25mm/50mrad projection, fresh absolute-clock guard/task-local I reset/all outer tokenrollback; no new terminal model or stop preference.',
            'All18q stop<.01rad, base/camera spans<5mm/fullorientation<.01rad, finalv<.02/w<.03, samepace1.05.'],
        known_risk='Earlier full-suite endpoint Z bias accumulated because subsequent new Tasks use actual measured starts. This version integrates fresh world-Z measurement error in native input only; it does not modify any accepted Task. Single6D convergence does not certify nine-case stability; all failures will remain failures.',
        readiness_changed=False,physical_full_verified=False)
    path=OUTPUT/'declaration.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    print(path);print(sha(path));print(' '.join(command))


if __name__=='__main__':main()
