#!/usr/bin/env python3
"""Freeze bounded all-axis input forecast and explicit endpointXY/yaw preference."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Robot_Go2/results/terminal_pose_full_preparation_01'
STOP=ROOT/'Robot_Go2/results/terminal_yaw_entry_cpu_01'
MOVING=ROOT/'Robot_Go2/results/bounded_moving_entry_cpu_01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stop=json.loads((STOP/'cpu_replay.json').read_text());moving=json.loads((MOVING/'summary.json').read_text())
    assert stop['cpu_gate_pass']is True and moving['cpu_gate_pass']is True
    for result in [stop,moving]:
        for name,value in result['files_sha256'].items():assert sha(ROOT/name)==value
    tests=ET.parse(STOP/'cpu_tests.xml').getroot()
    assert sum(int(s.get('tests','0'))for s in tests.iter('testsuite'))==127
    assert sum(int(s.get(k,'0'))for s in tests.iter('testsuite')for k in ['failures','errors','skipped'])==0
    OUT.mkdir(exist_ok=False);frozen=OUT/'sources';frozen.mkdir()
    paths=[ROOT/'Robot_Go2'/name for name in ['adaptive_runner.py','camera_coordinator.py','bounded_body_prediction.py',
        'terminal_allocator.py','terminal_yaw_allocator.py','world_z_feedback.py','optical_sampler.py','wbc_backend.py','actuator_limits.py','run_adaptive_wbc.sh']]
    paths += [ROOT/'Robot_G1'/name for name in ['velocity_servo.py','wbc_backend.py','legged_collision.py','pink_collision.py','upper_reference.py','contact_materials.py']]
    paths += [ROOT/'config/robot_sensor_mounts.json',Path(__file__)]
    for path in paths:
        name='g1_backend.py'if path==ROOT/'Robot_G1/wbc_backend.py'else'go2_backend.py'if path==ROOT/'Robot_Go2/wbc_backend.py'else path.name
        shutil.copy2(path,frozen/name)
    shutil.copytree(ROOT/'jev_agent_wbc',frozen/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    upstream=ROOT/'Robot_Go2/vendor/LeggedManip_Lab'
    shutil.copy2(upstream/'mujoco/deploy/policy/go2_piper/wbc/policy.pt',frozen/'policy.pt')
    shutil.copy2(upstream/'mujoco/robots/go2_piper/go2piper.xml',frozen/'go2piper.xml')
    mdp=upstream/'source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab'
    for rel in ['config/go2_piper/wbc_env_cfg.py','mdp/pose_command_wbc.py']:
        dest=frozen/'go2_upstream'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(mdp/rel,dest)
    cases='forward,backward,left,right,diagonal_left,diagonal_right,yaw_left,yaw_right,camera_shot'
    command=['Robot_Go2/run_adaptive_wbc.sh','--cases',cases,'--move-seconds','5','--stop-seconds','5',
        '--fine-collision','--go2-response-profile','measured_backward_v1','--go2-native-posture','x58_all_v1',
        '--go2-prediction-profile','bounded_ramp_history_v2','--go2-world-z-feedback','bounded_integral_v1',
        '--go2-terminal-profile','endpoint_xy58_15_yaw0_v1','--camera-excursion','diagnostic',
        '--ground-material','source_multiply','--parking-anchor-reset','--park-damping-scale','3','--rgb',
        '--output','Robot_Go2/results/adaptive_suite_terminal_pose_01']
    evidence=[STOP/'cpu_replay.json',STOP/'cpu_rows.json',STOP/'cpu_tests.xml',MOVING/'summary.json',MOVING/'rows.json',
        ROOT/'Robot_Go2/results/bounded_ramp_false_cpu_01/summary.json',ROOT/'Robot_Go2/results/terminal_yaw_allocation_cpu_02/cpu_replay.json',
        ROOT/'Robot_Go2/results/world_z_boundary_readonly_01/summary.json']
    declaration=dict(schema='go2_terminal_pose_full_declaration/v1',command=command,
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in paths},
        frozen_sha256={str(path.relative_to(frozen)):sha(path)for path in frozen.rglob('*')if path.is_file()},
        cpu_evidence_sha256={str(path.relative_to(ROOT)):sha(path)for path in evidence},
        source_matches_cpu=True,cpu_tests_passed=127,cpu_tests_skipped=0,
        separate_original_snapshots=dict(moving=2250,stop=2250),
        cpu_control_only_wall_ms=dict(moving=moving['control_only_wall_ms'],stop=stop['control_only_wall_ms']),
        cpu_scope='Frozen original measurements/history; no propagation of proposed inputs or learned-policy/physics simulation. Math/control-interface checks only.',
        pace=1,physics_hz=200,control_hz=50,physics_substeps_per_action=4,cases=cases.split(','),
        settle_s=2,move_s=5,stop_s=5,total_sim_s=92,
        prediction_profile='bounded_ramp_history_v2',terminal_profile='endpoint_xy58_15_yaw0_v1',
        prediction_nodes_full_s=[0,.2,.4,.6,.8,1,1.2],reserve_ramp_s=.8,
        position_reserve_m=[.035,.035,.015],orientation_reserve_rad=.015,
        original_native_input_caps=[.2,.2,.2],original_native_input_slew=[.5,.5,.8],
        prediction='Actual first input then original50Hz slew ramp to persistent bounded future target; actual150ms held history. Continuous x steady(u)=u*(.55 if negative else1),tau.35; forward lag uncalibrated. vy/yaw measured initial rate residual linearly decays over existing150ms inertia window, no additional tau.',
        body_operating_preference=dict(x_m=.58,y_m=.15,native_yaw_rad=0.,allocation_budget_s=1.6,original_slew_brake_budget_s=.4,
            literal_zero_deadline_offset_s=2.,literal_zero_remaining_s=3.,certificate='none: subordinate operating hypothesis only'),
        reference='Exact original accepted optical Task.end pose explicitly held during original5s stop; no knot/time/tolerance rewrite or extrapolation. WORLD Task at each boundary starts from fresh actual measured state.',
        native_posture_profile='x58_all_v1',park_damping_scale=3.,parking_camera_hold=False,
        world_z_feedback='bounded_integral_v1',ki_per_s=.2,world_z_integral_task_reset='unchanged zero reset',total_world_position_correction_cap_m=.025,
        unchanged=['Original9 directions with speed.1/yaw.12 and all5s move5s stop.',
            'Original source physics/material/PD/collision shapes+pairs/all18 learned actions/native q/qd/effort caps,4substep writes.',
            'Original Task50mm/.15rad tracking/final error,full.5s all18q<.01rad,base/camera<5mm/10mrad,final v<.02/w<.03,wall<=sim*1.05.',
            'Same fresh measured P+Ki.2/25mm projection,original native-domain hard check and every override tokenrollback; no bumpless bias initialization.',
            'Original FCL/current configuration/actuation checks,source history/fresh clock guards. Environment/swept collision and sensor avoidance uncertified.',
            'No liveRGB during timed physics; raw image/optical probe after timer only.'],
        known_limitations=['Operating forecast approximates measured root motion with fixed root height/full measured tilt and root-to-link relation; learned18/joints/contact/stability not predicted.',
            'Frozen late out-of-domain source samples stay infeasible; mathematical firstdiagR improvement is not full closed-loop acceptance.',
            'Accumulated Z debt from original freshTask integral reset remains intentionally unchanged and can still affect complete suite.',
            'Old full-terminal and all prior failures remain immutable; formal3gates stayFalse until complete+meaningful6D independent physical audits.'],
        physical_verified=False,formal_readiness_changed=False)
    p=OUT/'declaration.json';p.write_text(json.dumps(declaration,indent=2)+'\n');print(p);print(sha(p));print(' '.join(command))


if __name__=='__main__':main()
