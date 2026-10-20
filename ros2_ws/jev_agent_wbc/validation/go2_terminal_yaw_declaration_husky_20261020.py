#!/usr/bin/env python3
"""Freeze one original yaw-left diagnostic with explicit terminal endpoint hold."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'Robot_Go2/results/terminal_yaw_preparation_01'
CPU=ROOT/'Robot_Go2/results/terminal_entry_cpu_02'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result=json.loads((CPU/'cpu_replay.json').read_text())
    assert result['cpu_gate_pass'] is True and result['original_7node_forecast_retained'] is True
    for name in ['adaptive_runner.py','camera_coordinator.py','terminal_allocator.py']:
        path=ROOT/'Robot_Go2'/name
        assert sha(path)==result['files_sha256'][str(path.relative_to(ROOT))]
    tests=ET.parse(CPU/'cpu_tests.xml').getroot()
    assert sum(int(x.get('tests','0'))for x in tests.iter('testsuite'))==103
    assert sum(int(x.get(key,'0'))for x in tests.iter('testsuite')for key in ['failures','errors','skipped'])==0
    OUTPUT.mkdir(exist_ok=False);frozen=OUTPUT/'sources';frozen.mkdir()
    paths=[ROOT/'Robot_Go2'/name for name in ['adaptive_runner.py','camera_coordinator.py','optical_sampler.py','world_z_feedback.py','terminal_allocator.py','wbc_backend.py','actuator_limits.py']]
    paths += [ROOT/'Robot_G1'/name for name in ['velocity_servo.py','legged_collision.py','pink_collision.py','upper_reference.py','contact_materials.py']]
    paths += [ROOT/'config/robot_sensor_mounts.json',ROOT/'Robot_Go2/run_adaptive_wbc.sh',Path(__file__)]
    for path in paths:shutil.copy2(path,frozen/path.name)
    shutil.copytree(ROOT/'jev_agent_wbc',frozen/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    upstream=ROOT/'Robot_Go2/vendor/LeggedManip_Lab'
    shutil.copy2(upstream/'mujoco/deploy/policy/go2_piper/wbc/policy.pt',frozen/'policy.pt')
    shutil.copy2(upstream/'mujoco/robots/go2_piper/go2piper.xml',frozen/'go2piper.xml')
    mdp=upstream/'source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab'
    for rel in ['config/go2_piper/wbc_env_cfg.py','mdp/pose_command_wbc.py']:
        dest=frozen/'go2_upstream'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(mdp/rel,dest)
    command=['Robot_Go2/run_adaptive_wbc.sh','--cases','yaw_left','--move-seconds','5','--stop-seconds','5',
        '--fine-collision','--go2-response-profile','measured_backward_v1','--go2-native-posture','x58_all_v1',
        '--go2-prediction-profile','persistent_bias_12_v1','--go2-world-z-feedback','bounded_integral_v1',
        '--go2-terminal-profile','endpoint_xy58_15_v1','--camera-excursion','diagnostic',
        '--ground-material','source_multiply','--parking-anchor-reset','--park-damping-scale','3','--rgb',
        '--output','Robot_Go2/results/adaptive_yaw_left_terminal_01']
    evidence=[CPU/'cpu_replay.json',CPU/'cpu_rows.json',CPU/'cpu_tests.xml',
        ROOT/'Robot_TIAGo/wbc/results/go2_terminal_helper_review03/summary.json']
    declaration=dict(schema='go2_terminal_yaw_declaration/v1',command=command,
        pace=1,physics_hz=200,control_hz=50,physics_substeps_per_action=4,
        cases=['yaw_left'],settle_s=2,move_s=5,stop_s=5,total_sim_s=12,
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in paths},
        frozen_sha256={str(path.relative_to(frozen)):sha(path)for path in frozen.rglob('*')if path.is_file()},
        cpu_evidence_sha256={str(path.relative_to(ROOT)):sha(path)for path in evidence},
        source_matches_cpu=True,cpu_tests_passed=103,cpu_tests_skipped=0,
        cpu_replay_samples=result['samples'],cpu_control_only_wall_ms=result['control_only_wall_ms'],
        terminal_profile='endpoint_xy58_15_v1',park_damping_scale=3,parking_camera_hold=False,
        world_z_feedback='bounded_integral_v1',ki_per_s=.2,world_feedback_position_correction_total_cap_m=.025,
        allocation_budget_s=1.6,original_slew_brake_budget_s=.4,literal_zero_deadline_offset_s=2.,
        literal_zero_remaining_s=3.,terminal_target_xy_link0_m=[.58,.15],
        reference='Exact original optical Task.end pose copied at acceptance; explicit endpoint hold only. Original knots/times/tolerances stay immutable. No trajectory extrapolation.',
        prediction_nodes_s=[0,.2,.4,.6,.8,1.,1.2],reserve_ramp_s=.8,
        original_position_reserve_m=[.035,.035,.015],original_orientation_reserve_rad=.015,
        original_native_caps=[.2,.2,.2],original_native_slew=[.5,.5,.8],
        model='Bounded50Hz original-slew sequence,150ms held actual input delay, continuous vx steady map u*(.55 if negative else1), tau.35. Forward response uncalibrated; other axes measured150ms inertia then input, measured full root tilt/fixed height/root-to-link.',
        model_limitations=['Private operating workspace approximation, not a policy/contact/joint/stability prediction.',
            'CPU proposals use separate frozen real states/history; no learned18 closed-loop parking verified.',
            'Source x=.58/y=.15 are subordinate operating preferences, not training stability certificates.',
            'Original P+ZI corrected native command must pass live domain check; all override token rollbacks/FCL safety retain priority.'],
        explicit_failure='History/candidate/deadline invalid => public feasibleFalse and original-slew brake. Deadline infeasibility stays latched; no zero jump or silently retimed Task.',
        unchanged=['Original yaw .12rad/s for5s, fixed5s stop,2s settle; fresh fullWORLD Task authored from real boundary measurement.',
            'All18 learned actions stay active; no root/q writes after initialization. No native7 hold.',
            'Original physics/material/PD3/FCL shapes/pairs/effort/velocity/joint caps and4substep writes.',
            'Post2s still original1.2s7point reserved forecast from fresh real velocity/history while applied body inputs remain literal0.',
            'All18q last.5s span<.01rad, base/camera span<.005m/fullangle<.01rad, final v<.02/w<.03, original camera50mm/.15rad and wall<=sim*1.05.',
            'No timed liveRGB; optional raw/probe only after timer.'],
        physical_verified=False,formal_readiness_changed=False,diagnostic_only=True)
    path=OUTPUT/'declaration.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    print(path);print(sha(path));print(' '.join(command))


if __name__=='__main__':main()
