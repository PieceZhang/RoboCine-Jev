#!/usr/bin/env python3
"""Freeze the admission-only fresh measured Z-bias candidate; no physical claim."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'Robot_Go2/results/terminal_pose_full_preparation_03'
OUT=ROOT/'Robot_Go2/results/bumpless_z_full_preparation_01'
CPU=ROOT/'Robot_Go2/results/bumpless_z_seed_cpu_05'
TESTS=ROOT/'Robot_Go2/results/bumpless_z_seed_cpu_04/cpu_tests.xml'
INDEPENDENT=ROOT/'Robot_TIAGo/wbc/results/go2_bumpless_independent02'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior=json.loads((PREVIOUS/'declaration.json').read_text());cpu=json.loads((CPU/'summary.json').read_text())
    independent=json.loads((INDEPENDENT/'summary.json').read_text())
    assert cpu['cpu_gate_pass']is True and cpu['eligible_count']==6 and cpu['boundaries']==9
    assert independent['passed']is True and independent['checks']==28
    for name in ['world_z_bumpless.py','adaptive_runner.py','world_z_feedback.py']:
        assert sha(ROOT/'Robot_Go2'/name)==independent['source_sha256'][name]
    for name,value in cpu['files_sha256'].items():assert sha(ROOT/name)==value
    suites=list(ET.parse(TESTS).getroot().iter('testsuite'))
    count=sum(int(s.get('tests','0'))for s in suites)
    assert count==151 and sum(int(s.get(k,'0'))for s in suites for k in ['failures','errors','skipped'])==0
    stable=['camera_coordinator.py','bounded_body_prediction.py','terminal_allocator.py','terminal_yaw_allocator.py',
        'world_z_feedback.py','optical_sampler.py','wbc_backend.py','actuator_limits.py','run_adaptive_wbc.sh']
    assert all(sha(ROOT/'Robot_Go2'/name)==prior['source_sha256']['Robot_Go2/'+name]for name in stable)
    OUT.mkdir(exist_ok=False);frozen=OUT/'sources';shutil.copytree(PREVIOUS/'sources',frozen)
    actual=[ROOT/'Robot_Go2'/name for name in ['adaptive_runner.py','world_z_bumpless.py']+stable]
    actual += [ROOT/'Robot_G1'/name for name in ['velocity_servo.py','legged_collision.py','contact_materials.py']]
    actual += [ROOT/'config/robot_sensor_mounts.json']
    archived=[ROOT/'Robot_G1'/name for name in ['wbc_backend.py','pink_collision.py','upper_reference.py']]
    support=[ROOT/'test/test_wbc_go2_bumpless_z_seed.py',ROOT/'validation/go2_bumpless_z_boundary_preflight04_husky_20261020.py',Path(__file__)]
    for path in actual+archived+support:
        name='g1_backend.py'if path==ROOT/'Robot_G1/wbc_backend.py'else'go2_backend.py'if path==ROOT/'Robot_Go2/wbc_backend.py'else path.name
        shutil.copy2(path,frozen/name)
    shutil.rmtree(frozen/'jev_agent_wbc')
    shutil.copytree(ROOT/'jev_agent_wbc',frozen/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    for name,value in prior['frozen_sha256'].items():
        if name in ['policy.pt','go2piper.xml']or name.startswith('go2_upstream/'):
            assert sha(frozen/name)==value
    argv=prior['command'].copy();argv[-1]='Robot_Go2/results/adaptive_suite_bumpless_z_01'
    argv[argv.index('--go2-world-z-feedback')+2:argv.index('--go2-world-z-feedback')+2]=['--go2-world-z-seed','fresh_steady_native_z_v1']
    evidence=[CPU/'summary.json',TESTS,INDEPENDENT/'summary.json',INDEPENDENT/'sources/audit_go2_bumpless_seed.py',INDEPENDENT/'pytest.xml']
    assert all(path.is_file()for path in evidence)
    keep=['pace','physics_hz','control_hz','physics_substeps_per_action','cases','settle_s','move_s','stop_s','total_sim_s',
        'prediction_profile','terminal_profile','prediction_nodes_full_s','reserve_ramp_s','position_reserve_m','orientation_reserve_rad',
        'original_native_input_caps','original_native_input_slew','prediction','body_operating_preference','reference',
        'native_posture_profile','park_damping_scale','parking_camera_hold','world_z_feedback','ki_per_s','total_world_position_correction_cap_m']
    result={key:prior[key]for key in keep}
    result.update(schema='go2_bumpless_z_full_declaration/v1',command=argv,
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in actual+support},
        archived_unexecuted_g1_sources_sha256={str(path.relative_to(ROOT)):sha(path)for path in archived},
        frozen_sha256={str(path.relative_to(frozen)):sha(path)for path in frozen.rglob('*')if path.is_file()},
        cpu_evidence_sha256={str(path.relative_to(ROOT)):sha(path)for path in evidence},
        cpu_tests_passed=count,cpu_tests_skipped=0,independent_transaction_checks=28,physical_verified=False,formal_readiness_changed=False,
        admission_seed_profile='fresh_steady_native_z_v1',
        admission_seed=dict(original_full_task_sha=True,same_stamp_measured_camera_link_extrinsic=True,
            prior_actual_input_precision='float32 EE7 consumed by the unchanged learned policy',prior_max_age_s=.06,
            steady_window_s=.5,steady_window_samples=26,native_z_span_m=.001,actual_ee_z_span_m=.001,bias_span_m=.001,
            original_total_position_correction_cap_m=.025,first_task_seed_m=0.,unverified_seed_m=0.,
            source='previous actually consumed native7.z minus fresh original Task reference converted to raw native EE worldZ',
            commit='Only after all same-tick guards and actual float32 learned policy consumption succeed. Callback exception rolls seed to0 and fails run.',
            rollback='Any current outer guard, stale/domain/clock rejection, actual-input mismatch or next feedback before confirmation resets provisional seed to0; old confirmation token rejected.',
            full_sha_admission='Same task_id with different path/tolerance/observation is a new full-SHA admission; changed fresh context is a new uniquely identified event.'),
        cpu_scope='Nine separate frozen measured admissions: six of eight subsequent boundaries eligible, five exact continuity and one .910mm cap residual. Two unsteady windows remain original zero reset. CPU interface/math only; no learned-policy propagation or physical convergence claim.',
        performance_scope='Only9 admissions scan26 measured rows; subsequent frames forward original pinned WorldZFeedback. Per-frame metadata small calibration ID/status; full evidence written once after timer.',
        unchanged=['Original nine5s move+5s stop cases and2s settle, speed.1/yaw.12, no task retarget/retime or tolerance change.',
            'Same bounded_ramp_history_v2 and endpoint_xy58_15_yaw0_v1 helpers,1.6s allocation+.4s brake+3s literal0, originalcaps/slew/reserves.',
            'Original WorldZFeedback3d49 P+.2I,25mm total3Dcap, clock/token/domain and outerrollback; only eligible fresh Task initial bias differs.',
            'All18 learned actions, source PD/parkD3/source_multiply contact/material/geometry/FCL/200Hz50Hz4writes unchanged.',
            'Original complete.5s all18q<.01rad, base/camera5mm/10mrad, finalv<.02/w<.03 and wall<=sim*1.05.',
            'No liveRGB during timed physics; posttimer rawRGB and optical+Zprobe only.'],
        known_limitations=['Fresh steady calibration is an observed operating bias, not trained-domain/stability or environment avoidance proof.',
            '25mm cap may prevent full continuity; unsteady/unsafe/recovery/hold/missing evidence rejectsseed and may retain accumulatedZ.',
            'Existing root-motion predictor is an approximation, not contact/learned18 dynamics prediction.',
            'Formal control readiness remainsFalse until this exact profile passes FULL and meaningful6D with independent audits.'],
        unused_archive_scope='G1 backend/pink_collision/upper_reference are unconditionally copied for provenance but this Go2 branch does not load or execute them. They are not live executed dependencies.')
    path=OUT/'declaration.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(path);print(sha(path));print(' '.join(argv))


if __name__=='__main__':main()
