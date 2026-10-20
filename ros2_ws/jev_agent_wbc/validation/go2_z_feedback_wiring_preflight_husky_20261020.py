#!/usr/bin/env python3
"""Freeze the optional Z-input wrapper and replay unchanged physical samples."""
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
RUN=ROOT/'Robot_Go2/results/adaptive_suite_prediction_01'
OUTPUT=ROOT/'Robot_Go2/results/world_z_candidate_preparation_02'
sys.path.insert(0,str(ROOT))
from jev_agent_wbc.wbc_protocol import TimedTask


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.monotonic();OUTPUT.mkdir(exist_ok=False);frozen=OUTPUT/'sources';frozen.mkdir()
    files=[ROOT/'Robot_Go2/adaptive_runner.py',ROOT/'Robot_Go2/camera_coordinator.py',ROOT/'Robot_Go2/world_z_feedback.py',
        ROOT/'Robot_Go2/optical_sampler.py',ROOT/'Robot_Go2/wbc_backend.py',ROOT/'Robot_G1/velocity_servo.py',
        ROOT/'Robot_G1/legged_collision.py',ROOT/'Robot_G1/contact_materials.py',ROOT/'config/robot_sensor_mounts.json',
        ROOT/'test/test_wbc_go2_z_feedback_wiring.py',ROOT/'test/test_wbc_world_z_feedback.py',Path(__file__)]
    for path in files:shutil.copy2(path,frozen/path.name)
    shutil.copytree(ROOT/'jev_agent_wbc',frozen/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    (OUTPUT/'adaptive_runner.py.diff').write_text(''.join(difflib.unified_diff(
        (RUN/'sources/adaptive_runner.py').read_text().splitlines(keepends=True),
        (ROOT/'Robot_Go2/adaptive_runner.py').read_text().splitlines(keepends=True),fromfile='prediction_only/adaptive_runner.py',
        tofile='optional_z/adaptive_runner.py')))
    runner=load('z_preflight_runner',frozen/'adaptive_runner.py')
    module=load('z_preflight_helper',frozen/'world_z_feedback.py')
    backend=load('z_preflight_backend',frozen/'wbc_backend.py');backend.UPSTREAM_MDP=RUN/'sources/go2_upstream'
    coordinate=load('z_preflight_coordinator',frozen/'camera_coordinator.py')
    sampler_module=load('z_preflight_sampler',frozen/'optical_sampler.py')
    tasks=[TimedTask.from_dict(value) for value in json.loads((RUN/'timed_tasks.json').read_text())]
    samplers=[sampler_module.OpticalSampler(task) for task in tasks]
    rows=[json.loads(line) for line in (RUN/'measurements.jsonl').read_text().splitlines()]
    cases=[result['case'] for result in json.loads((RUN/'summary.json').read_text())['cases']]
    extrinsic=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    domain=backend.PolicyDomain();controller=module.WorldZFeedback(enabled=True,ki_per_s=.2)
    disabled=module.WorldZFeedback(enabled=False)
    checked=[];max_legacy_difference=0.;per_task={}
    for index,row in enumerate(rows):
        if row['case']=='settle':continue
        stamp=row['stamp_s']-.02;previous=rows[index-1];task_index=cases.index(row['case']);task=tasks[task_index]
        reference=samplers[task_index].sample(min(stamp,task.trajectory.end_s))
        measured=np.asarray(previous['T_world_camera_optical']);body=np.asarray(previous['T_world_base']);link=body.copy()
        link[:3,3]+=body[:3,:3]@np.array([0,0,.07])
        native,checked_domain=controller.feedback_native_command(backend,reference,measured,link,extrinsic,
            task_id=task.task_id,stamp_s=stamp,now_s=stamp,dt_s=.02,sample_seq=index,fresh=True,
            integration_allowed=row['camera_support'].get('feasible') is not False,domain=domain)
        old_native,old_check=coordinate.feedback_native_command(backend,reference,measured,link,extrinsic,domain=domain)
        disabled_native,disabled_check=disabled.feedback_native_command(backend,reference,measured,link,extrinsic,domain=domain)
        max_legacy_difference=max(max_legacy_difference,float(np.max(abs(old_native-disabled_native))))
        assert old_check['status']==disabled_check['status']
        overrides=[]
        if row.get('native_camera_parking',{}).get('active'):overrides.append('native_parking_hold')
        if row.get('native_domain_recovery',{}).get('active'):overrides.append('native_domain_recovery')
        if row.get('task_feedback',{}).get('constraints',{}).get('self_collision',{}).get('status')!='pass':overrides.append('collision_freshness_veto')
        if row['camera_support'].get('feasible') is False:overrides.append('original_camera_support_rejected')
        final=runner.finalize_native_feedback(controller,checked_domain,domain.check(native),overrides)
        report=final['world_z_feedback']
        per_task[row['case']]=dict(final_integral_m=controller.integral_m,
            first_camera_measurement_stamp_s=per_task.get(row['case'],{}).get('first_camera_measurement_stamp_s',stamp))
        checked.append(dict(stamp_s=stamp,case=row['case'],task_id=task.task_id,
            domain_status=final['status'],native=native.tolist(),report=report,
            camera_reference_changed=False))
    violations=[row for row in checked if row['report']['proposed_position_correction_norm_m']>.025+1e-12
        or np.linalg.norm(row['report']['proposed_orientation_correction_rad'])>.05+1e-12
        or abs(row['report']['integral_after_m'])>.025+1e-12]
    report=dict(schema='go2_z_wiring_cpu/v1',scope='4500 frozen actual-state command-time samples from the original9Tasks. No plant is integrated; measured errors cannot converge in this replay. No physical success claim.',
        samples=len(checked),per_task=per_task,maximum_disabled_legacy_native_difference=max_legacy_difference,
        correction_cap_violation_count=len(violations),domain_rejections=sum(row['domain_status']!='inside_outer_domain_unverified' for row in checked),
        rejected_corrected_command_count=sum(row['report']['domain_correction_rejected'] for row in checked),
        invalid_clock_count=sum(not row['report']['measurement_fresh_and_clock_valid']for row in checked),
        antiwindup_freeze_count=sum(row['report']['cap_antiwindup_frozen']for row in checked),
        outer_override_rollback_count=sum(row['report']['externally_rolled_back']for row in checked),
        maximum_integral_abs_m=max(abs(row['report']['integral_after_m'])for row in checked),
        maximum_position_correction_m=max(row['report']['proposed_position_correction_norm_m']for row in checked),
        task_reset_count=len(per_task),cpu_wall_s=time.monotonic()-started,
        source_trace_sha256={str((RUN/name).relative_to(ROOT)):sha(RUN/name)for name in ['measurements.jsonl','timed_tasks.json','provenance.json']})
    report['cpu_gate_pass']=bool(not violations and not report['domain_rejections'] and not report['invalid_clock_count'] and max_legacy_difference==0)
    (OUTPUT/'cpu_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUTPUT/'cpu_rows.json').write_text(json.dumps(checked,indent=2)+'\n')
    command=['Robot_Go2/run_adaptive_wbc.sh','--cases','camera_6d','--move-seconds','16','--stop-seconds','5',
        '--fine-collision','--go2-response-profile','measured_backward_v1','--go2-native-posture','x58_all_v1',
        '--go2-prediction-profile','persistent_bias_12_v1','--go2-world-z-feedback','bounded_integral_v1',
        '--camera-excursion','nontrivial','--ground-material','source_multiply','--parking-anchor-reset','--rgb',
        '--output','Robot_Go2/results/adaptive_camera6d_world_z_01']
    declaration=dict(schema='go2_z_wiring_declaration/v1',command=command,pace=1,total_sim_s=23,
        physics_hz=200,control_hz=50,physics_substeps=4,settle_s=2,original_task_duration_s=16,stop_s=5,
        change='Only optional already-audited world-Z I added before original full3D25mm projection, Ki=.2/s; task-local I reset with globally monotone absolute measurement clock/step. None default preserves original P/native allocator.',
        world_z_helper_sha256=sha(ROOT/'Robot_Go2/world_z_feedback.py'),
        override_rollback=['native_parking_hold','native_domain_recovery','collision_freshness_veto','original_camera_support_rejected','actual_native_domain_failure'],
        integration_veto=['original camera unsupported','original requested native outside domain','FCL evidence not fresh pass','missing/stale/nonmonotone measurement'],
        unchanged=['All18 learned actions/4substep writes/original physics/caps/FCL/WorldTask samples and absoluteclock.',
            'Successful predictor persistent_bias_12_v1/x58_all_v1/.8s reserve ramp and original caps/slew.',
            'Original strictstop5s, exact same camera/tolerance/meaningful6D/feedbackfalse/pace1.05 gates.',
            'No new terminal allocation/parking profile, no direct joint/root writes during measured time.',
            'During literalzero parking, if native EE still receives feedback and no outer override, fresh I may continue.'],
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in files},
        frozen_sha256={str(path.relative_to(frozen)):sha(path)for path in frozen.rglob('*')if path.is_file()},
        cpu_evidence_sha256={name:sha(OUTPUT/name)for name in ['cpu_replay.json','cpu_rows.json']},
        cpu_gate_pass=report['cpu_gate_pass'],physical_verified=False,formal_readiness_changed=False)
    path=OUTPUT/'declaration.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('DECLARATION',sha(path));print(' '.join(command))


if __name__=='__main__':main()
