#!/usr/bin/env python3
"""Existing native7 hold on preserved actual states; never freezes18 actions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'Robot_Go2/results/parking_hold_candidate_preparation_01'
sys.path.insert(0,str(ROOT))
from jev_agent_wbc.wbc_protocol import TimedTask


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(run_name,modules):
    folder=ROOT/'Robot_Go2/results'/run_name
    runner,coordinate,feedback,sampler_module,backend=modules
    backend.UPSTREAM_MDP=folder/'sources/go2_upstream';domain=backend.PolicyDomain()
    tasks=[TimedTask.from_dict(task) for task in json.loads((folder/'timed_tasks.json').read_text())]
    samples=[sampler_module.OpticalSampler(task) for task in tasks]
    summary=json.loads((folder/'summary.json').read_text());cases=[case['case']for case in summary['cases']]
    rows=[json.loads(line)for line in (folder/'measurements.jsonl').read_text().splitlines()]
    extrinsic=np.asarray(json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    controller=feedback.WorldZFeedback(enabled=True,ki_per_s=.2)
    current_case=None;records=[];case_reports={};events=[]
    for index,row in enumerate(rows):
        if row['case']=='settle':continue
        stamp=row['stamp_s']-.02;previous=rows[index-1];case=row['case'];task_index=cases.index(case);task=tasks[task_index]
        if case!=current_case:
            current_case=case;parking=coordinate.NativeCameraParking()
            case_reports[case]=dict(original_stop_pass=summary['cases'][task_index]['strict_stop']['passed'],
                captures=0,releases=0,first_capture_s_after_task_end=None,held_samples=0,stop_samples=0,
                native_domain_failure_count=0,original_actual_camera_threshold_release_count=0)
        body=np.asarray(previous['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07])
        reference=samples[task_index].sample(min(stamp,task.trajectory.end_s));camera=np.asarray(previous['T_world_camera_optical'])
        native,checked=controller.feedback_native_command(backend,reference,camera,link,extrinsic,
            task_id=task.task_id,stamp_s=stamp,now_s=stamp,dt_s=.02,sample_seq=index,fresh=True,
            integration_allowed=row['camera_support'].get('feasible') is not False,domain=domain)
        position_error=float(np.linalg.norm(reference[:3,3]-camera[:3,3]))
        angle_error=float(Rotation.from_matrix(reference[:3,:3]@camera[:3,:3].T).magnitude())
        prior_active=parking.command is not None;prior_count=parking.transitions
        release_by_base=bool(parking.base is not None and (np.linalg.norm(body[:3,3]-parking.base[:3,3])>.020
            or Rotation.from_matrix(body[:3,:3]@parking.base[:3,:3].T).magnitude()>.04))
        enabled=bool(row['phase']=='stop' and row['native_standing_mode'])
        held,parking_report=parking.update(native,reference,camera,body,np.asarray(previous['measured_body_velocity']),parking=enabled)
        original_rejected=checked.get('rejected_feedback_correction')
        final=domain.check(held)
        if original_rejected is not None:final['rejected_feedback_correction']=original_rejected
        final=runner.finalize_native_feedback(controller,checked,final,['native_parking_hold']if parking_report['active']else[])
        entry=case_reports[case];entry['native_domain_failure_count']+=final['status']!='inside_outer_domain_unverified'
        if row['phase']=='stop':
            entry['stop_samples']+=1;entry['held_samples']+=parking_report['active']
        transitions=parking.transitions-prior_count
        if transitions:
            release=prior_active and (not enabled or position_error>.040 or angle_error>.12 or release_by_base)
            capture=parking_report['active'] and (not prior_active or release)
            entry['captures']+=capture;entry['releases']+=release
            entry['original_actual_camera_threshold_release_count']+=release and (position_error>.040 or angle_error>.12)
            if capture and entry['first_capture_s_after_task_end']is None:
                entry['first_capture_s_after_task_end']=stamp-task.trajectory.end_s
            events.append(dict(case=case,stamp_s=stamp,stop_elapsed_s=stamp-task.trajectory.end_s,
                capture=bool(capture),release=bool(release),release_by_base=release_by_base,
                original_position_error_m=position_error,original_orientation_error_rad=angle_error,
                native_xyz=held[:3].tolist(),actual_domain_status=final['status']))
        records.append(dict(case=case,stamp_s=stamp,phase=row['phase'],hold_active=parking_report['active'],
            original_position_error_m=position_error,original_orientation_error_rad=angle_error,
            native_held=held.tolist(),unheld_native=native.tolist(),world_z_feedback=final['world_z_feedback']))
    for case,entry in case_reports.items():
        selected=[record for record in records if record['case']==case]
        tail=[record for record in selected if record['stamp_s']>=selected[-1]['stamp_s']-.5-1e-9]
        entry['last_half_second_hold_fraction']=float(np.mean([record['hold_active']for record in tail]))
        entry['last_half_second_native_xyz_span_m']=np.ptp(np.asarray([record['native_held']for record in tail])[:,:3],axis=0).tolist()
        entry['last_half_second_integral_span_m']=float(np.ptp([record['world_z_feedback']['integral_after_m']for record in tail]))
    return dict(run=run_name,samples=len(records),cases=case_reports,events=events,
        original_stop_passes_unchanged_in_frozen_trace=True,
        native_domain_failure_count=sum(entry['native_domain_failure_count']for entry in case_reports.values()),
        invalid_clock_count=sum(not record['world_z_feedback']['measurement_fresh_and_clock_valid']for record in records),
        rollback_count=sum(record['world_z_feedback']['externally_rolled_back']for record in records)),records


def main():
    started=time.monotonic();OUTPUT.mkdir(exist_ok=False);frozen=OUTPUT/'sources';frozen.mkdir()
    files=[ROOT/'Robot_Go2'/name for name in ['adaptive_runner.py','camera_coordinator.py','world_z_feedback.py','optical_sampler.py','wbc_backend.py']]
    files.append(Path(__file__))
    for path in files:shutil.copy2(path,frozen/path.name)
    modules=[load('hold_preflight_'+path.stem,frozen/path.name) for path in files[:5]]
    runs={};trace_files=[]
    for run in ['adaptive_camera6d_world_z_01','adaptive_suite_world_z_01']:
        report,rows=replay(run,modules);runs[run]=report
        path=OUTPUT/(run+'_cpu_rows.json');path.write_text(json.dumps(rows,indent=2)+'\n')
        trace_files.extend(ROOT/'Robot_Go2/results'/run/name for name in ['measurements.jsonl','timed_tasks.json','provenance.json'])
    report=dict(schema='go2_parking_hold_cpu/v1',scope='Read-only frozen actual-state eligibility/transaction replay. Native7 values may be held, but measured trajectories,18 learned actions/q and camera errors remain the original failed or passed physical data. Not a closed-loop improvement or physical success.',
        runs=runs,cpu_wall_s=time.monotonic()-started,
        cpu_gate_pass=all(not r['native_domain_failure_count']and not r['invalid_clock_count']for r in runs.values()),
        original_capture=dict(position_m=.030,orientation_rad=.09,base_linear_mps=.02,base_yaw_radps=.03),
        original_release=dict(position_m=.040,orientation_rad=.12,base_position_m=.020,base_rotation_rad=.04),
        original_thresholds_unchanged=True,files_sha256={str(path.relative_to(ROOT)):sha(path)for path in files+trace_files})
    (OUTPUT/'cpu_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    command=['Robot_Go2/run_adaptive_wbc.sh','--cases','camera_6d','--move-seconds','16','--stop-seconds','5',
        '--fine-collision','--go2-response-profile','measured_backward_v1','--go2-native-posture','x58_all_v1',
        '--go2-prediction-profile','persistent_bias_12_v1','--go2-world-z-feedback','bounded_integral_v1',
        '--camera-excursion','nontrivial','--ground-material','source_multiply','--parking-anchor-reset',
        '--parking-camera-hold','--rgb','--output','Robot_Go2/results/adaptive_camera6d_native_hold_01']
    declaration=dict(schema='go2_parking_hold_declaration/v1',command=command,pace=1,total_sim_s=23,
        physics_hz=200,control_hz=50,physics_substeps=4,settle_s=2,original_task_duration_s=16,stop_s=5,
        change='Only enable existing --parking-camera-hold at original hysteresis, preserving all18 learned actions. World-Z I transaction rolls back while native7 hold is active. No source changes.',
        unchanged=['Original WORLD Task/knot/time/tolerance/full optical pose and measured feedback errors.',
            'Same predictor/x58_all/Ki=.2/total25mm correction/caps/slew/physics/contact/FCL/four substep writes.',
            'Original five-second stop budget and all18q10mrad/base-camera5mm/orientation10mrad/finalv.02/w.03/pace1.05 thresholds.'],
        source_sha256={str(path.relative_to(ROOT)):sha(path)for path in files},
        frozen_sha256={str(path.relative_to(frozen)):sha(path)for path in frozen.rglob('*')if path.is_file()},
        cpu_evidence_sha256={path.name:sha(path)for path in OUTPUT.glob('*.json')},
        cpu_gate_pass=report['cpu_gate_pass'],physical_verified=False,formal_readiness_changed=False)
    path=OUTPUT/'declaration.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('DECLARATION',sha(path));print(' '.join(command))


if __name__=='__main__':main()
