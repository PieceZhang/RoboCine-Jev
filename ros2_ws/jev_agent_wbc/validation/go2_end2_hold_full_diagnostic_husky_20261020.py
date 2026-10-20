#!/usr/bin/env python3
"""Read-only analysis of failed complete native terminal allocation suite."""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_terminal_hold_01'
OUT=ROOT/'Robot_Go2/results/terminal_hold_full_raw_diagnostic_01'


def main():
    rows=[json.loads(line)for line in(RUN/'measurements.jsonl').read_text().splitlines()]
    summary=json.loads((RUN/'summary.json').read_text());records=[]
    for case in summary['cases']:
        actual=[row for row in rows if row['case']==case['case']]
        stop=[row for row in actual if row['phase']=='stop'];bad=[row for row in actual if row['camera_support']['feasible']is False]
        end=stop[0]['terminal_allocation']['reference_stamp_s'];late=[row for row in stop if row['stamp_s']-.02>=end+2-1e-9];tail=stop[-26:]
        def native_stat(key):
            v=np.asarray([row[key]for row in tail]);return dict(mean=v.mean(axis=0).tolist(),span=np.ptp(v,axis=0).tolist())
        zi=[row['actual_native_domain']['world_z_feedback']for row in tail]
        estimated=np.asarray([row['terminal_allocation'].get('model_final_target_error_xy_m',[np.nan,np.nan])for row in stop])
        details=[]
        for row in bad[:2]+bad[-2:]:
            details.append({key:row.get(key)for key in ['stamp_s','phase','position_error_m','orientation_error_rad','applied_policy_velocity',
                'actual_native_command','terminal_allocation','camera_support','native_domain_recovery','rejected_feedback_correction']})
        corrected=np.asarray([row['actual_native_command'][:3]for row in tail])-np.asarray([row['requested_native_command'][:3]for row in tail])
        records.append(dict(case=case['case'],passed=case['passed'],strict_stop=case['strict_stop'],
            position_rms_mm=1000*case['camera_position_rmse_m'],final_error_mm=1000*actual[-1]['position_error_m'],
            final_z_error_mm=1000*(np.asarray(actual[-1]['T_world_camera_optical'])[2,3]-np.asarray(actual[-1]['T_world_reference_optical'])[2,3]),
            infeasible_count=len(bad),infeasible_phase_counts=dict(collections.Counter(row['phase']for row in bad)),
            infeasible_reasons=dict(collections.Counter(row['camera_support']['reason']for row in bad)),
            infeasible_stamp_range_s=[bad[0]['stamp_s'],bad[-1]['stamp_s']]if bad else None,
            terminal_error_counts=dict(collections.Counter(row['terminal_allocation'].get('error')for row in stop if row['terminal_allocation'].get('error'))),
            min_forecast_margin=min(row['camera_support']['predicted_min_domain_margin']for row in stop),
            post2_max_abs_applied_input=float(np.max(abs(np.asarray([row['applied_policy_velocity']for row in late])))),
            phase_counts=dict(collections.Counter(row['terminal_allocation'].get('phase')for row in stop)),
            tail_actual_native7=native_stat('actual_native_command'),tail_uncorrected_native7=native_stat('requested_native_command'),
            tail_actual_minus_uncorrected_xyz_mean_m=corrected.mean(axis=0).tolist(),
            tail_zi_integral_mean_m=float(np.mean([entry['integral_after_m']for entry in zi])),
            tail_zi_integral_span_m=float(np.ptp([entry['integral_after_m']for entry in zi])),
            tail_zi_world_correction_mean_m=np.mean([entry['proposed_position_correction_m']for entry in zi],axis=0).tolist(),
            tail_zi_rollback_count=sum(entry['externally_rolled_back']for entry in zi),
            worst_stop_joint_name=tail[-1]['joint_names'][int(np.argmax(case['strict_stop']['joint_span']))],
            worst_stop_joint_target_span_rad=float(np.ptp(np.asarray([row['applied_joint_position_target']for row in tail])[:,int(np.argmax(case['strict_stop']['joint_span']))])),
            terminal_first_actual_native_xyz=stop[0]['actual_native_command'][:3],
            terminal_first_uncorrected_native_xyz=stop[0]['requested_native_command'][:3],
            terminal_first_desired_input=stop[0]['terminal_allocation'].get('desired_body_input'),
            terminal_after2_actual_native_xyz=late[0]['actual_native_command'][:3],
            terminal_after2_uncorrected_native_xyz=late[0]['requested_native_command'][:3],
            terminal_model_final_xy_estimate_error_start_m=estimated[0].tolist(),
            terminal_model_final_xy_estimate_error_after2_m=late[0]['terminal_allocation'].get('model_final_target_error_xy_m'),
            terminal_model_final_xy_estimate_error_last_m=estimated[-1].tolist(),
            terminal_model_final_yaw_error_start_rad=stop[0]['terminal_allocation'].get('model_final_native_yaw_error_rad'),
            terminal_model_final_yaw_error_after2_rad=late[0]['terminal_allocation'].get('model_final_native_yaw_error_rad'),
            actual_uncorrected_tail_xy_minus_operating_preference_m=(np.mean([row['requested_native_command'][:2]for row in tail],axis=0)-[.58,.15]).tolist(),
            tail_root_rpy_mean_rad=np.mean([Rotation.from_matrix(np.asarray(row['T_world_base'])[:3,:3]).as_euler('xyz')for row in tail],axis=0).tolist(),
            fcl_status_counts=dict(collections.Counter(row['fine_collision_evidence']['status']for row in actual)),
            infeasible_details=details))
    for record in records:
        actual=[row for row in rows if row['case']==record['case']]
        late=[row for row in actual if row.get('terminal_native_hold',{}).get('status')!='before_fixed_hold_window' and row.get('terminal_native_hold')]
        held=[row for row in late if row['terminal_native_hold']['active']]
        rejected=[row for row in late if row['terminal_native_hold']['status']not in ['actual_policy_consumed_hold','before_fixed_hold_window']]
        first=rejected[0]if rejected else None
        prior=next((rows[index-1]for index,row in enumerate(rows)if row is first),None)
        tail=held[-26:]
        commands=np.asarray([row['actual_native_command']for row in tail],dtype=np.float32)
        held_q=np.asarray([row['joint_positions']for row in tail])
        record['fixed_hold_diagnostic']=dict(active_samples=len(held),
            status_counts=dict(collections.Counter(row['terminal_native_hold']['status']for row in late)),
            first_cancel_issue_stamp_s=first['stamp_s']-.02 if first else None,
            first_cancel_original_measured_position_error_m=prior['position_error_m']if prior else None,
            first_cancel_original_measured_orientation_error_rad=prior['orientation_error_rad']if prior else None,
            last_active_window_s=tail[-1]['stamp_s']-tail[0]['stamp_s']if tail else None,
            last_active_native7_bit_identical=bool(len(commands)>0 and np.all(commands.view(np.uint32)==commands[0].view(np.uint32))),
            last_active_max_joint_span_rad=float(np.max(np.ptp(held_q,axis=0)))if tail else None,
            last_active_native7_span=np.ptp(commands,axis=0).astype(float).tolist()if tail else None,
            last_active_camera_xyz_span_m=np.ptp(np.asarray([row['T_world_camera_optical']for row in tail])[:,:3,3],axis=0).tolist()if tail else None)
    result=dict(schema='go2_terminal_full_readonly_diagnostic/v1',run=str(RUN.relative_to(ROOT)),
        physical_run_complete=summary['run_complete'],all_pass=summary['all_pass'],records=records,
        scope='Measured old raw records only; model final XY is an approximate preference estimate, not joint/contact/stability evidence. No parameter/Task/log edits.',
        sources_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()for path in
            [Path(__file__),RUN/'summary.json',RUN/'measurements.jsonl',RUN/'timed_tasks.json',RUN/'provenance.json',RUN/'world_z_seed_calibrations.json',RUN/'terminal_native_hold_manifests.json']})
    OUT.mkdir(exist_ok=False);(OUT/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{key:value for key,value in row.items()if key not in ['strict_stop','infeasible_details']}for row in records],indent=2))
    print(OUT/'diagnostic.json')


if __name__=='__main__':main()
