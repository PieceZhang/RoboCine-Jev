#!/usr/bin/env python3
"""Sparse original-Task/body lag and arm-compensation geometry, read-only CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def rz(angle):return Rotation.from_euler('z',angle).as_matrix()


def transform(pose):
    result=np.eye(4);result[:3,:3]=pose.rotation;result[:3,3]=pose.translation_m;return result


def rotation_distance(a,b):return float(Rotation.from_matrix(a@b.T).magnitude())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--native-dump',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=a.runs[0].resolve().parents[2];sys.path.insert(0,str(root))
    from jev_agent_wbc.wbc_protocol import TimedTask,optical_camera
    frozen=a.output/'sources';frozen.mkdir()
    shutil.copy2(__file__,frozen/Path(__file__).name)
    shutil.copy2(a.native_dump,frozen/a.native_dump.name)
    defaults=json.loads(a.native_dump.read_text())
    home=dict(zip(defaults['joint_names'],defaults['source_default_joint_positions']['values'][0]))
    right=[n for n in home if n.startswith('right_')and any(s in n for s in ('shoulder','elbow','wrist'))]
    assert len(right)==7 and all(abs(home[n])<1e-12 for n in right)
    records=[];sparse=[];hashes={};source_checks=[]
    for run in a.runs:
        tasks_data=json.loads((run/'timed_tasks.json').read_text())
        tasks=[TimedTask.from_dict(data)for data in tasks_data]
        summary=json.loads((run/'summary.json').read_text())
        assert len(tasks)==len(summary['cases'])
        wanted={round(stamp,9)for task in tasks for stamp in
            (task.trajectory.start_s,task.trajectory.start_s+1.,task.trajectory.end_s,task.trajectory.end_s+5.)}
        selected={};velocity_tails={task.task_id:[]for task in tasks}
        for line in (run/'measurements.jsonl').open():
            row=json.loads(line);stamp=row['stamp_s']
            if round(stamp,9)in wanted:selected[round(stamp,9)]=row
            for task in tasks:
                if task.trajectory.end_s-.5-1e-9<=stamp<=task.trajectory.end_s+1e-9:
                    velocity_tails[task.task_id].append(row['measured_body_velocity'])
        author=(run/'sources/adaptive_runner.py').read_text()
        assert 'heading=math.atan2(root_reference[1,0],root_reference[0,0])'in author
        assert "displacement=Rotation.from_euler('z',heading).as_matrix()@np.r_[velocity[:2]*t,0.]"in author
        assert 'T[:3,:3]=yaw_rotation@reference[:3,:3]'in author
        source_checks.append(dict(run=run.name,original_horizontal_yaw_authoring_verified=True))
        for source in (run/'timed_tasks.json',run/'summary.json',run/'measurements.jsonl',run/'sources/adaptive_runner.py'):
            hashes[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
        for task,data,case_summary in zip(tasks,tasks_data,summary['cases']):
            start=task.trajectory.start_s;end=task.trajectory.end_s;duration=end-start
            before=selected[round(start,9)];early=selected[round(start+1.,9)]
            last=selected[round(end,9)];stop=selected[round(end+5.,9)]
            observation_sha=hashlib.sha256(json.dumps(before['measured_observation'],sort_keys=True,
                separators=(',',':'),allow_nan=False).encode()).hexdigest()
            assert observation_sha==task.observation_id
            velocity=np.asarray(last['requested_body_velocity'],float)
            assert np.array_equal(velocity,np.asarray(case_summary['requested_vx_vy_wz']))
            B0=np.asarray(before['T_world_base']);C0=np.asarray(before['T_world_camera_optical'])
            heading=np.arctan2(B0[1,0],B0[0,0]);R_heading=rz(heading)
            L0=np.linalg.inv(B0)@C0
            C_start=transform(optical_camera(task.trajectory.sample(start)))
            C_ref=transform(optical_camera(task.trajectory.sample(end)))
            assert np.allclose(C_start,C0,atol=1e-9,rtol=0)
            assert np.allclose(C_ref,np.asarray(last['T_world_reference_optical']),atol=1e-9,rtol=0)
            def nominal(t):
                T=np.eye(4);T[:3,:3]=rz(velocity[2]*t)@B0[:3,:3]
                T[:3,3]=B0[:3,3]+R_heading@np.r_[velocity[:2]*t,0.]
                return T
            Bnom=nominal(duration);camera_reconstructed=Bnom@L0
            source_error=float(abs(camera_reconstructed-C_ref).max())
            assert source_error<1e-8,'Original endpoint must reconstruct from nominal base/admission camera'
            actual_B=np.asarray(last['T_world_base']);actual_C=np.asarray(last['T_world_camera_optical'])
            stop_B=np.asarray(stop['T_world_base']);stop_C=np.asarray(stop['T_world_camera_optical'])
            still_initial_arm=actual_B@L0
            compensation_required=C_ref[:3,3]-still_initial_arm[:3,3]
            compensation_actual=actual_C[:3,3]-still_initial_arm[:3,3]
            decomposition_error=compensation_actual-compensation_required-(actual_C[:3,3]-C_ref[:3,3])
            assert np.linalg.norm(decomposition_error)<1e-12
            B_early=np.asarray(early['T_world_base']);Bnom_early=nominal(1.)
            error_heading=R_heading.T@(actual_B[:3,3]-Bnom[:3,3])
            early_error_heading=R_heading.T@(B_early[:3,3]-Bnom_early[:3,3])
            planar_speed=float(np.linalg.norm(velocity[:2]))
            direction=velocity[:2]/planar_speed if planar_speed>0 else np.zeros(2)
            def yaw_error(actual,reference):
                difference=np.arctan2(actual[1,0],actual[0,0])-np.arctan2(reference[1,0],reference[0,0])
                return float(np.arctan2(np.sin(difference),np.cos(difference)))
            # Exact additive position decomposition at a specified order:
            # nominal -> actual horizontal translation -> actual yaw -> actual
            # remaining roll/pitch/height. Components need not be orthogonal.
            yaw_only=Bnom.copy();yaw_only[:3,:3]=rz(yaw_error(actual_B,Bnom))@Bnom[:3,:3]
            translation_compensation=np.r_[Bnom[:2,3]-actual_B[:2,3],0.]
            yaw_compensation=C_ref[:3,3]-(yaw_only@L0)[:3,3]
            other_compensation=compensation_required-translation_compensation-yaw_compensation
            stop_compensation=C_ref[:3,3]-(stop_B@L0)[:3,3]
            posture={}
            for label,row in [('admission',before),('plus1s',early),('move_end',last),('stop_end',stop)]:
                q=dict(zip(row['joint_names'],row['joint_positions']))
                posture[label]=dict(right7_norm_from_native_default_rad=float(np.linalg.norm([q[n]-home[n]for n in right])),
                    right7_joint_positions={n:q[n]for n in right},waist_yaw_rad=q['waist_yaw_joint'])
                sparse.append(dict(run=run.name,task_id=task.task_id,checkpoint=label,original_row=row))
            relative_required=np.linalg.inv(actual_B)@C_ref;relative_actual=np.linalg.inv(actual_B)@actual_C
            record=dict(run=run.name,case=case_summary['case'],task_id=task.task_id,
                task_sha256=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                absolute_start_s=start,absolute_move_end_s=end,duration_s=duration,requested_body_velocity=velocity.tolist(),
                original_endpoint_reconstruction_max_abs=source_error,
                actual_body_minus_nominal_in_admission_heading_m=error_heading.tolist(),
                plus1s_actual_body_minus_nominal_in_admission_heading_m=early_error_heading.tolist(),
                along_requested_translation_error_m=float(error_heading[:2]@direction)if planar_speed>0 else None,
                plus1s_along_requested_translation_error_m=float(early_error_heading[:2]@direction)if planar_speed>0 else None,
                yaw_error_at_move_end_rad=yaw_error(actual_B,Bnom),plus1s_yaw_error_rad=yaw_error(B_early,Bnom_early),
                full_body_orientation_error_rad=rotation_distance(actual_B[:3,:3],Bnom[:3,:3]),
                last_half_second_body_velocity_mean=np.mean(velocity_tails[task.task_id],axis=0).tolist(),
                world_camera_error_m=float(np.linalg.norm(actual_C[:3,3]-C_ref[:3,3])),
                world_camera_orientation_error_rad=rotation_distance(actual_C[:3,:3],C_ref[:3,:3]),
                admission_body_to_camera=L0.tolist(),actual_body_to_camera=relative_actual.tolist(),
                camera_if_admission_arm_pose_kept=still_initial_arm.tolist(),
                required_world_camera_compensation_m=compensation_required.tolist(),
                required_compensation_horizontal_translation_component_m=translation_compensation.tolist(),
                required_compensation_yaw_component_m=yaw_compensation.tolist(),
                required_compensation_remaining_height_tilt_component_m=other_compensation.tolist(),
                actual_world_camera_compensation_m=compensation_actual.tolist(),
                required_compensation_norm_m=float(np.linalg.norm(compensation_required)),
                actual_compensation_norm_m=float(np.linalg.norm(compensation_actual)),
                compensation_position_error_m=float(np.linalg.norm(compensation_actual-compensation_required)),
                required_body_relative_camera_translation_change_m=(relative_required[:3,3]-L0[:3,3]).tolist(),
                actual_body_relative_camera_translation_change_m=(relative_actual[:3,3]-L0[:3,3]).tolist(),
                required_body_relative_camera_rotation_change_rad=rotation_distance(relative_required[:3,:3],L0[:3,:3]),
                actual_body_relative_camera_rotation_change_rad=rotation_distance(relative_actual[:3,:3],L0[:3,:3]),
                stop_body_minus_nominal_end_in_admission_heading_m=(R_heading.T@(stop_B[:3,3]-Bnom[:3,3])).tolist(),
                stop_yaw_error_relative_nominal_end_rad=yaw_error(stop_B,Bnom),
                body_height_change_move_to_stop_m=float(stop_B[2,3]-actual_B[2,3]),
                stop_world_camera_error_m=float(np.linalg.norm(stop_C[:3,3]-C_ref[:3,3])),
                stop_world_camera_orientation_error_rad=rotation_distance(stop_C[:3,:3],C_ref[:3,:3]),
                stop_required_world_camera_compensation_m=stop_compensation.tolist(),
                stop_required_compensation_norm_m=float(np.linalg.norm(stop_compensation)),
                measured_posture_checkpoints=posture)
            records.append(record)
    (a.output/'sparse_original_rows.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n'for v in sparse))
    report=dict(schema='g1_task_endpoint_lag_independent/v1',physical_execution=False,
        scope='Sparse original admission/+1s/move-end/stop-end measured endpoints. Exact camera-position compensation decomposition is geometry, not causal actuation or policy-training support proof.',
        original_world_tasks_and_absolute_times_unchanged=True,source_checks=source_checks,
        home_reference='Actual native source_default_joint_positions from original30body mass diagnostic, matched by joint name. Right7 defaults all zero.',
        records=records,input_sha256=hashes,
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in frozen.iterdir()if p.is_file()})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(records=len(records),output=str(a.output)),indent=2))
    for r in records:
        q=r['measured_posture_checkpoints']
        print(json.dumps(dict(run=r['run'],case=r['case'],lag_m=r['along_requested_translation_error_m'],
            yaw_error_rad=r['yaw_error_at_move_end_rad'],compensation_required_m=r['required_compensation_norm_m'],
            actual_camera_error_m=r['world_camera_error_m'],
            right7_start_move_stop_norm=[q[k]['right7_norm_from_native_default_rad']for k in ('admission','move_end','stop_end')],
            right_elbow_start_move_stop=[q[k]['right7_joint_positions']['right_elbow_joint']for k in ('admission','move_end','stop_end')]),allow_nan=False))


if __name__=='__main__':main()
