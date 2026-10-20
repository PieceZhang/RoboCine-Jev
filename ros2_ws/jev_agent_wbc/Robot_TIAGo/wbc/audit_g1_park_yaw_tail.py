#!/usr/bin/env python3
"""Read-only fixed .5s G1 tail comparison; no causal or contact certificate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--native-dump',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    frozen=a.output/'sources';frozen.mkdir()
    reference_path=Path(__file__).with_name('audit_g1_actual_gravity.py')
    for source,name in [(Path(__file__),Path(__file__).name),(reference_path,reference_path.name),
                        (a.native_dump,'startup_physx_mass_properties.json'),
                        (a.baseline/'sources/g1_29dof.urdf','g1_29dof.urdf')]:
        shutil.copy2(source,frozen/name)
    spec=importlib.util.spec_from_file_location('independent_upper_com_xml',frozen/reference_path.name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ref=module.XmlReference(frozen/'g1_29dof.urdf',json.loads(a.native_dump.read_text()))
    upper=[]
    for name in ref.names:
        n=name
        while n!='pelvis':
            if ref.parent[n]['name']=='waist_yaw_joint':upper.append(name);break
            n=ref.parent[n]['parent']
    masses=dict(zip(ref.names,ref.masses));upper_mass=sum(masses[n]for n in upper)
    records=[];input_hashes={};frozen_tails=[]
    for label,run in [('baseline',a.baseline),('candidate',a.candidate)]:
        raw=run/'measurements.jsonl'
        rows=[json.loads(line)for line in raw.open()]
        for source in [raw,run/'summary.json',run/'provenance.json']:
            input_hashes[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
        for case in dict.fromkeys(row['case']for row in rows if row['case']!='settle'):
            case_rows=[row for row in rows if row['case']==case]
            end=case_rows[-1]['stamp_s']
            tail=[row for row in case_rows if row['stamp_s']>=end-.5-1e-9]
            assert len(tail)==26 and abs(tail[-1]['stamp_s']-tail[0]['stamp_s']-.5)<1e-9
            frozen_tails.extend(dict(run_label=label,original_row=row)for row in tail)
            names=tail[0]['joint_names'];q=np.asarray([r['joint_positions']for r in tail]);
            target=np.asarray([r['applied_joint_position_target']for r in tail])
            qd=np.asarray([r['measured_joint_velocity']for r in tail])
            tau=np.asarray([r['applied_joint_effort_estimate']for r in tail])
            states=[r['parking_transition']['mode']for r in tail]
            zero=[np.array_equal(r['applied_policy_velocity'],[0.,0.,0.])for r in tail]
            base_yaw=np.unwrap([np.arctan2(r['T_world_base'][1][0],r['T_world_base'][0][0])for r in tail])
            com=[]
            for row in tail:
                frames=ref.frames(dict(zip(names,row['joint_positions'])),np.eye(4))
                com.append(sum(masses[n]*(frames[n][1]+frames[n][0]@ref.centers_link[n])for n in upper)/upper_mass)
            com=np.asarray(com)
            selected_joints={}
            for name in ['waist_yaw_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint',
                         'right_shoulder_yaw_joint','left_hip_yaw_joint','right_hip_yaw_joint']:
                i=names.index(name);v=[r['pink_upper_velocity'].get(name,0.)for r in tail]
                selected_joints[name]=dict(actual_delta_rad=float(q[-1,i]-q[0,i]),actual_span_rad=float(np.ptp(q[:,i])),
                    drive_target_delta_rad=float(target[-1,i]-target[0,i]),
                    native_Pink_mean_qdot_rad_s=float(np.mean(v)),measured_mean_qdot_rad_s=float(qd[:,i].mean()),
                    target_lead_mean_rad=float((target[:,i]-q[:,i]).mean()),
                    target_lead_max_abs_rad=float(abs(target[:,i]-q[:,i]).max()),
                    applied_torque_mean_Nm=float(tau[:,i].mean()))
            action=np.asarray([r['native_policy_action']for r in tail])
            record=dict(run=label,directory=str(run.resolve()),case=case,tail_start_s=tail[0]['stamp_s'],tail_end_s=end,
                sample_count=len(tail),fixed_window_s=.5,parking_state_counts={v:states.count(v)for v in set(states)},
                literal_zero_input_samples=sum(zero),whole_window_is_PARK_zero=all(zero)and all(v=='PARK'for v in states),
                actual_base_yaw_delta_rad=float(base_yaw[-1]-base_yaw[0]),
                measured_body_yaw_rate_mean_rad_s=float(np.mean([r['measured_body_velocity'][2]for r in tail])),
                measured_world_z_angular_rate_mean_rad_s=float(np.mean([r['base_angular_velocity_world'][2]for r in tail])),
                actual_lower_hip_yaw_mean_delta_rad=float(np.mean([q[-1,names.index(n)]-q[0,names.index(n)]for n in
                    ['left_hip_yaw_joint','right_hip_yaw_joint']])),
                selected_joints=selected_joints,
                native_policy_action_max_span=float(np.ptp(action,axis=0).max()),
                estimated_upper_COM_pelvis_frame_mean_m=com.mean(axis=0).tolist(),
                estimated_upper_COM_pelvis_frame_delta_m=(com[-1]-com[0]).tolist(),
                estimated_upper_COM_pelvis_frame_axis_span_m=np.ptp(com,axis=0).tolist(),
                maximum_camera_position_error_m=max(r['position_error_m']for r in tail),
                maximum_camera_orientation_error_rad=max(r['orientation_error_rad']for r in tail))
            records.append(record)
    (a.output/'frozen_original_tails.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n'for r in frozen_tails))
    report=dict(schema='g1_park_yaw_tail_independent/v1',physical_execution=False,
        window_selection='All 26 samples in the fixed final .5s of every case; no favorable conditional subwindow.',
        upper_body_names=upper,actual_calibrated_upper_mass_kg=float(upper_mass),
        COM_method='Independent XML FK plus actual native 30-body mass/actor-COM calibration; measured joints. Estimate, not live COM sensor or dynamics certification.',
        records=records,
        conclusions=[
            'Candidate shot/yaw_right are full PARK/zero windows but still have positive pelvis yaw and both hip-yaw joints moving in the opposite direction. This is kinematic consistency, not a proof of motor/contact causality.',
            'Right shoulder pitch follows nonzero native Pink reference in those windows; target/actual lead is small. Stronger arm tracking alone does not remove the remaining requested motion.',
            'The calibrated upper COM moves by submillimetres in all fixed windows. Static COM position alone cannot establish yaw torque, contact slip or internal angular momentum.',
            'Left and candidate yaw_left windows include non-PARK states. Do not interpret their whole-window result as a clean zero-input parking experiment.',
            'Baseline and candidate have several changed controller gains and different trajectories/state histories; this is not a single-variable Kp counterfactual. Contact forces, foot slip and complete inertial dynamics are not logged/certified here.'
        ],
        input_sha256=input_hashes,
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in frozen.iterdir()if p.is_file()})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(records),actual_upper_mass_kg=float(upper_mass),output=str(a.output)),indent=2))


if __name__=='__main__':main()
