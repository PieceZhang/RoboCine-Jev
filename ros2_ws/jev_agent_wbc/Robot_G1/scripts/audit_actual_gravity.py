#!/usr/bin/env python3
"""Calibrate actual native masses/COMs and audit a separate gravity-only model.

No physics execution, inverse dynamics, collision or IK model is changed here.
The independent reference is potential-energy differentiation, with additional
hand-written URDF-FK validation performed by a separate audit.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    robot=Path(__file__).resolve().parents[1];urdf=robot/'kinematics/g1_29dof.urdf'
    dumpfile=robot/'results/default_upper_yaw_01/startup_physx_mass_properties.json'
    trace=robot/'results/adaptive_ground_sameprofile_01/measurements.jsonl'
    calibrationfile=robot/'results/adaptive_ground_sameprofile_01/pink_collision_initialization.json'
    source=robot/'gravity_model.py';spec=importlib.util.spec_from_file_location('actual_gravity',source)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    dump=json.loads(dumpfile.read_text());metadata=helper.calibrate(urdf,dump,digest(dumpfile))
    model=helper.ActualGravity(urdf,metadata)
    metadatafile=args.output_dir/'actual_gravity_model.json'
    metadatafile.write_text(json.dumps(metadata,indent=2)+'\n')
    offsets=json.loads(calibrationfile.read_text())['physical_body_frame_offsets']
    offset_error=max(float(np.max(np.abs(np.asarray(offset)-metadata['bodies'][name]['T_pin_body_from_actor'])))
                     for name,offset in offsets.items())
    rows=[json.loads(line) for line in trace.open()];poses=[]
    for case in sorted({r['case'] for r in rows if r['case']!='settle'}):
        subset=[r for r in rows if r['case']==case]
        for fraction in (0.,.25,.5,.75,1.):
            row=subset[round(fraction*(len(subset)-1))]
            poses.append((dict(zip(row['joint_names'],row['joint_positions'])),np.asarray(row['T_world_base']),row['case'],row['stamp_s']))
    poses.append((metadata['calibration_joint_positions'],np.asarray(metadata['T_world_pelvis']),'calibration',0.))
    upper=[name for name in list(model.model.names)[1:] if name=='waist_yaw_joint'
           or any(token in name for token in ('shoulder_','elbow_joint','wrist_'))]
    epsilon=1e-6;fd=[]
    for joints,pelvis,case,stamp in poses:
        torque=model.torque(joints,pelvis);errors={}
        for name in upper:
            plus=dict(joints);minus=dict(joints);plus[name]+=epsilon;minus[name]-=epsilon
            numerical=(model.potential(plus,pelvis)-model.potential(minus,pelvis))/(2*epsilon)
            errors[name]=dict(actual_model_gravity_nm=torque[name],potential_derivative_nm=numerical,
                             absolute_error_nm=abs(torque[name]-numerical))
        fd.append(dict(case=case,stamp_s=stamp,joints=errors))
    potential_error=max(value['absolute_error_nm'] for row in fd for value in row['joints'].values())
    original=pin.buildModelFromUrdf(str(urdf));originaldata=original.createData()
    indexed={round(r['stamp_s']*50):r for r in rows};deltas=[]
    for row in rows:
        if row['phase']!='stop' or not row.get('upper_gravity_feedforward'):continue
        acquisition=row.get('pink_runtime_recovery',{}).get('acquisition_stamp_s')
        measured=indexed.get(round(acquisition*50)) if acquisition is not None else None
        if measured is None:continue
        joints=dict(zip(measured['joint_names'],measured['joint_positions']));pelvis=np.asarray(measured['T_world_base'])
        original.gravity.linear=pelvis[:3,:3].T@np.array([0.,0.,-9.81])
        old=pin.computeGeneralizedGravity(original,originaldata,helper.configuration(original,joints))
        new=model.torque(joints,pelvis);entries={}
        for name in upper:
            index=original.joints[original.getJointId(name)].idx_v
            entries[name]=dict(old_gravity_nm=float(old[index]),actual_model_gravity_nm=new[name],
                old_minus_actual_nm=float(old[index]-new[name]),recorded_old_gravity_nm=row['upper_gravity_feedforward'][name])
        deltas.append(dict(case=row['case'],stamp_s=row['stamp_s'],joints=entries))
    reports={}
    for case in sorted({row['case'] for row in deltas}):
        subset=[r for r in deltas if r['case']==case];end=subset[-1]['stamp_s']
        tail=[r for r in subset if r['stamp_s']>=end-.5-1e-9]
        reports[case]={n:{key:float(np.mean([r['joints'][n][key] for r in tail])) for key in tail[0]['joints'][n]} for n in upper}
    reconstruction=max(abs(v['old_gravity_nm']-v['recorded_old_gravity_nm']) for r in deltas for v in r['joints'].values())
    total=sum(float(i.mass) for i in model.model.inertias)
    result=dict(schema='g1_actual_gravity_cpu_audit/v1',physical_execution=False,runtime_gravity_changed=False,
        passed=bool(potential_error<2e-6 and offset_error<2e-5 and reconstruction<1e-10
                    and abs(total-metadata['total_actual_mass_kg'])<1e-7),
        point_mass_model_total_kg=total,actual_total_kg=metadata['total_actual_mass_kg'],
        original_urdf_total_kg=float(sum(i.mass for i in original.inertias)),
        matched_previous_calibration_body_count=len(offsets),previous_body_offset_max_component_error=offset_error,
        finite_difference_pose_count=len(poses),finite_difference_joint_count=len(upper),finite_difference_epsilon_rad=epsilon,
        maximum_potential_derivative_error_nm=potential_error,old_recorded_gravity_reconstruction_max_error_nm=reconstruction,
        final_half_second_means=reports,
        warning='This CPU audit validates gravity accounting and transformations only. It does not establish physical tracking, stopping, or native policy yaw response.',
        source_sha256={str(p):digest(p) for p in [source,Path(__file__),urdf,dumpfile,trace,calibrationfile,metadatafile]})
    (args.output_dir/'finite_differences.json').write_text(json.dumps(fd,indent=2)+'\n')
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_sha256','final_half_second_means')},indent=2))
    for case,report in reports.items():print(case,'left wrist pitch',report['left_wrist_pitch_joint'])
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
