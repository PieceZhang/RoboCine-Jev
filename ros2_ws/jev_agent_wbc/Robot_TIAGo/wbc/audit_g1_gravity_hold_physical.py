#!/usr/bin/env python3
"""Read-only per-write static-gravity audit with independent genuine Pin values.

The frozen runtime's NumPy gravity implementation is never imported here. This
audits saved acquisitions/commands; source readback establishes their SDK origin,
but offline consistency does not independently authenticate a sensor or certify
the physical plant's stability.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();started=time.perf_counter()
    a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'sources';frozen.mkdir()
    inputs=['gravity_hold_initialization.json','actual_gravity_verification.json',
            'actuator_audit.json','arm_PD_preflight.json','provenance.json','summary.json']
    for name in inputs:shutil.copy2(a.run/name,frozen/name)
    for name in ['gravity_model.py','gravity_hold.py','g1_29dof.urdf',
                 'actual_gravity_model.json','adaptive_runner.py']:
        shutil.copy2(a.run/'sources'/name,frozen/name)
    for source in [Path(__file__),Path(__file__).with_name('gravity_hold_contract.py')]:
        shutil.copy2(source,frozen/source.name)
    contract=load('independent_write_metadata_contract',frozen/'gravity_hold_contract.py')
    pin_model=load('frozen_actual_mass_pin_reference',frozen/'gravity_model.py')
    metadata=json.loads((frozen/'actual_gravity_model.json').read_text())
    model=pin_model.ActualGravity(frozen/'g1_29dof.urdf',metadata)
    initialization=json.loads((frozen/'gravity_hold_initialization.json').read_text())
    binding=initialization['model_binding'];plant=binding['plant_joint_names'];owners=list(contract.OWNERS)
    ids=[plant.index(n)for n in owners];errors=[]
    def reject(where,reason):errors.append(dict(where=where,reason=reason))
    binding_checks={
        'canonical_owner_order':binding['canonical_upper_joint_names']==owners,
        'model_binding_digest':contract.digest(binding)==initialization['model_binding_id'],
        'gravity_source_hash':sha(frozen/'gravity_hold.py')==binding['gravity_hold_module_sha256'],
        'metadata_file_hash':sha(frozen/'actual_gravity_model.json')==binding['actual_metadata_file_sha256'],
        'urdf_hash':sha(frozen/'g1_29dof.urdf')==binding['source_urdf_sha256'],
        'native_dump_hash':metadata['native_dump_sha256']==binding['native_dump_sha256'],
        'native_startup_mass_com_fk_verified':json.loads((frozen/'actual_gravity_verification.json').read_text())['passed']is True,
        'native_explicit_pd_preflight':json.loads((frozen/'arm_PD_preflight.json').read_text())['passed']is True,
    }
    provenance=json.loads((frozen/'provenance.json').read_text())
    for name in ['gravity_model.py','gravity_hold.py','g1_29dof.urdf',
                 'actual_gravity_model.json','adaptive_runner.py']:
        binding_checks['provenance_'+name]=sha(frozen/name)==provenance['frozen_sha256'][name]
    for name,ok in binding_checks.items():
        if not ok:reject('initialization',name)
    # Source-order evidence is read-only and explicitly scoped. The numerical
    # model is independently Pin, not the frozen NumPy provider under audit.
    source=(frozen/'adaptive_runner.py').read_text()
    begin=source.index('for substep in range(4):')
    loop=source[begin:source.index('measurement_begin=',begin)]
    source_checks={
        'motion_lease_checked_before_local_model':loop.index('execution_lease_status')<loop.index('if gravity_hold_model is not None:'),
        'q_is_current_sdk_copy':'measured_q=robot.data.joint_pos[0].numpy().copy()'in loop,
        'pelvis_is_actor_pose_current_sdk_copy':'pelvis_pose=robot.data.body_link_pose_w[0,gravity_pelvis_body_id].numpy().copy()'in loop,
        'pelvis_wxyz_to_xyzw_conversion':'Rotation.from_quat(pelvis_pose[[4,5,6,3]])'in loop,
        'current_model_before_effort_write':loop.index('current_state_gravity_support(')<loop.index('robot.set_joint_effort_target'),
        'reference_ledger_commits_after_physical_write':loop.index('robot.write_data_to_sim()')<loop.index('last_upper_written=target[0,gravity_hold_ids].tolist()'),
        'physics_advance_then_sdk_update':'sim.step(render=False);robot.update(physics_dt)'in loop,
        'original_explicit_pd_model':'from isaaclab.actuators import IdealPDActuatorCfg'in source,
        'reference_mismatch_restored_before_rejection':loop.index('target[0,gravity_hold_ids]=torch.tensor(last_upper_written')<loop.index("raise ValueError('uncommitted upper reference refused"),
    }
    for name,ok in source_checks.items():
        if not ok:reject('frozen_source',name)
    previous=np.asarray(initialization['initial_upper_position_target15'],float)
    previous_row=None;sources=Counter();statuses=Counter();tick_count=0
    local_samples=[];maximum=0.;first_sample_q_checks=0;first_sample_R_checks=0
    row_errors=[];local_active=0;local_idle=0;mixed_ticks=0;max_f32_delta=0.
    with(a.run/'measurements.jsonl').open()as stream:
        for i,line in enumerate(stream):
            row=json.loads(line);tick_count+=1;stamp=float(row['stamp_s']);start=i*.02
            where=dict(index=i,stamp_s=stamp,case=row['case'],phase=row['phase'])
            if abs(stamp-(i+1)*.02)>1e-9:reject(where,'noncontiguous_control_clock')
            if row['joint_names']!=plant:reject(where,'row_physical29_order_mismatch')
            writes=row['upper_static_gravity_support']['writes'];expected=[None]*4
            fresh_flags=[]
            for s,write in enumerate(writes):
                sources[write['source']]+=1;fresh_flags.append(write['motion_ready'])
                if write['source']=='local_static_model':
                    if row.get('task_feedback')is None:local_idle+=1
                    else:local_active+=1
                    q=dict(zip(write['joint_names'],write['measured_q29']))
                    T=np.eye(4);T[:3,:3]=write['pelvis_R']
                    torque=model.torque(q,T);expected[s]=[torque[n]for n in owners]
                    delta=float(np.max(abs(np.asarray(write['computed_gravity15'])-expected[s])))
                    maximum=max(maximum,delta)
                    f32_delta=float(np.max(abs(np.asarray(write['applied_gravity15'])-np.asarray(expected[s],dtype=np.float32).astype(float))))
                    max_f32_delta=max(max_f32_delta,f32_delta)
                    local_samples.append(dict(**where,substep=s,physics_state_seq=write['physics_state_seq'],
                        sample_token=write['sample_token'],pin_torque15=expected[s],
                        computed_torque_max_error_Nm=delta,applied_float32_error_Nm=f32_delta))
            if any(fresh_flags)and not all(fresh_flags):mixed_ticks+=1
            q0=None;R0=None
            if previous_row is not None and writes[0]['source']=='local_static_model':
                q0=previous_row['joint_positions'];R0=np.asarray(previous_row['T_world_base'])[:3,:3]
                first_sample_q_checks+=1;first_sample_R_checks+=1
            check=contract.validate_tick(row['upper_static_gravity_support'],control_start_s=start,
                first_physics_state_seq=i*4,plant_joint_names=plant,
                model_binding_id=initialization['model_binding_id'],
                previous_position_target15=previous,motion_lease_report=row.get('upper_execution_lease'),
                expected_gravity15=expected,expected_initial_q29=q0,expected_initial_pelvis_R=R0)
            if not check['valid']:row_errors.append(dict(**where,errors=check['errors']))
            final=np.asarray(writes[-1]['position_target15'])
            if not np.array_equal(final,np.asarray(row['applied_joint_position_target'])[ids]):
                reject(where,'final_actual_write_differs_from_row_applied_position_target')
            feedback=row.get('task_feedback');statuses[row['controller_status']]+=1
            if not all(fresh_flags)and feedback is not None and feedback.get('reason')=='accepted':
                reject(where,'nonfresh_static_support_misreported_as_accepted_motion')
            previous=final;previous_row=row
    report=dict(schema='g1_fresh_static_gravity_physical_contract_audit/v1',
        passed=not errors and not row_errors,errors=errors,row_errors=row_errors,
        measured_control_rows=tick_count,physical_writes=sum(sources.values()),write_sources=dict(sources),
        local_static_active_writes=local_active,local_static_idle_writes=local_idle,
        genuine_pin_comparisons=len(local_samples)*15,maximum_computed_torque_error_Nm=maximum,
        maximum_applied_float32_error_Nm=max_f32_delta,
        prior_measured_row_q29_crosschecks=first_sample_q_checks,
        prior_measured_row_actor_R_crosschecks=first_sample_R_checks,mixed_lease_expiry_ticks=mixed_ticks,
        binding_checks=binding_checks,frozen_source_checks=source_checks,
        controller_status_counts=dict(statuses),local_sample_references=local_samples,
        run_path=str(a.run.resolve()),input_sha256={name:sha(a.run/name)for name in inputs+['measurements.jsonl']},
        frozen_sha256={p.name:sha(p)for p in frozen.iterdir()if p.is_file()},
        physical_summary_all_pass=json.loads((frozen/'summary.json').read_text())['all_pass'],
        public_feedback_reclassified=False,motion_lease_extended=False,
        scope='Source, clock, per-write held reference, independent current-state gravity and source binding. No stability, full-task acceptance or environment collision certification; intermediate SDK samples have frozen source/readback provenance, not an independent second physical sensor.',
        audit_wall_seconds=time.perf_counter()-started)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('local_sample_references','input_sha256','frozen_sha256','controller_status_counts')},indent=2))


if __name__=='__main__':main()
