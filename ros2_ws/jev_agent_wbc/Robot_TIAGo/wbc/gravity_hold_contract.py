"""Pure metadata checks for four writes; numerical references come from outside.

This never grants/extends a motion lease or changes public solver feedback.
An offline caller must provide independent Pin gravity for each local write.
Matching a declared clamp mode does not certify actuator implementation: inspect
its frozen source/caps and measured torque separately. No runtime is modified.
"""
import hashlib
import json

import numpy as np

OWNERS=('waist_yaw_joint',)+tuple(f'{s}_{j}_joint'for s in ('left','right')for j in
    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'))
DT=.005


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(value,shape):
    source=np.asarray(value,dtype=object)
    if source.shape!=shape or any(isinstance(x,(bool,np.bool_))or not isinstance(x,(int,float,np.integer,np.floating))for x in source.flat):
        raise ValueError('finite real scalar/vector/matrix required')
    result=np.asarray(value,float)
    if not np.isfinite(result).all():raise ValueError('nonfinite numeric evidence')
    return result


def sample_token(*,physics_state_seq,stamp_s,joint_names,positions_rad,pelvis_R):
    return digest(dict(physics_state_seq=physics_state_seq,stamp_s=float(stamp_s),joint_names=list(joint_names),
        positions_rad=finite(positions_rad,(29,)).tolist(),pelvis_R=finite(pelvis_R,(3,3)).tolist()))


def validate_tick(record,*,control_start_s,first_physics_state_seq,plant_joint_names,model_binding_id,
                  previous_position_target15,motion_lease_report,expected_gravity15=None,
                  expected_initial_q29=None,expected_initial_pelvis_R=None):
    """Return errors; independent expected_gravity15 is four vectors/dicts.

    ``None`` is allowed for nonlocal writes. A local write with no independent
    reference fails this offline audit rather than self-certifying its output.
    Initial q/R can be bound to the preceding physical raw row. Intermediate
    q/R remain recorded SDK inputs whose real source requires frozen runtime
    readback review; this metadata cannot prove sensor provenance by itself.
    """
    errors=[];checked=0;local_count=0
    def reject(code):
        if code not in errors:errors.append(code)
    try:
        start=float(finite(control_start_s,()))
        if type(first_physics_state_seq)is not int or first_physics_state_seq<0:raise ValueError('bad initial physics sequence')
        if (not isinstance(plant_joint_names,(list,tuple))or len(plant_joint_names)!=29
                or any(not isinstance(n,str)or not n for n in plant_joint_names)
                or len(set(plant_joint_names))!=29 or not set(OWNERS)<=set(plant_joint_names)):
            raise ValueError('actual named29 plant order required')
        if not isinstance(model_binding_id,str)or len(model_binding_id)!=64:raise ValueError('run model binding SHA required')
        if (not isinstance(record,dict)or record.get('schema')!='g1_upper_static_gravity_support/v1'
                or record.get('model_binding_id')!=model_binding_id or float(finite(record.get('physics_dt_s'),()))!=DT):
            raise ValueError('schema/model/physics interval mismatch')
        writes=record.get('writes')
        if not isinstance(writes,list)or len(writes)!=4:raise ValueError('all four actual physical writes required')
        previous=finite(previous_position_target15,(15,));nonfresh_seen=False
        lease_checks=(motion_lease_report or {}).get('checks',[])
        for s,write in enumerate(writes):
            stamp=start+s*DT;seq=first_physics_state_seq+s
            if (type(write.get('substep'))is not int or write['substep']!=s
                    or type(write.get('physics_state_seq'))is not int or write['physics_state_seq']!=seq
                    or abs(float(finite(write.get('write_stamp_s'),()))-stamp)>1e-9
                    or abs(float(finite(write.get('measured_stamp_s'),()))-stamp)>1e-9):
                reject('write_or_measurement_not_same_5ms_state')
            if (write.get('trajectory_reused')is not False or
                    write.get('effort_composition')!='existing_explicit_pd_plus_g_single_clamp'):
                reject('expired_motion_or_changed_effort_composition')
            fresh=write.get('motion_ready')
            if type(fresh)is not bool:raise ValueError('per-write motion readiness must be boolean')
            if fresh and nonfresh_seen:reject('motion_resumed_inside_tick_without_new_poll')
            if not fresh:nonfresh_seen=True
            position=finite(write.get('position_target15'),(15,));held=finite(write.get('held_reference15'),(15,))
            velocity=finite(write.get('velocity_target15'),(15,));worker=finite(write.get('worker_gravity15'),(15,))
            local=finite(write.get('local_gravity15'),(15,));applied=finite(write.get('applied_gravity15'),(15,))
            if not np.array_equal(held,previous):reject('held_reference_not_last_actual_write')
            if not np.array_equal(position,position.astype(np.float32).astype(float)):
                reject('position_target_not_actual_float32')
            source=write.get('source')
            if fresh:
                if (source!='worker_motion'or write.get('status')!='worker'or np.any(local!=0)
                        or not np.array_equal(applied,worker)or write.get('computed_gravity15')is not None):
                    reject('fresh_worker_and_local_gravity_sources_mixed')
                if s>=len(lease_checks)or lease_checks[s].get('valid')is not True:
                    reject('worker_gravity_without_original_valid_motion_lease')
                else:
                    check=lease_checks[s];sim=float(finite(check.get('sim_age_s'),()))
                    wall=float(finite(check.get('wall_age_s'),()));horizon=float(finite(check.get('execution_horizon_s'),()))
                    if abs(horizon-DT)>1e-9 or min(sim,wall)<-1e-9 or max(sim,wall)+horizon>.06+1e-9:
                        reject('worker_write_exceeds_original_60ms_motion_lease')
            else:
                if not np.array_equal(position,held)or np.any(velocity!=0)or np.any(worker!=0):
                    reject('nonfresh_write_reused_motion_reference_velocity_or_worker_gravity')
                if source=='zero_fallback':
                    if (write.get('status')!='rejected'or not isinstance(write.get('reason'),str)
                            or not write['reason']or np.any(local!=0)or np.any(applied!=0)
                            or write.get('computed_gravity15')is not None):
                        reject('invalid_input_fallback_not_zero')
                elif source=='local_static_model':
                    local_count+=1
                    if write.get('status')!='computed'or write.get('measurement_source')!='physx_sdk_current_after_update':
                        reject('unverified_local_static_source')
                    if write.get('joint_names')!=list(plant_joint_names):reject('physical_joint_order_mismatch')
                    q=finite(write.get('measured_q29'),(29,));R=finite(write.get('pelvis_R'),(3,3))
                    if (np.max(abs(R.T@R-np.eye(3)))>1e-6 or abs(np.linalg.det(R)-1.)>1e-6):
                        reject('invalid_current_pelvis_rotation')
                    token=sample_token(physics_state_seq=seq,stamp_s=stamp,joint_names=plant_joint_names,
                        positions_rad=q,pelvis_R=R)
                    if write.get('sample_token')!=token:reject('local_gravity_sample_binding_mismatch')
                    if s==0 and expected_initial_q29 is not None and not np.array_equal(q,finite(expected_initial_q29,(29,))):
                        reject('first_sample_differs_from_prior_measured29')
                    if s==0 and expected_initial_pelvis_R is not None and np.max(abs(R-finite(expected_initial_pelvis_R,(3,3))))>1e-6:
                        reject('first_sample_differs_from_prior_measured_pelvis')
                    computed=finite(write.get('computed_gravity15'),(15,))
                    if (not np.array_equal(local,computed.astype(np.float32).astype(float))or
                            not np.array_equal(applied,local)):
                        reject('local_force_was_clipped_summed_or_not_actual_float32')
                    expected=None if expected_gravity15 is None else expected_gravity15[s]
                    if expected is None:reject('independent_current_gravity_reference_missing')
                    else:
                        if isinstance(expected,dict):
                            if set(expected)!=set(OWNERS):raise ValueError('independent reference must cover exact15 owners')
                            expected=[expected[n]for n in OWNERS]
                        reference=finite(expected,(15,));checked+=1
                        if np.max(abs(computed-reference))>1e-8:reject('local_gravity_differs_from_independent_actual_mass_model')
                else:reject('unknown_nonfresh_gravity_source')
            previous=position
    except (KeyError,TypeError,ValueError,IndexError,AttributeError)as exc:
        reject('malformed_evidence:'+str(exc))
    return dict(valid=not errors,errors=errors,local_static_writes=local_count,
        independent_gravity_checks=checked,motion_lease_extended=False,
        scope='Per-write source, held reference, fresh sample and independent gravity consistency only; actuator/sensor provenance, public status and physical safety require their separate frozen-source/measured audits.')
