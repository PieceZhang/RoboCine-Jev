"""One native Pink displacement authorized for four real physical writes.

The request's application box is checked before creating this authorization.
This module neither rebases the frozen QP nor certifies physical braking.
"""
import copy
import hashlib
import json

import numpy as np


class FollowingExecutionRejected(ValueError):
    pass


def _vector(value):
    raw=np.asarray(value,dtype=object)
    if raw.shape!=(15,) or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) for v in raw):
        raise FollowingExecutionRejected('numeric_owned15_without_booleans_required')
    result=np.asarray(value,float)
    if result.shape!=(15,) or not np.isfinite(result).all():
        raise FollowingExecutionRejected('finite_owned15_vector_required')
    return result


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def authorize_native_step(*,task_context,packet_seq,acquisition_stamp_s,application_stamp_s,
                          committed_reference,integrator_before,velocity,proposed_reference,
                          actual_float32_target,antiwindup_active,dt_s=.02):
    """Reject clipping; recognize only the known IEEE float32 embedding.

    ``committed_reference`` is the true start-of-application ledger, already
    accepted by validate_application_manifest, not a planned snapshot.
    """
    if not isinstance(antiwindup_active,bool) or antiwindup_active:
        raise FollowingExecutionRejected('native_reference_clipped')
    if isinstance(dt_s,(bool,np.bool_)) or not np.isfinite(dt_s) or abs(dt_s-.02)>1e-12:
        raise FollowingExecutionRejected('original_native_dt_required')
    context=copy.deepcopy(task_context)
    if not isinstance(context,dict) or not context.get('task_id') or len(context.get('task_sha256',''))!=64:
        raise FollowingExecutionRejected('full_current_task_context_required')
    reference=copy.deepcopy(committed_reference)
    if (reference.get('source')!='physically_committed' or reference.get('task_id')!=context['task_id']
            or reference.get('task_sha256')!=context['task_sha256']):
        raise FollowingExecutionRejected('same_task_committed_reference_required')
    names=reference.get('joint_names')
    if not isinstance(names,list) or len(names)!=15 or len(set(names))!=15:
        raise FollowingExecutionRejected('owned_joint_order_required')
    start=_vector(reference['positions_rad']);before=_vector(integrator_before)
    native=_vector(velocity);proposed=_vector(proposed_reference);target=_vector(actual_float32_target)
    if np.any(np.abs(native)>.5+1e-10):
        raise FollowingExecutionRejected('original_native_speed_cap_required')
    if not np.array_equal(before.astype(np.float32).astype(float),start):
        raise FollowingExecutionRejected('integrator_does_not_match_actual_start_reference')
    expected=before+native*dt_s
    if not np.array_equal(expected,proposed):
        raise FollowingExecutionRejected('native_reference_projected_after_qp')
    if not np.array_equal(expected.astype(np.float32).astype(float),target):
        raise FollowingExecutionRejected('physical_target_not_known_float32_embedding')
    if any(isinstance(v,(bool,np.bool_)) for v in (acquisition_stamp_s,application_stamp_s)):
        raise FollowingExecutionRejected('numeric_acquisition_and_application_stamps_required')
    stamps=np.asarray([acquisition_stamp_s,application_stamp_s],float)
    if not np.isfinite(stamps).all() or stamps[0]<0 or stamps[1]<stamps[0] or stamps[1]+dt_s>stamps[0]+.06+1e-9:
        raise FollowingExecutionRejected('original_lease_must_cover_entire_step')
    if isinstance(packet_seq,bool) or not isinstance(packet_seq,int) or packet_seq<0:
        raise FollowingExecutionRejected('native_reply_seq_required')
    authorization=dict(schema='g1_native_reference_step/v1',task_context=context,packet_seq=packet_seq,
        acquisition_stamp_s=float(stamps[0]),application_stamp_s=float(stamps[1]),dt_s=float(dt_s),
        start_reference=reference,native_velocity_rad_s=native.tolist(),
        integrator_before_rad=before.tolist(),float64_proposal_rad=expected.tolist(),
        authorized_target_rad=target.tolist(),
        endpoint_float32_roundoff_rad=(target-expected).tolist(),
        carried_start_float32_roundoff_rad=(before-start).tolist(),
        native_step_was_clipped=False,physical_braking_certified=False)
    return dict(payload=authorization,sha256=_hash(authorization))


def validate_authorized_write(authorization,*,task_context,current_reference,target_to_write,
                              now_s,horizon_s=.005):
    """Only the two complete start/authorized vectors are permitted.

    No independent per-joint mixing, new projection, unplanned advance or
    return to the old target after the first new write is authorized.
    Original wall/sim packet lease checks remain separately mandatory.
    """
    payload=authorization.get('payload')
    if not isinstance(payload,dict) or authorization.get('sha256')!=_hash(payload):
        raise FollowingExecutionRejected('native_execution_authorization_corrupted')
    if task_context!=payload['task_context']:
        raise FollowingExecutionRejected('execution_cross_task')
    if (any(isinstance(v,(bool,np.bool_)) for v in (now_s,horizon_s)) or not np.isfinite([now_s,horizon_s]).all()
            or horizon_s<0 or now_s<payload['application_stamp_s']-1e-9):
        raise FollowingExecutionRejected('finite_current_execution_stamp_required')
    if now_s+horizon_s>payload['acquisition_stamp_s']+.06+1e-9:
        raise FollowingExecutionRejected('native_execution_lease_expired')
    initial=payload['start_reference'];current=current_reference
    if (current.get('source')!='physically_committed' or current.get('task_id')!=initial['task_id']
            or current.get('task_sha256')!=initial['task_sha256'] or current.get('joint_names')!=initial['joint_names']
            or current.get('commit_seq',-1)<initial['commit_seq']):
        raise FollowingExecutionRejected('current_physical_reference_context_mismatch')
    if (abs(float(current.get('snapshot_stamp_s',-1.))-now_s)>1e-9
            or float(current.get('commit_stamp_s',now_s+1.))>now_s+1e-9):
        raise FollowingExecutionRejected('fresh_actual_reference_snapshot_required')
    start=_vector(initial['positions_rad']);new=_vector(payload['authorized_target_rad'])
    actual=_vector(current['positions_rad']);target=_vector(target_to_write)
    actual_start=np.array_equal(actual,start);actual_new=np.array_equal(actual,new)
    target_start=np.array_equal(target,start);target_new=np.array_equal(target,new)
    if not (actual_start or actual_new) or not (target_start or target_new):
        raise FollowingExecutionRejected('reference_outside_atomic_authorized_vectors')
    if actual_new and not actual_start and target_start:
        raise FollowingExecutionRejected('reference_reversion_after_native_step_write')
    return dict(valid=True,authorization_sha256=authorization['sha256'],stamp_s=float(now_s),
        current_commit_seq=current['commit_seq'],current_reference_token=current['token'],
        current_is_start=bool(actual_start),current_is_authorized_target=bool(actual_new),
        requested_write_is_start=bool(target_start),requested_write_is_authorized_target=bool(target_new))


def validate_expired_reference_hold(*,current_reference,target_to_write,velocity_to_write,gravity_to_write):
    """A lease expiry permits only the current actually committed position.

    This is emergency position holding, not a fresh native displacement and
    not a renewal of an expired QP/reference authorization.
    """
    if current_reference.get('source')!='physically_committed':
        raise FollowingExecutionRejected('real_physical_position_hold_required')
    actual=_vector(current_reference['positions_rad']);target=_vector(target_to_write)
    if (not np.array_equal(actual,target) or np.any(_vector(velocity_to_write)!=0.)
            or np.any(_vector(gravity_to_write)!=0.)):
        raise FollowingExecutionRejected('expiry_must_hold_actual_reference_and_zero_velocity_gravity')
    return dict(fresh_native_motion_authorized=False,held_already_committed_reference=True,
        current_reference_token=current_reference['token'],current_commit_seq=current_reference['commit_seq'])
