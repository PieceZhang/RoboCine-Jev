"""Fresh full-Task binding, bounded measured seed and complete first-tick rollback."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from jev_agent_wbc.wbc_protocol import TimedTask,CameraTrajectory,semantic_camera
from jev_agent_wbc.representation import Pose

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


m=load('bumpless_seed_candidate',ROOT/'Robot_Go2/world_z_bumpless.py')
backend=load('bumpless_seed_backend',ROOT/'Robot_Go2/wbc_backend.py')


def fixture(native_z=.5):
    camera=np.eye(4);camera[:3,3]=[.5,.1,.52];link=np.eye(4);ext=np.eye(4)
    native=np.array([.5,.1,native_z,1,0,0,0],dtype=np.float32)
    pose=semantic_camera(Pose.from_rt(camera[:3,:3],camera[:3,3]))
    task=TimedTask('same_id',5.,'fresh_observation',CameraTrajectory(5.,(0.,1.),(pose,pose)),.05,.15)
    rows=[]
    for stamp in np.arange(4.5,5.001,.02):
        rows.append(dict(stamp_s=float(stamp),actual_native_command=native.astype(float).tolist(),T_world_camera_optical=camera.tolist(),
            task_feedback=dict(task_id='previous_task',feasible=None,constraints={family:dict(status='pass',stamp_s=float(stamp))for family in
                ['configuration','actuation','self_collision','policy_domain']}),fine_collision_evidence=dict(status='pass'),
            native_domain_recovery=dict(active=False),native_camera_parking=dict(active=False),
            actual_native_domain=dict(status='inside_outer_domain_unverified',world_z_feedback=dict(externally_rolled_back=False,native_correction_accepted=True)),
            requested_native_domain=dict(status='inside_outer_domain_unverified')))
    context=dict(now_s=5.,observation_stamp_s=5.,reference_optical=camera.copy(),measured_optical=camera.copy(),
        link0=link,extrinsic=ext,previous_applied_native=native,previous_issue_stamp_s=4.98,history_rows=rows,domain=backend.PolicyDomain())
    return task,context


def feedback(controller,task,context,stamp=5.,seq=250,**updates):
    values=dict(task_id=task.task_id,stamp_s=stamp,now_s=stamp,dt_s=.02,sample_seq=seq,fresh=True,integration_allowed=True,domain=context['domain'])
    values.update(updates)
    return controller.feedback_native_command(backend,context['reference_optical'],context['measured_optical'],context['link0'],context['extrinsic'],**values)


def test_seed_is_fresh_observed_bias_not_old_integral_and_commits_same_first_issue():
    task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True,ki_per_s=.2)
    controller.integral_m=.007;controller._last_stamp_s=4.98;controller._last_sample_seq=249
    decision=controller.admit(backend,task,**context)
    assert decision['eligible'] and abs(decision['seed_clipped_m']+.02)<1e-7
    assert controller.integral_m==0 and controller._last_stamp_s==4.98 and controller._last_sample_seq==249
    native,check=feedback(controller,task,context)
    assert abs(native[2]-.5)<1e-7 and abs(controller.integral_m+.02)<1e-7
    assert decision['transaction_status']=='prepared_waiting_actual_issue'
    assert check['world_z_feedback']['seed_pending_commit']
    assert controller.finish_seed_update(check['world_z_feedback']['update_id'],applied_native=native)
    assert decision['transaction_status']=='committed_to_native_input'
    assert abs(decision['remaining_native_z_jump_m'])<1e-7


@pytest.mark.parametrize('native_z',[.49,.55])
def test_more_than25mm_cannot_force_continuity_or_expand_original_total_cap(native_z):
    task,context=fixture(native_z);controller=m.BumplessWorldZFeedback(enabled=True)
    decision=controller.admit(backend,task,**context);native,check=feedback(controller,task,context)
    assert decision['eligible']and abs(decision['seed_clipped_m'])==.025
    assert abs(decision['expected_remaining_native_z_jump_m'])>.0049
    assert check['world_z_feedback']['proposed_position_correction_norm_m']<=.025+1e-12
    assert abs(native[2]-native_z)>.0049


@pytest.mark.parametrize('problem',['no_prior','stale','future','gap','native_unsteady','ee_unsteady','bias_unsteady','fcl','config','actuation','recovery','outer_override','source_mismatch','old_measurement','ref_mismatch','task_ref_mismatch','bool_time'])
def test_unverified_calibration_is_rejected_reset_zero(problem):
    task,context=fixture()
    if problem=='no_prior':context['previous_applied_native']=None
    elif problem=='stale':context['previous_issue_stamp_s']=4.9
    elif problem=='future':context['previous_issue_stamp_s']=5.01
    elif problem=='gap':context['history_rows'].pop(2)
    elif problem=='native_unsteady':context['history_rows'][0]['actual_native_command'][2]-=.002
    elif problem=='ee_unsteady':context['history_rows'][0]['T_world_camera_optical'][2][3]-=.002
    elif problem=='bias_unsteady':
        context['history_rows'][0]['actual_native_command'][2]-=.0008
        context['history_rows'][0]['T_world_camera_optical'][2][3]+=.0008
    elif problem=='fcl':context['history_rows'][-1]['fine_collision_evidence']['status']='unknown'
    elif problem in ('config','actuation'):context['history_rows'][-1]['task_feedback']['constraints']['configuration'if problem=='config'else'actuation']['status']='unknown'
    elif problem=='recovery':context['history_rows'][-1]['native_domain_recovery']['active']=True
    elif problem=='outer_override':context['history_rows'][-1]['actual_native_domain']['world_z_feedback']['externally_rolled_back']=True
    elif problem=='source_mismatch':context['previous_applied_native'][2]-=.001
    elif problem=='old_measurement':context['observation_stamp_s']=4.98
    elif problem=='ref_mismatch':context['reference_optical'][2,3]+=.001
    elif problem=='task_ref_mismatch':
        data=task.to_dict();data['camera_trajectory']['samples'][0]['pose']['translation_m'][2]+=.001;task=TimedTask.from_dict(data)
    else:context['now_s']=True
    controller=m.BumplessWorldZFeedback(enabled=True);controller.integral_m=.01
    decision=controller.admit(backend,task,**context)
    assert decision['eligible']is False and decision['seed_clipped_m']==0 and controller.integral_m==0


@pytest.mark.parametrize('blocked',['outer','clock','domain','integration','late','changed_matrix'])
def test_first_tick_rejection_includes_seed_in_rollback_and_never_retries(blocked):
    task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True)
    controller._last_stamp_s=4.98;controller._last_sample_seq=249
    decision=controller.admit(backend,task,**context)
    updates={}
    if blocked=='clock':updates['sample_seq']=249
    elif blocked=='integration':updates['integration_allowed']=False
    elif blocked=='late':updates.update(stamp_s=5.02,now_s=5.02,sample_seq=251)
    elif blocked=='changed_matrix':context['link0']=context['link0'].copy();context['link0'][0,3]+=.001
    elif blocked=='domain':
        original=context['domain']
        class RejectSeed:
            def check(self,native):
                result=original.check(native)
                if native[2]<.51:result['status']='out_of_policy_domain'
                return result
        context['domain']=RejectSeed()
    native,check=feedback(controller,task,context,**updates)
    token=check['world_z_feedback']['update_id']
    if blocked=='outer':assert controller.rollback_last_update('later_native_override',update_id=token)
    if blocked=='domain':controller.rollback_last_update('later_fcl_override',update_id=token)
    assert controller.integral_m==0
    assert not controller.finish_seed_update(token,applied_native=native)
    # Fresh recovery consumes normal20ms only; never replays admission seed.
    native2,check2=feedback(controller,task,context,stamp=5.04,seq=252)
    assert abs(controller.integral_m)<1e-10 and not check2['world_z_feedback'].get('seed_pending_commit',False)


@pytest.mark.parametrize('field',['path','tolerance','observation'])
def test_full_task_sha_changes_admission_even_when_task_id_reused(field):
    task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True)
    first=controller.admit(backend,task,**context)
    data=task.to_dict()
    if field=='path':data['camera_trajectory']['samples'][1]['pose']['translation_m'][0]+=.01
    elif field=='tolerance':data['position_tolerance_m']=.04
    else:data['observation_id']='different_observation'
    second=controller.admit(backend,data,**context)
    assert first['task_id']==second['task_id']and first['task_sha256']!=second['task_sha256']
    assert len(controller.calibrations)==2 and controller.integral_m==0


def test_late_outer_override_after_commit_still_reverts_seed_state():
    task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True)
    decision=controller.admit(backend,task,**context);native,check=feedback(controller,task,context)
    token=check['world_z_feedback']['update_id'];assert controller.finish_seed_update(token,applied_native=native)
    assert controller.rollback_last_update('actual_issue_cancelled',update_id=token)
    assert controller.integral_m==0 and decision['transaction_status']=='outer_override_rolled_back_reset_zero'


def test_first_commit_must_match_actual_float32_seeded_native_input():
    task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True)
    decision=controller.admit(backend,task,**context);native,check=feedback(controller,task,context)
    changed=native.copy();changed[2]+=.001
    assert not controller.finish_seed_update(check['world_z_feedback']['update_id'],applied_native=changed)
    assert controller.integral_m==0 and decision['transaction_status']=='outer_override_rolled_back_reset_zero'


def test_driver_policy_consumption_commits_only_after_all_actual_override_guards():
    driver=load('bumpless_finalization_driver',ROOT/'Robot_Go2/adaptive_runner.py')
    for reasons in [[],['native_parking_hold'],['native_domain_recovery'],['collision_freshness_veto'],['original_camera_support_rejected'],['terminal_rejected_forecast_body_brake']]:
        task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True)
        decision=controller.admit(backend,task,**context);native,check=feedback(controller,task,context)
        actual=driver.finalize_native_feedback(controller,check,dict(status='inside_outer_domain_unverified'),reasons)
        assert actual['world_z_feedback']is check['world_z_feedback']
        if reasons:
            assert controller.integral_m==0 and decision['transaction_status']=='outer_override_rolled_back_reset_zero'
            assert not actual['world_z_feedback']['seed_pending_commit']
        else:
            assert decision['transaction_status']=='prepared_waiting_actual_issue'
            consumed=[]
            def invoke():
                assert decision['transaction_status']=='prepared_waiting_actual_issue'
                consumed.append(True)
                return 'learned_policy_result'
            assert driver.consume_policy_with_seed_confirmation(invoke,controller,actual,np.asarray(native,dtype=np.float32))=='learned_policy_result'
            assert consumed==[True]
            assert abs(controller.integral_m+.02)<1e-7 and decision['transaction_status']=='committed_to_native_input'


def test_disabled_runner_path_preserves_original_base_integral_and_finalization():
    current=load('seed_disabled_current_driver',ROOT/'Robot_Go2/adaptive_runner.py')
    original=load('seed_disabled_original_driver',ROOT/'Robot_Go2/results/terminal_pose_full_preparation_03/sources/adaptive_runner.py')
    a=m._base.WorldZFeedback(enabled=True,ki_per_s=.2);b=m._base.WorldZFeedback(enabled=True,ki_per_s=.2)
    task,context=fixture();reference=context['reference_optical'].copy();measured=reference.copy();measured[2,3]+=.01
    for i,reasons in enumerate([[],['native_domain_recovery'],[],['collision_freshness_veto'],[]]):
        args=dict(task_id=task.task_id,stamp_s=5+i*.02,now_s=5+i*.02,dt_s=.02,sample_seq=250+i,fresh=True,integration_allowed=True,domain=context['domain'])
        na,ca=a.feedback_native_command(backend,reference,measured,context['link0'],context['extrinsic'],**args)
        nb,cb=b.feedback_native_command(backend,reference,measured,context['link0'],context['extrinsic'],**args)
        old=original.finalize_native_feedback(a,ca,dict(status='inside_outer_domain_unverified'),reasons)
        new=current.finalize_native_feedback(b,cb,dict(status='inside_outer_domain_unverified'),reasons)
        np.testing.assert_array_equal(na,nb)
        assert old==new and a.integral_m==b.integral_m


def test_same_task_sha_with_old_new_admission_measurement_cannot_reuse_commit():
    task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True)
    first=controller.admit(backend,task,**context);native,check=feedback(controller,task,context)
    controller.finish_seed_update(check['world_z_feedback']['update_id'],applied_native=native)
    stale=copy.deepcopy(context);stale['now_s']=5.1;stale['observation_stamp_s']=5.1
    rejected=controller.admit(backend,task,**stale)
    assert rejected is not first and rejected['task_sha256']==first['task_sha256']
    assert not rejected['eligible']and controller.integral_m==0
    assert rejected['calibration_id']!=first['calibration_id']


def test_unconfirmed_first_seed_is_rolled_back_before_next_measurement_and_cannot_finish_late():
    task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True)
    decision=controller.admit(backend,task,**context);native,check=feedback(controller,task,context)
    token=check['world_z_feedback']['update_id']
    native2,check2=feedback(controller,task,context,stamp=5.02,seq=251)
    assert abs(native2[2]-.52)<1e-9 and controller.integral_m==0
    assert not check2['world_z_feedback'].get('seed_pending_commit',False)
    assert not check['world_z_feedback']['seed_pending_commit']
    assert decision['transaction_status']=='outer_override_rolled_back_reset_zero'
    assert not controller.finish_seed_update(token,applied_native=native)


def test_policy_consumption_exception_rolls_back_unissued_seed_and_propagates_run_failure():
    driver=load('bumpless_policy_exception_driver',ROOT/'Robot_Go2/adaptive_runner.py')
    task,context=fixture();controller=m.BumplessWorldZFeedback(enabled=True)
    decision=controller.admit(backend,task,**context);native,check=feedback(controller,task,context)
    actual=driver.finalize_native_feedback(controller,check,dict(status='inside_outer_domain_unverified'),[])
    def failed_policy():raise RuntimeError('actual learned policy failed')
    with pytest.raises(RuntimeError,match='actual learned policy failed'):
        driver.consume_policy_with_seed_confirmation(failed_policy,controller,actual,np.asarray(native,dtype=np.float32))
    assert controller.integral_m==0 and not check['world_z_feedback']['seed_pending_commit']
    assert decision['transaction_status']=='outer_override_rolled_back_reset_zero'
    assert decision['rollback_reason']=='seed_policy_consumption_failed'
