"""Legged backend ownership, real Pink execution, and exact mixed-frame cameras."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


go2 = module('go2_native_backend', ROOT/'Robot_Go2/wbc_backend.py')
g1 = module('g1_pink_backend', ROOT/'Robot_G1/wbc_backend.py')
servo = module('legged_velocity_servo', ROOT/'Robot_G1/velocity_servo.py')
coordinator = module('go2_camera_coordinator', ROOT/'Robot_Go2/camera_coordinator.py')
velocity_audit = module('go2_velocity_audit', ROOT/'Robot_Go2/velocity_audit.py')
parking = module('g1_parking_supervisor', ROOT/'Robot_G1/parking_supervisor.py')
contact_materials = module('legged_contact_materials',ROOT/'Robot_G1/contact_materials.py')
pink_profile = module('g1_pink_profile',ROOT/'Robot_G1/pink_profile.py')
gravity_only = module('g1_actual_gravity_only',ROOT/'Robot_G1/gravity_model.py')
solver_feedback = module('g1_solver_feedback',ROOT/'Robot_G1/solver_feedback.py')
following_execution = module('g1_following_execution',ROOT/'Robot_G1/following_execution.py')
stand_yaw_assist = module('g1_stand_yaw_assist',ROOT/'Robot_G1/stand_yaw_assist.py')


def park_posture_refresh_fixture():
    owners=['waist_yaw_joint']+arm_pd_readback_fixture()['joint_names']
    measured={name:.01*i for i,name in enumerate(owners)}
    context=dict(schema='g1_solver_task_context/v1',task_id='current_accepted_task',
        camera_path_id='b'*64,task_sha256='a'*64)
    return dict(enabled=True,entered=True,already_refreshed=False,owner_names=owners,
        measured_joints=measured,stamp_s=8.24,task_context=context)


def test_g1_park_posture_refresh_captures_actual_once_per_Task_and_preserves_default():
    args=park_posture_refresh_fixture();result=pink_profile.first_park_posture_refresh(**args)
    assert result['targets']==args['measured_joints'] and result['stamp_s']==8.24
    assert result['task_context']==args['task_context'] and result['refresh_count']==1
    assert result['source']=='measured_physx_joint_pos_at_original_PARK_admission'
    assert not result['hard_joint_lock'] and result['controlled_upper_dofs']==15
    args['measured_joints']['right_elbow_joint']+=.3;args['task_context']['task_id']='next'
    assert result['targets']['right_elbow_joint']!=args['measured_joints']['right_elbow_joint']
    assert result['task_context']['task_id']=='current_accepted_task'
    for change in (dict(enabled=False),dict(entered=False),dict(already_refreshed=True)):
        # The old default and later admissions do not consume new q/preferences.
        assert pink_profile.first_park_posture_refresh(**(args|change)) is None
    args['already_refreshed']=False
    again=pink_profile.first_park_posture_refresh(**args)
    assert again['task_context']['task_id']=='next'


@pytest.mark.parametrize('change',[dict(stamp_s=True),dict(stamp_s=float('nan')),dict(owner_names=['bad']*15),
    dict(measured_joints={}),dict(task_context=dict(schema='invalid')),dict(entered=1)])
def test_g1_park_posture_refresh_rejects_unbound_or_unmeasured_capture(change):
    with pytest.raises(ValueError):pink_profile.first_park_posture_refresh(**(park_posture_refresh_fixture()|change))


def test_g1_park_posture_refresh_remains_a_genuine_soft_native_Pink_task_with_original_costs():
    pytest.importorskip('pinocchio');pytest.importorskip('pink')
    import yaml
    io=yaml.safe_load((ROOT/'Robot_G1/config/height_policy_io.yaml').read_text())
    names=io['articulations']['robot']['joint_names'];q=dict(zip(names,io['articulations']['robot']['default_joint_pos']))
    q['waist_yaw_joint']=.1
    solver=g1.PinkUpperBody(io['actions'][0]['joint_names'],names,np.eye(4),
        posture_home={n:0. for n in names},posture_cost=.03,damping_cost=.02,waist_posture_cost=.15)
    context=park_posture_refresh_fixture()['task_context']
    capture=pink_profile.first_park_posture_refresh(enabled=True,entered=True,already_refreshed=False,
        owner_names=solver.owners['pink'],measured_joints=q,stamp_s=1.,task_context=context)
    target=solver.predicted_camera(q,np.eye(4));target[0,3]+=.02
    velocity=solver.velocity(q,np.eye(4),target,.02,posture_reference=capture['targets'])
    assert set(velocity)==set(capture['targets']) and len(velocity)==15
    assert max(abs(v) for v in velocity.values())<=.02000001
    assert any(abs(v)>1e-4 for v in velocity.values())
    report=solver.last_posture_report
    assert report['waist_yaw_target_rad']==0. and report['other_posture_cost']==.03 and report['damping_cost']==.02
    assert report['camera_frame_task_unchanged'] and report['collision_barriers_unchanged'] and not report['waist_yaw_locked']


@pytest.mark.parametrize('moving,yaw,gain',[(True,.12,2.),(True,-.12,2.),(True,0.,1.),(False,.12,1.),(False,-.12,1.),(False,0.,1.)])
def test_g1_symmetric2_yaw_feedforward_only_changes_original_moving_yaw(moving,yaw,gain):
    profile='yaw_phase_symmetric2_v1';controller=servo.VelocityServo(**pink_profile.velocity_servo_overrides(profile))
    declared=pink_profile.phase_velocity_servo_overrides(profile,moving,yaw)
    assert declared['feedforward_gain']==declared['negative_feedforward_gain']==(1.,1.,gain)
    assert 'kp' not in declared and 'ki' not in declared
    controller.feedforward_gain=np.asarray(declared['feedforward_gain'])
    controller.negative_feedforward_gain=np.asarray(declared['negative_feedforward_gain'])
    np.testing.assert_array_equal(controller.ki,[1.2,1.2,1.])
    np.testing.assert_array_equal(controller.integral_limit,[.2,.2,.3])
    previous=np.zeros(3)
    for _ in range(50):
        command=controller.update([.1,-.1,yaw],[0.,0.,0.],.02)
        assert np.max(np.abs(command-previous)/np.array([.5,.5,.8]))<=.02000000001
        assert np.all(np.abs(command)<=[.5,.5,1.])
        assert np.all(np.abs(controller.integral)<=[.2,.2,.3])
        previous=command


def test_g1_symmetric2_yaw_feedforward_reverts_stop_and_keeps_anchor_feedback_unscaled():
    profile='yaw_phase_symmetric2_v1';current=servo.VelocityServo(**pink_profile.velocity_servo_overrides(profile))
    baseline=servo.VelocityServo()
    for c in (current,baseline):c.reset()
    schedule=pink_profile.phase_velocity_servo_overrides(profile,False,0.)
    current.feedforward_gain=np.asarray(schedule['feedforward_gain'])
    current.negative_feedforward_gain=np.asarray(schedule['negative_feedforward_gain'])
    for _ in range(20):
        a=current.update([0.,0.,0.],[.02,-.01,.02],.02,pose_error_body=[.01,-.01,-.025])
        b=baseline.update([0.,0.,0.],[.02,-.01,.02],.02,pose_error_body=[.01,-.01,-.025])
        np.testing.assert_array_equal(a,b)


def test_g1_symmetric2_actual_runner_native_factory_and_phase_start_stop_configuration():
    runner=module('g1_actual_velocity_factory_test',ROOT/'Robot_G1/adaptive_runner.py')
    selected=runner.native_g1_velocity_parameters(servo,'yaw_phase_symmetric2_v1')
    assert selected==servo.g1_profile('baseline')
    selected.update(pink_profile.velocity_servo_overrides('yaw_phase_symmetric2_v1'))
    actual=servo.VelocityServo(limits=(.5,.5,1.),**selected)
    for moving,requested,gain in [(True,.12,2.),(False,0.,1.),(True,-.12,2.),(False,0.,1.)]:
        phase=pink_profile.phase_velocity_servo_overrides('yaw_phase_symmetric2_v1',moving,requested)
        actual.feedforward_gain=np.array(phase['feedforward_gain'])
        actual.negative_feedforward_gain=np.array(phase['negative_feedforward_gain'])
        np.testing.assert_array_equal(actual.feedforward_gain,[1.,1.,gain])
        np.testing.assert_array_equal(actual.negative_feedforward_gain,[1.,1.,gain])
        np.testing.assert_array_equal(actual.ki,[1.2,1.2,1.])
        np.testing.assert_array_equal(actual.kp,[1.,1.,.8])
        actual.update([0.,0.,requested],[0.,0.,0.],.02)
        actual.reset()
    assert runner.native_g1_velocity_parameters(servo,'low_gain')==servo.g1_profile('low_gain')


@pytest.mark.parametrize('moving,yaw,ki,ff',[(True,.12,4.,2.),(True,-.12,4.,2.),
    (True,0.,1.,1.),(False,.12,1.,1.),(False,-.12,1.,1.),(False,0.,1.,1.)])
def test_g1_moving_yaw_ki4_actual_factory_keeps_original_bounds_and_restores_nonmoving_phase(moving,yaw,ki,ff):
    runner=module('g1_actual_ki4_factory_test',ROOT/'Robot_G1/adaptive_runner.py')
    profile='yaw_phase_symmetric2_ki4_v1'
    parameters=runner.native_g1_velocity_parameters(servo,profile)
    assert parameters==servo.g1_profile('baseline')
    parameters.update(pink_profile.velocity_servo_overrides(profile))
    controller=servo.VelocityServo(limits=(.5,.5,1.),**parameters)
    phase=pink_profile.phase_velocity_servo_overrides(profile,moving,yaw)
    controller.feedforward_gain=np.asarray(phase['feedforward_gain'])
    controller.negative_feedforward_gain=np.asarray(phase['negative_feedforward_gain'])
    controller.ki=np.asarray(phase['ki'])
    np.testing.assert_array_equal(controller.ki,[1.2,1.2,ki])
    np.testing.assert_array_equal(controller.feedforward_gain,[1.,1.,ff])
    np.testing.assert_array_equal(controller.negative_feedforward_gain,[1.,1.,ff])
    np.testing.assert_array_equal(controller.kp,[1.,1.,.8])
    np.testing.assert_array_equal(controller.integral_limit,[.2,.2,.3])
    np.testing.assert_array_equal(controller.slew,[.5,.5,.8])
    last=np.zeros(3)
    for _ in range(250):
        value=controller.update([0.,0.,yaw],[0.,0.,0.],.02)
        assert max(abs(value-last)/controller.slew)<=.02000000001
        assert np.all(abs(value)<=[.5,.5,1.]) and np.all(abs(controller.integral)<=[.2,.2,.3])
        last=value
    # The real stop boundary already resets I. Moving-yaw gain cannot amplify
    # the original stand/anchor correction after the phase changes.
    stop=pink_profile.phase_velocity_servo_overrides(profile,False,yaw)
    controller.ki=np.array(stop['ki']);controller.feedforward_gain=np.array(stop['feedforward_gain'])
    controller.negative_feedforward_gain=np.array(stop['negative_feedforward_gain']);controller.reset()
    baseline=servo.VelocityServo();baseline.reset()
    np.testing.assert_array_equal(controller.integral,[0.,0.,0.])
    np.testing.assert_array_equal(controller.update([0.,0.,0.],[.001,-.001,.01],.02,pose_error_body=[0.,0.,.02]),
        baseline.update([0.,0.,0.],[.001,-.001,.01],.02,pose_error_body=[0.,0.,.02]))


def test_g1_stand_yaw_assist_preserves_brake_settle_and_original_input_arrays():
    command=np.array([.1,-.1,.2]);previous=command.copy();before=command.copy()
    for mode in ('BRAKE','SETTLE'):
        result,report=stand_yaw_assist.command(command,previous,mode=mode,
            anchor_yaw_error_rad=-.02,measured_yaw_rate_rad_s=.025,dt_s=.02)
        np.testing.assert_array_equal(result,before)
        assert not report['active'] and not report['learned_joint_actions_modified']
    np.testing.assert_array_equal(command,before);np.testing.assert_array_equal(previous,before)


def test_g1_stand_yaw_assist_opposes_measured_drift_without_integrator_or_added_command():
    previous=np.zeros(3)
    for _ in range(20):
        result,report=stand_yaw_assist.command(np.zeros(3),previous,mode='PARK',
            anchor_yaw_error_rad=-.02,measured_yaw_rate_rad_s=.025,dt_s=.02)
        assert np.all(result[:2]==0.) and abs(result[2])<=.08
        assert np.max(np.abs(result-previous)/np.array([.5,.5,.8]))<=.02000000001
        assert not report['integrator_used'] and not report['physical_stopping_verified']
        previous=result
    assert result[2]==pytest.approx(-.05)
    # Instantaneous measured reversal, with the same target, remains slew-bound.
    result,_=stand_yaw_assist.command(np.zeros(3),previous,mode='PARK',
        anchor_yaw_error_rad=.04,measured_yaw_rate_rad_s=-.15,dt_s=.02)
    assert result[2]-previous[2]==pytest.approx(.016)


def test_g1_stand_yaw_assist_retains_original_supervisor_exit_and_never_changes_50mrad():
    supervisor=parking.ParkingSupervisor();supervisor.mode='PARK';supervisor.anchor=np.eye(4)
    body=np.eye(4);body[:3,:3]=Rotation.from_euler('z',.05001).as_matrix()
    native,transition=supervisor.update(1.,body,[0.,0.,.02],[0.,0.,-.04],[0.,0.,-.03],.02)
    assert transition['exited'] and transition['mode']=='BRAKE'
    result,report=stand_yaw_assist.command(native,[0.,0.,-.03],mode=transition['mode'],
        anchor_yaw_error_rad=transition['anchor_error_body'][2],measured_yaw_rate_rad_s=.02,dt_s=.02)
    np.testing.assert_array_equal(result,native);assert not report['active']
    for error in (-1.,1.):
        previous=np.zeros(3)
        for _ in range(10):
            previous,report=stand_yaw_assist.command(np.zeros(3),previous,mode='PARK',
                anchor_yaw_error_rad=error,measured_yaw_rate_rad_s=-error,dt_s=.02)
        assert abs(previous[2])==pytest.approx(.08)


@pytest.mark.parametrize('changes',[dict(anchor_yaw_error_rad=True),dict(measured_yaw_rate_rad_s=float('nan')),
    dict(dt_s=0.),dict(mode='WALK'),dict(previous_applied=[0.,0.,.2]),dict(previous_applied=[.1,0.,0.])])
def test_g1_stand_yaw_assist_rejects_invalid_state_or_impossible_cap_slew(changes):
    kwargs=dict(parking_command=np.zeros(3),previous_applied=np.zeros(3),mode='PARK',
        anchor_yaw_error_rad=0.,measured_yaw_rate_rad_s=0.,dt_s=.02)
    with pytest.raises(ValueError):stand_yaw_assist.command(**(kwargs|changes))


def arm_pd_readback_fixture():
    names=[side+'_'+joint+'_joint' for side in ('left','right') for joint in
        ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
    kp=[90.,60.,20.,60.,4.,4.,4.]*2;kd=[2.,1.,.4,1.,.2,.2,.2]*2
    caps=[25.,25.,25.,25.,25.,5.,5.]*2;speed=[37.,37.,37.,37.,37.,22.,22.]*2
    return dict(joint_names=names,stiffness_scale=2.,source_stiffness=kp,source_damping=kd,
        source_armature=[.03]*14,source_effort_caps=caps,source_velocity_caps=speed,
        actual_stiffness=np.asarray(kp)*2,actual_damping=np.asarray(kd)*5,actual_armature=[.03]*14,
        actual_effort_caps=caps,actual_velocity_caps=speed,engine_stiffness=[0.]*14,engine_damping=[0.]*14,
        upper_damping_scale=5.,physics_dt_s=.005)


def test_g1_arm_PD_preflight_uses_actual_armature_and_separates_explicit_from_engine_gains():
    data=arm_pd_readback_fixture();report=pink_profile.arm_pd_preflight(**data)
    assert report['passed'] and report['coupled_stability_verified'] is False
    assert report['actual_braking_certified'] is False
    assert min(report['actual_armature_minus_scalar_threshold_kg_m2'])==pytest.approx(.003875)
    assert max(report['scalar_necessary_inertia_threshold_kg_m2'])==pytest.approx(.026125)
    for changes,reason in [({'actual_armature':[.02]*14},'actual_armature_readback_mismatch'),
            ({'engine_stiffness':[1.]*14},'engine_stiffness_readback_mismatch'),
            ({'engine_damping':[1.]*14},'engine_damping_readback_mismatch'),
            ({'actual_effort_caps':[100.]*14},'actual_effort_caps_readback_mismatch'),
            ({'actual_velocity_caps':[100.]*14},'actual_velocity_caps_readback_mismatch')]:
        failed=pink_profile.arm_pd_preflight(**(data|changes))
        assert failed['passed'] is False and reason in failed['failures']


def test_g1_arm_PD_preflight_rejects_low_inertia_even_if_configuration_matches():
    data=arm_pd_readback_fixture()
    failed=pink_profile.arm_pd_preflight(**(data|dict(source_armature=[.001]*14,actual_armature=[.001]*14)))
    assert not failed['passed'] and failed['failures']==['scalar_armature_stability_screen_failed']
    # This gate cannot infer an actual full-body inertia from its scalar screen.
    assert not failed['coupled_stability_verified']


@pytest.mark.parametrize('changes',[dict(actual_armature=[float('nan')]*14),
    dict(actual_stiffness=[True]*14),dict(physics_dt_s=.01),dict(stiffness_scale=True),
    dict(joint_names=['waist_yaw_joint']*14)])
def test_g1_arm_PD_preflight_rejects_invalid_or_unrelated_native_parameters(changes):
    with pytest.raises(ValueError):pink_profile.arm_pd_preflight(**(arm_pd_readback_fixture()|changes))


def following_execution_fixture():
    helper=module('g1_following_reference_snapshots',ROOT/'Robot_G1/following_reserve.py')
    names=[f'owned{i}' for i in range(15)]
    context=dict(task_id='current',task_sha256='a'*64,camera_path_id='b'*64,schema='solver_context/v1')
    before=np.linspace(-2.3,2.3,15);start=before.astype(np.float32).astype(float)
    reference=helper.committed_reference_snapshot(task_id='current',task_sha256='a'*64,joint_names=names,
        positions_rad=start,commit_seq=5,commit_stamp_s=1.035,snapshot_stamp_s=1.04)
    velocity=np.linspace(-.5,.5,15);proposal=before+velocity*.02;target=proposal.astype(np.float32).astype(float)
    kwargs=dict(task_context=context,packet_seq=8,acquisition_stamp_s=1.,application_stamp_s=1.04,
        committed_reference=reference,integrator_before=before,velocity=velocity,proposed_reference=proposal,
        actual_float32_target=target,antiwindup_active=False)
    authorization=following_execution.authorize_native_step(**kwargs)
    return helper,context,reference,target,kwargs,authorization


def test_g1_following_execution_accepts_only_known_float32_embedding_and_keeps_ownership():
    _,context,reference,target,kwargs,authorization=following_execution_fixture()
    assert authorization['payload']['native_step_was_clipped'] is False
    assert np.max(np.abs(authorization['payload']['endpoint_float32_roundoff_rad']))>0
    check=following_execution.validate_authorized_write(authorization,task_context=context,
        current_reference=reference,target_to_write=target,now_s=1.04)
    assert check['current_is_start'] and check['requested_write_is_authorized_target']
    for change in (dict(antiwindup_active=True),dict(proposed_reference=np.asarray(kwargs['proposed_reference'])+.000001),
                   dict(actual_float32_target=np.asarray(target)+.000001),dict(velocity=np.full(15,.50001)),
                   dict(velocity=[True]*15),dict(acquisition_stamp_s=True),dict(dt_s=True)):
        with pytest.raises(following_execution.FollowingExecutionRejected):
            following_execution.authorize_native_step(**(kwargs|change))


def test_g1_following_execution_allows_its_own_one_step_but_rejects_partial_or_additional_advance():
    helper,context,reference,target,_,authorization=following_execution_fixture()
    current=helper.committed_reference_snapshot(task_id='current',task_sha256='a'*64,
        joint_names=reference['joint_names'],positions_rad=target,commit_seq=6,commit_stamp_s=1.04,snapshot_stamp_s=1.045)
    check=following_execution.validate_authorized_write(authorization,task_context=context,
        current_reference=current,target_to_write=target,now_s=1.045)
    assert check['current_is_authorized_target']
    for malformed in (np.asarray(target)+.001,np.r_[target[:7],reference['positions_rad'][7:]]):
        with pytest.raises(following_execution.FollowingExecutionRejected):
            following_execution.validate_authorized_write(authorization,task_context=context,
                current_reference=current,target_to_write=malformed,now_s=1.045)
    with pytest.raises(following_execution.FollowingExecutionRejected,match='reversion'):
        following_execution.validate_authorized_write(authorization,task_context=context,
            current_reference=current,target_to_write=reference['positions_rad'],now_s=1.045)
    with pytest.raises(following_execution.FollowingExecutionRejected,match='cross_task'):
        following_execution.validate_authorized_write(authorization,task_context=context|dict(task_sha256='c'*64),
            current_reference=current,target_to_write=target,now_s=1.045)


def test_g1_following_execution_retains_original_whole_step_and_substep_expiry():
    _,context,reference,target,kwargs,authorization=following_execution_fixture()
    for stamp in (1.04,1.045,1.05,1.055):
        following_execution.validate_authorized_write(authorization,task_context=context,
            current_reference=reference|dict(snapshot_stamp_s=stamp),target_to_write=target,now_s=stamp)
    following_execution.validate_authorized_write(authorization,task_context=context,
        current_reference=reference|dict(snapshot_stamp_s=1.06),target_to_write=target,now_s=1.06,horizon_s=0.)
    with pytest.raises(following_execution.FollowingExecutionRejected,match='expired'):
        following_execution.validate_authorized_write(authorization,task_context=context,
            current_reference=reference,target_to_write=target,now_s=1.06)
    with pytest.raises(following_execution.FollowingExecutionRejected,match='entire_step'):
        following_execution.authorize_native_step(**(kwargs|dict(application_stamp_s=1.06)))


def test_g1_following_expiry_holds_actual_start_or_written_target_without_renewing_motion():
    helper,context,reference,target,_,authorization=following_execution_fixture()
    for position,sequence in ((reference['positions_rad'],5),(target,6)):
        current=helper.committed_reference_snapshot(task_id='current',task_sha256='a'*64,
            joint_names=reference['joint_names'],positions_rad=position,commit_seq=sequence,
            commit_stamp_s=1.04,snapshot_stamp_s=1.061)
        with pytest.raises(following_execution.FollowingExecutionRejected,match='expired'):
            following_execution.validate_authorized_write(authorization,task_context=context,
                current_reference=current,target_to_write=target,now_s=1.061)
        result=following_execution.validate_expired_reference_hold(current_reference=current,
            target_to_write=position,velocity_to_write=np.zeros(15),gravity_to_write=np.zeros(15))
        assert not result['fresh_native_motion_authorized'] and result['held_already_committed_reference']
        with pytest.raises(following_execution.FollowingExecutionRejected):
            following_execution.validate_expired_reference_hold(current_reference=current,
                target_to_write=np.asarray(position)+.001,velocity_to_write=np.zeros(15),gravity_to_write=np.zeros(15))


def test_g1_terminal_collision_abort_cannot_be_hidden_by_passing_measured_prefix():
    runner=module('g1_native_summary',ROOT/'Robot_G1/adaptive_runner.py')
    result=dict(run_complete=False,all_pass=True,physical_tracking_and_stop_pass=True,
        fine_collision=dict(active_samples=1259,all_active_samples_fresh_pass=True),
        known_constraint_failure_free=True,task_infeasible_samples=0)
    abort=dict(stamp_s=27.18,collision=dict(status='fail',minimum_clearance_m=.004991),
        public_task_status=dict(reason='self_collision',feasible=False))
    runner.include_terminal_abort(result,abort,'a'*64)
    assert not result['all_pass'] and not result['physical_tracking_and_stop_pass']
    assert not result['known_constraint_failure_free'] and result['constraint_abort_detected']
    assert result['fine_collision']['completed_active_samples_fresh_pass']
    assert not result['fine_collision']['all_active_samples_fresh_pass']
    assert result['fine_collision']['active_samples']==1259 and result['task_infeasible_samples']==0
    assert result['task_infeasible_terminal_aborts']==1
    assert result['final_public_task_status']==abort['public_task_status']
    assert result['constraint_abort_evidence']['sha256']=='a'*64


@pytest.mark.parametrize('complete',[False,True])
def test_g1_complete_collision_gate_distinguishes_prefix_from_completed_run(complete):
    runner=module('g1_native_summary_normal',ROOT/'Robot_G1/adaptive_runner.py')
    result=dict(run_complete=complete,all_pass=complete,physical_tracking_and_stop_pass=complete,
        fine_collision=dict(active_samples=500,all_active_samples_fresh_pass=True),
        known_constraint_failure_free=True,task_infeasible_samples=0)
    runner.include_terminal_abort(result)
    assert result['fine_collision']['completed_active_samples_fresh_pass']
    assert result['fine_collision']['all_active_samples_fresh_pass']==complete
    assert result['known_constraint_failure_free'] and not result['constraint_abort_detected']
    assert result['terminal_constraint_abort'] is None and result['constraint_abort_evidence'] is None


def test_g1_public_solver_feedback_requires_current_valid_fresh_packet_and_preserves_uncertainty():
    accepted=dict(ready=True,ok=True,status='fresh_command')
    assert solver_feedback.public_support(accepted)==dict(reason='accepted',feasible=None)
    for status in ('idle','waiting_hold_upper_targets','stale_or_failed_result_dropped',
                   'superseded_task_hold_upper_targets'):
        assert solver_feedback.public_support(dict(ready=False,status=status))==dict(reason='command_stale',feasible=None)
    for status in ('worker_deadline_exceeded_hold_upper_targets','recovery_deadline_exceeded_hold_upper_targets'):
        assert solver_feedback.public_support(dict(ready=False,status=status))==dict(reason='deadline_exceeded',feasible=False)
    for packet in (dict(ready=False,status='stale_or_failed_result_dropped',error='QP failed'),
                   dict(ready=False,status='reply_context_mismatch'),dict(ready=False,failure_kind='transport_failure')):
        assert solver_feedback.public_support(packet)==dict(reason='solver_failure',feasible=False)
    assert solver_feedback.public_support(accepted,dict(status='fail'))==dict(reason='self_collision',feasible=False)
    assert solver_feedback.public_support(None)==dict(reason='command_stale',feasible=None)
    assert solver_feedback.public_support(dict(ready=False,status='reference_step_clipped_hold_upper_targets'))==dict(
        reason='configuration_limit',feasible=False)


def test_g1_moving_yaw_integral_schedule_keeps_original_stop_gains_input_limits_and_ownership():
    controller=servo.VelocityServo(**pink_profile.velocity_servo_overrides('yaw_pi08_2_ff1_v1'))
    for moving,request,expected_ki in ((True,.12,2.),(True,-.12,2.),(False,.12,1.),(True,0.,1.)):
        schedule=pink_profile.phase_velocity_servo_overrides('yaw_pi08_2_ff1_v1',moving,request)
        assert schedule['kp']==(1.,1.,.8) and schedule['ki']==(1.2,1.2,expected_ki)
        assert schedule['feedforward_gain']==schedule['negative_feedforward_gain']==(1.,1.,1.)
    np.testing.assert_array_equal(controller.slew,[.5,.5,.8])
    np.testing.assert_array_equal(controller.integral_limit,[.2,.2,.3])
    np.testing.assert_array_equal(controller.limits,[.5,.5,1.])


def test_genuine_pink_infeasible_qp_retains_exact_problem_without_retry_or_weaker_limits():
    pytest.importorskip('pinocchio');pytest.importorskip('pink')
    from scipy.optimize import linprog
    _,dump,metadata=actual_gravity_fixture();names=dump['joint_names']
    legs=[n for n in names if any(t in n for t in ('hip_','knee_','ankle_'))]
    solver=g1.PinkUpperBody(legs,names,np.eye(4),posture_cost=.03,damping_cost=.02)
    q=metadata['calibration_joint_positions'];pelvis=np.asarray(metadata['T_world_pelvis'])
    camera=solver.predicted_camera(q,pelvis)
    # A deliberately corrupted previous command makes the original .5 speed
    # and 1 rad/s² acceleration boxes incompatible. It must fail, not execute.
    solver.previous['left_shoulder_pitch_joint']=.8
    with pytest.raises(AssertionError):solver.velocity(q,pelvis,camera,.02)
    report=solver.last_qp_failure;problem=report['qp']
    assert report['qp_capture']=='native_pink_exception_frame'
    assert report['reconstructed_after_solver_failure'] is False and not report['execution_retried']
    assert solver.previous['left_shoulder_pitch_joint']==.8
    result=linprog(np.zeros(15),A_ub=problem['G'],b_ub=problem['h'],bounds=[(None,None)]*15,method='highs')
    assert not result.success and result.status==2
    assert sum(g['row_count'] for g in report['inequality_groups'])==len(problem['h'])


@pytest.mark.parametrize('result_name,old_gain,new_gain',[
    ('adaptive_actual_gravity_qpdiag_01',.1,.5),('adaptive_actual_gravity_gain_terminal_yaw_01',.5,1.)])
def test_genuine_pink_class_k_rate_solves_exact_measured_failure_without_changing_distance_or_native_boxes(result_name,old_gain,new_gain):
    pytest.importorskip('pinocchio');pytest.importorskip('pink')
    import yaml
    robot=ROOT/'Robot_G1';result=robot/'results'/result_name
    row=json.loads((result/'exact_qp_failures.json').read_text())[0]['pink_runtime_recovery']
    failure=row['qp_failure'];request=row['failed_request']
    dump=json.loads((robot/'results/default_upper_yaw_01/startup_physx_mass_properties.json').read_text())
    metadata=json.loads((result/'fine_collision_geometry.json').read_text())
    io=yaml.safe_load((robot/'config/height_policy_io.yaml').read_text());names=io['articulations']['robot']['joint_names']
    home=dict(zip(names,io['articulations']['robot']['default_joint_pos']))
    camera=json.loads((result/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    actors=np.asarray(dump['actor_world_poses_xyzw']['values'])[0]
    transforms=np.broadcast_to(np.eye(4),(len(actors),4,4)).copy()
    transforms[:,:3,:3]=Rotation.from_quat(actors[:,3:]).as_matrix();transforms[:,:3,3]=actors[:,:3]
    bodies=dict(zip(dump['body_names'],transforms));initial=dict(zip(dump['joint_names'],dump['joint_positions']['values'][0]))
    for gain,damping in ((old_gain,None),(new_gain,None),(new_gain,.5)):
        solver=g1.PinkUpperBody(io['actions'][0]['joint_names'],names,camera,
            posture_home=home,posture_cost=.03,damping_cost=.02,waist_posture_cost=.15,collision_gain=gain)
        solver.configure_collisions(metadata,initial,bodies['pelvis'],bodies)
        solver.previous.update(zip(failure['joint_names'],failure['previous_owned_velocity']))
        options={k:request.get(k) for k in ('posture_reference','posture_cost_override','damping_cost_override')}
        if damping is not None:options['damping_cost_override']=damping
        if gain==old_gain:
            with pytest.raises(AssertionError):solver.velocity(request['joints'],request['T_world_pelvis'],request['target'],request['dt'],**options)
            for name in ('P','q','G','h'):
                np.testing.assert_array_equal(solver.last_qp_failure['qp'][name],failure['qp'][name])
        else:
            velocity=solver.velocity(request['joints'],request['T_world_pelvis'],request['target'],request['dt'],**options)
            values=np.array([velocity[n] for n in failure['joint_names']])
            assert np.all(values>=np.asarray(failure['owned_velocity_lower'])-1e-8)
            assert np.all(values<=np.asarray(failure['owned_velocity_upper'])+1e-8)
            assert np.max(np.abs(values))<=.5+1e-8
            assert solver.last_collision_report['pink_barrier_minimum_m']==.0065
            assert solver.last_collision_report['pink_barrier_gain_per_s']==new_gain
            assert solver.last_collision_report['upper_controlled_pairs']==473
    for gain in (float('nan'),float('inf'),.099,1.001,True):
        with pytest.raises(ValueError):g1.PinkUpperBody(io['actions'][0]['joint_names'],names,camera,collision_gain=gain)


def actual_gravity_fixture():
    dump=json.loads((ROOT/'Robot_G1/results/default_upper_yaw_01/startup_physx_mass_properties.json').read_text())
    metadata=json.loads((ROOT/'Robot_G1/config/actual_gravity_model.json').read_text())
    solver=gravity_only.ActualGravity(ROOT/'Robot_G1/kinematics/g1_29dof.urdf',metadata)
    return solver,dump,metadata


def test_g1_fresh_static_gravity_uses_each_current_physical29_state_and_canonical_upper_ownership():
    pytest.importorskip('pinocchio')
    runner=module('g1_fresh_gravity_runner_test',ROOT/'Robot_G1/adaptive_runner.py')
    local_module=module('g1_fresh_gravity_model_test',ROOT/'Robot_G1/gravity_hold.py')
    reference,dump,metadata=actual_gravity_fixture()
    local=local_module.GravityHold(ROOT/'Robot_G1/kinematics/g1_29dof.urdf',metadata)
    names=dump['joint_names']  # Actual interleaved PhysX29 order, not URDF order.
    previous=None
    for substep in range(4):
        q={n:float(v+.01*substep*np.sin(i+.3))for i,(n,v)in enumerate(metadata['calibration_joint_positions'].items())}
        R=Rotation.from_euler('xyz',[.02*substep,-.1,.2]).as_matrix()
        original=np.array([q[n]for n in names]);T=np.eye(4);T[:3,:3]=R
        expected=reference.torque(q,T)
        gravity,report=runner.current_state_gravity_support(local,joint_names=names,
            joint_positions=original,pelvis_rotation=R,stamp_s=1.+substep*.005,physics_state_seq=200+substep)
        assert set(gravity)==set(local.upper_joint_names) and len(gravity)==15
        assert max(abs(gravity[n]-expected[n])for n in gravity)<2e-12
        assert report['physics_state_seq']==200+substep and report['stamp_s']==1.+substep*.005
        assert report['measurement_source']=='current_measured_physx'
        assert report['joint_names']==names and not report['trajectory_reused'] and not report['worker_packet_gravity_reused']
        np.testing.assert_array_equal(report['joint_positions_rad'],original)
        np.testing.assert_array_equal(report['pelvis_rotation_world'],R)
        if previous is not None:assert max(abs(gravity[n]-previous[n])for n in gravity)>1e-5
        previous=dict(gravity)
        original[:]=99.;R[:]=0.  # Report and output cannot alias later SDK updates.
        assert max(report['joint_positions_rad'])<99.
        assert np.linalg.det(report['pelvis_rotation_world'])==pytest.approx(1.)


@pytest.mark.parametrize('fault',['names','nan_q','bool_q','rotation','future_type','sequence'])
def test_g1_fresh_static_gravity_rejects_invalid_current_measurement_without_solver_reuse(fault):
    runner=module('g1_bad_fresh_gravity_runner_test',ROOT/'Robot_G1/adaptive_runner.py')
    local_module=module('g1_bad_fresh_gravity_model_test',ROOT/'Robot_G1/gravity_hold.py')
    metadata=json.loads((ROOT/'Robot_G1/config/actual_gravity_model.json').read_text())
    local=local_module.GravityHold(ROOT/'Robot_G1/kinematics/g1_29dof.urdf',metadata)
    args=dict(joint_names=list(local.joint_names),joint_positions=np.zeros(29),pelvis_rotation=np.eye(3),stamp_s=1.,physics_state_seq=200)
    if fault=='names':args['joint_names']=['missing']*29
    elif fault=='nan_q':args['joint_positions'][2]=np.nan
    elif fault=='bool_q':args['joint_positions']=np.zeros(29,dtype=bool)
    elif fault=='rotation':args['pelvis_rotation']=np.zeros((3,3))
    elif fault=='future_type':args['stamp_s']=True
    else:args['physics_state_seq']=-1
    with pytest.raises(ValueError):runner.current_state_gravity_support(local,**args)


@pytest.mark.parametrize('extra',[['--g1-gravity-profile','urdf'],
    ['--g1-gravity-profile','actual_physx_v1','--upper-mode','position_integral'],
    ['--g1-gravity-profile','actual_physx_v1','--upper-mode','bounded_integral','--g1-following-reserve']])
def test_g1_fresh_static_gravity_profile_rejects_incompatible_actual_runner_before_kit(tmp_path,extra):
    output=tmp_path/'should_not_start'
    result=subprocess.run([sys.executable,str(ROOT/'Robot_G1/adaptive_runner.py'),
        '--robot','g1','--output',str(output),'--g1-gravity-hold-profile','fresh_current_state_v1',*extra],
        capture_output=True,text=True,timeout=15.)
    assert result.returncode!=0 and any(reason in result.stderr for reason in (
        'fresh G1 static hold requires','actual G1 gravity requires its single explicit PD-plus-gravity capped execution path'))
    assert not output.exists() and 'Simulation App' not in result.stdout


def test_g1_actual_gravity_matches_potential_derivatives_for_all_upper_joints_and_tilted_base():
    pytest.importorskip('pinocchio')
    solver,dump,metadata=actual_gravity_fixture();joints=dict(metadata['calibration_joint_positions'])
    upper=[n for n in joints if n=='waist_yaw_joint' or any(x in n for x in ('shoulder_','elbow_joint','wrist_'))]
    for angle in (0.,.15,-.24):
        pelvis=np.asarray(metadata['T_world_pelvis']).copy()
        pelvis[:3,:3]=Rotation.from_euler('xyz',[angle,-.3,1.2]).as_matrix()
        q={name:float(value+.025*np.sin(i+angle)) for i,(name,value) in enumerate(joints.items())}
        values=solver.torque(q,pelvis)
        for name in upper:
            plus=dict(q);minus=dict(q);plus[name]+=1e-6;minus[name]-=1e-6
            derivative=(solver.potential(plus,pelvis)-solver.potential(minus,pelvis))/2e-6
            assert abs(values[name]-derivative)<2e-6
    assert len(upper)==15
    assert sum(i.mass for i in solver.model.inertias)==pytest.approx(33.00114506483078,abs=1e-10)


def test_g1_actual_gravity_checks_current_native_plant_and_rejects_mass_com_or_actor_changes():
    pytest.importorskip('pinocchio')
    solver,dump,metadata=actual_gravity_fixture();names=dump['body_names']
    actors=np.asarray(dump['actor_world_poses_xyzw']['values'])[0]
    transforms=np.broadcast_to(np.eye(4),(len(names),4,4)).copy()
    transforms[:,:3,:3]=Rotation.from_quat(actors[:,3:]).as_matrix();transforms[:,:3,3]=actors[:,:3]
    bodies=dict(zip(names,transforms));masses=np.asarray(dump['masses']['values'])[0]
    centers=np.asarray(dump['coms']['values'])[0,:,:3];joints=metadata['calibration_joint_positions']
    report=solver.verify_native(names,masses,centers,bodies,joints,bodies['pelvis'])
    assert report['passed'] and report['body_count']==30
    for mutation in ('mass','center','pose','missing'):
        m=masses.copy();c=centers.copy();b={n:T.copy() for n,T in bodies.items()}
        if mutation=='mass':m[0]+=.01
        if mutation=='center':c[1,0]+=.001
        if mutation=='pose':b[names[2]][0,3]+=.001
        if mutation=='missing':b.pop(names[-1])
        with pytest.raises(ValueError):solver.verify_native(names,m,c,b,joints,bodies['pelvis'])


def test_g1_actual_gravity_never_replaces_original_pink_kinematics_or_collision_mass_model():
    pytest.importorskip('pinocchio');pytest.importorskip('pink')
    _,dump,metadata=actual_gravity_fixture()
    names=dump['joint_names'];legs=[n for n in names if any(x in n for x in ('hip_','knee_','ankle_'))]
    original=g1.PinkUpperBody(legs,names,np.eye(4),posture_cost=.03,damping_cost=.02)
    measured=g1.PinkUpperBody(legs,names,np.eye(4),posture_cost=.03,damping_cost=.02,actual_gravity_metadata=metadata)
    q=metadata['calibration_joint_positions'];pelvis=np.asarray(metadata['T_world_pelvis'])
    target=original.predicted_camera(q,pelvis);target[0,3]+=.01
    np.testing.assert_array_equal(original.predicted_camera(q,pelvis),measured.predicted_camera(q,pelvis))
    assert original.velocity(q,pelvis,target,.02)==measured.velocity(q,pelvis,target,.02)
    assert sum(i.mass for i in measured.model.inertias)==pytest.approx(35.11514202)
    assert sum(i.mass for i in measured.actual_gravity.model.inertias)==pytest.approx(33.00114506483078)
    before=original.upper_gravity(q,pelvis);after=measured.upper_gravity(q,pelvis)
    assert set(before)==set(after)==set(original.owners['pink'])
    assert before['left_wrist_pitch_joint']-after['left_wrist_pitch_joint']==pytest.approx(-.233,abs=.005)


def test_g1_actual_gravity_rejects_partial_or_unbound_calibration_and_snapshots_metadata():
    pytest.importorskip('pinocchio')
    import copy
    solver,dump,metadata=actual_gravity_fixture();urdf=ROOT/'Robot_G1/kinematics/g1_29dof.urdf'
    metadata['bodies']['pelvis']['mass_kg']+=1.
    assert solver.metadata['bodies']['pelvis']['mass_kg']!=metadata['bodies']['pelvis']['mass_kg']
    for mutate in ('hash','missing','mode'):
        broken=copy.deepcopy(solver.metadata)
        if mutate=='hash':broken['source_urdf_sha256']='0'*64
        if mutate=='missing':broken['bodies'].pop('pelvis')
        if mutate=='mode':broken['rotational_inertia_used']=True
        with pytest.raises(ValueError):gravity_only.ActualGravity(urdf,broken)
    bad_dump=copy.deepcopy(dump);bad_dump['conventions']['coms_axes']='unknown'
    with pytest.raises(ValueError):gravity_only.calibrate(urdf,bad_dump,'0'*64)


def test_g1_actual_gravity_native_verification_uses_real_isolated_worker_without_kit_pin_import():
    pytest.importorskip('pinocchio');pytest.importorskip('pink')
    _,dump,metadata=actual_gravity_fixture();names=dump['joint_names']
    legs=[n for n in names if any(t in n for t in ('hip_','knee_','ankle_'))]
    actors=np.asarray(dump['actor_world_poses_xyzw']['values'])[0]
    transforms=np.broadcast_to(np.eye(4),(len(actors),4,4)).copy()
    transforms[:,:3,:3]=Rotation.from_quat(actors[:,3:]).as_matrix();transforms[:,:3,3]=actors[:,:3]
    bodies=dict(zip(dump['body_names'],transforms));masses=np.asarray(dump['masses']['values'])[0]
    centers=np.asarray(dump['coms']['values'])[0,:,:3]
    worker=g1.PinkProcess(legs,names,np.eye(4),ROOT/'Robot_G1/kinematics/g1_29dof.urdf',
        source_path=ROOT/'Robot_G1/wbc_backend.py',solver_options=dict(actual_gravity_metadata=metadata))
    try:
        worker.warmup(metadata['calibration_joint_positions'])
        report=worker.verify_actual_gravity(dump['body_names'],masses,centers,bodies,
            metadata['calibration_joint_positions'],bodies['pelvis'])
        assert report['passed'] and worker.pending is None and worker.needs_reset
        masses[0]+=.1
        with pytest.raises(RuntimeError,match='verification failed'):
            worker.verify_actual_gravity(dump['body_names'],masses,centers,bodies,
                metadata['calibration_joint_positions'],bodies['pelvis'])
    finally:worker.close()


def test_g1_pink_soft_costs_apply_only_in_declared_locomotion_phases():
    assert pink_profile.task_cost_overrides(True,True,.1,.1)==dict(posture_cost_override=.1,damping_cost_override=None)
    assert pink_profile.task_cost_overrides(True,False,.1,.1)==dict(posture_cost_override=None,damping_cost_override=.1)
    for moving in (True,False):
        assert pink_profile.task_cost_overrides(False,moving,.1,.1)==dict(posture_cost_override=None,damping_cost_override=None)
    assert pink_profile.task_cost_overrides(True,True)==dict(posture_cost_override=None,damping_cost_override=None)
    assert pink_profile.task_cost_overrides(True,False,None,.2)==dict(posture_cost_override=None,damping_cost_override=.2)
    assert pink_profile.task_cost_overrides(True,True,None,.2)==dict(posture_cost_override=None,damping_cost_override=None)
    for moving in (True,False):
        assert pink_profile.task_cost_overrides(False,moving,None,.2)==dict(posture_cost_override=None,damping_cost_override=None)
    with pytest.raises(ValueError):pink_profile.task_cost_overrides(True,False,None,.501)


def test_g1_phase_yaw_calibration_excludes_stop_anchor_and_preserves_native_slew():
    controller=servo.VelocityServo(**pink_profile.velocity_servo_overrides('yaw_phase_2_3_v1'))
    previous=np.zeros(3)
    for moving,requested in [(True,[0.,0.,.12]),(False,[0.,0.,0.]),(True,[0.,0.,-.12]),
                             (True,[.1,0.,0.]),(False,[0.,0.,0.])]:
        for _ in range(100):
            schedule=pink_profile.phase_velocity_servo_overrides('yaw_phase_2_3_v1',moving,requested[2])
            controller.feedforward_gain=np.array(schedule['feedforward_gain'])
            controller.negative_feedforward_gain=np.array(schedule['negative_feedforward_gain'])
            anchor=np.array([.005,-.005,-.04]) if not moving else None
            command=controller.update(requested,[0.,0.,0.],.02,pose_error_body=anchor)
            assert np.all(np.abs(command-previous)<=np.array([.5,.5,.8])*.02+1e-12)
            assert np.all(np.abs(command)<=np.array([.5,.5,1.])+1e-12)
            assert np.all(np.abs(controller.integral)<=np.array([.2,.2,.3])+1e-12)
            previous=command
        assert schedule['phase']==('moving_yaw' if moving and requested[2]!=0. else 'baseline')
        if not moving or requested[2]==0.:
            np.testing.assert_array_equal(controller.feedforward_gain,[1.,1.,1.])
            np.testing.assert_array_equal(controller.negative_feedforward_gain,[1.,1.,1.])
    assert pink_profile.phase_velocity_servo_overrides('yaw_gain2',False,0.) is None


def test_g1_execution_lease_covers_sim_and_wall_substeps_and_reports_expiry_as_unknown():
    packet=dict(ready=True,ok=True,status='fresh_command',stamp_s=1.,command_sent_wall_s=100.,
        lease_deadline_sim_s=1.06,lease_deadline_wall_s=100.06)
    assert solver_feedback.execution_lease_status(packet,stamp_s=1.04,wall_s=100.04,horizon_s=.02)['valid']
    for sim,wall,horizon in [(1.06,100.04,.02),(1.04,100.06,.02),(1.061,100.04,0.),(1.04,100.061,0.)]:
        report=solver_feedback.execution_lease_status(packet,stamp_s=sim,wall_s=wall,horizon_s=horizon)
        assert not report['valid'] and report['reason']=='command_stale'
    packet.update(ready=False,status='command_expired_during_execution_hold_upper_targets')
    assert solver_feedback.public_support(packet)==dict(reason='command_stale',feasible=None)


def test_g1_declared_yaw_feedforward_changes_only_supported_input_with_original_slew_and_antiwindup():
    overrides=pink_profile.velocity_servo_overrides('yaw_gain2')
    assert overrides==dict(feedforward_gain=(1.,1.,2.))
    controller=servo.VelocityServo(**overrides);previous=np.zeros(3)
    for desired in ([.1,-.1,.12],[0.,0.,-.12],[0.,0.,0.],[.5,-.5,1.],[-.5,.5,-1.]):
        for step in range(200):
            command=controller.update(desired,np.zeros(3),.02)
            assert np.all(np.abs(command)<=np.array([.5,.5,1.])+1e-12)
            assert np.all(np.abs(command-previous)<=np.array([.5,.5,.8])*.02+1e-12)
            assert np.all(np.abs(controller.integral)<=np.array([.2,.2,.3])+1e-12)
            if step==50:
                held=np.clip(np.zeros(3),previous-controller.slew*.02,previous+controller.slew*.02)
                integral_before=controller.integral_before_update.copy()
                controller.applied(held,transient_hold=True)
                np.testing.assert_array_equal(controller.integral,integral_before)
                previous=held
            else:previous=command
    request=np.array([.1,0.,.12]);copy=request.copy()
    controller.update(request,request,.02)
    np.testing.assert_array_equal(request,copy)
    assert pink_profile.velocity_servo_overrides('baseline')=={}
    with pytest.raises(ValueError):pink_profile.velocity_servo_overrides('unknown')


@pytest.mark.parametrize('posture,damping',[(float('nan'),.1),(.2,.1),(.1,-1.),(.1,float('inf')),(True,.1)])
def test_g1_pink_soft_costs_reject_unbounded_or_nonfinite_preferences(posture,damping):
    with pytest.raises(ValueError):pink_profile.task_cost_overrides(True,True,posture,damping)


def test_go2_soft_posture_is_disabled_equivalent_to_measured_executed_coordinator():
    executed=module('executed_go2_coordinator',ROOT/'Robot_Go2/results/adaptive_backward_ground_01/sources/camera_coordinator.py')
    for x in (.42,.50,.66):
        controllers=[m.CameraCoordinator(np.eye(4),backend=go2,response_profile='measured_backward_v1') for m in (executed,coordinator)]
        link=pose([0.,0.,.3],[0.,0.,0.])
        sample=lambda t:pose([x-.1*t,0.,.5],[0.,0.,0.])
        reports=[c.command(0.,sample,link,link,[-.18,0.,0.],[-.1,0.,0.],future_policy_velocity=[-.18,0.,0.],previous_applied=[-.18,0.,0.]) for c in controllers]
        np.testing.assert_allclose(reports[0]['base_command'],reports[1]['base_command'],atol=1e-12)
        np.testing.assert_allclose(reports[0]['native_command'],reports[1]['native_command'],atol=1e-12)
        assert reports[0]['camera_supported']==reports[1]['camera_supported']


def test_go2_soft_native_posture_changes_only_bounded_base_input_not_world_samples():
    c=coordinator.CameraCoordinator(np.eye(4),backend=go2,response_profile='measured_backward_v1')
    link=pose([0.,0.,.3],[0.,0.,0.])
    for t in np.arange(-.5,0.,.02):c.observe_applied_input(t,[-.18,0.,0.])
    samples={float(t):pose([.5-.1*t,0.,.5],[0.,0.,0.]) for t in np.linspace(0.,.8,5)}
    original={t:T.copy() for t,T in samples.items()}
    arguments=dict(future_policy_velocity=[-.18,0.,0.],previous_applied=[-.18,0.,0.])
    baseline=c.command(0.,samples.__getitem__,link,link,[-.18,0.,0.],[-.1,0.,0.],**arguments)
    soft=c.command(0.,samples.__getitem__,link,link,[-.18,0.,0.],[-.1,0.,0.],native_posture_x_m=.58,posture_weight=.2,**arguments)
    assert soft['camera_supported'] and soft['base_command'][0]<baseline['base_command'][0]
    assert np.max(np.abs(soft['base_command']))<=.2
    assert abs(soft['base_command'][0]+.18)<=.5*.02+1e-10
    for t,T in original.items():np.testing.assert_array_equal(samples[t],T)
    assert soft['native_posture_preference']['original_hard_workspace_unchanged']
    assert soft['camera_reference_changed'] is False


def test_go2_soft_posture_cannot_hide_current_domain_failure_or_incompatible_future():
    c=coordinator.CameraCoordinator(np.eye(4),backend=go2)
    link=pose([0.,0.,.3],[0.,0.,0.])
    outside=c.command(0.,lambda t:pose([.71,0.,.5],[0.,0.,0.]),link,link,[0.,0.,0.],[0.,0.,0.],native_posture_x_m=.58,posture_weight=.2)
    assert outside['reason']=='out_of_policy_domain' and outside['feasible'] is False
    conflict=c.command(0.,lambda t:pose([.5+.8*t,0.,.5],[0.,0.,0.]),link,link,[0.,0.,0.],[0.,0.,0.],native_posture_x_m=.58,posture_weight=.2)
    assert not conflict['camera_supported'] and conflict['feasible'] is False
    assert conflict['camera_reference_changed'] is False


def test_go2_soft_posture_absolute_schedule_is_gone_before_terminal_hold():
    weight=coordinator.operating_posture_weight
    assert weight(9.,10.,15.)==0.
    assert weight(10.,10.,15.)==0.
    assert weight(11.,10.,15.)==pytest.approx(.2)
    assert weight(15.25,10.,15.)==pytest.approx(.1)
    assert weight(15.5,10.,15.)==0. and weight(20.,10.,15.)==0.
    with pytest.raises(ValueError):weight(12.,15.,10.)


def test_physx_documented_material_combine_priority_preserves_coefficients():
    assert contact_materials.combine(.5,1.,'average','average')==('average',.75)
    assert contact_materials.combine(.5,1.,'average','multiply')==('multiply',.5)
    assert contact_materials.combine(.5,1.,'max','multiply')==('max',1.)
    with pytest.raises(ValueError,match='unknown'):
        contact_materials.combine(.5,1.,'unsupported','multiply')
    with pytest.raises(ValueError,match='nonnegative'):
        contact_materials.combine(-1.,1.,'average','average')


def test_native_material_readback_resolves_explicit_scene_default_ignoring_render_material():
    import os
    envroot=Path('/home/w84384651/miniconda3/envs/env_isaaclab');site=envroot/'lib/python3.11/site-packages/isaacsim'
    usd=next(iter((site/'extscache').glob('omni.usd.libs-*/pxr')),None)
    if usd is None:pytest.skip('native USD CPU distribution unavailable')
    libs=usd.parent;environment=dict(os.environ,LD_LIBRARY_PATH=f'{envroot}/lib:{libs}/bin:{site}/kit',PYTHONPATH=str(libs))
    script=r'''
import importlib.util
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf
s=importlib.util.spec_from_file_location('material_audit',__import__('sys').argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
stage=Usd.Stage.CreateInMemory();scene=UsdPhysics.Scene.Define(stage,'/physicsScene')
default=UsdShade.Material.Define(stage,'/physicsScene/defaultMaterial');api=UsdPhysics.MaterialAPI.Apply(default.GetPrim());api.CreateStaticFrictionAttr(.5);api.CreateDynamicFrictionAttr(.5);api.CreateRestitutionAttr(0.)
UsdShade.MaterialBindingAPI.Apply(scene.GetPrim()).Bind(default,UsdShade.Tokens.weakerThanDescendants,'physics')
UsdGeom.Xform.Define(stage,'/World/Robot');robot=UsdGeom.Cube.Define(stage,'/World/Robot/collider');UsdPhysics.CollisionAPI.Apply(robot.GetPrim())
visual=UsdShade.Material.Define(stage,'/Visual');UsdShade.MaterialBindingAPI.Apply(robot.GetPrim()).Bind(visual)
UsdGeom.Xform.Define(stage,'/World/Ground');ground=UsdGeom.Cube.Define(stage,'/World/Ground/collider');UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
mat=UsdShade.Material.Define(stage,'/GroundMaterial');api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());api.CreateStaticFrictionAttr(1.);api.CreateDynamicFrictionAttr(1.);api.CreateRestitutionAttr(0.)
UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
r=m.audit_stage(stage);assert r['all_collision_materials_explicitly_resolved'];assert r['robot_materials'][0]['binding_scope']=='explicit_physics_scene_default';assert r['expected_contact_pair_materials'][0]['effective_static_friction']==.75
mat.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode',Sdf.ValueTypeNames.Token).Set('multiply')
r=m.audit_stage(stage);assert r['expected_contact_pair_materials'][0]['effective_static_friction']==.5;assert r['robot_materials'][0]['static_friction']==.5
UsdPhysics.Scene.Define(stage,'/secondScene');assert not m.audit_stage(stage)['all_collision_materials_explicitly_resolved']
'''
    result=subprocess.run([str(envroot/'bin/python'),'-c',script,str(ROOT/'Robot_G1/contact_materials.py')],env=environment,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_parking_standing_input_does_not_wait_for_pi_filtered_zero_crossing():
    controller=parking.ParkingSupervisor()
    previous=np.array([0.,.12,0.]);body=np.eye(4)
    reports=[]
    for step in range(101):
        previous,report=controller.update(step*.02,body,[0.,.08,0.],[0.,.06,0.],previous,.02)
        reports.append(report)
    assert reports[-1]['mode']=='SETTLE'
    assert reports[-1]['literal_zero_input']
    assert not any(r['entered'] for r in reports)
    # A nonzero measured velocity cannot be labelled parked just because the
    # filtered PI state passed through zero during a delayed lateral oscillation.
    assert not reports[-1]['instantaneous_velocity_stable']


def test_parking_requires_sustained_raw_stability_and_exits_on_later_real_drift():
    controller=parking.ParkingSupervisor()
    previous=np.array([.1,0.,0.]);body=np.eye(4);entries=0
    for step in range(81):
        measured=[.08,0.,0.] if step<50 else [0.,0.,0.]
        previous,report=controller.update(step*.02,body,measured,[.06,0.,0.],previous,.02)
        entries+=report['entered']
    assert entries==1 and report['mode']=='PARK'
    body[1,3]=.026
    previous,report=controller.update(1.62,body,[0.,0.,0.],[0.,-.05,0.],previous,.02)
    assert report['mode']=='BRAKE' and report['exited']
    assert abs(previous[1])<=.01+1e-12
    assert report['camera_task_modified'] is False


def test_velocity_consistency_audit_keeps_distinct_native_and_pose_evidence():
    samples=[]
    for t in np.arange(.005, .105, .005):
        raw=np.r_[[.1*t, 0., .3], Rotation.from_rotvec([0., 0., .2*t]).as_quat()].tolist()
        # Synthetic split integration: preserve both signals, never silently
        # replace the native velocity with a more favourable pose derivative.
        native=[.1, 0., 0., .07, 0., .2]
        samples.append(dict(stamp_s=float(t),root_transform_raw_after_fk=raw,
            root_transform_raw_before_fk=raw,base_transform_raw=raw,
            root_velocity_raw_after_fk=native,root_velocity_raw_before_fk=native,
            base_com_velocity_raw=native,root_com_velocity_w=native))
    report=velocity_audit.analyze(samples,0)
    assert report['requested_base_is_articulation_root']
    assert report['max_root_base_native_velocity_difference']==0
    assert report['max_kinematic_refresh_root_orientation_change_rad']==pytest.approx(0.)
    np.testing.assert_allclose(report['full']['pose_angular_mean_world'],[0.,0.,.2],atol=1e-12)
    np.testing.assert_allclose(report['full']['native_angular_mean_world'],[.07,0.,.2],atol=1e-12)
    np.testing.assert_allclose(report['full']['angular_difference_rmse_rad_s'],[.07,0.,0.],atol=1e-12)
    assert report['finite_pose_velocity_replaces_native_velocity'] is False
    with pytest.raises(ValueError,match='ordered'):
        velocity_audit.analyze([samples[0],samples[0]],0)


def pose(position, rpy):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    result[:3, 3] = position
    return result


def test_go2_mixed_frame_roundtrip_preserves_tilted_camera_and_full_extrinsics():
    rng = np.random.default_rng(20261020)
    for _ in range(100):
        link = pose(rng.normal(size=3), rng.uniform([-.6, -.6, -3.], [.6, .6, 3.]))
        camera = pose(rng.normal(size=3), rng.uniform(-2., 2., size=3))
        extrinsic = pose(rng.uniform(-.2, .2, size=3), rng.uniform(-2., 2., size=3))
        command = go2.camera_to_mixed(camera, link, extrinsic)
        reconstructed = go2.mixed_to_world_camera(command, link, extrinsic)
        np.testing.assert_allclose(reconstructed, camera, atol=1e-12)
        ee = camera@np.linalg.inv(extrinsic)
        np.testing.assert_allclose(command[:2], (np.linalg.inv(link)@ee)[:2, 3], atol=1e-12)
        assert command[2] == pytest.approx(ee[2, 3])


def test_rotating_xy_then_overwriting_world_z_is_not_the_inverse_when_tilted():
    link = pose([1., 2., .5], [.35, -.4, .2])
    command = np.r_[.5, .1, .8, 1., 0., 0., 0.]
    exact = go2.mixed_to_world_ee(command, link)
    naive = link[:3, 3]+link[:3, :3]@np.r_[command[:2], 0.]
    naive[2] = command[2]
    assert np.linalg.norm(exact[:3, 3]-naive) > .01
    np.testing.assert_allclose((np.linalg.inv(link)@exact)[:2, 3], command[:2], atol=1e-12)


@pytest.mark.parametrize('attitude', [[0., np.pi/2, 0.], [np.pi/2, 0., 0.]])
def test_singular_mixed_frame_is_rejected(attitude):
    link = pose([0., 0., .3], attitude)
    with pytest.raises(ValueError, match='singular'):
        go2.camera_to_mixed(np.eye(4), link, np.eye(4))
    with pytest.raises(ValueError, match='singular'):
        go2.mixed_to_world_ee([.5, 0., .5, 1., 0., 0., 0.], link)


def test_g1_height_ownership_has_no_policy_pink_joint_overlap():
    import yaml
    data = yaml.safe_load((ROOT/'Robot_G1/config/height_policy_io.yaml').read_text())
    ownership = g1.ownership(data['actions'][0]['joint_names'], data['articulations']['robot']['joint_names'])
    assert len(ownership['policy']) == 12 and len(ownership['pink']) == 15
    assert not set(ownership['policy']) & set(ownership['pink'])
    assert set(ownership['hold']) == {'waist_roll_joint', 'waist_pitch_joint'}
    assert sum(x['shape'][0] for x in data['observations']['policy']) == 80
    with pytest.raises(ValueError):
        g1.ownership(list(ownership['policy'])+['waist_yaw_joint'], data['articulations']['robot']['joint_names'])


@pytest.mark.parametrize('xyz,euler,violations', [
    ([.5, 0., .5], [0., 0., 0.], []),
    ([.4, -.35, .1], [-3.14/4, 0., 3.14/3], []),
    ([-.236, 0., .5], [0., 0., 0.], ['x_link0']),
    ([.71, .36, .81], [0., 0., 0.], ['x_link0', 'y_link0', 'z_world']),
    ([.5, 0., .5], [0., -1., 0.], ['pitch_link0']),
])
def test_go2_upstream_outer_domain_only_excludes_without_certifying_feasibility(xyz, euler, violations):
    quat = Rotation.from_euler('xyz', euler).as_quat()[[3, 0, 1, 2]]
    command = np.r_[xyz, quat]; original = command.copy()
    result = go2.check_upstream_domain(command)
    assert result['violations'] == violations
    assert result['supported'] is (False if violations else None)
    assert result['feasible'] is None and result['checkpoint_training_coverage'] == 'unknown'
    np.testing.assert_array_equal(command, original)


def test_go2_missing_domain_metadata_is_unknown_and_current_executed_adapter_is_preserved(tmp_path):
    import ast
    result = go2.check_upstream_domain([.5, 0., .5, 1., 0., 0., 0.], config_path=tmp_path/'missing.py')
    assert result['supported'] is False and result['feasible'] is None
    assert result['status'] == 'capability_metadata_unknown'
    executed = ROOT/'Robot_Go2/results/wbc_camera_coordinated_01/wbc_backend_source.py'
    if executed.is_file():
        def functions(path):
            return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
        before, after = functions(executed), functions(ROOT/'Robot_Go2/wbc_backend.py')
        for name in ('rigid', 'camera_to_mixed', 'mixed_to_world_ee', 'mixed_to_world_camera', 'set_camera_target'):
            assert before[name] == after[name]


def test_real_installed_pink_converges_without_changing_policy_or_base_coordinates():
    python = ROOT/'Robot_G1/.venv/bin/python'
    if not python.is_file():
        pytest.skip('isolated G1 runtime is not installed')
    script = r'''
import importlib.util,json,numpy as np,yaml
from pathlib import Path
from scipy.spatial.transform import Rotation
p=Path(__import__('sys').argv[1]);s=importlib.util.spec_from_file_location('actual_g1_backend',p/'wbc_backend.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
io=yaml.safe_load((p/'config/height_policy_io.yaml').read_text());names=io['articulations']['robot']['joint_names'];q=dict(zip(names,io['articulations']['robot']['default_joint_pos']))
camera=np.eye(4);camera[:3,3]=[.2,.04,.11];camera[:3,:3]=np.column_stack([[0,-1,0],[0,0,-1],[1,0,0]])
c=m.PinkUpperBody(io['actions'][0]['joint_names'],names,camera,posture_home=q.copy(),posture_cost=.03,damping_cost=.02)
base=np.eye(4);base[:3,:3]=Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix();base[:3,3]=[1.,2.,.72]
reference=c.predicted_camera(q,base);reference[:3,3]+=[.015,-.01,.01]
reference[:3,:3]=Rotation.from_rotvec([.01,-.015,.02]).as_matrix()@reference[:3,:3]
locked={n:q[n] for n in c.owners['policy']+c.owners['hold']};previous={n:0. for n in c.owners['pink']};parked_preference=q.copy()
for _ in range(250):
 v=c.velocity(q,base,reference,.02,posture_reference=parked_preference)
 assert set(v)==set(c.owners['pink'])
 for name,speed in v.items():
  assert abs(speed)<=.5+1e-6 and abs(speed-previous[name])<=.02+1e-6
  q[name]+=speed*.02
 previous=v
assert all(q[n]==value for n,value in locked.items())
assert any(abs(q[n]-parked_preference[n])>.001 for n in c.owners['pink'])
actual=c.predicted_camera(q,base)
ep=np.linalg.norm(actual[:3,3]-reference[:3,3]);er=Rotation.from_matrix(actual[:3,:3]@reference[:3,:3].T).magnitude()
assert ep<.002 and er<.01,(ep,er)
assert c.model.nq==29 and c.model.nv==29
print(json.dumps(dict(position_error_m=ep,orientation_error_rad=er,actual_pink=True,free_base_optimized=False)))
'''
    result = subprocess.run([str(python), '-c', script, str(ROOT/'Robot_G1')], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['actual_pink']


def test_genuine_pink_soft_waist_preference_reallocates_camera_task_without_locking():
    script=r'''
import importlib.util,yaml,numpy as np,json
from pathlib import Path
from scipy.spatial.transform import Rotation
p=Path(__import__('sys').argv[1]);s=importlib.util.spec_from_file_location('backend',p/'wbc_backend.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
io=yaml.safe_load((p/'config/height_policy_io.yaml').read_text());names=io['articulations']['robot']['joint_names'];home=dict(zip(names,io['articulations']['robot']['default_joint_pos']))
cam=np.eye(4);cam[:3,3]=[.2,.04,.11];cam[:3,:3]=np.column_stack([[0,-1,0],[0,0,-1],[1,0,0]])
results=[]
for cost in [None,.15]:
 c=m.PinkUpperBody(io['actions'][0]['joint_names'],names,cam,posture_home=home.copy(),posture_cost=.03,damping_cost=.02,waist_posture_cost=cost)
 q=home.copy();q['waist_yaw_joint']=.25;preference=q.copy();locked={n:q[n] for n in c.owners['policy']+c.owners['hold']}
 T=c.predicted_camera(q,np.eye(4));T[:3,3]+=[0.,-.08,.03];T[:3,:3]=Rotation.from_rotvec([.1,.1,.2]).as_matrix()@T[:3,:3]
 for _ in range(300):
  v=c.velocity(q,np.eye(4),T,.02,posture_reference=preference)
  for n,u in v.items():q[n]+=u*.02
 actual=c.predicted_camera(q,np.eye(4));ep=np.linalg.norm(T[:3,3]-actual[:3,3]);er=Rotation.from_matrix(T[:3,:3]@actual[:3,:3].T).magnitude()
 assert ep<.005 and er<.02,(ep,er)
 assert all(q[n]==value for n,value in locked.items())
 if cost is not None:
  assert abs(q['waist_yaw_joint']-.25)>.1
  assert c.last_posture_report['waist_yaw_locked'] is False
  assert c.last_posture_report['camera_frame_task_unchanged']
 results.append(abs(q['waist_yaw_joint']))
assert results[1]<.01 and results[1]<results[0]/10,results
print(json.dumps(dict(genuine_pink=True,plain_and_weighted_final_waist_abs_rad=results)))
'''
    result=subprocess.run([str(ROOT/'Robot_G1/.venv/bin/python'),'-c',script,str(ROOT/'Robot_G1')],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['genuine_pink']


@pytest.mark.parametrize('soft_costs',[False,True])
def test_genuine_pink_collision_barrier_rejects_closing_camera_motion_without_leg_authority(soft_costs):
    python=ROOT/'Robot_G1/.venv/bin/python'
    if not python.is_file():pytest.skip('isolated native Pink unavailable')
    script=r'''
import importlib.util,yaml,numpy as np,pinocchio as pin
from pathlib import Path
from pink.barriers import SelfCollisionBarrier
p=Path(__import__('sys').argv[1]);s=importlib.util.spec_from_file_location('backend',p/'wbc_backend.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
io=yaml.safe_load((p/'config/height_policy_io.yaml').read_text());names=io['articulations']['robot']['joint_names'];q=dict(zip(names,io['articulations']['robot']['default_joint_pos']))
plain=m.PinkUpperBody(io['actions'][0]['joint_names'],names,np.eye(4),posture_home=q.copy(),posture_cost=.03,damping_cost=.02)
guard=m.PinkUpperBody(io['actions'][0]['joint_names'],names,np.eye(4),posture_home=q.copy(),posture_cost=.03,damping_cost=.02)
D=plain.model.createData();full=pin.neutral(plain.model)
for n,v in q.items():full[plain.model.joints[plain.model.getJointId(n)].idx_q]=v
pin.forwardKinematics(plain.model,D,full);pin.updateFramePlacements(plain.model,D)
poses={n:D.oMf[plain.model.getFrameId(n)].np for n in ['torso_link','right_wrist_yaw_link']}
center=np.linalg.inv(poses['torso_link'])@np.r_[poses['right_wrist_yaw_link'][:3,3]+[.05,0.,0.],1.]
meta=dict(shapes=[dict(name='torso_test',body='torso_link',kind='sphere',radius=.02,center_body=center[:3].tolist()),dict(name='wrist_test',body='right_wrist_yaw_link',kind='sphere',radius=.02,center_body=[0.,0.,0.])],allowed_pairs=[])
guard.configure_collisions(meta,q,np.eye(4),poses)
T=plain.predicted_camera(q,np.eye(4));T[0,3]+=.1
options=dict(posture_cost_override=.1,damping_cost_override=.1) if __import__('sys').argv[2]=='true' else {}
v0=plain.velocity(q,np.eye(4),T,.02,posture_reference=q.copy(),**options);v1=guard.velocity(q,np.eye(4),T,.02,posture_reference=q.copy(),**options)
c,names=guard.configuration(q);bar=SelfCollisionBarrier(1,gain=.1,safe_displacement_gain=0,d_min=.0065)
G,h=bar.compute_qp_inequalities(c,.02)
assert (G@(np.array([v0[x] for x in names])*.02)-h).max()>.001
assert (G@(np.array([v1[x] for x in names])*.02)-h).max()<1e-9
assert set(v1)==set(guard.owners['pink']) and not set(v1)&set(guard.owners['policy'])
assert max(abs(x) for x in v1.values())<=.02000001
assert guard.last_collision_report['barrier_type']=='pink.barriers.SelfCollisionBarrier'
'''
    result=subprocess.run([str(python),'-c',script,str(ROOT/'Robot_G1'),str(soft_costs).lower()],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_genuine_pink_new_soft_cost_api_preserves_default_and_camera_authority():
    script=r'''
import importlib.util,yaml,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
p=Path(__import__('sys').argv[1])
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
live=load('live_soft_cost',p/'wbc_backend.py');old=load('frozen_soft_cost',p/'results/adaptive_ground_sameprofile_01/sources/g1_backend.py')
io=yaml.safe_load((p/'config/height_policy_io.yaml').read_text());names=io['articulations']['robot']['joint_names'];home=dict(zip(names,io['articulations']['robot']['default_joint_pos']))
camera=np.eye(4);camera[:3,3]=[.2,.04,.11];camera[:3,:3]=np.column_stack([[0,-1,0],[0,0,-1],[1,0,0]])
make=lambda m:m.PinkUpperBody(io['actions'][0]['joint_names'],names,camera,p/'kinematics/g1_29dof.urdf',posture_home=home.copy(),posture_cost=.03,damping_cost=.02,waist_posture_cost=.15)
a,b=make(old),make(live);q=home.copy();q['waist_yaw_joint']=.12
target=a.predicted_camera(q,np.eye(4));target[:3,3]+=[.01,-.07,.02];target[:3,:3]=Rotation.from_rotvec([.1,.1,.25]).as_matrix()@target[:3,:3]
va=a.velocity(q,np.eye(4),target,.02);vb=b.velocity(q,np.eye(4),target,.02)
np.testing.assert_allclose([va[n] for n in a.owners['pink']],[vb[n] for n in b.owners['pink']],atol=1e-13,rtol=0.)
c=make(live);q=home.copy();initial=c.predicted_camera(q,np.eye(4));locked={n:q[n] for n in c.owners['policy']+c.owners['hold']}
target=initial.copy();target[:3,3]+=[.01,-.07,.02];target[:3,:3]=Rotation.from_rotvec([.1,.1,.25]).as_matrix()@target[:3,:3]
previous=dict.fromkeys(c.owners['pink'],0.)
for _ in range(250):
 v=c.velocity(q,np.eye(4),target,.02,posture_reference=home.copy(),posture_cost_override=.1,damping_cost_override=.1)
 assert set(v)==set(c.owners['pink']) and len(v)==15
 for n,u in v.items():
  assert abs(u)<=.5+1e-7 and abs(u-previous[n])<=.02+1e-7
  q[n]+=u*.02
 previous=v
actual=c.predicted_camera(q,np.eye(4));ep=np.linalg.norm(target[:3,3]-actual[:3,3]);er=Rotation.from_matrix(target[:3,:3]@actual[:3,:3].T).magnitude()
assert ep<.05 and er<.15,(ep,er)
assert np.linalg.norm(actual[:3,3]-initial[:3,3])>.025
assert Rotation.from_matrix(actual[:3,:3]@initial[:3,:3].T).magnitude()>.1
assert all(q[n]==value for n,value in locked.items())
assert c.last_posture_report['other_posture_cost']==.1 and c.last_posture_report['damping_cost']==.1
assert c.last_posture_report['optimized_upper_dofs']==15 and not c.last_posture_report['waist_yaw_locked']
for options in [dict(posture_cost_override=.2),dict(damping_cost_override=-1.)]:
 try:c.velocity(q,np.eye(4),target,.02,**options)
 except ValueError:pass
 else:raise AssertionError('invalid soft preference accepted')
'''
    result=subprocess.run([str(ROOT/'Robot_G1/.venv/bin/python'),'-c',script,str(ROOT/'Robot_G1')],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_pink_timeout_does_not_leave_an_unbounded_hold_only_success(monkeypatch):
    import time
    worker = object.__new__(g1.PinkProcess)
    worker.max_age_s = .06
    worker.pending = dict(request=dict(seq=1, stamp_s=0.), sent_wall_s=time.monotonic()-1.)
    worker.failure_count = 0; worker.needs_reset = False
    terminated = []
    worker.process = type('Worker', (), {'terminate': lambda self: terminated.append(True)})()
    monkeypatch.setattr(worker, '_receive', lambda wait: None)
    result = worker.poll(.02)
    assert not result['ready'] and 'deadline_exceeded' in result['status']
    assert terminated and worker.failure_count == 3 and worker.needs_reset


def recovery_worker(monkeypatch):
    import time
    from types import SimpleNamespace
    worker=object.__new__(g1.PinkProcess)
    worker.max_age_s=.06;worker.failure_count=0;worker.needs_reset=False
    worker._recovery_started_wall_s=None;worker.cancelled=None;worker._last_state_stamp=1.
    worker.pending=dict(request=dict(seq=4,stamp_s=1.),sent_wall_s=time.monotonic(),
        measured_camera=np.eye(4),measured_bodies=None)
    worker._seq=5;terminated=[]
    worker.process=SimpleNamespace(poll=lambda:None,terminate=lambda:terminated.append(True))
    sent=[]
    def send(request):
        request['seq']=worker._seq;worker._seq+=1;sent.append(request)
        worker.pending=dict(request=request,sent_wall_s=time.monotonic())
    monkeypatch.setattr(worker,'_send',send)
    return worker,sent,terminated


def test_pink_direct_recovery_clock_survives_fresh_resume_before_state_is_cleared(monkeypatch):
    import time
    worker,_,_=recovery_worker(monkeypatch);context=task_context_fixture('direct_clock')
    clock=[100.04];monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    worker.pending.update(sent_wall_s=100.)
    worker.pending['request']['task_context']=context
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=False,seq=4,stamp_s=1.,task_context=context,error='real solver failure'))
    failed=worker.poll(1.02,current_context=context)
    report=failed['recovery_clock']
    assert report['event']=='failure_detected' and report['fault_detected_wall_s']==100.04
    assert report['recovery_started_wall_s']==100.04 and report['recovery_started_stamp_s']==1.02
    assert report['recovery_started_context']==context and report['elapsed_wall_s']==0.
    assert report['budget_s']==pytest.approx(.18) and worker.failure_count==1
    assert worker.submit({},np.eye(4),None,1.02,np.eye(4),task_context=context)
    clock[0]=100.06
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=5,stamp_s=1.02,task_context=context,velocity={}))
    resumed=worker.poll(1.04,current_context=context,execution_horizon_s=.02)
    direct=resumed['recovery_clock']
    assert resumed['ready'] and direct['event']=='resumed'
    assert direct['sample_wall_s']==100.06 and direct['elapsed_wall_s']==pytest.approx(.02)
    assert direct['fault_detected_wall_s']==100.04 and direct['poll_interval_wall_s']==pytest.approx(.02)
    assert worker._recovery_started_wall_s is None and worker._recovery_started_context is None
    assert failed['recovery_clock']==report  # Earlier evidence is not mutated by the reset.


def test_pink_direct_clock_distinguishes_main_poll_stall_from_new_recovery_window(monkeypatch):
    import time
    worker,_,_=recovery_worker(monkeypatch)
    monkeypatch.setattr(time,'monotonic',lambda:100.4)
    worker.pending['sent_wall_s']=100.;worker._last_poll_wall_s=100.
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=4,stamp_s=1.,velocity={}))
    packet=worker.poll(1.02)
    assert not packet['ready'] and packet['status']=='stale_or_failed_result_dropped'
    report=packet['recovery_clock']
    assert report['poll_interval_wall_s']==pytest.approx(.4)
    assert report['recovery_started_wall_s']==report['fault_detected_wall_s']==100.4
    assert report['elapsed_wall_s']==0. and packet['wall_age_s']==pytest.approx(.4)
    assert report['original_motion_lease_s']==.06


def test_g1_gc_timing_observer_never_changes_collection_policy_and_removes_callback(monkeypatch):
    import gc
    import time
    runner=module('g1_gc_diagnostic_test',ROOT/'Robot_G1/adaptive_runner.py')
    before=(gc.isenabled(),gc.get_threshold(),list(gc.callbacks));clock=[20.]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    recorder=runner.GarbageCollectionTiming(True)
    with pytest.raises(RuntimeError),recorder:
        assert recorder._observe in gc.callbacks
        recorder._observe('start',dict(generation=2));clock[0]=20.4
        recorder._observe('stop',dict(generation=2,collected=123,uncollectable=0))
        raise RuntimeError('test cleanup')
    assert (gc.isenabled(),gc.get_threshold(),list(gc.callbacks))==before
    assert recorder.events==[dict(generation=2,start_wall_s=20.,end_wall_s=20.4,
        elapsed_wall_s=pytest.approx(.4),collected=123,uncollectable=0)]


def task_context_fixture(name):
    return g1.context_helper().task_context(dict(schema='jev_wbc/v1',kind='task',task_id=name,
        camera_trajectory=dict(test_path=name)))


@pytest.mark.parametrize('corrupt',[None,'evidence_sha','acquisition','seq','task_sha','binding_type'])
def test_following_real_pending_request_fingerprint_must_match_report_and_outer_metadata(monkeypatch,corrupt):
    import copy
    import hashlib
    worker,_,_=recovery_worker(monkeypatch);context=task_context_fixture('guarded')
    evidence=dict(acquisition_stamp_s=1.,measured_qd_rad_s=[0.]*15,source='transport_unit_fixture')
    expected=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    worker.pending['request'].update(task_context=context,following_evidence=evidence)
    report=dict(evidence_sha256=expected,acquisition_stamp_s=1.,worker_request_binding=dict(
        seq=4,stamp_s=1.,task_sha256=context['task_sha256'],evidence_sha256=expected))
    if corrupt=='evidence_sha':report['evidence_sha256']='0'*64
    if corrupt=='acquisition':report['acquisition_stamp_s']=.98
    if corrupt=='seq':report['worker_request_binding']['seq']=3
    if corrupt=='task_sha':report['worker_request_binding']['task_sha256']='0'*64
    if corrupt=='binding_type':report['worker_request_binding']=[]
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=4,stamp_s=1.,task_context=context,
        following_reserve_report=copy.deepcopy(report),velocity={'waist_yaw_joint':.01},predicted_camera=np.eye(4).tolist()))
    packet=worker.poll(1.02,current_context=context,execution_horizon_s=.02)
    if corrupt is None:
        assert packet['ready'] and worker.failure_count==0
    else:
        assert not packet['ready'] and packet['failure_kind']=='transport_failure'
        assert worker.failure_count==1 and solver_feedback.public_support(packet)==dict(reason='solver_failure',feasible=False)


def test_pink_still_fresh_previous_task_is_discarded_without_solver_fault(monkeypatch):
    worker,sent,_=recovery_worker(monkeypatch);a=task_context_fixture('A');b=task_context_fixture('B')
    worker.pending['request']['task_context']=a
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=4,stamp_s=1.,task_context=a,
        velocity={'waist_yaw_joint':.2}))
    packet=worker.poll(1.02,current_context=b)
    assert not packet['ready'] and packet['status']=='superseded_task_hold_upper_targets'
    assert 'velocity' not in packet and worker.failure_count==0 and worker.needs_reset
    assert solver_feedback.public_support(packet)==dict(reason='command_stale',feasible=None)
    assert worker.submit({},np.eye(4),None,1.02,np.eye(4),task_context=b)
    assert sent[-1]['reset_command'] and sent[-1]['task_context']==b
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=5,stamp_s=1.02,task_context=b,
        velocity={'waist_yaw_joint':.01}))
    assert worker.poll(1.04,current_context=b)['ready']


def test_pink_task_transition_cancels_even_a_fresh_recovery_request_without_starting_fault_timer(monkeypatch):
    worker,sent,_=recovery_worker(monkeypatch);a=task_context_fixture('A');b=task_context_fixture('B')
    worker.pending['request']['task_context']=a;worker.pending['recovery_request']=True
    monkeypatch.setattr(worker,'_receive',lambda wait:None)
    packet=worker.poll(1.02,current_context=b)
    assert packet['replacement_allowed'] and packet['status']=='superseded_task_hold_upper_targets'
    assert worker.pending is None and worker.cancelled['cancellation_reason']=='superseded_task'
    assert worker.failure_count==0 and worker._recovery_started_wall_s is None
    assert worker.submit({},np.eye(4),None,1.02,np.eye(4),task_context=b)
    assert len(sent)==1 and sent[-1]['reset_command']


@pytest.mark.parametrize('echo',[None,task_context_fixture('wrong')])
def test_pink_current_task_missing_or_wrong_echo_is_transport_failure(monkeypatch,echo):
    worker,_,_=recovery_worker(monkeypatch);context=task_context_fixture('current')
    worker.pending['request']['task_context']=context
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=4,stamp_s=1.,task_context=echo))
    packet=worker.poll(1.02,current_context=context)
    assert not packet['ready'] and packet['status']=='reply_context_mismatch'
    assert worker.failure_count==1 and worker.needs_reset and 'velocity' not in packet
    assert solver_feedback.public_support(packet)==dict(reason='solver_failure',feasible=False)


def test_pink_real_worker_echoes_current_task_and_rejects_previous_fresh_task():
    pytest.importorskip('pinocchio');pytest.importorskip('pink')
    import time
    _,dump,metadata=actual_gravity_fixture();names=dump['joint_names']
    legs=[n for n in names if any(t in n for t in ('hip_','knee_','ankle_'))]
    worker=g1.PinkProcess(legs,names,np.eye(4),ROOT/'Robot_G1/kinematics/g1_29dof.urdf',
        source_path=ROOT/'Robot_G1/wbc_backend.py')
    a=task_context_fixture('physical_A');b=task_context_fixture('physical_B')
    try:
        q=metadata['calibration_joint_positions'];worker.warmup(q)
        assert worker.submit(q,np.eye(4),None,1.,np.eye(4),task_context=a)
        packet=worker.poll(1.02,current_context=b)
        assert not packet['ready'] and packet['status']=='superseded_task_hold_upper_targets'
        assert worker.failure_count==0
        assert worker.submit(q,np.eye(4),None,1.02,np.eye(4),task_context=b)
        deadline=time.monotonic()+.055
        while time.monotonic()<deadline:
            packet=worker.poll(1.04,current_context=b)
            if packet['ready']:break
        assert packet['ready'] and packet['task_context']==b and packet['task_context_status']=='same_task'
        assert worker.failure_count==0
        assert solver_feedback.public_support(packet)==dict(reason='accepted',feasible=None)
    finally:worker.close()


def test_pink_cancelled_work_never_resumes_but_new_measurement_can_recover_in_original_lease(monkeypatch):
    worker,sent,_=recovery_worker(monkeypatch)
    monkeypatch.setattr(worker,'_receive',lambda wait:None)
    first=worker.poll(1.02)
    assert not first['ready'] and first['replacement_allowed']
    assert worker.pending is None and worker.cancelled['request']['seq']==4
    measured=pose([.01,0.,0.],[0.,0.,0.])
    with pytest.raises(ValueError):worker.submit({},np.eye(4),None,1.,measured)
    assert worker.submit({},np.eye(4),None,1.02,measured)
    assert len(sent)==1 and sent[0]['reset_command']
    replies=iter([dict(ok=True,seq=4,stamp_s=1.,velocity={'waist_yaw_joint':9.}),
                  dict(ok=True,seq=5,stamp_s=1.02,velocity={'waist_yaw_joint':.01})])
    monkeypatch.setattr(worker,'_receive',lambda wait:next(replies))
    result=worker.poll(1.04)
    assert result['ready'] and result['seq']==5 and result['velocity']['waist_yaw_joint']==.01
    assert result['discarded_results'][0]['seq']==4
    np.testing.assert_array_equal(result['acquisition_camera'],measured)
    assert worker.pending is None and worker.cancelled is None and worker._recovery_started_wall_s is None


def test_pink_queue_is_bounded_and_only_the_fresh_replacement_may_wait_inside_original_lease(monkeypatch):
    worker,sent,_=recovery_worker(monkeypatch)
    monkeypatch.setattr(worker,'_receive',lambda wait:None)
    worker.poll(1.02);assert worker.submit({},np.eye(4),None,1.02,np.eye(4))
    second=worker.poll(1.04)
    assert second['outstanding_requests']==2 and not second['replacement_allowed']
    assert not worker.submit({},np.eye(4),None,1.04,np.eye(4)) and len(sent)==1
    replies=iter([dict(ok=True,seq=4,stamp_s=1.),dict(ok=True,seq=5,stamp_s=1.02,velocity={'waist_yaw_joint':.1})])
    monkeypatch.setattr(worker,'_receive',lambda wait:next(replies))
    result=worker.poll(1.06)
    assert result['ready'] and result['seq']==5 and result['acquisition_stamp_s']==1.02
    assert result['discarded_results'][0]['seq']==4
    assert worker.submit({},np.eye(4),None,1.06,np.eye(4)) and not sent[-1]['reset_command']


@pytest.mark.parametrize('wall_age,sim_age',[(.061,.02),(.02,.061),(.02,-.001)])
def test_pink_fresh_replacement_still_requires_original_wall_and_sim_clock_lease(monkeypatch,wall_age,sim_age):
    import time
    worker,_,_=recovery_worker(monkeypatch)
    worker.pending['sent_wall_s']=time.monotonic()-wall_age
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=4,stamp_s=1.,velocity={'waist_yaw_joint':.1}))
    result=worker.poll(1.+sim_age)
    assert not result['ready'] and 'velocity' not in result and worker.needs_reset


@pytest.mark.parametrize('wall_age,sim_age,valid',[(.04,.04,True),(.06,.04,False),(.04,.06,False),(.041,.04,False),(-.001,.02,False)])
def test_pink_original_dual_lease_must_cover_entire_declared_execution_tick(monkeypatch,wall_age,sim_age,valid):
    import time
    worker,_,_=recovery_worker(monkeypatch);worker.pending['sent_wall_s']=100.
    monkeypatch.setattr(time,'monotonic',lambda:100.+wall_age)
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=4,stamp_s=1.,velocity={'waist_yaw_joint':.1}))
    packet=worker.poll(1.+sim_age,execution_horizon_s=.02)
    assert packet['ready']==valid
    if valid:
        assert packet['lease_deadline_sim_s']==pytest.approx(1.06)
        assert packet['accepted_wall_age_s']==pytest.approx(.04)
    else:assert 'velocity' not in packet


def test_pink_progress_without_any_fresh_recovery_cannot_extend_original_deadline(monkeypatch):
    import time
    worker,_,terminated=recovery_worker(monkeypatch)
    worker._recovery_started_wall_s=time.monotonic()-.181
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=4,stamp_s=1.))
    result=worker.poll(1.02)
    assert 'deadline_exceeded' in result['status'] and terminated and worker.failure_count==3


def test_pink_binary_framing_drains_two_buffered_packets_without_waiting_for_more_bytes():
    import os
    from types import SimpleNamespace
    read_fd,write_fd=os.pipe()
    try:
        os.write(write_fd,b'{"seq":4}\n{"seq":5}\n')
        worker=object.__new__(g1.PinkProcess);worker._receive_buffer=b''
        with os.fdopen(read_fd,'rb',buffering=0) as stream:
            worker.process=SimpleNamespace(stdout=stream)
            assert worker._receive(0.)==dict(seq=4)
            assert worker._receive(0.)==dict(seq=5)
    finally:os.close(write_fd)


def test_pink_real_child_double_response_drains_local_bytes_before_selecting_fd_again():
    child=subprocess.Popen([sys.executable,'-u','-c',
        'import os; os.write(1,b\'{"seq":4}\\n{"seq":5}\\n\')'],stdout=subprocess.PIPE,bufsize=0)
    worker=object.__new__(g1.PinkProcess);worker._receive_buffer=b'';worker.process=child
    try:
        assert worker._receive(2.)==dict(seq=4)
        assert worker._receive(0.)==dict(seq=5)
        assert worker._receive_buffer==b''
    finally:child.wait(timeout=2.);child.stdout.close()


def test_pink_real_child_partial_line_never_blocks_past_bounded_poll():
    import time
    child=subprocess.Popen([sys.executable,'-u','-c',
        'import os; os.write(1,b\'{"seq":4\'); os.read(0,1); os.write(1,b\'}\\n\')'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
    worker=object.__new__(g1.PinkProcess);worker._receive_buffer=b'';worker.process=child
    try:
        start=time.monotonic();assert worker._receive(.05) is None
        assert time.monotonic()-start<.5
        child.stdin.write(b'x');child.stdin.flush()
        assert worker._receive(2.)==dict(seq=4)
    finally:
        if child.poll() is None:child.terminate()
        child.wait(timeout=2.);child.stdin.close();child.stdout.close()


def test_pink_main_stall_with_completed_expired_reply_drops_it_then_requires_fresh_state(monkeypatch):
    import time
    worker,sent,terminated=recovery_worker(monkeypatch)
    worker.pending['sent_wall_s']=time.monotonic()-.181
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=4,stamp_s=1.,velocity={'waist_yaw_joint':.5}))
    result=worker.poll(1.2)
    assert not result['ready'] and result['status']=='stale_or_failed_result_dropped'
    assert not terminated and 'velocity' not in result and worker.pending is None
    assert worker.submit({},np.eye(4),None,1.2,np.eye(4)) and sent[-1]['reset_command']
    monkeypatch.setattr(worker,'_receive',lambda wait:dict(ok=True,seq=5,stamp_s=1.2,velocity={'waist_yaw_joint':.01}))
    resumed=worker.poll(1.22)
    assert resumed['ready'] and resumed['seq']==5 and resumed['velocity']['waist_yaw_joint']==.01


def test_transient_upper_hold_brakes_at_native_slew_without_unwinding_or_accumulating_integral():
    controller=servo.VelocityServo()
    controller.command=np.array([.1,-.1,.4]);controller.integral=np.array([.05,-.05,.25])
    initial=controller.integral.copy();previous=controller.command.copy()
    for _ in range(3):
        controller.update([.1,-.1,.12],[.07,-.07,.03],.02)
        actual=np.clip(np.zeros(3),previous-controller.slew*.02,previous+controller.slew*.02)
        controller.applied(actual,transient_hold=True)
        np.testing.assert_array_equal(controller.integral,initial)
        assert np.all(np.abs(actual-previous)<=controller.slew*.02+1e-12)
        previous=actual
    resumed=controller.update([.1,-.1,.12],[.07,-.07,.03],.02)
    assert np.all(np.abs(resumed-previous)<=controller.slew*.02+1e-12)
    assert resumed[2]>.3  # Recovery starts from the actual bounded brake, never from zero.


def test_servo_ordinary_updates_and_workspace_vetoes_preserve_frozen_executed_behavior():
    executed=module('executed_velocity_servo',ROOT/'Robot_G1/results/adaptive_waist_settle_01/sources/velocity_servo.py')
    old,new=executed.VelocityServo(),servo.VelocityServo()
    generator=np.random.default_rng(87)
    for step in range(300):
        desired=generator.uniform(-.15,.15,3);measured=generator.uniform(-.2,.2,3)
        np.testing.assert_array_equal(old.update(desired,measured,.02),new.update(desired,measured,.02))
        if step%11==0:
            applied=new.command*.3;old.applied(applied);new.applied(applied)
        np.testing.assert_array_equal(old.integral,new.integral)
        np.testing.assert_array_equal(old.command,new.command)


@pytest.mark.parametrize('held,reply_seq,reply_stamp,current_stamp', [
    (True, 4, 1., 1.02), (False, 3, 1., 1.02), (False, 4, .5, 1.02), (False, 4, 1., 1.2),
])
def test_pink_late_or_mismatched_response_cannot_resume_old_velocity(
        monkeypatch, held, reply_seq, reply_stamp, current_stamp):
    import time
    worker = object.__new__(g1.PinkProcess)
    worker.max_age_s = .06; worker.failure_count = 0; worker.needs_reset = False
    worker.pending = dict(request=dict(seq=4, stamp_s=1.), sent_wall_s=time.monotonic(),
                          held_since_submit=held, measured_camera=np.eye(4))
    monkeypatch.setattr(worker, '_receive', lambda wait: dict(ok=True, seq=reply_seq, stamp_s=reply_stamp,
        velocity={'waist_yaw_joint': .2}, predicted_camera=np.eye(4).tolist()))
    result = worker.poll(current_stamp)
    assert not result['ready'] and 'velocity' not in result
    assert worker.pending is None and worker.needs_reset and worker.failure_count == 1


def test_measured_velocity_servo_removes_deadband_bias_reverses_and_brakes_without_windup():
    controller = servo.VelocityServo()
    measured = np.zeros(3); previous = np.zeros(3)
    endpoints = []
    for target in ([.10, -.08, .12], [-.10, .08, -.12], [0., 0., 0.]):
        for _ in range(600):
            applied = controller.update(target, measured, .02)
            assert np.all(np.abs(applied) <= controller.limits+1e-10)
            assert np.all(np.abs(applied-previous) <= controller.slew*.02+1e-10)
            response = .7*np.sign(applied)*np.maximum(np.abs(applied)-.04, 0.)
            measured += .1*(response-measured)
            previous = applied
        endpoints.append(measured.copy())
    np.testing.assert_allclose(endpoints, [[.10, -.08, .12], [-.10, .08, -.12], [0., 0., 0.]], atol=.002)
    assert np.max(np.abs(controller.integral)) < .1


def test_support_guard_back_calculation_does_not_accumulate_hidden_velocity_request():
    controller = servo.VelocityServo(limits=(.2, .2, .2))
    for _ in range(300):
        controller.update([.2, 0., 0.], [0., 0., 0.], .02)
        controller.applied([0., 0., 0.])
    assert np.max(np.abs(controller.integral)) < .03
    assert np.max(np.abs(controller.update([0., 0., 0.], [0., 0., 0.], .02))) < .011


def test_stop_anchor_uses_physical_body_axes_including_nonzero_world_heading():
    current = pose([1., 2., .7], [0., 0., np.pi/2])
    anchor = pose([1.01, 1.98, .7], [0., 0., np.pi/2+.03])
    np.testing.assert_allclose(servo.stop_pose_error(anchor, current), [-.02, -.01, .03], atol=1e-12)


def test_measured_brake_anchor_latch_prevents_same_tick_stand_exit_then_detects_new_drift():
    anchor=np.eye(4);current=pose([.0268,.0156,0.],[0.,0.,0.]);velocity=[.007,0.,.005]
    old,_,negative=servo.parking_transition(anchor,current,velocity,2.,False,reset_anchor_on_entry=False)
    assert not old and negative['entered'] and negative['exited']
    parked,latched,event=servo.parking_transition(anchor,current,velocity,2.,False,reset_anchor_on_entry=True)
    assert parked and event['entered'] and not event['exited']
    np.testing.assert_array_equal(latched,current)
    later=current.copy();later[0,3]+=.026
    parked,_,event=servo.parking_transition(latched,later,velocity,2.5,True,reset_anchor_on_entry=True)
    assert not parked and event['exited'] and not event['entered']
    parked,_,_=servo.parking_transition(anchor,current,[.03,0.,0.],2.,False,reset_anchor_on_entry=True)
    assert not parked


def test_go2_camera_priority_guard_allows_supported_motion_and_limits_conflicting_static_task():
    guard = coordinator.CameraCoordinator(np.eye(4), backend=go2)
    link = pose([0., 0., .35], [0., 0., 0.]); body = pose([0., 0., .28], [0., 0., 0.])
    def static(t):
        return pose([.5, 0., .5], [0., 0., 0.])
    def moving(t):
        return pose([.5+.1*t, 0., .5], [0., 0., 0.])
    conflict = guard.command(0., static, link, body, [.2, 0., 0.], [0., 0., 0.])
    assert conflict['camera_supported'] and not conflict['base_velocity_feasible']
    assert .01 < conflict['base_command'][0] < .15
    assert conflict['camera_reference_changed'] is False
    support = guard.command(0., moving, link, body, [.1, 0., 0.], [.1, 0., 0.])
    assert support['camera_supported'] and support['base_velocity_feasible']
    np.testing.assert_allclose(support['base_command'], [.1, 0., 0.], atol=1e-8)


def test_go2_camera_priority_horizon_brakes_before_leaving_static_world_workspace():
    guard = coordinator.CameraCoordinator(np.eye(4), backend=go2)
    link = pose([0., 0., .35], [0., 0., 0.]); body = pose([0., 0., .28], [0., 0., 0.])
    target = pose([.5, 0., .5], [0., 0., 0.]); original = target.copy()
    measured = np.zeros(3); dt = .02; positions = []
    for step in range(200):
        result = guard.command(step*dt, lambda t: target, link, body, [.2, 0., 0.], measured)
        assert result['camera_supported']
        measured += dt/.15*(result['base_command']-measured)
        link[:2, 3] += measured[:2]*dt; body[:2, 3] += measured[:2]*dt
        positions.append(go2.camera_to_mixed(target, link, np.eye(4))[0])
    assert min(positions) > .4 and link[0, 3] > .02
    assert abs(measured[0]) < .005
    np.testing.assert_array_equal(target, original)


def test_compiled_policy_domain_matches_pinned_file_checks_without_runtime_parsing():
    domain=go2.PolicyDomain()
    for xyz,euler in [([.5,0.,.5],[0.,0.,0.]),([-.2,0.,.5],[0.,0.,0.]),([.5,0.,.5],[0.,-.9,0.])]:
        command=np.r_[xyz,Rotation.from_euler('xyz',euler).as_quat()[[3,0,1,2]]]
        compiled,direct=domain.check(command),go2.check_upstream_domain(command)
        for key in ['status','supported','feasible','violations','sources_sha256']:
            assert compiled[key]==direct[key]


def test_actual_fcl_worker_distinguishes_original_sphere_clearance_and_overlap(tmp_path):
    import select
    python=ROOT/'Robot_TIAGo/wbc/.venv/bin/python'
    if not python.is_file():pytest.skip('isolated native FCL worker interpreter unavailable')
    collision=module('legged_collision_test',ROOT/'Robot_G1/legged_collision.py')
    metadata=tmp_path/'shapes.json'
    metadata.write_text(json.dumps(dict(shapes=[dict(name=n,body=n,kind='sphere',radius=.1,center_body=[0.,0.,0.])
        for n in ['a','b']],allowed_pairs=[])))
    worker=collision.DistanceProcess(python,ROOT/'Robot_G1/legged_collision.py',metadata,tmp_path/'stderr.log')
    try:
        for stamp,separation,expected in [(0.,.25,'pass'),(.02,.15,'fail')]:
            assert worker.submit(stamp,{'a':np.eye(4),'b':pose([separation,0.,0.],[0.,0.,0.])})
            assert select.select([worker.process.stdout],[],[],1.)[0]
            result=worker.poll(stamp+.02)
            assert result['status']==expected
            assert result['minimum_clearance_lower_bound_m']==pytest.approx(separation-.2,abs=1e-6)
    finally:worker.close()


def test_native_usd_extraction_honors_nearest_disabled_api_guide_and_instance_proxies():
    import os
    envroot=Path('/home/w84384651/miniconda3/envs/env_isaaclab')
    site=envroot/'lib/python3.11/site-packages/isaacsim'
    usd=next(iter((site/'extscache').glob('omni.usd.libs-*/pxr')),None)
    if usd is None:pytest.skip('native USD CPU distribution unavailable')
    libs=usd.parent
    environment=dict(os.environ,LD_LIBRARY_PATH=f'{envroot}/lib:{libs}/bin:{site}/kit',PYTHONPATH=str(libs))
    script=r'''
import importlib.util,numpy as np
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics
p=Path(__import__('sys').argv[1]);s=importlib.util.spec_from_file_location('actual_extract',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.)
UsdGeom.Xform.Define(stage,'/Robot');body=UsdGeom.Xform.Define(stage,'/Robot/body');UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
parent=UsdGeom.Xform.Define(stage,'/Robot/body/collisions');UsdPhysics.CollisionAPI.Apply(parent.GetPrim())
live=UsdGeom.Sphere.Define(stage,'/Robot/body/collisions/live');live.CreateRadiusAttr(.1);live.CreatePurposeAttr('guide')
disabled=UsdGeom.Sphere.Define(stage,'/Robot/body/collisions/disabled');UsdPhysics.CollisionAPI.Apply(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
prototype=UsdGeom.Xform.Define(stage,'/Prototype');UsdPhysics.RigidBodyAPI.Apply(prototype.GetPrim())
sphere=UsdGeom.Sphere.Define(stage,'/Prototype/sphere');sphere.CreateRadiusAttr(.1);UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
instance=stage.DefinePrim('/Robot/inst','Xform');instance.GetReferences().AddInternalReference('/Prototype');instance.SetInstanceable(True)
metadata=m.extract_collision_geometry(stage,'/Robot')
assert {x['name'] for x in metadata['shapes']}=={'/Robot/body/collisions/live','/Robot/inst/sphere'},metadata
assert {x['body'] for x in metadata['shapes']}=={'body','inst'}
'''
    result=subprocess.run([str(envroot/'bin/python'),'-c',script,str(ROOT/'Robot_G1/legged_collision.py')],env=environment,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_go2_motor_caps_come_from_same_pinned_model_and_cover_all_18_joints():
    limits=module('piper_actuator_limits',ROOT/'Robot_Go2/actuator_limits.py')
    result=limits.load_motor_limits(ROOT/'Robot_Go2/vendor/LeggedManip_Lab/mujoco/robots/go2_piper/go2piper.xml')
    caps=result['effort_limits_nm']
    assert len(caps)==18 and caps['FL_hip_joint']==23.7 and caps['RR_calf_joint']==45.43
    assert [caps[f'joint{i}'] for i in range(1,7)]==[20.,20.,15.,7.,5.,5.]


def test_feedback_rejection_reports_domain_of_applied_command_separately():
    link=pose([0.,0.,.35],[0.,0.,0.]);reference=pose([.405,0.,.5],[0.,0.,0.])
    measured=reference.copy();measured[0,3]+=.04
    actual,check=coordinator.feedback_native_command(go2,reference,measured,link,np.eye(4))
    assert check['status']=='inside_outer_domain_unverified'
    assert check['rejected_feedback_correction']['domain_check']['status']=='out_of_policy_domain'
    np.testing.assert_allclose(actual,go2.camera_to_mixed(reference,link,np.eye(4)))


def test_native_parking_holds_input_only_after_braking_and_releases_on_world_error():
    parking=coordinator.NativeCameraParking();T=pose([.5,0.,.5],[0.,0.,0.]);body=np.eye(4)
    native=np.array([.5,0.,.5,1.,0.,0.,0.]);original=T.copy()
    _,status=parking.update(native,T,T,body,[.1,0.,0.],parking=True)
    assert not status['active']
    held,status=parking.update(native,T,T,body,[0.,0.,0.],parking=True)
    assert status['active']
    perturbed=native.copy();perturbed[0]+=.01
    result,status=parking.update(perturbed,T,T,body,[0.,0.,0.],parking=True)
    np.testing.assert_array_equal(result,held)
    camera=T.copy();camera[0,3]+=.041
    result,status=parking.update(perturbed,T,camera,body,[0.,0.,0.],parking=True)
    assert not status['active'];np.testing.assert_array_equal(result,perturbed)
    np.testing.assert_array_equal(T,original)


def test_nontrivial_six_axis_fixture_defeats_initial_pose_hold_without_changing_tolerance():
    runner=module('legged_fixture_test',ROOT/'Robot_G1/adaptive_runner.py')
    samples=[runner.six_axis_excursion(t,16.,True) for t in np.linspace(0.,16.,801)]
    p=np.array([x[0] for x in samples]);r=np.array([x[1] for x in samples])
    assert np.all(np.ptp(p,axis=0)>.06) and np.linalg.norm(np.ptp(p,axis=0))>.10
    assert np.all(np.ptp(r,axis=0)>.25)
    assert np.sqrt(np.mean(np.sum(p*p,axis=1)))>.05
    assert np.sqrt(np.mean(np.sum(r*r,axis=1)))>.15
    np.testing.assert_allclose(p[[0,-1]],0.,atol=1e-15)
    np.testing.assert_allclose(r[[0,-1]],0.,atol=1e-15)


@pytest.mark.parametrize('initial,future,measured', [(-.03,-.18,-.02),(-.18,0.,-.1),(0.,-.18,0.)])
def test_delayed_ramp_prediction_matches_independent_numerical_response(initial,future,measured):
    duration=.8;dt=.00005;position=0.;v=measured
    for t in np.arange(0.,duration,dt):
        if t<.15:position+=measured*dt;continue
        u=initial+np.clip(future-initial,-.5*(t-.15),.5*(t-.15))
        position+=v*dt;v+=dt/.35*(.55*u-v)
    exact=coordinator.CameraCoordinator.delayed_ramp_displacement(measured,initial,future,duration)
    assert exact==pytest.approx(position,abs=1e-5)


def test_measured_backward_feedforward_and_horizon_keep_original_world_path_in_outer_domain():
    guard=coordinator.CameraCoordinator(np.eye(4),backend=go2,response_profile='measured_backward_v1')
    controller=servo.VelocityServo(limits=(.2,.2,.2),**servo.go2_profile('measured_backward_v1'))
    link=pose([0.,0.,.35],[0.,0.,0.]);body=pose([0.,0.,.28],[0.,0.,0.])
    previous=np.zeros(3);measured=np.zeros(3);history=[np.zeros(3) for _ in range(8)];xs=[];velocities=[]
    def target(t):return pose([.5-.1*min(t,5.),0.,.5],[0.,0.,0.])
    for step in range(250):
        stamp=step*.02;preferred=controller.update([-.1,0.,0.],measured,.02)
        support=guard.command(stamp,target,link,body,preferred,measured,max_lookahead_s=5.-stamp,
            future_policy_velocity=[-.1/.55,0.,0.],previous_applied=previous,command_dt=.02)
        assert support['camera_supported']
        actual=np.asarray(support['base_command']);assert np.max(np.abs(actual-previous))<=.010000001
        assert np.max(np.abs(actual))<=.2
        controller.applied(actual);previous=actual;history.append(actual);delayed=history.pop(0)
        guard.observe_applied_input(stamp,actual)
        measured+=.02/.35*(.55*delayed-measured)
        link[:2,3]+=measured[:2]*.02;body[:2,3]+=measured[:2]*.02
        xs.append(go2.camera_to_mixed(target(stamp+.02),link,np.eye(4))[0]);velocities.append(measured[0])
    assert min(xs)>.4 and abs(np.mean(velocities[125:])+.1)<.025


def test_domain_recovery_never_sends_outside_ee_and_recovers_from_fresh_state():
    guard=coordinator.NativeDomainRecovery(go2.PolicyDomain(),go2)
    guard.seed([.5,0.,.5,1.,0.,0.,0.]);link=pose([0.,0.,.35],[0.,0.,0.]);body=pose([0.,0.,.28],[0.,0.,0.])
    original=pose([.35,0.,.5],[0.,0.,0.]);unchanged=original.copy();previous=np.zeros(3);recovered=False
    for step in range(100):
        requested=go2.camera_to_mixed(original,link,np.eye(4))
        native,base,report=guard.update(requested,link,body,step*.02,previous,.02)
        assert go2.check_upstream_domain(native)['status']=='inside_outer_domain_unverified'
        if report.get('recovered_from_fresh_measurement'):recovered=True;break
        assert report['active'] and np.max(np.abs(base-previous))<=.010000001
        previous=base;link[:2,3]+=.55*base[:2]*.02;body[:2,3]+=.55*base[:2]*.02
    assert recovered;np.testing.assert_array_equal(original,unchanged)


def test_domain_recovery_brakes_after_fixed_budget_and_does_not_invent_height_control():
    guard=coordinator.NativeDomainRecovery(go2.PolicyDomain(),go2);guard.seed([.5,0.,.5,1.,0.,0.,0.])
    previous=np.zeros(3)
    for step in range(150):
        native,base,report=guard.update([-10.,0.,.5,1.,0.,0.,0.],np.eye(4),np.eye(4),step*.02,previous,.02)
        assert np.max(np.abs(base-previous))<=.010000001;previous=base
    assert report['recovery_timed_out'] and np.max(np.abs(base))==0.
    native,base,report=guard.update([.5,0.,.9,1.,0.,0.,0.],np.eye(4),np.eye(4),3.,previous,.02)
    assert report['unsupported_recovery_axes']==['z_world'] and not np.any(base)


def test_bounded_upper_reference_holds_position_for_zero_velocity_and_unwinds_lead():
    module_ref=module('upper_reference_test',ROOT/'Robot_G1/upper_reference.py')
    reference=module_ref.UpperReference(['a','b'],[0.,0.],[[-1.,1.],[-1.,1.]])
    actual=np.zeros(2)
    for _ in range(100):
        target,report=reference.update(actual,[.2,-.1],.02)
        actual+=(target-actual)*.2
        assert np.max(np.abs(target-actual))<=.05
    fixed=target.copy()
    for _ in range(100):
        actual+=(fixed-actual)*.2
        target,report=reference.update(actual,[0.,0.],.02)
        np.testing.assert_allclose(target,fixed,atol=1e-12)
    target,report=reference.update(actual-.2,[0.,0.],.02)
    assert report['antiwindup_active'] and report['maximum_reference_lead_rad']<=.050000001
    with pytest.raises(ValueError):reference.update(actual,[.51,0.],.02)


@pytest.mark.parametrize('delay_s,gain',[(.2,.8),(.4,1.),(.6,1.2)])
def test_low_gain_g1_servo_handles_delayed_velocity_response_reversal_and_stop(delay_s,gain):
    controller=servo.VelocityServo(**servo.g1_profile('low_gain'))
    dt=.02;history=[np.zeros(3) for _ in range(round(delay_s/dt))];measured=np.zeros(3);end=[]
    for desired in [[.1,0.,0.],[-.1,0.,0.],[0.,0.,0.]]:
        for _ in range(1200):
            command=controller.update(desired,measured,dt)
            history.append(command);delayed=history.pop(0)
            measured+=dt/.2*(gain*delayed-measured)
            assert np.isfinite(measured).all() and np.max(np.abs(measured))<.3
        end.append(measured.copy())
    np.testing.assert_allclose(end,[[.1,0.,0.],[-.1,0.,0.],[0.,0.,0.]],atol=.002)
