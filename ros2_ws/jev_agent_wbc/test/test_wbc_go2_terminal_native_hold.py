"""Fixed-clock native7 transaction and narrowly verified measured hold calibration."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from jev_agent_wbc.wbc_protocol import TimedTask,CameraTrajectory,semantic_camera
from jev_agent_wbc.representation import Pose
ROOT=Path(__file__).resolve().parents[1]
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
h=load('hold_unit',ROOT/'Robot_Go2/terminal_native_hold.py')
z=load('hold_seed_unit',ROOT/'test/test_wbc_go2_bumpless_z_seed.py')


def setup_hold():
 pose=semantic_camera(Pose.from_rt(np.eye(3),[.5,.1,.52]))
 task=TimedTask('previous_task',0.,'oldobs',CameraTrajectory(0.,(0.,2.),(pose,pose)),.05,.15)
 obj=h.TerminalNativeHold(task)
 native=np.array([.5,.1,.5,1,0,0,0],dtype=np.float32)
 args=dict(stamp_s=4.,measurement_stamp_s=4.,previous_consumed_native=native.copy(),previous_issue_stamp_s=3.98,
  previous_consumption_seq=199,previous_body_input=np.zeros(3),issued_body_input=np.zeros(3),domain=z.backend.PolicyDomain(),
  collision_evidence=dict(status='pass',stamp_s=4.),original_support_ok=True,world_position_error_m=.01,world_orientation_error_rad=.02)
 return obj,native,args


def test_fixed_hold_uses_previously_consumed_float32_and_commits_only_current_policy_tick():
 obj,native,args=setup_hold();candidate=native.astype(float);candidate[0]+=.01
 selected,report=obj.select(candidate,**args)
 assert np.array_equal(selected.astype(np.float32).view(np.uint32),native.view(np.uint32))
 assert not report['active']and report['status']=='prepared_waiting_actual_policy_consumption'
 assert obj.confirm(report,actual_native=selected,actual_body_input=np.zeros(3),stamp_s=4.,consumption_seq=200,override_reasons=['terminal_native_hold'])
 assert report['active']and obj.manifest()['first_consumption']==dict(seq=200,stamp_s=4.)
 for step in range(201,350):
  stamp=step*.02;args.update(stamp_s=stamp,measurement_stamp_s=stamp,previous_issue_stamp_s=stamp-.02,previous_consumption_seq=step-1,collision_evidence=dict(status='pass',stamp_s=stamp))
  actual,r=obj.select(candidate,**args)
  assert h.same_native_bits(actual,native)
  assert obj.confirm(r,actual_native=actual,actual_body_input=np.zeros(3),stamp_s=stamp,consumption_seq=step,override_reasons=['terminal_native_hold'])
 assert not obj.in_window(7.)
 new=h.TerminalNativeHold(obj.task);assert new.command is None and new.first_consumption is None


@pytest.mark.parametrize('fault',['fcl_stale','fcl_unknown','moving','previous_moving','measurement_old','prior_old','support','error','nan_error','missed_tick'])
def test_capture_never_uses_unsafe_stale_or_unsupported_input(fault):
 obj,native,args=setup_hold()
 if fault=='fcl_stale':args['collision_evidence']['stamp_s']=3.9
 elif fault=='fcl_unknown':args['collision_evidence']['status']='unknown'
 elif fault=='moving':args['issued_body_input'][0]=.0001
 elif fault=='previous_moving':args['previous_body_input'][0]=.0001
 elif fault=='measurement_old':args['measurement_stamp_s']=3.98
 elif fault=='prior_old':args['previous_issue_stamp_s']=3.9
 elif fault=='support':args['original_support_ok']=False
 elif fault=='error':args['world_position_error_m']=.051
 elif fault=='nan_error':args['world_position_error_m']=float('nan')
 else:args.update(stamp_s=4.04,measurement_stamp_s=4.04)
 actual,r=obj.select(native,**args)
 assert not r['selected']and obj.cancelled and obj.command is None
 np.testing.assert_array_equal(actual,native)


@pytest.mark.parametrize('fault',['native_mismatch','body_veto','FCL_veto','late','wrong_seq','wrong_task'])
def test_later_actual_input_override_cannot_be_marked_a_consumed_hold(fault):
 obj,native,args=setup_hold();actual,r=obj.select(native,**args)
 kw=dict(actual_native=actual,actual_body_input=np.zeros(3),stamp_s=4.,consumption_seq=200,override_reasons=['terminal_native_hold'])
 if fault=='native_mismatch':kw['actual_native']=actual.copy();kw['actual_native'][0]+=.01
 elif fault=='body_veto':kw['actual_body_input'][0]=.01
 elif fault=='FCL_veto':kw['override_reasons'].append('collision_freshness_veto')
 elif fault=='late':kw['stamp_s']=4.02
 elif fault=='wrong_seq':kw['consumption_seq']=201
 else:r['task_sha256']='wrong'
 assert not obj.confirm(r,**kw)and not r['active']


def test_unconfirmed_hold_cannot_leak_to_next_tick_or_confirm_old_report():
 obj,native,args=setup_hold();actual,r=obj.select(native,**args)
 args.update(stamp_s=4.02,measurement_stamp_s=4.02,previous_issue_stamp_s=4.,previous_consumption_seq=200)
 _,r2=obj.select(native,**args)
 assert not r2['selected']and obj.command is None
 assert not obj.confirm(r,actual_native=actual,actual_body_input=np.zeros(3),stamp_s=4.,consumption_seq=200,override_reasons=['terminal_native_hold'])


def calibration_context():
 task,context=z.fixture();data=task.to_dict();data['issued_at_s']=7.;data['camera_trajectory']['start_s']=7.;task=TimedTask.from_dict(data)
 context.update(now_s=7.,observation_stamp_s=7.,previous_issue_stamp_s=6.98)
 obj,native,args=setup_hold();actual,report=obj.select(native,**args)
 obj.confirm(report,actual_native=actual,actual_body_input=np.zeros(3),stamp_s=4.,consumption_seq=200,override_reasons=['terminal_native_hold'])
 for row in context['history_rows']:
  row['stamp_s']+=2
  for value in row['task_feedback']['constraints'].values():value['stamp_s']+=2
  row['applied_policy_velocity']=[0.,0.,0.]
  row['terminal_native_hold']=dict(report)
  row['terminal_native_hold'].update(selected_issue_stamp_s=row['stamp_s']-.02,
   selected_consumption_seq=round((row['stamp_s']-.02)/.02))
  zi=row['actual_native_domain']['world_z_feedback'];zi.update(integration_allowed=False,externally_rolled_back=True,
   external_rollback_reason='terminal_native_hold',native_correction_accepted=False,integral_before_m=-.02,integral_after_m=-.02,
   sample_seq=round((row['stamp_s']-.02)/.02))
 context['previous_terminal_hold']=obj.manifest()
 return task,context


def test_only_bound_verified_intentional_hold_window_can_seed_from_actual_stable_bias():
 task,context=calibration_context();controller=z.m.BumplessWorldZFeedback(enabled=True)
 r=controller.admit(z.backend,task,**context)
 assert r['eligible']and r['prior_calibration_source']=='verified_fixed_terminal_actual_native_hold'
 assert abs(r['seed_clipped_m']+.02)<1e-7
 assert controller.integral_m==0


@pytest.mark.parametrize('fault',['old_parking','recovery','other_rollback','infeasible','bit_changed','body_input','task_sha','task_end','first_token','first_clock','missing_manifest','native_unsteady','ee_unsteady','integrated','FCL','domain','copied_old_tick_report'])
def test_any_non_dedicated_or_unsafe_hold_evidence_keeps_original_reset_zero(fault):
 task,context=calibration_context();row=context['history_rows'][3]
 if fault=='old_parking':row['native_camera_parking']['active']=True
 elif fault=='recovery':row['native_domain_recovery']['active']=True
 elif fault=='other_rollback':row['actual_native_domain']['world_z_feedback']['external_rollback_reason']+=' ,collision_freshness_veto'
 elif fault=='infeasible':row['task_feedback']['feasible']=False
 elif fault=='bit_changed':row['actual_native_command'][0]=float(np.nextafter(np.float32(.5),np.float32(1)))
 elif fault=='body_input':row['applied_policy_velocity'][0]=.0001
 elif fault=='task_sha':row['terminal_native_hold']['task_sha256']='bad'
 elif fault=='task_end':context['previous_terminal_hold']['task_end_s']+=.01
 elif fault=='first_token':row['terminal_native_hold']['first_consumption_seq']+=1
 elif fault=='first_clock':context['previous_terminal_hold']['first_consumption']['stamp_s']+=.02
 elif fault=='missing_manifest':context['previous_terminal_hold']=None
 elif fault=='native_unsteady':row['actual_native_command'][2]+=.002
 elif fault=='ee_unsteady':row['T_world_camera_optical'][2][3]+=.002
 elif fault=='integrated':row['actual_native_domain']['world_z_feedback']['integral_after_m']+=.001
 elif fault=='FCL':row['fine_collision_evidence']['status']='unknown'
 elif fault=='copied_old_tick_report':row['terminal_native_hold']['selected_issue_stamp_s']-=.02
 else:row['requested_native_domain']['status']='out_of_policy_domain'
 controller=z.m.BumplessWorldZFeedback(enabled=True);r=controller.admit(z.backend,task,**context)
 assert not r['eligible']and controller.integral_m==0


def test_runtime_select_before_domain_and_confirm_after_actual_policy_and_original_feedback_stays_live():
 source=(ROOT/'Robot_Go2/adaptive_runner.py').read_text()
 select=source.index('native,terminal_hold_report=terminal_native_hold.select(')
 recovery=source.index('native,recovery_command,recovery_report=domain_recovery.update(',select)
 finalize=source.index('corrected=finalize_native_feedback(',recovery)
 f32=source.index('ee=torch.tensor(native,dtype=torch.float32)',finalize)
 policy=source.index('action=policy(torch.cat(histories',f32)
 confirm=source.index('terminal_native_hold.confirm(',policy)
 physics=source.index('sim.step(render=False)',confirm)
 assert select<recovery<finalize<f32<policy<confirm<physics
 assert "row['terminal_native_hold']=terminal_hold_report"in source
 assert 'hold_terminal=True,constraints=constraints' in source
