"""Fresh actual output continuity does not certify a steady tracking bias."""
import copy,importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('continuity_fixture',ROOT/'test/test_wbc_go2_bumpless_z_seed.py');z=importlib.util.module_from_spec(spec);spec.loader.exec_module(z)
PROFILE='fresh_actual_output_continuity_v2'
def new():return z.m.BumplessWorldZFeedback(enabled=True,initialization_profile=PROFILE)

@pytest.mark.parametrize('variation',['native','measured','bias'])
def test_observed_small_motion_no_longer_means_reset_output_or_claim_steady_bias(variation):
 task,c=z.fixture()
 if variation=='native':c['history_rows'][0]['actual_native_command'][2]+=.002
 elif variation=='measured':c['history_rows'][0]['T_world_camera_optical'][2][3]+=.002
 else:
  c['history_rows'][0]['actual_native_command'][2]+=.0008;c['history_rows'][0]['T_world_camera_optical'][2][3]-=.0008
 old=z.m.BumplessWorldZFeedback(enabled=True);legacy=old.admit(z.backend,task,**copy.deepcopy(c))
 assert not legacy['eligible']and old.integral_m==0
 control=new();record=control.admit(z.backend,task,**c);native,check=z.feedback(control,task,c)
 assert record['eligible']and not record['steady_bias_calibration_claim']and not record['steady_window_within_legacy_1mm']
 assert record['initialization_profile']==PROFILE and record['reason']=='fresh_actual_output_continuity'
 assert check['world_z_feedback']['seed_pending_commit']
 assert control.finish_seed_update(check['world_z_feedback']['update_id'],applied_native=np.asarray(native,dtype=np.float32))
 assert native[2]==.5

@pytest.mark.parametrize('fault',['first_task','expired','future','old_observation','old_issue','missing_history','FCL','configuration','domain','recovery','other_veto','infeasible','wrong_actual_native','wrong_reference'])
def test_continuity_preserves_original_freshness_consumption_and_real_guard_rejections(fault):
 task,c=z.fixture()
 if fault=='first_task':c['previous_applied_native']=None
 elif fault=='expired':c['previous_issue_stamp_s']=4.9
 elif fault=='future':c['previous_issue_stamp_s']=5.01
 elif fault=='old_observation':c['observation_stamp_s']=4.98
 elif fault=='old_issue':c['now_s']=5.02
 elif fault=='missing_history':c['history_rows'].pop(3)
 elif fault=='FCL':c['history_rows'][-1]['fine_collision_evidence']['status']='unknown'
 elif fault=='configuration':c['history_rows'][-1]['task_feedback']['constraints']['configuration']['status']='fail'
 elif fault=='domain':c['history_rows'][-1]['actual_native_domain']['status']='out_of_policy_domain'
 elif fault=='recovery':c['history_rows'][-1]['native_domain_recovery']['active']=True
 elif fault=='other_veto':c['history_rows'][-1]['actual_native_domain']['world_z_feedback']['externally_rolled_back']=True
 elif fault=='infeasible':c['history_rows'][-1]['task_feedback']['feasible']=False
 elif fault=='wrong_actual_native':c['previous_applied_native'][2]+=.001
 else:c['reference_optical'][2,3]+=.001
 controller=new();r=controller.admit(z.backend,task,**c)
 assert not r['eligible']and controller.integral_m==0

@pytest.mark.parametrize('native_z',[.49,.55])
def test_bound25mm_keeps_explicit_residual_instead_of_claiming_exact_continuity(native_z):
 task,c=z.fixture(native_z);controller=new();r=controller.admit(z.backend,task,**c);native,check=z.feedback(controller,task,c)
 assert r['eligible']and abs(r['seed_clipped_m'])==.025
 assert abs(r['expected_remaining_native_z_jump_m'])>.0049
 assert check['world_z_feedback']['proposed_position_correction_norm_m']<=.025+1e-12


def test_output_seed_still_waits_real_first_policy_consumption_and_rolls_back_faults():
 task,c=z.fixture();controller=new();r=controller.admit(z.backend,task,**c);native,check=z.feedback(controller,task,c)
 token=check['world_z_feedback']['update_id'];assert r['transaction_status']=='prepared_waiting_actual_issue'
 controller.rollback_last_update('collision_freshness_veto',update_id=token)
 assert not controller.finish_seed_update(token,applied_native=native)and controller.integral_m==0


def test_output_seed_never_propagates_unconfirmed_first_tick_or_accepts_late_confirmation():
 task,c=z.fixture();controller=new();controller.admit(z.backend,task,**c);native,check=z.feedback(controller,task,c)
 token=check['world_z_feedback']['update_id'];native2,_=z.feedback(controller,task,c,stamp=5.02,seq=251)
 assert controller.integral_m==0 and abs(native2[2]-.52)<1e-12
 assert not controller.finish_seed_update(token,applied_native=native)
