"""Fresh-only vy feedback, real nonzero forecast and final safety brake routing."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('test_lateral_brake',ROOT/'Robot_Go2/lateral_stand_brake.py')
b=load('test_lateral_backend',ROOT/'Robot_Go2/wbc_backend.py')
r=load('test_lateral_driver',ROOT/'Robot_Go2/adaptive_runner.py')

def fixture(vy=.02,previous=None):
 previous=np.zeros(3)if previous is None else np.asarray(previous,float)
 return dict(stamp_s=10.,measurement_stamp_s=10.,task_end_s=8.,stop_seconds=5.,body=np.eye(4),link=np.eye(4),
  measured_velocity=np.array([.01,vy,-.01]),previous_applied=previous.copy(),
  input_history=[(10.+i*.02,previous.copy())for i in range(-32,0)],prediction_intervals=[0,.2,.4,.6,.8,1.,1.2])

@pytest.mark.parametrize('vy',[-.2,-.061,-.02,-.001,0,.001,.02,.061,.2])
def test_formula_cap_slew_and_original_axis_meaning(vy):
 kw=fixture(vy);c=m.LateralStandBrake(b,**kw)
 assert c.raw[0]==c.raw[2]==c.first[0]==c.first[2]==0
 assert c.raw[1]==-.5*vy and abs(c.target[1])<=.03
 np.testing.assert_array_equal(c.first,np.clip(c.target,-m.SLEW*.02,m.SLEW*.02))
 assert c.manifest()['requested_stop_velocity']==[0,0,0]
 np.testing.assert_array_equal(c.finalize(c.first),c.first)
 assert c.manifest()['final_feedback_only']


def test_forecast_preserves_actual_first_pending_history_and_nonzero_feedback_plan():
 c=m.LateralStandBrake(b,**fixture(.02))
 np.testing.assert_array_equal(c.predict_link(c.first,0),np.eye(4))
 predicted=c.predict_link(c.first,1.2)
 assert abs(predicted[1,3])>.001
 trajectory=c.model.trajectory(c.first,c.target)
 np.testing.assert_array_equal(trajectory['issued'][0],c.first)
 assert np.max(np.abs(np.diff(trajectory['issued'],axis=0)))<=.5*.02+1e-12
 assert np.all(trajectory['issued'][:,0]==0)and np.all(trajectory['issued'][:,2]==0)
 assert len(c.model.queries)==7
 with pytest.raises(ValueError,match='actual locked first'):c.predict_link(np.zeros(3),.2)


@pytest.mark.parametrize('fault',['stale_measurement','late_window','before_window','gap_history','missing_history','wrong_previous','nan_velocity','nan_time','bad_previous_axis','assist_cap_previous'])
def test_unknown_or_inconsistent_feedback_context_cannot_issue_braking_motion(fault):
 kw=fixture()
 if fault=='stale_measurement':kw['measurement_stamp_s']-=.02
 elif fault=='late_window':kw['stamp_s']=13.;kw['measurement_stamp_s']=13.
 elif fault=='before_window':kw['stamp_s']=9.98;kw['measurement_stamp_s']=9.98
 elif fault=='gap_history':kw['input_history'].pop(3)
 elif fault=='missing_history':kw['input_history']=[]
 elif fault=='wrong_previous':kw['previous_applied'][1]=.001
 elif fault=='nan_velocity':kw['measured_velocity'][1]=float('nan')
 elif fault=='nan_time':kw['stamp_s']=float('nan')
 elif fault=='bad_previous_axis':kw['previous_applied'][0]=.001
 else:kw['previous_applied'][1]=.04
 with pytest.raises(ValueError):m.LateralStandBrake(b,**kw)


@pytest.mark.parametrize('veto',[['original_forecast_rejected'],['collision_freshness_veto'],['native_domain_recovery'],['prior_terminal_deadline_failure'],[]])
def test_actual_driver_final_route_brakes_on_every_veto_and_records_public_failure(veto):
 c=m.LateralStandBrake(b,**fixture(.04,previous=[0,-.02,0]));support=dict(feasible=None,camera_supported=True)
 overrides=[];issued=r.finalize_active_stand_brake(c,c.first,support,overrides,veto)
 if veto:
  np.testing.assert_array_equal(issued,m.safe_brake(c.previous));assert support['feasible']is False
  assert 'active_brake_safety_veto'in overrides and not c.manifest()['final_feedback_only']
 else:
  np.testing.assert_array_equal(issued,c.first);assert support['feasible']is None and not overrides
 assert np.all(abs(issued-c.previous)<=m.SLEW*.02+1e-12)


def test_unexpected_later_recovery_cannot_restart_x_or_yaw_or_silently_change_feedback():
 c=m.LateralStandBrake(b,**fixture(.02));support=dict(feasible=None);overrides=[]
 issued=r.finalize_active_stand_brake(c,np.array([.01,.01,.01]),support,overrides,[])
 np.testing.assert_array_equal(issued,np.zeros(3));assert support['feasible']is False
 assert c.manifest()['veto_reasons']==['later_body_input_override']
