"""Two-axis actual-velocity brake retains the original final safety route."""
import ast,sys
from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('planar_brake',ROOT/'Robot_Go2/lateral_stand_brake.py')
b=load('planar_backend',ROOT/'Robot_Go2/wbc_backend.py')
r=load('planar_runner',ROOT/'Robot_Go2/adaptive_runner.py')

def fixture(velocity=(.02,-.04,.01),previous=(0,0,0),profile='vxy_p05_v1'):
    previous=np.asarray(previous,float)
    return dict(stamp_s=10.,measurement_stamp_s=10.,task_end_s=8.,stop_seconds=5.,
        body=np.eye(4),link=np.eye(4),measured_velocity=np.asarray(velocity,float),previous_applied=previous,
        input_history=[(10.+i*.02,previous.copy())for i in range(-32,0)],
        prediction_intervals=[0,.2,.4,.6,.8,1.,1.2],profile=profile)

@pytest.mark.parametrize('velocity',[(.2,-.2,0),(-.02,.04,.1),(.001,-.001,-.1),(0,0,0)])
@pytest.mark.parametrize('previous',[(0,0,0),(.03,-.03,0),(-.03,.03,0)])
def test_fresh_xy_negative_feedback_cap_slew_and_locked_forecast(velocity,previous):
    c=m.LateralStandBrake(b,**fixture(velocity,previous))
    np.testing.assert_array_equal(c.raw,[-.5*velocity[0],-.5*velocity[1],0])
    np.testing.assert_array_equal(c.target,np.clip(c.raw,[-.03,-.03,0],[.03,.03,0]))
    np.testing.assert_array_equal(c.first,np.clip(c.target,np.asarray(previous)-m.SLEW*.02,np.asarray(previous)+m.SLEW*.02))
    assert c.first[2]==0 and np.all(abs(c.first[:2])<=.03)
    trajectory=c.model.trajectory(c.first,c.target)
    np.testing.assert_array_equal(trajectory['issued'][0],c.first)
    assert np.all(abs(np.diff(trajectory['issued'],axis=0))<=m.SLEW*.02+1e-12)
    np.testing.assert_array_equal(c.predict_link(c.first,0),np.eye(4))
    assert c.manifest()['requested_stop_velocity']==[0,0,0]
    assert c.manifest()['profile']=='vxy_p05_v1' and len(c.model.queries)==7

@pytest.mark.parametrize('veto',[['original_forecast_rejected'],['collision_freshness_veto'],['native_domain_recovery'],['prior_terminal_deadline_failure'],[]])
def test_final_shared_route_safely_brakes_both_axes_and_marks_failure(veto):
    c=m.LateralStandBrake(b,**fixture(previous=(.02,-.02,0)));support=dict(feasible=None);overrides=[]
    issued=r.finalize_active_stand_brake(c,c.first,support,overrides,veto)
    np.testing.assert_array_equal(issued,m.safe_brake(c.previous)if veto else c.first)
    if veto:assert support['feasible']is False and overrides==['active_brake_safety_veto']
    assert np.all(abs(issued-c.previous)<=m.SLEW*.02+1e-12)

@pytest.mark.parametrize('fault',['stale','missing_history','wrong_previous','oversize_x','nonzero_yaw','unknown_profile'])
def test_inconsistent_actual_evidence_rejected(fault):
    kw=fixture()
    if fault=='stale':kw['measurement_stamp_s']-=.02
    elif fault=='missing_history':kw['input_history']=[]
    elif fault=='wrong_previous':kw['previous_applied'][0]=.001
    elif fault=='oversize_x':kw['previous_applied'][0]=.031
    elif fault=='nonzero_yaw':kw['previous_applied'][2]=.001
    else:kw['profile']='unverified'
    with pytest.raises(ValueError):m.LateralStandBrake(b,**kw)

def test_legacy_vy_default_and_explicit_profile_identical():
    kw=fixture(velocity=(.02,-.04,.01));kw.pop('profile')
    a=m.LateralStandBrake(b,**kw);z=m.LateralStandBrake(b,**kw,profile='vy_p05_v1')
    assert a.manifest()==z.manifest()
    np.testing.assert_array_equal(a.first,z.first)
    for t in a.model.queries:np.testing.assert_array_equal(a.predict_link(a.first,t),z.predict_link(z.first,t))

def test_real_runner_passes_explicit_profile_to_same_helper():
    tree=ast.parse((ROOT/'Robot_Go2/adaptive_runner.py').read_text())
    call=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='LateralStandBrake')
    assert next(ast.unparse(k.value)for k in call.keywords if k.arg=='profile')=='a.go2_active_stand_brake'

@pytest.mark.parametrize('valid',[True,False])
def test_actual_cli_preserves_required_fine_collision_and_dispatches_explicit_profile(monkeypatch,tmp_path,valid):
    argv=['adaptive_runner.py','--robot','go2','--output',str(tmp_path/'out'),'--stop-seconds','5',
        '--go2-active-stand-brake','vxy_p05_v1','--go2-terminal-profile','endpoint_xy58_15_yaw0_v1',
        '--go2-prediction-profile','bounded_ramp_history_v2','--go2-world-z-feedback','bounded_integral_v1']
    if valid:argv.append('--fine-collision')
    monkeypatch.setattr(sys,'argv',argv)
    class Dispatch(Exception):pass
    def dispatch(args):
        assert valid and args[args.index('--go2-active-stand-brake')+1]=='vxy_p05_v1'
        raise Dispatch
    monkeypatch.setattr(r.subprocess,'run',dispatch)
    with pytest.raises(Dispatch if valid else ValueError):r.main()
