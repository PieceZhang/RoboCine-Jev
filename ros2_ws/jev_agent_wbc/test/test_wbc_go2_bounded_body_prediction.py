"""Actual first input, pending history and bounded future ramp invariants."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


m=load('bounded_native_forecast',ROOT/'Robot_Go2/bounded_body_prediction.py')
backend=load('bounded_native_backend',ROOT/'Robot_Go2/wbc_backend.py')


def context(previous=None,measured=None,history=None,mode='linear_residual'):
    previous=np.zeros(3)if previous is None else np.asarray(previous)
    body=np.eye(4);body[2,3]=.3;link=body.copy();link[2,3]+=.07
    return m.BoundedBodyPrediction(backend,stamp_s=5.,body=body,link=link,
        measured_velocity=np.zeros(3)if measured is None else measured,previous_applied=previous,
        input_history=history or[(5-k*.02,previous.copy())for k in range(32,0,-1)],pending_mode=mode)


@pytest.mark.parametrize('mode',['linear_residual','measured_hold'])
def test_first_value_original_limits_slew_and_measured_initial_transform(mode):
    c=context(mode=mode);first=np.array([.01,-.01,.016]);data=c.trajectory(first,[.2,-.2,.2])
    np.testing.assert_array_equal(data['issued'][0],first)
    assert np.all(abs(data['issued'])<=m.CAPS+1e-12)
    assert np.all(abs(np.diff(np.vstack([c.previous,data['issued']]),axis=0))<=m.SLEW*.02+1e-12)
    np.testing.assert_array_equal(c.predict_link(first,[.2,-.2,.2],0),c.link)


@pytest.mark.parametrize('first,future', [([.02,0,0],[0,0,0]),([0,0,0],[.21,0,0])])
def test_unissuable_first_or_overcap_future_rejected(first,future):
    with pytest.raises(ValueError):context().trajectory(first,future)


def test_forecast_does_not_replace_history_or_measured_initial_state():
    c=context(measured=[.03,.04,.05]);d=context(measured=[0,0,0])
    assert np.linalg.norm(c.predict_link([0,0,0],[.1,.1,.1],.2)-d.predict_link([0,0,0],[.1,.1,.1],.2))>1e-4
    h=[(5-k*.02,np.array([.1,.1,.1])if k>=3 else np.zeros(3))for k in range(32,0,-1)]
    e=context(history=h)
    assert np.linalg.norm(e.predict_link([0,0,0],[.1,.1,.1],.2)-d.predict_link([0,0,0],[.1,.1,.1],.2))>1e-4


@pytest.mark.parametrize('kind',['stale','future','gap','wrongprevious','short'])
def test_false_history_fails_closed(kind):
    h=[(5-k*.02,np.zeros(3))for k in range(32,0,-1)]
    if kind=='stale':h=[(t-.02,u)for t,u in h]
    elif kind=='future':h[-1]=(5.,np.zeros(3))
    elif kind=='gap':h.pop(10)
    elif kind=='wrongprevious':h[-1]=(h[-1][0],np.array([.01,0,0]))
    else:h=h[-4:]
    with pytest.raises(ValueError):context(history=h)


def test_zero_crossing_model_is_continuous_all_axes():
    c=context();a=c.predict_link([-1e-9]*3,[-1e-9]*3,1.2);b=c.predict_link([1e-9]*3,[1e-9]*3,1.2)
    assert np.linalg.norm(a-b)<1e-7


def test_future_plan_moves_more_than_frozen_first_pulse():
    c=context();first=[.01,-.01,0]
    ramp=c.predict_link(first,[.15,-.15,0],1.2);frozen=c.predict_link(first,first,1.2)
    assert ramp[0,3]-frozen[0,3]>.05
    assert ramp[1,3]-frozen[1,3]<-.08


def test_first_positive_slew_pulse_is_not_falsely_held_for_whole_horizon():
    coordinator=load('bounded_model_coordinator',ROOT/'Robot_Go2/camera_coordinator.py')
    body=np.eye(4);body[2,3]=.3;link=body.copy();link[2,3]+=.07
    def sample(stamp):
        value=np.eye(4);value[:3,3]=[.56+.1*(stamp-5),.21-.1*(stamp-5),.5];return value
    def run(profile,posture=0):
        c=coordinator.CameraCoordinator(np.eye(4),backend=backend,response_profile='measured_backward_v1',prediction_profile=profile)
        c.input_history=[(5-k*.02,np.zeros(3))for k in range(32,0,-1)]
        return c.command(5,sample,link,body,[.01,-.01,0],[0,0,0],max_lookahead_s=1.2,
            future_policy_velocity=[.1,-.1,0],previous_applied=np.zeros(3),command_dt=.02,
            native_posture_x_m=.58,posture_weight=posture)
    old=run('persistent_bias_12_v1');new=run('bounded_ramp_history_v2')
    assert old['feasible']is False and old['predicted_min_domain_margin']<0
    assert new['feasible']is not False and new['predicted_min_domain_margin']>.03
    np.testing.assert_array_equal(new['base_command'],[.01,-.01,0])
    assert new['prediction_intervals_s']==old['prediction_intervals_s']
    # The optional soft preference differentiates at the original slew edge.
    soft=run('bounded_ramp_history_v2',.2)
    assert np.all(abs(np.asarray(soft['base_command']))<=m.SLEW*.02+1e-9)


def test_new_profile_needs_actual_history_and_original_control_interval():
    coordinator=load('bounded_model_clock_coordinator',ROOT/'Robot_Go2/camera_coordinator.py')
    c=coordinator.CameraCoordinator(np.eye(4),backend=backend,prediction_profile='bounded_ramp_history_v2')
    body=np.eye(4);camera=np.eye(4);camera[:3,3]=[.5,0,.5]
    with pytest.raises(ValueError,match='history'):
        c.command(5,lambda t:camera,body,body,[0,0,0],[0,0,0],previous_applied=np.zeros(3))
