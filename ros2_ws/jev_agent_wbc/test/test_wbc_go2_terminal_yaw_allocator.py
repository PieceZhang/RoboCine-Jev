"""CPU-only yaw allocation: original endpoint, real history and original limits."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


helper=load('yaw_terminal_candidate',ROOT/'Robot_Go2/terminal_yaw_allocator.py')
backend=load('yaw_terminal_backend',ROOT/'Robot_Go2/wbc_backend.py')


def context(yaw=.2,elapsed=0.,previous=None):
    previous=np.zeros(3)if previous is None else np.asarray(previous)
    body=np.eye(4);body[2,3]=.3;link=body.copy();link[2,3]+=.07
    endpoint=np.eye(4);endpoint[:3,3]=[.5,.25,.5];endpoint[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
    stamp=5+elapsed
    return helper.TerminalContext(backend,stamp_s=stamp,task_end_s=5,stop_seconds=5,
        endpoint_optical=endpoint,extrinsic=np.eye(4),body=body,link=link,
        measured_velocity=np.zeros(3),previous_applied=previous,
        input_history=[(stamp-k*.02,previous.copy())for k in range(32,0,-1)])


@pytest.mark.parametrize('yaw',[-.3,-.1,.1,.3])
def test_yaw_preference_reduces_model_error_without_modifying_world_target(yaw):
    c=context(yaw);original=c.endpoint.copy();initial=c._preference_error(c._first(np.zeros(3)),np.zeros(3))
    c.solve();after=c._preference_error(c.proposal,c.desired)
    assert abs(after[2])<abs(initial[2])
    assert np.sign(c.desired[2])==np.sign(yaw)
    assert np.all(abs(c.proposal)<=helper.SLEW*.02+1e-9)
    np.testing.assert_array_equal(c.endpoint,original)
    assert c.manifest()['reference_stamp_s']==5
    assert c.manifest()['original_world_task_modified']is False


@pytest.mark.parametrize('elapsed,previous',[(1.6,[.1,-.1,.1]),(2.,[.01,-.01,.016]),(4.5,[0,0,0])])
def test_original_brake_and_literalzero_deadline_remain_authoritative(elapsed,previous):
    c=context(.2,elapsed,previous);first=c.solve()
    np.testing.assert_allclose(first,np.clip(0,np.asarray(previous)-helper.SLEW*.02,np.asarray(previous)+helper.SLEW*.02),atol=1e-9)
    if elapsed>=2:np.testing.assert_array_equal(first,np.zeros(3))
    for t in [0,.2,.4,.6,.8,1.,1.2]:assert np.isfinite(c.predict_link(first,t)).all()


def test_invalid_history_or_deadline_still_fails_closed():
    with pytest.raises(helper.TerminalDeadlineInfeasible):context(elapsed=1.98,previous=[.2,0,0])


def test_zero_crossing_of_three_axis_local_solver_is_continuous():
    left=context(-1e-9);right=context(1e-9);left.solve();right.solve()
    assert np.linalg.norm(left.desired-right.desired)<1e-5


def test_terminal_and_moving_models_share_actual_pending_input_dynamics():
    prediction=load('yaw_pending_shared_model',ROOT/'Robot_Go2/bounded_body_prediction.py')
    c=context(.2);first=c.solve()
    history=[(c.stamp+t,u)for t,u in zip(c.history_times,c.history_inputs)]
    moving=prediction.BoundedBodyPrediction(backend,stamp_s=c.stamp,body=c.body,link=c.link,
        measured_velocity=c.measured,previous_applied=c.previous,input_history=history,prediction_intervals=[0,.2,.4,.6,.8,1,1.2])
    for t in [0,.2,.4,.6,.8,1,1.2]:
        np.testing.assert_allclose(c.predict_link(first,t),moving.predict_link(first,c.desired,t),atol=1e-12,rtol=0)
