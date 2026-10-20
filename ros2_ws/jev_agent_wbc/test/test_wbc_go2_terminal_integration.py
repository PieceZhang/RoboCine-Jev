"""Terminal planner and original domain guard share the actual first input."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


backend=module('terminal_live_backend',ROOT/'Robot_Go2/wbc_backend.py')
allocator=module('terminal_live_allocator',ROOT/'Robot_Go2/terminal_allocator.py')
coordinator=module('terminal_live_coordinator',ROOT/'Robot_Go2/camera_coordinator.py')


def call(elapsed=0.,previous=None,restrict_workspace=False):
    stamp=5+elapsed;previous=np.zeros(3)if previous is None else np.asarray(previous)
    body=np.eye(4);body[2,3]=.3;link=body.copy();link[2,3]+=.07
    endpoint=np.eye(4);endpoint[:3,3]=[.5,.25,.5]
    c=coordinator.CameraCoordinator(np.eye(4),backend=backend,prediction_profile='persistent_bias_12_v1')
    if restrict_workspace:c.reserve[:2]=.3
    horizon=min(1.2,5-elapsed)
    context=allocator.TerminalContext(backend,stamp_s=stamp,task_end_s=5,stop_seconds=5,
        endpoint_optical=endpoint,extrinsic=np.eye(4),body=body,link=link,measured_velocity=np.zeros(3),
        previous_applied=previous,input_history=[(stamp-.02*k,previous.copy())for k in range(32,0,-1)],
        prediction_intervals=c.prediction_intervals(horizon))
    proposal=context.solve();seen=[]
    def predict(first,t):
        seen.append(np.asarray(first).copy())
        np.testing.assert_allclose(first,context.guard_applied(first),atol=1e-9,rtol=0)
        return context.predict_link(first,t)
    result=c.command(stamp,lambda t:endpoint.copy(),link,body,proposal,np.zeros(3),
        max_lookahead_s=horizon,previous_applied=previous,command_dt=.02,
        prediction_provider=predict,locked_input=context.phase!='allocate')
    return context,result,seen


def test_original_camera_domain_guard_uses_explicit_slew_brake_prediction():
    context,result,seen=call()
    assert result['planned_slew_brake_model']
    assert result['camera_supported'] and result['feasible']is not False
    assert result['predicted_min_domain_margin']>=-1e-7
    np.testing.assert_allclose(result['base_command'],context.guard_applied(result['base_command']))
    assert seen


def test_derivatives_at_slew_boundary_never_query_unissuable_first_input():
    context,result,seen=call(restrict_workspace=True)
    assert result['feasible']is False
    assert len(seen)>7
    assert np.all(abs(np.asarray(result['base_command']))<=allocator.SLEW*.02+1e-9)


@pytest.mark.parametrize('elapsed,previous',[(1.6,[-.2,.2,.2]),(2.,[.01,-.01,.016]),(4.5,[0,0,0])])
def test_brake_or_zero_is_locked_to_exact_original_slew_input(elapsed,previous):
    context,result,seen=call(elapsed,previous)
    assert result['locked_input']
    np.testing.assert_array_equal(result['base_command'],context.proposal)
    if elapsed>=2:np.testing.assert_array_equal(result['base_command'],[0,0,0])


def test_wire_retains_original_reference_and_standing_label_is_actual_applied_input():
    source=(ROOT/'Robot_Go2/adaptive_runner.py').read_text();tree=ast.parse(source)
    assert "sample=lambda t:terminal_endpoint.copy()"in source
    assert "desired_camera,_=session.reference(min(stamp,task_end))"in source
    assert "native_standing_mode=(bool(not moving and np.all(policy_command==0))"in source
    # First-input plans and safety-override guards are tied to the same context.
    solve=source.index('policy_command=terminal_context.solve()')
    projection=source.index('prediction_provider=terminal_context.predict_link')
    final_guard=source.index('issued,forced_brake=final_terminal_body_command(policy_command')
    assert solve<projection<final_guard


def test_terminal_model_rejects_a_second_competing_soft_posture_cost():
    c=coordinator.CameraCoordinator(np.eye(4),backend=backend)
    with pytest.raises(ValueError,match='owns'):
        c.command(0,lambda t:np.eye(4),np.eye(4),np.eye(4),np.zeros(3),np.zeros(3),
            prediction_provider=lambda v,t:np.eye(4),posture_weight=.2,native_posture_x_m=.58)


@pytest.mark.parametrize('history_kind',['missing','stale'])
@pytest.mark.parametrize('deadline_failed',[False,True])
def test_actual_driver_rejection_branch_does_not_retry_invalid_history(history_kind,deadline_failed):
    driver=module('terminal_rejected_live_driver',ROOT/'Robot_Go2/adaptive_runner.py')
    c=coordinator.CameraCoordinator(np.eye(4),backend=backend,prediction_profile='bounded_ramp_history_v2')
    if history_kind=='stale':c.input_history=[(5-.02*k-.1,np.zeros(3))for k in range(32,0,-1)]
    body=np.eye(4);camera=np.eye(4);camera[:3,3]=[.5,.1,.5]
    previous=np.array([.05,-.05,.04]);seen=[]
    # Reproduces the context constructor's missing/stale-history failure.
    with pytest.raises(ValueError,match='history'):
        allocator.TerminalContext(backend,stamp_s=5,task_end_s=5,stop_seconds=5,
            endpoint_optical=camera,extrinsic=np.eye(4),body=body,link=body,
            measured_velocity=np.zeros(3),previous_applied=previous,input_history=c.input_history)
    def invalid_retry(*args,**kwargs):raise AssertionError('invalid history must never reach a second forecast')
    c.command=invalid_retry
    def sample(stamp):seen.append(stamp);return camera.copy()
    result=driver.camera_support_with_terminal_rejection(c,5,sample,body,body,np.zeros(3),np.zeros(3),
        previous,.02,0.,{},'fresh context history unavailable',deadline_failed)
    np.testing.assert_allclose(result['base_command'],[.04,-.04,.024],atol=1e-12)
    np.testing.assert_allclose(result['native_command'][:3],[.5,.1,.5],atol=1e-12)
    assert seen==[5]and result['feasible']is False
    assert result['reason']==('tracking_limit'if deadline_failed else'command_stale')
    assert result['forecast_status']=='unavailable_not_evaluated'
    assert result['predicted_min_domain_margin']is None
    assert not result['camera_supported']and not result['base_velocity_feasible']


def test_actual_driver_dense_normal_branch_equals_original_direct_command():
    driver=module('terminal_normal_live_driver',ROOT/'Robot_Go2/adaptive_runner.py')
    c=coordinator.CameraCoordinator(np.eye(4),backend=backend,prediction_profile='bounded_ramp_history_v2')
    c.input_history=[(5-.02*k,np.zeros(3))for k in range(32,0,-1)]
    body=np.eye(4);camera=np.eye(4);camera[:3,3]=[.5,.1,.5]
    forecast=dict(previous_applied=np.zeros(3),command_dt=.02,future_policy_velocity=np.zeros(3))
    expected=c.command(5,lambda t:camera.copy(),body,body,np.zeros(3),np.zeros(3),max_lookahead_s=1.2,**forecast)
    actual=driver.camera_support_with_terminal_rejection(c,5,lambda t:camera.copy(),body,body,np.zeros(3),np.zeros(3),
        np.zeros(3),.02,1.2,forecast,None,False)
    assert actual.keys()==expected.keys()
    for key in actual:
        if isinstance(actual[key],np.ndarray):np.testing.assert_array_equal(actual[key],expected[key])
        else:assert actual[key]==expected[key]


@pytest.mark.parametrize('error,latched', [('missing fresh history',False),(None,True),('deadline infeasible',True)])
def test_actual_final_route_brakes_after_later_native_domain_recovery(error,latched):
    driver=module('terminal_late_veto_driver',ROOT/'Robot_Go2/adaptive_runner.py')
    class UnsafeRecoveryContext:
        def guard_applied(self,command):raise AssertionError('failed context must not authorize body recovery')
    # A later recovery proposes nonzero motion; native7 can remain legal while
    # final body input must independently keep original-slew braking.
    recovery=np.array([-.02,.02,-.04]);previous=np.array([.05,-.05,.04])
    issued,forced=driver.final_terminal_body_command(recovery,previous,.02,UnsafeRecoveryContext(),error,latched)
    assert forced is True
    np.testing.assert_allclose(issued,[.04,-.04,.024],atol=1e-12)
    assert np.all(abs(issued-previous)<=allocator.SLEW*.02+1e-12)


def test_actual_final_route_normal_fresh_context_retains_original_projection():
    driver=module('terminal_fresh_final_driver',ROOT/'Robot_Go2/adaptive_runner.py')
    context,result,_=call();command=np.asarray(result['base_command'])
    issued,forced=driver.final_terminal_body_command(command,context.previous,.02,context,None,False)
    assert forced is False
    np.testing.assert_array_equal(issued,context.guard_applied(command))
