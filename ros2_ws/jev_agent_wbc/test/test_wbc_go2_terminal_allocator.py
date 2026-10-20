"""Fresh terminal model invariants and actual issued-input deadline, no Isaac."""
import importlib.util
from pathlib import Path
import itertools
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


allocator=module('terminal_allocator_test',ROOT/'Robot_Go2/terminal_allocator.py')
backend=module('terminal_allocator_backend',ROOT/'Robot_Go2/wbc_backend.py')


def context(elapsed=0.,previous=None,**overrides):
    body=np.eye(4);body[2,3]=.3;link=body.copy();link[2,3]+=.07
    endpoint=np.eye(4);endpoint[:3,3]=[.5,.25,.5]
    options=dict(stamp_s=5+elapsed,task_end_s=5,stop_seconds=5,endpoint_optical=endpoint,extrinsic=np.eye(4),
        body=body,link=link,measured_velocity=np.zeros(3),previous_applied=np.zeros(3)if previous is None else previous,
        input_history=[(5-.02*k,[0,0,0])for k in range(32,0,-1)])
    options.update(overrides)
    return allocator.TerminalContext(backend,**options)


def test_fresh_model_allocates_xy_without_changing_original_endpoint():
    c=context();endpoint=c.endpoint.copy();issued=c.solve()
    assert issued[0]<0 and issued[1]>0
    assert np.all(abs(issued)<=np.array([.01,.01,.016])+1e-12)
    np.testing.assert_array_equal(c.endpoint,endpoint)
    assert not c.endpoint.flags.writeable
    assert np.linalg.norm(c.model_target_error)<.003
    assert c.manifest()['reference_stamp_s']==5
    assert c.manifest()['reference_mode']=='explicit_original_endpoint_hold'


@pytest.mark.parametrize('previous',list(itertools.product([-.2,.2],repeat=3)))
def test_arbitrary_body_reversal_and_final_veto_obey_caps_slew_and_zero_by2(previous):
    previous=np.asarray(previous);history=[(5-.02*k,previous.copy())for k in range(32,0,-1)]
    for step in range(250):
        elapsed=step*.02
        c=context(elapsed,previous=previous,input_history=history)
        c.solve()
        # Even an opposite requested domain recovery must respect this same
        # original deadline; public infeasible feedback is retained by caller.
        issued=c.guard_applied(-np.sign(previous+.000001)*.2)
        assert np.all(abs(issued)<=.2+1e-12)
        assert np.all(abs(issued-previous)<=np.array([.5,.5,.8])*.02+1e-9)
        if elapsed>=2.-1e-9:assert np.all(issued==0)
        history.append((5+elapsed,issued.copy()));history=history[-32:]
        previous=issued


def test_planned_braking_is_part_of_prediction():
    previous=np.array([-.2,.2,.2])
    c=context(1.6,previous=previous,input_history=[(6.6-.02*k,previous.copy())for k in range(32,0,-1)]);c.solve()
    assert c.phase=='brake'
    data=c._trajectory(c.proposal,c.desired)
    commands=data['issued']
    assert np.all(commands[np.where(c.command_times>=.4-1e-9)]==0)
    assert np.max(abs(np.diff(np.vstack([c.previous,commands]),axis=0)),axis=0).tolist()==pytest.approx([.01,.01,.016])


def test_prediction_is_exact_at_original_measurement_and_accepts_only_explicit_queries():
    c=context();c.solve()
    np.testing.assert_allclose(c.predict_link(c.proposal,0),c.link,atol=1e-14)
    with pytest.raises(ValueError):c.predict_link(c.proposal,.12345)
    with pytest.raises(ValueError):c.predict_link([np.nan,0,0],.2)
    with pytest.raises(ValueError):c.predict_link([.21,0,0],.2)
    with pytest.raises(ValueError):c.predict_link([.2,0,0],.2)


@pytest.mark.parametrize('overrides',[dict(stamp_s=4.9),dict(stop_seconds=4.99),dict(measured_velocity=[np.nan,0,0]),
    dict(previous_applied=[.21,0,0]),dict(input_history=[(4.98,[0,0,0]),(4.4,[0,0,0])])])
def test_invalid_time_state_or_history_rejected(overrides):
    with pytest.raises(ValueError):context(**overrides)


@pytest.mark.parametrize('elapsed,previous',[(1.98,[.2,0,0]),(2.,[.011,0,0]),(2.1,[0,0,.017])])
def test_late_context_does_not_fake_zero_or_violate_slew(elapsed,previous):
    previous=np.asarray(previous)
    with pytest.raises(allocator.TerminalDeadlineInfeasible)as caught:context(elapsed,previous=previous)
    safe=caught.value.safe_brake_command
    assert np.all(abs(safe-previous)<=allocator.SLEW*.02+1e-12)
    assert np.any(safe!=0)


def test_zero_crossing_prediction_is_continuous_with_same_measured_initial_state():
    previous=np.array([.1,0,0]);history=[(5-.02*k,previous.copy())for k in range(32,0,-1)]
    c=context(previous=previous,input_history=history,measured_velocity=[.08,0,0]);c.solve()
    # Isolate the two constant target sides of0. Both predictions retain the
    # identical initial velocity/old input history/filter, without a model swap.
    left=c._trajectory(c._first(np.array([-1e-9,0,0])),np.array([-1e-9,0,0]))['xyz'][-1]
    right=c._trajectory(c._first(np.array([1e-9,0,0])),np.array([1e-9,0,0]))['xyz'][-1]
    assert np.linalg.norm(left-right)<1e-7


@pytest.mark.parametrize('history',[[],[(4.9,[0,0,0])]*9,
    [(5-.02*k,[0,0,0])for k in range(33,1,-1)],
    [(5-.02*k,[0,0,0])for k in range(31,-1,-1)],
    [(5-.02*k,[.01,0,0])for k in range(32,0,-1)]])
def test_missing_stale_future_or_wrong_last_applied_history_cannot_be_used(history):
    with pytest.raises(ValueError):context(input_history=history)


@pytest.mark.parametrize('previous',[.0115,.010001])
def test_non_grid_absolute_deadline_rejects_late_unbrakeable_input(previous):
    # Absolute Task.end5.003 => deadline7.003. At current update7.000 only
    # this update can make zero; the next7.020 is already too late.
    with pytest.raises(allocator.TerminalDeadlineInfeasible):
        context(elapsed=2.,task_end_s=5.003,previous=np.array([previous,0,0]))


def test_non_grid_deadline_brakes_at_last_update_and_does_not_retime_task():
    previous=np.array([.009,0,0]);stamp=7.;end=5.003
    c=context(elapsed=2.,task_end_s=end,previous=previous,
        input_history=[(stamp-.02*k,previous.copy())for k in range(32,0,-1)])
    command=c.solve()
    np.testing.assert_array_equal(command,[0,0,0])
    assert c.manifest()['reference_stamp_s']==end
    assert c.manifest()['zero_input_deadline_s']==pytest.approx(7.003)
    assert c.zero_updates_remaining==0


def test_non_grid_allocation_changes_to_brake_before_1p6_not_after_it():
    times=np.arange(6)*.02
    commands=allocator.schedule(np.array([.2,0,0]),np.array([.2,0,0]),np.array([.2,0,0]),.023,times)
    # Current relative time1.577 is still allocate. Next update1.597 must
    # already brake, since its ZOH extends beyond the1.600 phase boundary.
    assert commands[0,0]==.2
    assert commands[1,0]==pytest.approx(.19)
