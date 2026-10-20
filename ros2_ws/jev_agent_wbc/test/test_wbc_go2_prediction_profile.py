"""Optional prediction candidate: original task/caps/reserve stay binding."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


coordinator=load('prediction_candidate',ROOT/'Robot_Go2/camera_coordinator.py')
backend=load('prediction_backend',ROOT/'Robot_Go2/wbc_backend.py')
runner=load('prediction_runner',ROOT/'Robot_Go2/adaptive_runner.py')


def model(profile='persistent_bias_12_v1'):
    return coordinator.CameraCoordinator(np.eye(4),backend=backend,response_profile='measured_backward_v1',
        prediction_profile=profile)


def test_extended_nodes_preserve_original_nodes_and_reserve():
    c=model()
    assert c.horizon_s==1.2
    np.testing.assert_allclose(c.prediction_intervals(1.2),[0,.2,.4,.6,.8,1.,1.2])
    np.testing.assert_array_equal(c.reserve,[.035,.035,.015])


@pytest.mark.parametrize('horizon',[0.,.01,.3,.8,1.,1.2])
def test_prediction_never_extrapolates_original_task(horizon):
    nodes=model().prediction_intervals(horizon)
    assert nodes[0]==0 and nodes[-1]==horizon
    assert np.all(nodes<=horizon)
    np.testing.assert_array_equal(nodes[:5] if horizon else nodes,np.linspace(0,min(horizon,.8),5) if horizon else [0])


def test_bias_persists_without_posture_and_clips_only_future_input():
    c=model();v=np.array([-.1,.1,.1]);preferred=np.array([0.,.03,-.1]);future=np.array([-.15,.2,.1])
    np.testing.assert_allclose(c.future_plan(v,preferred,future,0),[-.2,.2,.2])
    np.testing.assert_array_equal(model('legacy').future_plan(v,preferred,future,0),future)
    assert c.future_plan(v,preferred,None,0) is None
    np.testing.assert_array_equal(c.future_plan(v,preferred,None,.2),v)


def test_actual_future_gradient_exists_without_posture():
    c=model();body=np.eye(4);link=body.copy();link[2,3]=.07
    preferred=np.zeros(3);future=np.zeros(3);measured=np.zeros(3)
    def predicted(v):
        return c.predicted_link(link,body,v,.8,measured,c.future_plan(v,preferred,future,0),stamp_s=0)[0,3]
    assert predicted(np.array([-.01,0,0])) < predicted(np.zeros(3))-1e-4


def test_caps_slew_and_nonlinear_rejection_are_preserved():
    c=model();previous=np.array([.1,.05,.1]);reference=np.eye(4);reference[:3,3]=[.5,0,.5]
    sample=lambda t:reference.copy()
    result=c.command(0,sample,np.eye(4),np.eye(4),np.zeros(3),np.zeros(3),
        future_policy_velocity=np.zeros(3),previous_applied=previous)
    command=np.asarray(result['base_command'])
    assert np.all(abs(command)<=.2+1e-9)
    assert np.all(abs(command-previous)<=np.array([.5,.5,.8])*.02+1e-9)
    assert result['reserve_ramp_horizon_s']==.8
    assert result['prediction_intervals_s']==pytest.approx([0,.2,.4,.6,.8,1,1.2])
    outside=reference.copy();outside[0,3]=-1
    rejected=c.command(0,lambda t:outside,np.eye(4),np.eye(4),np.zeros(3),np.zeros(3),
        future_policy_velocity=np.zeros(3),previous_applied=previous)
    assert rejected['feasible'] is False
    assert rejected['reason']=='out_of_policy_domain'
    assert rejected['camera_reference_changed'] is False


def test_all_task_preference_includes_zero_preferred_camera_6d():
    assert runner.native_posture_enabled('x58_all_v1',[0,0,0])
    assert not runner.native_posture_enabled('x58_moving_v1',[0,0,0])
    assert runner.native_posture_enabled('x58_moving_v1',[.1,0,0])
    assert not runner.native_posture_enabled('none',[.1,0,0])


@pytest.mark.parametrize('horizon',[float('nan'),float('inf'),-.1,1.21])
def test_invalid_prediction_horizons_rejected(horizon):
    with pytest.raises(ValueError):model().prediction_intervals(horizon)


def test_legacy_outputs_are_unchanged():
    frozen=load('legacy_predictor',ROOT/'Robot_Go2/results/adaptive_camera6d_softpose_01/sources/camera_coordinator.py')
    controllers=[m.CameraCoordinator(np.eye(4),backend=backend,response_profile='measured_backward_v1') for m in (frozen,coordinator)]
    reference=np.eye(4);reference[:3,3]=[.5,0,.5]
    for stamp in [0.,.02,.04]:
        results=[c.command(stamp,lambda t:reference.copy(),np.eye(4),np.eye(4),np.zeros(3),np.zeros(3),
            future_policy_velocity=np.zeros(3),previous_applied=np.zeros(3)) for c in controllers]
        np.testing.assert_array_equal(results[0]['base_command'],results[1]['base_command'])
        for key in ('reason','feasible','predicted_min_domain_margin','camera_reference_changed'):
            assert results[0][key]==results[1][key]
        for c in controllers:c.observe_applied_input(stamp,results[0]['base_command'])
