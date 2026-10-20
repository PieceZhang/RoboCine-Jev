import importlib.util,json,sys,ast
from pathlib import Path
import numpy as np
import pytest
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('default_upper_diagnostic',Path(__file__).with_name('default_upper_yaw.py'));d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from jev_agent_wbc.representation import Pose,Box,Observation
from jev_agent_wbc.wbc_protocol import TimedTask,CameraTrajectory,Capabilities


def observation(stamp):return Observation(stamp,Pose((0.,0.,.78)),Pose((.2,0.,.3)),{'mobile_platform':(Box((0.,0.,0.),(.2,.2,.3)),),'camera_manipulator':(Box((.2,0.,.3),(.1,.1,.1)),)})
def task(obs):return TimedTask('diagnostic_task',obs.stamp_s,obs.id,CameraTrajectory(obs.stamp_s,(0.,5.),(obs.world_from_camera,obs.world_from_camera)),.05,.15)


def test_default_goal_with_original_reference_lead_and_limits():
    for measured in [np.array([0.,.03]),np.array([.2,-.2])]:
        result,report=d.bounded_home_reference(measured,[0,0],[[-1,1],[-1,1]])
        assert np.max(abs(result-measured))<=.05+1e-15
        assert not report['dynamic_pink_frame_task_control']
        np.testing.assert_allclose(result,np.clip([0,0],measured-.05,measured+.05))


@pytest.mark.parametrize('measured,home,limits',[
 ([float('nan')],[0.],[[-1.,1.]]),([0.],[2.],[[-1.,1.]]),([2.],[0.],[[-1.,1.]]),([0.],[0.],[[1.,-1.]])])
def test_invalid_or_outside_original_state_never_produces_target(measured,home,limits):
    with pytest.raises(ValueError):d.bounded_home_reference(measured,home,limits)


def test_no_positive_camera_capability_or_fake_acceptance_even_at_zero_error():
    with pytest.raises(ValueError):d.DefaultUpperDiagnosticSession(Capabilities('bad',True,frozenset(['timed_camera_se3'])))
    s=d.DefaultUpperDiagnosticSession(Capabilities('diagnostic',False,frozenset()))
    obs=observation(1.);t=task(obs);answer=s.accept(t,obs,1.)
    assert answer['supported'] is False and answer['feasible'] is None and answer['diagnostic_reference_only']
    T,_=s.reference(1.);assert T.shape==(4,4)
    feedback=s.feedback(observation(1.02),feasible=True,reason='accepted',hold_terminal=True,now_s=1.02)
    assert feedback['supported'] is False and feedback['feasible'] is None and feedback['diagnostic_only']
    assert feedback['reason']=='unsupported_request' and feedback['camera_within_tolerance']
    with pytest.raises(ValueError):s.reference(6.01)
    with pytest.raises(ValueError):s.feedback(observation(1.02),now_s=1.02)


def test_stale_or_wrong_binding_reference_rejected():
    s=d.DefaultUpperDiagnosticSession(Capabilities('diagnostic',False,frozenset()));obs=observation(1.)
    with pytest.raises(ValueError):s.accept(task(obs),obs,1.2)
    with pytest.raises(ValueError):s.accept(task(obs),observation(1.02),1.02)


def test_generated_source_disables_control_preserves_physics_and_forces_diagnostic_result():
    b=P/'Robot_G1/results'/d.BASELINE/'sources';runner=(b/'adaptive_runner.py').read_text();backend=(b/'g1_backend.py').read_text();new,worker=d.candidate_sources(runner,backend)
    assert 'desired_camera if session else None,stamp,camera,dt' not in new
    assert 'None,stamp,camera,dt,measured_bodies=' in new
    assert 'diagnostic_only=True,all_pass=False,wbc_acceptance_pass=False' in new
    assert 'public_timed_task_actual_ingress=False' in new
    assert 'if len(geometry.shapes)!=35' in new
    old_ast,new_ast=ast.parse(runner),ast.parse(new)
    selected=lambda tree:[ast.dump(x) for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id in ['robot','sim'] and x.func.attr in ['step','update','write_data_to_sim','write_root_state_to_sim','write_joint_state_to_sim','set_joint_position_target']]
    assert selected(old_ast)==selected(new_ast)
    assert 'None if request[\'target\'] is None else solver.last_collision_report' in worker
    with pytest.raises(ValueError):d.candidate_sources('not the frozen source',backend)
