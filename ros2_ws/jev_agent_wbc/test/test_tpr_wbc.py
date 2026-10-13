"""Behavioral contracts for trajectory-preserving passive avoidance."""
from types import SimpleNamespace
import numpy as np
import pytest

from jev_agent_wbc.planning import ExecutionWBC
from jev_agent_wbc.tpr_wbc import qp, AsyncReconfigurationPreview, ReconfigurationPreview, response_coefficients, response_velocity
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
from jev_agent_wbc.self_collision import SelfCollision
from jev_agent_wbc.actuation import advance_command
from jev_agent_wbc.safety import solve_with_certificate


STATE = np.r_[0.,0.,0.,0.,-.4,1.7,0.,-1.3,0.]


class Reference:
    shots = []
    duration = 20.
    sha256 = 'immutable-received-task'
    def sample(self, t):
        p,r,_ = whole_body_kinematics(STATE)
        return p,r,np.zeros(6)


def scene(center=(3.,3.,.3), velocity=(0.,0.,0.)):
    return dict(format='jvs_lidar_surfels/v1',surfel_centers=[center],surfel_radius=.07,
                surfel_velocities=[velocity],surfel_track_ids=[1],observation_stamp=0.,minimum_clearance_m=.025)


def controller(observed=None):
    c = ExecutionWBC(Limits(command_ramp=True),scene=scene() if observed is None else observed)
    c.set_trajectory(Reference())
    return c


def test_secondary_preference_cannot_move_fixed_camera_even_when_enormous():
    c = controller()
    c.reconfiguration.update = lambda *args: np.ones(8)*1e4
    p,r,J = whole_body_kinematics(STATE)
    u,info = c.solve(STATE,p,r,np.zeros(6),.02,commanded_q=STATE[3:])
    assert np.linalg.norm(u) > .001
    np.testing.assert_allclose(J@u,0.,atol=1e-9)
    assert info['tpr_report']['camera_task_feasible']
    assert max(abs(u)-c.limits.acceleration*.02) < 1e-6


def test_nearby_mover_reconfigures_base_and_arm_without_changing_reference():
    c = controller(scene((-.13,0.,.48),(.012,0.,0.)))
    x=STATE.copy(); received=c.trajectory
    for tick in range(15):
        t=tick*.02;c.set_time(t);old=c.previous.copy()
        u,info=c.solve(x,*received.sample(t),.02,trajectory_time=t,commanded_q=x[3:])
        assert info['tpr_status'] != 'degraded'
        x=advance_command(x,old,u,.02,True)
    assert abs(x[0]-STATE[0]) > 1e-5
    assert np.linalg.norm(x[3:]-STATE[3:]) > 1e-4
    p,r,_=whole_body_kinematics(x)
    np.testing.assert_allclose(p,received.sample(0.)[0],atol=5e-5)
    np.testing.assert_allclose(r,received.sample(0.)[1],atol=1e-7)
    assert c.trajectory is received


def test_unreachable_camera_acceleration_is_explicit_degradation():
    c=controller()
    p,r,_=whole_body_kinematics(STATE)
    c.reconfiguration.update=lambda *args: np.zeros(8)
    _,info=c.solve(STATE,p+[0.,0.,1.],r,np.zeros(6),.02)
    assert info['tpr_status']=='degraded'
    assert not info['tpr_report']['camera_task_feasible']
    assert np.linalg.norm(info['camera_acceleration_residual']) > 1.


def test_received_moving_position_and_orientation_are_tracked_together():
    from scipy.spatial.transform import Rotation
    class MovingReference(Reference):
        def sample(self,t):
            x=STATE.copy();x[3]+=.02*(1-np.cos(t));x[4]+=.01*(1-np.cos(t))
            p,r,J=whole_body_kinematics(x)
            velocity=np.zeros(8);velocity[2:4]=[.02*np.sin(t),.01*np.sin(t)]
            return p,r,J@velocity
    c=controller();reference=MovingReference();c.set_trajectory(reference)
    x=STATE.copy()
    for tick in range(60):
        t=tick*.02;c.set_time(t);old=c.previous.copy();c.measured_control=old.copy()
        u,info=c.solve(x,*reference.sample(t),.02,trajectory_time=t,commanded_q=x[3:])
        assert info['tpr_status']!='degraded'
        x=advance_command(x,old,u,.02,True)
    p,r,_=whole_body_kinematics(x);target,rotation,_=reference.sample(1.2)
    assert np.linalg.norm(p-target)<2e-4
    assert Rotation.from_matrix(r@rotation.T).magnitude()<2e-4
    assert np.linalg.norm(target-reference.sample(0.)[0])>.001
    assert Rotation.from_matrix(rotation@reference.sample(0.)[1].T).magnitude()>.001


def test_safety_failure_never_calls_legacy_escape_or_brake():
    c=controller()
    count=[]
    def failure(*args,**kwargs):
        count.append(kwargs)
        raise RuntimeError('empty safety set')
    c.solve=failure
    with pytest.raises(RuntimeError,match='empty safety set'):
        solve_with_certificate(c,STATE,*c.trajectory.sample(0.),.02)
    assert len(count)==1


def test_measured_velocity_mismatch_cannot_be_hidden_by_last_command():
    c=controller();c.measured_control=np.ones(8)
    with pytest.raises(RuntimeError):
        c.solve(STATE,*c.trajectory.sample(0.),.02)
    np.testing.assert_array_equal(c.previous,np.zeros(8))


def test_operating_command_bounds_and_measured_hardware_bounds_are_both_kept():
    c=controller();c.limits.acceleration[1]=.03
    c.measured_limits=Limits();c.measured_control=np.zeros(8);c.measured_control[1]=.002
    c.reconfiguration.update=lambda *args:np.zeros(8)
    u,info=c.solve(STATE,*c.trajectory.sample(0.),.02)
    assert abs(u[1]) <= .03*.02+1e-6
    assert np.all(abs(u-c.measured_control) <= c.measured_limits.acceleration*.02+1e-6)
    assert info['tpr_report']['actual_acceleration_limits']==c.measured_limits.acceleration.tolist()


def test_collision_prediction_snapshots_do_not_mutate_live_observation():
    from jev_agent_wbc.dynamic_collision import DynamicCollision
    original=DynamicCollision(scene((-.3,0.,.4),(.1,.2,0.)),0.)
    centers,radii=original.centers.copy(),original.radii.copy()
    predicted=original.at_time(1.)
    np.testing.assert_allclose(predicted.centers,centers+[[.1,.2,0.]])
    np.testing.assert_allclose(predicted.radii,radii+.035)
    predicted.set_time(2.)
    np.testing.assert_array_equal(original.centers,centers)
    np.testing.assert_array_equal(original.radii,radii)
    assert original.physical_time==0.
    original.set_time(.5)
    np.testing.assert_allclose(predicted.centers,centers+[[.2,.4,0.]])


def test_hierarchy_keeps_exact_equalities_at_active_limits():
    rng=np.random.default_rng(19)
    J=rng.normal(size=(6,8));feasible=np.r_[.01,-.01,np.zeros(6)]
    A=np.vstack([np.eye(8),J]);lo=np.r_[np.full(8,-.01),J@feasible];hi=np.r_[np.full(8,.01),J@feasible]
    u,residual=qp(np.eye(8),np.arange(8)*-1e4,A,lo,hi)
    np.testing.assert_allclose(J@u,J@feasible,atol=1e-10)
    assert residual < 1e-6


def test_identified_response_matches_endpoint_and_integrated_velocity():
    from scipy.integrate import quad
    old=np.arange(8)*.01;cmd0=-old;cmd1=old+.03
    tau=np.array([0.,.08,.002,.02,.03,.05,.1,.2]);h=.25
    decay,beta,gain=response_coefficients(tau,h)
    endpoint=decay*old+beta*cmd0+gain*cmd1
    np.testing.assert_allclose(endpoint,response_velocity(old,cmd0,cmd1,tau,h,h),atol=1e-12)
    average=tau/h*(old-endpoint)+.5*(cmd0+cmd1)
    for j in range(8):
        integral=quad(lambda t:response_velocity(old,cmd0,cmd1,tau,h,t)[j],0,h)[0]/h
        assert abs(integral-average[j]) < 1e-9


def test_response_aware_task_constrains_actual_velocity_and_command_limits():
    c=controller();c.limits.response_time[1]=.08;c.measured_control=np.zeros(8)
    c.reconfiguration.update=lambda *args:np.array([0.,.3,0.,0.,0.,0.,0.,0.])
    _,_,J=whole_body_kinematics(STATE)
    command,info=c.solve(STATE,*c.trajectory.sample(0.),.02)
    predicted=np.array(info['predicted_actual_velocity'])
    np.testing.assert_allclose(J@predicted,0.,atol=1e-9)
    assert np.linalg.norm(predicted-command) > 1e-4
    assert np.max(np.abs(command)-c.limits.acceleration*.02) < 1e-6
    decay,beta,gain=response_coefficients(c.limits.response_time,.02)
    np.testing.assert_allclose(predicted,gain*command,atol=1e-12)


def test_self_geometry_is_rigid_motion_invariant_and_covers_camera_pairs():
    c=SelfCollision();x=STATE.copy();x[:3]=[1.,2.,.9]
    np.testing.assert_allclose(c.distances(STATE),c.distances(x),atol=1e-12)
    assert any(c.geometry.shapes[j][1]=='camera_link' for i,j in c.pairs)
    assert c.distances(STATE).min() > c.margin


def test_self_collision_analytic_gradients_match_distance_queries():
    c=SelfCollision();x=STATE.copy();x[[3,6,8]]=.1
    _,gradient=c.terms(x)
    for j in range(3,9):
        a,b=x.copy(),x.copy();a[j]+=1e-6;b[j]-=1e-6
        np.testing.assert_allclose(gradient[:,j],(c.distances(a)-c.distances(b))/2e-6,atol=1e-7)


def test_demo_motion_uncertainty_covers_unseen_future_curve():
    # Current position and velocity only. Check the calibrated linear envelope
    # against independent environment truth throughout the prediction horizon.
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/demo_tpr_wbc.py'
    spec=importlib.util.spec_from_file_location('tpr_demo_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for period in (12.,24.,36.):
        rate=module.scene(0.,period)['motion_uncertainty_rate_m_s']
        for t in np.linspace(0.,period,31):
            p,v=module.flight(t,period)
            for future in np.linspace(.01,1.5,13):
                actual=module.flight(t+future,period)[0]
                assert np.linalg.norm(actual-p-v*future) <= rate*future


def test_async_preview_rejects_wrong_reference_stale_or_diverged_state():
    c=AsyncReconfigurationPreview();ref=Reference();c.reset(ref)
    result=ReconfigurationPreview();result.updated=0.
    result.times=np.array([0.,1.]);result.controls=np.zeros((2,8));result.states=np.tile(STATE,(2,1))
    c.pending_generation=c.generation
    assert c.accept(result,STATE,.2)
    assert not c.accept(result,STATE,.61)
    moved=STATE.copy();moved[0]+=.04
    assert not c.accept(result,moved,.2)
    c.info={'tpr_prediction_valid':True,'tpr_prediction_failure':'old reference'}
    c.reset(Reference())
    assert not c.accept(result,STATE,.2)
    assert c.info=={'tpr_prediction_valid':False}
