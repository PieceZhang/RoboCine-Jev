"""PhysX input adaptation preserves measured geometry and shared CPU defaults."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from physx_compat import OrientedSearch, subject_from_returns, SolverAPI, MotorTargets, generalized_gravity, freeze_task_inputs
from jev_agent_wbc.kinematic_feasibility import KinematicSearch
from jev_agent_wbc.whole_body_qp import Limits
from jev_agent_wbc.dynamic_collision import DynamicCollision


@pytest.mark.parametrize('angles',[(0.,0.,0.),(.03,-.02,.7)])
def test_oriented_proxy_preserves_camera_and_body_geometry(angles):
    state=np.array([.1,-.2,angles[2],0.,-.9,2.1,0.,-1.2,0.])
    br=Rotation.from_euler('xyz',angles).as_matrix()
    scene=dict(format='jvs_lidar_surfels/v1',surfel_centers=[[.5,.4,.6]],
        surfel_radius=.1, minimum_clearance_m=.025,
        surfel_radii=[.1],surfel_velocities=[[.1,0.,0.]],surfel_track_ids=[1],observation_stamp=0.)
    search=KinematicSearch(state,np.zeros(8),0.,.002,Limits(),base_rotation=br,scene=scene)
    proxy=OrientedSearch(search)
    assert proxy.tilt is None and search.tilt is not None
    x=state.copy();x[2]+=.2
    for a,b in zip(proxy.fk(x),search.fk(x)):np.testing.assert_array_equal(a,b)
    rotated=Rotation.from_euler('z',x[2]).as_matrix()@search.tilt
    for offset in (0.,.4):
        actual=proxy.collision.at_time(offset).clearance(x,.002,gradient=True)
        expected=search.collision.at_time(offset).clearance(x,.002,rotated,gradient=True)
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-12)


def test_partial_visible_subject_surface_does_not_bias_center():
    model=dict(id='subject',position=[1.5,.2,.8],radius=.13,height=.36)
    theta,z=np.meshgrid(np.linspace(np.pi-.7,np.pi+.7,12),np.linspace(-.14,.14,7))
    points=np.column_stack([1.5+.13*np.cos(theta.ravel()),.2+.13*np.sin(theta.ravel()),.8+z.ravel()])
    assert np.linalg.norm(.5*(points.min(axis=0)+points.max(axis=0))-model['position'])>.1
    estimate=subject_from_returns(points,model,2.)
    assert estimate['valid']
    np.testing.assert_allclose(estimate['position'],model['position'],atol=1e-9)
    assert estimate['stamp_s']==2.


def test_insufficient_returns_remain_unknown():
    model=dict(id='subject',position=[1.5,.2,.8],radius=.13,height=.36)
    estimate=subject_from_returns([[1.37,.2,.8]],model,0.)
    assert not estimate['valid'] and not estimate['orientation_valid']


def test_solver_api_preserves_feasible_and_infeasible_results():
    import osqp
    from scipy import sparse
    api=SolverAPI(osqp)
    qp=api.OSQP()
    qp.setup(P=sparse.eye(1,format='csc'),q=np.array([-2.]),A=sparse.eye(1,format='csc'),
             l=np.array([0.]),u=np.array([1.]),verbose=False)
    result=qp.solve()
    assert result.info.status_val==1 and abs(result.x[0]-1.)<.002
    assert result.info.prim_res<.002
    qp=api.OSQP()
    qp.setup(P=sparse.eye(1,format='csc'),q=np.zeros(1),A=sparse.csc_matrix([[1.],[1.]]),
             l=np.array([1.,-np.inf]),u=np.array([np.inf,0.]),verbose=False)
    assert qp.solve().info.status_val in (3,4)


def test_physical_motor_interface_keeps_cpu_caps_and_handles_wheel_coupling():
    limits=Limits();motors=MotorTargets(limits)
    requested=np.array([.3,1.,.8,-.7,.6,.5,-.4,.3])
    previous=np.zeros(8)
    for k in range(1000):
        previous_control=motors.control.copy()
        current=motors.step(requested,.001)
        assert np.all(np.abs(current-previous)<=motors.acceleration*.001+1e-12)
        assert np.all(np.abs(motors.control-previous_control)<=limits.acceleration*.001+1e-12)
        previous=current
    np.testing.assert_allclose(current,np.r_[limits.wheels@requested,requested[2:]],atol=1e-10)
    assert limits.command_ramp is False



@pytest.mark.parametrize('root_dofs',[0,6])
def test_gravity_adapter_handles_fixed_and_floating_base_layouts(root_dofs):
    raw=np.arange(14+root_dofs,dtype=float)
    ix=np.array([1,4,6,9,10,11,12,13])
    np.testing.assert_array_equal(generalized_gravity(raw,14,ix),raw[root_dofs+ix])


def test_unknown_gravity_layout_is_rejected():
    with pytest.raises(RuntimeError):generalized_gravity(np.zeros(21),14,[1])


def test_motor_slew_uses_actual_velocity_when_actuator_lags():
    limits=Limits();motors=MotorTargets(limits)
    motors.control=np.full(8,.2)
    actual=np.zeros(8)
    target=motors.step(np.ones(8),.001,actual)
    assert np.all(abs(motors.control-actual)<=limits.acceleration*.001+1e-12)
    assert abs(target[0])<=limits.wheel_acceleration*.001+1e-12


def test_worker_transport_freezes_current_observation_before_async_serialization():
    cloud=np.array([[.1,.2,.3]])
    scene=dict(observation_stamp=1.2,surfel_centers=cloud,tracks=[dict(id=1,velocity=[.1,0,0])])
    args,kwargs=freeze_task_inputs((scene,),dict(history=[dict(t=1.2)]))
    scene['new_key']=True;cloud[:]=9.;scene['tracks'][0]['velocity'][0]=8.
    assert 'new_key' not in args[0]
    np.testing.assert_array_equal(args[0]['surfel_centers'],[[.1,.2,.3]])
    assert args[0]['tracks'][0]['velocity'][0]==.1
    assert kwargs['history'][0]['t']==1.2


def test_wheel_encoder_anchor_does_not_amplify_chassis_slip():
    limits=Limits();motors=MotorTargets(limits)
    wheels=np.array([1.,2.]);measured=np.zeros(8);measured[1]=.8
    target=motors.step(np.ones(8),.001,measured,measured_wheels=wheels)
    assert np.all(abs(target[:2]-wheels)<=limits.wheel_acceleration*.001+1e-12)


def test_motor_brakes_before_unchanged_cpu_position_margin():
    from jev_agent_wbc.jvs_kinematics import JOINT_UPPER
    limits=Limits();motors=MotorTargets(limits)
    positions=JOINT_UPPER-limits.position_margin-2e-5
    target=motors.step(np.ones(8),.001,np.zeros(8),positions)
    np.testing.assert_allclose(target[2:],0.,atol=1e-11)
    assert limits.position_margin==.01
