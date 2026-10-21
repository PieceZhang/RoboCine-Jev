"""Native Giskard physical bridge constraints, not a substitute for GPU evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROBOT=Path(__file__).resolve().parents[1]/'Robot_TIAGo'
sys.path[:0]=[str(ROBOT),str(ROBOT/'wbc')]
from collision_model import CollisionBarriers
from physical_control import CameraController


def test_spatial_com_shift_matches_independent_urdf_pose_differences():
    """Full world-frame linear/angular validation at several bodies/postures."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from source_state import source_state
    from physical_control import spatial_jacobian_at_offset
    spec=json.loads((ROBOT/'robot.json').read_text())
    document=ET.parse(ROBOT/'urdf/tiago_physics.urdf').getroot()
    epsilon=1e-6
    for index,q in enumerate([np.array([j['home'] for j in spec['joints']]),
            np.array([.1,.65,-.35,-.8,1.35,.25,.2,-.3]),np.array([.28,.2,-.9,-.2,1.85,-.3,.75,.4])]):
        R=Rotation.from_euler('xyz',[.17*index,-.13*index,.63*index]).as_matrix()
        poses,origin_js=source_state(spec,q)
        for name in ['base_link','torso_lift_link','arm_2_link','gripper_link']:
            inertial=document.find(f"link[@name='{name}']/inertial/origin")
            local_com=np.fromstring(inertial.get('xyz','0 0 0'),sep=' ') if inertial is not None else np.zeros(3)
            rotated_com=R@poses[name][:3,:3]@local_com
            origin=origin_js[name][:,2:].copy();origin[:3]=R@origin[:3];origin[3:]=R@origin[3:]
            # COM Jacobian is computed independently from finite-difference
            # transforms; it is never constructed by reversing the shift.
            com_j=np.zeros((6,8));actor_j=np.zeros((6,8))
            for column in range(8):
                qm=q.copy();qp=q.copy();qm[column]-=epsilon;qp[column]+=epsilon
                minus=source_state(spec,qm)[0][name];plus=source_state(spec,qp)[0][name]
                com_j[:3,column]=R@((plus[:3,:3]@local_com+plus[:3,3])-(minus[:3,:3]@local_com+minus[:3,3]))/(2*epsilon)
                actor_j[:3,column]=R@(plus[:3,3]-minus[:3,3])/(2*epsilon)
                omega=R@Rotation.from_matrix(plus[:3,:3]@minus[:3,:3].T).as_rotvec()/(2*epsilon)
                com_j[3:,column]=actor_j[3:,column]=omega
            shifted=spatial_jacobian_at_offset(com_j,-rotated_com)
            np.testing.assert_allclose(shifted,actor_j,atol=4e-9)
            np.testing.assert_allclose(shifted,origin,atol=4e-9)
            batched=spatial_jacobian_at_offset(np.stack([com_j,com_j]),np.stack([-rotated_com,-rotated_com]))
            np.testing.assert_allclose(batched,np.stack([actor_j,actor_j]),atol=4e-9)


def test_parking_latches_measured_position_only_after_deceleration_and_releases():
    from physical_control import WheelParkingServo,wheel_speeds
    parking=WheelParkingServo()
    q=np.array([123.,-24.])
    assert not parking.update(0.,True,np.zeros(2),q,[.1,0.])['active']
    for i in range(2):assert not parking.update(.01*(i+1),True,np.zeros(2),q,[.01,-.01])['active']
    entered=parking.update(.03,True,np.zeros(2),q,[.01,-.01])
    assert entered['active'] and entered['kp']==300.
    np.testing.assert_array_equal(entered['target_rad'],q)
    np.testing.assert_array_equal(parking.update(.04,True,np.zeros(2),q+1.,[.2,.2])['target_rad'],q)
    # A permitted 100Hz native velocity slew releases without adding any new
    # wheel velocity or position target. Continuous encoder angle is unwrapped.
    spec=json.loads((ROBOT/'robot.json').read_text())
    body_command=np.array([.0045,.0065])
    wheel_command=wheel_speeds(*body_command,spec['wheel_radius_m'],spec['wheel_separation_m'])
    released=parking.update(.05,True,wheel_command,q,[0.,0.])
    assert not released['active'] and released['kp']==0. and released['target_rad'] is None
    np.testing.assert_allclose(wheel_command,wheel_speeds(*body_command,spec['wheel_radius_m'],spec['wheel_separation_m']))
    assert [e['mode'] for e in parking.events]==['position_hold','velocity']


def box_fixture(distance=.04):
    metadata=dict(shapes=[dict(name='fixed',body='fixed',kind='box',size=[1.,1.,1.],T_body_shape=np.eye(4).tolist()),
        dict(name='moving',body='moving',kind='box',size=[1.,1.,1.],T_body_shape=np.eye(4).tolist())],allowed_pairs=[])
    poses={'fixed':np.eye(4),'moving':np.eye(4)}
    poses['moving'][0,3]=1.+distance
    js={'fixed':np.zeros((6,10)),'moving':np.zeros((6,10))};js['moving'][0,2]=1.
    return metadata,poses,js


def test_fcl_nearest_points_produce_correct_separation_gradient():
    pytest.importorskip('fcl')
    metadata,poses,js=box_fixture()
    A,b,report=CollisionBarriers(metadata).query(poses,js)
    np.testing.assert_allclose(A[0],np.eye(10)[2],atol=1e-10)
    np.testing.assert_allclose(b,[-3*(.04-.005)],atol=1e-10)
    assert report['active_pair_count']==1


def test_real_giskard_solver_respects_hard_self_separation():
    pytest.importorskip('piqp');pytest.importorskip('fcl')
    from native_solver import NativeGiskardSolver
    metadata,poses,js=box_fixture(distance=.025)
    spec=json.loads((ROBOT/'robot.json').read_text())
    solver=NativeGiskardSolver(spec,metadata)
    previous=np.zeros(10);previous[2]=-.06
    request=dict(q=np.array([j['home'] for j in spec['joints']]),previous=previous,
        poses=poses,body_jacobians=js,mode='actuator',desired_rates=np.r_[0.,0.,-.2,np.zeros(7)])
    result=solver.solve(request)
    assert result['solver'].startswith('giskardpy.') and not result['big_ball_mode']
    assert -.06-1e-7<=result['command'][2]<=-.05+1e-7
    assert result['hard_constraint_violation']<1e-6


def test_genuine_stationary_projection_does_not_create_crawl_inside_safe_barrier():
    """The exact optimum is zero when all hard limits include zero strictly."""
    pytest.importorskip('piqp');pytest.importorskip('fcl')
    from native_solver import NativeGiskardSolver
    metadata,poses,js=box_fixture(distance=.074)
    spec=json.loads((ROBOT/'robot.json').read_text())
    solver=NativeGiskardSolver(spec,metadata)
    answer=solver.solve(dict(q=np.array([j['home'] for j in spec['joints']]),
        previous=np.zeros(10),poses=poses,body_jacobians=js,mode='actuator',desired_rates=np.zeros(10)))
    assert answer['collision']['active_pair_count']==1
    assert np.max(np.abs(answer['command']))<1e-9
    assert answer['hard_constraint_violation']<1e-6


def test_native_projection_reallocates_camera_twist_under_original_slew_limits():
    pytest.importorskip('piqp');pytest.importorskip('fcl')
    from native_solver import NativeGiskardSolver
    metadata,poses,js=box_fixture(distance=.4)
    spec=json.loads((ROBOT/'robot.json').read_text())
    solver=NativeGiskardSolver(spec,metadata)
    previous=np.zeros(10);previous[3]=.09
    desired=np.zeros(10);desired[0]=.02;desired[3]=.07
    J=np.zeros((6,10));J[0,0]=J[0,3]=1.
    request=dict(q=np.array([j['home'] for j in spec['joints']]),previous=previous,
        poses=poses,body_jacobians=js,mode='actuator',desired_rates=desired,jacobian=J)
    ordinary=solver.solve(request)
    reallocated=solver.solve(dict(request,preserve_proposed_camera_twist=True))
    assert np.linalg.norm(J@(reallocated['command']-desired))<.1*np.linalg.norm(J@(ordinary['command']-desired))
    assert np.all(np.abs(reallocated['command']-previous)<=solver.accel*.01+1e-9)
    assert reallocated['hard_constraint_violation']<1e-6
    assert not reallocated['big_ball_mode']


def test_official_allowed_pair_is_explicitly_filtered():
    pytest.importorskip('fcl')
    metadata,poses,js=box_fixture(distance=-.01)
    metadata['allowed_pairs']=[['fixed','moving']]
    A,b,_=CollisionBarriers(metadata).query(poses,js)
    assert A.shape==(0,10) and b.shape==(0,)


def test_unallowed_penetration_is_rejected():
    pytest.importorskip('fcl')
    metadata,poses,js=box_fixture(distance=-.01)
    with pytest.raises(RuntimeError,match='Nonallowed physical collision'):
        CollisionBarriers(metadata).query(poses,js)


def test_uncontrollable_nonallowed_penetration_is_still_rejected():
    pytest.importorskip('fcl')
    metadata,poses,js=box_fixture(distance=-.01)
    js['moving'][:]=0.
    with pytest.raises(RuntimeError,match='Nonallowed physical collision'):
        CollisionBarriers(metadata).query(poses,js)


def test_independent_guard_preserves_braking_and_acceleration():
    spec=json.loads((ROBOT/'robot.json').read_text())
    guard=CameraController(spec['joints'],spec['wheel_radius_m'],spec['wheel_separation_m'])
    q=guard.home.copy();q[0]=guard.upper[0]-.002
    u=guard.guard(q,np.ones(10)*100.)
    assert np.all(np.abs(u)<=guard.accel*.01+1e-10)
    assert u[2]<=1e-10


def test_measured_rates_are_independently_checked_even_with_issued_native_state():
    spec=json.loads((ROBOT/'robot.json').read_text())
    guard=CameraController(spec['joints'],spec['wheel_radius_m'],spec['wheel_separation_m'])
    measured=np.zeros(10);measured[2]=.05
    assert guard.measured_motion_check(guard.home,measured)['raw_measured_velocity_limits_checked']
    q=guard.home.copy();q[0]=guard.upper[0]-.0021
    with pytest.raises(RuntimeError,match='stopping reserve'):
        guard.measured_motion_check(q,measured)
    measured[2]=.08
    with pytest.raises(RuntimeError,match='velocity exceeds'):
        guard.measured_motion_check(guard.home,measured)


def test_full_dof_guard_includes_original_passive_and_wheel_feedback():
    import xml.etree.ElementTree as ET
    from physical_control import PhysicalDofGuard
    spec=json.loads((ROBOT/'robot.json').read_text())
    nodes=ET.parse(ROBOT/'urdf/tiago_physics.urdf').getroot().findall('joint')
    names=[node.get('name') for node in nodes if node.get('type')!='fixed']
    guard=PhysicalDofGuard(names,nodes,spec['left_wheels']+spec['right_wheels'])
    q=np.zeros(14);qd=np.zeros(14)
    assert guard.check(q,qd)['raw_physical_dofs_checked']==14
    qd[names.index('head_2_joint')]=3.1
    with pytest.raises(RuntimeError,match='full-articulation'):
        guard.check(q,qd)
    qd[:]=0.;qd[names.index('wheel_left_joint')]=8.1
    with pytest.raises(RuntimeError,match='full-articulation'):
        guard.check(q,qd)


def test_native_entry_does_not_call_historical_optimizer():
    source=(ROBOT/'giskard_runtime.py').read_text()
    assert "wbc/physical_control.py" in source
    assert 'wbc.velocity(' not in source
    assert 'native.command(packet)' in source
    assert "session.reference(absolute_t)" in source


def test_complete_world_executor_rejects_rebound_and_preserves_original_limits(tmp_path,monkeypatch):
    """Genuine upstream tasks solve, then unchanged policy bounds project output."""
    pytest.importorskip('probabilistic_model');pytest.importorskip('fcl')
    monkeypatch.setenv('JAX_PLATFORMS','cpu')
    monkeypatch.setenv('XDG_CACHE_HOME',str(ROBOT/'wbc/cache'))
    from full_framework import FullGiskardExecutor
    from source_state import source_state
    spec=json.loads((ROBOT/'robot.json').read_text())
    metadata=json.loads((ROBOT/'wbc/results/collision_geometry_01.json').read_text())
    solver=FullGiskardExecutor(spec,metadata,tmp_path/'native_meshes')
    q=np.array([j['home'] for j in spec['joints']]);poses,js=source_state(spec,q)
    measured=np.zeros(10);measured[4]=.114;measured[2]=spec['joints'][0]['velocity']
    request=dict(q=q,previous=np.zeros(10),measured_rates=measured,poses=poses,body_jacobians=js,
        mode='actuator',desired_rates=np.zeros(10),sample_s=0.,phase='cpu_test')
    result=solver.solve(request)
    assert result['full_world_executor'] and not result['big_ball_mode']
    assert abs(result['upstream_proposal'][4])<1e-6
    assert abs(result['upstream_proposal'][2])<1e-6
    assert result['hard_constraint_violation']<1e-6
    assert len(solver.world.active_degrees_of_freedom)==10
    for dof,joint in zip(solver.dofs[2:],spec['joints']):
        assert dof.limits.lower.position==joint['lower']
        assert dof.limits.upper.position==joint['upper']
        assert dof.limits.upper.velocity==joint['velocity']
    request['desired_rates']=np.r_[.2,0.,np.zeros(8)];request['sample_s']=.01
    result=solver.solve(request)
    assert result['upstream_proposal'][0]>1e-6
    assert 0.<result['command'][0]<=.0045+1e-7
