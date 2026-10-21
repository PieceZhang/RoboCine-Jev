"""TIAGo genuine-Giskard task model: independent mathematical and source checks."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROBOT = Path(__file__).resolve().parents[1]/'Robot_TIAGo'
sys.path.insert(0,str(ROBOT))
sys.path.insert(0,str(ROBOT/'wbc'))
from wbc_backend import (ARM_JOINTS,CAMERA_FRAME,TimedCameraReference,
                         camera_mount,differential_wheel_rates,model_urdf)
from kinematics import TiagoKinematics,transform


def test_original_source_limits_are_retained_in_augmented_model():
    original = ET.parse(ROBOT/'urdf/tiago_physics.urdf').getroot()
    augmented = ET.fromstring(model_urdf())
    assert len(augmented.findall('joint'))==len(original.findall('joint'))+1
    assert len(augmented.findall('link'))==len(original.findall('link'))+1
    for name in ARM_JOINTS:
        assert original.find(f"joint[@name='{name}']/limit").attrib == augmented.find(f"joint[@name='{name}']/limit").attrib


def test_optical_frame_is_exact_canonical_sensor_transform():
    parent,mount = camera_mount()
    document = ET.fromstring(model_urdf())
    joint = document.find("joint[@name='tiago_camera_optical_fixed']")
    assert joint.find('parent').get('link')==parent=='gripper_link'
    assert joint.find('child').get('link')==CAMERA_FRAME
    xyz = np.fromstring(joint.find('origin').get('xyz'),sep=' ')
    rpy = np.fromstring(joint.find('origin').get('rpy'),sep=' ')
    np.testing.assert_allclose(transform(xyz,rpy),mount,atol=1e-14)
    sensor=json.loads((ROBOT/'sensors.json').read_text())['robots']['tiago']['camera']
    np.testing.assert_allclose(mount[:3,2],sensor['forward_parent'],atol=1e-14)
    np.testing.assert_allclose(-mount[:3,1],sensor['up_parent'],atol=1e-14)
    np.testing.assert_allclose(mount[:3,:3].T@mount[:3,:3],np.eye(3),atol=1e-14)
    assert np.linalg.det(mount[:3,:3])==pytest.approx(1.,abs=1e-14)


@pytest.mark.parametrize('seed',range(5))
def test_true_urdf_spatial_jacobian_matches_finite_differences(seed):
    model = TiagoKinematics()
    rng = np.random.default_rng(seed)
    q = model.lower+(model.upper-model.lower)*rng.uniform(.25,.75,8)
    base = transform(rng.normal(size=3),rng.uniform(-.4,.4,3))
    pose,J = model.pose_and_jacobian(base,q)
    numerical = np.zeros((6,10))
    h = 1e-6
    for i in range(10):
        rates = np.eye(10)[i]
        bp,qp = model.integrate(base,q,rates,h)
        bm,qm = model.integrate(base,q,-rates,h)
        plus,_ = model.pose_and_jacobian(bp,qp)
        minus,_ = model.pose_and_jacobian(bm,qm)
        numerical[:3,i] = (plus[:3,3]-minus[:3,3])/(2*h)
        numerical[3:,i] = (Rotation.from_matrix(plus[:3,:3])*Rotation.from_matrix(minus[:3,:3]).inv()).as_rotvec()/(2*h)
    np.testing.assert_allclose(J,numerical,atol=4e-9,rtol=1e-7)


def test_differential_drive_mapping_has_no_lateral_coordinate():
    model = TiagoKinematics()
    base = transform([.2,-.1,.01],[.02,-.03,.7])
    rates = np.r_[.2,.3,np.zeros(8)]
    next_base,_ = model.integrate(base,model.home,rates,.1)
    local_delta = base[:3,:3].T@(next_base[:3,3]-base[:3,3])
    np.testing.assert_allclose(local_delta,[.2*np.sin(.03)/.3,.2*(1-np.cos(.03))/.3,0.],atol=1e-14)
    wheels = differential_wheel_rates(.2,.3)
    radius,separation = model.profile['wheel_radius_m'],model.profile['wheel_separation_m']
    np.testing.assert_allclose([(wheels[0]+wheels[1])*radius/2,(wheels[1]-wheels[0])*radius/separation],[.2,.3])


def reference():
    return TimedCameraReference([dict(t_s=float(t),position_m=[t*.01,0,0],
        linear_velocity_world_mps=[.01,0,0],quaternion_xyzw=Rotation.from_rotvec([.02*t,.03*t,-.05*t]).as_quat())
        for t in (0,1,2)])


def test_continuous_pose_and_world_twist_at_authored_knot():
    curve = reference()
    left,at,right = [curve.evaluate(t) for t in (1-1e-6,1,1+1e-6)]
    np.testing.assert_allclose(left.twist_world,right.twist_world,atol=1e-8)
    np.testing.assert_allclose(at.twist_world,[.01,0,0,.02,.03,-.05],atol=1e-9)
    np.testing.assert_allclose(at.T_world_optical[:3,3],[.01,0,0],atol=1e-14)
    assert tuple(curve.times)==(0,1,2)


@pytest.mark.parametrize('t',[-.1,2.1,float('nan')])
def test_reference_cannot_silently_extrapolate_or_retime(t):
    with pytest.raises(ValueError):
        reference().evaluate(t)


def test_authored_linear_velocity_is_required():
    with pytest.raises(KeyError):
        TimedCameraReference([dict(t_s=t,position_m=[0,0,0],quaternion_xyzw=[0,0,0,1]) for t in (0,1)])


def test_genuine_framework_solver_honors_source_limit_with_soft_tracking():
    pytest.importorskip('piqp',reason='Run wbc/.venv for genuine upstream solver tests')
    from core_task_smoke import GenuineGiskardRateTask,check_reject_infeasible
    model = TiagoKinematics()
    q = model.home.copy()
    q[0] = model.upper[0]
    base = transform()
    _,J = model.pose_and_jacobian(base,q)
    task = GenuineGiskardRateTask(model)
    command,_,residual,violation,_ = task.solve(J,np.array([0,0,.5,0,0,0.]),q,np.zeros(10))
    assert task.source_path.is_relative_to(ROBOT/'wbc/vendor/cognitive_robot_abstract_machine')
    assert not task.solver.big_ball_mode
    assert command[2]<1e-7
    assert q[0]+command[2]*.01<=model.upper[0]+1e-8
    assert residual<1e-5 and violation<1e-6
    assert check_reject_infeasible(task)
