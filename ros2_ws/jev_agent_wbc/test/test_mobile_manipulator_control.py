"""Physical camera Jacobian and bounded nonholonomic command contracts."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mobile_manipulator import control
from mobile_manipulator.control import CameraController, camera_jacobian, wheel_speeds, validate_trajectory


def joint(name='arm', lower=-1., upper=1., home=0., velocity=.4):
    return dict(name=name, lower=lower, upper=upper, home=home, velocity=velocity, acceleration=.8)


def test_wheel_kinematics_preserve_forward_speed_and_yaw_rate():
    radius, separation = .0985, .4044
    for v, yaw in [(.2, 0.), (-.2, 0.), (0., .3), (.1, -.2)]:
        left, right = wheel_speeds(v, yaw, radius, separation)
        assert (left+right)*radius/2 == pytest.approx(v)
        assert (right-left)*radius/separation == pytest.approx(yaw)
    for radius, separation in [(0., .4), (.1, 0.), (-.1, .4)]:
        with pytest.raises(ValueError):
            wheel_speeds(.1, .2, radius, separation)


def test_camera_jacobian_matches_offset_and_tilted_base_finite_differences():
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler('xyz', [.2, -.3, .7]).as_matrix()
    base[:3, 3] = [1., -2., .2]
    camera = np.array([1.4, -1.7, 1.2])
    offset = np.array([.06, -.04, .1])
    measured_link_j = np.random.default_rng(52).normal(size=(6, 3))
    actual = camera_jacobian(base, camera, measured_link_j, offset)
    eps = 1e-7
    expected = np.zeros_like(actual)
    expected[:3, 0] = ((camera+eps*base[:3, 0])-camera)/eps
    up = base[:3, 2]
    expected[:3, 1] = (base[:3, 3]+Rotation.from_rotvec(eps*up).apply(camera-base[:3, 3])-camera)/eps
    expected[3:, 1] = up
    link_origin = camera-offset
    for index in range(3):
        linear, angular = measured_link_j[:3, index], measured_link_j[3:, index]
        moved = link_origin+eps*linear+Rotation.from_rotvec(eps*angular).apply(offset)
        expected[:3, index+2] = (moved-camera)/eps
        expected[3:, index+2] = angular
    np.testing.assert_allclose(actual, expected, atol=3e-7)


def test_repeated_qp_commands_obey_speed_acceleration_wheel_and_position_bounds():
    controller = CameraController([joint(), joint('torso', 0., .35, .18, .07)], .0985, .4044)
    q = controller.home.copy()
    J = np.random.default_rng(9).normal(size=(6, 4))
    for index in range(45):
        previous = controller.previous.copy()
        error = np.array([.3*np.sin(index/8), -.2, .1, .1, -.1, .2])
        speed = controller.velocity(q, J, error)
        assert np.all(np.abs(speed) <= controller.speed+1e-7)
        assert np.all(np.abs(speed-previous) <= controller.accel*controller.dt+1e-7)
        assert np.all(np.abs(wheel_speeds(*speed[:2], controller.radius, controller.separation)) <= controller.wheel_limit+1e-7)
        q += speed[2:]*controller.dt
        assert np.all(q >= controller.lower+.002-1e-7)
        assert np.all(q <= controller.upper-.002+1e-7)


def test_arm_only_leaves_nonholonomic_base_stationary():
    controller = CameraController([joint()], .165, .555)
    speed = controller.velocity([0.], np.ones((6, 3)), np.ones(6), arm_only=True)
    np.testing.assert_array_equal(speed[:2], [0., 0.])
    assert speed[2] > 0.


def test_unrecoverable_joint_limit_does_not_overwrite_last_command():
    controller = CameraController([joint()], .165, .555)
    before = controller.previous.copy()
    with pytest.raises(RuntimeError, match='recovery'):
        controller.velocity([1.2], np.ones((6, 3)), np.zeros(6))
    np.testing.assert_array_equal(controller.previous, before)


@pytest.mark.parametrize('success,candidate', [(False, [0., 0., 0.]), (True, [1e3, 0., 0.]), (True, [np.nan, 0., 0.])])
def test_failed_or_out_of_bounds_solver_result_never_becomes_command(monkeypatch, success, candidate):
    controller = CameraController([joint()], .165, .555)
    monkeypatch.setattr(control, 'minimize', lambda *a, **kw: SimpleNamespace(
        success=success, x=np.asarray(candidate), message='injected solver result'))
    before = controller.previous.copy()
    with pytest.raises(RuntimeError, match='rejected'):
        controller.velocity([0.], np.ones((6, 3)), np.ones(6))
    np.testing.assert_array_equal(controller.previous, before)


@pytest.mark.parametrize('q,jacobian,error', [([np.nan], np.ones((6, 3)), np.zeros(6)),
                                          ([0.], np.full((6, 3), np.inf), np.zeros(6)),
                                          ([0.], np.ones((6, 3)), np.full(6, np.nan)),
                                          ([0., 0.], np.ones((6, 3)), np.zeros(6))])
def test_nonfinite_or_mismatched_state_is_rejected(q, jacobian, error):
    controller = CameraController([joint()], .165, .555)
    with pytest.raises(ValueError):
        controller.velocity(q, jacobian, error)


def trajectory():
    return [dict(t_s=0., position_m=[0., 0., .5], quaternion_xyzw=[0., 0., 0., 1.]),
            dict(t_s=1., position_m=[1., 0., .5], quaternion_xyzw=[0., 0., 1., 0.])]


def test_valid_world_optical_trajectory_interpolates_position_and_rotation():
    stamps, positions, rotations = validate_trajectory(trajectory())
    np.testing.assert_array_equal(stamps, [0., 1.])
    np.testing.assert_array_equal(positions, [[0., 0., .5], [1., 0., .5]])
    np.testing.assert_allclose(rotations(.5).apply([1., 0., 0.]), [0., 1., 0.], atol=1e-14)


@pytest.mark.parametrize('field,value', [('t_s', np.nan), ('t_s', np.inf), ('t_s', 0.), ('t_s', -.5),
                                       ('t_s', .001), ('position_m', [1., 2.]),
                                       ('position_m', [1., 2., np.inf]),
                                       ('quaternion_xyzw', [0., 0., 0., 0.]),
                                       ('quaternion_xyzw', [0., 0., 0., 2.]),
                                       ('quaternion_xyzw', [0., np.nan, 0., 1.])])
def test_malformed_trajectory_samples_are_rejected_before_simulation(field, value):
    samples = trajectory()
    samples[-1][field] = value
    with pytest.raises(ValueError):
        validate_trajectory(samples)


@pytest.mark.parametrize('samples', [[], {}, None, trajectory()[:1]])
def test_incomplete_trajectory_is_rejected(samples):
    with pytest.raises(ValueError):
        validate_trajectory(samples)


@pytest.mark.parametrize('robot_name', ['Robot_Husky_UR5', 'Robot_TIAGo'])
def test_sustained_targets_respect_every_arm_and_torso_limit_then_reverse(robot_name):
    profile = json.loads((Path(__file__).resolve().parents[1]/robot_name/'robot.json').read_text())
    specifications = profile['joints']
    for selected in range(len(specifications)):
        controller = CameraController(specifications, profile['wheel_radius_m'], profile['wheel_separation_m'])
        q = controller.home.copy()
        jacobian = np.zeros((6, len(specifications)+2))
        jacobian[0, selected+2] = 1.
        for direction in [1., -1.]:
            initial_position = q[selected]
            boundary = controller.upper[selected]-.002 if direction > 0 else controller.lower[selected]+.002
            maximum_steps = int(np.ceil((abs(boundary-q[selected])/controller.speed[selected+2]
                                         +2*controller.speed[selected+2]/controller.accel[selected+2]+1.)/controller.dt))
            settled_steps = 0
            for _ in range(maximum_steps):
                previous = controller.previous.copy()
                error = np.array([direction*20., 0., 0., 0., 0., 0.])
                command = controller.velocity(q, jacobian, error)
                assert np.all(np.abs(command-previous) <= controller.accel*controller.dt+1e-7)
                assert np.all(np.abs(command) <= controller.speed+1e-7)
                q += command[2:]*controller.dt
                assert np.all(q >= controller.lower+.002-1e-9)
                assert np.all(q <= controller.upper-.002+1e-9)
                if abs(command[selected+2]) < 1e-6 and direction*(q[selected]-initial_position) > .01:
                    settled_steps += 1
                    if settled_steps == 10:
                        break
                else:
                    settled_steps = 0
            assert settled_steps == 10, specifications[selected]['name']


@pytest.mark.parametrize('prismatic', [False, True])
@pytest.mark.parametrize('upper_boundary', [False, True])
def test_redundant_task_keeps_joint_reserve_under_tracking_perturbations(prismatic, upper_boundary):
    specification = joint('torso' if prismatic else 'arm', 0., .35 if prismatic else 1.,
                          .18 if prismatic else .5, .07 if prismatic else .4)
    controller = CameraController([specification], .0985, .4044)
    q = np.array([specification['upper']-.004 if upper_boundary else .004])
    jacobian = np.zeros((6, 3))
    jacobian[0, 0] = jacobian[0, 2] = 1.
    if not prismatic:
        jacobian[4, 2] = 1.
    direction = 1. if upper_boundary else -1.
    minimum_reserve = float('inf')
    for index in range(300):
        # Bounded physical tracking error alternately moves toward and away
        # from the nearby bound; the task can instead use the mobile base.
        q[0] += direction*(.001 if index % 2 == 0 else -.001)
        previous = controller.previous.copy()
        error = np.zeros(6)
        error[0] = direction*(.025 if index < 150 else -.025)
        command = controller.velocity(q, jacobian, error)
        assert np.all(np.abs(command-previous) <= controller.accel*controller.dt+1e-7)
        q += command[2:]*controller.dt
        assert np.all(q >= controller.lower+.002-1e-9)
        assert np.all(q <= controller.upper-.002+1e-9)
        minimum_reserve = min(minimum_reserve, float(np.minimum(q-controller.lower, controller.upper-q).min()))
    assert minimum_reserve > .002
    assert (specification['upper']-q[0] if upper_boundary else q[0]) > .01
