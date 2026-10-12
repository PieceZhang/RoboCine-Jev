"""Combo4 camera priority must leave hard feasibility and other shots intact."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jvs_mock.dynamic_predictive_qp import DynamicPredictiveQP
from jvs_mock.shot_predictive_qp import ShotPredictiveQP


def controller(profile=None):
    instance = DynamicPredictiveQP.__new__(DynamicPredictiveQP)
    instance.tracking_gain, instance.tracking_weight = 8., 10.
    trajectory = SimpleNamespace(document={} if profile is None else {'dynamic_camera_tracking': profile}, shots=[])
    instance.set_trajectory(trajectory)
    return instance


def test_unconfigured_dynamic_task_keeps_existing_behavior():
    c = controller()
    error, feedforward = np.arange(6.)*.01, np.arange(6.)*.02
    expected = ShotPredictiveQP.camera_task(c, error, feedforward, .3)
    actual = c.camera_task(error, feedforward, .3)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_recovery_cap_preserves_feedforward_and_error_direction():
    c = controller(dict(position_gain=4., position_recovery_speed_m_s=.1, orientation_weight_scale=100.))
    error = np.array([.1, .2, -.1, .01, -.02, .03])
    ff = np.array([.1, -.1, .2, .02, .01, -.01])
    desired, weights = c.camera_task(error, ff, 0.)
    np.testing.assert_allclose(desired[:3]-ff[:3], .1*error[:3]/np.linalg.norm(error[:3]))
    np.testing.assert_allclose(desired[3:], ff[3:]+8.*error[3:])
    np.testing.assert_allclose(weights, [10000.]*3+[180000.]*3)
    small = error*.001
    np.testing.assert_allclose(c.camera_task(small, ff, 0.)[0][:3], ff[:3]+4*small[:3])


@pytest.mark.parametrize('profile', [{'unknown': 1.}, {'position_gain': 0.},
                                    {'orientation_weight_scale': np.nan}, [],
                                    {'position_gain': 'fast'}, {'position_gain': True},
                                    {'joint_reserve_rad': 1.}])
def test_rejects_invalid_profile(profile):
    with pytest.raises(ValueError, match='dynamic_camera_tracking'):
        controller(profile)


def test_tracking_profile_does_not_relax_actuator_bounds():
    from jvs_mock.whole_body_qp import Limits
    from jvs_mock.arm_comfort import COMFORT_POSE
    c = controller()
    c.limits, c.previous = Limits(), np.zeros(8)
    original = c.bounds(COMFORT_POSE, .02, COMFORT_POSE.copy())
    c.set_trajectory(SimpleNamespace(shots=[], document={'dynamic_camera_tracking': dict(
        orientation_weight_scale=100., position_gain=8., position_recovery_speed_m_s=.2,
        joint_reserve_rad=.15, joint_reserve_weight=10000.)}))
    for a, b in zip(original, c.bounds(COMFORT_POSE, .02, COMFORT_POSE.copy())):
        np.testing.assert_array_equal(a, b)


def test_reference_audit_detects_slow_roll_on_top_of_a_moving_shot():
    path = Path(__file__).resolve().parents[1]/'scripts/audit_camera_reference.py'
    spec = importlib.util.spec_from_file_location('reference_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    t = np.linspace(0., 10., 1001)
    planned = Rotation.from_euler('zy', np.c_[.1*t, .02*t])
    wobble = 4.*np.sin(2*np.pi*.2*t)
    actual = planned*Rotation.from_euler('x', wobble, degrees=True)
    roll, angle = module.orientation_errors(actual, planned)
    np.testing.assert_allclose(roll, wobble, atol=1e-12)
    assert angle.max() > 3.99
    np.testing.assert_allclose(module.orientation_errors(planned, planned), 0., atol=1e-12)


@pytest.mark.parametrize('preparing,brake', [(False, False), (True, False), (False, True)])
def test_reserve_recenters_near_limit_without_changing_hard_rows(monkeypatch, preparing, brake):
    from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
    from jvs_mock.jvs_kinematics import JOINT_LOWER, JOINT_UPPER
    c = controller(dict(joint_reserve_rad=.1, joint_reserve_weight=10000.))
    c.preparing = preparing
    q = (JOINT_LOWER+JOINT_UPPER)/2
    q[4], q[5] = JOINT_UPPER[4]-.03, JOINT_LOWER[5]+.04
    state = np.r_[np.zeros(3), q]
    matrix, lower, upper = np.eye(8), -np.ones(8), np.ones(8)
    observed = {}

    def capture(self, state, z, br, dt, H, g, A, lo, hi, brake):
        observed.update(H=H.copy(), g=g.copy())
        np.testing.assert_array_equal(A, matrix)
        np.testing.assert_array_equal(lo, lower)
        np.testing.assert_array_equal(hi, upper)
        raise StopIteration

    monkeypatch.setattr(ObstaclePredictiveQP, 'execution_collision', capture)
    with pytest.raises(StopIteration):
        c.execution_collision(state, 0., np.eye(3), .02, np.eye(8), np.zeros(8),
                              matrix, lower, upper, brake)
    velocity = -np.linalg.solve(observed['H'], observed['g'])
    if preparing or brake:
        np.testing.assert_array_equal(velocity, np.zeros(8))
    else:
        assert velocity[6] < 0 < velocity[7]
        np.testing.assert_array_equal(velocity[:6], np.zeros(6))
