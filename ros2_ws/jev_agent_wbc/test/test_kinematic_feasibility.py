"""Adversarial state, timing, redundancy and actuator checks for the FK gate."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.kinematic_feasibility import KinematicSearch, installation_error, POLICY
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics, WholeBodyQP
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER
from jev_agent_wbc.kc_svsdf import advance
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.task_space_planning import MeasuredCameraPrediction

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    runtime = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    state = np.array(runtime['initial_state'])
    original = OriginalReference(ROOT/'scenarios/movein/task.json', runtime['subject']['position'])
    return state, runtime['base_z'], original


class Line:
    def __init__(self, state, z, velocity=(0., 0., 0.), duration=1.):
        self.p, self.r, _ = whole_body_kinematics(state, z)
        self.velocity = np.array(velocity)
        self.duration = duration

    def sample(self, t):
        return self.p+min(t, self.duration)*self.velocity, self.r.copy(), np.r_[self.velocity, [0., 0., 0.]]


def test_effective_jacobian_obeys_nonholonomic_flow(fixture):
    state, z, _ = fixture
    state[2] = .71
    p, r, j = whole_body_kinematics(state, z)
    for axis in np.eye(8):
        pp, rr, _ = whole_body_kinematics(advance(state, axis, 1e-6), z)
        derivative = np.r_[(pp-p)/1e-6, Rotation.from_matrix(rr@r.T).as_rotvec()/1e-6]
        np.testing.assert_allclose(j@axis, derivative, atol=5e-7)
    np.testing.assert_allclose(j[:3, 0], [np.cos(.71), np.sin(.71), 0.])
    assert j.shape == (6, 8)  # no fictitious independent base lateral input


def test_full_witness_independently_respects_actuator_limits(fixture, monkeypatch):
    state, z, original = fixture
    monkeypatch.setattr(WholeBodyQP, 'solve', lambda *a, **k: pytest.fail('execution WBC invoked'))
    limits = Limits()
    search = KinematicSearch(state, np.zeros(8), 0., z, limits, measured_control=np.zeros(8))
    result = search.lift(MeasuredCameraPrediction(original, state, z, 0., 1.5), 8.4)
    assert result['feasible'], result['reason']
    q = np.array([r['state'][3:] for r in result['trace']])
    u = np.array([r['control'] for r in result['trace']])
    dt = np.diff([r['t'] for r in result['trace']])[:, None]
    assert np.all(q >= JOINT_LOWER-1e-6) and np.all(q <= JOINT_UPPER+1e-6)
    assert np.max(abs(u)/limits.velocity) <= 1+1e-6
    assert np.max(abs(np.diff(u, axis=0))/(dt*limits.acceleration)) <= 1+1e-5
    wheels = u@limits.wheels.T
    assert np.max(abs(wheels)) <= limits.wheel_speed+1e-6
    assert np.max(abs(np.diff(wheels, axis=0))/dt) <= limits.wheel_acceleration+1e-5
    assert result['min_singular_value'] >= POLICY.min_singular_value
    assert result['velocity_source'] == 'measured_control'
    assert np.max(abs(u[-1])) < 1e-6


def test_unreachable_height_is_rejected(fixture):
    state, z, _ = fixture
    result = KinematicSearch(state, np.zeros(8), 0., z, Limits()).lift(Line(state, z, (0., 0., 3.)), 1.)
    assert not result['feasible'] and result['reason'] == 'kinematic_tracking_unreachable'


def test_current_velocity_changes_stationary_path_feasibility(fixture):
    state, z, _ = fixture
    limits = replace(Limits(), acceleration=np.full(8, .02), wheel_acceleration=.1)
    still = KinematicSearch(state, np.zeros(8), 0., z, limits)
    moving = KinematicSearch(state, np.r_[.4, np.zeros(7)], 0., z, limits,
                              measured_control=np.r_[.4, np.zeros(7)])
    path = Line(state, z, duration=30.)
    assert still.lift(path, .4)['feasible']
    assert not moving.lift(path, .4)['feasible']


def test_position_inside_limits_can_be_outside_braking_envelope(fixture):
    state, z, _ = fixture
    state[4] = JOINT_UPPER[1]-.001
    velocity = np.zeros(8); velocity[3] = .9
    search = KinematicSearch(state, velocity, 0., z, Limits())
    assert JOINT_LOWER[1] < state[4] < JOINT_UPPER[1]
    result = search.lift(Line(state, z, duration=8.), .2)
    assert not result['feasible']
    assert result['reason'] == 'braking_or_acceleration_infeasible'


def test_coupled_wheel_speed_is_not_independent_base_limits(fixture):
    state, z, _ = fixture
    limits = replace(Limits(), wheel_speed=5.)
    velocity = np.r_[.3, 1., np.zeros(6)]
    assert np.all(abs(velocity) < limits.velocity)
    search = KinematicSearch(state, velocity, 0., z, limits)
    assert search.initial_error() == 'initial_velocity_limit'


def test_singular_pose_is_rejected_even_when_camera_can_hold():
    state = np.zeros(9)
    search = KinematicSearch(state, np.zeros(8), 0., 0., Limits())
    result = search.lift(Line(state, 0.), .2)
    assert not result['feasible'] and result['reason'] == 'initial_near_singular_configuration'


def test_prior_is_directional_near_joint_boundary(fixture):
    state, z, _ = fixture
    state[5] = JOINT_UPPER[2]-.0101
    search = KinematicSearch(state, np.zeros(8), 0., z, Limits())
    prior = search.prior()
    assert prior['control_upper'][4] < abs(prior['control_lower'][4])*.1
    assert len(prior['control_matrix']) == 10  # eight actuator + coupled wheel bounds
    radial = np.array([1., 0., 0.])
    above, below = [search.amplitude(route, .12, radial) for route in ('above', 'below')]
    assert above != pytest.approx(below)
    assert search.amplitude('above', .22, radial) == .22


def test_same_camera_pose_cannot_hide_changed_redundant_configuration(fixture):
    state, z, _ = fixture
    search = KinematicSearch(state, np.zeros(8), 0., z, Limits())
    witness = search.lift(Line(state, z), .2)
    assert witness['feasible']
    changed = state.copy()
    changed[2] += .2; changed[3] -= .2
    # The first arm joint and base yaw cancel in orientation; translate the
    # base to compensate any mount offset and recover exactly the same FK.
    p, r, _ = whole_body_kinematics(changed, z)
    changed[:2] += search.p[:2]-p[:2]
    pp, rr, _ = whole_body_kinematics(changed, z)
    np.testing.assert_allclose(pp, search.p, atol=1e-12)
    np.testing.assert_allclose(rr, search.r, atol=1e-12)
    assert installation_error(witness, changed, np.zeros(8), 0., Limits()) == 'configuration_left_verified_kinematic_neighborhood'
    velocity = np.r_[.2, np.zeros(7)]
    assert installation_error(witness, state, velocity, 0., Limits()) == 'velocity_left_verified_kinematic_neighborhood'


@pytest.mark.parametrize('value', [[0.]*7, [float('nan')]*8])
def test_invalid_measured_velocity_never_falls_back(fixture, value):
    state, z, _ = fixture
    with pytest.raises(ValueError, match='current velocity'):
        KinematicSearch(state, np.zeros(8), 0., z, Limits(), measured_control=value)


def test_horizon_budget_fails_closed(fixture):
    state, z, _ = fixture
    result = KinematicSearch(state, np.zeros(8), 0., z, Limits()).lift(Line(state, z, duration=50.), 50.)
    assert result['reason'] == 'kinematic_sample_budget_exceeded' and not result['feasible']


def test_low_acceleration_prevents_fast_camera_motion_despite_high_velocity_cap(fixture):
    state, z, _ = fixture
    path = Line(state, z, (.2, 0., 0.), duration=10.)
    limits = replace(Limits(), acceleration=np.full(8, .001), wheel_acceleration=.01)
    result = KinematicSearch(state, np.zeros(8), 0., z, limits).lift(path, .5)
    assert not result['feasible'] and result['reason'] == 'kinematic_tracking_unreachable'
