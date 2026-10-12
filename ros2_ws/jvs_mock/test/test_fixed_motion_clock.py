"""Physical action timing despite tracking lag, and adaptive connector timing."""
from types import SimpleNamespace

import numpy as np
import pytest

from jvs_mock.shot_execution import FixedMotionShotClock
from jvs_mock.dynamic_predictive_qp import DynamicPredictiveQP
from jvs_mock.whole_body_qp import Limits


def trajectory():
    shots = [dict(type='orbit', start_s=0., end_s=3., motion_start_s=1., motion_end_s=2.),
             dict(type='pan', start_s=3., end_s=5., motion_start_s=4., motion_end_s=5.)]
    return SimpleNamespace(shots=shots, transition=lambda t: {} if 2. <= t < 4. else None)


@pytest.mark.parametrize('lag', [0., .02, .5])
def test_tracking_lag_does_not_retime_an_action(lag):
    clock = FixedMotionShotClock()
    clock.set_trajectory(trajectory())
    clock.time = 1.
    clock.rate = .15
    for _ in range(50):
        clock.advance(.02, lag, lag)
    assert clock.time == pytest.approx(2.)
    assert clock.rate == pytest.approx(1.)


def test_connector_preparation_remains_adaptive_and_action_boundary_is_split():
    clock = FixedMotionShotClock()
    clock.set_trajectory(trajectory())
    clock.time, clock.rate = 3.997, .15
    t, rate = clock.advance(.04, .4, .4, transition=True, preparing=True)
    # Twenty milliseconds finish the connector; the other twenty execute pan.
    assert t == pytest.approx(4.02)
    assert rate == 1.
    clock.time, clock.rate = 2.1, .15
    t, rate = clock.advance(.02, .4, .4, transition=True, preparing=True)
    assert t == pytest.approx(2.103)
    assert rate == .15


def test_dynamic_replanning_uses_physical_time_even_in_a_slow_connector():
    scene = dict(format='jvs_lidar_surfels/v1', observation_stamp=0.,
                 surfel_centers=np.array([[2., 0., .5]]), surfel_radius=.04)
    planner = DynamicPredictiveQP(Limits(), scene=scene)
    planner.last_plan = 3.
    planner.last_plan_physical_time = 10.
    planner.path_rate = .15
    planner.set_time(10.19)
    assert not planner.replan_due(3.)
    planner.set_time(10.2)
    assert planner.replan_due(3.)
    planner.pending = object()
    assert not planner.replan_due(3.)
    planner.pending = None
    planner.last_plan = -np.inf
    assert planner.replan_due(3.)
