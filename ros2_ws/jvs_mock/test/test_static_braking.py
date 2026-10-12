"""Early braking for a newly observed voxel preserves hard feasibility."""
import json
from pathlib import Path

import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import linprog

from jvs_mock.kc_svsdf import advance
from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
from jvs_mock.whole_body_qp import Limits


FIXTURE = json.loads((Path(__file__).parent/'fixtures/static_voxel_overtake_braking.json').read_text())


def scene(point):
    return dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([point], float),
                surfel_radius=FIXTURE['surfel_radius'], minimum_clearance_m=FIXTURE['minimum_clearance_m'])


def constraints(controller, state, dt, commanded_q=None):
    A, lo, hi = controller.bounds(state[3:], dt, state[3:] if commanded_q is None else commanded_q)
    return controller.execution_collision(state, FIXTURE['base_z'], None, dt,
        np.eye(8), np.array([.1, 0., 0., 0., 0., 0., 0., 0.]), A, lo, hi)


class OriginalBarrier(ObstaclePredictiveQP):
    def static_barrier_terms(self, state, distances, gradients, braking_control=None):
        return super().static_barrier_terms(state, distances, gradients, None)


def approach(controller_class):
    observed = scene(FIXTURE['earlier_point'])
    controller = controller_class(Limits(), scene=observed)
    controller.preparing = True
    state = np.array(FIXTURE['state'])
    # Approach the real measured chassis witness while asking for the same
    # modest 0.1 m/s base motion. At 1.2 s a nearer voxel becomes visible.
    state[:2] -= -.1*np.array([np.cos(state[2]), np.sin(state[2])])*1.2
    minimum, largest_acceleration, first_braking = np.inf, 0., None
    for tick in range(180):
        if tick == 60:
            controller.update_collision_map({**observed,
                'surfel_centers': np.array([FIXTURE['observed_point']])})
        H, g, A, lo, hi = constraints(controller, state, .02)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(H), q=g, A=sparse.csc_matrix(A), l=lo, u=hi,
            verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=10000, polish=False)
        solver.warm_start(x=controller.previous)
        result = solver.solve()
        if result.info.status_val != 1:
            return dict(completed=False, failed_tick=tick)
        command = result.x
        assert np.max(np.maximum(lo-A@command, A@command-hi)) < 1e-7
        largest_acceleration = max(largest_acceleration, abs(command[0]-controller.previous[0])/.02)
        if tick > 10 and first_braking is None and command[0] > -.095:
            first_braking = tick
        certificate = controller.collision.sweep(state, command, .02,
            FIXTURE['base_z'], tolerance=.0001, max_samples=33)
        minimum = min(minimum, float(certificate['lower'].min()))
        controller.previous = command.copy()
        state = advance(state, command, .02)
    return dict(completed=True, minimum_clearance=minimum,
                largest_acceleration=largest_acceleration, first_braking_tick=first_braking)


def test_real_overtake_witness_starts_braking_before_new_voxel_causes_infeasibility():
    original = approach(OriginalBarrier)
    assert not original['completed'] and original['failed_tick'] == 60
    protected = approach(ObstaclePredictiveQP)
    assert protected['completed']
    assert protected['first_braking_tick'] < 60
    assert protected['minimum_clearance'] > FIXTURE['minimum_clearance_m']
    assert protected['largest_acceleration'] <= Limits().acceleration[0]+1e-7


def test_already_infeasible_recorded_state_is_not_made_feasible_by_relaxing_safety():
    state = np.array(FIXTURE['state'])
    controllers = [cls(Limits(), scene=scene(FIXTURE['observed_point']))
                   for cls in (OriginalBarrier, ObstaclePredictiveQP)]
    for controller in controllers:
        controller.preparing = True
        controller.previous = np.array(FIXTURE['previous'])
        _, _, A, lo, hi = constraints(controller, state, FIXTURE['dt'], np.array(FIXTURE['commanded_q']))
        finite = np.isfinite(hi)
        result = linprog(np.zeros(8), A_ub=np.vstack([-A, A[finite]]),
            b_ub=np.r_[-lo, hi[finite]], bounds=[(None, None)]*8, method='highs')
        assert not result.success


def test_early_braking_band_never_loosens_an_original_collision_row():
    controller = ObstaclePredictiveQP(Limits(), scene=scene(FIXTURE['earlier_point']))
    state = np.array(FIXTURE['state'])
    centers, radii = controller.collision.active_obstacles(state, FIXTURE['base_z'], per_body=None)
    distances, gradients = controller.collision.geometry.primitive_distances(state, centers, FIXTURE['base_z'])
    distances -= radii[:, None]
    rows, original, mask = controller.static_barrier_terms(state, distances, gradients)
    protected_rows, protected, protected_mask = controller.static_barrier_terms(
        state, distances, gradients, np.zeros(8))
    np.testing.assert_array_equal(protected_rows, rows)
    assert np.all(protected >= original)
    assert np.all(protected_mask[mask])
    assert np.any(protected > original)


def test_common_braking_reference_respects_coupled_wheel_acceleration_bounds():
    limits = Limits(acceleration=np.array([1., 10., 2., 2., 2., 2., 2., 2.]), wheel_acceleration=.5)
    controller = ObstaclePredictiveQP(limits, scene=scene(FIXTURE['earlier_point']))
    controller.previous[:2] = [.2, .3]
    A, lo, hi = controller.bounds(np.array(FIXTURE['state'])[3:], .02)
    # Braking each chassis axis independently exceeds one wheel's deceleration.
    assert controller.hardware_braking_control(A, lo, hi) is None
