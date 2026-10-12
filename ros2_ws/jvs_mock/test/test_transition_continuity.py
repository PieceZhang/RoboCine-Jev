"""Stationary camera holds must not make the base guide jump or reverse."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
import pytest

from jvs_mock.arm_comfort import COMFORT_POSE
from jvs_mock.jvs_kinematics import JOINT_NAMES, JOINT_LOWER, JOINT_UPPER
from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
from jvs_mock.shot_execution import ShotClock
from jvs_mock.shot_predictive_qp import wrap
from jvs_mock.shot_preview_worker import AsyncObstaclePredictiveQP
from jvs_mock.whole_body_qp import whole_body_kinematics
from jvs_mock.wbc_trajectory import CameraTrajectory


def test_controller_joint_bounds_match_the_imported_robot_and_command_interfaces():
    root = Path(__file__).resolve().parents[1]
    robot = ElementTree.parse(root/'urdf/jvs_mock.urdf.xacro')
    interfaces = ElementTree.parse(root/'urdf/ros2_control.xacro')
    for index, name in enumerate(JOINT_NAMES):
        limit = robot.find(f'.//joint[@name="{name}"]/limit')
        interface = interfaces.find(f'.//joint[@name="{name}"]/command_interface[@name="position"]')
        assert JOINT_LOWER[index] == pytest.approx(float(limit.attrib['lower']), abs=1e-12)
        assert JOINT_UPPER[index] == pytest.approx(float(limit.attrib['upper']), abs=1e-12)
        assert JOINT_LOWER[index] == pytest.approx(float(interface.find('param[@name="min"]').text), abs=1e-12)
        assert JOINT_UPPER[index] == pytest.approx(float(interface.find('param[@name="max"]').text), abs=1e-12)


def test_combo3_orbit_direction_accounts_for_camera_pose_fit():
    root = Path(__file__).resolve().parents[1]
    p, r, _ = whole_body_kinematics(np.zeros(9))
    trajectory = CameraTrajectory(root/'trajectories/shot_combo3.json', p, r, np.eye(3))
    controller = ObstaclePredictiveQP(scene=dict(format='jvs_obstacles/v1',
        minimum_clearance_m=.025, obstacles=[dict(type='sphere',
        name='far', center=[3., 3., .2], radius=.1)]))
    controller.set_trajectory(trajectory)
    controller.route = SimpleNamespace(sample=lambda t: trajectory.sample(t)[0][:2],
        safe_segment=lambda *_: True, reachable=lambda _: True)
    state = np.r_[0., 0., 0., COMFORT_POSE]
    # The reverse guide looked slightly more comfortable while its bounded IK
    # left a larger camera pose residual. The real orbit uses the forward guide.
    orbit = next(shot for shot in trajectory.shots if shot['type'] == 'orbit')
    probe = orbit['motion_start_s'] + (orbit['motion_end_s']-orbit['motion_start_s'])/18.
    assert controller._base_reference(state, probe)[2] == 1


def test_identical_hold_ik_inputs_share_a_solve_and_changed_seed_invalidates_it():
    controller = ObstaclePredictiveQP(scene=dict(format='jvs_obstacles/v1',
        minimum_clearance_m=.025, obstacles=[dict(type='sphere',
        name='far', center=[3., 3., .2], radius=.1)]))
    position, rotation, _ = whole_body_kinematics(np.r_[np.zeros(3), COMFORT_POSE])
    controller.trajectory = SimpleNamespace(sample=lambda _: (position, rotation, np.zeros(6)))
    state = np.r_[np.zeros(3), COMFORT_POSE]
    from jvs_mock.arm_comfort import preferred_posture
    with patch('jvs_mock.arm_comfort.preferred_posture', wraps=preferred_posture) as solve:
        first = controller.comfort_reference(state, 20., initial_q=state[3:])
        second = controller.comfort_reference(state, 23., initial_q=state[3:])
        np.testing.assert_array_equal(first[1], second[1])
        assert solve.call_count == 1
        state[3] += .01
        controller.comfort_reference(state, 23., initial_q=state[3:])
        assert solve.call_count == 2
        controller.reference_base_z += .001
        controller.comfort_reference(state, 23., initial_q=state[3:])
        assert solve.call_count == 3
        rotation[:] = np.roll(rotation, 1, axis=1)
        controller.comfort_reference(state, 23., initial_q=state[3:])
        assert solve.call_count == 4


@pytest.fixture
def parked_camera():
    controller = ObstaclePredictiveQP(scene=dict(format='jvs_obstacles/v1',
        minimum_clearance_m=.025, obstacles=[dict(type='sphere',
        name='far', center=[3., 3., .2], radius=.1)]))
    controller.trajectory = SimpleNamespace(
        sample=lambda _: (np.array([0., 0., .7]), np.eye(3), np.zeros(6)))
    controller.route = SimpleNamespace(sample=lambda t: np.array([0., t]),
        reachable=lambda _: True, safe_segment=lambda *_: True)
    window = dict(start_s=20., end_s=24., reference_s=24.5)
    with patch('jvs_mock.arm_comfort.preferred_posture',
               return_value=(COMFORT_POSE.copy(), np.zeros(3))):
        with patch.object(controller, '_base_reference',
                          return_value=(0., np.pi/2, 1)):
            yield controller, window


def test_position_noise_at_old_parking_threshold_keeps_guide_continuous(parked_camera):
    controller, window = parked_camera
    references = []
    for distance in [.07001, .06999, .07002, .06998]:
        state = np.zeros(9)
        state[:2] = -distance*np.array([.8, .6])
        state[2] = np.pi/2
        references.append(controller.transition_reference(state, 22., window))
    speeds, headings, _ = np.array(references).T
    assert np.min(np.abs(speeds)) > .001
    assert np.max(np.abs(np.diff(speeds))) < .001
    assert np.max(np.abs(wrap(np.diff(headings)))) < .002


def test_outgoing_stopped_guide_has_no_distance_threshold_jump():
    controller = ObstaclePredictiveQP(scene=dict(format='jvs_obstacles/v1',
        minimum_clearance_m=.025, obstacles=[dict(type='sphere',
        name='far', center=[3., 3., .2], radius=.1)]))
    context = dict(shot_type='moveout', shot_phase=0., next_shot='orbit',
                   segment_index=0, start_s=24., end_s=40.)
    controller.trajectory = SimpleNamespace(tangent=lambda _: 0.,
        shot_context=lambda _: context)
    controller.route = SimpleNamespace(sample=lambda t: np.array([max(t-25., 0.), 0.]))
    controller.directions[('base_guide', 0, 40.)] = 1
    references = []
    with patch.object(controller, 'comfort_base_position', return_value=np.zeros(2)), \
         patch.object(controller, 'terminal_approach', return_value=False):
        for distance in [.02501, .02499, .02502, .02498]:
            state = np.zeros(9)
            state[:2] = -distance*np.array([.8, .6])
            references.append(controller._base_reference(state, 24.5))
    speeds, headings, _ = np.array(references).T
    assert np.max(np.abs(np.diff(speeds))) < .001
    assert np.max(np.abs(wrap(np.diff(headings)))) < .002


def test_measured_posture_keeps_stopped_comfort_goal_on_same_branch():
    controller = ObstaclePredictiveQP(scene=dict(format='jvs_obstacles/v1',
        minimum_clearance_m=.025, obstacles=[dict(type='sphere',
        name='far', center=[3., 3., .2], radius=.1)]))
    rotation = np.array([[.9543153981972984, .25865296397463194, .14960202201397518],
                         [-.2555674564879122, .9659702872324408, -.03983314409352087],
                         [-.15481406895961128, -.00022002548450154522, .9879435994230416]])
    controller.trajectory = SimpleNamespace(sample=lambda _: (np.array([0., 0., .978677]), rotation, np.zeros(6)))
    controller.route = SimpleNamespace(sample=lambda _: np.zeros(2), safe_segment=lambda *_: True)
    offsets = []
    # Two successive measured postures from the Pan -> MoveOut junction.
    with patch.object(controller, 'terminal_approach', return_value=False):
        for yaw, joints in [(.220140618, [-.6701, -.2034, .3338, .9102, .025, -.6981]),
                            (.191192275, [-.6543, -.2103, .3403, .9239, .025, -.6981])]:
            offsets.append(controller.comfort_base_position(np.r_[0., 0., yaw, joints], 24.))
    assert np.linalg.norm(offsets[1]-offsets[0]) < .01


def test_heading_noise_cannot_flip_parking_drive_direction(parked_camera):
    controller, window = parked_camera
    references = []
    for perturbation in [-1e-5, 1e-5, -2e-5, 2e-5]:
        state = np.zeros(9)
        state[0], state[2] = -.2, -np.pi/2 + perturbation
        references.append(controller.transition_reference(state, 22., window))
    _, headings, directions = np.array(references).T
    assert len(set(directions)) == 1
    assert np.max(np.abs(wrap(np.diff(headings)))) < .001


def test_hold_entry_and_exit_match_adjacent_motion_guides(parked_camera):
    controller, window = parked_camera
    state = np.zeros(9)
    state[0] = -.2

    def adjacent(_, time):
        return (.08, .7, 1) if time < 20. else (.02, np.pi/2, 1)

    with patch.object(controller, '_base_reference', side_effect=adjacent):
        entry = controller.transition_reference(state, 20., window)
        exit_ = controller.transition_reference(state, 24.-1e-7, window)
    np.testing.assert_allclose(entry[:2], [.08, .7], atol=1e-6)
    np.testing.assert_allclose(exit_[:2], [.02, np.pi/2], atol=1e-6)


def test_next_shot_joins_prepared_heading_before_lateral_recovery(parked_camera):
    controller, window = parked_camera
    controller.trajectory.shots = [dict(start_s=0., end_s=22., motion_end_s=20.),
                                  dict(start_s=22., end_s=40., motion_start_s=24.)]
    state = np.zeros(9)
    state[0] = -.2
    with patch.object(controller, '_base_reference', return_value=(.02, -2.8, 1)):
        exit_ = controller.transition_reference(state, 24.-1e-7, window)
    outgoing = (.02, -2.8, 1)
    entry = controller.follow_transition(24.+1e-7, outgoing)
    joined = controller.follow_transition(25.-1e-7, outgoing)
    assert wrap(entry[1]-exit_[1]) == pytest.approx(0., abs=1e-6)
    assert wrap(joined[1]-outgoing[1]) == pytest.approx(0., abs=1e-6)


def test_following_guide_cannot_flip_its_half_turn_branch(parked_camera):
    controller, window = parked_camera
    controller.trajectory.shots = [dict(start_s=0., end_s=22., motion_end_s=20.),
                                  dict(start_s=22., end_s=40., motion_start_s=24.)]
    controller.directions[('transition_heading', 20., 24.)] = 0.
    headings = [controller.follow_transition(24.5, (0., np.pi+noise, 1))[1]
                for noise in [-1e-5, 1e-5, -2e-5, 2e-5]]
    assert np.max(np.abs(wrap(np.diff(headings)))) < .001


def test_parking_does_not_recompute_the_incoming_guide(parked_camera):
    controller, window = parked_camera
    state = np.zeros(9)
    state[0] = -.2
    with patch.object(controller, '_base_reference', return_value=(.02, .7, 1)):
        entry = controller.transition_reference(state, 20., window)
    with patch.object(controller, '_base_reference', return_value=(.02, 2.8, 1)):
        later = controller.transition_reference(state, 20.+1e-6, window)
    np.testing.assert_allclose(entry[:2], later[:2], atol=1e-6)


def test_preview_does_not_latch_a_hypothetical_hold_entry(parked_camera):
    controller, window = parked_camera
    state = np.zeros(9)
    state[0] = -.2
    controller.planning = True
    with patch.object(controller, '_base_reference', return_value=(.02, 2.8, 1)):
        controller.transition_reference(state, 20., window)
    controller.planning = False
    with patch.object(controller, '_base_reference', return_value=(.02, .7, 1)):
        entry = controller.transition_reference(state, 20., window)
    assert entry[1] == pytest.approx(.7)


def test_arrived_base_prepares_heading_without_translation(parked_camera):
    controller, window = parked_camera
    speed, heading, _ = controller.transition_reference(np.zeros(9), 22., window)
    assert speed == pytest.approx(0.)
    assert wrap(heading - np.pi/2) == pytest.approx(0.)


def test_stationary_projected_guide_uses_future_heading(parked_camera):
    controller, window = parked_camera
    controller.route.sample = lambda _: np.zeros(2)
    with patch.object(controller, '_base_reference', return_value=(0., 1.2, 1)):
        speed, heading, _ = controller.transition_reference(np.zeros(9), 22., window)
    assert speed == pytest.approx(0.)
    assert heading == pytest.approx(1.2)


def test_arrival_heading_does_not_jump_at_half_turn(parked_camera):
    controller, window = parked_camera
    controller.route.sample = lambda t: np.array([t, 0.])
    references = []
    for lateral in [-1e-5, 1e-5, -2e-5, 2e-5]:
        state = np.zeros(9)
        state[0], state[1], state[2] = .07, lateral, np.pi
        references.append(controller.transition_reference(state, 22., window))
    _, headings, directions = np.array(references).T
    assert len(set(directions)) == 1
    assert np.max(np.abs(wrap(np.diff(headings)))) < .002


def test_parking_heading_stays_continuous_while_turning_past_pi(parked_camera):
    controller, window = parked_camera
    controller.route.sample = lambda t: np.array([t, 0.])
    headings = []
    for angle in np.linspace(2.6, 2*np.pi-2.4, 300):
        state = np.zeros(9)
        state[:2] = -.07*np.array([np.cos(angle), np.sin(angle)])
        state[2] = 2.6
        headings.append(controller.transition_reference(state, 22., window)[1])
    assert np.max(np.abs(wrap(np.diff(headings)))) < .01


def test_arrival_noise_cannot_restart_parking(parked_camera):
    controller, window = parked_camera
    state = np.zeros(9)
    state[0] = -.2
    controller.transition_reference(state, 21.5, window)
    state[0] = -.0399
    controller.transition_reference(state, 22., window)
    references = []
    for distance in [.0401, .0399, .0402, .0398]:
        state[0] = -distance
        references.append(controller.transition_reference(state, 22.6, window))
    speeds, headings, _ = np.array(references).T
    np.testing.assert_allclose(speeds, 0., atol=1e-8)
    np.testing.assert_allclose(headings, np.pi/2, atol=1e-8)


def test_blocked_parking_reserves_time_for_next_heading(parked_camera):
    controller, window = parked_camera
    controller.trajectory.transition = lambda _: window
    state = np.zeros(9)
    state[0] = -.2
    controller.transition_reference(state, 21.5, window)
    assert controller.transition_preparing(21.5)
    controller.transition_reference(state, 22., window)
    speed, heading, _ = controller.transition_reference(state, 22.6, window)
    assert not controller.transition_preparing(22.6)
    assert speed == pytest.approx(0.)
    assert heading == pytest.approx(np.pi/2)


def test_clock_keeps_time_for_parking_even_with_aligned_heading():
    clock = ShotClock()
    for _ in range(200):
        clock.advance(.02, 0., 0., transition=True, preparing=True)
    assert clock.rate == pytest.approx(.15)
    for _ in range(200):
        previous = clock.rate
        clock.advance(.02, 0., 0., transition=True, preparing=False)
        assert abs(clock.rate-previous) <= .00800001
    assert clock.rate == pytest.approx(.7)


def test_moving_connector_does_not_park_for_a_stationary_camera():
    controller = ObstaclePredictiveQP(scene=dict(format='jvs_obstacles/v1',
        minimum_clearance_m=.025, obstacles=[dict(type='sphere',
        name='far', center=[3., 3., .2], radius=.1)]))
    controller.trajectory = SimpleNamespace(
        sample=lambda t: (np.array([.01*t, 0., .7]), np.eye(3), np.array([.01, 0., 0., 0., 0., 0.])),
        transition=lambda _: dict(start_s=20., end_s=24.))
    controller.directions[('transition_parking', 20., 24.)] = 1
    assert controller.stationary_transition(22.) is None
    assert controller.hold_weight(22.) == 0.
    assert not controller.transition_preparing(22.)


def test_stationary_hold_prioritizes_camera_over_conflicting_base_preparation():
    state = np.r_[np.zeros(3), COMFORT_POSE]
    position, rotation, jacobian = whole_body_kinematics(state)
    camera_speeds = []
    for holding in [False, True]:
        controller = ObstaclePredictiveQP(scene=dict(format='jvs_obstacles/v1',
            minimum_clearance_m=.025, obstacles=[dict(type='sphere',
            name='far', center=[3., 3., .2], radius=.1)]))
        controller.trajectory = SimpleNamespace(sample=lambda _: (position, rotation, np.zeros(6)),
            transition=lambda _: dict(start_s=20., end_s=24.) if holding else None)
        controller.last_plan = np.inf
        context = dict(shot_type='neutral', shot_phase=.5, next_shot='neutral')

        def prepare(_, z, br, dt, hessian, gradient, matrix, lower, upper, brake):
            # A deliberately competing preparation target, inside the same QP.
            hessian[:2, :2] += np.diag([1200., 500.])
            gradient[:2] -= np.array([1200., 500.])*[.15, .5]
            return hessian, gradient, matrix, lower, upper

        with patch.object(controller, 'context', return_value=context), \
             patch.object(controller, 'motion_weights', return_value=dict(neutral=1., orbit=0., pan=0., moveout=0.)), \
             patch.object(controller, 'heading', return_value=(0., 1)), \
             patch.object(controller, 'execution_collision', side_effect=prepare):
            command, info = controller.solve(state, position, rotation, np.zeros(6), .02,
                                             trajectory_time=22.)
        camera_speeds.append(np.linalg.norm(jacobian@command))
        assert info['qp_violation'] <= 1e-6
        assert info['command_position_violation'] <= 1e-6
    assert camera_speeds[1] < .8*camera_speeds[0]


@pytest.mark.parametrize('older_branch', [None, -1])
def test_completed_preview_preserves_execution_parking_branch(older_branch):
    scene = dict(format='jvs_obstacles/v1', minimum_clearance_m=.025,
        obstacles=[dict(type='sphere', name='far', center=[3., 3., .2], radius=.1)])
    controller = AsyncObstaclePredictiveQP(scene=scene)
    key, turn_key = ('transition', 20., 24.), ('transition_entry_turn', 20., 24.)
    guide_key = ('base_guide', 1, 36.)
    try:
        controller.directions.update({key: 1, turn_key: 1, guide_key: 1})
        controller.transition_entries[(20., 24.)] = (.02, .7, 1)
        directions = {} if older_branch is None else {key: older_branch, turn_key: older_branch, guide_key: older_branch}
        completed = Future()
        completed.set_result(dict(directions=directions, transition_entries={}))
        controller.pending = completed
        with patch.object(ObstaclePredictiveQP, 'solve', return_value=(np.zeros(8), {})):
            controller.solve(trajectory_time=22., dt=.02)
        assert controller.directions[key] == 1
        assert controller.directions[turn_key] == 1
        assert controller.directions[guide_key] == 1
        assert controller.transition_entries[(20., 24.)] == (.02, .7, 1)
    finally:
        controller.close()


def test_stale_preview_supplies_guide_costs_without_committing_future_branches():
    scene = dict(format='jvs_obstacles/v1', minimum_clearance_m=.025,
        obstacles=[dict(type='sphere', name='far', center=[3., 3., .2], radius=.1)])
    controller = AsyncObstaclePredictiveQP(scene=scene)
    current, upcoming = ('base_guide', 0, 22.), ('base_guide', 1, 40.)
    try:
        controller.directions[current] = 1
        controller.execution_clock = 1.
        completed = Future()
        completed.set_result(dict(route=SimpleNamespace(), directions={current: -1, upcoming: -1},
                                  guide_costs={upcoming: (.5, 1.3)},
                                  plan=np.ones((controller.n, 8)), reference=np.ones(8)))
        controller.pending = completed
        with patch.object(ObstaclePredictiveQP, 'solve', return_value=(np.zeros(8), {})):
            controller.solve(trajectory_time=22., dt=.02)
        assert controller.directions[current] == 1
        assert upcoming not in controller.directions
        assert controller.guide_costs[upcoming] == (.5, 1.3)
        assert controller.plan is None
        np.testing.assert_array_equal(controller.reference, 0.)
        assert controller.stale_plans == 1
    finally:
        controller.close()
