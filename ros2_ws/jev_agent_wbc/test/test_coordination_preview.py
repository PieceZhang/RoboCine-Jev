"""Cross-phase reserve, ramp dynamics, and live preview isolation regressions."""
from concurrent.futures import Future
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.coordination import CoordinationPreview, AsyncCoordinationPreview
from jev_agent_wbc.reference import OriginalReference, smoothstep
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER
from jev_agent_wbc.planning import make_wbc
from jev_agent_wbc.actuation import advance_command
from jev_agent_wbc.safety import solve_with_certificate

ROOT = Path(__file__).resolve().parents[1]


def composite(pan=.35, moving=False):
    state = np.array([0., 0., 0., 0., -.9, 2.1, 0., -1.2, 0.])
    p, _, _ = whole_body_kinematics(state, .002)
    subject = p+np.array([1.55, 0., 0.])
    samples = []
    for t in np.linspace(0., 8., 81):
        angle = .16*smoothstep(min(t/4., 1.))
        R = Rotation.from_euler('z', angle).as_matrix()
        position = subject+R@(p-subject)
        if t > 4.:
            if moving:
                position += np.array([.1*smoothstep((t-4.)/4.), 0., 0.])
            angle += pan*smoothstep((t-4.)/4.)
        samples.append(dict(t=float(t), position=position.tolist(), quaternion_xyzw=Rotation.from_euler('z', angle).as_quat().tolist()))
    task = dict(reference_camera_trajectory=dict(frame='world_z_up_camera_link', samples=samples),
                shot_phase=[dict(name='arc', motion='orbit', start_s=0., end_s=4.), dict(name='pan', motion='pan', start_s=4., end_s=8.)],
                subject=dict(id='blue_subject', description='blue subject'), intent='Orbit then pan continuously within eight seconds.')
    return OriginalReference(task, subject), state


def matched_limits():
    lim = Limits(command_ramp=True, position_margin=.03, deadline_s=8.)
    lim.velocity *= .95; lim.wheel_speed *= .95; lim.acceleration *= .4
    lim.acceleration[1] = .15; lim.wheel_acceleration *= .4
    return lim


def scene(t=0.):
    return dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[3., 3., .2]]),
                surfel_radius=.025, observation_stamp=t, minimum_clearance_m=.025)


def test_ramp_jacobians_and_previous_velocity_are_part_of_dynamics():
    rng = np.random.default_rng(13)
    x, old, new = rng.normal(size=9), rng.normal(size=8), rng.normal(size=8)
    y, A, B = CoordinationPreview.step(x, old, new, .4)
    for vector, analytical, index in [(x, A, 0), (old, B, 1), (new, B, 2)]:
        numerical = np.empty_like(analytical)
        for j in range(len(vector)):
            args = [x.copy(), old.copy(), new.copy(), .4]
            args[index][j] += 1e-6
            plus = CoordinationPreview.step(*args)[0]
            args[index][j] -= 2e-6
            minus = CoordinationPreview.step(*args)[0]
            numerical[:, j] = (plus-minus)/2e-6
        np.testing.assert_allclose(numerical, analytical, atol=1e-8)
    assert not np.allclose(y, CoordinationPreview.step(x, np.zeros(8), new, .4)[0])


def test_stationary_geometry_not_pan_name_enables_coordination():
    preview = CoordinationPreview()
    preview.set_trajectory(composite()[0]); assert preview.enabled
    preview.set_trajectory(composite(moving=True)[0]); assert not preview.enabled


def test_changed_target_discards_old_controls_even_when_original_hash_matches():
    reference, _ = composite()
    preview = CoordinationPreview(); preview.set_trajectory(reference)
    preview.controls = np.ones((2, 8)); preview.times = np.array([0., 8.])
    changed = deepcopy(reference)
    assert changed.sha256 == reference.sha256
    preview.set_trajectory(changed)
    assert preview.controls is None and preview.times is None


def test_opposite_future_pan_changes_allocation_before_the_junction():
    plans = []
    for direction in (-.35, .35):
        reference, state = composite(direction)
        preview = CoordinationPreview(); preview.set_trajectory(reference)
        assert preview.update(state, np.zeros(8), state[3:], 0., .02, .002, matched_limits()) is not None
        assert preview.info['coordination_horizon_end_s'] == 8.
        assert not preview.info['coordination_certificate']
        plans.append(preview.controls.copy())
    assert np.max(np.abs(plans[0][:10]-plans[1][:10])) > .01


def test_cross_phase_closed_loop_reserves_arm_and_finishes_without_retiming():
    reference, state = composite(); original_hash = reference.sha256
    limits = matched_limits(); controller = make_wbc(limits, scene(), reference)
    qcmd = state[3:].copy(); errors = []; peak_turn = 0.
    for k in range(401):
        t = k*.02; controller.update_collision_map(scene(t)); controller.set_time(t)
        previous = controller.previous.copy()
        u, info = solve_with_certificate(controller, state, *reference.sample(t), .02,
            base_z=.002, commanded_q=qcmd, trajectory_time=t, brake=t >= 8.)
        errors.append([info['position_error'], info['orientation_error']])
        peak_turn = max(peak_turn, abs(u[1]))
        assert np.max(np.abs(u-previous)-limits.acceleration*.02) < 1e-6
        assert np.all(state[3:] >= JOINT_LOWER) and np.all(state[3:] <= JOINT_UPPER)
        if t < 8.:
            state = advance_command(state, previous, u, .02, True)
            qcmd += .01*(previous[2:]+u[2:])
    assert reference.sha256 == original_hash and reference.duration == 8.
    assert max(e[0] for e in errors) < .05
    assert errors[200][1] < .1 and errors[-1][1] < .1
    assert np.linalg.norm(u) < 1e-5
    assert np.min(np.minimum(state[3:]-JOINT_LOWER, JOINT_UPPER-state[3:])) > .03


def test_preview_cannot_override_fast_actuator_bounds_or_braking():
    reference, state = composite()
    controller = make_wbc(matched_limits(), scene(), reference)
    controller.coordination.update = lambda *args, **kwargs: np.full(8, 1000.)
    controller.set_time(0.)
    u, _ = controller.solve(state, *reference.sample(0.), .02, base_z=.002, commanded_q=state[3:])
    assert np.max(np.abs(u)-controller.limits.acceleration*.02) <= 1e-6
    def forbidden(*args, **kwargs):
        raise AssertionError('braking must not invoke a preview')
    controller.coordination.update = forbidden
    previous = controller.previous.copy()
    u, info = controller.solve(state, *reference.sample(.02), .02, base_z=.002, commanded_q=state[3:], brake=True)
    assert not info['coordination_active']
    assert np.max(np.abs(u-previous)-controller.limits.acceleration*.02) <= 1e-6


def fake_result(reference, state):
    result = CoordinationPreview(); result.set_trajectory(reference)
    result.times = np.array([0., 8.]); result.controls = np.zeros((2, 8)); result.states = np.tile(state, (2, 1))
    return result


def test_async_rejects_expired_wrong_generation_and_diverged_state():
    reference, state = composite()
    live = AsyncCoordinationPreview(); live.set_trajectory(reference)
    result = fake_result(reference, state)
    live.pending_generation = live.generation; live.pending_t = 0.
    assert live.accept(result, state, .2)
    assert not live.accept(result, state, .61)
    shifted = state.copy(); shifted[0] += .081
    assert not live.accept(result, shifted, .2)
    live.pending_generation -= 1
    assert not live.accept(result, state, .2)


def test_async_pending_job_never_blocks_and_old_guide_expires():
    reference, state = composite()
    live = AsyncCoordinationPreview(); live.set_trajectory(reference)
    live.pending = Future()  # deliberately never completed
    live.requested = 0.
    live.controls = np.ones((2, 8)); live.times = np.array([0., 8.]); live.updated = 0.
    value = live.update(state, np.zeros(8), state[3:], .2, .02, .002, matched_limits())
    np.testing.assert_allclose(value, np.ones(8))
    assert live.update(state, np.zeros(8), state[3:], .61, .02, .002, matched_limits()) is None
    assert not live.pending.done()


def test_existing_single_motion_scenarios_keep_original_allocation():
    for folder in (ROOT/'scenarios').iterdir():
        if not (folder/'task.json').is_file():
            continue
        runtime = json.loads((folder/'runtime_scene.json').read_text())
        reference = OriginalReference(folder/'task.json', runtime['subject']['position'])
        preview = CoordinationPreview(); preview.set_trajectory(reference)
        assert not preview.enabled, folder.name


def test_fast_deadline_survives_missing_async_preview():
    reference, state = composite()
    controller = make_wbc(Limits(), scene(), reference)
    assert controller.limits.deadline_s == reference.duration
    controller.coordination.update = lambda *args, **kwargs: None
    controller.previous = np.ones(8)*.01
    controller.set_time(7.98)
    u, _ = controller.solve(state, *reference.sample(7.98), .02, base_z=.002,
                            commanded_q=state[3:], trajectory_time=7.98)
    np.testing.assert_allclose(u, np.zeros(8), atol=1e-6)
