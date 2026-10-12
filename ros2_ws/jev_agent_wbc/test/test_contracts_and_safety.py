import copy
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.contracts import Envelope, PARAMETERS, validate_task
from jev_agent_wbc.reference import OriginalReference, DeformedReference
from jev_agent_wbc.evaluation import visual_errors, observation_angle, smoothness
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.whole_body_qp import WholeBodyQP, Limits
from jev_agent_wbc.jev import AsyncJev, SemanticDecider


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def original():
    scene = json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    return OriginalReference(ROOT/'scenarios/movein/task.json', scene['subject']['position'])


def envelope(original, **kwargs):
    defaults = dict(reference_sha256=original.sha256, decision_id='test', event='difficulty', issued_at_s=0.,
        start_s=.2, end_s=8., blend_s=1., bounds={'radial_m': [-.2, .3]}, protected=('fixed_progress',), terminal='retain')
    defaults.update(kwargs)
    return Envelope(**defaults)


def test_task_has_no_robot_scene_or_visual_tolerance(original):
    for field in ('visual_tolerance', 'robot', 'obstacle_trajectory'):
        task = original.task
        task[field] = {}
        with pytest.raises(ValueError):
            validate_task(task)
    task = original.task
    task['reference_camera_trajectory']['samples'][2]['position'][0] = float('nan')
    with pytest.raises(ValueError):
        validate_task(task)


def test_task_copy_cannot_rewrite_original(original):
    p = original.sample(3.)[0]
    task = original.task
    task['reference_camera_trajectory']['samples'][15]['position'][0] += 10.
    np.testing.assert_allclose(original.sample(3.)[0], p)


def test_envelope_rejects_nonfinite_inverted_time_wrong_reference(original):
    caps = dict.fromkeys(PARAMETERS, 1.)
    for e in (envelope(original, end_s=9.), envelope(original, reference_sha256='wrong'),
              envelope(original, bounds={'radial_m': [0., float('nan')]}),
              envelope(original, bounds={'height_m': [1., -.5]})):
        with pytest.raises(ValueError):
            e.validate(original.sha256, original.duration, caps)


def test_optimizer_cannot_exceed_permission(original):
    with pytest.raises(ValueError):
        DeformedReference(original, envelope(original), [.31, 0., 0., 0., 0., 0.])


def test_optimizer_checks_authorized_boundary_when_small_variants_fail(original, monkeypatch):
    from jev_agent_wbc import planning
    from jev_agent_wbc.contracts import Validation
    permission = envelope(original, bounds={'radial_m': [0., .3]})
    # Equal coarse scores for retain/recover occur when their difference lies
    # beyond the ranking horizon. Full-body feasibility may require a larger
    # permitted displacement and an in-budget return.
    def rank(path, *args, **kwargs):
        return Validation(True, 'ranking', cost=float(np.linalg.norm(path.values)))
    checked = []
    def validate(path, *args, **kwargs):
        checked.append((float(path.values[0]), path.return_s))
        feasible = path.values[0] > .29 and path.return_s is not None
        return Validation(feasible, 'full_budget', cost=1.)
    monkeypatch.setattr(planning, 'rank_candidate', rank)
    monkeypatch.setattr(planning, 'validate_rollout', validate)
    state = np.array(json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())['initial_state'])
    result = planning.optimize_envelope(original, original, permission, '', {}, state,
                                       np.zeros(8), 0., .002, Limits())
    assert 'path' in result, checked
    assert result['path'].values[0] == pytest.approx(.3)
    assert result['path'].return_s is not None


def test_easier_modified_target_cannot_outweigh_original_reference_loss(original, monkeypatch):
    from jev_agent_wbc import planning
    from jev_agent_wbc.contracts import Validation
    permission = envelope(original, bounds={'radial_m': [0., .3]})
    monkeypatch.setattr(planning, 'rank_candidate',
        lambda *a, **k: Validation(True, 'ranking', cost=0.))
    def validate(path, *args, **kwargs):
        changed = np.linalg.norm(path.values) > 1e-8
        # A changed target is easier to track, but its executed trajectory is
        # farther from the original. All candidates satisfy physical checks.
        return Validation(True, 'verified', cost=1. if changed else 100.,
                          original_objective=2. if changed else 1.)
    monkeypatch.setattr(planning, 'validate_rollout', validate)
    state = np.array(json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())['initial_state'])
    result = planning.optimize_envelope(original, original, permission, '', {}, state,
                                       np.zeros(8), 0., .002, Limits())
    np.testing.assert_array_equal(result['path'].values, np.zeros(6))


def test_repeated_relaxation_remains_relative_to_original_and_pickles(original):
    one = DeformedReference(original, envelope(original), [.2, 0., 0., 0., 0., 0.])
    two = DeformedReference(original, envelope(original, start_s=2., decision_id='second'),
                            [.2, 0., 0., 0., 0., 0.], previous=one)
    np.testing.assert_allclose(np.linalg.norm(two.sample(5.)[0]-original.sample(5.)[0]), .2, atol=1e-6)
    restored = pickle.loads(pickle.dumps(two))
    np.testing.assert_allclose(restored.sample(5.)[0], two.sample(5.)[0])


def test_c2_activation_and_recovery_do_not_add_time(original):
    modified = DeformedReference(original, envelope(original, terminal='recover'), [.2, 0., 0., 0., 0., 0.], activation_s=.5)
    assert modified.duration == original.duration
    for t in (.5, 8.):
        np.testing.assert_allclose(modified.sample(t)[0], original.sample(t)[0], atol=1e-9)
        for h in (1e-3, 5e-4):
            change = modified.parameters(t+h)-modified.parameters(t-h)
            assert np.linalg.norm(change/(2*h)) < 1e-5


def test_moving_obstacle_sweep_detects_collision_between_clear_endpoints():
    scene = dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[0., -.65, .13]]),
                 surfel_radius=.03, surfel_velocities=np.array([[0., 1.3, 0.]]),
                 surfel_track_ids=np.array([1]), observation_stamp=0., minimum_clearance_m=.025)
    collision = DynamicCollision(scene, uncertainty_rate=0.)
    assert collision.clearance(np.zeros(9)).min() > .1
    assert collision.at_time(1.).clearance(np.zeros(9)).min() > .1
    assert collision.sweep(np.zeros(9), np.zeros(8), 1.)['upper'].min() < 0.


def test_velocity_acceleration_and_nonholonomic_wheel_bounds():
    controller = WholeBodyQP(Limits())
    from jev_agent_wbc.arm_comfort import COMFORT_POSE
    from jev_agent_wbc.whole_body_qp import whole_body_kinematics
    state = np.r_[np.zeros(3), COMFORT_POSE]
    p, r, _ = whole_body_kinematics(state)
    u, result = controller.solve(state, p+[10., 10., 10.], r, np.zeros(6), .02, commanded_q=state[3:])
    assert np.all(np.abs(u) <= .02*controller.limits.acceleration+1e-6)
    assert np.all(np.abs(controller.limits.wheels@u) <= .02*controller.limits.wheel_acceleration+1e-6)
    assert result['command_diff_drive_violation'] == 0.


def test_invalid_masks_are_not_zero_error_and_scale_is_relative_area():
    ref = np.zeros((100, 100), bool)
    ref[40:60, 40:60] = True
    actual = np.zeros_like(ref)
    actual[40:60, 50:70] = True
    value = visual_errors(ref, actual)
    assert value['composition_error'] == pytest.approx(.1)
    assert value['subject_scale_error'] == 0.
    assert visual_errors(ref, np.zeros_like(ref))['composition_error'] is None
    assert not visual_errors(None, actual)['valid']


def test_subject_coordinates_and_missing_orientation():
    subject = dict(valid=True, orientation_valid=True, position=[0., 0., 0.], rotation=np.eye(3))
    assert observation_angle([1, 0, 0], [0, 1, 0], subject) == pytest.approx(np.pi/2)
    turned = {**subject, 'rotation': Rotation.from_euler('z', np.pi/2).as_matrix()}
    assert observation_angle([1, 0, 0], [0, 1, 0], subject, turned) == pytest.approx(0., abs=1e-7)
    assert observation_angle([1, 0, 0], [0, 1, 0], {**subject, 'orientation_valid': False}) is None


def test_smoothness_uses_physical_time_and_rotation_manifold():
    t = np.linspace(0., 1., 101)
    p = np.column_stack([t**3, 0*t, 0*t])
    q = Rotation.from_euler('z', .2*t**2).as_quat()
    value = smoothness(t, p, q)
    assert value['translation_jerk_rms_m_s3'] == pytest.approx(6., abs=1e-8)
    assert value['angular_acceleration_rms_rad_s2'] == pytest.approx(.4, abs=1e-8)


def test_async_jev_does_not_wait_for_slow_model():
    class Slow:
        def decide(self, evidence):
            time.sleep(.15)
            raise RuntimeError('offline')
    worker = AsyncJev(Slow())
    started = time.monotonic()
    assert worker.submit({})
    assert worker.poll() is None
    assert time.monotonic()-started < .05
    assert not worker.submit({})
    time.sleep(.17)
    assert 'offline' in worker.poll()['error']
    worker.close()


def test_no_cross_package_import_dependency():
    import ast
    for folder in ('jev_agent_wbc', 'scripts', 'launch'):
        for path in (ROOT/folder).glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').startswith('jvs_mock'), str(path)
                if isinstance(node, ast.Import):
                    assert not any(alias.name.startswith('jvs_mock') for alias in node.names), str(path)


def test_settled_bounds_do_not_hide_invalid_transition_union(original):
    e = envelope(original, bounds={'radial_m': [-10., 10.]}, settled_bounds={'radial_m': [0., .1]})
    with pytest.raises(ValueError):
        e.validate(original.sha256, original.duration, dict.fromkeys(PARAMETERS, 1.))


def test_recovery_transition_preserves_pose_before_activation(original):
    from dataclasses import replace
    one = DeformedReference(original, envelope(original, protected=('subject_aim',)), [.2, 0., 0., 0., 0., 0.])
    narrowed = envelope(original, start_s=2., protected=(), bounds={'radial_m': [0., .3]},
                        settled_bounds={'radial_m': [0., .05]})
    two = DeformedReference(original, narrowed, [.05, 0., 0., 0., 0., 0.], one, 2.2)
    for t in (1., 2.2):
        for a, b in zip(one.sample(t)[:2], two.sample(t)[:2]):
            np.testing.assert_allclose(a, b, atol=1e-10)
    assert two.parameters(4.)[0] == pytest.approx(.05)


def test_ramp_can_stop_inside_interval_without_false_infeasibility():
    from jev_agent_wbc.jvs_kinematics import JOINT_UPPER, JOINT_LOWER
    limits = Limits(command_ramp=True, position_margin=.03)
    controller = WholeBodyQP(limits)
    q = .5*(JOINT_UPPER+JOINT_LOWER)
    q[4] = JOINT_UPPER[4]-.03-.000015
    controller.previous[6] = .00257
    _, low, high = controller.bounds(q, .02, q)
    assert low[6] <= high[6] < 0.
    v0, v1 = controller.previous[6], high[6]
    maximum_travel = .02*v0*v0/(2*(v0-v1))
    assert maximum_travel <= .000015+1e-12


def test_numeric_stopping_residual_does_not_create_large_false_bound():
    from jev_agent_wbc.jvs_kinematics import JOINT_UPPER, JOINT_LOWER
    controller = WholeBodyQP(Limits(command_ramp=True, position_margin=.03))
    q = .5*(JOINT_UPPER+JOINT_LOWER)
    q[5] = JOINT_LOWER[5]+.03
    controller.previous[7] = -2e-7
    _, low, high = controller.bounds(q, .02, q)
    assert np.all(low <= high)


def test_command_reserve_roundoff_does_not_relax_physical_joint_boundary():
    from jev_agent_wbc.jvs_kinematics import JOINT_UPPER, JOINT_LOWER
    controller = WholeBodyQP(Limits(command_ramp=True, position_margin=.03,
                                   acceleration=np.ones(8)))
    measured = (JOINT_UPPER+JOINT_LOWER)/2
    target = measured.copy()
    velocity = .003125804174836811
    target[5] = JOINT_LOWER[5]+.03+.5*velocity**2-2e-9
    controller.previous[7] = -velocity
    _, low, high = controller.bounds(measured, .02, target)
    assert np.all(low <= high)
    assert high[7] == pytest.approx(-velocity+.02)
    measured[5] = JOINT_LOWER[5]+.5*velocity**2-2e-9
    _, low, high = controller.bounds(measured, .02, target)
    assert low[7] > high[7]


def test_exact_permission_numbers_survive_prompt_compaction():
    from jev_agent_wbc.jev import compact_evidence
    plan = {'bounds': {'radial_m': [0., .0234375]}, 'start_s': .12345678}
    output = compact_evidence({'complete_plan_to_authorize': plan, 'measurement': .12345678})
    assert output['complete_plan_to_authorize'] == plan
    assert output['measurement'] == .1235


def test_numeric_bound_effects_use_runtime_subject_geometry():
    from jev_agent_wbc.reference import bound_effect_estimate
    evidence = {'runtime_subject_position': [0., 0., 0.],
                'reference_samples': [{'position': [-2., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.]}]}
    effect = bound_effect_estimate(evidence, 'radial_m', .2)
    np.testing.assert_allclose(effect['inverse_square_area_ratio_range'], [4/2.2**2]*2)
    assert effect['max_view_angle_change_rad'] == pytest.approx(0.)


def test_moving_proxy_keeps_radius_in_prediction_and_static_filter():
    scene = dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[3., 0., .2], [0., .8, .2]]),
        surfel_radius=.03, surfel_radii=np.array([.03, .2]), surfel_velocities=np.array([[0., 0., 0.], [0., -.1, 0.]]),
        surfel_track_ids=np.array([0, 1]), observation_stamp=0., minimum_clearance_m=.025)
    collision = DynamicCollision(scene, uncertainty_rate=0.)
    np.testing.assert_allclose(collision.at_time(1.).radii, [.03, .2])
    np.testing.assert_allclose(collision.static_collision().radii, [.03])


def test_direct_fallback_never_certifies_current_penetration():
    from jev_agent_wbc.safety import certified_escape
    from jev_agent_wbc.arm_comfort import COMFORT_POSE
    from jev_agent_wbc.whole_body_qp import whole_body_kinematics
    state = np.r_[np.zeros(3), COMFORT_POSE]
    controller = WholeBodyQP()
    controller.collision = DynamicCollision(dict(format='jvs_lidar_surfels/v1',
        surfel_centers=np.array([[0., 0., .13]]), surfel_radius=.10,
        observation_stamp=0., minimum_clearance_m=.025))
    p, r, _ = whole_body_kinematics(state)
    with pytest.raises(RuntimeError, match='passes direct dynamic sweep'):
        certified_escape(controller, state, p, r, .02, commanded_q=state[3:])


def test_jev_actual_protocol_builds_numeric_permission_without_text_json(original):
    class Client:
        def ask(self, state, questions):
            reply = {}
            for key, question in questions.items():
                if question['type'] == 'choice':
                    value = {'direction': 'radial_m_positive', 'complement': 'none',
                             'window': 'immediate', 'terminal': 'recover'}[key]
                    reply[key] = {'type': 'choice', 'choice': value}
                else:
                    reply[key] = {'type': 'noul', 'noul': .99}
            return reply
    evidence = dict(time_s=1., duration_s=8., reference_sha256=original.sha256, event='difficulty',
        revision=4, physical_caps=dict.fromkeys(PARAMETERS, 1.), active_envelope=None,
        directions=[dict(id='radial_m_positive', parameter='radial_m', sign=1, unit='m', physical_cap=.8)],
        time_windows={'immediate': dict(start_s=1.1, end_s=8., blend_s=1.)})
    permission, recovery = SemanticDecider(Client(), search_rounds=3).decide(evidence)
    assert permission.bounds['radial_m'] == pytest.approx([0., .7])
    assert permission.terminal == 'recover'
    assert permission.end_s == original.duration
    assert 'fixed_progress' in permission.protected


def test_fast_sweep_is_lower_bound_of_dense_all_surface_reference():
    from jev_agent_wbc.arm_comfort import COMFORT_POSE
    from jev_agent_wbc.kc_svsdf import advance
    rng = np.random.default_rng(731)
    for _ in range(4):
        centers = rng.uniform([-2., -2., .05], [2., 2., 1.5], (35, 3))
        velocities = np.zeros_like(centers)
        velocities[:3] = rng.uniform(-.4, .4, (3, 3))
        scene = dict(format='jvs_lidar_surfels/v1', surfel_centers=centers, surfel_radius=.025,
            surfel_radii=rng.uniform(.02, .08, len(centers)), surfel_velocities=velocities,
            surfel_track_ids=np.r_[np.ones(3, int), np.zeros(32, int)], observation_stamp=0., minimum_clearance_m=.025)
        collision = DynamicCollision(scene, uncertainty_rate=.035)
        state = np.r_[0., 0., .3, COMFORT_POSE]
        command = rng.uniform(-.15, .15, 8)
        result = collision.sweep(state, command, .08, max_samples=9)
        dense = np.array([collision.clearance_to(advance(state, command, t),
            centers+velocities*t, scene['surfel_radii']+collision.radius_rates*t).min(axis=0)
            for t in np.linspace(0., .08, 101)]).min(axis=0)
        assert np.all(result['lower'].min(axis=0) <= dense+1e-9)


def test_compact_eight_ball_model_contains_observed_box_and_surfel_extent():
    extent = np.array([.07, .11, .09])
    half = extent/2
    centers = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*half
    points = np.random.default_rng(12).uniform(-extent, extent, (500, 3))
    distance = np.linalg.norm(points[:, None]-centers[None], axis=2).min(axis=1)
    assert np.max(distance)+.035*np.sqrt(3)/2 < np.linalg.norm(half)+1.5*.035


def test_original_deadline_reserves_deceleration_and_forces_zero_terminal_command():
    from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER
    controller = WholeBodyQP(Limits(deadline_s=8.))
    q = (JOINT_LOWER+JOINT_UPPER)/2
    controller.time_s = 7.98
    controller.previous = .5*.02*controller.limits.acceleration
    _, lower, upper = controller.bounds(q, .02, q)
    np.testing.assert_allclose(lower, 0., atol=1e-12)
    np.testing.assert_allclose(upper, 0., atol=1e-12)


def test_motion_direction_protection_includes_activation(original):
    from jev_agent_wbc.planning import permission_violation
    permitted = envelope(original, protected=('motion_direction',))
    modified = DeformedReference(original, permitted, [.2, 0., 0., 0., 0., 0.])
    assert permission_violation(modified, 0., 8.) == 'protected_motion_direction_reversed'
    assert permission_violation(modified, 2., 6.) is None


def test_retain_recovery_reauthorizes_old_bounds_without_new_numeric_search(original):
    class Client:
        def __init__(self): self.queries = []
        def ask(self, state, questions):
            self.queries.append(set(questions))
            return {k: ({'type': 'choice', 'choice': {'recovery': 'retain', 'window': 'immediate',
                    'terminal': 'retain'}[k]} if q['type'] == 'choice' else
                    {'type': 'noul', 'noul': .99}) for k, q in questions.items()}
    client = Client()
    old = envelope(original).to_dict()
    evidence = dict(time_s=3., duration_s=8., reference_sha256=original.sha256, event='recovery',
        revision=4, physical_caps=dict.fromkeys(PARAMETERS, 1.), active_envelope=old,
        directions=[dict(id='radial_m_positive', parameter='radial_m', sign=1, unit='m', physical_cap=.8)],
        time_windows={'immediate': dict(start_s=3.1, end_s=8., blend_s=1.)})
    permission, recovery = SemanticDecider(client).decide(evidence)
    assert recovery == 'retain'
    assert permission.settled_bounds['radial_m'] == old['bounds']['radial_m']
    assert len(client.queries) == 3


def test_original_angular_feedforward_matches_world_rotation_derivative(original):
    task = original.task
    for row in task['reference_camera_trajectory']['samples']:
        t = row['t']
        row['quaternion_xyzw'] = Rotation.from_euler('ZYX', [.1*t, .03*t*t, .2*np.sin(t)]).as_quat().tolist()
    reference = OriginalReference(task, original.subject)
    t, h = 3.2, 1e-5
    _, r, ff = reference.sample(t)
    ra, rb = reference.sample(t-h)[1], reference.sample(t+h)[1]
    numerical = Rotation.from_matrix(rb@ra.T).as_rotvec()/(2*h)
    np.testing.assert_allclose(ff[3:], numerical, atol=1e-7)


def test_preroll_is_observed_moving_history_continuous_at_shot_start():
    from jev_agent_wbc.scenarios import RuntimeScene
    scene = RuntimeScene(ROOT/'scenarios/pedestal_up/runtime_scene.json')
    a, b = scene.obstacles(-.8)[0], scene.obstacles(0.)[0]
    np.testing.assert_allclose(a['center']+.8*a['velocity'], b['center'])
    np.testing.assert_allclose(a['velocity'], b['velocity'])
    assert np.linalg.norm(a['velocity']) > 0.


def test_refined_ramp_certificate_bounds_dense_ramp_with_tilted_base():
    from jev_agent_wbc.actuation import certify_control_ramp
    from jev_agent_wbc.arm_comfort import COMFORT_POSE
    from jev_agent_wbc.kc_svsdf import advance
    state = np.r_[0., 0., .3, COMFORT_POSE]
    rotation = Rotation.from_euler('xyz', [.03, -.02, .3]).as_matrix()
    scene = dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[0., .8, .3]]),
        surfel_radius=.04, surfel_velocities=np.array([[0., -.15, 0.]]),
        surfel_track_ids=np.array([1]), observation_stamp=0., minimum_clearance_m=.025)
    old = np.array([.1, .2, .1, -.1, .05, .1, -.03, .04])
    new = old+np.array([.01, -.03, .02, .02, -.01, .01, .02, -.01])
    bound = certify_control_ramp(scene, state, old, new, 0., .08, steps=8, base_rotation=rotation)
    robot = state.copy()
    collision = DynamicCollision(scene)
    minimum = np.inf
    h = .0002
    for i in range(401):
        t = i*h
        collision.set_time(t)
        current_rotation = Rotation.from_euler('z', robot[2]-state[2]).as_matrix()@rotation
        minimum = min(minimum, collision.clearance(robot, base_rotation=current_rotation).min())
        if i < 400:
            robot = advance(robot, old+(t+.5*h)/.08*(new-old), h)
    assert bound <= minimum+1e-9
