"""Private Go2 input feedback: original cap/domain and clock/state regressions."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


feedback = module('go2_world_z_test', ROOT/'Robot_Go2/world_z_feedback.py')
backend = module('go2_world_z_backend', ROOT/'Robot_Go2/wbc_backend.py')
legacy = module('go2_world_z_legacy', ROOT/'Robot_Go2/camera_coordinator.py')


class Inside:
    def check(self, native):
        return dict(status='inside_outer_domain_unverified', command=np.asarray(native).tolist())


def pose(position=(.5, 0., .5), rotvec=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return result


def step(controller, index, error=(0., 0., .01), **kwargs):
    reference = pose()
    measured = reference.copy()
    measured[:3, 3] -= error
    stamp = (index+1)*.02
    options = dict(task_id='task', stamp_s=stamp, now_s=stamp, dt_s=.02,
                   sample_seq=index, domain=Inside())
    options.update(kwargs)
    return controller.feedback_native_command(backend, reference, measured,
        np.eye(4), np.eye(4), **options)


def test_disabled_exactly_matches_original_p_for_nontrivial_se3_and_never_mutates_inputs():
    rng = np.random.default_rng(731)
    controller = feedback.WorldZFeedback(enabled=False)
    for _ in range(80):
        reference = pose(rng.uniform(-.5, .8, 3), rng.normal(size=3)*.3)
        measured = pose(reference[:3, 3]+rng.normal(size=3)*.1, rng.normal(size=3)*.3)
        link = pose(rng.normal(size=3)*.2, rng.normal(size=3)*.15)
        extrinsic = pose(rng.normal(size=3)*.1, rng.normal(size=3)*.3)
        originals = [a.copy() for a in (reference, measured, link, extrinsic)]
        expected, expected_check = legacy.feedback_native_command(
            backend, reference, measured, link, extrinsic, domain=Inside())
        actual, actual_check = controller.feedback_native_command(
            backend, reference, measured, link, extrinsic, domain=Inside())
        np.testing.assert_array_equal(actual, expected)
        actual_check.pop('world_z_feedback')
        assert actual_check == expected_check
        assert controller.integral_m == 0.
        for actual_input, original in zip((reference, measured, link, extrinsic), originals):
            np.testing.assert_array_equal(actual_input, original)


def test_disabled_preserves_real_domain_rejection_and_original_fallback():
    reference = pose((.405, 0., .5))
    measured = reference.copy(); measured[0, 3] += .04
    link = pose((0., 0., .35))
    expected, before = legacy.feedback_native_command(backend, reference, measured, link, np.eye(4))
    actual, after = feedback.WorldZFeedback().feedback_native_command(
        backend, reference, measured, link, np.eye(4))
    after.pop('world_z_feedback')
    np.testing.assert_array_equal(actual, expected)
    assert before == after and after['status'] == 'inside_outer_domain_unverified'
    assert after['rejected_feedback_correction']['domain_check']['status'] == 'out_of_policy_domain'


def test_integral_is_world_z_even_with_rotated_base_and_camera_extrinsic():
    controller = feedback.WorldZFeedback(enabled=True)
    reference = pose((.5, .1, .6), (.2, -.1, .4))
    measured = reference.copy(); measured[2, 3] -= .01
    link = pose((.1, -.2, .3), (.3, -.2, .7))
    extrinsic = pose((.2, .04, .11), (-.2, .4, -.1))
    native, check = controller.feedback_native_command(backend, reference, measured, link, extrinsic,
        task_id='world', stamp_s=.02, now_s=.02, sample_seq=0, domain=Inside())
    compiled = backend.mixed_to_world_camera(native, link, extrinsic)
    np.testing.assert_allclose(compiled[:3, 3]-reference[:3, 3], [0., 0., .00354], atol=1e-15)
    np.testing.assert_allclose(compiled[:3, :3], reference[:3, :3], atol=1e-15)
    assert check['world_z_feedback']['integral_after_m'] == pytest.approx(.00004)


def test_constant_bias_bounded_by_shared_total_cap_then_opposite_error_unwinds():
    controller = feedback.WorldZFeedback(enabled=True)
    reports = []
    for index in range(1000):
        _, check = step(controller, index)
        reports.append(check['world_z_feedback'])
    assert 0.021 < controller.integral_m < .025
    assert reports[-1]['cap_antiwindup_frozen']
    assert all(r['proposed_position_correction_norm_m'] <= .025+1e-14 for r in reports)
    old = controller.integral_m
    for index in range(1000, 1700):
        step(controller, index, error=(0., 0., -.01))
    assert controller.integral_m < 0. and controller.integral_m < old


def test_xy_p_and_z_i_share_one_norm_cap_without_extra_per_axis_allowance():
    controller = feedback.WorldZFeedback(enabled=True)
    for index in range(500):
        _, check = step(controller, index, error=(.04, .04, .03))
        report = check['world_z_feedback']
        correction = np.asarray(report['proposed_position_correction_m'])
        assert np.linalg.norm(correction) <= .025+1e-14
    assert report['cap_antiwindup_frozen']
    assert controller.integral_m < .025


def test_p_already_saturated_cannot_wind_up_same_direction_integral():
    controller = feedback.WorldZFeedback(enabled=True)
    for index in range(100):
        _, check = step(controller, index, error=(0., 0., .1))
    assert controller.integral_m == 0.
    assert check['world_z_feedback']['cap_antiwindup_frozen']


def test_integration_veto_retains_true_fresh_p_feedback_and_existing_integral():
    controller = feedback.WorldZFeedback(enabled=True)
    step(controller, 0)
    before = controller.integral_m
    native, check = step(controller, 1, integration_allowed=False)
    report = check['world_z_feedback']
    assert report['measurement_fresh_and_clock_valid']
    assert not report['integral_updated'] and controller.integral_m == before
    np.testing.assert_allclose(native[:3], [.5, 0., .5+.0035+before])


def test_domain_rejection_uses_original_reference_and_rolls_back_current_integral():
    class RejectCorrection:
        def check(self, command):
            return dict(status='out_of_policy_domain' if command[2] > .5 else 'inside_outer_domain_unverified')
    controller = feedback.WorldZFeedback(enabled=True)
    step(controller, 0)
    before = controller.integral_m
    native, check = step(controller, 1, domain=RejectCorrection())
    np.testing.assert_array_equal(native, [.5, 0., .5, 1., 0., 0., 0.])
    assert controller.integral_m == before
    assert check['status'] == 'inside_outer_domain_unverified'
    assert check['rejected_feedback_correction']['domain_check']['status'] == 'out_of_policy_domain'
    assert check['world_z_feedback']['domain_correction_rejected']
    step(controller, 2)
    assert controller.integral_m == pytest.approx(before+.00004)


def test_fallback_outside_real_domain_is_never_reported_as_inside():
    class Outside:
        def check(self, command):
            return dict(status='out_of_policy_domain')
    native, check = step(feedback.WorldZFeedback(enabled=True), 0, domain=Outside())
    assert check['status'] == 'out_of_policy_domain'
    np.testing.assert_array_equal(native, [.5, 0., .5, 1., 0., 0., 0.])


def test_missing_measurement_freezes_then_reanchors_without_integrating_missing_gap():
    controller = feedback.WorldZFeedback(enabled=True)
    step(controller, 0)
    before = controller.integral_m
    native, check = controller.feedback_native_command(backend, pose(), None, np.eye(4), np.eye(4),
        task_id='task', stamp_s=None, now_s=.04, fresh=False, domain=Inside())
    np.testing.assert_array_equal(native, [.5, 0., .5, 1., 0., 0., 0.])
    assert controller.integral_m == before
    assert check['world_z_feedback']['clock_status'] == 'missing_fresh_measurement'
    _, recovered = step(controller, 8)
    assert controller.integral_m == before
    assert recovered['world_z_feedback']['integration_dt_s'] == 0.
    step(controller, 9)
    assert controller.integral_m == pytest.approx(before+.00004)


@pytest.mark.parametrize('override,reason', [
    (dict(stamp_s=.02, now_s=.04), 'nonmonotone_measurement_stamp'),
    (dict(stamp_s=.01, now_s=.04), 'nonmonotone_measurement_stamp'),
    (dict(stamp_s=.04, now_s=.03), 'future_or_stale_measurement'),
    (dict(stamp_s=.04, now_s=.2), 'future_or_stale_measurement'),
    (dict(stamp_s=float('nan')), 'invalid_measurement_time_or_dt'),
    (dict(dt_s=.2), 'invalid_measurement_time_or_dt'),
    (dict(dt_s=0.), 'invalid_measurement_time_or_dt'),
    (dict(dt_s=.01), 'measurement_gap_or_dt_mismatch'),
    (dict(sample_seq=0), 'nonmonotone_sample_sequence'),
    (dict(sample_seq=True), 'invalid_sample_sequence'),
])
def test_invalid_time_and_sequence_cannot_update_integral_or_apply_stale_p(override, reason):
    controller = feedback.WorldZFeedback(enabled=True)
    step(controller, 0)
    before = controller.integral_m
    native, check = step(controller, 1, **override)
    assert controller.integral_m == before
    assert check['world_z_feedback']['clock_status'] == reason
    json.dumps(check, allow_nan=False)
    np.testing.assert_array_equal(native, [.5, 0., .5, 1., 0., 0., 0.])


def test_new_task_clears_integral_but_cannot_reset_absolute_clock_or_sequence():
    controller = feedback.WorldZFeedback(enabled=True)
    for index in range(10):
        step(controller, index)
    _, check = step(controller, 10, task_id='new', error=(0., 0., 0.))
    assert controller.integral_m == 0. and controller.task_id == 'new'
    assert check['world_z_feedback']['integral_before_m'] == 0.
    step(controller, 11, task_id='new')
    before = controller.integral_m
    step(controller, 12, task_id='old_late', stamp_s=.1, now_s=.26)
    assert controller.task_id == 'new' and controller.integral_m == before
    controller.reset('third')
    _, check = step(controller, 12, task_id='third', sample_seq=2)
    assert check['world_z_feedback']['clock_status'] == 'nonmonotone_sample_sequence'
    assert controller.integral_m == 0.


def test_external_override_rollback_is_once_only_token_bound_and_consumes_time():
    controller = feedback.WorldZFeedback(enabled=True)
    _, first = step(controller, 0)
    token = first['world_z_feedback']['update_id']
    assert controller.rollback_last_update('outer_domain_recovery', update_id=token)
    assert controller.integral_m == 0.
    assert first['world_z_feedback']['externally_rolled_back']
    assert not controller.rollback_last_update('duplicate', update_id=token)
    _, second = step(controller, 1)
    assert controller.integral_m == pytest.approx(.00004)
    assert not controller.rollback_last_update('late_old_request', update_id=token)
    assert controller.integral_m == pytest.approx(.00004)
    assert controller.rollback_last_update('parking_hold', update_id=second['world_z_feedback']['update_id'])
    assert controller.integral_m == 0.
    _, duplicate = step(controller, 1)
    assert not duplicate['world_z_feedback']['measurement_fresh_and_clock_valid']


def test_reset_invalidates_previous_task_rollback_token():
    controller = feedback.WorldZFeedback(enabled=True)
    _, check = step(controller, 0)
    controller.reset('new')
    assert not controller.rollback_last_update('late', update_id=check['world_z_feedback']['update_id'])


def test_orientation_cap_is_the_original_fifty_milliradians():
    controller = feedback.WorldZFeedback(enabled=True)
    reference = pose()
    measured = pose(rotvec=(1., .4, -.2))
    native, check = controller.feedback_native_command(backend, reference, measured, np.eye(4), np.eye(4),
        task_id='rot', stamp_s=.02, now_s=.02, domain=Inside())
    compiled = backend.mixed_to_world_camera(native, np.eye(4), np.eye(4))
    assert Rotation.from_matrix(compiled[:3, :3]).magnitude() == pytest.approx(.05)
    assert np.linalg.norm(check['world_z_feedback']['proposed_orientation_correction_rad']) == pytest.approx(.05)
