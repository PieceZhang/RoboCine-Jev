"""Detection timestamps and occlusion errors in independent demo acceptance."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_dynamic_demos.py'
SPEC = importlib.util.spec_from_file_location('dynamic_demo_analysis', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def truth():
    return {'Cart': (np.array([0., 3.]), np.zeros((2, 3)), np.zeros((2, 3)))}


def snapshot(stamp, predicted, measured, measured_at=0.):
    return dict(stamp=stamp, tracks=[dict(id=28, moving=True, center=predicted,
        measured_center=measured, measurement_stamp=measured_at,
        velocity=[0., .07, 0.], surfaces=4)])


def test_occluded_forecast_mismatch_is_retained_without_inventing_a_detection():
    # The apartment cart's last sparse face was measured within 18 cm of its
    # centre. A 2.4 s extrapolation then exceeded the 30 cm matching threshold.
    history = [snapshot(2.4, [.1285, .2949, 0.], [.1287, .1224, 0.])]
    report = audit.match_moving_tracks(history, truth())
    assert report['unmatched_moving_observations'] == 0
    assert report['predicted_unmatched_moving_observations'] == 1
    assert report['predicted_center_error_max_m'] > .30
    assert report['predicted_unmatched_moving_observation_details'][0]['prediction_age_s'] == 2.4


def test_a_false_measured_target_still_fails_even_when_its_forecast_matches():
    report = audit.match_moving_tracks([snapshot(2., [0., 0., 0.], [1., 0., 0.])], truth())
    assert report['unmatched_moving_observations'] == 1
    assert report['predicted_unmatched_moving_observations'] == 0


def test_cached_detection_is_counted_once_but_each_bad_forecast_is_reported():
    history = [snapshot(t, [.4, 0., 0.], [.1, 0., 0.]) for t in (2.1, 2.2, 2.3, 2.4)]
    report = audit.match_moving_tracks(history, truth())
    assert report['measured_moving_observations'] == 1
    assert report['predicted_unmatched_moving_observations'] == 4


def test_detection_velocity_is_compared_at_measurement_time():
    tracks = {'Cart': (np.array([0., 2.]), np.zeros((2, 3)), np.array([[0., .07, 0.], [0., 0., 0.]]))}
    report = audit.match_moving_tracks([snapshot(2., [0., .14, 0.], [0., 0., 0.])], tracks)
    assert report['matched_moving_observations_by_obstacle'] == {'Cart': 1}
    assert report['matched_velocity_error_rms_m_s'] == pytest.approx(0.)


@pytest.mark.parametrize('speed,passes', [(1., True), (.8, False)])
def test_motion_timing_audit_rejects_an_action_stretched_in_physical_time(speed, passes):
    rows = [dict(time_s=str(t), wall_elapsed_s=str(t/speed), path_rate=str(speed), dt_s='.02')
            for t in np.linspace(0., 2., 101)]
    shots = [dict(type='pan', start_s=0., end_s=2., motion_start_s=.2, motion_end_s=1.8)]
    result = audit.audit_motion_timing(rows, shots)
    assert result['motion_timing_pass'] is passes
    assert result['motion_timing'][0]['physical_duration_s'] == pytest.approx(1.6/speed)
