"""The jitter audit detects fast shake without flagging smooth pi crossings."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jvs_mock.camera_stability import motion_residual


def test_smooth_large_turn_crossing_pi_has_no_false_quaternion_spike():
    t = np.arange(0., 8., .017)
    yaw = np.deg2rad(160.) + .25*t
    grid, dp, dr = motion_residual(t, np.c_[.1*t, 0*t, 0*t], Rotation.from_euler('z', yaw))
    interior = (grid > .5) & (grid < 7.4)
    assert np.max(np.rad2deg(np.linalg.norm(dr[interior], axis=1))) < .01
    assert np.max(np.linalg.norm(dp[interior], axis=1)) < 1e-6


def test_detects_roll_shake_even_when_forward_direction_does_not_change():
    t = np.arange(0., 8., .001)
    roll = np.deg2rad(.4)*np.sin(2*np.pi*9*t)
    grid, _, dr = motion_residual(t, np.zeros((len(t), 3)), Rotation.from_euler('x', roll))
    angle = np.rad2deg(np.linalg.norm(dr[(grid > .5) & (grid < 7.4)], axis=1))
    assert np.sqrt(np.mean(angle**2)) > .25
    assert np.max(angle) > .35


def test_irregular_physical_samples_keep_the_same_linear_motion():
    t = np.cumsum(np.tile([.016, .017, .033], 150))
    grid, dp, dr = motion_residual(t, np.c_[.03*t, -.04*t, 0*t], Rotation.from_euler('y', .02*t))
    keep = (grid > t[0]+.5) & (grid < t[-1]-.5)
    assert np.max(np.linalg.norm(dp[keep], axis=1)) < 1e-6
    assert np.max(np.linalg.norm(dr[keep], axis=1)) < 1e-5


@pytest.mark.parametrize('offset', [0., -.01])
def test_rejects_repeated_or_reversed_physical_time(offset):
    t = np.arange(100.)*.02
    t[50] = t[49]+offset
    with pytest.raises(ValueError, match='strictly increasing'):
        motion_residual(t, np.zeros((len(t), 3)), Rotation.identity(len(t)))


@pytest.mark.parametrize('shake,expected', [(False, True), (True, False)])
def test_physical_audit_uses_camera_body_measurements_at_all_three_rates(monkeypatch, tmp_path,
                                                                        shake, expected):
    from jvs_mock import camera_stability as audit
    for name in ('camera_physics.csv', 'telemetry.csv'):
        (tmp_path / name).write_text('mock measurements')
    physical_time = np.arange(-.1, 5., .001)
    roll = np.deg2rad(.4)*np.sin(2*np.pi*9*physical_time) if shake else 0*physical_time
    monkeypatch.setattr(audit, 'load_physics', lambda _: (
        physical_time, np.zeros((len(physical_time), 3)), Rotation.from_euler('x', roll)))
    control_time = np.arange(0., 5., .02)
    rows = [dict(command_omega=.3, transition_active=int(i%2==0), motion_active=int(i%2==1))
            for i in range(len(control_time))]
    # Opposite ROS/FK estimates cannot determine the physical-camera result.
    telemetry_roll = np.zeros(len(control_time)) if shake else .4*np.sin(18*np.pi*control_time)
    monkeypatch.setattr(audit, 'load_telemetry', lambda _: (
        control_time, np.zeros((len(control_time), 3)),
        Rotation.from_euler('x', telemetry_roll), rows))
    report = audit.summarize_physics(tmp_path)
    assert report['camera_stability_pass'] == expected
    assert set(report['frequencies']) == {'30', '60', '1000'}
    assert all(item['pass_'] == expected for item in report['frequencies'].values())
    assert not report['image_stabilization']


def test_slow_closed_loop_wobble_cannot_pass_a_quiet_high_frequency_audit(monkeypatch, tmp_path):
    from jvs_mock import camera_stability as audit
    for name in ('camera_physics.csv', 'telemetry.csv'):
        (tmp_path/name).write_text('slow oscillation regression fixture')
    time = np.arange(-.1, 20., .001)
    rotation = Rotation.from_euler('x', np.deg2rad(4.)*np.sin(2*np.pi*.9*time))
    monkeypatch.setattr(audit, 'load_physics', lambda _: (time, np.zeros((len(time), 3)), rotation))
    control = np.arange(0., 20., .02)
    rows = [dict(command_omega=.3, transition_active=1, motion_active=0) for _ in control]
    monkeypatch.setattr(audit, 'load_telemetry', lambda _: (
        control, np.zeros((len(control), 3)), Rotation.identity(len(control)), rows))
    report = audit.summarize_physics(tmp_path)
    assert all(item['fast_shake_pass'] for item in report['frequencies'].values())
    assert all(not item['slow_wobble_pass'] for item in report['frequencies'].values())
    assert not report['camera_stability_pass']
