import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.virtual_stabilizer import VirtualStabilizer, StabilizerConfig, se3_log, se3_exp


def test_se3_coupled_translation_roundtrip():
    for a in (1e-9, .2, 2.8):
        p = np.array([.2, -.4, .1])
        r = Rotation.from_rotvec(np.array([1., 2., -1.])*a/np.sqrt(6)).as_matrix()
        p2, r2 = se3_exp(se3_log(p, r))
        np.testing.assert_allclose(p2, p, atol=1e-12)
        np.testing.assert_allclose(r2, r, atol=1e-12)


def test_perfect_fast_orbit_and_constant_tracking_bias_are_preserved():
    f = VirtualStabilizer()
    bias = np.array([.03, -.02, .01, .01, .02, -.03])
    dp, dr = se3_exp(bias)
    for t in np.arange(0., 2., .002):
        rp = np.array([np.cos(4*t), np.sin(4*t), .5*t])
        rr = Rotation.from_euler('z', 4*t).as_matrix()
        ap, ar = rp+rr@dp, rr@dr
        p, r, _ = f.update(t, rp, rr, ap, ar)
        np.testing.assert_allclose(p, ap, atol=1e-12)
        np.testing.assert_allclose(r, ar, atol=1e-12)


def test_high_frequency_attenuation_and_bounded_correction():
    f = VirtualStabilizer()
    raw, smooth = [], []
    for t in np.arange(0., 4., .001):
        e = .01*np.sin(2*np.pi*10*t)
        p, r, info = f.update(t, np.zeros(3), np.eye(3), np.array([e, 0., 0.]),
                              Rotation.from_euler('z', e).as_matrix())
        if t > 1.:
            raw.append(e); smooth.append(p[0])
        assert info['translation_correction_m'] <= .12+1e-12
        assert info['rotation_correction_deg'] <= 8.+1e-12
    assert np.std(smooth) < .02*np.std(raw)
    p, r, info = f.update(4., np.zeros(3), np.eye(3), np.ones(3), Rotation.from_euler('z', 1.).as_matrix())
    assert np.linalg.norm(p-np.ones(3)) <= .12+1e-12
    assert np.linalg.norm(Rotation.from_matrix(Rotation.from_euler('z', 1.).as_matrix().T@r).as_rotvec()) <= np.deg2rad(8)+1e-12


def test_clock_reset_reinitializes_to_actual_pose():
    f = VirtualStabilizer()
    for t in (0., .01, .02, 0.):
        p, r, info = f.update(t, np.zeros(3), np.eye(3), np.array([.1*t, 0., 0.]), np.eye(3))
    assert info['reset']
    np.testing.assert_allclose(p, 0.)


def test_slow_error_passes_after_settling():
    f = VirtualStabilizer()
    raw, smooth = [], []
    for t in np.arange(0., 30., .01):
        e = .01*np.sin(2*np.pi*.05*t)
        p, _, _ = f.update(t, np.zeros(3), np.eye(3), np.array([e, 0., 0.]), np.eye(3))
        if t > 10.:
            raw.append(e); smooth.append(p[0])
    assert .95 < np.std(smooth)/np.std(raw) < 1.05


def test_diagnostics_are_json_serializable_on_every_update():
    import json
    f = VirtualStabilizer()
    for t in (0., .001, .002):
        _, _, info = f.update(t, np.zeros(3), np.eye(3), np.ones(3)*t, np.eye(3))
        json.dumps(info, allow_nan=False)
