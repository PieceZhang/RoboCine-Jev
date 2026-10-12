"""Retime layer tests — each class aligned 1:1 with the Kotlin motion-math
suite (TrapezoidalVelocityProfileTest / ControlLimitsTest / CoordinateTransformsTest,
tolerance 1e-12), plus the retime_trajectory semantics (caps provably satisfied,
path shape preserved, exact-integrator regression)."""

import math

import numpy as np
import pytest

from cinema_traj.core.retime import (
    MINI3_DEMO_CAPS, BodyVelocityFrd, ControlCommand, ControlLimits,
    DroneCapabilities, TrapezoidalVelocityProfile, body_frd_to_world_ned,
    check_caps, integrate_body_frd_horizontal, look_vec_ned, look_vec_world,
    ned_yaw_to_world_yaw, retime_trajectory, world_ned_to_body_frd,
    world_yaw_to_ned_yaw, world_yup_to_ned)

TOL = 1e-12


def profile(duration, velocity, acceleration):
    return TrapezoidalVelocityProfile.create(duration, velocity, acceleration)


# --------------------------------------------------------------------------- #
# TrapezoidalVelocityProfile — Kotlin TrapezoidalVelocityProfileTest
# --------------------------------------------------------------------------- #
class TestTrapezoidalProfile:
    def test_long_profile_has_all_three_phases(self):
        p = profile(6.0, 1.0, 0.5)
        assert p.ramp_duration == pytest.approx(2.0, abs=TOL)
        assert p.cruise_duration == pytest.approx(2.0, abs=TOL)
        assert p.total_distance == pytest.approx(4.0, abs=TOL)
        s = p.sample(1.0)
        assert s.position == pytest.approx(0.25, abs=TOL)
        assert s.velocity == pytest.approx(0.5, abs=TOL)
        s = p.sample(3.0)
        assert s.position == pytest.approx(2.0, abs=TOL)
        assert s.velocity == pytest.approx(1.0, abs=TOL)
        s = p.sample(6.0)
        assert s.position == pytest.approx(4.0, abs=TOL)
        assert s.velocity == pytest.approx(0.0, abs=TOL)

    def test_short_profile_degrades_to_triangle(self):
        p = profile(1.0, 1.0, 1.0)
        assert p.peak_velocity == pytest.approx(0.5, abs=TOL)
        assert p.cruise_duration == pytest.approx(0.0, abs=TOL)
        assert p.total_distance == pytest.approx(0.25, abs=TOL)

    def test_negative_profile_mirrors_positive(self):
        pos, neg = profile(6.0, 1.0, 0.5), profile(6.0, -1.0, 0.5)
        for t in (0.0, 1.0, 3.0, 5.0, 6.0):
            a, b = pos.sample(t), neg.sample(t)
            assert b.position == pytest.approx(-a.position, abs=TOL)
            assert b.velocity == pytest.approx(-a.velocity, abs=TOL)

    def test_all_samples_inside_limits(self):
        p = profile(3.0, 1.0, 0.5)
        for i in range(301):
            s = p.sample(i / 100.0)
            assert abs(s.velocity) <= 1.0 + TOL
            assert abs(s.acceleration) <= 0.5 + TOL
            assert math.isfinite(s.position)

    def test_zero_and_tiny_profiles_remain_finite(self):
        zero = profile(0.0, 1.0, 0.5)
        tiny = profile(1e-9, 1.0, 0.5)
        for s in (zero.sample(0.0), tiny.sample(1e-9)):
            assert math.isfinite(s.position) and math.isfinite(s.velocity)

    def test_sample_clamps_past_end(self):
        p = profile(6.0, 1.0, 0.5)
        s = p.sample(100.0)
        assert s.elapsed == pytest.approx(6.0, abs=TOL)
        assert s.position == pytest.approx(p.total_distance, abs=TOL)
        assert s.velocity == 0.0 and s.acceleration == 0.0

    def test_non_positive_acceleration_rejected(self):
        with pytest.raises(ValueError):
            profile(1.0, 1.0, 0.0)


# --------------------------------------------------------------------------- #
# ControlLimits — Kotlin ControlLimitsTest (Mini3DemoProfile limits)
# --------------------------------------------------------------------------- #
def velocity(forward=0.0, right=0.0, down=0.0):
    return BodyVelocityFrd(forward, right, down)


class TestControlLimits:
    def test_horizontal_clamp_preserves_direction(self):
        r = ControlLimits.clamp_velocity(velocity(3.0, 4.0, 0.0), MINI3_DEMO_CAPS)
        assert r.forward == pytest.approx(0.6, abs=TOL)
        assert r.right == pytest.approx(0.8, abs=TOL)
        assert math.hypot(r.forward, r.right) == pytest.approx(1.0, abs=TOL)

    def test_ascent_descent_clamped_independently(self):
        descent = ControlLimits.clamp_velocity(velocity(down=2.0), MINI3_DEMO_CAPS)
        ascent = ControlLimits.clamp_velocity(velocity(down=-2.0), MINI3_DEMO_CAPS)
        assert descent.down == pytest.approx(0.5, abs=TOL)
        assert ascent.down == pytest.approx(-0.5, abs=TOL)

    def test_one_20hz_step_obeyes_acceleration(self):
        r = ControlLimits.limit_velocity_step(
            velocity(), velocity(forward=1.0, down=-0.5), MINI3_DEMO_CAPS, 0.05)
        assert r.forward == pytest.approx(0.025, abs=TOL)     # 0.5 m/s² × 0.05 s
        assert r.down == pytest.approx(-0.015, abs=TOL)       # 0.3 m/s² × 0.05 s

    def test_zero_elapsed_cannot_change_velocity(self):
        prev = velocity(0.2, -0.1, 0.05)
        r = ControlLimits.limit_velocity_step(
            prev, velocity(1.0, 1.0, -0.5), MINI3_DEMO_CAPS, 0.0)
        assert r == prev

    def test_yaw_gimbal_rates_clamped(self):
        r = ControlLimits.limit_command(
            ControlCommand.zero(),
            ControlCommand(velocity(), yaw_rate=10.0, gimbal_pitch_rate=-10.0),
            MINI3_DEMO_CAPS, 0.05)
        assert r.yaw_rate == pytest.approx(MINI3_DEMO_CAPS.max_yaw_rate, abs=TOL)
        assert r.gimbal_pitch_rate == pytest.approx(
            -MINI3_DEMO_CAPS.max_gimbal_pitch_rate, abs=TOL)
        assert math.isfinite(r.velocity.forward)

    def test_caps_validation_rejects_bad_values(self):
        with pytest.raises(ValueError):
            DroneCapabilities(0.0, 0.5, 0.5, 0.5, 0.3, 0.3, -1.5, 1.0, 0.3)
        with pytest.raises(ValueError):
            DroneCapabilities(1.0, 0.5, 0.5, 0.5, 0.3, 0.3, 1.0, -1.0, 0.3)


# --------------------------------------------------------------------------- #
# FRD / NED transforms — Kotlin CoordinateTransformsTest + cinema_traj adapters
# --------------------------------------------------------------------------- #
class TestCoordinateTransforms:
    def test_world_frd_roundtrip(self):
        v = BodyVelocityFrd(0.7, -0.2, 0.1)
        for yaw in (0.0, 0.7, -2.3, 5.0):
            n, e, d = body_frd_to_world_ned(v, yaw)
            w = world_ned_to_body_frd(n, e, d, yaw)
            assert (w.forward, w.right, w.down) == \
                pytest.approx((v.forward, v.right, v.down), abs=TOL)

    def test_zero_yaw_forward_is_north(self):
        v = world_ned_to_body_frd(1.0, 0.0, 0.2, 0.0)
        assert v.forward == pytest.approx(1.0, abs=TOL)
        assert v.right == pytest.approx(0.0, abs=TOL)
        assert v.down == pytest.approx(0.2, abs=TOL)

    def test_exact_integration_matches_zero_yaw_rate(self):
        v = BodyVelocityFrd(0.8, 0.1, 0.0)
        dn, de = integrate_body_frd_horizontal(v, 0.3, 0.0, 2.0)
        wn, we, _ = body_frd_to_world_ned(v, 0.3)
        assert dn == pytest.approx(2.0 * wn, abs=TOL)
        assert de == pytest.approx(2.0 * we, abs=TOL)

    def test_exact_integration_beats_euler_when_yawing(self):
        # quarter turn at 1 rad/s while flying forward: quarter-circle arc,
        # net displacement (sin T, 1 − cos T) = (1, 1) — Euler would overshoot
        v = BodyVelocityFrd(1.0, 0.0, 0.0)
        dn, de = integrate_body_frd_horizontal(v, 0.0, 1.0, math.pi / 2)
        assert dn == pytest.approx(1.0, abs=TOL)
        assert de == pytest.approx(1.0, abs=TOL)

    def test_ned_mapping_of_world_yup(self):
        assert world_yup_to_ned((1.0, 2.0, 3.0)) == (1.0, 3.0, -2.0)

    def test_yaw_conversion_pinned_by_look_vectors(self):
        # known world heading: yaw_w=0 looks +Z ⇒ NED yaw 90 (east); +30° etc.
        for yaw_w in (0.0, 30.0, 90.0, 180.0, -120.0):
            yaw_n = world_yaw_to_ned_yaw(yaw_w)
            back = ned_yaw_to_world_yaw(yaw_n)
            assert abs((back - yaw_w + 180.0) % 360.0 - 180.0) < 1e-9  # wrap-equal
            lw = look_vec_world(yaw_w, 0.0)
            ln = look_vec_ned(yaw_n, 0.0)
            # same physical direction expressed in the two frames
            np.testing.assert_allclose([ln[0], -ln[2], ln[1]], lw, atol=1e-12)
        assert world_yaw_to_ned_yaw(0.0) == pytest.approx(90.0)

    def test_look_vec_roundtrip_through_pitch(self):
        for yaw_w in (-170.0, -45.0, 0.0, 60.0, 135.0):
            for pitch in (-80.0, -20.0, 0.0, 45.0):
                lw = look_vec_world(yaw_w, pitch)
                yaw_back = math.degrees(math.atan2(lw[0], lw[2]))
                pitch_back = math.degrees(math.asin(lw[1]))
                assert yaw_back == pytest.approx(yaw_w, abs=1e-9)
                assert pitch_back == pytest.approx(pitch, abs=1e-9)

    def test_ned_look_matches_se3_look_at(self):
        # dual check against the authoritative export math: look_at_R → yaw/pitch
        from cinema_traj.common.se3 import look_at_R, rotation_to_yaw_pitch
        rng = np.random.default_rng(7)
        for _ in range(50):
            eye = rng.uniform(-2, 2, 3) + [0, 1.5, 0]
            target = rng.uniform(-2, 2, 3) + [0, 1.2, 0]
            R = look_at_R(eye, target)
            yaw_w, pitch = rotation_to_yaw_pitch(R)
            f_n = look_vec_ned(world_yaw_to_ned_yaw(yaw_w), pitch)
            np.testing.assert_allclose(
                [f_n[0], -f_n[2], f_n[1]], (target - eye) / np.linalg.norm(target - eye),
                atol=1e-9)


# --------------------------------------------------------------------------- #
# retime_trajectory semantics
# --------------------------------------------------------------------------- #
def _line_samples(n=40, speed=5.0, dt=0.1, yaw_step_deg=5.0):
    """Straight +X line at constant `speed` (m/s), yaw sweeping at yaw_step/dt."""
    return [{"t": i * dt, "x": i * dt * speed, "y": 1.0, "z": 0.0,
             "yaw_deg": yaw_step_deg * i, "pitch_deg": -10.0 + 0.2 * i,
             "roll_deg": 0.0, "fov_deg": 55.0, "segment": "seg"} for i in range(n)]


class TestRetimeTrajectory:
    def test_overspeed_input_gets_stretched_and_compliant(self):
        samples = _line_samples(speed=5.0)                    # 5 m/s ≫ 1 m/s cap
        res = retime_trajectory(samples, MINI3_DEMO_CAPS)
        assert res.duration > samples[-1]["t"]                # stretched
        assert res.scale > 1.0
        assert check_caps(res.samples, res.commands, MINI3_DEMO_CAPS) == []

    def test_compliant_input_keeps_duration(self):
        # 0.2 m/s, 10°/s yaw, 2°/s pitch — everything inside the Mini3 caps
        samples = _line_samples(speed=0.2, yaw_step_deg=1.0)
        res = retime_trajectory(samples, MINI3_DEMO_CAPS)
        assert res.duration == pytest.approx(samples[-1]["t"], abs=1e-6)
        assert res.scale == pytest.approx(1.0, abs=1e-6)

    def test_path_geometry_preserved(self):
        samples = _line_samples(speed=5.0)
        res = retime_trajectory(samples, MINI3_DEMO_CAPS)
        # every retimed position lies on the original piecewise-linear path
        # (here: the straight +X line, y=1, z=0)
        for s in res.samples:
            assert s["y"] == pytest.approx(1.0, abs=1e-9)
            assert s["z"] == pytest.approx(0.0, abs=1e-9)
            assert 0.0 <= s["x"] <= samples[-1]["x"] + 1e-9
        # endpoints preserved
        assert res.samples[0]["x"] == pytest.approx(0.0, abs=1e-9)
        assert res.samples[-1]["x"] == pytest.approx(samples[-1]["x"], abs=1e-6)

    def test_zero_velocity_at_both_ends(self):
        res = retime_trajectory(_line_samples(speed=3.0), MINI3_DEMO_CAPS)
        first, last = res.commands[0], res.commands[-1]
        assert math.hypot(first["forward"], first["right"]) == pytest.approx(0.0, abs=0.15)
        assert last["forward"] == 0.0 and last["down"] == 0.0

    def test_yaw_rate_cap_enforced(self):
        # violent 180° yaw flip in one 0.1 s step
        samples = [{"t": i * 0.1, "x": 0.0, "y": 1.0, "z": 0.0,
                    "yaw_deg": 0.0 if i < 10 else 180.0,
                    "pitch_deg": 0.0, "roll_deg": 0.0, "fov_deg": 55.0,
                    "segment": "pan"} for i in range(20)]
        res = retime_trajectory(samples, MINI3_DEMO_CAPS)
        assert check_caps(res.samples, res.commands, MINI3_DEMO_CAPS) == []
        assert res.duration > samples[-1]["t"]

    def test_custom_caps_respected(self):
        caps = DroneCapabilities(0.4, 0.2, 0.2, 0.25, 0.15, math.radians(45.0),
                                 -math.pi / 2, math.pi / 3, math.radians(30.0))
        res = retime_trajectory(_line_samples(speed=1.0), caps)
        assert check_caps(res.samples, res.commands, caps) == []

    def test_short_input_handled(self):
        samples = _line_samples(n=2, speed=1.0)
        res = retime_trajectory(samples, MINI3_DEMO_CAPS)
        assert len(res.samples) >= 2
        with pytest.raises(ValueError):
            retime_trajectory(samples[:1])

    def test_static_trajectory_zero_commands(self):
        samples = [{"t": i * 0.5, "x": 0.0, "y": 1.0, "z": 0.0, "yaw_deg": 10.0 * i,
                    "pitch_deg": 0.0, "roll_deg": 0.0, "fov_deg": 55.0,
                    "segment": "pan"} for i in range(5)]
        res = retime_trajectory(samples, MINI3_DEMO_CAPS)
        for c in res.commands:
            assert c["down"] == 0.0

    def test_zoh_commands_reproduce_path_exactly(self):
        """Exact-integrator regression (Kotlin MotionTrajectorySimulator math):
        holding each ZOH body command over its interval reproduces the sampled
        NED positions to high accuracy."""
        rng = np.random.default_rng(3)
        samples = [{"t": i * 0.1,
                    "x": float(rng.uniform(-1, 1)), "y": 1.0 + 0.3 * math.sin(0.2 * i),
                    "z": float(rng.uniform(-1, 1)),
                    "yaw_deg": 30.0 * math.sin(0.1 * i), "pitch_deg": -20.0 + 5.0 * i / 40,
                    "roll_deg": 0.0, "fov_deg": 55.0, "segment": "wander"}
                   for i in range(40)]
        res = retime_trajectory(samples, MINI3_DEMO_CAPS)
        n0, e0, d0 = world_yup_to_ned(
            (res.samples[0]["x"], res.samples[0]["y"], res.samples[0]["z"]))
        yaw_n = math.radians(world_yaw_to_ned_yaw(res.samples[0]["yaw_deg"]))
        for k, c in enumerate(res.commands[:-1]):             # last is terminal zero
            dt = res.samples[k + 1]["t"] - res.samples[k]["t"]
            v = BodyVelocityFrd(c["forward"], c["right"], c["down"])
            yaw_rate = math.radians(c["yaw_rate_degps"])      # already NED convention
            dn, de = integrate_body_frd_horizontal(v, yaw_n, yaw_rate, dt)
            n0 += dn
            e0 += de
            d0 += c["down"] * dt
            yaw_n += yaw_rate * dt
            x, y, z = res.samples[k + 1]["x"], res.samples[k + 1]["y"], res.samples[k + 1]["z"]
            # 5e-4 m budget: output yaw/pitch are rounded to 0.001°, the test
            # re-derives yaw from those rounded samples ⇒ small allowed drift
            assert (n0, e0, d0) == pytest.approx(world_yup_to_ned((x, y, z)), abs=5e-4)
