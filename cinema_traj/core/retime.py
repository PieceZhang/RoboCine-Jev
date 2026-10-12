"""Retime layer — execution math ported from the third-party Kotlin package
``trajectory-simulator-0.1.0-SNAPSHOT`` (modules motion-math / motion-domain).

cinema_traj plans WHERE the camera goes (geometry, occlusion, collision) but
only WARNS when the exported setpoints exceed speed limits. This module adds
the missing execution layer: re-time the geometric path so the result provably
satisfies drone velocity / acceleration / rate caps. The path SHAPE is never
clipped — only the time axis stretches.

Ported 1:1 from Kotlin (same formulas, same edge cases, tests aligned at 1e-12):

  * ``TrapezoidalVelocityProfile.create/sample``  ← motion-math/…/TrapezoidalVelocityProfile.kt
  * ``ControlLimits.*``                           ← motion-math/…/ControlLimits.kt
  * ``MINI3_DEMO_CAPS``                           ← motion-domain/…/AircraftProfile.kt (Mini3DemoProfile)
  * ``integrate_body_frd_horizontal``             ← motion-math/…/CoordinateTransforms.kt

Coordinate conventions (pinned by tests):

  * cinema_traj world frame is Y-up: yaw 0 → +Z, positive toward +X, pitch up.
  * NED (Kotlin / REP-103 world): north=+X(world), east=+Z(world), down=−Y(world);
    yaw_ned = 90° − yaw_world (0 → north, positive clockwise from above);
    gimbal pitch positive UP in both conventions (Mini3 range −90°…+60°).
  * FRD body velocity: forward/right/down; down = +descent, −ascent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Platform contract — what the shared retime loop needs from a platform
# --------------------------------------------------------------------------- #
class PlatformCaps:
    """Contract the shared retime loop imposes on a platform's capability
    object.

    The loop owns the platform-independent execution math: arc-length
    reparameterization, forward/backward tangential-acceleration passes,
    yaw/pitch slew limiting (fields ``max_yaw_rate``, ``max_gimbal_pitch_rate``,
    ``min/max_gimbal_pitch``, ``command_period``), and the stretch-until-provable
    iteration. Everything platform-specific is delegated to four methods:

      * ``tangential_accel()``          — scalar cap for the v² = v₀² + 2a·s passes
      * ``vertex_speed_limits(dirn, dt)`` — per-vertex speed cap (N vertices
        from N−1 unit segment directions; rest at both ends)
      * ``interval_factor(dp, dts, yaw_l, pitch_l, dt)`` — worst stretch factor
        over the retimed intervals (≥1 means "stretch more")
      * ``build_commands(ts, dp, dts, yaw_l, pitch_l)`` — per-interval command
        stream artifact (platform schema)

    Reference implementations: :class:`DroneCapabilities` (Kotlin 1:1 port)
    and ``JarvisCapabilities`` (cinema_traj.core.jarvis, mobile manipulator
    with Jacobian-projected caps)."""

    command_period: float = 0.05
    max_yaw_rate: float = 0.0
    max_gimbal_pitch_rate: float = 0.0
    min_gimbal_pitch: float = 0.0
    max_gimbal_pitch: float = 0.0

    def tangential_accel(self) -> float:
        raise NotImplementedError

    def vertex_speed_limits(self, dirn: np.ndarray, dt_out: float) -> np.ndarray:
        raise NotImplementedError

    def interval_factor(self, dp: np.ndarray, dts: np.ndarray,
                        yaw_l: np.ndarray, pitch_l: np.ndarray,
                        dt_out: float) -> float:
        raise NotImplementedError

    def build_commands(self, ts: np.ndarray, dp: np.ndarray, dts: np.ndarray,
                       yaw_l: np.ndarray, pitch_l: np.ndarray) -> list:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Value types (mirror of Kotlin domain classes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScalarMotionSample:
    elapsed: float
    position: float
    velocity: float
    acceleration: float


@dataclass(frozen=True)
class BodyVelocityFrd:
    forward: float
    right: float
    down: float


@dataclass(frozen=True)
class ControlCommand:
    velocity: BodyVelocityFrd
    yaw_rate: float                    # rad/s
    gimbal_pitch_rate: float           # rad/s

    @classmethod
    def zero(cls) -> "ControlCommand":
        return cls(BodyVelocityFrd(0.0, 0.0, 0.0), 0.0, 0.0)


@dataclass(frozen=True)
class DroneCapabilities(PlatformCaps):
    max_horizontal_speed: float        # m/s
    max_ascent_speed: float            # m/s (positive magnitude)
    max_descent_speed: float           # m/s (positive magnitude)
    max_horizontal_acceleration: float # m/s²
    max_vertical_acceleration: float   # m/s²
    max_yaw_rate: float                # rad/s
    min_gimbal_pitch: float            # rad
    max_gimbal_pitch: float            # rad
    max_gimbal_pitch_rate: float       # rad/s
    command_period: float = 0.05       # s (20 Hz ZOH command stream)

    def __post_init__(self):
        positives = [self.max_horizontal_speed, self.max_ascent_speed,
                     self.max_descent_speed, self.max_horizontal_acceleration,
                     self.max_vertical_acceleration, self.max_yaw_rate,
                     self.max_gimbal_pitch_rate, self.command_period]
        if any(v <= 0.0 for v in positives):
            raise ValueError("all speed/acceleration/rate capabilities must be > 0")
        if self.min_gimbal_pitch >= self.max_gimbal_pitch:
            raise ValueError("min_gimbal_pitch must be less than max_gimbal_pitch")

    # -- PlatformCaps: drone-specific execution math (1:1 Kotlin port) ----- #
    def tangential_accel(self) -> float:
        return min(self.max_horizontal_acceleration, self.max_vertical_acceleration)

    def vertex_speed_limits(self, dirn: np.ndarray, dt_out: float) -> np.ndarray:
        """Per-vertex caps: horizontal corner slowdown + horizontal/vertical
        component limits (drone kinematics — direction-agnostic lift)."""
        N = dirn.shape[0] + 1
        v_lim = np.full(N, np.inf)
        for i in range(1, N - 1):
            a, b = dirn[i - 1], dirn[i]
            ha, hb = math.hypot(a[0], a[2]), math.hypot(b[0], b[2])
            if ha > _EPS and hb > _EPS:                          # horizontal corner
                d_h = math.hypot(a[0] / ha - b[0] / hb, a[2] / ha - b[2] / hb)
                v_lim[i] = min(v_lim[i],
                               self.max_horizontal_acceleration * dt_out / max(d_h, _EPS))
            v_lim[i] = min(v_lim[i],
                           self.max_vertical_acceleration * dt_out / max(abs(b[1] - a[1]), _EPS),
                           self.max_horizontal_speed / max(ha, hb, _EPS),
                           min(self.max_ascent_speed, self.max_descent_speed) / max(abs(a[1]), abs(b[1]), _EPS))
        v_lim[0] = v_lim[-1] = 0.0                               # rest at both ends
        return v_lim

    def interval_factor(self, dp: np.ndarray, dts: np.ndarray,
                        yaw_l: np.ndarray, pitch_l: np.ndarray,
                        dt_out: float) -> float:
        """Worst stretch factor over intervals: speed / acceleration / ZOH
        command-step checks against the drone caps."""
        h_speed = np.hypot(dp[:, 0], dp[:, 2]) / dts
        vy = dp[:, 1] / dts
        factor = max(
            float((h_speed / self.max_horizontal_speed).max(initial=0.0)),
            float((vy / self.max_descent_speed).max(initial=0.0)),
            float((-vy / self.max_ascent_speed).max(initial=0.0)))
        v_world = dp / dts[:, None]
        dt_mid = (dts[:-1] + dts[1:]) / 2.0
        acc = np.diff(v_world, axis=0) / dt_mid[:, None]
        factor = max(factor,
                     math.sqrt(float((np.hypot(acc[:, 0], acc[:, 2]) /
                                      self.max_horizontal_acceleration).max(initial=0.0))),
                     math.sqrt(float((np.abs(acc[:, 1]) /
                                      self.max_vertical_acceleration).max(initial=0.0))))
        # body-frame ZOH command steps (Kotlin limitVelocityStep semantics —
        # includes the yaw-rotation shear between consecutive FRD vectors)
        yaw_n = np.radians(np.asarray([world_yaw_to_ned_yaw(math.degrees(y))
                                       for y in yaw_l[:-1]]))
        fwd = (np.cos(yaw_n) * dp[:, 0] / dts + np.sin(yaw_n) * dp[:, 2] / dts)
        rgt = (-np.sin(yaw_n) * dp[:, 0] / dts + np.cos(yaw_n) * dp[:, 2] / dts)
        dwn = -dp[:, 1] / dts
        dv_h = np.hypot(np.diff(fwd, prepend=0.0), np.diff(rgt, prepend=0.0))
        dv_v = np.abs(np.diff(dwn, prepend=0.0))
        factor = max(factor,
                     float((dv_h / (self.max_horizontal_acceleration * dt_out)).max(initial=0.0)),
                     float((dv_v / (self.max_vertical_acceleration * dt_out)).max(initial=0.0)))
        return float(factor)

    def build_commands(self, ts: np.ndarray, dp: np.ndarray, dts: np.ndarray,
                       yaw_l: np.ndarray, pitch_l: np.ndarray) -> list:
        """ZOH body commands: world Δp over [t_k, t_k+1] → FRD at yaw(t_k)."""
        yaw_n = np.radians(np.asarray([world_yaw_to_ned_yaw(math.degrees(y))
                                       for y in yaw_l[:-1]]))
        fwd = (np.cos(yaw_n) * dp[:, 0] / dts + np.sin(yaw_n) * dp[:, 2] / dts)
        rgt = (-np.sin(yaw_n) * dp[:, 0] / dts + np.cos(yaw_n) * dp[:, 2] / dts)
        dwn = -dp[:, 1] / dts
        commands = []
        for k in range(len(ts) - 1):
            dt = dts[k]
            yaw_rate = -math.degrees((yaw_l[k + 1] - yaw_l[k]) / dt)
            pitch_rate = math.degrees((pitch_l[k + 1] - pitch_l[k]) / dt)
            commands.append({
                "t": round(float(ts[k]), 6),
                "forward": float(fwd[k]), "right": float(rgt[k]),
                "down": float(dwn[k]),
                "yaw_rate_degps": yaw_rate,
                "gimbal_pitch_rate_degps": pitch_rate})
        commands.append({"t": round(float(ts[-1]), 6), "forward": 0.0, "right": 0.0,
                         "down": 0.0, "yaw_rate_degps": 0.0,
                         "gimbal_pitch_rate_degps": 0.0})    # terminal zero command
        return commands


#: Conservative DJI Mini 3 filming-demo limits (NOT the aircraft maxima).
MINI3_DEMO_CAPS = DroneCapabilities(
    max_horizontal_speed=1.0, max_ascent_speed=0.5, max_descent_speed=0.5,
    max_horizontal_acceleration=0.5, max_vertical_acceleration=0.3,
    max_yaw_rate=math.radians(20.0),
    min_gimbal_pitch=-math.pi / 2.0, max_gimbal_pitch=math.pi / 3.0,
    max_gimbal_pitch_rate=math.radians(20.0),
    command_period=0.05)


# --------------------------------------------------------------------------- #
# TrapezoidalVelocityProfile — exact port, tests aligned at 1e-12
# --------------------------------------------------------------------------- #
class TrapezoidalVelocityProfile:
    """Scalar position profile over [0, duration]: accelerate at a_max to a
    peak velocity (≤ |v_req|), cruise, decelerate back to zero. Degrades to a
    triangular profile when the duration is too short to reach v_req."""

    def __init__(self, duration: float, requested_velocity: float,
                 max_acceleration: float, ramp_duration: float,
                 cruise_duration: float, peak_velocity: float,
                 total_distance: float):
        self.duration = duration
        self.requested_velocity = requested_velocity
        self.max_acceleration = max_acceleration
        self.ramp_duration = ramp_duration
        self.cruise_duration = cruise_duration
        self.peak_velocity = peak_velocity
        self.total_distance = total_distance

    @classmethod
    def create(cls, duration: float, requested_velocity: float,
               max_acceleration: float) -> "TrapezoidalVelocityProfile":
        if max_acceleration <= 0.0:
            raise ValueError("maxAcceleration must be greater than zero")
        if duration == 0.0 or requested_velocity == 0.0:
            return cls(duration, requested_velocity, max_acceleration,
                       ramp_duration=0.0, cruise_duration=duration,
                       peak_velocity=0.0, total_distance=0.0)
        requested_magnitude = abs(requested_velocity)
        ramp = min(requested_magnitude / max_acceleration, duration / 2.0)
        cruise = duration - 2.0 * ramp
        peak_magnitude = min(requested_magnitude, max_acceleration * ramp)
        peak = math.copysign(peak_magnitude, requested_velocity)
        distance = peak * (ramp + cruise)
        return cls(duration, requested_velocity, max_acceleration,
                   ramp, cruise, peak, distance)

    def sample(self, elapsed: float) -> ScalarMotionSample:
        if self.duration == 0.0 or self.peak_velocity == 0.0:
            return ScalarMotionSample(min(elapsed, self.duration), 0.0, 0.0, 0.0)
        time = min(elapsed, self.duration)
        if time >= self.duration:
            return ScalarMotionSample(time, self.total_distance, 0.0, 0.0)
        direction = math.copysign(1.0, self.peak_velocity)
        a = self.max_acceleration
        ramp, cruise, peak = self.ramp_duration, self.cruise_duration, self.peak_velocity
        accel_distance = 0.5 * direction * a * ramp * ramp
        if time <= ramp:
            pos = 0.5 * direction * a * time * time
            vel = direction * a * time
            acc = direction * a
        elif time <= ramp + cruise:
            pos = accel_distance + peak * (time - ramp)
            vel, acc = peak, 0.0
        else:
            de = time - ramp - cruise
            pos = accel_distance + peak * cruise + peak * de - 0.5 * direction * a * de * de
            vel = peak - direction * a * de
            acc = -direction * a
        return ScalarMotionSample(time, pos, vel, acc)


# --------------------------------------------------------------------------- #
# ControlLimits — exact port, tests aligned at 1e-12
# --------------------------------------------------------------------------- #
class ControlLimits:
    @staticmethod
    def clamp_velocity(requested: BodyVelocityFrd,
                       caps: DroneCapabilities) -> BodyVelocityFrd:
        """Clamp speed preserving the requested horizontal direction."""
        hmag = math.hypot(requested.forward, requested.right)
        scale = (caps.max_horizontal_speed / hmag
                 if hmag > caps.max_horizontal_speed and hmag > 0.0 else 1.0)
        bounded_down = min(max(requested.down, -caps.max_ascent_speed),
                           caps.max_descent_speed)
        return BodyVelocityFrd(requested.forward * scale,
                               requested.right * scale, bounded_down)

    @staticmethod
    def limit_velocity_step(previous: BodyVelocityFrd, requested: BodyVelocityFrd,
                            caps: DroneCapabilities, elapsed: float) -> BodyVelocityFrd:
        """Limit one velocity step by acceleration × elapsed time."""
        prev = ControlLimits.clamp_velocity(previous, caps)
        target = ControlLimits.clamp_velocity(requested, caps)
        max_h_delta = caps.max_horizontal_acceleration * elapsed
        fd = target.forward - prev.forward
        rd = target.right - prev.right
        hd = math.hypot(fd, rd)
        d_scale = max_h_delta / hd if hd > max_h_delta and hd > 0.0 else 1.0
        max_v_delta = caps.max_vertical_acceleration * elapsed
        dd = min(max(target.down - prev.down, -max_v_delta), max_v_delta)
        return BodyVelocityFrd(prev.forward + fd * d_scale,
                               prev.right + rd * d_scale, prev.down + dd)

    @staticmethod
    def clamp_yaw_rate(rate: float, caps: DroneCapabilities) -> float:
        return min(max(rate, -caps.max_yaw_rate), caps.max_yaw_rate)

    @staticmethod
    def clamp_gimbal_pitch_rate(rate: float, caps: DroneCapabilities) -> float:
        return min(max(rate, -caps.max_gimbal_pitch_rate), caps.max_gimbal_pitch_rate)

    @staticmethod
    def limit_command(previous: ControlCommand, requested: ControlCommand,
                      caps: DroneCapabilities, elapsed: float) -> ControlCommand:
        return ControlCommand(
            velocity=ControlLimits.limit_velocity_step(
                previous.velocity, requested.velocity, caps, elapsed),
            yaw_rate=ControlLimits.clamp_yaw_rate(requested.yaw_rate, caps),
            gimbal_pitch_rate=ControlLimits.clamp_gimbal_pitch_rate(
                requested.gimbal_pitch_rate, caps))


# --------------------------------------------------------------------------- #
# NED / FRD coordinate transforms (ports + cinema_traj adapters)
# --------------------------------------------------------------------------- #
def world_ned_to_body_frd(north: float, east: float, down: float,
                          yaw: float) -> BodyVelocityFrd:
    c, s = math.cos(yaw), math.sin(yaw)
    return BodyVelocityFrd(c * north + s * east, -s * north + c * east, down)


def body_frd_to_world_ned(v: BodyVelocityFrd, yaw: float) -> tuple[float, float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v.forward - s * v.right, s * v.forward + c * v.right, v.down)


def integrate_body_frd_horizontal(v: BodyVelocityFrd, yaw_start: float,
                                  yaw_rate: float, duration: float) -> tuple[float, float]:
    """Exact (north, east) displacement while a body velocity is held and yaw
    changes linearly — closed form, no Euler error."""
    if abs(yaw_rate) <= _EPS:
        north, east, _ = body_frd_to_world_ned(v, yaw_start)
        return north * duration, east * duration
    yaw_end = yaw_start + yaw_rate * duration
    north = (v.forward * (math.sin(yaw_end) - math.sin(yaw_start)) +
             v.right * (math.cos(yaw_end) - math.cos(yaw_start))) / yaw_rate
    east = (-v.forward * (math.cos(yaw_end) - math.cos(yaw_start)) +
            v.right * (math.sin(yaw_end) - math.sin(yaw_start))) / yaw_rate
    return north, east


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def world_yup_to_ned(p) -> tuple[float, float, float]:
    """(x, y_up, z) → (north, east, down): north=+X, east=+Z, down=−Y."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return (x, z, -y)


def world_yaw_to_ned_yaw(yaw_world_deg: float) -> float:
    """World yaw (0 → +Z, positive toward +X) → NED yaw (0 → north, positive
    clockwise from above). Heading vector sin/cos swap ⇒ 90° − yaw."""
    return wrap_deg(90.0 - yaw_world_deg)


def ned_yaw_to_world_yaw(yaw_ned_deg: float) -> float:
    return wrap_deg(90.0 - yaw_ned_deg)


def look_vec_world(yaw_world_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit look vector in the Y-up world frame."""
    y, p = math.radians(yaw_world_deg), math.radians(pitch_deg)
    return np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])


def look_vec_ned(yaw_ned_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit look vector in NED (gimbal pitch positive up ⇒ down = −sin)."""
    y, p = math.radians(yaw_ned_deg), math.radians(pitch_deg)
    return np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])


# --------------------------------------------------------------------------- #
# Retiming: arc-length reparameterization under caps
# --------------------------------------------------------------------------- #
@dataclass
class RetimeResult:
    samples: list = field(default_factory=list)    # same schema as export samples
    commands: list = field(default_factory=list)   # ZOH body commands per interval
    duration: float = 0.0
    scale: float = 1.0                             # new duration / original

    @property
    def n_samples(self) -> int:
        return len(self.samples)


def _min_trapezoid_duration(distance: float, v_cap: float, a_cap: float) -> float:
    """Minimal duration for a trapezoid covering ``distance`` within (v_cap, a_cap):
    triangular regime T = 2√(S/a) below the v²/a knee, else S/v + v/a."""
    if distance <= 0.0:
        return 0.0
    if distance <= v_cap * v_cap / a_cap:
        return 2.0 * math.sqrt(distance / a_cap)
    return distance / v_cap + v_cap / a_cap


def check_caps(samples: list, commands: list, caps: DroneCapabilities,
               tol: float = 1e-6, rate_tol: float = 1e-3) -> list:
    """Verify a (samples, commands) pair against caps. Returns violation strings
    (empty ⇒ provably inside limits). Accel/step uses the Kotlin 20 Hz semantics."""
    warns = []
    for k, (a, b) in enumerate(zip(samples, samples[1:])):
        dt = max(b["t"] - a["t"], 1e-9)
        dx, dy, dz = b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"]
        h = math.hypot(dx, dz) / dt
        up, down = -dy / dt, dy / dt
        if h > caps.max_horizontal_speed + tol:
            warns.append(f"t={b['t']:.2f}: horizontal speed {h:.3f} > {caps.max_horizontal_speed}")
        if up > caps.max_ascent_speed + tol:
            warns.append(f"t={b['t']:.2f}: ascent {up:.3f} > {caps.max_ascent_speed}")
        if down > caps.max_descent_speed + tol:
            warns.append(f"t={b['t']:.2f}: descent {down:.3f} > {caps.max_descent_speed}")
        dyaw = abs(wrap_deg(b["yaw_deg"] - a["yaw_deg"])) / dt
        if dyaw > math.degrees(caps.max_yaw_rate) + rate_tol:
            warns.append(f"t={b['t']:.2f}: yaw rate {dyaw:.1f}°/s > "
                         f"{math.degrees(caps.max_yaw_rate):.1f}")
        dpitch = abs(b["pitch_deg"] - a["pitch_deg"]) / dt
        if dpitch > math.degrees(caps.max_gimbal_pitch_rate) + rate_tol:
            warns.append(f"t={b['t']:.2f}: gimbal rate {dpitch:.1f}°/s > "
                         f"{math.degrees(caps.max_gimbal_pitch_rate):.1f}")
    for k, (c0, c1) in enumerate(zip(commands, commands[1:])):
        dt = caps.command_period
        dv = (c1["forward"] - c0["forward"], c1["right"] - c0["right"])
        if math.hypot(*dv) > caps.max_horizontal_acceleration * dt + tol:
            warns.append(f"cmd {k}: horizontal Δv {math.hypot(*dv):.4f} > "
                         f"{caps.max_horizontal_acceleration * dt:.4f}")
        if abs(c1["down"] - c0["down"]) > caps.max_vertical_acceleration * dt + tol:
            warns.append(f"cmd {k}: vertical Δv exceeds limit")
    return warns


def retime_trajectory(samples: list, caps: PlatformCaps = MINI3_DEMO_CAPS,
                      dt_out: float | None = None, keep_duration: bool = True,
                      max_stretch_iters: int = 64,
                      yaw0: float | None = None,
                      pitch0: float | None = None) -> RetimeResult:
    """Re-time an exported trajectory so it provably satisfies ``caps``.

    Shared loop for every platform (see ``PlatformCaps``); the drone formulas
    live in ``DroneCapabilities`` (1:1 Kotlin port), the mobile-manipulator
    Jacobian projection in ``cinema_traj.core.jarvis.JarvisCapabilities``.

    ``yaw0``/``pitch0`` (rad, optional): warm-start the slew limiters with the
    PREVIOUS stream's final heading — per-segment retiming chains them so the
    look-yaw jump at a segment boundary is ramped (while at rest) instead of
    teleporting the base parking spot. Single-call (drone) usage leaves them
    unset → identical behavior.

    The geometric path (positions, yaw, gimbal pitch — piecewise linear between
    input samples) is preserved EXACTLY; only the time law changes:

      * positions follow a forward/backward velocity profile over the arc
        length (classic time-optimal profiling): full speed on straights,
        slowed to ``a·dt/Δdirection`` at corners (the speed at which the 20 Hz
        setpoint stream itself respects the acceleration caps), at rest at
        both ends;
      * yaw / gimbal follow linear ramp-rate profiles (slew-rate limiters)
        sampled on the same arc-length parameter — desired heading JUMPS at
        segment boundaries (look target switches), which no time stretch can
        fix, so the executable track lags the desire by ≤ rate × dt and a
        stationary terminal tail settles the final pose;
      * gimbal pitch saturates into the physical range (−90°…+60° for Mini 3).

    A verification loop re-checks every output interval against the caps and
    stretches all speeds by a common factor on any violation (stretch, never
    clip); failure to converge raises rather than emitting an unsafe stream.

    Returns ``RetimeResult`` with output ``samples`` on a ``dt_out`` grid
    (default: the 20 Hz ZOH command period) plus per-interval body commands.
    """
    if len(samples) < 2:
        raise ValueError("retime_trajectory needs >= 2 samples")
    dt_out = caps.command_period if dt_out is None else float(dt_out)
    P0 = np.asarray([[s["x"], s["y"], s["z"]] for s in samples], dtype=np.float64)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P0, axis=0), axis=1) > 1e-9])
    t0 = max(float(samples[-1]["t"] - samples[0]["t"]), _EPS)
    yaw_cap, pitch_cap = caps.max_yaw_rate, caps.max_gimbal_pitch_rate
    pitch_lo = max(caps.min_gimbal_pitch, -math.pi / 2 + 1e-9)
    pitch_hi = min(caps.max_gimbal_pitch, math.pi / 2 - 1e-9)

    if keep.sum() < 2:
        # degenerate: stationary position (pure pan/tilt) — walk yaw/pitch on a
        # uniform grid, slew-limited; stretch only for terminal catch-up.
        yaw_src = np.unwrap([math.radians(s["yaw_deg"]) for s in samples])
        pitch_src = np.asarray([math.radians(s["pitch_deg"]) for s in samples])
        t_src = np.asarray([s["t"] for s in samples], dtype=np.float64)
        t_src = t_src - t_src[0]
        T = t0
        for _ in range(max_stretch_iters):
            n_steps = max(1, int(math.ceil(T / dt_out - 1e-9)))
            ts = np.asarray([min(k * dt_out, T) for k in range(n_steps + 1)])
            yaw_l = _slew_limit(np.interp(ts, t_src, yaw_src), yaw_cap, np.diff(ts),
                                start=yaw0)
            pitch_l = _slew_limit(np.clip(np.interp(ts, t_src, pitch_src), pitch_lo, pitch_hi),
                                  pitch_cap, np.diff(ts), start=pitch0)
            lag = max(abs(yaw_l[-1] - np.interp(T, t_src, yaw_src)) / yaw_cap,
                      abs(pitch_l[-1] - np.interp(T, t_src, pitch_src)) / pitch_cap)
            if lag <= 0.02:
                break
            T += lag + dt_out
        p0 = P0[0]
        new_samples = [{"t": round(float(t), 6), "x": p0[0], "y": p0[1], "z": p0[2],
                        "yaw_deg": round(math.degrees(ya), 3),
                        "pitch_deg": round(math.degrees(pa), 3), "roll_deg": 0.0,
                        "fov_deg": samples[0].get("fov_deg") or 55.0,
                        "segment": samples[-1].get("segment") or ""}
                       for t, ya, pa in zip(ts, yaw_l, pitch_l)]
        commands = [{"t": round(float(ts[k]), 6), "forward": 0.0, "right": 0.0,
                     "down": 0.0,
                     "yaw_rate_degps": -math.degrees((yaw_l[k + 1] - yaw_l[k]) / dt_out),
                     "gimbal_pitch_rate_degps": math.degrees(
                         (pitch_l[k + 1] - pitch_l[k]) / dt_out)}
                    for k in range(len(ts) - 1)]
        commands.append({"t": round(float(ts[-1]), 6), "forward": 0.0, "right": 0.0,
                         "down": 0.0, "yaw_rate_degps": 0.0, "gimbal_pitch_rate_degps": 0.0})
        return RetimeResult(samples=new_samples, commands=commands,
                            duration=float(ts[-1]), scale=float(ts[-1]) / t0)

    P = P0[keep]
    yaw = np.unwrap([math.radians(s["yaw_deg"]) for s, k in zip(samples, keep) if k])
    pitch = np.asarray([math.radians(s["pitch_deg"]) for s, k in zip(samples, keep) if k])
    seg_labels = np.asarray([s.get("segment") or "" for s, k in zip(samples, keep) if k],
                            dtype=object)

    seg = np.diff(P, axis=0)                                 # (N-1, 3)
    ds = np.linalg.norm(seg, axis=1)
    S = float(ds.sum())
    s_arc = np.concatenate([[0.0], np.cumsum(ds)])
    u_arc = s_arc / S if S > 0 else np.linspace(0.0, 1.0, len(P))
    N = len(P)
    a_t = caps.tangential_accel()

    # -- per-vertex speed limits: platform caps + corners --------------------- #
    dirn = seg / np.maximum(ds[:, None], _EPS)               # unit segment dirs
    v_lim = caps.vertex_speed_limits(dirn, dt_out)

    # -- forward/backward passes with tangential acceleration ------------------ #
    v = v_lim.copy()
    for i in range(N - 1):
        v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2.0 * a_t * ds[i]))
    for i in range(N - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_t * ds[i]))
    v = np.maximum(v, 1e-6)
    v[-1] = 0.0
    # per-segment linear velocity ramp: accel = (v1²−v0²)/(2·ds) ≤ a_t by construction
    dt_seg = 2.0 * ds / np.maximum(v[:-1] + v[1:], 1e-9)     # segment durations at λ=1
    t_seg_end = np.concatenate([[0.0], np.cumsum(dt_seg)])   # vertex times at λ=1
    a_seg = (v[1:] ** 2 - v[:-1] ** 2) / np.maximum(2.0 * ds, _EPS)

    lam = max(1.0, t0 / t_seg_end[-1]) if keep_duration else 1.0  # never compress time

    def _u_at(ts: np.ndarray) -> np.ndarray:
        """Path parameter u at times ts for time scale λ (quadratic per segment)."""
        ends = t_seg_end * lam
        j = np.clip(np.searchsorted(ends, ts, side="right") - 1, 0, len(ds) - 1)
        tau = ts - ends[j]
        s = s_arc[j] + (v[j] / lam) * tau + 0.5 * (a_seg[j] / lam ** 2) * tau * tau
        if S > 0:
            return np.clip(s / S, 0.0, 1.0)
        return np.clip(ts / max(ends[-1], _EPS), 0.0, 1.0)   # degenerate: pure pan

    def _interp(u: np.ndarray):
        """positions / yaw / pitch at path parameter u ∈ [0, 1] (by arc length)."""
        u = np.clip(u, 0.0, 1.0)
        i = np.clip(np.searchsorted(u_arc, u, side="right") - 1, 0, len(P) - 2)
        u0, u1 = u_arc[i], u_arc[i + 1]
        f = np.where(u1 > u0, (u - u0) / np.maximum(u1 - u0, _EPS), 0.0)
        f = np.clip(f, 0.0, 1.0)
        pos = P[i] + f[:, None] * (P[i + 1] - P[i])
        yaw_i = yaw[i] + f * (yaw[i + 1] - yaw[i])
        pitch_i = pitch[i] + f * (pitch[i + 1] - pitch[i])
        return pos, yaw_i, pitch_i, seg_labels[i]

    yaw_cap, pitch_cap = caps.max_yaw_rate, caps.max_gimbal_pitch_rate
    pitch_lo = max(caps.min_gimbal_pitch, -math.pi / 2 + 1e-9)
    pitch_hi = min(caps.max_gimbal_pitch, math.pi / 2 - 1e-9)

    for _ in range(max_stretch_iters):
        T = float(t_seg_end[-1] * lam)
        n_steps = max(1, int(math.ceil(T / dt_out - 1e-9)))
        ts = np.asarray([min(k * dt_out, T) for k in range(n_steps + 1)])
        pos, yaw_des, pitch_des, seg_o = _interp(_u_at(ts))

        pitch_cmd = np.clip(pitch_des, pitch_lo, pitch_hi)
        dts = np.diff(ts)
        yaw_l = _slew_limit(yaw_des, yaw_cap, dts, start=yaw0)
        pitch_l = _slew_limit(pitch_cmd, pitch_cap, dts, start=pitch0)

        # terminal catch-up: the look target may jump at the very last frames,
        # leaving a slew deficit that extra PATH time cannot absorb (zero arc
        # left). Append a stationary tail — the profile already ends at rest.
        lag = max(abs(yaw_l[-1] - yaw_des[-1]) / yaw_cap,
                  abs(pitch_l[-1] - pitch_cmd[-1]) / pitch_cap)
        if lag > 0.02:
            tail = int(math.ceil((lag + 2.0 * dt_out) / dt_out))
            ts = np.concatenate([ts, ts[-1] + dt_out * np.arange(1, tail + 1)])
            pos, yaw_des, pitch_des, seg_o = _interp(_u_at(ts))
            pitch_cmd = np.clip(pitch_des, pitch_lo, pitch_hi)
            dts = np.diff(ts)
            yaw_l = _slew_limit(yaw_des, yaw_cap, dts, start=yaw0)
            pitch_l = _slew_limit(pitch_cmd, pitch_cap, dts, start=pitch0)
            lag = max(abs(yaw_l[-1] - yaw_des[-1]) / yaw_cap,
                      abs(pitch_l[-1] - pitch_cmd[-1]) / pitch_cap)

        # verify: speeds ∝ 1/λ, accelerations ∝ 1/λ² (platform caps)
        dp = np.diff(pos, axis=0)
        factor = max(1.0, caps.interval_factor(dp, dts, yaw_l, pitch_l, dt_out))

        if factor <= 1.0 + 1e-9 and lag <= 0.02:
            T_end = float(ts[-1])
            new_samples = [{"t": round(float(t), 6),
                            "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                            "yaw_deg": round(math.degrees(ya), 3),
                            "pitch_deg": round(math.degrees(pa), 3), "roll_deg": 0.0,
                            "fov_deg": samples[0].get("fov_deg") or 55.0,
                            "segment": str(sg)}
                           for t, p, ya, pa, sg in zip(ts, pos, yaw_l, pitch_l, seg_o)]

            commands = caps.build_commands(ts, dp, dts, yaw_l, pitch_l)
            return RetimeResult(samples=new_samples, commands=commands,
                                duration=T_end, scale=T_end / t0)
        lam = max(lam * factor * 1.000001, lam + lag / max(t_seg_end[-1], _EPS))
    raise RuntimeError("retime: could not satisfy caps within "
                       f"{max_stretch_iters} stretch iterations (path too kinky?)")


def _slew_limit(desired: np.ndarray, rate: float, dts: np.ndarray,
                start: float | None = None) -> np.ndarray:
    """Slew-rate limiter: track `desired` with |Δ/Δt| ≤ rate (linear ramp-rate).
    ``start`` warm-starts the tracked heading (per-segment chaining)."""
    out = np.empty_like(desired)
    out[0] = desired[0] if start is None else float(start)
    for k in range(len(dts)):
        step = min(max(desired[k + 1] - out[k], -rate * dts[k]), rate * dts[k])
        out[k + 1] = out[k] + step
    return out
