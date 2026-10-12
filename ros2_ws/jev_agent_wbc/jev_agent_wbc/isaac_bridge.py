"""Pure helpers shared by the Isaac Sim ROS command adapter and tests."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


ARM_JOINT_NAMES: Tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 7))
WHEEL_JOINT_NAMES: Tuple[str, ...] = ("left_wheel_joint", "right_wheel_joint")


def arm_positions(values: Iterable[float]) -> list[float]:
    """Validate one complete arm position command."""

    positions = [float(value) for value in values]
    if len(positions) != len(ARM_JOINT_NAMES):
        raise ValueError(
            f"expected {len(ARM_JOINT_NAMES)} arm positions, got {len(positions)}"
        )
    if not all(math.isfinite(value) for value in positions):
        raise ValueError("arm position command contains a non-finite value")
    return positions


def limited_twist(
    linear_x: float,
    angular_z: float,
    max_linear: float,
    max_angular: float,
) -> tuple[float, float]:
    """Return finite, symmetrically limited planar velocity components."""

    values = (linear_x, angular_z, max_linear, max_angular)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("velocity command and limits must be finite")
    if max_linear <= 0.0 or max_angular <= 0.0:
        raise ValueError("velocity limits must be positive")
    return (
        max(-max_linear, min(max_linear, float(linear_x))),
        max(-max_angular, min(max_angular, float(angular_z))),
    )


def acceleration_limited_twist(
    target: tuple[float, float],
    previous: tuple[float, float],
    dt: float,
    max_linear_acceleration: float,
    max_angular_acceleration: float,
) -> tuple[float, float]:
    """Apply the symmetric acceleration limits used by diff_drive_controller."""
    values = (*target, *previous, dt, max_linear_acceleration, max_angular_acceleration)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("acceleration inputs must be finite")
    if dt < 0.0 or min(max_linear_acceleration, max_angular_acceleration) <= 0.0:
        raise ValueError("time step must be nonnegative and acceleration limits positive")
    linear_step = max_linear_acceleration * dt
    angular_step = max_angular_acceleration * dt
    return (
        previous[0] + max(-linear_step, min(linear_step, target[0] - previous[0])),
        previous[1] + max(-angular_step, min(angular_step, target[1] - previous[1])),
    )


def differential_wheel_velocities(
    linear_x: float,
    angular_z: float,
    wheel_radius: float,
    wheel_separation: float,
) -> list[float]:
    """Convert one planar velocity command to left/right wheel rates."""

    values = (linear_x, angular_z, wheel_radius, wheel_separation)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("velocity command and wheel geometry must be finite")
    if wheel_radius <= 0.0 or wheel_separation <= 0.0:
        raise ValueError("wheel geometry must be positive")
    half_track_rotation = float(angular_z) * float(wheel_separation) * 0.5
    return [
        (float(linear_x) - half_track_rotation) / float(wheel_radius),
        (float(linear_x) + half_track_rotation) / float(wheel_radius),
    ]
