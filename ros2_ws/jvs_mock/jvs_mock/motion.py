"""Pure helpers for deterministic mock arm commands."""

from __future__ import annotations

import math


JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# Symmetric amplitudes about zero, each strictly inside its URDF limits.
# joint5 has only +2 degrees of positive travel, so its test amplitude is small.
SAFE_AMPLITUDES = (0.35, 0.30, 0.40, 0.35, 0.03, 0.35)
FREQUENCY_MULTIPLIERS = (1.0, 0.75, 0.55, 0.85, 0.65, 1.15)


def sample_joint_positions(
    elapsed_s: float,
    period_s: float = 10.0,
    amplitude_scale: float = 1.0,
) -> list[float]:
    """Return six smooth, bounded position commands in joint1..joint6 order."""
    if period_s <= 0.0:
        raise ValueError("period_s must be positive")
    if not 0.0 <= amplitude_scale <= 1.0:
        raise ValueError("amplitude_scale must be in [0, 1]")
    phase = 2.0 * math.pi * max(0.0, elapsed_s) / period_s
    return [
        amplitude_scale * amplitude * math.sin(phase * multiplier)
        for amplitude, multiplier in zip(SAFE_AMPLITUDES, FREQUENCY_MULTIPLIERS)
    ]

