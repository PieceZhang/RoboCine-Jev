"""SE(3) helpers: look_at rotation, spherical offsets, easing, angle wrap.

World frame is +Y up, right-handed, meters. Camera convention: the rotation
matrix columns are (right, up, forward) and the camera looks along ``forward``
— orientation is always representable as (yaw, pitch) with roll = 0, which is
what the export adapters emit.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

WORLD_UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------- #
# look_at
# --------------------------------------------------------------------------- #
def look_at_R(eye, target, up=WORLD_UP) -> np.ndarray:
    """Rotation (3,3), columns = (right, up, forward), looking eye→target.

    Handles the degenerate forward ∥ up case by nudging up.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    f = target - eye
    n = np.linalg.norm(f)
    if n < 1e-9:
        return np.eye(3)
    f = f / n
    up = np.asarray(up, dtype=np.float64)
    if np.linalg.norm(np.cross(f, up)) < 1e-6:          # looking straight up/down
        up = up + np.array([1e-3, 0.0, 0.0])
        up = up / np.linalg.norm(up)
    r = np.cross(up, f)                      # right-handed: det([r,u,f]) = +1
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=1)


def look_at_R_torch(eye, target, up=None):
    """Differentiable look_at. eye/target: (F,3) or (3,). Returns (F,3,3) or (3,3)."""
    if not _HAS_TORCH:
        raise RuntimeError("torch not available")
    single = eye.dim() == 1
    eye = eye.unsqueeze(0) if single else eye
    target = target.unsqueeze(0) if single else target
    upv = torch.tensor([0.0, 1.0, 0.0], dtype=eye.dtype, device=eye.device) if up is None else up
    f = target - eye
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    # nudge degenerate frames slightly (gradient-safe constant offset)
    cross_fu = torch.cross(f, upv.expand_as(f), dim=-1)
    degenerate = cross_fu.norm(dim=-1, keepdim=True) < 1e-6
    up_adj = torch.where(degenerate, upv + torch.tensor([1e-3, 0.0, 0.0], dtype=eye.dtype,
                                                        device=eye.device), upv.expand_as(f))
    up_adj = up_adj / up_adj.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    r = torch.cross(up_adj, f, dim=-1)          # right-handed: det([r,u,f]) = +1
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    u = torch.cross(f, r, dim=-1)
    R = torch.stack([r, u, f], dim=-1)                  # (F,3,3), columns right/up/forward
    return R.squeeze(0) if single else R


def rotation_to_yaw_pitch(R) -> tuple[float, float]:
    """(right,up,forward)-column rotation → (yaw_deg, pitch_deg).

    yaw = atan2(f.x, f.z) (0 → +Z, positive toward +X), pitch = asin(f.y).
    """
    f = np.asarray(R, dtype=np.float64)[:, 2]
    yaw = math.degrees(math.atan2(f[0], f[2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, f[1]))))
    return yaw, pitch


# --------------------------------------------------------------------------- #
# Spherical ↔ cartesian
# --------------------------------------------------------------------------- #
def sph_offset(az_rad: float, el_rad: float, dist: float) -> np.ndarray:
    """Spherical → cartesian offset. Azimuth in the XZ plane from +X toward +Z;
    elevation above the horizontal plane."""
    ce = math.cos(el_rad)
    return np.array([dist * ce * math.cos(az_rad),
                     dist * math.sin(el_rad),
                     dist * ce * math.sin(az_rad)])


def cart_to_sph(v) -> tuple[float, float, float]:
    v = np.asarray(v, dtype=np.float64)
    dist = float(np.linalg.norm(v))
    if dist < 1e-12:
        return 0.0, 0.0, 0.0
    az = math.atan2(v[2], v[0])
    el = math.asin(max(-1.0, min(1.0, v[1] / dist)))
    return az, el, dist


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# --------------------------------------------------------------------------- #
# Angles & easing
# --------------------------------------------------------------------------- #
def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_lerp(a: float, b: float, u: float) -> float:
    """Shortest-arc interpolation between two angles."""
    return a + wrap_angle(b - a) * u


def ease_in_out(t):
    """Smoothstep ease-in-out σ(t): σ(0)=0, σ(1)=1, zero velocity at ends."""
    if _HAS_TORCH and isinstance(t, torch.Tensor):
        tc = torch.clamp(t, 0.0, 1.0)
        return tc * tc * (3.0 - 2.0 * tc)
    tc = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return tc * tc * (3.0 - 2.0 * tc)
