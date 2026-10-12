"""Oriented bounding box: exact SDF (numpy + torch), corners, PCA fit, ray slabs.

Conventions (package-wide): right-handed world, meters, +Y up. OBB rotation is
a 3x3 matrix whose COLUMNS are the local axes expressed in world frame, so
``p_local = (p_world - center) @ rotation`` for row-vector batches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # torch is an optional (traj/depth extra) dependency — numpy paths work without it
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# OBB dataclass
# --------------------------------------------------------------------------- #
@dataclass
class OBB:
    center: np.ndarray            # (3,)
    rotation: np.ndarray          # (3,3) columns = local axes in world
    extents: np.ndarray           # (3,) full extents (not half)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.extents = np.asarray(self.extents, dtype=np.float64).reshape(3)

    # -- serialization ------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict) -> "OBB":
        return cls(d["center"], d["rotation"], d["extents"])

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "rotation": self.rotation.tolist(),
            "extents": self.extents.tolist(),
        }

    # -- geometry ----------------------------------------------------------- #
    @property
    def half(self) -> np.ndarray:
        return self.extents / 2.0

    def to_local(self, p: np.ndarray) -> np.ndarray:
        """World points (N,3) or (3,) → OBB-local coordinates."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        return (p - self.center) @ self.rotation

    def to_world(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        return q @ self.rotation.T + self.center

    def corners(self) -> np.ndarray:
        """8 corners in world frame, (8,3)."""
        h = self.half
        offs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return offs * h @ self.rotation.T + self.center

    def contains(self, p: np.ndarray) -> np.ndarray | bool:
        q = np.abs(self.to_local(p))
        return np.all(q <= self.half, axis=-1)

    # -- exact SDF ---------------------------------------------------------- #
    def sdf(self, p: np.ndarray) -> np.ndarray:
        """Exact box SDF (negative inside). Accepts (N,3) or (3,); scalar in,
        scalar out."""
        single = np.ndim(p) == 1
        q = np.abs(self.to_local(p)) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        out = outside + inside
        return float(out[0]) if single else out

    def sdf_torch(self, p) -> "torch.Tensor":
        """Differentiable exact box SDF. p: (N,3) float tensor."""
        if not _HAS_TORCH:
            raise RuntimeError("torch not available")
        center = torch.as_tensor(self.center, dtype=p.dtype, device=p.device)
        rot = torch.as_tensor(self.rotation, dtype=p.dtype, device=p.device)
        half = torch.as_tensor(self.half, dtype=p.dtype, device=p.device)
        q = (p - center) @ rot
        q = q.abs() - half
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        return outside + inside

    # -- misc --------------------------------------------------------------- #
    def largest_face_normal(self) -> np.ndarray:
        """World direction of the axis with the smallest extent — i.e. the
        normal of the box's two largest faces (useful for wall objects)."""
        axis = int(np.argmin(self.extents))
        return self.rotation[:, axis].copy()

    def max_half_extent(self) -> float:
        return float(np.max(self.half))


# --------------------------------------------------------------------------- #
# Ray ↔ OBB slab intersection (batched)
# --------------------------------------------------------------------------- #
def ray_obb_intersect(origins: np.ndarray, dirs: np.ndarray, obb: OBB):
    """Batched ray-OBB slab test.

    origins, dirs: (N,3). dirs need not be normalized (t is in units of |dirs|).
    Returns (t_enter, t_exit), each (N,); rays that miss have t_exit < t_enter.
    """
    o = obb.to_local(origins)
    d = dirs @ obb.rotation
    h = obb.half
    t_enter = np.full(o.shape[0], -np.inf)
    t_exit = np.full(o.shape[0], np.inf)
    for i in range(3):
        di = d[:, i]
        oi = o[:, i]
        parallel = np.abs(di) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h[i] - oi) / di
            t2 = (h[i] - oi) / di
        t_lo = np.minimum(t1, t2)
        t_hi = np.maximum(t1, t2)
        t_lo = np.where(parallel, -np.inf, t_lo)
        t_hi = np.where(parallel, np.inf, t_hi)
        # parallel rays outside the slab miss immediately
        miss = parallel & (np.abs(oi) > h[i])
        t_hi = np.where(miss, -np.inf, t_hi)
        t_enter = np.maximum(t_enter, t_lo)
        t_exit = np.minimum(t_exit, t_hi)
    return t_enter, t_exit


# --------------------------------------------------------------------------- #
# PCA OBB fit (extractor)
# --------------------------------------------------------------------------- #
def fit_obb_pca(points: np.ndarray, snap_axes: bool = True,
                pct: tuple = (5.0, 95.0)) -> OBB:
    """Fit an OBB to a point cloud via PCA on the covariance.

    Extents use robust percentile range along each principal axis (default
    5–95%) so a few stray points cannot inflate the box. ``snap_axes`` rotates
    each principal axis to the nearest world axis when within 8° — indoor
    furniture is overwhelmingly axis-aligned in practice.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        raise ValueError(f"need >=3 points, got {pts.shape}")
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    # fix sign ambiguity: make each axis point toward +world where possible
    for i in range(3):
        if evecs[np.argmax(np.abs(evecs[:, i])), i] < 0:
            evecs[:, i] *= -1.0
    R = evecs
    if snap_axes:
        R = _snap_rotation(R, tol_deg=8.0)
        # re-orthonormalize after snapping
        u, _, vt = np.linalg.svd(R)
        R = u @ vt
        if np.linalg.det(R) < 0:                # keep right-handed
            R[:, 2] *= -1.0
    local = (pts - center) @ R
    lo = np.percentile(local, pct[0], axis=0)
    hi = np.percentile(local, pct[1], axis=0)
    extents = hi - lo
    extents = np.maximum(extents, 1e-3)
    center = center + R @ ((lo + hi) / 2.0)
    return OBB(center, R, extents)


def _snap_rotation(R: np.ndarray, tol_deg: float) -> np.ndarray:
    """Rotate each column to the nearest world axis if within tol."""
    snapped = R.copy()
    used: set[int] = set()
    cols = []
    # snap the most axis-aligned column first so axis assignment stays injective
    order = sorted(range(3), key=lambda i: -np.max(np.abs(R[:, i])))
    for i in order:
        col = snapped[:, i]
        best_axis, best_dot = None, -1.0
        for a in range(3):
            if a in used:
                continue
            dot = abs(col[a])
            if dot > best_dot:
                best_dot, best_axis = dot, a
        if best_axis is not None and best_dot >= math.cos(math.radians(tol_deg)):
            new = np.zeros(3)
            new[best_axis] = 1.0 if col[best_axis] > 0 else -1.0
            cols.append((i, new))
            used.add(best_axis)
    for i, new in cols:
        snapped[:, i] = new
    return snapped
