"""Differentiable scene SDF: analytic union of OBB signed distance fields.

Deviation from CinemaTraj (deliberate, documented): the paper voxelizes the
OBB mesh (adaptive voxelization → morphological closing → marching cubes →
flood-fill sign → 256³ grid + trilinear interpolation). We instead evaluate
the union analytically, Φ(p) = min_i sdf_obb_i(p), which is

  * exact for box scenes (the paper's grid is itself built from OBBs and adds
    discretization error),
  * dependency-free (no meshing stack), fully vectorized, CPU-fast,
  * differentiable everywhere except box-bisector surfaces — a subgradient is
    fine under Adam + gradient clipping; an optional log-sum-exp soft-min
    (``soft_min_k``) smooths those seams if oscillation is ever observed.

The interface is designed so a grid SDF built from a raw mesh could replace it
later without touching the optimizer.

Performance note: all boxes are evaluated in ONE broadcasted batch — stacked
center/rotation/half-extent arrays are precomputed once in ``__init__`` and
torch tensors are cached per (device, dtype) in ``_torch_cache`` — instead of
a Python loop of per-OBB tensors. If the OBB list is mutated after
construction, call ``invalidate_cache()``.
"""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

from .obb import OBB


_SCENE_SDF_CACHE: dict = {}


def scene_sdf_cached(scene, soft_min_k=None):
    """Per-process memo of the main collision SceneSDF for a scene.

    build.run and optimize.run construct byte-identical fields for the
    same scene; queries are read-only, so the second construction in a
    process is pure rework. Keyed on (scene_id, soft_min_k)."""
    from .cloud_sdf import cloud_sdf_from_scene
    key = (getattr(scene, "scene_id", id(scene)), soft_min_k)
    if key not in _SCENE_SDF_CACHE:
        _SCENE_SDF_CACHE[key] = SceneSDF(
            scene.collision_obbs(), scene.room_bounds(),
            soft_min_k=soft_min_k, cloud=cloud_sdf_from_scene(scene))
    return _SCENE_SDF_CACHE[key]


class SceneSDF:
    """Union SDF over collision OBBs (structures + objects) with room bounds.

    The dynamic subject capsule is NOT part of this static field — it is added
    by the optimizer at each sampled time (see cinema_traj.core.costs).
    """

    def __init__(self, obbs: list[OBB], room_bounds: list[tuple[np.ndarray, np.ndarray]],
                 soft_min_k: float | None = None, cloud=None):
        self.obbs = list(obbs)
        self.room_bounds = [(
            np.asarray(lo, dtype=np.float64).reshape(3),
            np.asarray(hi, dtype=np.float64).reshape(3),
        ) for lo, hi in room_bounds]
        self.soft_min_k = soft_min_k
        # optional ground-truth field from the real point cloud (geometry the
        # boxes don't cover: thin decor, detail, fuzz). Pseudo-signed:
        # distance − hit_margin; combined by plain min everywhere.
        self.cloud = cloud
        self._torch_cache: dict = {}
        # stacked per-box arrays, (N,3) centers / (N,3,3) rotations (columns =
        # local axes, so local = (p − c) @ R) / (N,3) half extents
        self._c = np.stack([o.center for o in self.obbs], axis=0) if self.obbs \
            else np.zeros((0, 3))
        self._R = np.stack([o.rotation for o in self.obbs], axis=0) if self.obbs \
            else np.zeros((0, 3, 3))
        self._h = np.stack([o.half for o in self.obbs], axis=0) if self.obbs \
            else np.zeros((0, 3))

    def invalidate_cache(self) -> None:
        """Rebuild stacked arrays after OBB mutation (rare; tests only)."""
        self._torch_cache.clear()
        self._c = np.stack([o.center for o in self.obbs], axis=0) if self.obbs \
            else np.zeros((0, 3))
        self._R = np.stack([o.rotation for o in self.obbs], axis=0) if self.obbs \
            else np.zeros((0, 3, 3))
        self._h = np.stack([o.half for o in self.obbs], axis=0) if self.obbs \
            else np.zeros((0, 3))

    def restricted_to(self, aabb, margin: float = 0.5) -> "SceneSDF":
        """Pruned view keeping only boxes that can interact with ``aabb``.

        Boxes whose own AABB clears the query box by more than ``margin`` are
        dropped: at ≥ margin clearance both cost terms they could contribute
        (hinge ρ(g) and softplus_β(δ−Φ), β=20) are < 1e-4, so the optimizer's
        trajectory is unaffected while the batched SDF shrinks. Returns
        ``self`` when nothing is pruned."""
        if not self.obbs:
            return self
        lo, hi = (np.asarray(aabb[0], float), np.asarray(aabb[1], float))
        keep = [o for o in self.obbs
                if self._box_near(o, lo, hi, margin)]
        if len(keep) == len(self.obbs):
            return self
        pruned = SceneSDF(keep, self.room_bounds, self.soft_min_k)
        pruned.cloud = self.cloud                     # cloud field is global
        return pruned

    @staticmethod
    def _box_near(o: OBB, lo: np.ndarray, hi: np.ndarray, margin: float) -> bool:
        cs = o.corners()
        return bool(np.all(cs.max(axis=0) >= lo - margin)
                    and np.all(cs.min(axis=0) <= hi + margin))

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    _EMPTY_DIST = 1e6          # no obstacles → effectively infinite clearance

    def _box_vals_np(self, p2: np.ndarray) -> np.ndarray:
        """Per-box SDF values, (N, M)."""
        diff = p2[None, :, :] - self._c[:, None, :]          # (N,M,3)
        q = np.abs(diff @ self._R) - self._h[:, None, :]     # (N,M,3)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(q.max(axis=-1), 0.0)
        return outside + inside

    def numpy_query(self, p: np.ndarray) -> np.ndarray:
        """Φ at world points (N,3) or (3,) — numpy, no autograd."""
        p2 = np.atleast_2d(np.asarray(p, dtype=np.float64))
        if not self.obbs:
            out = np.full(p2.shape[0], self._EMPTY_DIST)
        else:
            out = self._reduce(self._box_vals_np(p2))
        if self.cloud is not None:
            out = np.minimum(out, self.cloud.numpy_query(p2))
        return out if np.ndim(p) == 2 else float(out[0])

    def numpy_query_nearest(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(Φ values (M,), nearest box index (M,)) under the plain min — used
        to attribute residual collisions to a specific scene element."""
        p2 = np.atleast_2d(np.asarray(p, dtype=np.float64))
        vals = self._box_vals_np(p2)                          # (N,M)
        idx = vals.argmin(axis=0)
        return vals.min(axis=0), idx

    def box_label(self, i: int) -> str:
        """Human-readable label for box ``i`` (id + rotation), if the caller
        attached ``labels`` in scene order (structures then objects)."""
        labels = getattr(self, "labels", None)
        return labels[i] if labels and 0 <= i < len(labels) else f"box#{i}"

    def __call__(self, p) -> "torch.Tensor":
        """Differentiable Φ. p: (N,3) torch tensor."""
        if not _HAS_TORCH:
            raise RuntimeError("torch not available")
        if not self.obbs:
            out = torch.full((p.shape[0],), self._EMPTY_DIST, dtype=p.dtype,
                             device=p.device)
        else:
            key = (str(p.device), p.dtype)
            if key not in self._torch_cache:
                self._torch_cache[key] = (
                    torch.as_tensor(self._c, dtype=p.dtype, device=p.device),
                    torch.as_tensor(self._R, dtype=p.dtype, device=p.device),
                    torch.as_tensor(self._h, dtype=p.dtype, device=p.device),
                )
            c, R, h = self._torch_cache[key]
            diff = p.unsqueeze(0) - c.unsqueeze(1)                # (N,M,3)
            q = (diff @ R).abs() - h.unsqueeze(1)                 # (N,M,3)
            outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
            inside = torch.clamp(q.max(dim=-1).values, max=0.0)
            vals = outside + inside                               # (N,M)
            out = self._reduce_torch(vals)
        if self.cloud is not None:
            out = torch.minimum(out, self.cloud.torch_query(p))
        return out

    def _reduce(self, vals: np.ndarray) -> np.ndarray:
        if self.soft_min_k is None:
            return vals.min(axis=0)
        k = self.soft_min_k
        return -(np.log(np.sum(np.exp(-k * vals), axis=0))) / k

    def _reduce_torch(self, vals):
        if self.soft_min_k is None:
            return vals.min(dim=0).values
        k = self.soft_min_k
        return -torch.logsumexp(-k * vals, dim=0) / k

    # ------------------------------------------------------------------ #
    # Visibility ray march (occlusion cost helper)
    # ------------------------------------------------------------------ #
    def ray_march_hits(self, origins, targets, k_steps: int = 32,
                       near_skip: float = 0.1, target_mask_dist: float | None = None):
        """March rays origins→targets; return per-ray stacked SDF samples.

        origins/targets: (N,3) torch tensors, differentiable. Returns
        hits (N, K) of Φ along each ray (samples within ``near_skip`` of the
        origin and within ``target_mask_dist`` of the target are masked to 0 so
        they never contribute — target self-occlusion guard, paper §D.5).
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch not available")
        ray = targets - origins
        dist = ray.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dirn = ray / dist
        s = torch.linspace(near_skip, 1.0, k_steps, dtype=origins.dtype,
                           device=origins.device).view(1, k_steps)
        along = s * dist                                   # (N,K) distance travelled
        pts = origins.unsqueeze(1) + dirn.unsqueeze(1) * s.unsqueeze(-1) * dist.unsqueeze(-1)
        hits = self(pts.reshape(-1, 3)).view(pts.shape[0], k_steps)
        if target_mask_dist is not None:
            # zero samples within target_mask_dist of the endpoint (self-occlusion guard)
            near_target = along > (dist - target_mask_dist)
            hits = torch.where(near_target, torch.zeros_like(hits), hits)
        return hits

    # ------------------------------------------------------------------ #
    # Rooms / bounds
    # ------------------------------------------------------------------ #
    def room_bounds_of_point(self, p) -> tuple[np.ndarray, np.ndarray] | None:
        """Bounds of the room containing p (geometric containment), else None."""
        q = np.asarray(p, dtype=np.float64).reshape(3)
        for lo, hi in self.room_bounds:
            if np.all(q >= lo) and np.all(q <= hi):
                return lo, hi
        return self.room_bounds[0] if self.room_bounds else None

    def scene_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.room_bounds:
            raise ValueError("no room bounds in scene")
        lo = np.min([lo for lo, _ in self.room_bounds], axis=0)
        hi = np.max([hi for _, hi in self.room_bounds], axis=0)
        return lo, hi
