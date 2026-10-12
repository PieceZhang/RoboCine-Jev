"""Unsigned distance-to-cloud voxel grid — ground-truth collision field.

The scene-graph OBB union covers furniture it knows about, but REAL scans
(and 3DGS dumps) contain geometry the boxes miss: thin decor, detail,
floaters/fuzz. Trajectories that are OBB-clear can still graze real points
(PR #2's known issue: the orbit was 0% OBB-collision while 8 frames came
within 0.10 m of actual cloud points).

``CloudSDFGrid`` voxelizes the cloud, computes EXACT nearest-point distance
inside a ``band_m`` neighborhood (729 offset shells — every cell in a ±band
window is enumerated, so within the band this is the true L2 distance, not a
chamfer approximation; distances are measured to occupied CELL CENTERS, so
budget ±half a voxel diagonal — ~±0.04 m at the default 0.05 m voxel — when
choosing hit_margin), and caps distances at ``band_m`` beyond it — the
optimizer only ever cares below its clearance margin. Queries:

  * ``numpy_query(p)``  — pseudo-SIGNED: distance − hit_margin
  * ``torch_query(p)``  — differentiable trilinear interpolation of the same
                          grid (gradients flow to camera positions)

Both query paths share the same cell-frame math (``_cell``), so numpy and
torch return identical values.
"""

from __future__ import annotations

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


class CloudSDFGrid:
    def __init__(self, points: np.ndarray, lo, hi, voxel: float = 0.05,
                 band_m: float = 0.30, hit_margin: float = 0.10):
        points = np.asarray(points, dtype=np.float64)
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        self.voxel = float(voxel)
        self.band_m = float(band_m)
        self.hit_margin = float(hit_margin)
        self.origin = lo                                  # grid corner (cell center + voxel/2)
        grid_shape = np.maximum(np.ceil((hi - lo) / voxel).astype(int), 1)
        self.shape = tuple(int(s) for s in grid_shape)

        # occupancy: which cell centers hold at least one cloud point
        cell_idx = np.floor((points - lo) / voxel).astype(int)
        inside_grid = np.all((cell_idx >= 0) & (cell_idx < np.array(self.shape)), axis=1)
        occupied = np.zeros(self.shape, dtype=bool)
        occupied[cell_idx[inside_grid, 0], cell_idx[inside_grid, 1],
                 cell_idx[inside_grid, 2]] = True

        # exact L2 distance to the nearest occupied cell within the band:
        # enumerate every offset in the ±band-radius window once (cheaper and
        # exact where a two-pass chamfer would only approximate)
        window_radius = int(np.ceil(band_m / voxel))
        offsets = np.array([[dx, dy, dz]
                            for dx in range(-window_radius, window_radius + 1)
                            for dy in range(-window_radius, window_radius + 1)
                            for dz in range(-window_radius, window_radius + 1)],
                           dtype=int)
        dist = np.full(self.shape, np.inf, dtype=np.float32)
        padded = np.zeros(np.array(self.shape) + 2 * window_radius, dtype=bool)
        padded[window_radius:window_radius + self.shape[0],
               window_radius:window_radius + self.shape[1],
               window_radius:window_radius + self.shape[2]] = occupied
        for offset in offsets:
            cell_dist = float(np.linalg.norm(offset) * voxel)
            if cell_dist > band_m:
                continue
            occupied_shifted = padded[
                window_radius - offset[0]:window_radius - offset[0] + self.shape[0],
                window_radius - offset[1]:window_radius - offset[1] + self.shape[1],
                window_radius - offset[2]:window_radius - offset[2] + self.shape[2]]
            np.minimum(dist, np.where(occupied_shifted, cell_dist, np.inf), out=dist)
        dist[np.isinf(dist)] = band_m                     # beyond band: ≥ band
        self.dist = dist
        self._torch_grid = None

    # ------------------------------------------------------------------ #
    # Queries (pseudo-signed: distance − hit_margin)
    # ------------------------------------------------------------------ #
    def _cell(self, points_xyz: np.ndarray):
        """Shared cell-frame math for both query paths: the 2×2×2 corner
        indices (``corner_lo``/``corner_hi``, clamped to the grid) and the
        fractional position ``frac`` between them."""
        cell_pos = (points_xyz - self.origin) / self.voxel - 0.5
        floor_idx = np.floor(cell_pos)
        last_cell = np.array(self.shape) - 1
        corner_lo = np.minimum(np.clip(floor_idx, 0, None), last_cell).astype(int)
        corner_hi = np.minimum(np.clip(floor_idx + 1, 0, None), last_cell).astype(int)
        frac = np.clip(cell_pos - corner_lo, 0.0, 1.0)
        return corner_lo, corner_hi, frac

    def numpy_query(self, p) -> np.ndarray | float:
        points_xyz = np.atleast_2d(np.asarray(p, dtype=np.float64))
        lo_idx, hi_idx, frac = self._cell(points_xyz)

        def corner_value(ix, iy, iz):
            return self.dist[ix, iy, iz]

        # trilinear: interpolate along z, then y, then x
        z_lo_y_lo = corner_value(lo_idx[:, 0], lo_idx[:, 1], lo_idx[:, 2]) * (1 - frac[:, 2]) \
            + corner_value(lo_idx[:, 0], hi_idx[:, 1], lo_idx[:, 2]) * frac[:, 2]
        z_hi_y_lo = corner_value(lo_idx[:, 0], lo_idx[:, 1], hi_idx[:, 2]) * (1 - frac[:, 2]) \
            + corner_value(lo_idx[:, 0], hi_idx[:, 1], hi_idx[:, 2]) * frac[:, 2]
        z_lo_y_hi = corner_value(hi_idx[:, 0], lo_idx[:, 1], lo_idx[:, 2]) * (1 - frac[:, 2]) \
            + corner_value(hi_idx[:, 0], hi_idx[:, 1], lo_idx[:, 2]) * frac[:, 2]
        z_hi_y_hi = corner_value(hi_idx[:, 0], lo_idx[:, 1], hi_idx[:, 2]) * (1 - frac[:, 2]) \
            + corner_value(hi_idx[:, 0], hi_idx[:, 1], hi_idx[:, 2]) * frac[:, 2]
        y_lo = z_lo_y_lo * (1 - frac[:, 1]) + z_hi_y_lo * frac[:, 1]
        y_hi = z_lo_y_hi * (1 - frac[:, 1]) + z_hi_y_hi * frac[:, 1]
        distance = y_lo * (1 - frac[:, 0]) + y_hi * frac[:, 0]

        out = distance - self.hit_margin
        return out if np.ndim(p) == 2 else float(out[0])

    def torch_query(self, p):
        if not _HAS_TORCH:
            raise RuntimeError("torch not available")
        if self._torch_grid is None or self._torch_grid.dtype != p.dtype \
                or self._torch_grid.device != p.device:
            self._torch_grid = torch.as_tensor(self.dist, dtype=p.dtype,
                                               device=p.device)
        grid = self._torch_grid
        cell_pos = (p - p.new_tensor(self.origin)) / self.voxel - 0.5
        floor_idx = torch.floor(cell_pos)
        last_cell = p.new_tensor(self.shape) - 1
        lo_idx = torch.minimum(torch.clamp(floor_idx, min=0), last_cell).long()
        hi_idx = torch.minimum(torch.clamp(floor_idx + 1, min=0), last_cell).long()
        frac = torch.clamp(cell_pos - lo_idx.to(cell_pos.dtype), 0.0, 1.0)
        x_lo, y_lo, z_lo = lo_idx[:, 0], lo_idx[:, 1], lo_idx[:, 2]
        x_hi, y_hi, z_hi = hi_idx[:, 0], hi_idx[:, 1], hi_idx[:, 2]
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

        # trilinear: interpolate along z, then y, then x
        z_lo_y_lo = grid[x_lo, y_lo, z_lo] * (1 - fz) + grid[x_lo, y_hi, z_lo] * fz
        z_hi_y_lo = grid[x_lo, y_lo, z_hi] * (1 - fz) + grid[x_lo, y_hi, z_hi] * fz
        z_lo_y_hi = grid[x_hi, y_lo, z_lo] * (1 - fz) + grid[x_hi, y_hi, z_lo] * fz
        z_hi_y_hi = grid[x_hi, y_lo, z_hi] * (1 - fz) + grid[x_hi, y_hi, z_hi] * fz
        y_lo_val = z_lo_y_lo * (1 - fy) + z_hi_y_lo * fy
        y_hi_val = z_lo_y_hi * (1 - fy) + z_hi_y_hi * fy
        return y_lo_val * (1 - fx) + y_hi_val * fx - self.hit_margin


def cloud_sdf_from_scene(scene, voxel: float = 0.05, band_m: float = 0.30,
                         hit_margin: float = 0.10) -> CloudSDFGrid | None:
    """Build the cloud field from a scene's provenance (source_map + map_up +
    map_shift), clipped to the first room's AABB. Best-effort: any missing
    provenance/file just returns None (the OBB field still runs)."""
    provenance = getattr(scene, "provenance", {}) or {}
    source = provenance.get("source_map", "")
    if not source or not scene.rooms:
        return None
    from pathlib import Path

    cloud_path = Path(source)
    if not cloud_path.is_absolute():
        cloud_path = Path(__file__).resolve().parents[2] / cloud_path
    if not cloud_path.exists():
        return None
    try:
        from cinema_traj.stages.extract import _to_yup, load_points, voxel_downsample

        points = np.asarray(load_points(str(cloud_path)), dtype=np.float64)
        points = _to_yup(points, up=str(provenance.get("map_up", "y")))
        points = points + np.asarray(provenance.get("map_shift") or [0.0, 0.0, 0.0])
        points = voxel_downsample(points, voxel)
        grid_lo = scene.rooms[0].bounds_min - 0.6
        grid_hi = scene.rooms[0].bounds_max + 0.6
        return CloudSDFGrid(points, grid_lo, grid_hi, voxel=voxel,
                            band_m=band_m, hit_margin=hit_margin)
    except Exception:                                      # noqa: BLE001
        return None
