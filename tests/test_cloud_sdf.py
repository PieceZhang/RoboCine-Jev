"""CloudSDFGrid tests — exact band distances, margin semantics, torch
consistency/differentiability, SceneSDF combination, and scene loader."""

import numpy as np
import pytest
import torch

from cinema_traj.common.cloud_sdf import CloudSDFGrid, cloud_sdf_from_scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.common.scene import Scene

ROOM = [(np.array([-2.0, -1.0, -2.0]), np.array([2.0, 2.0, 2.0]))]


def _grid(points, **kw):
    return CloudSDFGrid(np.asarray(points, float), [-2.0, -1.0, -2.0],
                        [2.0, 2.0, 2.0], **kw)


class TestGrid:
    def test_exact_band_distance(self):
        g = _grid([[0.0, 1.0, 0.0]])
        # distance is to the nearest occupied CELL CENTER — quantized by up
        # to half a voxel diagonal (~0.04 m at voxel 0.05), stated in the
        # module docstring; margins must budget for it
        assert g.numpy_query([0.13, 1.0, 0.0]) == pytest.approx(0.03, abs=0.05)
        assert g.numpy_query([0.03, 1.0, 0.0]) < 0.0
        assert g.numpy_query([0.0, 1.0, 0.0]) < 0.0        # at the point itself

    def test_beyond_band_caps(self):
        g = _grid([[0.0, 1.0, 0.0]], band_m=0.30)
        far = g.numpy_query([1.8, 1.0, 1.8])       # no window hit
        assert far == pytest.approx(0.30 - 0.10, abs=1e-6)

    def test_nearest_of_many(self):
        g = _grid([[0.0, 1.0, 0.0], [0.5, 1.0, 0.0]])
        # 0.25 m off the first point, well inside the band
        assert g.numpy_query([0.25, 1.0, 0.0]) == pytest.approx(0.15, abs=0.06)

    def test_torch_matches_numpy_and_diffs(self):
        g = _grid([[0.0, 1.0, 0.0], [1.0, 1.2, 0.3]])
        p = np.array([[0.14, 1.02, 0.05], [0.6, 1.1, 0.1], [0.01, 1.0, 0.0]])
        npv = g.numpy_query(p)
        tp = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        tv = g.torch_query(tp)
        np.testing.assert_allclose(tv.detach().numpy(), npv, atol=2e-2)
        tv.sum().backward()
        # gradient points away from the nearest point (nonzero, finite)
        assert torch.isfinite(tp.grad).all() and float(tp.grad.abs().sum()) > 0


class TestSceneSDFCombine:
    def test_min_combines_obb_and_cloud(self):
        from cinema_traj.common.obb import OBB

        box = OBB(np.array([1.5, 0.0, 1.5]), np.eye(3), np.ones(3) * 0.6)
        g = _grid([[0.0, 1.0, 0.0]])
        sdf = SceneSDF([box], ROOM, cloud=g)
        # near the cloud point: cloud dominates although the box is far
        assert sdf.numpy_query([0.05, 1.0, 0.0]) < 0.0
        # near the box: box dominates (deep inside; extents are FULL sizes)
        assert sdf.numpy_query([1.5, 0.0, 1.5]) < -0.2
        # clean space: both positive
        assert sdf.numpy_query([-1.5, 1.0, -1.5]) > 0.0
        # torch path agrees on sign
        t = sdf(torch.tensor([[0.05, 1.0, 0.0]], dtype=torch.float64))
        assert float(t) < 0.0

    def test_restricted_to_keeps_cloud(self):
        from cinema_traj.common.obb import OBB

        box = OBB(np.array([1.5, 0.0, 1.5]), np.eye(3), np.ones(3) * 0.6)
        sdf = SceneSDF([box], ROOM, cloud=_grid([[0.0, 1.0, 0.0]]))
        pruned = sdf.restricted_to(([-0.1, 0.8, -0.1], [0.1, 1.2, 0.1]))
        assert pruned is not sdf
        assert pruned.cloud is sdf.cloud
        assert pruned.numpy_query([0.05, 1.0, 0.0]) < 0.0


class TestFromScene:
    def test_loader_builds_grid_from_npz(self, tmp_path):
        pts = np.random.default_rng(0).uniform([0, 0, 0], [2, 2, 2], (500, 3))
        f = tmp_path / "cloud.npz"
        np.savez(f, points=pts)
        scene = Scene.from_dict({
            "scene_id": "s",
            "rooms": [{"id": "r", "label": "r", "bounds_min": [0, 0, 0],
                       "bounds_max": [2, 2, 2]}],
            "structures": [], "objects": [], "doors": [],
            "provenance": {"source_map": str(f)},
        })
        g = cloud_sdf_from_scene(scene)
        assert g is not None
        assert g.numpy_query(pts[0]) < 0.0            # a real point is a hit

    def test_missing_source_returns_none(self):
        scene = Scene.from_dict({
            "scene_id": "s",
            "rooms": [{"id": "r", "label": "r", "bounds_min": [0, 0, 0],
                       "bounds_max": [2, 2, 2]}],
            "structures": [], "objects": [], "doors": [],
        })
        assert cloud_sdf_from_scene(scene) is None
