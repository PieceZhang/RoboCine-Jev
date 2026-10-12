"""Viewer embedding tests — point-cloud packing roundtrip, map_shift /
floor auto-alignment, and anchors/cloud keys in the generated HTML data."""

import base64
import json
from pathlib import Path

import numpy as np

from cinema_traj.stages.view import build_html, pack_point_cloud

REPO = Path(__file__).resolve().parents[1]


def _unpack(cloud: dict) -> np.ndarray:
    q = np.frombuffer(base64.b64decode(cloud["b64"]), dtype="<i2").reshape(-1, 3)
    return q * cloud["scale"] + np.asarray(cloud["origin"])


class TestPackPointCloud:
    def test_roundtrip_within_quantization(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform([0, -5.4, -1], [7, -0.6, 3], size=(5000, 3))
        cloud = pack_point_cloud(pts, max_pts=2000, voxel=0.05)
        assert cloud["count"] == 2000
        back = _unpack(cloud)
        # every decoded point must be within one quantization step of a real point
        d = np.linalg.norm(back[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
        assert d.max() <= cloud["scale"] * 1.5 + 1e-9

    def test_count_cap_and_small_input(self):
        pts = np.random.default_rng(2).uniform(-3, 3, (50, 3))
        cloud = pack_point_cloud(pts, max_pts=10, voxel=0.05)
        assert cloud["count"] == 10
        assert pack_point_cloud(None) is None
        assert pack_point_cloud(np.zeros((2, 3))) is None

    def test_int16_range(self):
        pts = np.random.default_rng(3).uniform(-40, 40, (100, 3))
        cloud = pack_point_cloud(pts, max_pts=100, voxel=0.05)
        q = np.frombuffer(base64.b64decode(cloud["b64"]), dtype="<i2")
        assert np.abs(q).max() <= 32767


class TestCloudAlignment:
    def test_map_shift_applied(self, tmp_path, monkeypatch):
        # cloud floor at y=-5.4, scene floor at y=0, map_shift lifts it
        pts = np.random.default_rng(4).uniform([0, -5.4, 0], [6, -0.6, 3], (800, 3))
        cloud_file = tmp_path / "cloud.npz"
        np.savez(cloud_file, points=pts)
        scene = {"scene_id": "s", "rooms": [{"bounds_min": [0, -0.05, 0],
                                             "bounds_max": [6, 4.8, 3]}],
                 "provenance": {"source_map": str(cloud_file),
                                "map_shift": [0.0, 5.4, 0.0]}}
        from cinema_traj.stages import view as V
        cloud = V._load_source_cloud(scene, max_pts=300)
        back = _unpack(cloud)
        assert back[:, 1].min() >= -0.6 - cloud["scale"]   # floor ≈ 0

    def test_floor_autoalign_without_map_shift(self, tmp_path):
        pts = np.random.default_rng(5).uniform([0, -5.4, 0], [6, -0.6, 3], (800, 3))
        pts[:100, 1] = -5.4                                # dense floor plane
        cloud_file = tmp_path / "cloud.npz"
        np.savez(cloud_file, points=pts)
        scene = {"scene_id": "s", "rooms": [{"bounds_min": [0, -0.05, 0],
                                             "bounds_max": [6, 4.8, 3]}],
                 "provenance": {"source_map": str(cloud_file)}}
        from cinema_traj.stages import view as V
        cloud = V._load_source_cloud(scene, max_pts=300)
        back = _unpack(cloud)
        assert abs(back[:, 1].min() + 0.05) < 0.35          # aligned to scene floor

    def test_missing_file_returns_none(self):
        from cinema_traj.stages import view as V
        scene = {"scene_id": "s", "provenance": {"source_map": "nope.ply"}}
        assert V._load_source_cloud(scene) is None

    def test_gs_colors_non_ply_returns_none_fast(self):
        # regression: _read_gs_colors used to loop forever on non-PLY inputs
        # (npz/pcd have no end_header marker; read(1) at EOF returns b"")
        import time
        from cinema_traj.stages import view as V
        npz = Path(__file__).parent / "_tmp_nocolor.npz"
        np.savez(npz, points=np.zeros((3, 3)))
        try:
            t0 = time.monotonic()
            assert V._read_gs_colors(npz) is None
            assert time.monotonic() - t0 < 2.0, "non-PLY input must not loop"
        finally:
            npz.unlink()


class TestCloudClearance:
    def test_annotate_distances(self):
        from cinema_traj.stages.view import _annotate_cloud_clearance
        pts = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.5], [0.5, 1.0, 0.0],
                        [5.0, 1.0, 5.0], [5.0, 1.5, 5.0]])   # pack needs ≥4
        cloud = pack_point_cloud(pts, voxel=0.01)
        frames = [{"pos": [0.05, 1.0, 0.0]},            # 5 cm off (0,1,0)
                  {"pos": [2.0, 1.0, 2.0]}]              # ~3.5 m away
        n_near, n_hit = _annotate_cloud_clearance(frames, cloud)
        assert abs(frames[0]["cloud_m"] - 0.05) < 0.02
        assert frames[1]["cloud_m"] > 1.0
        assert n_hit == 1 and n_near == 1

    def test_no_cloud_noop(self):
        from cinema_traj.stages.view import _annotate_cloud_clearance
        frames = [{"pos": [0, 0, 0]}]
        assert _annotate_cloud_clearance(frames, None) == (0, 0)
        assert "cloud_m" not in frames[0]


class TestBuildHtml:
    def test_anchors_and_cloud_embedded(self):
        scene = {"scene_id": "t", "rooms": [], "structures": [], "objects": []}
        traj = {"frames": [{"t": 0.0, "pos": [0, 0, 0], "look": [1, 0, 0],
                            "segment": "s", "fov_deg": 55.0}],
                "metrics": {"frames": 1}}
        cloud = pack_point_cloud(np.random.default_rng(6).uniform(-1, 1, (50, 3)))
        html = build_html(scene, traj,
                          anchors={"subject_01": {"pos": [1, 1, 1], "score": 0.98}},
                          cloud=cloud)
        assert '"anchors":{"subject_01"' in html.replace(" ", "")
        assert "atob" in html and "drawCloud" in html
        # the demo pipeline path (no anchors/cloud) still renders
        assert "const DATA" in build_html(scene, traj)
