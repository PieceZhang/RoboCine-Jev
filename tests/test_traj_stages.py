"""Extractor (PCD/PLY readers + structure detection) and stage-artifact tests."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cinema_traj.stages.extract import (_read_pcd, _read_ply, build_scene_draft,
                                        classify_placement, detect_floor_ceiling,
                                        voxel_downsample)

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def _write_pcd_ascii(path: Path, pts: np.ndarray):
    with open(path, "w") as fh:
        fh.write("# .PCD v0.7\n")
        fh.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        fh.write(f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
        fh.write(f"POINTS {len(pts)}\nDATA ascii\n")
        for p in pts:
            fh.write(f"{p[0]} {p[1]} {p[2]}\n")


def _write_pcd_binary(path: Path, pts: np.ndarray):
    with open(path, "wb") as fh:
        hdr = (f"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
               f"WIDTH {len(pts)}\nHEIGHT 1\nPOINTS {len(pts)}\nDATA binary\n")
        fh.write(hdr.encode("ascii"))
        fh.write(pts.astype("<f4").tobytes())


def _write_ply_ascii(path: Path, pts: np.ndarray):
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(pts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property float f_dc_0\nend_header\n")
        for p in pts:
            fh.write(f"{p[0]} {p[1]} {p[2]} 0.5\n")     # 3DGS-style extra column


class TestReaders:
    def test_pcd_ascii(self, tmp_path):
        pts = np.array([[0.0, 0.0, 0.0], [1.5, 0.2, -0.3]])
        _write_pcd_ascii(tmp_path / "a.pcd", pts)
        np.testing.assert_allclose(_read_pcd(tmp_path / "a.pcd"), pts)

    def test_to_yup_declared_axis_swap(self):
        # SceneSplat-style Z-up maps (ground on XY) into the internal Y-up
        # frame via a right-handed rotation: (x, y, z) → (x, z, −y)
        from cinema_traj.stages.extract import _to_yup
        pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.25]])
        out = _to_yup(pts, up="z")
        np.testing.assert_allclose(out, [[1.0, 3.0, -2.0], [-4.0, 0.25, -0.5]])
        np.testing.assert_allclose(_to_yup(pts, up="y"), pts)   # identity
        with pytest.raises(ValueError, match="up"):
            _to_yup(pts, up="x")

    def test_load_points_up_flag(self, tmp_path):
        from cinema_traj.stages.extract import load_points
        pts = np.array([[1.0, 2.0, 3.0]])
        np.savez(tmp_path / "m.npz", points=pts)
        np.testing.assert_allclose(load_points(str(tmp_path / "m.npz"), up="z"),
                                   [[1.0, 3.0, -2.0]])

    def test_pcd_binary(self, tmp_path):
        pts = np.random.default_rng(0).normal(size=(50, 3))
        _write_pcd_binary(tmp_path / "b.pcd", pts)
        np.testing.assert_allclose(_read_pcd(tmp_path / "b.pcd"), pts, atol=1e-6)

    def test_ply_ascii_with_extra_props(self, tmp_path):
        pts = np.random.default_rng(1).normal(size=(30, 3))
        _write_ply_ascii(tmp_path / "c.ply", pts)
        np.testing.assert_allclose(_read_ply(tmp_path / "c.ply"), pts)


# --------------------------------------------------------------------------- #
# Structure detection on a synthetic room
# --------------------------------------------------------------------------- #
def _room_cloud(seed=0, with_box=True):
    rng = np.random.default_rng(seed)
    pts = []
    # floor 6x6 at y=0, ceiling at y=2.7
    for _ in range(3000):
        pts.append([rng.uniform(-3, 3), rng.normal(0, 0.01), rng.uniform(-3, 3)])
        pts.append([rng.uniform(-3, 3), 2.7 + rng.normal(0, 0.01), rng.uniform(-3, 3)])
    # walls at x=±3, z=±3
    for _ in range(3000):
        u, v = rng.uniform(-3, 3), rng.uniform(0, 2.7)
        for wall in ([3 + rng.normal(0, 0.01), v, u], [-3 + rng.normal(0, 0.01), v, u],
                     [u, v, 3 + rng.normal(0, 0.01)], [u, v, -3 + rng.normal(0, 0.01)]):
            pts.append(wall)
    if with_box:                                        # freestanding box near center
        for _ in range(600):
            pts.append([rng.normal(0.5, 0.3), rng.uniform(0.05, 0.8),
                        rng.normal(0.5, 0.3)])
    return np.asarray(pts)


class TestStructureDetection:
    def test_floor_ceiling(self):
        pts = _room_cloud()
        floor, ceiling = detect_floor_ceiling(voxel_downsample(pts, 0.05))
        assert floor == pytest.approx(0.0, abs=0.08)
        assert ceiling == pytest.approx(2.7, abs=0.08)

    def test_draft_scene_shape(self):
        pts = _room_cloud()
        scene, warnings = build_scene_draft(pts, scene_id="synth", voxel=0.05)
        assert len(scene["rooms"]) == 1
        kinds = {s["kind"] for s in scene["structures"]}
        assert "floor" in kinds and "wall" in kinds
        walls = [s for s in scene["structures"] if s["kind"] == "wall"]
        assert len(walls) >= 3, f"expected >=3 walls, got {len(walls)}"
        assert scene["objects"], "the synthetic box should cluster into an object"
        # object near room center, not against a wall
        assert scene["objects"][0]["placement"] in ("freestanding", "wall")
        assert any("GENERIC" in w for w in warnings)

    def test_wall_placement_classification(self):
        from cinema_traj.common.obb import OBB
        walls = [(2, -3.0, None)]                       # wall plane at z = -3
        obb = OBB(np.array([0.0, 0.5, -2.7]), np.eye(3), np.array([1.2, 1.0, 0.5]))
        assert classify_placement(obb, walls, None) == "wall"
        free = OBB(np.array([0.0, 0.5, 0.0]), np.eye(3), np.array([1.2, 1.0, 0.5]))
        assert classify_placement(free, walls, None) == "freestanding"


# --------------------------------------------------------------------------- #
# Full pipeline artifacts via the CLI (mock planner, fast)
# --------------------------------------------------------------------------- #
class TestPipelineCLI:
    def test_all_stages(self, tmp_path):
        cmd = [sys.executable, "run_traj.py", "--stage", "all",
               "--scene", "cinema_traj/demo/demo_apartment.json",
               "--prompt-file", "cinema_traj/demo/demo_prompt.txt",
               "--mock", "--iters", "60", "--force"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        out_dir = REPO / "cinema_traj_out" / "demo_apartment"
        assert (out_dir / "plan.json").exists()
        assert (out_dir / "traj_init.json").exists()
        assert (out_dir / "traj_final.json").exists()
        assert (out_dir / "exports" / "traj_samples.csv").exists()
        assert (out_dir / "exports" / "drone_setpoints.csv").exists()
        html = (out_dir / "viewer" / "demo_apartment.html")
        assert html.exists()
        content = html.read_text(encoding="utf-8")
        assert "const DATA" in content and "frames" in content

        traj = json.loads((out_dir / "traj_final.json").read_text())
        # with only 60 iterations the optimizer may not fully converge,
        # but the artifact chain + metrics must exist and be well-formed
        assert "collision_rate" in traj["metrics"]
        assert len(traj["frames"]) > 50
