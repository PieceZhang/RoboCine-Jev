"""Subject-injection tool + demo-room PCD generator tests."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from add_subject_track import (build_subject_track, parse_waypoints,  # noqa: E402
                               static_frame, walk_frames)
from make_demo_room_pcd import build_demo_room, write_pcd_binary  # noqa: E402

from cinema_traj.common.scene import load_scene  # noqa: E402
from cinema_traj.stages.extract import _read_pcd  # noqa: E402


class TestParseWaypoints:
    def test_basic(self):
        assert parse_waypoints("-2,0 -2,2 1,2") == [(-2.0, 0.0), (-2.0, 2.0), (1.0, 2.0)]

    def test_rejects_single_and_malformed(self):
        with pytest.raises(ValueError):
            parse_waypoints("1,2")
        with pytest.raises(ValueError):
            parse_waypoints("1,2 oops")


class TestWalkFrames:
    def test_frame_count_duration_and_speed(self):
        # 2 m straight leg at 1 m/s + 3 s pause at the end waypoint
        frames = walk_frames([(-2.0, 0.0), (0.0, 0.0)], speed=1.0, height=1.75,
                             pauses=[0.0, 3.0])
        assert frames[-1]["t"] == pytest.approx(5.0, abs=0.15)   # 2 s walk + 3 s pause
        assert all(b["t"] - a["t"] <= 0.1 + 1e-9 for a, b in zip(frames, frames[1:]))
        # actual walking speed from consecutive frames
        d = [math.dist(a["pos"], b["pos"]) / (b["t"] - a["t"])
             for a, b in zip(frames, frames[1:]) if b["t"] - a["t"] > 0]
        walking = [v for v in d if v > 1e-6]
        assert max(walking) <= 1.0 + 0.15                        # ≈ speed, never above

    def test_heading_is_travel_direction(self):
        # leg toward +Z ⇒ heading = atan2(dz, dx) = +π/2 (0 → +X convention)
        frames = walk_frames([(0.0, 0.0), (0.0, 2.0)], speed=1.0, height=1.75,
                             pauses=[0.0, 0.0])
        assert frames[0]["heading"] == pytest.approx(math.pi / 2, abs=1e-9)
        # leg toward +X ⇒ heading 0
        frames = walk_frames([(0.0, 0.0), (2.0, 0.0)], speed=1.0, height=1.75,
                             pauses=[0.0, 0.0])
        assert all(f["heading"] == pytest.approx(0.0, abs=1e-9) for f in frames)

    def test_pause_holds_position(self):
        frames = walk_frames([(0.0, 0.0), (1.0, 0.0)], speed=1.0, height=1.75,
                             pauses=[0.0, 2.0])
        tail = frames[-20:]
        assert all(f["pos"] == [1.0, 0.0, 0.0] for f in tail)

    def test_feet_on_floor(self):
        frames = walk_frames([(0.0, 0.0), (1.0, 1.0)], 1.0, 1.9, [0, 0])
        assert all(f["pos"][1] == 0.0 for f in frames)


class TestStaticMode:
    def test_single_point(self):
        frames = static_frame(1.5, -0.5, 30.0)
        assert len(frames) == 1
        assert frames[0]["pos"] == [1.5, 0.0, -0.5]
        assert frames[0]["heading"] == pytest.approx(math.radians(30.0))

    def test_build_subject_track_static(self):
        class A:
            static, waypoints, heading = "1.0,1.5", None, 30.0
            speed = height = 1.75
            pause = None
            subject_id = "dummy"
        track = build_subject_track(A())
        assert track["_mode"] == "static"
        assert track["subject_id"] == "dummy"
        assert len(track["frames"]) == 1
        assert track["key_points_local"]["head"] == 0.92


class TestSceneWriteback:
    def test_written_scene_loads(self, tmp_path):
        scene = {"scene_id": "t", "rooms": [
            {"id": "r1", "label": "r1", "bounds_min": [-3, 0, -3], "bounds_max": [3, 3, 3]}],
            "structures": [], "objects": [], "doors": []}
        p = tmp_path / "scene.json"
        p.write_text(json.dumps(scene), encoding="utf-8")

        cmd = [sys.executable, str(REPO / "scripts" / "add_subject_track.py"),
               "--scene", str(p), "--waypoints", "-2,0 -2,2 1,2",
               "--speed", "1.0", "--pause", "0,3,0"]
        import subprocess
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr

        loaded = load_scene(p)                         # validates via Scene.validate()
        assert loaded.subject is not None
        assert loaded.subject.subject_id == "subject_01"
        assert loaded.subject.t1 > 5.0                 # 2+√8 s walk + 3 s pause
        # backup created next to the scene
        baks = list(tmp_path.glob("scene.json.presubject-*"))
        assert baks, "backup_once should have produced a backup"


class TestMakeDemoRoomPcd:
    def test_roundtrip_through_read_pcd(self, tmp_path):
        pts = build_demo_room(seed=0)
        p = tmp_path / "demo_room.pcd"
        write_pcd_binary(p, pts)
        back = _read_pcd(p)
        assert len(back) == len(pts)
        np.testing.assert_allclose(back, pts, atol=1e-6)

    def test_geometry_is_the_expected_room(self):
        pts = build_demo_room(seed=1)
        assert pts[:, 0].min() >= -3.1 and pts[:, 0].max() <= 3.1
        assert pts[:, 1].min() >= -0.1 and pts[:, 1].max() <= 2.8
        assert pts[:, 2].min() >= -3.1 and pts[:, 2].max() <= 3.1
        # the three boxes contribute surface points above 0.7 m (tall target box)
        assert (pts[:, 1] > 0.7).sum() > 500

    def test_extract_detects_structure(self):
        from cinema_traj.stages.extract import build_scene_draft
        scene, warnings = build_scene_draft(build_demo_room(seed=0),
                                            scene_id="demo_room", voxel=0.05)
        kinds = {s["kind"] for s in scene["structures"]}
        assert "floor" in kinds and "wall" in kinds and "ceiling" in kinds
        assert scene["objects"], "the boxes should cluster into objects"
        assert any("GENERIC" in w for w in warnings)
