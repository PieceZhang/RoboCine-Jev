"""make_gz_world tests — axis conversion, SDF structure, subject track schema.

Pure python (xml.etree), runs on macOS without gz/ROS.
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from make_gz_world import (  # noqa: E402
    _rpy_from_matrix, _world_to_gz_pose, build_world_sdf, subject_track_to_samples,
)

SCENE = {
    "scene_id": "gz_test",
    "rooms": [{"id": "room_01", "label": "room", "bounds_min": [0.0, 0.0, 0.0],
               "bounds_max": [4.0, 2.8, 3.0]}],
    "structures": [
        {"id": "floor", "kind": "floor", "room": "room_01",
         "obb": {"center": [2.0, -0.05, 1.5], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "extents": [4.0, 0.1, 3.0]}},
        {"id": "ceiling", "kind": "ceiling", "room": "room_01",
         "obb": {"center": [2.0, 2.85, 1.5], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "extents": [4.0, 0.1, 3.0]}},
        {"id": "wall_north", "kind": "wall", "room": "room_01",
         "obb": {"center": [2.0, 1.4, 3.0], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "extents": [4.0, 2.8, 0.12]}},
    ],
    "objects": [
        {"id": "sofa_01", "label": "沙发 sofa", "room": "room_01",
         "obb": {"center": [1.0, 0.4, 0.5], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "extents": [1.8, 0.8, 0.9]}, "placement": "freestanding"},
        {"id": "rug_01", "label": "地毯 rug", "room": "room_01",
         "obb": {"center": [2.0, 0.03, 1.5], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 "extents": [2.0, 0.06, 1.4]}, "placement": "freestanding"},
    ],
    "doors": [],
    "subject_track": {"subject_id": "subject_01", "height_m": 1.65,
                      "frames": [{"t": 0.0, "pos": [1.0, 0.0, 1.0], "heading": 0.0},
                                 {"t": 2.0, "pos": [3.0, 0.0, 1.0], "heading": 0.0}],
                      "key_points_local": {"head": 0.92, "chest": 0.72}},
}


class TestAxisConversion:
    def test_position_swap(self):
        # world (x=2.6, y=2.4, z=1.7) → gz (2.6, 1.7, 2.4)
        pose = _world_to_gz_pose([2.6, 2.4, 1.7], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert pose.startswith("2.6000 1.7000 2.4000")

    def test_yaw_rotation_carries(self):
        # world rotation about +y is the same physical spin; expressed in the
        # gz frame (mirrored handedness around up) it becomes -90° about +z
        c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
        Ry = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        pose = _world_to_gz_pose([0, 0, 0], Ry)
        vals = [float(v) for v in pose.split()]
        assert abs(vals[5] + math.pi / 2) < 1e-3       # gz yaw = -90°

    def test_rpy_roundtrip_identity(self):
        r, p, y = _rpy_from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert abs(r) < 1e-9 and abs(p) < 1e-9 and abs(y) < 1e-9


class TestWorldSdf:
    def test_structure_and_models(self):
        sdf = build_world_sdf(SCENE, "gz_test")
        root = ET.fromstring(sdf)
        world = root.find("world")
        assert world.get("name") == "gz_test"
        names = {m.get("name") for m in world.findall("model")}
        assert {"ground_plane", "room_floor", "wall_north", "sofa_01", "rug_01",
                "camera_drone", "subject_01"} <= names
        assert "ceiling" not in names                          # open top
        # wall_north: world z=3.0 → gz y=3.0, y-center 1.4 → gz z=1.4
        wn = next(m for m in world.findall("model") if m.get("name") == "wall_north")
        assert wn.find("pose").text.startswith("2.0000 3.0000 1.4000")
        # rug is visual-only (flat), sofa has collision
        sofa = next(m for m in world.findall("model") if m.get("name") == "sofa_01")
        rug = next(m for m in world.findall("model") if m.get("name") == "rug_01")
        assert sofa.find("link/collision") is not None
        assert rug.find("link/collision") is None

    def test_no_model_includes(self):
        import re
        sdf = re.sub(r"<!--.*?-->", "", build_world_sdf(SCENE, "gz_test"), flags=re.S)
        assert "model://" not in sdf and "<include" not in sdf   # fully offline


class TestSubjectTrack:
    def test_schema_matches_traj_loader(self):
        doc = subject_track_to_samples(SCENE["subject_track"], step_s=0.5)
        assert doc["frame"]["up"] == "+Y"
        ss = doc["samples"]
        assert ss[0]["t"] == 0.0 and ss[-1]["t"] == 2.0
        assert ss[0]["x"] == 1.0 and ss[-1]["x"] == 3.0
        # loadable by the bridge's rclpy-free loader
        bridge = REPO / "ros2_ws" / "cinema_traj_bridge"
        sys.path.insert(0, str(bridge))
        from cinema_traj_bridge import traj_loader  # noqa: E402

        out = Path(__file__).parent / "_gz_subject_tmp.json"
        out.write_text(json.dumps(doc), encoding="utf-8")
        try:
            samples = traj_loader.load_samples(str(out))
            assert len(samples) >= 2 and samples[0]["yaw_deg"] == 0.0
        finally:
            out.unlink()
