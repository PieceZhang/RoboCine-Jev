"""simview_isaac build-mode regression: USD authoring.

Pure-python tests — build mode needs only numpy, no Isaac Sim. Runs against the
committed demo_room scene (has a subject track) with a synthetic traj_final.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _mini_traj(tmp_path: Path) -> Path:
    frames = []
    for i in range(11):                        # 1.0 s @ 10 Hz
        t = i * 0.1
        frames.append({"segment": "seg", "type": "move_in", "t": t,
                       "pos": [2.0 - 0.1 * t, 1.2, 1.5], "look": [0, 0.6, 0],
                       "fov_deg": 55.0, "colliding": False, "occluded": False,
                       "clearance_m": 1.0})
    data = {"scene_id": "demo_room", "plan_id": "test", "metrics": {},
            "segments": [], "frames": frames}
    p = tmp_path / "traj_final.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_simview_build_produces_valid_stage(tmp_path):
    from tools import simview_isaac as sv        # noqa: import via repo root

    traj = _mini_traj(tmp_path)
    out = tmp_path / "simview.usda"
    meta = sv.build_usda(traj, REPO / "cinema_traj" / "scenes" / "demo_room.json",
                         out, points="none")
    txt = out.read_text(encoding="utf-8")

    assert txt.startswith("#usda 1.0")
    assert txt.count("def Mesh") == 10           # 6 structures + 3 objects + cam hull
    assert 'def Camera "ShotCam"' in txt
    # constant FOV → static focalLength (fixed-lens convention); timeSamples
    # only appear when the trajectory actually zooms (see next test)
    assert "float focalLength = " in txt and "focalLength.timeSamples" not in txt
    assert meta["fov_static"] is True
    assert 'def SphereLight "CamLight"' in txt
    assert 'def Capsule "Body"' in txt           # demo_room has a subject
    assert meta["duration_s"] == 1.0
    assert meta["n_codes"] == 30
    assert txt.count("{") == txt.count("}")      # brace balance sanity


def test_simview_build_zoom_keeps_fov_samples(tmp_path):
    """A trajectory with per-frame FOV (Hitchcock zoom) keeps
    focalLength.timeSamples."""
    from tools import simview_isaac as sv

    traj = _mini_traj(tmp_path)
    data = json.loads(traj.read_text())
    for i, f in enumerate(data["frames"]):
        f["fov_deg"] = 55.0 - 20.0 * (i / 10.0)     # 55° → 35° zoom-in
    traj.write_text(json.dumps(data))
    out = tmp_path / "simview_zoom.usda"
    meta = sv.build_usda(traj, REPO / "cinema_traj" / "scenes" / "demo_room.json",
                         out, points="none")
    txt = out.read_text(encoding="utf-8")
    assert "focalLength.timeSamples" in txt
    assert meta["fov_static"] is False
    assert txt.count("{") == txt.count("}")


def test_simview_build_with_cast(tmp_path):
    """--cast bakes a rigged GLB (posed via glb_pose_bake) as seated figures.

    Uses the committed Xbot.glb in BIND pose (anim=None) with floor alignment —
    deterministic, no KayKit download needed.
    """
    import numpy as np
    from tools import simview_isaac as sv

    xbot = REPO / "datasets" / "simview_assets" / "Xbot.glb"
    if not xbot.exists():                        # asset not fetched on CI
        import pytest
        pytest.skip("Xbot.glb not downloaded")

    cast = tmp_path / "cast.json"
    cast.write_text(json.dumps({"characters": [
        {"id": "Alpha", "glb": str(xbot), "height_m": 1.6,
         "seat": [1.0, 0.0, 1.0], "heading_deg": 180.0, "align": "floor"},
        {"id": "Beta", "glb": str(xbot), "height_m": 1.7,
         "seat": [2.0, 0.4, 1.0], "heading_deg": 0.0, "align": "floor",
         "color": [0.2, 0.4, 0.8]},
    ]}), encoding="utf-8")
    traj = _mini_traj(tmp_path)
    out = tmp_path / "simview.usda"
    meta = sv.build_usda(traj, REPO / "cinema_traj" / "scenes" / "demo_room.json",
                         out, points="none", cast=str(cast))
    txt = out.read_text(encoding="utf-8")

    assert 'def Scope "Cast"' in txt
    assert 'def Xform "Alpha"' in txt and 'def Xform "Beta"' in txt
    assert txt.count('def Mesh "Figure"') == 2   # one baked mesh per character
    assert meta["cast_ids"] == ["Alpha", "Beta"]
    # Beta asked for (0.2, 0.4, 0.8)
    assert "(0.20000, 0.40000, 0.80000)" in txt
    assert txt.count("{") == txt.count("}")


def test_cast_bake_alignment():
    """Floor alignment pins the bbox feet-centre exactly at the local origin;
    seat alignment succeeds (= hips found) on a rigged humanoid."""
    import numpy as np
    from tools import simview_isaac as sv

    xbot = REPO / "datasets" / "simview_assets" / "Xbot.glb"
    if not xbot.exists():
        import pytest
        pytest.skip("Xbot.glb not downloaded")

    c = sv._bake_cast_character({"id": "t", "glb": str(xbot),
                                 "height_m": 1.6, "seat": [3, 0.4, 1],
                                 "heading_deg": 180, "align": "floor"}, 0)
    V = c["V"]
    assert abs(V[:, 1].min()) < 1e-9             # feet exactly at local y=0
    assert abs(V[:, 1].max() - 1.6) < 0.05       # scaled to requested height
    assert c["F"].shape[1] == 3 and V.shape[1] == 3
    seat = sv._bake_cast_character({"id": "t", "glb": str(xbot),
                                    "height_m": 1.6, "seat": [0, 0, 0],
                                    "align": "seat"}, 0)   # must not raise
    assert seat["V"].shape[1] == 3


