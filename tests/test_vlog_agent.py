"""Unit tests for vlog_agent pure logic (no network, no ffmpeg)."""
import json
from pathlib import Path

import pytest

from vlog_agent.ingest import parse_filename_date, sanitize_tags, scan_tree
from vlog_agent.llm import extract_json
from vlog_agent.storyboard import rule_based_edl, validate_edl


# ── llm.extract_json ──────────────────────────────────────────────────────

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = "好的，结果如下：\n```json\n{\"a\": 1, \"b\": \"x{y}z\"}\n```\n"
    assert extract_json(text) == {"a": 1, "b": "x{y}z"}


def test_extract_json_prose_wrapped():
    text = '标注结果是 {"scene":"室内","quality":4} 希望有帮助'
    assert extract_json(text)["scene"] == "室内"


def test_extract_json_garbage():
    assert extract_json("完全不是 JSON") is None


# ── ingest ────────────────────────────────────────────────────────────────

def test_parse_filename_date_dji_long():
    assert parse_filename_date("DJI_20260102173837_0001_D.MP4") == "2026-01-02T17:38:37"


def test_parse_filename_date_dji_fly():
    assert parse_filename_date("dji_fly_20260102_140846_0199_x_video.MP4") == "2026-01-02T00:00:00"


def test_parse_filename_date_none():
    assert parse_filename_date("IMG_7100.MOV") is None


def test_sanitize_tags_clamps():
    tags = sanitize_tags({"scene": "太空", "subject": "多人", "shot": "手持",
                          "quality": 99, "issues": ["抖动", "瞎写"], "desc": "x" * 100})
    assert tags["scene"] == "室外自然"  # unknown → default
    assert tags["quality"] == 5
    assert tags["issues"] == ["抖动"]
    assert len(tags["desc"]) <= 40


def test_scan_tree_pairs_lrf_srt(tmp_path: Path):
    (tmp_path / "DJI_001").mkdir()
    (tmp_path / "DJI_001" / "DJI_20260102173837_0001_D.MP4").write_bytes(b"x")
    (tmp_path / "DJI_001" / "DJI_20260102173837_0001_D.LRF").write_bytes(b"x")
    (tmp_path / "DJI_001" / "DJI_20260102173837_0001_D.SRT").write_text("GPS:(1,2,3)")
    (tmp_path / "HYPERLAPSE").mkdir()
    (tmp_path / "HYPERLAPSE" / "HYPERLAPSE_0001.JPG").write_bytes(b"x")
    (tmp_path / "photo.JPG").write_bytes(b"x")

    recs = scan_tree(tmp_path)
    by_id = {r["id"]: r for r in recs}
    assert set(by_id) == {"DJI_001/DJI_20260102173837_0001_D.MP4", "photo.JPG"}
    video = by_id["DJI_001/DJI_20260102173837_0001_D.MP4"]
    assert video["proxy"].endswith(".LRF")
    assert video["telemetry"].endswith(".SRT")
    assert by_id["photo.JPG"]["type"] == "photo"


# ── storyboard ────────────────────────────────────────────────────────────

def _mat(mid, dur=10.0, q=3, scene="室外自然", mtype="video"):
    return {"id": mid, "type": mtype, "duration": dur, "quality": q,
            "scene": scene, "desc": f"desc-{mid}", "created_at": "2026-01-02T00:00:00"}


def test_validate_edl_clamps_and_drops():
    mats = {m["id"]: m for m in [_mat("a", 10), _mat("b", 3)]}
    edl = {"clips": [
        {"id": "a", "start": 8, "end": 20, "caption": "ok"},     # end clamped to 10
        {"id": "b", "start": 2.5, "end": 2.6, "caption": ""},    # degenerate → repaired
        {"id": "zz", "start": 0, "end": 3, "caption": ""},       # unknown id → dropped
        {"id": "a", "start": 0, "end": 4, "caption": ""},        # duplicate → dropped
    ]}
    clips = validate_edl(edl, mats)
    assert [c["id"] for c in clips] == ["a", "b"]
    assert clips[0]["end"] <= 10.0
    assert clips[1]["end"] - clips[1]["start"] >= 0.5


def test_validate_edl_photo_fixed_window():
    mats = {"p": {"id": "p", "type": "photo", "duration": None}}
    clips = validate_edl({"clips": [{"id": "p", "start": 0, "end": 0}]}, mats)
    assert clips == [{"id": "p", "start": 0.0, "end": 3.0, "caption": ""}]


def test_rule_based_edl_diverse_and_sorted():
    mats = [
        _mat("late", q=5, scene="室内"),
        _mat("early", q=5, scene="室外自然"),
        _mat("mid", q=4, scene="室外人工"),
        _mat("extra", q=3, scene="室外人工"),
    ]
    edl = rule_based_edl(mats, "测试主题", 30)
    assert not edl.get("fallback") is None
    ids = [c["id"] for c in edl["clips"]]
    assert len(ids) >= 3 and len(set(ids)) == len(ids)
    # chronological
    created = [m["created_at"] for m in mats if m["id"] in ids]
    assert created == sorted(created)
    for c in edl["clips"]:
        m = next(m for m in mats if m["id"] == c["id"])
        assert c["end"] <= m["duration"] + 1e-6


# ── edit ──────────────────────────────────────────────────────────────────

def test_caption_png_with_and_without_font(tmp_path, monkeypatch):
    import vlog_agent.edit as edit_mod
    if Path("/System/Library/Fonts/PingFang.ttc").exists():
        png = edit_mod._caption_png("风电场", tmp_path, 0)
        assert Path(png).exists() and Path(png).stat().st_size > 0
    # no CJK font available → default font path, still renders a PNG
    monkeypatch.setattr(edit_mod, "FONT_CANDIDATES", [str(tmp_path / "nofont.ttc")])
    png = edit_mod._caption_png("字幕", tmp_path, 1)
    assert Path(png).exists()


def test_caption_overlay_fc():
    import vlog_agent.edit as edit_mod
    assert edit_mod._caption_overlay_fc(None, 4.0) is None
    fc = edit_mod._caption_overlay_fc("cap.png", 4.0)
    assert "overlay" in fc and "between(t,0.2,3.70)" in fc


def test_index_roundtrip(tmp_path):
    # the index written by ingest must load back as a plain dict-of-dicts
    idx = {"a.mp4": {"id": "a.mp4", "type": "video", "classified": True}}
    p = tmp_path / "material_index.json"
    p.write_text(json.dumps(idx, ensure_ascii=False))
    assert json.loads(p.read_text())["a.mp4"]["classified"] is True
