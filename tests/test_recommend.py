"""Recommendation adapter tests — tolerant parser, zh→atomic mapping, and
plan generation against the REAL upstream output shape (JS callback literal).

RAW_LITERALE embeds the actual datasets/CameraMovementsRecommand_output_json/
results.json content; when the file is present on disk we additionally parse
it live so format drift is caught early.
"""

from pathlib import Path

import json

import pytest

from cinema_traj.common.scene import Scene
from cinema_traj.core.plan_schema import validate_plan
from cinema_traj.stages.recommend import (
    canonicalize, map_movement, parse_raw_recommendation, to_plan,
)

REPO = Path(__file__).resolve().parents[1]
RAW_FILE = REPO / "datasets/CameraMovementsRecommand_output_json/results.json"
BATCH_FILE = REPO / "datasets/CameraMovementsRecommand_output_json/possible_results.json"

RAW_LITERAL = (
    '{"func":"callbackRecommend","params":[{"jimeng-2026-05-08-5028-横屏-女生-办公居家酒店-'
    '推镜特写-2.mp4", tags:{"场景类别：室内/居家环境_酒店、商场、广场（机场车站）_工作环境, '
    '被拍摄主体：单人/中青年女性, 场景特征：狭小封闭空间；运镜类型：前推运镜", '
    '“屏幕方向“:"横屏", “初始拍摄角度“： 水平， “运镜方向“： 前, "运镜节奏": 中, '
    '“运镜难度“： 1星"}}]}'
)

SUBJECT_SCENE = Scene.from_dict({
    "scene_id": "rec_test",
    "rooms": [{"id": "room", "label": "room",
               "bounds_min": [-4.0, 0.0, -4.0], "bounds_max": [4.0, 3.0, 4.0]}],
    "structures": [],
    "objects": [{"id": "desk_01", "label": "desk", "room": "room",
                 "obb": {"center": [1.0, 0.4, 1.0], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "extents": [1.4, 0.8, 0.7]}, "placement": "freestanding"}],
    "doors": [],
    "subject_track": {"subject_id": "subject_01", "height_m": 1.65,
                      "frames": [{"t": 0.0, "pos": [-1.0, 0.0, 0.0], "heading": 0.5}],
                      "key_points_local": {"head": 0.92, "chest": 0.72}},
})


class TestParse:
    def test_parses_the_real_literal(self):
        parsed = parse_raw_recommendation(RAW_LITERAL)
        assert parsed["video"].endswith("推镜特写-2.mp4")
        f = parsed["fields"]
        assert "前推运镜" in f["运镜类型"]
        assert "单人" in f["被拍摄主体"]
        assert f["屏幕方向"] == "横屏"
        assert f["初始拍摄角度"] == "水平"
        assert f["运镜方向"] == "前"
        assert f["运镜节奏"] == "中"
        assert "1" in f["运镜难度"]

    @pytest.mark.skipif(not RAW_FILE.exists(), reason="datasets file not present")
    def test_parses_live_file(self):
        parsed = parse_raw_recommendation(RAW_FILE.read_text(encoding="utf-8"))
        assert parsed["fields"]["运镜类型"]

    def test_legal_json_fast_path(self):
        raw = ('{"func":"callbackRecommend","params":[{"video":"a.mp4",'
               '"tags":{"运镜类型":"环绕运镜","movement_speed":0.8,"start_angle_deg":-15}}]}')
        parsed = parse_raw_recommendation(raw)
        assert parsed["fields"]["运镜类型"] == "环绕运镜"
        assert parsed["fields"]["movement_speed"] == "0.8"

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            parse_raw_recommendation("not a recommendation at all")


class TestBatch:
    # the 9-video batch in possible_results.json: (index, movement, atomic)
    BATCH_CASES = [
        (0, "前推运镜_希区柯克", "move_in"),
        (1, "前推运镜", "move_in"),
        (2, "横摇_主角跟随", "track_follow"),
        (3, "环视运镜", "orbit_quarter"),
        (4, "横摇_横移运镜", "pan_left"),
        (5, "后拉运镜", "move_out"),
        (6, "横摇", "pan_left"),
        (7, "前推运镜_过肩镜头", "track_lead"),
        (8, "环视运镜", "orbit_quarter"),
    ]

    @pytest.mark.skipif(not BATCH_FILE.exists(), reason="batch file not present")
    @pytest.mark.parametrize("idx,movement,atomic", BATCH_CASES)
    def test_batch_entries(self, idx, movement, atomic):
        parsed = parse_raw_recommendation(
            BATCH_FILE.read_text(encoding="utf-8"), index=idx)
        assert parsed["fields"]["运镜类型"] == movement
        assert map_movement(parsed["fields"]["运镜类型"]) == atomic
        assert parsed["video"].endswith(".mp4")

    def test_legacy_video_map_shape(self):
        raw = json.dumps({"a.mp4": "运镜类型：后拉运镜；被拍摄主体：单人",
                          "b.mp4": "运镜类型：环视运镜"})
        p0 = parse_raw_recommendation(raw, index=0)
        assert p0["video"] == "a.mp4" and p0["fields"]["运镜类型"] == "后拉运镜"
        p1 = parse_raw_recommendation(raw, index=1)
        assert p1["video"] == "b.mp4" and map_movement(p1["fields"]["运镜类型"]) == "orbit_quarter"


class TestCanonicalize:
    def test_fields_and_defaults(self):
        rec = canonicalize(parse_raw_recommendation(RAW_LITERAL), source="results.json")
        assert rec["schema"] == "cinema_traj/recommendation/v1"
        assert rec["movement_type_atomic"] == "move_in"
        assert rec["movement_speed"] == 0.6            # 节奏「中」→ 默认
        assert rec["start_angle_deg"] == 0.0           # 「水平」→ 默认
        assert set(rec["defaults_filled"]) == {"movement_speed", "start_angle_deg"}
        assert rec["difficulty"] == 1
        assert "室内" in rec["scene_category"][0]
        assert "单人" in rec["subject"]

    def test_upstream_numbers_win_over_defaults(self):
        raw = RAW_LITERAL.replace("1星", "1星\", \"movement_speed\": 0.9, "
                                        "\"start_angle_deg\": -25")
        rec = canonicalize(parse_raw_recommendation(raw))
        assert rec["movement_speed"] == 0.9
        assert rec["start_angle_deg"] == -25.0
        assert rec["defaults_filled"] == []


class TestMapping:
    @pytest.mark.parametrize("zh,atomic", [
        ("前推运镜", "move_in"), ("拉远镜头", "move_out"), ("环绕跟随拍摄", "track_orbit"),
        ("跟拍", "track_follow"), ("过肩跟拍", "track_lead"), ("360度环绕", "orbit_full"),
        ("半圈环绕", "orbit_half"), ("环绕运镜", "orbit_quarter"), ("环视运镜", "orbit_quarter"),
        ("环顾四周", "orbit_quarter"), ("升降镜头", "crane"),
        ("推拉变焦", "zoom_in_out"), ("向左摇镜", "pan_left"), ("横摇", "pan_left"),
        ("横移运镜", "pan_left"), ("俯视下摇", "tilt_down"),
        ("固定机位", "static"),
    ])
    def test_table(self, zh, atomic):
        assert map_movement(zh) == atomic

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="未识别"):
            map_movement("量子运镜")


class TestPlan:
    def test_push_in_on_subject_validates(self):
        rec = canonicalize(parse_raw_recommendation(RAW_LITERAL))
        plan = to_plan(rec, SUBJECT_SCENE)
        assert plan["object_sequence"] == ["subject_01"]
        assert plan["steps"][1]["kind"] == "object_traj"
        assert plan["steps"][1]["type"] == "move_in"
        assert 2.0 <= plan["steps"][1]["duration_s"] <= 20.0
        # anchor close-up: the video filename says 特写
        assert plan["steps"][0]["distance"] == "close"
        assert validate_plan(plan, SUBJECT_SCENE) == []

    def test_push_in_on_subject_builds(self):
        from cinema_traj.core.trajectories import DollySegment
        from cinema_traj.stages.build import build_trajectory
        rec = canonicalize(parse_raw_recommendation(RAW_LITERAL))
        plan = to_plan(rec, SUBJECT_SCENE)
        segments, _ = build_trajectory(SUBJECT_SCENE, plan)
        assert len(segments) == 1 and isinstance(segments[0], DollySegment)
        seg = segments[0]
        assert seg.target_id == "subject_01"
        assert seg.target_mode == "subject_head"      # look keeps tracking her head
        # dolly aims at the chest key point at t_start, not the feet
        assert seg.fix["center"][1] == pytest.approx(0.72 * 1.65, abs=1e-6)

    def test_track_movement_needs_subject(self):
        rec = canonicalize(parse_raw_recommendation(RAW_LITERAL))
        rec["movement_type"] = "跟拍"
        rec["movement_type_atomic"] = "track_follow"
        scene = Scene.from_dict({k: v for k, v in SUBJECT_SCENE.to_dict().items()
                                 if k != "subject_track"})
        with pytest.raises(ValueError, match="subject_track"):
            to_plan(rec, scene)

    def test_falls_back_to_object_without_subject(self):
        rec = canonicalize(parse_raw_recommendation(RAW_LITERAL))
        rec["subject"] = ""
        scene = Scene.from_dict({k: v for k, v in SUBJECT_SCENE.to_dict().items()
                                 if k != "subject_track"})
        plan = to_plan(rec, scene)
        assert plan["object_sequence"] == ["desk_01"]
        assert validate_plan(plan, scene) == []
