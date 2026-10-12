"""Recommendation adapter: upstream 运镜推荐 results.json → canonical
recommendation JSON + validated 1-3-1 atomic plan.

The upstream ``callbackRecommend`` output is a JS callback literal, not JSON:
bare keys (``tags:``), full-width punctuation (“”：，), one field's value
smuggling several ``key：value`` pairs inside a single string, and the
contract fields ``movement_speed`` / ``start_angle_deg`` missing entirely.
``parse_raw_recommendation`` reads it WITHOUT eval: normalize punctuation,
then slice known-key occurrences positionally. When upstream starts emitting
legal JSON the fast path picks it up unchanged.

Downstream contract (``cinema_traj/recommendation/v1``):

    movement_type_atomic  zh 运镜 → atomic type (move_in, orbit_half, …)
    movement_speed        m/s — used to size 段时长; defaulted from 运镜节奏
    start_angle_deg       initial gimbal pitch (+up); defaulted from 初始拍摄角度
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common import constants as C
from cinema_traj.common.scene import Scene, load_scene
from cinema_traj.core.plan_schema import normalize_plan, validate_plan
from vlm_intention.common.atomic_io import atomic_write_json
from vlm_intention.common.config import load_config

SCHEMA = "cinema_traj/recommendation/v1"

# --------------------------------------------------------------------------- #
# 中文运镜 → 原子类型（长的关键字优先；track_* 需要 scene 里挂了 subject_track）
# --------------------------------------------------------------------------- #
_MOVEMENT_ZH_MAP: list[tuple[tuple, str]] = [
    (("环绕跟随", "边环绕边跟", "跟拍环绕", "跟随环绕"), "track_orbit"),
    (("过肩", "前瞻机位", "lead"), "track_lead"),
    (("跟拍", "跟随", "跟踪", "跟镜", "跟移"), "track_follow"),
    (("前推", "推近", "推进", "移近", "推镜", "缓推", "快推", "推到"), "move_in"),
    (("后拉", "拉远", "拉出", "退镜", "远离", "拉镜"), "move_out"),
    (("全周环绕", "360度", "360°", "整圈", "全周"), "orbit_full"),
    (("半圈环绕", "180度", "180°", "半圈", "绕半"), "orbit_half"),
    (("环绕", "环视", "环顾", "旋转运镜", "绕行", "旋转"), "orbit_quarter"),
    (("升降", "摇臂", "升降弧线"), "crane"),
    (("推拉变焦", "变焦"), "zoom_in_out"),
    (("横移", "左移", "右移", "平移", "滑移", "左移摇", "平移摇", "truck", "track"), "lateral_truck"),
    (("向左摇", "左摇", "横摇"), "pan_left"),
    (("摇镜", "右摇", "向右摇"), "pan_right"),
    (("上仰", "仰视", "上摇", "竖摇"), "tilt_up"),
    (("俯视", "下俯", "下摇"), "tilt_down"),
    (("固定", "定机位", "静止", "定格"), "static"),
]

# 节奏 → 备用 movement_speed（上游缺字段时的缺省，m/s）
_RHYTHM_SPEED = {"慢": 0.3, "缓": 0.3, "中": 0.6, "快": 1.0, "急": 1.2}
_RHYTHM_DURATION_SCALE = {"慢": 1.4, "缓": 1.4, "中": 1.0, "快": 0.75, "急": 0.65}

# 初始拍摄角度 → start_angle_deg（云台俯仰，正 = 仰）
_START_ANGLE_ZH = {"水平": 0.0, "平视": 0.0, "仰视": 30.0, "仰拍": 30.0,
                   "俯视": -30.0, "俯拍": -30.0, "低角度": 20.0, "高角度": -20.0}

# 已知字段名（解析切片的锚点）；英文合同字段也在内，兼容上游将来改好
_KNOWN_KEYS = ["场景类别", "被拍摄主体", "场景特征", "运镜类型", "屏幕方向",
               "初始拍摄角度", "运镜方向", "运镜节奏", "运镜难度",
               "movement_speed", "start_angle_deg"]

_PUNCT = {"“": '"', "”": '"', "‘": '"', "’": '"', "：": ":", "，": ",",
          "　": " "}


def _normalize(text: str) -> str:
    for k, v in _PUNCT.items():
        text = text.replace(k, v)
    return text


def parse_raw_recommendation(raw: str, index: int = 0) -> dict:
    """Tolerant reader for recommendation output (no eval).

    Order: legal-JSON fast path first (the repaired emitter format, incl.
    multi-video batches — ``index`` picks the entry), then the legacy
    {video: "k：v；…"} mapping, then the regex slicer for the raw JS
    callback literal. Returns {video, fields{key: value}}.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, dict) and isinstance(obj.get("params"), list) and obj["params"]:
        entry = obj["params"][max(0, min(index, len(obj["params"]) - 1))]
        tags = entry.get("tags", {})
        if isinstance(tags, dict) and tags:
            fields = {str(k): str(v) for k, v in tags.items()
                      if not isinstance(v, (dict, list))}
            return {"video": entry.get("video"), "fields": fields}

    if isinstance(obj, dict) and obj and all(isinstance(v, str) for v in obj.values()):
        items = list(obj.items())                  # legacy video → tag string
        video, tagstr = items[max(0, min(index, len(items) - 1))]
        fields = {}
        for part in re.split(r"[;；]", _normalize(tagstr)):
            if ":" in part:
                k, v = part.split(":", 1)
                fields[k.strip()] = v.strip()
        if fields:
            return {"video": video, "fields": fields}

    text = _normalize(raw)
    video = None
    m = re.search(r'"([^"]+?\.(?:mp4|mov|avi|webm|mkv))"', text, re.IGNORECASE)
    if m:
        video = m.group(1)

    # a key may be followed by a stray quote before the colon (“屏幕方向“：)
    key_pat = re.compile(r"(?:^|[{,\s\"';；])(" + "|".join(_KNOWN_KEYS) + r")\s*[\"']?\s*:")
    hits = [(m.start(1), m.end(), m.group(1)) for m in key_pat.finditer(text)]
    fields: dict[str, str] = {}
    for i, (ks, vs, key) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        val = text[vs:end]
        val = re.split(r"[}\]]", val)[0]
        # a value may run up to the next `", "key":` — strip JSON-ish debris
        val = re.sub(r"^[\s\",;；{}]+|[\s\",;；{}]+$", "", val)
        fields.setdefault(key, val)

    if not fields and not video:
        raise ValueError("无法从推荐输出中解析出任何字段 — 检查上游 results.json 格式")
    return {"video": video, "fields": fields}


def map_movement(movement_zh: str) -> str:
    """中文运镜名 → 原子类型；未识别时抛错（宁可失败不可错拍）。"""
    m = movement_zh.strip()
    for keywords, atomic in _MOVEMENT_ZH_MAP:
        if any(k in m for k in keywords):
            return atomic
    raise ValueError(f"未识别的运镜类型: {movement_zh!r} —"
                     f"请在 cinema_traj/stages/recommend.py 的 _MOVEMENT_ZH_MAP 补映射")


def canonicalize(parsed: dict, source: str = "") -> dict:
    """Raw fields → contract JSON；缺失合同字段按节奏/角度推断并记录。"""
    f = parsed["fields"]
    defaults_filled = []

    movement_zh = f.get("运镜类型", "").strip()
    atomic = map_movement(movement_zh) if movement_zh else None

    speed = _to_float(f.get("movement_speed"))
    rhythm = f.get("运镜节奏", "中").strip()
    if speed is None:
        speed = next((v for k, v in _RHYTHM_SPEED.items() if k in rhythm), 0.6)
        defaults_filled.append("movement_speed")

    start_angle = _to_float(f.get("start_angle_deg"))
    angle_zh = f.get("初始拍摄角度", "水平").strip()
    if start_angle is None:
        start_angle = next((v for k, v in _START_ANGLE_ZH.items() if k in angle_zh), 0.0)
        defaults_filled.append("start_angle_deg")

    diff = re.search(r"(\d+)", f.get("运镜难度", "") or "")
    return {
        "schema": SCHEMA,
        "source": source,
        "video": parsed.get("video"),
        "scene_category": [s.strip() for s in re.split(r"[_/]", f.get("场景类别", "")) if s.strip()],
        "subject": f.get("被拍摄主体", "").strip(),
        "scene_features": f.get("场景特征", "").strip(),
        "movement_type": movement_zh,
        "movement_type_atomic": atomic,
        "orientation": f.get("屏幕方向", "").strip(),
        "start_angle_deg": float(start_angle),
        "movement_direction": f.get("运镜方向", "").strip(),
        "movement_speed": float(speed),
        "movement_rhythm": rhythm,
        "difficulty": int(diff.group(1)) if diff else None,
        "defaults_filled": defaults_filled,
    }


def _to_float(v) -> float | None:
    if v is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


# --------------------------------------------------------------------------- #
# Canonical recommendation → 1-3-1 plan (single target, deterministic)
# --------------------------------------------------------------------------- #
def _pick_target(rec: dict, scene: Scene) -> tuple:
    """主体优先：推荐里写了被拍摄主体且场景挂了 subject_track → 拍主体。"""
    if scene.subject is not None and rec.get("subject"):
        return scene.subject.subject_id, True
    if scene.objects:
        return scene.objects[0].id, False
    raise ValueError("场景既无主体也无物体，无法选目标")


def _elevation_from_angle(start_angle_deg: float) -> str:
    if start_angle_deg <= -20.0:
        return "high"
    if start_angle_deg >= 20.0:
        return "low"
    return "medium"


def _distance_from_text(rec: dict) -> str:
    text = " ".join(filter(None, [rec.get("movement_type", ""), rec.get("video") or ""]))
    if "特写" in text or "近景" in text:
        return "close"
    if "远景" in text or "全景" in text:
        return "far"
    return "medium"


def to_plan(rec: dict, scene: Scene) -> dict:
    """单目标 1-3-1 计划：A → T。时长由 movement_speed 推算（推/拉按行程，
    其余按默认时长乘节奏系数）。"""
    target, has_subject = _pick_target(rec, scene)
    atomic = rec["movement_type_atomic"]
    kind = "subject_traj" if atomic in C.SUBJECT_TRAJ_TYPES else "object_traj"
    if kind == "subject_traj" and scene.subject is None:
        raise ValueError("推荐映射到主体运镜但场景没有 subject_track")

    scale = next((v for k, v in _RHYTHM_DURATION_SCALE.items()
                  if k in rec.get("movement_rhythm", "中")), 1.0)
    if atomic in ("move_in", "move_out"):
        push_m = 1.2                                  # DollySegment 默认行程（0.55/1.45×r0≈1.8m）
        duration = max(2.0, min(20.0, push_m / max(rec["movement_speed"], 0.05))) * scale
    else:
        duration = C.DEFAULT_DURATION * scale
    duration = round(min(duration, 60.0), 1)

    step_type = atomic
    if atomic.startswith("pan") and rec.get("movement_direction"):
        if "左" in rec["movement_direction"] and atomic != "pan_left":
            step_type = "pan_left"
        elif "右" in rec["movement_direction"] and atomic != "pan_right":
            step_type = "pan_right"

    prompt = (f"运镜推荐适配：{rec.get('movement_type', atomic)}（{atomic}），"
              f"目标 {target}，节奏 {rec.get('movement_rhythm', '')}，"
              f"速度 {rec['movement_speed']} m/s，起始角 {rec['start_angle_deg']}°")
    anchor_step = {"step": 1, "kind": "anchor", "target": target,
                   "elevation": _elevation_from_angle(rec["start_angle_deg"]),
                   "distance": _distance_from_text(rec)}
    if has_subject:
        # measured contract angle (水平 → 0.0) rides next to the fuzzy label —
        # only subject shots can interpret it (the face has a known height)
        anchor_step["start_angle_deg"] = rec["start_angle_deg"]
    plan = {
        "plan_id": "recommend",
        "prompt": prompt,
        "model": f"recommend-adapter/{SCHEMA}",
        "user_specified_order": True,
        "object_sequence": [target],
        "steps": [
            anchor_step,
            {"step": 2, "kind": kind, "target": target,
             "type": step_type, "duration_s": duration},
        ],
        "reasoning": f"上游运镜推荐 {rec.get('movement_type')}→{atomic}"
                     f"（默认补齐: {', '.join(rec.get('defaults_filled') or []) or '无'}）",
    }
    plan = normalize_plan(plan, scene)
    errs = validate_plan(plan, scene)
    if errs:
        raise ValueError("recommend 计划未通过校验:\n- " + "\n- ".join(errs))
    return plan


# --------------------------------------------------------------------------- #
# Stage entry
# --------------------------------------------------------------------------- #
def run(cfg=None, raw_path=None, scene_path=None, video_index: int = 0,
        force=False) -> dict:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    scene = load_scene(scene_path or ct.get("scene", ""))
    raw_path = raw_path or ct.get("recommendation", "")
    if not raw_path:
        raise ValueError("需要 --raw 指定上游推荐 results.json")

    out_dir = _out_dir(cfg, scene)
    plan_path = out_dir / "plan.json"
    if plan_path.exists() and not force:
        from vlm_intention.common.atomic_io import load_json
        cached = load_json(plan_path)
        if cached and cached.get("model", "").startswith("recommend-adapter"):
            print(f"↩️  复用已有 recommend 计划: {plan_path}")
            return cached

    raw = Path(raw_path).read_text(encoding="utf-8")
    rec = canonicalize(parse_raw_recommendation(raw, index=video_index),
                       source=str(raw_path))
    plan = to_plan(rec, scene)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "recommendation.json", rec)
    atomic_write_json(plan_path, plan)
    print(f"✅ recommendation → {out_dir / 'recommendation.json'}")
    print(f"   [{video_index}] {rec.get('video') or ''}")
    print(f"   {rec['movement_type']} → {rec['movement_type_atomic']}"
          f"（speed {rec['movement_speed']} m/s，start {rec['start_angle_deg']}°，"
          f"默认补齐: {', '.join(rec['defaults_filled']) or '无'}）")
    print(f"✅ plan → {plan_path}（目标 {plan['object_sequence']}，"
          f"{plan['steps'][-1]['type']} {plan['steps'][-1]['duration_s']}s）")
    return plan


def _out_dir(cfg, scene) -> Path:
    ct = cfg.get("cinema_traj", {})
    base = ct.get("outputs_dir", "cinema_traj_out")
    p = Path(base)
    if not p.is_absolute():
        from vlm_intention.common.config import repo_root
        p = repo_root() / p
    return p / scene.scene_id


if __name__ == "__main__":                                # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="运镜推荐 results.json → 规范 JSON + 原子计划")
    ap.add_argument("--raw", required=True, help="上游推荐输出（JSON 或 JS 字面量）")
    ap.add_argument("--scene", required=True, help="场景 JSON")
    ap.add_argument("--video-index", type=int, default=0,
                    help="多视频批次里选第 N 条（默认 0）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, raw_path=a.raw, scene_path=a.scene, video_index=a.video_index,
        force=a.force)
