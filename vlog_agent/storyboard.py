"""Storyboard stage: material_index.json + user brief → edl.json.

The LLM picks clips and orders them narratively; every clip is validated
against the index (id exists, bounds, length) and repaired by clamping. Two
failed LLM rounds fall back to a deterministic rule-based cut so the pipeline
always produces an editable EDL — same philosophy as cinema_traj's
plan-with-validation-retry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .llm import LMSClient, extract_json

MIN_CLIP_S = 2.0
MAX_CLIP_S = 6.0
MAX_CLIP_S_ABS = 8.0

STORYBOARD_SYSTEM = "你是专业 vlog 剪辑师。只输出一个 JSON 对象，不要解释。"


def editable_materials(index: dict, include_photos: bool = False) -> list[dict]:
    """Materials usable for cutting: classified, decodable, with frames."""
    out = []
    for mat in index.values():
        if not mat.get("classified") or mat.get("error"):
            continue
        if mat["type"] != "video" and not include_photos:
            continue
        dur = mat.get("duration") or 0
        if mat["type"] == "video" and dur < 1.0:
            continue
        out.append(mat)
    return out


def _summary_for_llm(mats: list[dict]) -> list[dict]:
    return [
        {
            "id": m["id"],
            "类型": m["type"],
            "时长": m.get("duration"),
            "画面": m.get("desc") or "",
            "场景": m.get("scene"),
            "主体": m.get("subject"),
            "运镜": m.get("shot"),
            "质量": m.get("quality"),
        }
        for m in mats
    ]


def storyboard_prompt(mats: list[dict], brief: str, target_s: int) -> str:
    return (
        f"素材列表（JSON）：{json.dumps(_summary_for_llm(mats), ensure_ascii=False)}\n"
        f"vlog 主题：{brief}\n"
        f"目标成片约 {target_s} 秒。\n"
        "规则：\n"
        f"1. 选 3-8 个片段，按叙事顺序排列；\n"
        f"2. 每个片段给 start 秒和 end 秒，长度 {MIN_CLIP_S:.0f}-{MAX_CLIP_S:.0f} 秒，"
        "不能超过该素材的时长；\n"
        "3. 同一素材最多用一次；优先用质量高的素材；\n"
        "4. 每个片段写一个不超过10字的中文短字幕。\n"
        '输出格式：{"title":"vlog标题","clips":[{"id":"素材的id","start":数字,"end":数字,"caption":"字幕"}]}\n'
        "只输出 JSON。"
    )


def validate_edl(edl: dict, mats_by_id: dict[str, dict]) -> list[dict]:
    """Clamp each clip to reality; drop unusable ones. Returns valid clips."""
    valid: list[dict] = []
    seen: set[str] = set()
    for clip in edl.get("clips", []):
        if not isinstance(clip, dict):
            continue
        mid = clip.get("id")
        if mid not in mats_by_id or mid in seen:
            continue
        mat = mats_by_id[mid]
        caption = str(clip.get("caption", "")).strip()[:12]
        if mat["type"] == "photo":  # stills have no duration; fixed 3s window
            valid.append({"id": mid, "start": 0.0, "end": 3.0, "caption": caption})
            seen.add(mid)
            continue
        dur = float(mat.get("duration") or 0)
        try:
            start = float(clip.get("start", 0))
            end = float(clip.get("end", 0))
        except (TypeError, ValueError):
            continue
        start = max(0.0, min(start, max(0.0, dur - 0.5)))
        end = min(end, dur) if dur > 0 else end
        if end - start < 1.0:
            # degenerate range → give it a sane window from the requested start
            start = min(start, max(0.0, dur - MIN_CLIP_S)) if dur > MIN_CLIP_S else 0.0
            end = min(dur, start + MIN_CLIP_S) if dur > 0 else end
            if end - start < 0.5:
                continue
        length = end - start
        if length > MAX_CLIP_S_ABS:
            end = start + MAX_CLIP_S_ABS
        valid.append({"id": mid, "start": round(start, 2), "end": round(end, 2),
                      "caption": caption})
        seen.add(mid)
    return valid


def rule_based_edl(mats: list[dict], brief: str, target_s: int) -> dict:
    """Deterministic fallback: top-quality, scene-diverse, chronological."""
    def sort_key(m):
        q = m.get("quality") or 3
        if isinstance(q, dict):
            q = 3
        return (-int(q), m.get("created_at") or "")

    picked: list[dict] = []
    used_scenes: set[str] = set()
    for m in sorted(mats, key=sort_key):
        scene = m.get("scene")
        if scene and scene in used_scenes and len(picked) < 3:
            continue  # first pass: prefer scene diversity
        picked.append(m)
        if scene:
            used_scenes.add(scene)
    for m in sorted(mats, key=sort_key):  # second pass: fill up
        if m in picked:
            continue
        picked.append(m)
        if len(picked) >= 8:
            break

    picked = picked[: max(3, min(8, target_s // 4))]
    picked.sort(key=lambda m: m.get("created_at") or "")
    clips = []
    for m in picked:
        dur = float(m.get("duration") or 0)
        start = min(dur * 0.1, max(0.0, dur - MIN_CLIP_S)) if dur > MIN_CLIP_S else 0.0
        length = min(MAX_CLIP_S, dur - start) if dur > 0 else 0
        if length < 1.0:
            continue
        desc = m.get("desc") or Path(m["id"]).stem
        clips.append({"id": m["id"], "start": round(start, 2),
                      "end": round(start + length, 2), "caption": desc[:10]})
    return {"title": brief[:16] or "自动 vlog", "clips": clips, "fallback": True}


def run_storyboard(
    index_path: Path,
    out_dir: Path,
    brief: str,
    target_s: int = 30,
    api_url: str = "",
    model: Optional[str] = None,
    include_photos: bool = False,
) -> dict:
    """Generate + validate the EDL; write out_dir/edl.json."""
    index = json.loads(Path(index_path).read_text())
    mats = editable_materials(index, include_photos=include_photos)
    if len(mats) < 2:
        raise RuntimeError(
            f"only {len(mats)} usable materials — run ingest first / lower the bar")
    mats_by_id = {m["id"]: m for m in mats}

    edl: Optional[dict] = None
    client = LMSClient(api_url=api_url, model=model)
    prompt = storyboard_prompt(mats, brief, target_s)
    feedback = ""
    for attempt in range(2):
        reply = client.vision(
            prompt + feedback, [], system=STORYBOARD_SYSTEM, max_tokens=900,
            temperature=0.3,
        )
        cand = extract_json(reply)
        if cand and cand.get("clips"):
            clips = validate_edl(cand, mats_by_id)
            if len(clips) >= 2:
                edl = {"title": str(cand.get("title", "")).strip()[:30] or brief[:16],
                       "clips": clips}
                break
        feedback = ("\n注意：上一次输出无法通过校验。id 必须逐字取自素材列表的 id 字段，"
                    "start/end 必须是数字且在素材时长内，只输出一个 JSON 对象。")

    if edl is None:
        edl = rule_based_edl(mats, brief, target_s)

    edl["brief"] = brief
    edl["target_s"] = target_s
    edl_path = out_dir / "edl.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    edl_path.write_text(json.dumps(edl, ensure_ascii=False, indent=1))
    return {"edl_path": str(edl_path), "edl": edl}
