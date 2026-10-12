"""Ingest stage: scan a footage tree → probe meta → extract frames → VLM classify.

Output: material_index.json (one record per mp4/mov/jpg), consumed by the
storyboard stage.

Design notes:
- DJI .LRF is a low-res MP4 proxy sharing the stem of the full clip: frame
  extraction prefers it so a spinning HDD never seeks the 2-4GB original.
- .SRT next to a DJI video is flight telemetry; the first GPS fix is captured.
- Incremental + atomic: already-classified ids are skipped unless force=True;
  the index is written via tmp+rename so an interrupted run never corrupts it.
- HYPERLAPSE/PANORAMA dirs hold frame sequences / stitched panoramas, not
  normal clips — skipped.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .llm import LMSClient, extract_json
from .media import probe_media, run_ffmpeg, stem_hash

VIDEO_EXTS = {".mp4", ".mov"}
PHOTO_EXTS = {".jpg", ".jpeg"}
HEIC_EXT = ".heic"
PROXY_EXT = ".lrf"
TELEMETRY_EXT = ".srt"
SKIP_DIRS = {"HYPERLAPSE", "PANORAMA"}

# DJI_20260102173837_0001_D / dji_fly_20260102_140846 — filename date beats mtime.
_DATE_PATTERNS = [re.compile(r"(?:19|20)\d{12}"), re.compile(r"(?:19|20)\d{6}")]

SCENE_VALUES = ["室内", "室外人工", "室外自然"]
SUBJECT_VALUES = ["单人", "多人", "空镜", "物体"]
SHOT_VALUES = ["航拍", "地面", "手持", "其他"]
ISSUE_VALUES = ["抖动", "过曝", "欠曝", "模糊", "无"]

CLASSIFY_SYSTEM = "你是视频素材标注助手。只输出一个 JSON 对象，不要解释，不要输出其他文字。"


def parse_filename_date(name: str) -> Optional[str]:
    """Extract an ISO date from common DJI naming, else None."""
    for pat in _DATE_PATTERNS:
        m = pat.search(name)
        if not m:
            continue
        s = m.group(0)
        try:
            if len(s) == 14:
                return datetime.strptime(s, "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%S")
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%dT00:00:00")
        except ValueError:
            continue
    return None


def scan_tree(src_root: Path) -> list[dict]:
    """Walk the tree and build raw material records (no VLM yet)."""
    src_root = Path(src_root).resolve()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d not in SKIP_DIRS
                             and not d.endswith(".photoslibrary"))
        for fn in sorted(filenames):
            if not fn.startswith("."):
                files.append(Path(dirpath) / fn)

    stems: dict[str, list[Path]] = {}
    for f in files:
        stems.setdefault((f.parent, f.stem.lower()), []).append(f)

    records: list[dict] = []
    for f in files:
        ext = f.suffix.lower()
        rel = f.relative_to(src_root).as_posix()
        if ext in VIDEO_EXTS or ext == HEIC_EXT or ext in PHOTO_EXTS:
            siblings = stems[(f.parent, f.stem.lower())]
            proxy = next((s for s in siblings if s.suffix.lower() == PROXY_EXT), None)
            telemetry = next((s for s in siblings if s.suffix.lower() == TELEMETRY_EXT), None)
            records.append({
                "id": rel,
                "type": "photo" if ext in PHOTO_EXTS or ext == HEIC_EXT else "video",
                "path": str(f),
                "proxy": str(proxy) if proxy else None,
                "telemetry": str(telemetry) if telemetry else None,
            })
    return records


def first_gps(srt_path: Path) -> Optional[str]:
    """First GPS fix from a DJI telemetry SRT (kept raw, format varies by model)."""
    try:
        text = srt_path.read_text(errors="ignore")
    except OSError:
        return None
    m = re.search(r"GPS:\(([^)]*)\)", text)
    return m.group(1) if m else None


def created_at(rec: dict, probe: dict) -> Optional[str]:
    if probe.get("creation_time"):
        return probe["creation_time"]
    fn_date = parse_filename_date(Path(rec["path"]).name)
    if fn_date:
        return fn_date
    mtime = os.path.getmtime(rec["path"])
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_frames(rec: dict, frames_dir: Path, max_frames: int) -> list[str]:
    """Pull up to max_frames thumbnails, preferring the LRF proxy."""
    src = rec["proxy"] or rec["path"]
    duration = probe_media(Path(src)).get("duration") or 0.0
    out_paths: list[str] = []
    cache = frames_dir / stem_hash(rec["id"])
    cache.mkdir(parents=True, exist_ok=True)

    if rec["type"] == "photo":
        out = cache / "f0.jpg"
        if not out.exists():
            run_ffmpeg(["-i", rec["path"], "-frames:v", "1",
                        "-vf", "scale=w=640:h=640:force_original_aspect_ratio=decrease",
                        str(out)])
        return [str(out)]

    if duration <= 0:
        return []
    if duration < 3.0:
        fractions = [0.5]
    else:
        fractions = [0.15, 0.5, 0.85][: max_frames]
    for k, frac in enumerate(fractions):
        out = cache / f"f{k}.jpg"
        if not out.exists():
            run_ffmpeg(["-ss", f"{duration * frac:.2f}", "-i", src, "-frames:v", "1",
                        "-vf", "scale=w=640:h=640:force_original_aspect_ratio=decrease",
                        "-q:v", "4", str(out)])
        out_paths.append(str(out))
    return out_paths


def classify_prompt(rec: dict, dir_hint: str) -> str:
    kind = "一张照片" if rec["type"] == "photo" else f"同一段{'视频' if rec['type'] == 'video' else '素材'}的几帧截图"
    return (
        f"以下是{kind}（拍摄目录：{dir_hint}）。请输出 JSON，字段固定为：\n"
        '{"scene":"室内|室外人工|室外自然","subject":"单人|多人|空镜|物体",'
        '"shot":"航拍|地面|手持|其他","quality":1到5的整数,'
        '"issues":["抖动|过曝|欠曝|模糊|无 中的一项或多项"],"desc":"不超过25字的中文画面描述"}\n'
        "只输出 JSON。"
    )


def sanitize_tags(tags: dict) -> dict:
    """Clamp free-form VLM output to the fixed vocab; unknown → fallback value."""
    def pick(value, allowed, default):
        return value if value in allowed else default

    issues = tags.get("issues")
    if not isinstance(issues, list):
        issues = [issues] if isinstance(issues, str) else []
    issues = [i for i in issues if i in ISSUE_VALUES] or ["无"]
    try:
        quality = max(1, min(5, int(tags.get("quality", 3))))
    except (TypeError, ValueError):
        quality = 3
    return {
        "scene": pick(tags.get("scene"), SCENE_VALUES, "室外自然"),
        "subject": pick(tags.get("subject"), SUBJECT_VALUES, "空镜"),
        "shot": pick(tags.get("shot"), SHOT_VALUES, "其他"),
        "quality": quality,
        "issues": issues,
        "desc": str(tags.get("desc", "")).strip()[:40],
    }


def classify_material(client: LMSClient, rec: dict, frames: list[str], dir_hint: str) -> dict:
    """VLM call with one parse-failure retry; returns the sanitized tag dict."""
    prompt = classify_prompt(rec, dir_hint)
    reply = client.vision(prompt, frames, system=CLASSIFY_SYSTEM, max_tokens=400)
    tags = extract_json(reply)
    if tags is None:
        reply = client.vision(
            prompt + "\n注意：上一次输出无法解析。只输出一个 JSON 对象，以 { 开头、} 结尾。",
            frames, system=CLASSIFY_SYSTEM, max_tokens=400,
        )
        tags = extract_json(reply)
    if tags is None:
        raise ValueError(f"VLM output not parseable as JSON: {reply[:120]!r}")
    return sanitize_tags(tags)


def run_ingest(
    src: Path,
    out_dir: Path,
    api_url: str,
    model: Optional[str] = None,
    force: bool = False,
    max_frames: int = 3,
    dry_vlm: bool = False,
) -> dict:
    """Scan + classify + write material_index.json. Returns a summary dict."""
    src, out_dir = Path(src), Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "material_index.json"

    index: dict = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())

    client = None if dry_vlm else LMSClient(api_url=api_url, model=model)
    records = scan_tree(src)
    stats = {"total": len(records), "classified": 0, "skipped": 0, "failed": 0, "dry": 0}

    for rec in records:
        prev = index.get(rec["id"])
        if prev and prev.get("classified") and not force:
            stats["skipped"] += 1
            continue

        probe = probe_media(Path(rec["path"]))
        dir_hint = Path(rec["path"]).parent.name
        mat = {
            **rec,
            "duration": probe["duration"],
            "width": probe["width"],
            "height": probe["height"],
            "fps": probe["fps"],
            "codec": probe["codec"],
            "has_audio": probe["has_audio"],
            "device": probe["encoder"],
            "created_at": created_at(rec, probe),
            "dir_hint": dir_hint,
            "gps": first_gps(Path(rec["telemetry"])) if rec["telemetry"] else None,
            "frames": extract_frames(rec, frames_dir, max_frames),
            "classified": False,
            "model": None,
            "tags": None,
            "quality": None,
            "desc": "",
            "error": None,
        }

        if dry_vlm:
            mat["error"] = "dry_vlm"
            stats["dry"] += 1
        else:
            try:
                assert client is not None
                tags = classify_material(client, mat, mat["frames"], dir_hint)
                mat.update(tags)
                mat["classified"] = True
                mat["model"] = client.model
                stats["classified"] += 1
            except Exception as exc:  # keep going: one bad file must not kill the batch
                mat["error"] = str(exc)[:300]
                stats["failed"] += 1

        index[rec["id"]] = mat
        _atomic_write(index_path, index)  # checkpoint after every file

    return {"index_path": str(index_path), "stats": stats, "index": index}


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)
