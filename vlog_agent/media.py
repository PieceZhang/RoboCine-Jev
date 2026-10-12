"""ffprobe/ffmpeg helpers shared by ingest and edit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def ffprobe_json(path: Path) -> dict | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or not r.stdout:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def probe_media(path: Path) -> dict:
    """Normalized meta: duration/size/fps/creation_time/encoder/has_audio."""
    out: dict = {"duration": None, "width": None, "height": None, "fps": None,
                 "codec": None, "creation_time": None, "encoder": None,
                 "has_audio": False}
    data = ffprobe_json(path)
    if data is None:
        return out
    fmt = data.get("format", {})
    try:
        out["duration"] = round(float(fmt.get("duration")), 3)
    except (TypeError, ValueError):
        pass
    tags = fmt.get("tags") or {}
    out["creation_time"] = tags.get("creation_time")
    out["encoder"] = tags.get("encoder")
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and out["codec"] is None:
            out["codec"] = s.get("codec_name")
            out["width"], out["height"] = s.get("width"), s.get("height")
            rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or ""
            try:
                num, den = rate.split("/")
                if int(den) != 0:
                    out["fps"] = round(int(num) / int(den), 3)
            except (ValueError, ZeroDivisionError):
                pass
        elif s.get("codec_type") == "audio":
            out["has_audio"] = True
    return out


def run_ffmpeg(args: list[str], timeout: int = 900) -> None:
    """Run ffmpeg, raising with the stderr tail on failure."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:8])}…\n{r.stderr[-800:]}")


def stem_hash(relpath: str) -> str:
    """Stable short id for frame-cache directories."""
    return hashlib.sha1(relpath.encode()).hexdigest()[:12]
