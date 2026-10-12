"""Edit stage: execute an EDL with ffmpeg.

Every clip is normalized to the same container parameters (H.264 1920x1080
30fps + AAC 48k stereo) so the concat demuxer can stream-copy them together:
  video clip  → -ss/-t trim from the ORIGINAL file (not the LRF proxy),
                scale+pad (contain, black bars), optional caption overlay
  photo       → 3s still with a slow zoompan + silent audio track
Captions are rendered to transparent PNGs with PIL (CJK-safe via PingFang)
and composited with the overlay filter — deliberately NOT drawtext/subtitles,
which many ffmpeg builds ship without (no freetype/libass).

Input index bookkeeping per render (one ffmpeg process per clip):
  0 = main input (trimmed video or looping photo)
  1 = caption PNG          (only when a caption exists)
  last = silent audio      (only when the main input has no audio)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .media import ffprobe_json, run_ffmpeg

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
OUT_W, OUT_H, OUT_FPS = 1920, 1080, 30
PHOTO_CLIP_S = 3.0
CAPTION_H = 170  # caption strip height above the bottom edge


def _find_font() -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return ""


def _caption_png(caption: str, workdir: Path, idx: int) -> str:
    """Render a centered CJK caption on a transparent 1920x140 strip."""
    png = workdir / f"cap_{idx}.png"
    img = Image.new("RGBA", (OUT_W, 140), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    path = _find_font()
    font = ImageFont.truetype(path, 64) if path else ImageFont.load_default()
    stroke_w = 3 if path else 0
    box = draw.textbbox((0, 0), caption, font=font, stroke_width=stroke_w)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text(((OUT_W - tw) / 2 - box[0], (140 - th) / 2 - box[1]), caption,
              font=font, fill=(255, 255, 255, 255),
              stroke_width=stroke_w, stroke_fill=(0, 0, 0, 200))
    img.save(png)
    return str(png)


def _caption_overlay_fc(cap_png: str | None, dur: float) -> str | None:
    if not cap_png:
        return None
    end = max(0.5, dur - 0.3)
    return (f"[bg][1:v]overlay=0:{OUT_H - CAPTION_H}:"
            f"enable='between(t,0.2,{end:.2f})'[v]")


VIDEO_BASE_VF = (
    f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
    f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,fps={OUT_FPS}"
)


def cut_video_clip(mat: dict, clip: dict, out_path: Path, workdir: Path, idx: int) -> float:
    """Trim + normalize one video clip. Returns the clip duration."""
    start, end = float(clip["start"]), float(clip["end"])
    dur = max(0.5, end - start)

    cap_png = None
    if clip.get("caption") and _find_font():
        cap_png = _caption_png(clip["caption"], workdir, idx)

    cmd = ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", mat["path"]]
    if cap_png:
        cmd += ["-i", cap_png]
    if mat.get("has_audio"):
        amap = ["-map", "0:a:0", "-af", "aresample=48000", "-ac", "2"]
    else:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        amap = ["-map", f"{2 if cap_png else 1}:a:0"]

    fc = _caption_overlay_fc(cap_png, dur)
    if fc:
        vmap = ["-filter_complex", f"[0:v]{VIDEO_BASE_VF}[bg];{fc}", "-map", "[v]"]
    else:
        vmap = ["-map", "0:v:0", "-vf", VIDEO_BASE_VF]

    cmd += [*vmap, *amap,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
            "-t", f"{dur:.3f}", "-avoid_negative_ts", "make_zero", str(out_path)]
    run_ffmpeg(cmd)
    return dur


def cut_photo_clip(mat: dict, clip: dict, out_path: Path, workdir: Path, idx: int) -> float:
    """3s Ken-Burns-ish still: upscale → center crop → slow zoompan → silent audio."""
    dur = float(clip.get("end") or PHOTO_CLIP_S)
    frames = max(1, int(dur * OUT_FPS))

    cap_png = None
    if clip.get("caption") and _find_font():
        cap_png = _caption_png(clip["caption"], workdir, idx)

    photo_vf = (
        f"scale=2400:-2,crop=2400:1350,"
        f"zoompan=z='min(1+0.0012*in,1.18)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s={OUT_W}x{OUT_H}:fps={OUT_FPS},format=yuv420p"
    )
    cmd = ["-loop", "1", "-i", mat["path"]]
    if cap_png:
        cmd += ["-i", cap_png]
    cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]

    fc = _caption_overlay_fc(cap_png, dur)
    if fc:
        vmap = ["-filter_complex", f"[0:v]{photo_vf}[bg];{fc}", "-map", "[v]"]
    else:
        vmap = ["-map", "0:v:0", "-vf", photo_vf]

    cmd += [*vmap, "-map", f"{2 if cap_png else 1}:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
            "-t", f"{dur:.3f}", str(out_path)]
    run_ffmpeg(cmd)
    return dur


def _concat(clips: list[Path], out_path: Path) -> None:
    lst = out_path.parent / "concat_list.txt"
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out_path)])
    except RuntimeError:
        # stream-copy concat is strict; re-encode as the safety net
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(out_path)])


def run_edit(index_path: Path, edl_path: Path, out_root: Path) -> dict:
    index = json.loads(Path(index_path).read_text())
    edl = json.loads(Path(edl_path).read_text())
    ts = time.strftime("%Y%m%d_%H%M%S")
    workdir = out_root / "edits" / ts
    clips_dir = workdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    report_clips: list[dict] = []
    for i, clip in enumerate(edl.get("clips", [])):
        mat = index.get(clip["id"])
        if mat is None:
            report_clips.append({"clip": i, "id": clip["id"], "error": "id not in index"})
            continue
        out_i = clips_dir / f"clip_{i:03d}.mp4"
        try:
            if mat["type"] == "photo":
                dur = cut_photo_clip(mat, clip, out_i, clips_dir, i)
            else:
                dur = cut_video_clip(mat, clip, out_i, clips_dir, i)
        except Exception as exc:
            report_clips.append({"clip": i, "id": clip["id"], "error": str(exc)[:200]})
            continue
        parts.append(out_i)
        report_clips.append({"clip": i, "id": clip["id"], "start": clip["start"],
                             "end": clip["end"], "caption": clip.get("caption", ""),
                             "file": str(out_i), "rendered_s": round(dur, 2)})

    if len(parts) < 1:
        raise RuntimeError("no clip rendered successfully — nothing to concatenate")

    final = workdir / "vlog.mp4"
    _concat(parts, final)

    probe = ffprobe_json(final) or {}
    total = float((probe.get("format") or {}).get("duration") or 0)
    report = {
        "output": str(final),
        "title": edl.get("title", ""),
        "total_s": round(total, 2),
        "clips": report_clips,
    }
    (workdir / "edit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    return report
