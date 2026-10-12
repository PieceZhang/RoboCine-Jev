"""Stage: mine — extract more frames from source videos + depth-score them.

Data-redistribution sampling: the raw 1-keyframe-per-video set is obstacle-sparse;
mining more frames from obstacle-dense / indoor videos raises the positive rate.
Unifies mine_obstacle_frames.py (pilot: pool A+B, per-direction top-K candidates)
and run_mine_full.py (full: every mp4 × N frames, depth-labeled).

Requires system ffmpeg/ffprobe on PATH.
"""
import json
import subprocess
from pathlib import Path

from ..common.atomic_io import atomic_write_json, load_json
from ..common.config import load_config
from ..common.depth_model import DepthObstacleAnnotator
from ..common.geometry import score_frame
from ..common.paths import resolve_base, discover_videos, mp4_for
from ..common.schema import DIRS, build_annotation_record
from ..common.thresholds import FRAC_THRESH, RANK_THRESH

PILOT_POOL_B_N = 20
PILOT_TOP_K_PER_DIR = 8


def get_duration(mp4) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except (ValueError, TypeError):
        return 0.0


def extract_frames(mp4, out_dir, times):
    """ffmpeg-seek one frame per relative timestamp in `times`. Returns PNG paths."""
    out_dir = Path(out_dir)
    dur = get_duration(mp4)
    if dur < 0.3:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, tfrac in enumerate(times):
        out = out_dir / f"{Path(mp4).stem}_f{i:02d}.png"
        if out.exists():
            frames.append(out); continue
        t = dur * tfrac
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}",
             "-i", str(mp4), "-frames:v", "1", str(out)],
            capture_output=True)
        if out.exists():
            frames.append(out)
    return frames


def list_positive_keyframe_mp4s(base, cfg):
    """Pool A: mp4s whose existing keyframe is positive in any direction."""
    depth = load_json(base / cfg.outputs.depth_annotations)
    out = []
    for k, v in depth.items():
        if any(v.get(d) for d in DIRS):
            mp4 = base / mp4_for(k)
            if mp4.exists():
                out.append(mp4)
    return sorted(set(out))


def list_indoor_mp4s(base, cfg):
    """Pool B: indoor-scene mp4s (from filter_manifest.json if present, else scan)."""
    manifest = base / "filter_manifest.json"
    if manifest.exists():
        m = load_json(manifest)
        return sorted((base / p for p in m.get("indoor_videos", []) if (base / p).exists()),
                      key=str)
    # fallback: scan video_labels via env classification
    from ..common import env as envmod
    out = []
    for vl in base.glob("**/标注tag/video_labels.json"):
        if "depth_maps" in vl.parts:
            continue
        scene_dir = vl.parent.parent
        try:
            d = json.load(open(vl, encoding="utf-8"))
        except Exception:
            continue
        for name, s in d.items():
            if isinstance(s, str) and envmod.env_from_scene(
                    envmod.parse_label_fields(s).get("场景类别")) == "indoor":
                p = scene_dir / name
                if p.exists():
                    out.append(p)
    return sorted(set(out), key=str)


def _score(ann, frame):
    env = ann.classify_env(frame)
    depth = ann.infer_depth(frame, env)
    if depth is None:
        return None
    s = score_frame(depth, ann.dist_thresh, ann.frac_thresh, RANK_THRESH,
                    ann.vert_frac, ann.depth_min, ann.depth_max)
    return s, env, ann.env_model_id.get(env, ann.model_indoor)


# ── full mode: extract + depth-label every video's frames ──
def run_full(cfg=None, frames_per_video=None):
    cfg = cfg or load_config()
    base = resolve_base(cfg)
    times = cfg.sampling.frame_times[: (frames_per_video or cfg.sampling.frames_per_video)]
    out_dir = Path(cfg.paths.full_frames_dir)
    mined_json = base / cfg.outputs.mined_annotations

    mp4s = discover_videos(base)
    print(f"🎬 {len(mp4s)} 个 mp4，每视频抽 {len(times)} 帧")
    done = load_json(mined_json, default={})
    if done:
        print(f"📂 续传：已标注 {len(done)} 帧")

    print("\n[1/2] 抽帧...")
    all_frames = []
    for i, mp4 in enumerate(mp4s):
        rel_parent = Path(mp4).parent.relative_to(base)
        all_frames.extend(extract_frames(mp4, out_dir / rel_parent, times))
        if (i + 1) % 100 == 0:
            print(f"  抽帧 {i+1}/{len(mp4s)}，累计 {len(all_frames)}")
    print(f"  共抽出 {len(all_frames)} 帧")

    print("\n[2/2] 深度标注...")
    ann = DepthObstacleAnnotator(base, read_only=True, device=None, cfg=cfg)
    n_new = 0
    for i, frame in enumerate(all_frames):
        rel = str(frame.relative_to(base))
        if rel in done:
            continue
        res = _score(ann, frame)
        if res is None:
            continue
        s, env, model_id = res
        directions = {d: bool(s["strict"][d] > ann.frac_thresh) for d in DIRS}
        done[rel] = build_annotation_record(directions, frame.name, rel, model_id,
                                            source="mined_depth")
        n_new += 1
        if (i + 1) % 30 == 0:
            print(f"  进度 {i+1}/{len(all_frames)} | 新标 {n_new}")
        if n_new % 50 == 0:
            atomic_write_json(mined_json, done)
    atomic_write_json(mined_json, done)
    print(f"\n✅ mined JSON: {len(done)} 条 → {mined_json.name}")
    for d in DIRS:
        n = sum(1 for v in done.values() if v[d])
        print(f"  {d}: {n}/{len(done)} = {n/max(len(done),1)*100:.1f}%")


# ── pilot mode: extract + score + per-direction top-K candidates ──
def run_pilot(cfg=None):
    cfg = cfg or load_config()
    base = resolve_base(cfg)
    times = cfg.sampling.frame_times
    out_dir = Path(cfg.paths.pilot_frames_dir)
    cands_json = out_dir / "_candidates.json"

    import random
    poolA = list_positive_keyframe_mp4s(base, cfg)
    indoor = list_indoor_mp4s(base, cfg)
    rng = random.Random(0)
    poolB = sorted(rng.sample(indoor, min(PILOT_POOL_B_N, len(indoor))), key=str)
    pool = poolA + [p for p in poolB if p not in set(poolA)]
    print(f"池 A(keyframe 正样本视频) {len(poolA)} | 池 B(室内采样) {len(poolB)} | 合计 {len(pool)}")

    print("\n[1/3] 抽帧...")
    all_frames = []
    for mp4 in pool:
        all_frames.extend(extract_frames(mp4, out_dir / Path(mp4).parent.name, times))
    print(f"  共抽出 {len(all_frames)} 帧")

    print("\n[2/3] 深度打分（CPU）...")
    ann = DepthObstacleAnnotator(base, read_only=True, device=None, cfg=cfg)
    scored = []
    for i, frame in enumerate(all_frames):
        res = _score(ann, frame)
        if res is None:
            continue
        s, env, _ = res
        scored.append({"frame": str(frame.relative_to(base)),
                       "from_video": Path(frame).stem.rsplit("_f", 1)[0],
                       "scene": Path(frame).parent.name, **s})
        if (i + 1) % 30 == 0:
            print(f"  打分 {i+1}/{len(all_frames)}")

    print(f"\n[3/3] 每方向 top-{PILOT_TOP_K_PER_DIR} 候选...")
    by_dir = {d: [] for d in DIRS}
    for item in scored:
        for d in DIRS:
            by_dir[d].append((item["loose"][d], item))
    selected = {}                       # frame -> item (union across directions)
    for d in DIRS:
        picked = 0
        for _, item in sorted(by_dir[d], key=lambda x: x[0], reverse=True):
            if item["frame"] in selected:
                continue
            selected[item["frame"]] = item
            picked += 1
            if picked >= PILOT_TOP_K_PER_DIR:
                break
    candidates = sorted(selected.values(), key=lambda x: x["frame"])
    for d in DIRS:
        n_pos = sum(1 for it in scored if it["strict"][d] > FRAC_THRESH)
        top = max(by_dir[d], key=lambda x: x[0])[0]
        print(f"  {d}: strict 正样本 {n_pos}/{len(scored)} | top loose near_ratio={top:.3f}")
    atomic_write_json(cands_json, candidates)
    print(f"\n✅ 候选 {len(candidates)} 张 → {cands_json}")


def run(cfg=None, mode="full", frames_per_video=None):
    if mode == "pilot":
        run_pilot(cfg)
    else:
        run_full(cfg, frames_per_video=frames_per_video)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="full")
    ap.add_argument("--frames", type=int)
    a = ap.parse_args()
    run(mode=a.mode, frames_per_video=a.frames)
