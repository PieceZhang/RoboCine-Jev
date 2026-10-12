"""Path resolution + filesystem discovery.

Consolidates the BASE literals, mp4_for/mp4_rel_for, the realpath traversal
guard, and recursive image/video enumeration that were copy-pasted across the
original scripts.
"""
import os
from pathlib import Path

# Dirs that hold derived/cache frames — excluded from raw image discovery.
_CACHE_DIRS = ("depth_maps", "抽帧_full", "抽帧_pilot")
_IMG_EXTS = (".jpg", ".jpeg", ".png")


def resolve_base(cfg):
    """The corpus root (paths.base) as an absolute Path."""
    return Path(cfg.paths.base).expanduser()


def mp4_for(keyframe_rel):
    """Map a 标注tag PNG keyframe key -> its sibling source .mp4.
    '<scene>/标注tag/NAME.png' -> '<scene>/NAME.mp4'."""
    d = os.path.dirname(keyframe_rel)          # .../标注tag
    scene = os.path.dirname(d)                 # .../<scene>
    base = os.path.splitext(os.path.basename(keyframe_rel))[0]
    return os.path.join(scene, base + ".mp4")


def mp4_rel_for(frame_rel):
    """Map an extracted-frame key -> its source .mp4.
    '抽帧_full|抽帧_pilot/<scene>/<stem>_f<NN>.png' -> '<scene>/<stem>.mp4'.
    (Uses the working algorithm; the original serve_mined_gt copy had a buggy
    double-reassign of the path variable before this loop.)"""
    p = frame_rel
    for pre in ("抽帧_full/", "抽帧_pilot/"):
        if p.startswith(pre):
            p = p[len(pre):]
            break
    stem = os.path.splitext(os.path.basename(p))[0]
    if "_f" in stem:
        stem = stem.rsplit("_f", 1)[0]
    return os.path.join(os.path.dirname(p), stem + ".mp4")


def safe_resolve(base, rel):
    """Resolve rel under base, blocking path traversal. Returns (abs_path, ok)."""
    if not rel:
        return None, False
    full = os.path.realpath(os.path.join(str(base), rel))
    base_real = os.path.realpath(str(base))
    if full != base_real and not full.startswith(base_real + os.sep):
        return None, False
    return full, os.path.isfile(full)


def discover_images(base, exclude_cache=True):
    """Recursive jpg/jpeg/png discovery (case-insensitive), skipping cache dirs."""
    base = Path(base)
    out = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if exclude_cache and any(seg in _CACHE_DIRS for seg in p.parts):
            continue
        if p.suffix.lower() in _IMG_EXTS:
            out.append(p)
    return sorted(set(out))


def discover_videos(base):
    """List all .mp4 files under base (any depth)."""
    base = Path(base)
    return sorted(p for p in base.rglob("*.mp4") if p.is_file())
