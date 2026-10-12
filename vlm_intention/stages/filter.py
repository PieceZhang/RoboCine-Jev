"""Stage: filter — explicit scene / indoor-outdoor selection manifest.

Promotes the implicit 筛选 step (previously buried in depth_annotation's env
loading and mine's pool selection) to a first-class stage. Walks every
标注tag/video_labels.json, classifies each scene indoor/outdoor, and emits a
manifest consumed by annotate_depth (env routing) and mine (indoor pool B).
"""
import json
from collections import defaultdict
from pathlib import Path

from ..common.atomic_io import atomic_write_json
from ..common.config import load_config
from ..common.env import load_scene_env, env_from_scene, parse_label_fields
from ..common.paths import resolve_base


def run(cfg=None):
    cfg = cfg or load_config()
    base = resolve_base(cfg)
    scenes = {}                 # scene_name -> info
    indoor_videos, outdoor_videos = [], []

    for vf in sorted(base.glob("**/标注tag/video_labels.json")):
        if "depth_maps" in vf.parts:
            continue
        scene_dir = vf.parent.parent          # .../<scene>
        scene = scene_dir.name
        try:
            d = json.load(open(vf, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        envs = defaultdict(int)
        for name, s in d.items():             # name = mp4 filename in this scene
            if not isinstance(s, str):
                continue
            env = env_from_scene(parse_label_fields(s).get("场景类别")) or "unknown"
            envs[env] += 1
            mp4 = scene_dir / name
            if mp4.exists():
                (indoor_videos if env == "indoor" else outdoor_videos).append(
                    str(mp4.relative_to(base)))
        dominant = max(envs, key=envs.get) if envs else "unknown"
        scenes[scene] = {"dominant_env": dominant, "env_counts": dict(envs),
                          "n_labeled": sum(envs.values())}

    manifest = {
        "n_scenes": len(scenes),
        "n_indoor_videos": len(indoor_videos),
        "n_outdoor_videos": len(outdoor_videos),
        "scenes": scenes,
        "indoor_videos": sorted(set(indoor_videos)),     # mine pool B candidates
        "outdoor_videos": sorted(set(outdoor_videos)),
    }
    out_path = base / "filter_manifest.json"
    atomic_write_json(out_path, manifest)
    print(f"🎬 {len(scenes)} 场景 | 室内视频 {len(indoor_videos)} | 室外视频 {len(outdoor_videos)}")
    indoor_scenes = [s for s, v in scenes.items() if v["dominant_env"] == "indoor"]
    print(f"   室内场景({len(indoor_scenes)}): {', '.join(sorted(indoor_scenes)[:8])}{'…' if len(indoor_scenes)>8 else ''}")
    print(f"✅ 清单 → {out_path.name}")
    return manifest


if __name__ == "__main__":
    run()
