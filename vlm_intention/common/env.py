"""Indoor/outdoor scene classification.

Scene-category header in each 标注tag/video_labels.json -> 'indoor'/'outdoor',
with a path-keyword fallback. Migrated verbatim from depth_annotation.py.
"""
from pathlib import Path
from typing import Optional

# Path-keyword fallback when the 场景类别 header is ambiguous/missing.
INDOOR_KEYWORDS = [
    "室内", "居家", "酒店", "咖啡厅", "花店", "工作环境", "艺术馆", "图书馆",
    "音乐厅", "舞蹈房", "健身房", "酒吧", "舞厅", "饭店", "商场", "广场",
    "机场", "车站", "教室", "阶梯教室",
]


def parse_label_fields(s: str) -> dict:
    """Parse '场景类别：x；被拍摄主体：y；运镜类型：z' into a dict."""
    out = {}
    for part in (s or "").replace("；", ";").split(";"):
        part = part.strip()
        for sep in ("：", ":"):
            if sep in part:
                k, _, v = part.partition(sep)
                out[k.strip()] = v.strip()
                break
    return out


def env_from_scene(scene_str: Optional[str]) -> Optional[str]:
    """场景类别 header -> 'indoor' / 'outdoor' / None."""
    s = (scene_str or "").strip()
    if s.startswith("室内"):
        return "indoor"
    if s.startswith("室外"):
        return "outdoor"
    return None


def env_from_path(rel_path: str, keywords=None, default="outdoor") -> str:
    """Keyword fallback on the path; defaults to outdoor."""
    for kw in (keywords or INDOOR_KEYWORDS):
        if kw in rel_path:
            return "indoor"
    return default


def load_scene_env(base) -> dict:
    """{image_stem: 'indoor'|'outdoor'} from every 标注tag/video_labels.json under base."""
    base = Path(base)
    idx = {}
    for vf in base.glob("**/标注tag/video_labels.json"):
        if "depth_maps" in vf.parts:
            continue
        try:
            d = __import__("json").load(open(vf, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict):
            for name, s in d.items():
                if not isinstance(s, str):
                    continue
                env = env_from_scene(parse_label_fields(s).get("场景类别"))
                if env:
                    idx[Path(name).stem] = env
    return idx


def classify_env(image_path, scene_env_index=None, keywords=None, default="outdoor") -> str:
    """Resolve env for an image: scene index by stem first, then path-keyword fallback."""
    stem = Path(image_path).stem if not isinstance(image_path, str) else Path(image_path).stem
    if scene_env_index is not None and stem in scene_env_index:
        return scene_env_index[stem]
    return env_from_path(str(image_path), keywords=keywords, default=default)
