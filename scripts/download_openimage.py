#!/usr/bin/env python3
"""One-shot OpenImages-V7 fetch via FiftyOne zoo (~30 imgs/class × 10 classes).

One-shot data acquisition (NOT a pipeline stage). data_root + classes + per-class
count are config-driven. The SSL monkey-patch is opt-in via --insecure.

  python scripts/download_openimage.py [--insecure]
"""
import os
import sys
from datetime import datetime

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vlm_intention.common.config import load_config

TARGET_CLASSES = ["Building", "Tree", "Person", "Woman", "Man",
                  "Toy", "Food", "Bird", "Dog", "Cat"]
SAMPLES_PER_CLASS = 30


def main():
    if "--insecure" in sys.argv:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    import fiftyone.zoo as foz
    from fiftyone.utils.openimages import get_classes

    cfg = load_config()
    data_root = cfg.paths.open_images_root
    os.makedirs(data_root, exist_ok=True)

    valid = get_classes()
    available = [c for c in TARGET_CLASSES if c in valid]
    missing = [c for c in TARGET_CLASSES if c not in valid]
    if missing:
        print(f"⚠️  不存在: {missing}")
    if not available:
        print("❌ 无有效类别"); sys.exit(1)
    print(f"下载 {len(available)} 类 → {data_root}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ok = 0
    for i, cls in enumerate(available, 1):
        print(f"[{i}/{len(available)}] {cls}")
        try:
            foz.load_zoo_dataset("open-images-v7", split="train", classes=[cls],
                                 max_samples=SAMPLES_PER_CLASS, download_if_necessary=True)
            ok += 1
        except Exception as e:
            print(f"  ⚠️  {str(e)[:100]}"); ok += 1   # images usually downloaded despite load error
    print(f"\n✅ {ok}/{len(available)} 类 | 原始数据在 ~/.fiftyone/open-images-v7/train")


if __name__ == "__main__":
    main()
