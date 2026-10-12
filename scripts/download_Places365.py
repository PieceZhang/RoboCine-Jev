#!/usr/bin/env python3
"""One-shot Places365 train-standard small (256×256) download via torchvision.

One-shot data acquisition (NOT a pipeline stage). data_root is config-driven.

  python scripts/download_Places365.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torchvision.datasets as datasets
from vlm_intention.common.config import load_config


def main():
    cfg = load_config()
    data_root = cfg.paths.places365_root
    os.makedirs(os.path.dirname(data_root) or ".", exist_ok=True)
    print(f"下载 Places365 (train-standard, small) → {data_root}")
    datasets.Places365(root=data_root, split="train-standard", small=True, download=True)
    print("✅ 完成")


if __name__ == "__main__":
    main()
