"""Stage: balance — merge annotation sources + greedily select a training set
where every direction reaches >= target_positive_rate (default 35%).

Migrated from build_training_set.py. Preserves current behavior:
  • only keys whose file exists under paths.base are kept (Open-Images, on a
    different base, is loaded then excluded — set balance.include_external_sources
    to admit it after copying under base);
  • target N = floor(rarest_direction_positives / target_positive_rate);
  • fill remainder with all-negative frames so the model sees "no obstacle".
"""
import math
import os
from pathlib import Path

from ..common.atomic_io import atomic_write_json, load_json
from ..common.config import load_config
from ..common.paths import resolve_base
from ..common.schema import DIRS, build_annotation_record
from ..common.thresholds import TARGET_RATE


def load_merged(cfg):
    base = resolve_base(cfg)
    merged = {}
    skipped = 0
    include_ext = bool(cfg.balance.get("include_external_sources", False)) \
        if hasattr(cfg.balance, "get") else cfg.balance.include_external_sources
    for src in cfg.balance.sources:
        is_ext = isinstance(src, dict) and "external" in src
        p = Path(src["external"]) if is_ext else (base / src)
        if not p.exists():
            print(f"  ⚠ 缺 {p}")
            continue
        for k, v in load_json(p).items():
            if is_ext and not include_ext:
                skipped += 1
                continue
            if os.path.exists(str(base / k)):
                merged[k] = v
            else:
                skipped += 1
    if skipped:
        print(f"  ℹ 跳过 {skipped} 条不在 paths.base 下的帧（如 OI，base 不同）")
    return merged


def build(merged, target_rate=TARGET_RATE):
    items = [(k, {d: bool(v.get(d)) for d in DIRS}) for k, v in merged.items()]
    pos = {d: sum(1 for _, b in items if b[d]) for d in DIRS}
    rarest = min(pos, key=pos.get)
    target = int(pos[rarest] / target_rate)
    need = {d: math.ceil(target_rate * target) for d in DIRS}

    pool = list(items)
    selected = []
    cur = {d: 0 for d in DIRS}
    while len(selected) < target and pool:
        deficit = [d for d in DIRS if cur[d] < need[d]]
        if deficit:
            best = max(pool, key=lambda kv: (
                sum(1 for d in deficit if kv[1][d]),
                sum(1 for d in DIRS if kv[1][d])))
        else:
            best = min(pool, key=lambda kv: sum(1 for d in DIRS if kv[1][d]))
        selected.append(best)
        pool.remove(best)
        for d in DIRS:
            cur[d] += 1 if best[1][d] else 0
    return selected, target, need, cur


def run(cfg=None):
    cfg = cfg or load_config()
    base = resolve_base(cfg)
    merged = load_merged(cfg)
    target_rate = cfg.thresholds.target_positive_rate
    print(f"合并去重: {len(merged)} 帧 | 目标正样本率 {target_rate}")
    selected, target, need, cur = build(merged, target_rate=target_rate)
    n = len(selected)
    print(f"\n目标训练集规模 = {n}（最稀疏方向 {min(cur.values())} 正样本约束）")
    print(f"{'方向':<4}{'正样本':>6}{'占比':>8}{'达标':>6}")
    all_ok = True
    for d in DIRS:
        r = cur[d] / n if n else 0
        ok = r >= target_rate
        all_ok &= ok
        print(f"{d:<6}{cur[d]:>6}{r*100:>7.1f}%{'✅' if ok else '❌':>5}")

    out = {}
    for k, b in selected:
        orig = merged[k]
        rec = build_annotation_record(
            b, k.split("/")[-1], k, orig.get("model", ""), source="curated_balanced")
        out[k] = rec
    out_path = base / cfg.outputs.train_annotations
    atomic_write_json(out_path, out)
    print(f"\n{'✅' if all_ok else '⚠️'} 已写出 {n} 条 → {out_path.name}")
    return out


if __name__ == "__main__":
    run()
