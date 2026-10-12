"""Stage: calibrate (OPTIONAL QA) — depth vs VLM consistency + threshold grid search.

Migrated from calibrate_depth.py. Uses common.geometry.close_frac_from_hist and
common.depth_model. NOT part of the linear raw->train flow (invoke explicitly).

  python -m vlm_intention.stages.calibrate [--no-grid] [--reinfer]
"""
import pickle
from pathlib import Path

import numpy as np

from ..common.config import load_config
from ..common.depth_model import DepthObstacleAnnotator
from ..common.geometry import close_frac_from_hist
from ..common.paths import resolve_base
from ..common.schema import DIRS, HORIZ_DIRS, VERT_DIRS
from ..common.atomic_io import atomic_write_json, load_json

N_BINS = 200
HORIZ_DIST_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
VERT_DIST_GRID = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
FRAC_GRID = [0.02, 0.05, 0.10, 0.15, 0.20]


def metrics(pred, ref):
    pred = np.asarray(pred, bool); ref = np.asarray(ref, bool)
    tp = int(np.sum(pred & ref)); fp = int(np.sum(pred & ~ref))
    fn = int(np.sum(~pred & ref)); tn = int(np.sum(~pred & ~ref))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "balanced_acc": 0.5 * (tpr + tnr),
            "accuracy": (tp + tn) / max(len(ref), 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _table(title, rows):
    print(f"\n{title}")
    print(f"{'方向':<6}{'P':>8}{'R':>8}{'F1':>8}{'BalAcc':>8}{'Acc':>8}  (tp/fp/fn/tn)")
    for d in DIRS:
        m = rows[d]
        print(f"{d:<6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
              f"{m['balanced_acc']:>8.3f}{m['accuracy']:>8.3f}  "
              f"({m['tp']}/{m['fp']}/{m['fn']}/{m['tn']})")


def _build_histograms(ann, keys, cache, force, depth_min, depth_max):
    if cache.exists() and not force:
        print(f"📂 载入直方图缓存: {cache.name}")
        return pickle.load(open(cache, "rb"))
    bin_edges = np.linspace(depth_min, depth_max, N_BINS + 1)
    hists = {"bin_edges": bin_edges, "hists": {}}
    print(f"🔮 推理 {len(keys)} 张构建直方图...")
    for i, k in enumerate(keys):
        depth = ann.infer_depth(ann.base_path / k, ann.classify_env(ann.base_path / k))
        if depth is None:
            continue
        regions = ann.region_depths(depth)
        per = {}
        for d, arr in regions.items():
            valid = arr[np.isfinite(arr) & (arr >= depth_min) & (arr <= depth_max)]
            per[d] = np.histogram(valid, bins=bin_edges)[0].astype(np.int64)
        hists["hists"][k] = per
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(keys)}]")
    pickle.dump(hists, open(cache, "wb"))
    print(f"💾 直方图缓存 -> {cache.name}")
    return hists


def run(cfg=None, grid=True, force_reinfer=False, top_n=None):
    cfg = cfg or load_config()
    base = resolve_base(cfg)
    vlm = load_json(base / cfg.outputs.vlm_annotations)
    depth_json = load_json(base / cfg.outputs.depth_annotations)
    if not depth_json:
        print(f"⚠️  未找到 {cfg.outputs.depth_annotations}，先跑 annotate_depth")
        return

    keys = sorted(set(vlm) & set(depth_json))
    print(f"🔗 对齐 {len(keys)} / VLM {len(vlm)} / 深度 {len(depth_json)}")
    rows = {d: metrics([depth_json[k].get(d, False) for k in keys],
                       [vlm[k].get(d, False) for k in keys]) for d in DIRS}
    _table("A. 当前阈值一致性（与 VLM 标注）", rows)

    if not grid:
        return
    ann = DepthObstacleAnnotator(base, read_only=True, cfg=cfg)
    top_n = top_n or (cfg.sampling.calibration_top_n if "calibration_top_n" in cfg.sampling else 20)
    cache = base / cfg.outputs.depth_histograms_cache
    hist = _build_histograms(ann, sorted(vlm), cache, force_reinfer,
                             ann.depth_min, ann.depth_max)
    be = hist["bin_edges"]; hm = hist["hists"]
    keys = [k for k in sorted(vlm) if k in hm]

    print(f"\nB. 网格搜索最优阈值（按 F1）")
    recommended, best_preds = {}, {}
    for d in DIRS:
        grid_d = HORIZ_DIST_GRID if d in HORIZ_DIRS else VERT_DIST_GRID
        ref = np.array([vlm[k].get(d, False) for k in keys])
        best = None
        for dist in grid_d:
            for frac in FRAC_GRID:
                pred = np.array([close_frac_from_hist(hm[k][d], be, dist) > frac for k in keys])
                m = metrics(pred, ref)
                if best is None or m["f1"] > best[0]["f1"]:
                    best = (m, dist, frac, pred)
        m, dist, frac, pred = best
        recommended[d] = (dist, frac); best_preds[d] = pred
        print(f"  {d}: F1={m['f1']:.3f} (P={m['precision']:.3f} R={m['recall']:.3f})  "
              f"<- DIST={dist}, FRAC={frac}")
    print("\n📋 推荐阈值（回填 config.yaml thresholds.dist_thresh / frac_thresh）:")
    print("  " + ", ".join(f"{d}: dist={recommended[d][0]}, frac={recommended[d][1]}" for d in DIRS))

    mism = sorted(((sum(1 for d in DIRS if bool(best_preds[d][i]) != bool(vlm[k].get(d, False))), k)
                   for i, k in enumerate(keys)), reverse=True)
    out = [{"mismatch_dirs": n, "relative_path": k, "vlm": {d: vlm[k].get(d) for d in DIRS}}
           for n, k in mism[:top_n]]
    atomic_write_json(base / cfg.outputs.calibration_disagreements, out)
    print(f"\n🔀 分歧 top{top_n} -> {cfg.outputs.calibration_disagreements}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--reinfer", action="store_true")
    a = ap.parse_args()
    run(grid=not a.no_grid, force_reinfer=a.reinfer)
