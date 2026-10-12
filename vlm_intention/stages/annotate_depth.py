"""Stage: annotate_depth — main labeler (Depth-Anything-V2 5-direction).

Driver over common.depth_model.DepthObstacleAnnotator. Migrated from
run_depth_annotation.py. Writes obstacle_annotations_depth.json + .meta.json.
"""
from ..common.config import load_config
from ..common.depth_model import DepthObstacleAnnotator
from ..common.schema import DIRS


def run(cfg=None, force=False, device=None, limit=None, per_category=None,
        start_from=0):
    cfg = cfg or load_config()
    ann = DepthObstacleAnnotator(cfg.paths.base, device=device, force=force, cfg=cfg)

    print("\n🚀 深度障碍物标注")
    print("=" * 70)
    print("🎯 阈值: " + "  ".join(f"{d}<{ann.dist_thresh[d]}m" for d in DIRS)
          + f"  |  占比>{ann.frac_thresh}")
    print(f"🏠/🌳 室内外均→{ann.model_indoor.split('/')[-1]}"
          + ("（outdoor_alias=indoor）" if ann.env_model_id['outdoor'] == ann.model_indoor else ""))
    if ann.stale and not force:
        print("⚠️  配置已变更但未加 --force：将跳过所有已存在条目（=不重算）。")
    print("=" * 70)

    if per_category:
        sample = ann.get_images_per_category(per_category)
        print(f"📊 每类别 {per_category} 张 → 采样 {len(sample)} 张（seed=0）")
        ann.batch_annotate(images=sample)
        print("\n🎉 采样标注完成！")
        ann.get_statistics()
        return

    total = len(ann.get_all_images())
    existing = len(ann.existing_annotations)
    remaining = total - existing
    print(f"📊 进度: {existing}/{total} | ⏳ 剩余: {remaining}"
          + ("  (force：全部重算)" if force else ""))
    if force:
        ann.batch_annotate(max_images=limit, start_from=0)
        print("\n🎉 全量重算完成！")
    elif remaining > 0:
        print("\n🎯 开始处理...")
        ann.batch_annotate(max_images=limit, start_from=existing)
        print("\n🎉 完成！")
    else:
        print("✅ 已完成全部标注（如需用新配置重算请加 --force）")
    ann.get_statistics()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-category", type=int)
    ap.add_argument("--start-from", type=int, default=0)
    a = ap.parse_args()
    run(force=a.force, device="cpu" if a.cpu else None, limit=a.limit,
        per_category=a.per_category, start_from=a.start_from)
