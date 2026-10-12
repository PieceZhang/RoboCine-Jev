#!/usr/bin/env python
"""1000 条图片测试集自动标注 CLI（LM Studio qwen3-vl）。

用法：
    .venv/bin/python run_annotator.py --input <图片目录> --out annotator_out/pilot \
        [--limit N] [--force] [--model ...] [--api-url ...] \
        [--stage {all,label,report}] [--dry-run] [--no-scene-feature] [--no-peiti] [--keep-raw]

行为：
- 逐张标注（场景大类/小类、主体大类/小类、培体、场景特征、运镜 top5 推荐），
  结果逐条追加 annotator_out/<name>/annotations.jsonl（fsync），断点续跑：
  重新执行时跳过已有 id（--force 覆盖重标）。
- 标注完（或 --stage report）生成 report.json + summary.csv。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_annotator.annotate import annotate_one  # noqa: E402
from video_annotator.llm import DEFAULT_API_URL, LMSClient  # noqa: E402
from video_annotator.report import generate_report, load_records  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".tif", ".tiff"}


def collect_images(input_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )
    if limit is not None:
        files = files[:limit]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="图片标注管线（LM Studio VLM）")
    ap.add_argument("--input", required=True, help="图片目录（递归扫描）")
    ap.add_argument("--out", default="annotator_out/pilot", help="输出目录")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 张")
    ap.add_argument("--force", action="store_true", help="覆盖已有标注重新跑")
    ap.add_argument("--model", default=None, help="模型名（默认 /v1/models 自动发现 qwen3-vl）")
    ap.add_argument("--api-url", default=DEFAULT_API_URL, help="LM Studio chat/completions 端点")
    ap.add_argument("--stage", choices=["all", "label", "report"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="只清点输入，不调用模型")
    ap.add_argument("--no-scene-feature", action="store_true", help="不标场景特征")
    ap.add_argument("--no-peiti", action="store_true", help="不标培体")
    ap.add_argument("--keep-raw", action="store_true", help="记录中保留模型原始 JSON")
    ap.add_argument("--max-px", type=int, default=1024, help="送入模型的缩图边长上限")
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"[annotator] 输入目录不存在：{input_dir}")
        return 2
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "annotations.jsonl"

    images = collect_images(input_dir, args.limit)
    print(f"[annotator] 输入目录：{input_dir}")
    print(f"[annotator] 待处理图片：{len(images)} 张（limit={args.limit}）-> {jsonl_path}")
    if args.dry_run:
        for p in images[:20]:
            print("  -", p.relative_to(input_dir))
        if len(images) > 20:
            print(f"  ... 共 {len(images)} 张")
        return 0

    if not images:
        print("[annotator] 没有图片可处理")
        return 2

    existing: dict[str, dict] = {}
    if jsonl_path.exists():
        existing = {r.get("id"): r for r in load_records(jsonl_path)}
        print(f"[annotator] 断点续跑：已有 {len(existing)} 条，跳过已存在 id")

    todo = [p for p in images if args.force
            or p.relative_to(input_dir).as_posix() not in existing
            or existing[p.relative_to(input_dir).as_posix()].get("status") == "error"]
    if not todo:
        print("[annotator] 所有图片均已标注")

    client = None
    n_new = 0
    t_start = time.time()
    if todo and args.stage in ("all", "label"):
        client = LMSClient(api_url=args.api_url, model=args.model, max_px=args.max_px)
        print(f"[annotator] 模型：{client.model}  端点：{args.api_url}")

    for i, img in enumerate(todo, 1):
        if args.stage == "report":
            break
        record = annotate_one(
            client, img,
            record_id=img.relative_to(input_dir).as_posix(),  # 相对路径做 id，避免同名冲突
            scene_feature=not args.no_scene_feature,
            peiti=not args.no_peiti,
            keep_raw=args.keep_raw,
        )
        existing[record["id"]] = record
        n_new += 1
        # 逐条原子落盘（tmp + os.replace + fsync），中途断电/中断可续跑
        tmp = jsonl_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in existing.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, jsonl_path)
        labels = record.get("labels") or {}
        mv = "/".join(t["name"] for t in (record.get("movement_top5") or [])[:3])
        print(f"  [{i}/{len(todo)}] {record['id']}  {record['status']}  "
              f"{labels.get('场景类别')}  {labels.get('被拍摄主体')}  "
              f"培体={labels.get('培体') or '-'}  top3={mv}  {record['latency_s']}s")

    if n_new:
        dt = time.time() - t_start
        print(f"[annotator] 新标注 {n_new} 条，耗时 {dt/60:.1f} 分钟"
              f"（均 {dt/max(n_new,1):.1f}s/张）")

    # 汇总报告
    report_path = generate_report(jsonl_path, out_dir)
    print(f"[annotator] 报告：{report_path}")
    print(f"[annotator] 表格：{out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
