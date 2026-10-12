#!/usr/bin/env python
"""vlog_agent 编排器：素材库 → 分类索引 → 叙事 EDL → ffmpeg 成片。

用法（仓库根目录）：
  # ① ingest：扫描素材目录，抽帧 + VLM 分类 → vlog_agent_out/material_index.json
  python run_vlog.py --stage ingest --src /path/to/素材 [--force] [--dry-vlm]
  # ② storyboard：读索引 + 主题 → vlog_agent_out/edl.json
  python run_vlog.py --stage storyboard --brief "新丰韶关自驾旅拍" --target 30
  # ③ edit：按 EDL 出片 → vlog_agent_out/edits/<ts>/vlog.mp4
  python run_vlog.py --stage edit
  # 一次跑完：
  python run_vlog.py --stage all --src /path/to/素材 --brief "..."

模型：LM Studio（OpenAI 兼容端点，默认 http://127.0.0.1:1234）。
--model 不给时自动用 /v1/models 里第一个非 embedding 模型。
--dry-vlm 跳过 VLM 调用（离线调试管线用，分类字段为空）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vlog_agent.edit import run_edit
from vlog_agent.ingest import run_ingest
from vlog_agent.storyboard import run_storyboard

DEFAULT_OUT = Path(__file__).parent / "vlog_agent_out"
DEFAULT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["ingest", "storyboard", "edit", "all"])
    ap.add_argument("--src", type=Path, help="素材根目录（ingest/all 必填）")
    ap.add_argument("--brief", default="旅行 vlog", help="vlog 主题（storyboard/all）")
    ap.add_argument("--target", type=int, default=30, help="目标成片秒数")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="产物根目录")
    ap.add_argument("--api-url", default=DEFAULT_API_URL)
    ap.add_argument("--model", default=None, help="LM Studio 模型 id（默认自动探测）")
    ap.add_argument("--force", action="store_true", help="重分类已有条目")
    ap.add_argument("--max-frames", type=int, default=3)
    ap.add_argument("--photos", action="store_true", help="storyboard 允许插入照片片段")
    ap.add_argument("--dry-vlm", action="store_true", help="跳过 VLM（离线调试）")
    args = ap.parse_args(argv)

    if args.stage in ("ingest", "all") and args.src is None:
        ap.error("--src is required for ingest/all")

    if args.stage in ("ingest", "all"):
        print(f"① ingest: {args.src}")
        summary = run_ingest(args.src, args.out, api_url=args.api_url,
                             model=args.model, force=args.force,
                             max_frames=args.max_frames, dry_vlm=args.dry_vlm)
        print(f"   {summary['stats']}  → {summary['index_path']}")

    if args.stage in ("storyboard", "all"):
        print(f"② storyboard: brief={args.brief!r} target={args.target}s")
        result = run_storyboard(args.out / "material_index.json", args.out,
                                brief=args.brief, target_s=args.target,
                                api_url=args.api_url, model=args.model,
                                include_photos=args.photos)
        edl = result["edl"]
        src = "rule-based fallback" if edl.get("fallback") else "LLM"
        print(f"   [{src}] {edl['title']}: {len(edl['clips'])} clips → {result['edl_path']}")

    if args.stage in ("edit", "all"):
        print("③ edit:")
        report = run_edit(args.out / "material_index.json",
                          args.out / "edl.json", args.out)
        ok = sum(1 for c in report["clips"] if "error" not in c)
        print(f"   {ok}/{len(report['clips'])} clips, {report['total_s']}s → {report['output']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
