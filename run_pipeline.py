#!/usr/bin/env python3
"""VLMIntention pipeline orchestrator.

Runs a single stage or the whole raw→filter→annotate→mine→balance flow.

  python run_pipeline.py --stage all --config config.yaml
  python run_pipeline.py --stage balance
  python run_pipeline.py --stage annotate_depth --force --cpu
  python run_pipeline.py --stage mine --mode full --frames 5

Stages with many flags are also runnable directly with full --help:
  python -m vlm_intention.stages.annotate_depth --help
"""
import argparse
import sys
import time
from importlib import import_module
from pathlib import Path

# Ensure the repo root is on sys.path even if invoked from another CWD
# (removes the old sys.path.insert(0, '<ws>') hacks from the runner scripts).
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vlm_intention.common.config import load_config           # noqa: E402


def _stage(name):
    """Lazy stage import — each stage only loads its OWN deps (torch/transformers
    for depth+mine+calibrate, requests for vlm). balance/filter stay usable even
    when those optional deps are absent."""
    return import_module(f"vlm_intention.stages.{name}")


STAGES = {
    "filter": lambda cfg, a: _stage("filter").run(cfg),
    "annotate_vlm": lambda cfg, a: _stage("annotate_vlm").run(
        cfg, max_images=a.limit, start_from=a.start_from),
    "annotate_depth": lambda cfg, a: _stage("annotate_depth").run(
        cfg, force=a.force, device="cpu" if a.cpu else None,
        limit=a.limit, per_category=a.per_category, start_from=a.start_from),
    "calibrate": lambda cfg, a: _stage("calibrate").run(
        cfg, grid=not a.no_grid, force_reinfer=a.reinfer),
    "mine": lambda cfg, a: _stage("mine").run(cfg, mode=a.mode, frames_per_video=a.frames),
    "balance": lambda cfg, a: _stage("balance").run(cfg),
}
ALL_ORDER = ["filter", "annotate_vlm", "annotate_depth", "mine", "balance"]


def main():
    ap = argparse.ArgumentParser(description="VLMIntention pipeline orchestrator")
    ap.add_argument("--stage", default="all",
                    choices=["all"] + list(STAGES))
    ap.add_argument("--config", default=None, help="config.yaml path (default: shipped)")
    # Pass-through flags (ignored by stages that don't use them)
    ap.add_argument("--force", action="store_true", help="annotate_depth: full recompute")
    ap.add_argument("--cpu", action="store_true", help="annotate_depth: CPU (MPS-deadlock fallback)")
    ap.add_argument("--limit", type=int, help="annotate_vlm/depth: cap image count")
    ap.add_argument("--start-from", type=int, default=0)
    ap.add_argument("--per-category", type=int, help="annotate_depth: Places365-style sampling")
    ap.add_argument("--no-grid", action="store_true", help="calibrate: skip grid search")
    ap.add_argument("--reinfer", action="store_true", help="calibrate: re-infer histograms")
    ap.add_argument("--mode", choices=["pilot", "full"], default="full", help="mine mode")
    ap.add_argument("--frames", type=int, help="mine full: frames per video")
    args = ap.parse_args()

    cfg = load_config(args.config)
    order = ALL_ORDER if args.stage == "all" else [args.stage]
    for name in order:
        print(f"\n{'='*70}\n▶ STAGE: {name}\n{'='*70}")
        t0 = time.time()
        STAGES[name](cfg, args)
        print(f"⏱️  {name}: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
