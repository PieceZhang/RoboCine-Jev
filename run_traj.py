#!/usr/bin/env python3
"""cinema_traj orchestrator — From 3D map, Motion Prompt, Subject Pose, to Trajectory.

Usage:
    python run_traj.py --stage all --scene cinema_traj/demo/demo_apartment.json \
        --prompt-file cinema_traj/demo/demo_prompt.txt --mock

Stages: extract → recommend/plan → build → optimize → export → retime → view
        (all = plan…view；export 发现超速会自动连带 retime；
         recommend 用上游运镜推荐 results.json 直接生成 plan.json)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from importlib import import_module
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vlm_intention.common.config import load_config


def _traj_final_path(cfg, a):
    """traj_final.json for the ACTIVE scene: --scene wins over config."""
    from cinema_traj.stages.export import _base
    ct = cfg.get("cinema_traj", {})
    scene = a.scene or ct.get("scene", "demo_apartment.json")
    return _base(cfg) / Path(scene).stem / "traj_final.json"


STAGES = {
    "extract": lambda cfg, a: import_module("cinema_traj.stages.extract").run(
        cfg, map_path=a.map, out_path=a.scene_out, voxel=a.voxel,
        wall_peak=a.wall_peak, up=a.up, review=a.review, force=a.force),
    "recommend": lambda cfg, a: import_module("cinema_traj.stages.recommend").run(
        cfg, raw_path=a.raw, scene_path=a.scene, video_index=a.video_index,
        force=a.force),
    "plan": lambda cfg, a: import_module("cinema_traj.stages.plan").run(
        cfg, scene_path=a.scene, prompt=a.prompt, prompt_file=a.prompt_file,
        mock=a.mock, force=a.force),
    "build": lambda cfg, a: import_module("cinema_traj.stages.build").run(
        cfg, scene_path=a.scene, force=a.force),
    "optimize": lambda cfg, a: import_module("cinema_traj.stages.optimize").run(
        cfg, scene_path=a.scene, device=a.device, max_iters=a.iters, force=a.force,
        collision_scene=a.collision_scene),
    "reach": lambda cfg, a: import_module("cinema_traj.stages.reach").run(
        cfg, scene_path=a.scene, traj_path=a.traj),
    "export": lambda cfg, a: import_module("cinema_traj.stages.export").run(
        cfg, traj_path=_traj_final_path(cfg, a), out_dir=None, force=a.force,
        scene_path=a.scene),
    "retime": lambda cfg, a: import_module("cinema_traj.stages.retime_stage").run(
        cfg, traj_path=_traj_final_path(cfg, a), out_dir=None, force=a.force,
        scene_path=a.scene),
    "view": lambda cfg, a: import_module("cinema_traj.stages.view").run(
        cfg, traj_path=_traj_final_path(cfg, a), out_path=None, scene_path=a.scene,
        cloud_voxel=a.cloud_voxel, max_points=a.cloud_max_points,
        cloud_margin=a.cloud_margin),
}
ALL_ORDER = ["plan", "build", "optimize", "export", "retime", "view"]


def main() -> None:
    ap = argparse.ArgumentParser(description="cinema_traj 流水线编排器")
    ap.add_argument("--stage", required=True,
                    choices=["all"] + list(STAGES), help="要运行的阶段")
    ap.add_argument("--config", help="自定义 config.yaml（深合并覆盖默认）")
    ap.add_argument("--scene", help="场景 JSON 路径（plan/build/optimize）")
    ap.add_argument("--map", help="PCD/PLY 点云路径（extract）")
    ap.add_argument("--raw", help="上游运镜推荐 results.json（recommend）")
    ap.add_argument("--video-index", type=int, default=0,
                    help="recommend：多视频批次里选第 N 条（默认 0）")
    ap.add_argument("--scene-out", help="extract 输出的 scene JSON 路径")
    ap.add_argument("--prompt", help="运镜请求文本")
    ap.add_argument("--prompt-file", help="运镜请求文件")
    ap.add_argument("--mock", action="store_true", help="plan 阶段用 MockPlanner（离线）")
    ap.add_argument("--device", default=None, help="optimize 设备（默认 cpu）")
    ap.add_argument("--iters", type=int, help="每段最大优化迭代数")
    ap.add_argument("--collision-scene", default=None,
                    help="optimize：用另一个（更稠密的）场景碰撞场审判帧与指标，"
                         "自身场景数据保留在 metrics.self_scene（交叉审计，"
                         "如 --scene 半场景 --collision-scene 全场景）")
    ap.add_argument("--voxel", type=float, default=0.05, help="extract 体素降采样粒度")
    ap.add_argument("--wall-peak", type=float, default=6.0,
                    help="extract 墙面峰阈值（×直方图中位数）；3DGS 点云建议 3.0")
    ap.add_argument("--up", default="y", choices=["y", "z"],
                    help="extract 地图高度轴约定（z=地面在XY，SceneSplat Replica "
                         "3DGS 用 z；入口旋转成内部 Y-up）")
    ap.add_argument("--review", action="store_true", help="extract 打印 JSON 草稿")
    ap.add_argument("--cloud-voxel", type=float, default=None,
                    help="view 嵌入点云体素粒度 m（默认 0.05；越小越密、"
                         "HTML 越大、旋转越慢）")
    ap.add_argument("--cloud-max-points", type=int, default=None,
                    help="view 嵌入点云上限（默认 60000）")
    ap.add_argument("--cloud-margin", type=float, default=None,
                    help="view 点云裁剪 = 房间边界外扩米数（默认 1.5；-1 = 不裁剪，"
                         "整朵点云）")
    ap.add_argument("--traj", help="reach：指定 traj_final.json（缺省用输出目录，"
                                   "不存在则回退 traj_init.json）")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    names = ALL_ORDER if a.stage == "all" else [a.stage]
    for name in names:
        print(f"\n════════════ {name} ════════════")
        t0 = time.time()
        STAGES[name](cfg, a)
        print(f"── {name} 完成，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
