"""Reach stage: traj_final.json → reach_report.json (Phase-1 whole-body
reachability gate — wheeled base + mast + 6-axis arm + tablet camera).

Reads the finished camera trajectory, builds the SAME collision field the
optimize stage uses (OBB union + real point cloud when the scene carries one),
and answers: can the robot physically place the camera at every sampled pose?
Failures carry per-segment reasons (height band / arm reach / base placement +
nearest obstacle). Robot dimensions come from ``cinema_traj.robot`` in
config.yaml (defaults in cinema_traj.core.reachability.DEFAULT_ROBOT —
measure the real robot and override).

Usage:
    python -m cinema_traj.stages.reach --scene cinema_traj/scenes/X.json
    python run_traj.py --stage reach --scene cinema_traj/scenes/X.json
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common.cloud_sdf import cloud_sdf_from_scene
from cinema_traj.common.scene import Scene, load_scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core.reachability import check_reachability
from cinema_traj.core.trajectories import Segment
from vlm_intention.common.atomic_io import atomic_write_json, load_json
from vlm_intention.common.config import load_config


def run(cfg=None, scene_path=None, traj_path=None) -> dict:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    scene: Scene = load_scene(scene_path or ct.get("scene", ""))
    out_dir = _out_dir(cfg, scene)
    traj_path = Path(traj_path) if traj_path else out_dir / "traj_final.json"
    if not traj_path.exists():                       # fall back to traj_init
        traj_path = out_dir / "traj_init.json"

    traj = load_json(traj_path)
    segments = [Segment.from_dict(d) for d in traj["segments"]]

    sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds(),
                   cloud=cloud_sdf_from_scene(scene))
    if sdf.cloud is not None:
        g = sdf.cloud.shape
        print(f"☁️  点云碰撞场: {g[0]}x{g[1]}x{g[2]} @ {sdf.cloud.voxel}m")

    robot_cfg = ct.get("robot", {})
    report = check_reachability(segments, scene, sdf, robot_cfg)
    report["scene_id"] = scene.scene_id
    report["source"] = str(traj_path)
    r = report["robot"]
    print(f"🦾 可达性检查: {len(segments)} 段（底盘半径 {r['base_radius_m']}m | "
          f"臂展 {r['arm_reach_m'][0]}–{r['arm_reach_m'][1]}m | "
          f"相机高度带 {r['camera_height_range_m'][0]}–"
          f"{r['camera_height_range_m'][1]}m）")

    out_path = out_dir / "reach_report.json"
    atomic_write_json(out_path, report)
    if report["status"] == "failed":
        print(f"❌ 不可达: 机器人无法复现该轨迹 → {out_path}")
        for reason in report["reasons"]:
            print(f"   原因: {reason}")
    else:
        print(f"✅ reach_report → {out_path}（全部帧可达）")
    return report


def _out_dir(cfg, scene) -> Path:
    ct = cfg.get("cinema_traj", {})
    base = ct.get("outputs_dir", "cinema_traj_out")
    p = Path(base)
    if not p.is_absolute():
        from vlm_intention.common.config import repo_root
        p = repo_root() / p
    return p / scene.scene_id


if __name__ == "__main__":                            # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="一期全身可达性检查")
    ap.add_argument("--scene")
    ap.add_argument("--traj", help="traj_final.json（缺省回退 traj_init.json）")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, scene_path=a.scene, traj_path=a.traj)
