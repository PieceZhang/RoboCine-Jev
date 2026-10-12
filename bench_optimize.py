"""Benchmark for optimize_trajectory: loads a traj_init.json + scene, times the
two-pass optimization, reports iteration totals and residual collisions."""
import json
import sys
import time
from pathlib import Path

ROOT = Path("/Users/siyu/Downloads/Project/Project_Robot/VisionThinker")
sys.path.insert(0, str(ROOT))

from cinema_traj.common.scene import load_scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core.optimizer import OptCfg, optimize_trajectory
from cinema_traj.core.trajectories import Segment


def main(scene_path, traj_path):
    scene = load_scene(scene_path)
    if str(traj_path).startswith("mock:"):
        from cinema_traj.stages.build import build_trajectory
        from cinema_traj.stages.plan import MockPlanner
        plan = MockPlanner().plan(str(traj_path)[5:], scene)
        segments, _ = build_trajectory(scene, plan)
    else:
        init = json.loads(Path(traj_path).read_text())
        segments = [Segment.from_dict(d) for d in init["segments"]]
    sdf = SceneSDF(scene.collision_obbs(), scene.room_bounds())
    cfg = OptCfg()
    t0 = time.perf_counter()
    segs, log = optimize_trajectory(segments, scene, sdf, scene.subject, cfg)
    dt = time.perf_counter() - t0
    iters = sum(r.get("iters", 0) for r in log)
    reasons = {}
    for r in log:
        reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1
    col = [(r["segment"], r.get("collision_rate"))
           for r in log if r.get("collision_rate", 0) > 0]
    worst = min(worst_pos(segs, scene, sdf), default=None)
    print(f"BENCH scene={Path(scene_path).stem} segs={len(segs)} "
          f"obbs={len(sdf.obbs)} time={dt:.1f}s iters={iters} reasons={reasons}")
    print(f"     colliding segments: {col if col else 'none'} | final min clearance: {worst}")


def worst_pos(segs, scene, sdf):
    import numpy as np
    w = []
    for seg in segs:
        th, _ = seg.sample_times()
        p = seg.positions(th, None, scene.subject).numpy()
        w.append(float(sdf.numpy_query(p).min()))
    return w


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
