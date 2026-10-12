"""Optimize stage: traj_init.json → traj_final.json (two-pass SDF optimization
+ sampled frames + metrics). Device defaults to CPU (this repo has documented
MPS deadlocks); ``--device mps`` opts in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common import constants as C
from cinema_traj.common.scene import Scene, load_scene
from cinema_traj.common.cloud_sdf import cloud_sdf_from_scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core import costs
from cinema_traj.core.anchors import _point_visible, points_visible_batch
from cinema_traj.core.optimizer import OptCfg, optimize_trajectory
from cinema_traj.core.trajectories import Segment
from vlm_intention.common.atomic_io import atomic_write_json, load_json
from vlm_intention.common.config import load_config


# --------------------------------------------------------------------------- #
# Frame sampling + metrics
# --------------------------------------------------------------------------- #
def _coll_items(scene: Scene) -> list:
    """(owner_id, OBB) pairs — id-keyed so a target can be excluded in any scene."""
    return [(s.id, s.obb) for s in scene.structures] + \
           [(o.id, o.obb) for o in scene.objects]


def sample_frames(segments: list, scene: Scene, sdf: SceneSDF,
                  audit: tuple | None = None) -> tuple:
    """Sample every segment at its frame rate; compute per-frame collision /
    occlusion flags; returns (frames, metrics).

    audit: optional (SceneSDF, audit_scene) — a DENSER collision field the same
    trajectory is judged against (e.g. optimize on the sparse half scene but
    audit on the --full one). Frame flags + metrics then measure reality; the
    optimizing field's numbers are reported under metrics["self_scene"] so the
    two worlds' disagreement is printed, never hidden."""

    frames = []
    cam_path_m = 0.0

    # the judge of every frame: the audit world when given, else the
    # optimizer's own world (never both at once — the contrast is the point)
    judge_sdf, judge_scene = audit if audit else (sdf, scene)
    judge_obbs = _coll_items(judge_scene)           # (owner_id, OBB) pairs
    auditing = audit is not None

    for seg in segments:
        progress, t_seconds = seg.sample_times()    # progress∈[0,1]; t_seconds = wall clock
        cam_pos_frames = seg.positions(progress, None, scene.subject).numpy()
        look_at_frames = seg.look_target(progress, t_seconds, scene.subject).numpy()
        fov_frames = seg.fov_deg(progress)          # None for non-zoom segments
        fov_list = fov_frames.numpy() if fov_frames is not None \
            else [None] * len(cam_pos_frames)

        # batched clearances: one query per segment per world (the audit
        # branch also needs the optimizing world's numbers for the contrast)
        clearance_all = np.asarray(judge_sdf.numpy_query(cam_pos_frames))
        clearance_self_all = np.asarray(sdf.numpy_query(cam_pos_frames)) if auditing else None
        seg_blockers = [b for owner_id, b in judge_obbs
                        if owner_id != seg.target_id
                        # subject targets: also drop the matching person_<id>
                        # cast obstacle OBB — the look target IS the subject's
                        # own body (head sits ~0.13 m below the OBB top), so
                        # counting it flags EVERY frame of a subject shot as
                        # occluded and the reported occlusion rate is garbage
                        and owner_id.lower() != f"person_{seg.target_id.lower()}"]
        if seg.type.startswith("pan"):
            # a pan ACROSS a row of people sweeps its look through the
            # other seated bodies by design (measured: pan_right across
            # the sofa trio, 22 frames cross person_Dad to look past him
            # at the Kid). Passing actors = the shot's semantics, not
            # occlusion — judge pans on non-person blockers only.
            person_ids = {owner_id for owner_id, _ in judge_obbs
                          if owner_id.lower().startswith("person_")}
            seg_blockers = [b for owner_id, b in judge_obbs
                            if owner_id not in person_ids
                            and owner_id != seg.target_id
                            and owner_id.lower() != f"person_{seg.target_id.lower()}"]

        # occlusion, batched over frames (was a per-frame×per-blocker
        # python slab loop): occluded iff a blocker's slab straddles the
        # ray AND the blocker does not contain the look point (framing,
        # not blocking). LED-look segments (orbit lead / truck lead) judge
        # the SUBJECT ray (camera → head), not the look-point ray.
        if seg.type in ("orbit_quarter", "orbit_half", "orbit_full",
                        "arc_swing") and scene.subject is not None:
            _ray_t = np.array([scene.subject.key_point(float(t), "head")
                               for t in t_seconds])
        else:
            _ray_t = look_at_frames
        _occl_flags = ~points_visible_batch(cam_pos_frames, _ray_t,
                                            look_at_frames, seg_blockers)
        prev_cam_pos = None
        for i, (cam_pos, look_at, t_sec, fov_deg) in enumerate(
                zip(cam_pos_frames, look_at_frames, t_seconds, fov_list)):
            if prev_cam_pos is not None:
                cam_path_m += float(np.linalg.norm(cam_pos - prev_cam_pos))
            prev_cam_pos = cam_pos

            clearance_m = float(clearance_all[i])
            occluded = bool(_occl_flags[i])
            frame = {
                "segment": seg.name, "type": seg.type, "t": round(float(t_sec), 3),
                "pos": [round(float(x), 4) for x in cam_pos],
                "look": [round(float(x), 4) for x in look_at],
                "fov_deg": round(float(fov_deg), 2) if fov_deg is not None else None,
                "colliding": clearance_m < 0.0,
                "occluded": bool(occluded),
                "clearance_m": round(clearance_m, 3),
            }
            if auditing:
                frame["clearance_self_m"] = round(float(clearance_self_all[i]), 3)
            frames.append(frame)

    metrics = compute_metrics(frames, segments, scene)
    if auditing:
        metrics["collision_field"] = f"audit:{judge_scene.scene_id}"
        n = max(len(frames), 1)
        metrics["self_scene"] = {
            "scene_id": scene.scene_id,
            "collision_rate": round(sum(1 for f in frames
                                        if f["clearance_self_m"] < 0.0) / n, 4),
            "min_clearance_m": round(min((f["clearance_self_m"] for f in frames),
                                         default=0.0), 3),
        }
    return frames, metrics


def compute_metrics(frames: list, segments: list, scene: Scene) -> dict:
    n = max(len(frames), 1)
    n_colliding = sum(1 for f in frames if f["colliding"])
    n_occluded = sum(1 for f in frames if f["occluded"])
    min_clearance = min((f["clearance_m"] for f in frames), default=0.0)
    cam_path_m = 0.0
    prev_pos = None
    for f in frames:
        pos = np.asarray(f["pos"])
        if prev_pos is not None:
            cam_path_m += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
    # NOTE (object_coverage): planned/visited are built from the same segments,
    # so today this ratio is structurally 1.0 — kept for schema compatibility;
    # a real coverage metric (e.g. by visibility) is future work.
    planned = {s.target_id for s in segments if s.type != "arc"}
    visited = {s.target_id for s in segments if s.type != "arc"}
    return {
        "frames": n,
        "collision_rate": round(n_colliding / n, 4),
        "occlusion_rate": round(n_occluded / n, 4),
        "min_clearance_m": round(min_clearance, 3),
        "path_length_m": round(cam_path_m, 2),
        "object_coverage": round(len(visited) / max(len(planned), 1), 3),
        "segments": len(segments),
        "subject_present": scene.subject is not None,
    }


def verify_collision_free(segments: list, scene: Scene, sdf: SceneSDF,
                          fps_mult: int = 2) -> dict:
    """Dense post-optimization collision audit — the plan's hard gate.

    Re-samples every segment at ``fps_mult``× the export frame rate (catches
    collisions between the optimizer's sample points) against the static SDF
    and, when a subject is present, the per-time subject capsule. Returns a
    report dict: ``status`` is "ok" or "failed"; ``violations`` carries one
    entry per colliding segment with the worst sample's time, position,
    penetration depth and the nearest scene element (the "reason").
    """
    labels = ([f"structure {s.kind} '{s.id}'" for s in scene.structures]
              + [f"object '{o.label}' ({o.id})" for o in scene.objects])
    violations = []
    for seg in segments:
        th, t_abs = seg.sample_times(fps=C.DEFAULT_FPS * fps_mult)
        pos = seg.positions(th, None, scene.subject).numpy()
        box_clearance, nearest_box = sdf.numpy_query_nearest(pos)
        if sdf.cloud is not None:                           # real-cloud geometry
            box_clearance = np.minimum(box_clearance, sdf.cloud.numpy_query(pos))
        subject_is_nearest = np.zeros(len(box_clearance), dtype=bool)
        if scene.subject is not None:
            capsule_dist = scene.subject.capsule_sdf_rows(pos, t_abs)
            subject_is_nearest = capsule_dist < box_clearance
            box_clearance = np.minimum(box_clearance, capsule_dist)
        colliding = box_clearance < 0.0
        if not colliding.any():
            continue
        worst_idx = int(np.argmin(np.where(colliding, box_clearance, np.inf)))
        if subject_is_nearest[worst_idx]:
            nearest = f"subject '{scene.subject.subject_id}'"
        elif sdf.cloud is not None and \
                sdf.cloud.numpy_query(pos[worst_idx:worst_idx + 1])[0] < box_clearance[worst_idx]:
            nearest = "point cloud (real geometry)"
        elif nearest_box[worst_idx] < len(labels):
            nearest = labels[nearest_box[worst_idx]]
        else:
            nearest = "unknown"
        violations.append({
            "segment": seg.name,
            "type": seg.type,
            "frames_colliding": int(colliding.sum()),
            "frames_total": int(len(box_clearance)),
            "worst_clearance_m": round(float(box_clearance[worst_idx]), 4),
            "worst_t": round(float(t_abs[worst_idx]), 3),
            "worst_pos": [round(float(x), 3) for x in pos[worst_idx]],
            "nearest_obstacle": nearest,
        })
    ok = not violations
    reasons = [
        f"segment {v['segment']} ({v['type']}): {v['frames_colliding']}/{v['frames_total']} "
        f"采样点碰撞，最深穿透 {abs(v['worst_clearance_m']):.3f}m @ t={v['worst_t']}s "
        f"pos={v['worst_pos']}，最近障碍 {v['nearest_obstacle']}"
        for v in violations
    ]
    return {"status": "ok" if ok else "failed",
            "fps_used": C.DEFAULT_FPS * fps_mult,
            "violations": violations, "reasons": reasons}


# --------------------------------------------------------------------------- #
# Stage entry
# --------------------------------------------------------------------------- #
def run(cfg=None, scene_path=None, traj_path=None, device=None,
        max_iters=None, force=False, collision_scene: str | None = None) -> dict:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    scene = load_scene(scene_path or ct.get("scene", ""))
    out_dir = _out_dir(cfg, scene)
    traj_path = Path(traj_path) if traj_path else out_dir / "traj_init.json"
    out_path = out_dir / "traj_final.json"
    audit = None
    if collision_scene:
        audit_scene = load_scene(collision_scene)
        audit = (SceneSDF(audit_scene.collision_obbs(), audit_scene.room_bounds()),
                 audit_scene)
        print(f"🔍 碰撞审计场景: {audit_scene.scene_id}（帧标记/指标按该场景区，自身场景作为对照）")

    init = load_json(traj_path)
    segments = [Segment.from_dict(d) for d in init["segments"]]
    print(f"🛡️  两阶段优化: {len(segments)} 段（device={device or 'cpu'}）")

    ocfg = OptCfg(ct.get("optimizer", {}))
    if device:
        ocfg.device = device
    if max_iters:
        ocfg.max_iters = int(max_iters)

    from cinema_traj.common.sdf import scene_sdf_cached
    sdf = scene_sdf_cached(scene,
                            soft_min_k=ct.get("optimizer", {}).get("soft_min_k"))
    if sdf.cloud is not None:
        grid_shape = sdf.cloud.shape
        print(f"☁️  点云碰撞场: {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]} "
              f"@ {sdf.cloud.voxel}m（真实云几何并入碰撞/遮挡）")

    # occlusion judge: same field MINUS the subject's own person_<id> cast
    # obstacle — the look target (head key point) sits ~0.13 m BELOW that
    # OBB's top face, so counting it as an occluder fires a permanent
    # occlusion cost on every subject shot. Measured consequence: the
    # optimizer drives the camera LOW to steepen the ray through her own
    # body box — into base+IK unreachable geometry (orbit radius 0.9 at
    # height 0.305 = 25% unreachable). Collision still uses the FULL field
    # (the body remains a hard obstacle there).
    occl_sdf = sdf
    if scene.subject is not None and any(
            o.id.lower() == f"person_{scene.subject.subject_id.lower()}"
            for o in scene.objects):
        occl_obbs = [s.obb for s in scene.structures] + \
            [o.obb for o in scene.objects
             if o.id.lower() != f"person_{scene.subject.subject_id.lower()}"]
        occl_sdf = SceneSDF(occl_obbs, scene.room_bounds(),
                            soft_min_k=ct.get("optimizer", {}).get("soft_min_k"),
                            cloud=sdf.cloud)
        print(f"👁️  遮挡场剔除 person_{scene.subject.subject_id}"
              f"（主体自身 OBB 只算碰撞障碍，不算遮挡物）")

    # jarvis platform: stalled arcs get a discrete-α fallback scored on
    # (collision, reachability) — the callback keeps core/optimizer
    # platform-agnostic
    arc_refine = None
    jarvis_ctx = None
    if ct.get("platform", "drone") == "jarvis":
        from cinema_traj.core.jarvis import (build_base_sdf, jarvis_caps_from_config)
        jc = jarvis_caps_from_config(ct.get("jarvis", {}))
        base_sdf_j = build_base_sdf(scene)

        def _arc_evaluate(seg):
            import torch
            th, t_abs = seg.sample_times(fps=C.DEFAULT_FPS)
            pos = seg.positions(th, None, scene.subject).numpy()
            coll, _min = costs.collision_stats(
                torch.as_tensor(pos, dtype=torch.float64), sdf)
            rep = jarvis_ctx["verify"]([seg], scene.subject, base_sdf_j, jc)
            return {"collision": float(coll),
                    "reachable": float(rep["reachable_frac"])}

        from cinema_traj.core.jarvis import verify_reachable_trajectory as _vrt
        jarvis_ctx = {"verify": _vrt}
        arc_refine = _arc_evaluate

    # jarvis platform: the differential-base drivability cost (dead-zone
    # hinge + tangent-back base parking clearance) steers the optimizer
    # into shapes a diff base can actually drive — without it the export
    # gate rejects shapes after the fact and the planner has no recourse
    drivable = ct.get("platform", "drone") == "jarvis"
    segments, log = optimize_trajectory(segments, scene, sdf, scene.subject, ocfg,
                                        arc_refine=arc_refine, drivable_on=drivable,
                                        occl_sdf=occl_sdf)

    # hard gate: dense re-sample against the optimizing field (incl. the real
    # cloud + subject capsule); with --collision-scene the AUDIT field judges
    gate_sdf = audit[0] if audit else sdf
    gate_scene = audit[1] if audit else scene
    verification = verify_collision_free(segments, gate_scene, gate_sdf)

    # jarvis platform: reachability is the second hard gate — retiming can
    # slow a trajectory but never fix an unreachable shape, so every sampled
    # waypoint must admit a (base, arm) configuration
    jarvis_report = None
    if ct.get("platform", "drone") == "jarvis":
        from cinema_traj.core.jarvis import verify_reachable_trajectory
        print(f"🤖 Jarvis 可达性审计（相机带 [{jc.cinematic_z_lo:.2f}, "
              f"{jc.cinematic_z_hi:.2f}]m）…")
        jarvis_report = verify_reachable_trajectory(segments, scene.subject,
                                                    base_sdf_j, jc)
        # ARC (transition) segments are exempt from the hard gate: the
        # export's tiered mode parks them as DWELL/HOLD configurations
        # (ik_stream hold_segments=("arc",) + jarvis.allow_unreachable —
        # the camera stream stays exactly as planned, the robot pauses and
        # re-aims). Arcs are not shot, so a held waypoint is designed
        # behaviour, not a defect; SHOOTING segments stay hard-failed.
        arc_viol = [v for v in jarvis_report.get("violations", [])
                    if str(v.get("type", "")) == "arc"]
        shoot_viol = [v for v in jarvis_report.get("violations", [])
                      if str(v.get("type", "")) != "arc"]
        if arc_viol:
            print(f"   ⚠️  arc 过渡段 {len(arc_viol)} 个采样点不可达"
                  f"（导出时按驻足保持构型处理，非拍摄段）")
            jarvis_report = {**jarvis_report,
                             "violations": shoot_viol,
                             "reasons": [f'{v["segment"]} ({v["type"]}) @ t={v["t"]}s '
                                         f'pos={v["pos"]} 不可达（底盘落点+臂 IK 无解）'
                                         for v in shoot_viol],
                             "status": "failed" if shoot_viol else "ok"}
        if jarvis_report["status"] == "failed":
            verification["status"] = "failed"
            verification.setdefault("reasons", []).extend(jarvis_report["reasons"])

    frames, metrics = sample_frames(segments, scene, sdf, audit=audit)

    out = {
        "scene_id": scene.scene_id,
        "plan_id": init.get("plan_id"),
        "status": verification["status"],
        "verification": verification,
        "metrics": metrics,
        "optimization_log": log,
        "segments": [s.to_dict() for s in segments],
        "frames": frames,
    }
    if jarvis_report is not None:
        out["jarvis_reachability"] = jarvis_report
    atomic_write_json(out_path, out)
    if verification["status"] == "failed":
        print(f"❌ 规划失败: 轨迹存在碰撞/不可达 → {out_path}")
        for r in verification["reasons"]:
            print(f"   原因: {r}")
    else:
        print(f"✅ traj_final → {out_path}（碰撞终验通过）")
    fld = f"（审计 {audit_scene.scene_id}）" if audit else ""
    print(f"   碰撞率 {metrics['collision_rate']:.2%} | 遮挡率 {metrics['occlusion_rate']:.2%} "
          f"| 最小间隙 {metrics['min_clearance_m']}m | 路径长 {metrics['path_length_m']}m{fld}")
    if jarvis_report is not None:
        print(f"   🤖 可达率 {jarvis_report['reachable_frac']:.2%}"
              f"（{'✅ 全部可达' if jarvis_report['status'] == 'ok' else '❌ 存在不可达采样点'}）")
    if "self_scene" in metrics:
        s = metrics["self_scene"]
        print(f"   ↳ 自身场景 {s['scene_id']}: 碰撞 {s['collision_rate']:.2%} | "
              f"最小间隙 {s['min_clearance_m']}m")
    return out


def _out_dir(cfg, scene) -> Path:
    ct = cfg.get("cinema_traj", {})
    base = ct.get("outputs_dir", "cinema_traj_out")
    p = Path(base)
    if not p.is_absolute():
        from vlm_intention.common.config import repo_root
        p = repo_root() / p
    return p / scene.scene_id


if __name__ == "__main__":                                # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="两阶段 SDF 轨迹优化")
    ap.add_argument("--scene")
    ap.add_argument("--traj", help="traj_init.json 路径")
    ap.add_argument("--device", default=None, help="cpu | mps")
    ap.add_argument("--iters", type=int, help="每段最大迭代数")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, scene_path=a.scene, traj_path=a.traj, device=a.device,
        max_iters=a.iters, force=a.force)
