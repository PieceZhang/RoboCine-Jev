#!/usr/bin/env python3
"""Inject a synthetic subject track (假人) into a scene JSON — two modes:

  walking : waypoints walked at a fixed speed, frames at dt=0.1, heading =
            travel direction; optional per-waypoint pauses
  static  : one standing pose

    # 走路：3 个路点，1.0 m/s，在第 2 个路点停 3 秒
    python scripts/add_subject_track.py --scene scene.json \
        --waypoints "-2,0 -2,2 1,2" --speed 1.0 --pause 0,3,0

    # 静态站立：面朝 30°（世界系 yaw，0 → +Z，正值偏向 +X）
    python scripts/add_subject_track.py --scene scene.json --static 1.0,1.5 --heading 30

heading 约定与 SubjectTrack/_heading_offset 一致：0 → +X，朝 +Z 增大
（atan2(dz, dx)）。写回前 backup_once 备份原 scene。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlm_intention.common.atomic_io import atomic_write_json, backup_once

DT = 0.1


def parse_waypoints(spec: str) -> list[tuple[float, float]]:
    """'x,z x,z …' → [(x, z), …]"""
    out = []
    for tok in spec.split():
        parts = tok.split(",")
        if len(parts) != 2:
            raise ValueError(f"waypoint must be 'x,z', got: {tok!r}")
        out.append((float(parts[0]), float(parts[1])))
    if len(out) < 2:
        raise ValueError("walking mode needs >= 2 waypoints")
    return out


def walk_frames(waypoints, speed: float, height: float, pauses, dt: float = DT):
    """Frames walking waypoint→waypoint at `speed`; hold position during pauses.
    heading = atan2(dz, dx) of each travel leg (0 → +X, toward +Z)."""
    frames, t = [], 0.0
    frames.append({"t": round(t, 3),
                   "pos": [waypoints[0][0], 0.0, waypoints[0][1]],
                   "heading": _leg_heading(waypoints[0], waypoints[1])})
    for i, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
        heading = _leg_heading(a, b)
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n_steps = max(1, int(math.ceil(dist / speed / dt))) if dist > 1e-9 else 1
        step_u = [k / n_steps for k in range(1, n_steps + 1)]
        for u in step_u:
            t += dist / speed / n_steps
            frames.append({"t": round(t, 3),
                           "pos": [a[0] + u * (b[0] - a[0]), 0.0,
                                   a[1] + u * (b[1] - a[1])],
                           "heading": heading})
        pause = pauses[i + 1] if i + 1 < len(pauses) else 0.0
        if pause > 0:                                       # hold at waypoint i+1
            for _ in range(int(round(pause / dt))):
                t += dt
                frames.append({"t": round(t, 3), "pos": [b[0], 0.0, b[1]],
                               "heading": heading})
    return frames


def _leg_heading(a, b) -> float:
    """Heading convention matching _heading_offset: 0 → +X, toward +Z."""
    return math.atan2(b[1] - a[1], b[0] - a[0])


def static_frame(x: float, z: float, heading_deg: float):
    return [{"t": 0.0, "pos": [x, 0.0, z], "heading": math.radians(heading_deg)}]


def build_subject_track(args) -> dict:
    if args.static:
        x, z = (float(v) for v in args.static.split(","))
        frames = static_frame(x, z, args.heading)
        mode = "static"
    else:
        waypoints = parse_waypoints(args.waypoints)
        pauses = [float(p) for p in args.pause.split(",")] if args.pause else []
        if len(pauses) > len(waypoints):
            raise ValueError(f"--pause has {len(pauses)} values for "
                             f"{len(waypoints)} waypoints")
        pauses += [0.0] * (len(waypoints) - len(pauses))
        frames = walk_frames(waypoints, args.speed, args.height, pauses)
        mode = "walk"
    return {"subject_id": args.subject_id, "height_m": args.height,
            "frames": frames, "key_points_local": {"head": 0.92, "chest": 0.72},
            "_mode": mode}


def main() -> None:
    ap = argparse.ArgumentParser(description="向 scene JSON 注入假人 subject_track")
    ap.add_argument("--scene", required=True, help="场景 JSON（原地更新）")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--waypoints", help='走路路点 "x,z x,z …"（世界系 XZ 平面）')
    mode.add_argument("--static", help='静态位置 "x,z"')
    ap.add_argument("--speed", type=float, default=1.0, help="步行速度 m/s")
    ap.add_argument("--height", type=float, default=1.75, help="身高 m")
    ap.add_argument("--pause", help="各路点停留秒数（逗号分隔，缺省 0）")
    ap.add_argument("--heading", type=float, default=0.0,
                    help="静态模式朝向（deg，世界系 yaw：0 → +Z，正偏向 +X）")
    ap.add_argument("--subject-id", default="subject_01")
    a = ap.parse_args()

    scene_path = Path(a.scene)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    track = build_subject_track(a)
    mode = track.pop("_mode")

    bak = backup_once(scene_path, tag="presubject")
    scene["subject_track"] = track
    atomic_write_json(scene_path, scene)

    n = len(track["frames"])
    p0, p1 = track["frames"][0]["pos"], track["frames"][-1]["pos"]
    dur = track["frames"][-1]["t"] - track["frames"][0]["t"]
    print(f"🚶 subject_track[{mode}] → {scene_path}（{n} 帧，{dur:.1f}s，"
          f"{p0[:3]} → {p1[:3]}，身高 {track['height_m']}m）")
    if bak:
        print(f"   备份: {bak}")


if __name__ == "__main__":
    main()
