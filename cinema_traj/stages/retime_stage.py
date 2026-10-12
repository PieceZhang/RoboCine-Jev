"""Retime stage: exported trajectory → capability-provable setpoints.

Loads traj_final.json, re-times the geometric path under drone caps (math
ported from the Kotlin trajectory-simulator's motion-math, see
cinema_traj/core/retime.py) and writes:

  exports/traj_samples_retimed.json   — same schema as traj_samples.json
  exports/drone_setpoints_retimed.csv — Y-up setpoints + ZOH body commands
  exports/simulator_csv/camera_traj.csv + summary.json — NED v1 format,
    OVERWRITTEN with the retimed stream (the executable trajectory)

Run standalone or via run_traj.py --stage retime. The export stage invokes it
automatically when the raw setpoints exceed speed limits.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.core.retime import (MINI3_DEMO_CAPS, DroneCapabilities,
                                     check_caps, retime_trajectory)
from cinema_traj.stages.export import frame_pose, write_ned_csv, write_ned_summary
from vlm_intention.common.atomic_io import atomic_write_json, load_json
from vlm_intention.common.config import load_config


def caps_from_config(rcfg: dict) -> DroneCapabilities:
    import math
    caps = DroneCapabilities(
        max_horizontal_speed=float(rcfg.get("max_horizontal_speed",
                                            MINI3_DEMO_CAPS.max_horizontal_speed)),
        max_ascent_speed=float(rcfg.get("max_ascent_speed",
                                        MINI3_DEMO_CAPS.max_ascent_speed)),
        max_descent_speed=float(rcfg.get("max_descent_speed",
                                         MINI3_DEMO_CAPS.max_descent_speed)),
        max_horizontal_acceleration=float(rcfg.get("max_horizontal_acceleration",
                                                   MINI3_DEMO_CAPS.max_horizontal_acceleration)),
        max_vertical_acceleration=float(rcfg.get("max_vertical_acceleration",
                                                 MINI3_DEMO_CAPS.max_vertical_acceleration)),
        max_yaw_rate=math.radians(float(rcfg.get("max_yaw_rate_deg", 20.0))),
        min_gimbal_pitch=math.radians(float(rcfg.get("min_gimbal_pitch_deg", -90.0))),
        max_gimbal_pitch=math.radians(float(rcfg.get("max_gimbal_pitch_deg", 60.0))),
        max_gimbal_pitch_rate=math.radians(float(rcfg.get("max_gimbal_pitch_rate_deg",
                                                          20.0))),
        command_period=float(rcfg.get("command_period", 0.05)))
    return caps


def retime_all(traj_final: dict, out_dir: Path, caps: DroneCapabilities,
               dt_out: float | None = None) -> dict:
    frames = traj_final.get("frames", [])
    if len(frames) < 2:
        raise ValueError("traj_final needs >= 2 frames to retime")
    samples = [frame_pose(f) for f in frames]
    res = retime_trajectory(samples, caps, dt_out=dt_out)

    violations = check_caps(res.samples, res.commands, caps)
    if violations:                                          # pragma: no cover
        raise AssertionError("retime produced cap violations:\n  "
                             + "\n  ".join(violations[:10]))

    exp = out_dir / "exports"
    exp.mkdir(parents=True, exist_ok=True)

    # -- retimed generic JSON ------------------------------------------------ #
    doc = {"scene_id": traj_final.get("scene_id"),
           "frame": {"up": "+Y", "units": "meters", "yaw_zero": "+Z",
                     "note": "world frame; yaw 0 → +Z, positive toward +X, roll 0"},
           "metrics": traj_final.get("metrics"),
           "retime": {"duration_s": round(res.duration, 3),
                      "scale": round(res.scale, 3),
                      "caps": {"max_horizontal_speed": caps.max_horizontal_speed,
                               "max_ascent_speed": caps.max_ascent_speed,
                               "max_descent_speed": caps.max_descent_speed}},
           "count": len(res.samples), "samples": res.samples}
    atomic_write_json(exp / "traj_samples_retimed.json", doc)

    # -- retimed Y-up setpoints + ZOH body commands --------------------------- #
    with open(exp / "drone_setpoints_retimed.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# retimed setpoint stream (ZOH body command held from t_k",
                    "to t_k+1): t_s", "x_m", "y_m(up)", "z_m", "yaw_deg",
                    "pitch_deg", "forward_mps", "right_mps", "down_mps",
                    "yaw_rate_degps", "gimbal_pitch_rate_degps", "segment"])
        for s, c in zip(res.samples, res.commands):
            w.writerow([s["t"], s["x"], s["y"], s["z"], s["yaw_deg"], s["pitch_deg"],
                        c["forward"], c["right"], c["down"],
                        c["yaw_rate_degps"], c["gimbal_pitch_rate_degps"],
                        s["segment"] or ""])

    # -- NED simulator CSV (retimed = executable trajectory) ------------------ #
    sim_dir = exp / "simulator_csv"
    sim_dir.mkdir(parents=True, exist_ok=True)
    entry = write_ned_csv(res.samples, res.commands, sim_dir / "camera_traj.csv",
                          profile_name="retimed",
                          extra_meta={"source": "cinema_traj retime",
                                      "retime_scale": round(res.scale, 6)})
    write_ned_summary(sim_dir / "summary.json", [entry],
                      notes="no roll (always 0) / fov columns in CSV; yaw 0 → north, "
                            "positive clockwise from above; gimbal pitch positive up")
    return {"exports_dir": str(exp), "samples": len(res.samples),
            "duration": res.duration, "scale": res.scale,
            "ned_csv": str(sim_dir / "camera_traj.csv")}


# --------------------------------------------------------------------------- #
# Stage entry
# --------------------------------------------------------------------------- #
def run(cfg=None, traj_path=None, out_dir=None, force=False,
        scene_path=None) -> dict:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    if ct.get("platform", "drone") == "jarvis":
        # platform switch: the jarvis chain retimes under Jacobian-projected
        # caps, IK's the base/arm configuration stream and verifies it exactly
        from cinema_traj.stages import jarvis_export
        return jarvis_export.run(cfg, traj_path=traj_path, out_dir=out_dir,
                                 force=force, scene_path=scene_path)
    traj_path = Path(traj_path) if traj_path else None
    if traj_path is None:
        scene_id = Path(ct.get("scene", "demo_apartment.json")).stem
        traj_path = _base(cfg) / scene_id / "traj_final.json"
    traj = load_json(traj_path)
    out_dir = Path(out_dir) if out_dir else traj_path.parent

    rcfg = ct.get("retime", {})
    caps = caps_from_config(rcfg)
    res = retime_all(traj, out_dir, caps,
                     dt_out=float(rcfg["command_period"]) if rcfg.get("command_period") else None)
    print(f"⏱️  重定时 → {res['exports_dir']}（{res['samples']} 个设定点 @ "
          f"{1.0 / caps.command_period:.0f} Hz，时长 {res['duration']:.1f}s "
          f"×{res['scale']:.2f}）")
    print(f"   NED simulator CSV（重定时后）→ {res['ned_csv']}")
    return res


def _base(cfg) -> Path:
    ct = cfg.get("cinema_traj", {})
    p = Path(ct.get("outputs_dir", "cinema_traj_out"))
    if p.is_absolute():
        return p
    from vlm_intention.common.config import repo_root
    return repo_root() / p


if __name__ == "__main__":                                # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="轨迹重定时（满足机体能力约束）")
    ap.add_argument("--traj", help="traj_final.json 路径")
    ap.add_argument("--out-dir", help="输出目录（默认与 traj 同目录）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, traj_path=a.traj, out_dir=a.out_dir, force=a.force)
