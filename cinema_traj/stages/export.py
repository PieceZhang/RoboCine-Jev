"""Export stage: traj_final.json → generic samples JSON / CSV / drone setpoint CSV
/ NED simulator CSV (camera-motion-offline-trajectory/v1).

World frame is preserved (+Y up, meters, yaw=0 → +Z, yaw positive toward +X,
roll always 0). Platform adapters (MAVLink missions, ROS goals …) convert from
the generic table later. The NED CSV uses the third-party trajectory-simulator
exchange format: north=+X, east=+Z, down=−Y, yaw 0 → north (positive clockwise
from above), gimbal pitch positive up — conversions live in cinema_traj.core.retime
and are pinned by tests/test_retime.py.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common.robot_frame import quaternion_robot
from cinema_traj.common.se3 import look_at_R, rotation_to_yaw_pitch
from cinema_traj.core.retime import (world_yup_to_ned, world_yaw_to_ned_yaw,
                                     wrap_deg)
from vlm_intention.common.atomic_io import atomic_write_json, load_json
from vlm_intention.common.config import load_config

NED_CSV_COLUMNS = ["t_s", "north_m", "east_m", "down_m", "yaw_deg", "gimbal_pitch_deg",
                   "forward_mps", "right_mps", "down_mps", "yaw_rate_degps",
                   "gimbal_pitch_rate_degps"]


def frame_pose(frame: dict) -> dict:
    """One generic sample: t, position, yaw/pitch (deg, from look_at), fov,
    plus ``robot_quat_wxyz`` — the robot-frame camera orientation (X+ right,
    Y+ down, Z+ forward; the robot's definition), body→world, qw qx qy qz.
    The frame *declaration* lives once in the JSON ``frame`` note, not per row."""
    R = look_at_R(frame["pos"], frame["look"])
    yaw, pitch = rotation_to_yaw_pitch(R)
    q_wxyz = quaternion_robot(frame["pos"], frame["look"])
    return {
        "t": frame["t"],
        "x": frame["pos"][0], "y": frame["pos"][1], "z": frame["pos"][2],
        "yaw_deg": round(yaw, 2), "pitch_deg": round(pitch, 2), "roll_deg": 0.0,
        "robot_quat_wxyz": [round(v, 6) for v in q_wxyz],
        "fov_deg": frame.get("fov_deg") or 55.0,
        "segment": frame.get("segment"),
    }


def check_speed_limits(samples: list, max_speed: float, max_accel: float) -> list:
    """Warn (list of strings) where consecutive setpoints exceed limits."""
    warns = []
    for a, b in zip(samples, samples[1:]):
        dt = max(b["t"] - a["t"], 1e-6)
        v = math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"])) / dt
        if v > max_speed:
            warns.append(f"t={b['t']}: speed {v:.2f} m/s > {max_speed} ({b['segment']})")
    return warns


# --------------------------------------------------------------------------- #
# NED simulator CSV (camera-motion-offline-trajectory/v1) — replayable by the
# third-party Kotlin trajectory-simulator tooling
# --------------------------------------------------------------------------- #
def write_ned_csv(samples: list, commands: list | None, out_csv: Path,
                  trajectory_name: str = "cinema_traj",
                  profile_name: str = "raw", extra_meta: dict | None = None) -> dict:
    """Write one NED CSV (columns identical to the Kotlin offline exporter) and
    return its summary entry. ``commands`` (ZOH body rates per interval) is
    optional; without it the rate columns are derived from finite differences
    of the sample stream itself."""
    rows = []
    for k, s in enumerate(samples):
        north, east, down = world_yup_to_ned((s["x"], s["y"], s["z"]))
        c = commands[k] if commands and k < len(commands) else None
        if c is not None:
            fwd, right, dvel = c["forward"], c["right"], c["down"]
            yaw_rate, pitch_rate = c["yaw_rate_degps"], c["gimbal_pitch_rate_degps"]
        else:
            a = samples[max(k - 1, 0)]
            dt = max(s["t"] - a["t"], 1e-6)
            fwd = right = dvel = yaw_rate = pitch_rate = 0.0
            if k > 0:
                dn, de, dd = world_yup_to_ned((s["x"] - a["x"], s["y"] - a["y"],
                                               s["z"] - a["z"]))
                yaw_n = math.radians(world_yaw_to_ned_yaw(s["yaw_deg"]))
                cn, sn = math.cos(yaw_n), math.sin(yaw_n)
                fwd, right = cn * dn + sn * de, -sn * dn + cn * de
                dvel = dd / dt
                fwd, right = fwd / dt, right / dt
                yaw_rate = -wrap_deg(s["yaw_deg"] - a["yaw_deg"]) / dt
                pitch_rate = (s["pitch_deg"] - a["pitch_deg"]) / dt
        rows.append([s["t"], north, east, down,
                     world_yaw_to_ned_yaw(s["yaw_deg"]), s["pitch_deg"],
                     fwd, right, dvel, yaw_rate, pitch_rate])

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(NED_CSV_COLUMNS)
        for r in rows:
            w.writerow([f"{v:.9f}" for v in r])

    last = rows[-1]
    entry = {"motion": trajectory_name, "profile": profile_name,
             "duration_seconds": round(float(last[0]), 9),
             "sample_count": len(rows), "csv": out_csv.name,
             "final": {"north_m": round(float(last[1]), 9), "east_m": round(float(last[2]), 9),
                       "down_m": round(float(last[3]), 9), "yaw_deg": round(float(last[4]), 9),
                       "gimbal_pitch_deg": round(float(last[5]), 9)}}
    if extra_meta:
        entry.update(extra_meta)
    return entry


def write_ned_summary(summary_path: Path, entries: list, notes: str = "") -> None:
    summary_path.write_text(json.dumps({
        "schema": "camera-motion-offline-trajectory/v1",
        "coordinate_system": {"world": "NED", "body": "FRD"},
        "ideal_kinematics_only": True,
        "notes": notes,
        "trajectories": entries}, ensure_ascii=False, indent=2), encoding="utf-8")


def export_all(traj_final: dict, out_dir: Path, control_dt: float = 0.1,
               max_speed: float = 2.0, max_accel: float = 2.0) -> dict:
    frames = traj_final.get("frames", [])
    if not frames:
        raise ValueError("traj_final has no frames")
    samples = [frame_pose(f) for f in frames]
    exp = out_dir / "exports"
    exp.mkdir(parents=True, exist_ok=True)

    # -- generic JSON -------------------------------------------------------- #
    doc = {"scene_id": traj_final.get("scene_id"),
           "frame": {"up": "+Y", "units": "meters", "yaw_zero": "+Z",
                     "note": "world frame; yaw 0 → +Z, positive toward +X, roll 0; "
                             "robot_quat_wxyz = camera orientation in the robot's "
                             "body frame (X+ right, Y+ down, Z+ forward), body→world"},
           "metrics": traj_final.get("metrics"),
           "count": len(samples), "samples": samples}
    atomic_write_json(exp / "traj_samples.json", doc)

    # -- generic CSV ---------------------------------------------------------- #
    with open(exp / "traj_samples.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["t", "x", "y", "z", "yaw_deg", "pitch_deg",
                                           "roll_deg", "robot_quat_wxyz", "fov_deg",
                                           "segment"])
        w.writeheader()
        w.writerows(samples)

    # -- drone setpoints (position + yaw; speed-limited check) ---------------- #
    warns = check_speed_limits(samples, max_speed, max_accel)
    with open(exp / "drone_setpoints.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# setpoint stream: t_s", "x_m", "y_m(up)", "z_m", "yaw_deg", "comment"])
        for s in samples:
            w.writerow([s["t"], s["x"], s["y"], s["z"], s["yaw_deg"],
                        s["segment"] or ""])

    # -- NED simulator CSV (v1 exchange format) -------------------------------- #
    sim_dir = exp / "simulator_csv"
    sim_dir.mkdir(parents=True, exist_ok=True)
    entry = write_ned_csv(samples, None, sim_dir / "camera_traj.csv",
                          extra_meta={"source": "cinema_traj export (pre-retime)"})
    write_ned_summary(sim_dir / "summary.json", [entry],
                      notes="no roll (always 0) / fov columns in CSV; yaw 0 → north, "
                            "positive clockwise from above; gimbal pitch positive up")
    return {"exports_dir": str(exp), "samples": len(samples), "warnings": warns,
            "ned_csv": str(sim_dir / "camera_traj.csv")}


# --------------------------------------------------------------------------- #
# Stage entry
# --------------------------------------------------------------------------- #
def run(cfg=None, traj_path=None, out_dir=None, force=False,
        scene_path=None) -> dict:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    traj_path = Path(traj_path) if traj_path else None
    if traj_path is None:
        scene_id = Path(ct.get("scene", "demo_apartment.json")).stem
        traj_path = _base(cfg) / scene_id / "traj_final.json"
    traj = load_json(traj_path)
    out_dir = Path(out_dir) if out_dir else traj_path.parent

    ecfg = ct.get("export", {})
    platform = ct.get("platform", "drone")
    if platform == "jarvis":
        # the floor robot saturates far below the drone trigger — gate the
        # auto-retime on its base cap so /cam_desired_traj never ships raw
        from cinema_traj.core.jarvis import jarvis_caps_from_config
        trigger_speed = jarvis_caps_from_config(ct.get("jarvis", {})).max_base_linear_vel
    else:
        trigger_speed = float(ecfg.get("max_speed", 2.0))
    res = export_all(traj, out_dir, control_dt=float(ecfg.get("control_dt", 0.1)),
                     max_speed=trigger_speed,
                     max_accel=float(ecfg.get("max_accel", 2.0)))
    print(f"📦 导出 → {res['exports_dir']}（{res['samples']} 个采样点）")
    print(f"   NED simulator CSV → {res['ned_csv']}")
    if res["warnings"]:
        n = len(res["warnings"])
        rcfg = ct.get("retime", {})
        if rcfg.get("auto", True):
            if platform == "jarvis":
                print(f"   ⚠️  {n} 处超过 {trigger_speed:.2f} m/s → 自动 Jarvis 重定时"
                      f"（构型流 IK + 可跟踪性验证）")
            else:
                print(f"   ⚠️  {n} 处超过速度限制 → 自动重定时（retime 阶段）")
            from cinema_traj.stages.retime_stage import run as run_retime
            run_retime(cfg, traj_path=traj_path, out_dir=out_dir, force=force,
                       scene_path=scene_path)
        else:
            print(f"   ⚠️  {n} 处超过速度限制（cinema_traj.retime.auto=false，"
                  f"运行 --stage retime 生成安全设定点）")
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

    ap = argparse.ArgumentParser(description="轨迹导出（JSON / CSV / 无人机设定点）")
    ap.add_argument("--traj", help="traj_final.json 路径")
    ap.add_argument("--out-dir", help="导出目录（默认与 traj 同目录）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, traj_path=a.traj, out_dir=a.out_dir, force=a.force)
