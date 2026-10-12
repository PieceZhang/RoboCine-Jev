#!/usr/bin/env python3
"""Convert a cinema_traj exported trajectory to the Jarvis (jvs_mock) arm
format: ``jvs_relative_task_space_trajectory/v1``.

PLANNER = Jarvis (2-wheel differential base + 6-axis arm, PR #11). The arm's
head camera follows the REP-103 OPTICAL convention — the URDF declares
``camera_optical_frame`` with rpy (−90°, 0, −90°) from ``camera_link``,
i.e. **Z+ FORWARD, X+ RIGHT, Y+ DOWN** — and that is the frame every pose in
the converted file is expressed in:

  * translation is the camera's motion in the STARTING optical frame;
  * rotation is the relative optical-frame orientation, as fixed-axis
    RPY [roll, pitch, yaw] consumed by ``jvs_kinematics.rpy_matrix``.

    python scripts/export_jarvis_trajectory.py \
        --traj cinema_traj_out/frl_apartment_0/exports/traj_samples.json \
        --out  cinema_traj_out/frl_apartment_0/exports/jarvis_task_space.json
    # → feed to: ros2 launch jvs_mock jvs_follow_traj.launch.py trajectory_file:=…

Resample rate: the jvs follower IK-solves every sample, so we default to
10 Hz (the schema's reference rate) regardless of the export density.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common.robot_frame import (  # noqa: E402
    quaternion_robot, rotation_robot)

FORMAT = "jvs_relative_task_space_trajectory/v1"
DEFAULT_RATE_HZ = 10.0
_world_up = np.array([0.0, 1.0, 0.0])


def _rotation_from_quaternion(q_wxyz) -> np.ndarray:
    """Body→world rotation matrix from (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rotation_to_rpy(R: np.ndarray) -> tuple:
    """Rotation matrix → fixed-axis (roll, pitch, yaw) — the inverse of
    ``jvs_kinematics.rpy_matrix`` (Rz(yaw)·Ry(pitch)·Rx(roll))."""
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:                     # gimbal lock: ±90°
        yaw = 0.0
        roll = math.atan2(-R[1, 2], R[1, 1]) if pitch > 0 else \
            math.atan2(R[1, 2], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def load_export_samples(path: Path) -> list:
    """traj_samples[_retimed].json → [{t, x, y, z, robot_quat_wxyz}]."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    frame_note = (doc.get("frame") or {}).get("note", "")
    samples = sorted(doc["samples"], key=lambda s: s["t"])
    if len(samples) < 2:
        raise ValueError(f"{path}: need >= 2 samples")
    missing = [s for s in samples if "robot_quat_wxyz" not in s]
    if missing:
        # retimed streams (retime_trajectory) rebuild sample dicts carrying
        # only yaw/pitch — rebuild the robot-frame quaternion from them using
        # the SAME look-at convention as export's frame_pose (world +Y up:
        # yaw = atan2(f.x, f.z), pitch = asin(f.y))
        if any("yaw_deg" not in s or "pitch_deg" not in s for s in missing):
            raise ValueError(
                f"{path}: samples lack robot_quat_wxyz AND yaw/pitch — re-run "
                "`run_traj.py --stage export` (robot camera frame missing)")
        for s in missing:
            yaw = math.radians(float(s["yaw_deg"]))
            pitch = math.radians(float(s["pitch_deg"]))
            fwd = np.array([math.sin(yaw) * math.cos(pitch),
                            math.sin(pitch),
                            math.cos(yaw) * math.cos(pitch)])
            pos = np.array([float(s["x"]), float(s["y"]), float(s["z"])])
            s["robot_quat_wxyz"] = [float(v) for v in
                                    quaternion_robot(pos, pos + fwd)]
    return samples, frame_note


def convert(samples: list, rate_hz: float = DEFAULT_RATE_HZ,
            frame_id: str = "base_link") -> dict:
    """Absolute robot-frame poses → relative xyz + RPY (jvs v1 schema).

    Positions come straight from the export; orientations come from the
    stored robot quaternions (optical frame). Sample i's relative pose is
    expressed in the OPTICAL FRAME OF SAMPLE 0 — exactly what
    ``relative_pose_target(initial_camera_link=…)``/``base_link`` compose
    back into absolute pose on the robot."""
    t_all = np.array([s["t"] for s in samples])
    t0, t1 = float(t_all[0]), float(t_all[-1])
    duration = t1 - t0
    n_out = max(int(round(duration * rate_hz)) + 1, 2)
    ts = np.linspace(t0, t1, n_out)

    def interp(key):
        vals = np.array([s[key] for s in samples])
        return np.interp(ts, t_all, vals, axis=0) if vals.ndim > 1 else \
            np.interp(ts, t_all, vals)

    xyz = np.stack([interp("x"), interp("y"), interp("z")], axis=1)

    # quaternions interpolate on the manifold: slerp via rotation matrices
    # (nlerp is not accurate enough for IK ground truth; matrices → matrix
    # log/exp would be heavier — short intervals make slerp-per-pair fine)
    quats = np.array([s["robot_quat_wxyz"] for s in samples])

    def q_to_R(q):
        return _rotation_from_quaternion(q)

    def slerp(qa, qb, u):
        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb, dot = -qb, -dot
        dot = min(dot, 1.0)
        theta = math.acos(dot)
        if theta < 1e-6:
            return qa + u * (qb - qa)
        s0 = math.sin((1 - u) * theta) / math.sin(theta)
        s1 = math.sin(u * theta) / math.sin(theta)
        return s0 * qa + s1 * qb

    # reference = sample 0's OPTICAL frame; every delta is composed in it
    R_ref = q_to_R(quats[0])
    R_ref_inv = R_ref.T
    out = []
    for k, t in enumerate(ts):
        # nearest bracket for orientation interpolation
        j = min(np.searchsorted(t_all, t), len(t_all) - 1)
        j0 = max(j - 1, 0)
        span = t_all[j] - t_all[j0] if j > j0 else 1.0
        u = 0.0 if span <= 0 else float(np.clip((t - t_all[j0]) / span, 0, 1))
        R_k = q_to_R(slerp(quats[j0], quats[j], u))
        # relative in the STARTING optical frame: ΔR = R_ref⁻¹·R_k,
        # Δp = R_ref⁻¹·(p_k − p_0)
        dp_world = xyz[k] - xyz[0]
        dp = R_ref_inv @ dp_world
        dR = R_ref_inv @ R_k
        # rpy_matrix returns R = Rz·Ry·Rx — decompose ΔR the same way
        roll, pitch, yaw = _rotation_to_rpy(dR)
        out.append({"time_from_start": round(float(t - t0), 6),
                    "pose": [round(float(dp[0]), 9), round(float(dp[1]), 9),
                             round(float(dp[2]), 9), round(roll, 9),
                             round(pitch, 9), round(yaw, 9)]})

    return {
        "format": FORMAT,
        "description": ("RoboCine camera trajectory converted for the Jarvis "
                        "arm (PR #11). Poses are RELATIVE to sample 0 and "
                        "expressed in the head-camera OPTICAL frame "
                        "(Z+ forward, X+ right, Y+ down) — the URDF's "
                        "camera_optical_frame convention."),
        "relative_to": "current_camera_link",
        "frame_id": frame_id,
        "dof_order": ["x", "y", "z", "roll", "pitch", "yaw"],
        "units": {"translation": "m", "rotation": "rad"},
        "camera_frame": "camera_optical_frame: Z+ FORWARD, X+ RIGHT, Y+ DOWN",
        "sample_rate_hz": float(rate_hz),
        "duration_s": round(duration, 6),
        "samples": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="cinema_traj exports → Jarvis jvs_relative_task_space/v1 "
                    "(optical frame Z+ fwd / X+ right / Y+ down)")
    ap.add_argument("--traj", required=True,
                    help="traj_samples[_retimed].json from the export stage")
    ap.add_argument("--out", required=True, help="output jvs trajectory JSON")
    ap.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ,
                    help=f"resample rate (default {DEFAULT_RATE_HZ})")
    ap.add_argument("--frame-id", default="base_link",
                    choices=["base_link", "initial_camera_link"])
    a = ap.parse_args()

    samples, _ = load_export_samples(Path(a.traj))
    doc = convert(samples, rate_hz=a.rate_hz, frame_id=a.frame_id)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"🤖 Jarvis trajectory → {a.out}")
    print(f"   {len(doc['samples'])} 样本 @ {doc['sample_rate_hz']}Hz · "
          f"{doc['duration_s']}s · frame_id={doc['frame_id']} · "
          f"光学系 {doc['camera_frame']}")


if __name__ == "__main__":
    main()
