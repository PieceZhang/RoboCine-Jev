"""Jarvis export stage: traj_final.json → provably trackable camera stream +
base/arm configuration stream (platform ``jarvis``).

Pipeline (the jarvis twin of retime_stage.retime_all):

  1. frames → camera samples (export.frame_pose, schema unchanged);
  2. PER-SEGMENT re-time under :class:`JarvisCapabilities` — the SHARED retime
     loop stretches each segment's time axis independently until the
     Jacobian-projected ``v_max(û)`` holds (path shape never clipped); a slow
     arm-driven arc stretches only itself, never the shooting segments;
  3. per retimed sample: :func:`camera_reachable` (warm-started, previous
     (base yaw, arm extension) branch preferred) → base pose + arm joints.
     Holds are TIERED: an unreachable ARC waypoint dwells (robot pauses, config
     frozen) when ``jarvis.allow_unreachable``; an unreachable SHOOTING-segment
     waypoint always fails the export — the build gate exists so these never
     get here;
  4. exact verification: :func:`check_trackable` finite-differences the
     configuration stream against every cap — residual violations stretch the
     offending SEGMENT (pure time scale, geometry untouched) and redo;
  5. write:
       exports/traj_samples_retimed.json + exports/simulator_csv/camera_traj.csv
         — camera stream, schema IDENTICAL to the drone path, so the
         ``/cam_desired_traj`` consumption chain (traj_server → tracking node)
         works unchanged;
       exports/jarvis_traj.json      — t, base [x, z, yaw], q1..6, camera pose
                                        (the offline joint feedforward);
       exports/jarvis_setpoints.csv  — per-interval v_base / ω_base / q̇.

Run standalone or via run_traj.py --stage retime (platform: jarvis routes
here); the export stage triggers it automatically on overspeed setpoints.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common.scene import load_scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.core.jarvis import (
    Q_HI, Q_LO, JARVIS_DEMO_CAPS, JarvisCapabilities,
                                     check_trackable, config_commands,
                                     camera_reachable, jarvis_caps_from_config,
                                     trackability_factor)
from cinema_traj.core.retime import retime_trajectory
from cinema_traj.stages.export import frame_pose, write_ned_csv, write_ned_summary
from vlm_intention.common.atomic_io import atomic_write_json, load_json
from vlm_intention.common.config import load_config


# --------------------------------------------------------------------------- #
# Configuration stream IK
# --------------------------------------------------------------------------- #
def ik_stream(samples: list, sdf: SceneSDF, caps: JarvisCapabilities,
              allow_unreachable: bool = False,
              hold_segments: tuple = ("arc",)) -> tuple:
    """Per-sample (base pose, arm joints) via camera_reachable, warm-started
    and branch-pinned to the previous solution for continuity. Returns
    (configs, failures, holds) — configs rows ``{"t", "base", "q"}`` (None at
    failed samples, aligned with ``samples``). Stationary waypoints park the
    base at its previous heading (the arm sweeps, the base does not spin).

    Holds are TIERED: an unreachable sample whose segment label matches
    ``hold_segments`` (default: transition arcs only) HOLDS the previous
    configuration when ``allow_unreachable`` — the robot dwells (a cinematic
    pause + re-aim) while the camera stream stays exactly as planned. An
    unreachable SHOOTING-segment sample is a failure in every mode: the build
    gate exists precisely so these never reach the export."""
    configs: list = []
    failures: list = []
    holds = 0
    prev_q = None
    prev_branch = None                        # (absolute base yaw, d)
    prev_config = None
    prev_pos = None

    # BATCHED PRE-PASS over all samples (tiered disc oracle): trusted reps
    # are verified solutions carrying their base pose; the sequential
    # branch-pinned chain below runs ONLY for rows the batch cannot
    # certify (rng₅-only or nothing) — where the warm-start chain was
    # load-bearing. Removes the 3-fallback sequential search on healthy
    # streams (single-move 横移 paid ~36 s across the stretch rounds).
    bres = None
    try:
        from cinema_traj.core.jarvis_batched import camera_reachable_batch
        _poses = np.array([[s["x"], s["y"], s["z"]] for s in samples],
                          dtype=np.float64)
        _yaws = np.array([math.radians(float(s["yaw_deg"])) for s in samples])
        _pitches = np.array([math.radians(float(s["pitch_deg"]))
                             for s in samples])
        _tans = np.full(len(samples), np.nan)
        for k, s in enumerate(samples):
            prv = samples[max(k - 1, 0)]
            nxt = samples[min(k + 1, len(samples) - 1)]
            v = np.array([nxt["x"] - prv["x"], nxt["z"] - prv["z"]])
            if float(np.linalg.norm(v)) > 1e-4:
                _tans[k] = math.atan2(v[0], v[1])
        bres = camera_reachable_batch(_poses, _yaws, _pitches, _tans,
                                      sdf, caps)
    except Exception:
        bres = None
    for k, s in enumerate(samples):
        pos = np.array([s["x"], s["y"], s["z"]])
        yaw = math.radians(float(s["yaw_deg"]))
        pitch = math.radians(float(s["pitch_deg"]))
        # base heading: path tangent (the base must drive its path) —
        # CENTER-differenced (prev↔next), matching the plan gate's tangent
        # computation. The retimed stream's local speed profile makes a
        # FORWARD difference at mid-segment acceleration peaks deviate up
        # to ~23° from the geometric tangent (measured: swing t=0.75,
        # −78° forward vs −91° centered → −78° has no base+arm solution,
        # −101° does; the plan gate passed the same pose at −91°). First
        # sample keeps the forward neighbour (no prev yet); stationary
        # samples keep the base parked.
        tangent = yaw
        if k == 0:
            nxt = samples[min(k + 1, len(samples) - 1)]
            v = np.array([nxt["x"] - s["x"], nxt["z"] - s["z"]])
            if float(np.linalg.norm(v)) > 1e-4:
                tangent = math.atan2(v[0], v[1])
        else:
            prv = samples[k - 1]
            nxt = samples[min(k + 1, len(samples) - 1)]
            v = np.array([nxt["x"] - prv["x"], nxt["z"] - prv["z"]])
            if float(np.linalg.norm(v)) > 1e-4:
                tangent = math.atan2(v[0], v[1])
            elif prev_branch is not None:
                tangent = None                # stationary: keep the base parked
        r = None
        if bres is not None and bres["reachable"][k] and \
                bres["base"][k] is not None:
            _b = bres["base"][k]
            r = {"base": (_b[0], _b[1], _b[2]), "q": np.asarray(bres["q"][k]),
                 "d": 0.0, "yaw_offset": 0.0}
            if tangent is None:
                tangent = _b[2]
        if r is None:
            r = camera_reachable(pos, yaw, pitch, sdf, tangent_yaw=tangent,
                                 seed_q=prev_q, caps=caps, prefer=prev_branch)
        if r is None:
            # branch-pinned warm start can dead-end on isolated samples
            # (measured: swing t=0.75 — the plan gate's COLD solve passes
            # the same pose). Retry fully cold before declaring failure: a
            # cold solution proves the pose reachable; branch continuity
            # recovers on the next sample.
            r = camera_reachable(pos, yaw, pitch, sdf, tangent_yaw=tangent,
                                 caps=caps)
        if r is None:
            # second fallback: park behind the LOOK (stationary convention)
            # — retimed join rows can produce transient tangents whose
            # driving-park candidates all miss (measured: truck t=15.56
            # cold-solves OK under every probed tangent, so the failure is
            # a stream-transient, not geometry)
            r = camera_reachable(pos, yaw, pitch, sdf, tangent_yaw=None,
                                 caps=caps)
        seg_label = str(s.get("segment") or "")
        if r is None:
            if allow_unreachable and prev_config is not None and \
                    seg_label.startswith(hold_segments):
                holds += 1                    # dwell: freeze the last valid config
                configs.append({"t": float(s["t"]), "base": prev_config["base"],
                                "q": prev_config["q"]})
                continue
            failures.append({"index": k, "t": s["t"],
                             "pos": [s["x"], s["y"], s["z"]],
                             "segment": seg_label})
            configs.append(None)
            continue
        # clip arm joints onto the URDF box: arm_ik's damped solve accepts
        # poses within its position tolerance, which can leave a joint a few
        # 1e-4 rad PAST a limit (measured: q3=2.6184 vs hi 2.618 from the
        # look-park fallback) — check_trackable then rejects the stream.
        q_c = np.clip(np.asarray(r["q"], dtype=np.float64),
                       np.asarray(Q_LO) + 1e-4, np.asarray(Q_HI) - 1e-4)
        prev_q = q_c
        prev_branch = (r["base"][2], r["d"])
        prev_pos = pos
        prev_config = {"base": r["base"], "q": q_c}
        configs.append({"t": float(s["t"]), "base": r["base"], "q": q_c})
    return configs, failures, holds


def _insert_join_turns(samples: list, configs: list, caps) -> tuple:
    """Turn-in-place transitions at segment joins whose base step is not
    drivable.

    Per-segment IK re-parks the base at every join (TANGENT-parked while a
    segment drives vs LOOK-parked at rest) — the joined stream's base
    heading/position then STEP by up to ~0.8 rad / ~0.2 m between two
    adjacent configs. No differential base can execute that step: the
    tracker pivots in place instead of translating and rides tens of cm
    off the line for the whole next segment (measured R36-R48). The camera
    is AT REST at every join, so the repair is exact: ramp rows (camera
    static, base position+yaw linear) spanning a window that EXCEEDS the
    available gap, and the timeline after the join shifts by the NET
    extension (turn_t − gap) so times stay MONOTONIC. A first version
    overlapped the ramp with the successor's raw timestamp — a
    non-monotonic ff stream that corrupted every downstream segment
    (worse than no repair). Returns (configs, samples, n_rows)."""
    dt_out = caps.command_period
    join_need = {}                            # join index i → turn_t
    for i in range(len(configs) - 1):
        ca, cb = configs[i], configs[i + 1]
        if ca is None or cb is None:
            continue
        # ONLY true segment joins (label gate on the parallel samples:
        # config rows carry no segment label)
        if (str(samples[i].get("segment") or "") ==
                str(samples[min(i + 1, len(samples) - 1)].get("segment") or "")):
            continue
        pa, pb = np.asarray(ca["base"], dtype=np.float64), \
            np.asarray(cb["base"], dtype=np.float64)
        dyaw = abs(_wrap2(float(pb[2]) - float(pa[2])))
        dpos = float(np.hypot(*(pb[:2] - pa[:2])))
        gap = max(float(cb["t"]) - float(ca["t"]), dt_out)
        need = max(dyaw / 0.40, dpos / 0.15, 0.0)
        if need > gap - 1e-9:
            join_need[i] = max(need + 0.5, 1.0)
    if not join_need:
        return configs, samples, 0
    out_s: list = []
    out_c: list = []
    shift = 0.0
    n = 0
    for i in range(len(configs)):
        ca = configs[i]
        out_s.append(dict(samples[i],
                          t=round(float(samples[i]["t"]) + shift, 6)))
        out_c.append(dict(ca, t=round(float(ca["t"]) + shift, 6)))
        if i in join_need:
            cb = configs[i + 1]
            pa = np.asarray(ca["base"], dtype=np.float64)
            pb = np.asarray(cb["base"], dtype=np.float64)
            turn_t = join_need[i]
            gap = max(float(cb["t"]) - float(ca["t"]), dt_out)
            n_rows = max(int(round(turn_t / dt_out)), 2)
            sa = samples[i]
            t0 = float(ca["t"]) + shift
            for j in range(1, n_rows + 1):     # rows strictly inside (ca, cb']
                f = j / (n_rows + 1)
                t_j = round(t0 + f * turn_t, 6)
                base_j = [pa[0] + f * (pb[0] - pa[0]),
                          pa[1] + f * (pb[1] - pa[1]),
                          float(pa[2]) + f * _wrap2(float(pb[2]) - float(pa[2]))]
                q_j = [a + f * (b - a) for a, b in zip(ca["q"], cb["q"])]
                out_s.append(dict(sa, t=t_j))  # camera static at the join
                out_c.append({"t": t_j,
                              "base": [round(v, 4) for v in base_j],
                              "q": [round(v, 4) for v in q_j],
                              "cam": sa.get("cam") or [sa["x"], sa["y"], sa["z"]],
                              "yaw_deg": sa["yaw_deg"],
                              "pitch_deg": sa["pitch_deg"],
                              "segment": "join_turn"})
                n += 1
            shift += turn_t - gap              # NET timeline extension
    return out_c, out_s, n


def _wrap2(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _split_by_segment(samples: list) -> list:
    """Consecutive runs of one segment label → list of sample groups."""
    groups: list = []
    for s in samples:
        label = str(s.get("segment") or "")
        if groups and groups[-1][0] == label:
            groups[-1][1].append(s)
        else:
            groups.append((label, [s]))
    return groups


def _segment_runs(samples: list) -> list:
    """(label, first_index, last_index) per consecutive segment run."""
    runs: list = []
    for i, s in enumerate(samples):
        label = str(s.get("segment") or "")
        if runs and runs[-1][0] == label:
            runs[-1][2] = i
        else:
            runs.append([label, i, i])
    return [(label, lo, hi) for label, lo, hi in runs]


def _retime_per_segment(samples: list, caps, dt_out: float) -> list:
    """Retime each shooting/transition segment INDEPENDENTLY and rejoin.

    A global retime lets one slow segment (an arc whose look swing is
    arm-driven) inflate the whole movie (kujiale: ×162, 2764.8 s). Segments
    start/end at rest (ease_in_out + the retime loop's rest-at-ends rule), so
    independent stretches rejoin cleanly. The look yaw/pitch are CHAINED via
    the slew warm start (yaw0/pitch0) — a look-target switch at a segment
    boundary ramps while at rest instead of teleporting the base parking
    spot. Each group is shifted to max(its original start, the previous
    group's stretched end) — the at-rest join turns any gap into a harmless
    dwell."""
    out: list = []
    t_cursor = None
    warm = (None, None)                       # (yaw, pitch) from the last group
    for label, group in _split_by_segment(samples):
        if len(group) < 2:                    # single-frame group → dwell pad
            s0 = dict(group[0])
            s1 = dict(s0, t=round(float(s0["t"]) + max(dt_out, 0.05), 6))
            group = [s0, s1]
        res_g = retime_trajectory(group, caps, dt_out=dt_out, keep_duration=False,
                                  yaw0=warm[0], pitch0=warm[1])
        # keep_duration=False re-profiles TIME-OPTIMALLY: a short relocation
        # comes back FASTER than planned (measured: a 0.5 m arc_1 hop, 9 s in
        # the plan, re-profiled to 2.2 s — the robot must then sprint, brake
        # and re-park at full caps, and the tracker lags it by 35-58 cm).
        # The plan's durations ARE the cinematography (运镜节奏): stretch any
        # segment that came back shorter back to its planned span. Slower is
        # always cap-safe, so the stretch is provably trackable.
        in_span = float(group[-1]["t"]) - float(group[0]["t"])
        out_span = float(res_g.samples[-1]["t"]) - float(res_g.samples[0]["t"])
        if out_span < in_span - 1e-6 and out_span > 1e-9:
            f = in_span / out_span
            res_g.samples = [dict(s, t=round(float(s["t"]) * f, 6))
                             for s in res_g.samples]
        # a stretched predecessor may overrun the successor's raw start, and
        # the successor's first sample (t=0 in its own frame) must not share
        # a timestamp with it — the at-rest join absorbs a one-period dwell
        t_off = float(group[0]["t"]) if t_cursor is None else \
            max(float(group[0]["t"]), t_cursor + dt_out)
        out.extend(dict(s, t=round(float(s["t"]) + t_off, 6))
                   for s in res_g.samples)
        t_cursor = float(out[-1]["t"])
        last = out[-1]
        warm = (math.radians(float(last["yaw_deg"])),
                math.radians(float(last["pitch_deg"])))
    return out


def jarvis_retime_all(traj_final: dict, out_dir: Path, caps: JarvisCapabilities,
                      sdf: SceneSDF, dt_out: float | None = None,
                      max_rounds: int = 4,
                      allow_unreachable: bool = False) -> dict:
    """Retime + IK + exact trackability verification (see module docstring).

    Per-segment time laws: each segment is retimed independently (arcs get
    their own budget — a slow arm-driven look swing must not inflate the
    movie), and the residual exact-verification stretch is applied per
    segment too. Shooting-segment unreachables always fail; arc unreachables
    dwell (hold) when ``allow_unreachable``."""
    frames = traj_final.get("frames", [])
    if len(frames) < 2:
        raise ValueError("traj_final needs >= 2 frames to retime")
    samples = [frame_pose(f) for f in frames]

    out_samples = _retime_per_segment(samples, caps,
                                      dt_out if dt_out is not None
                                      else caps.command_period)
    configs, failures, holds = ik_stream(out_samples, sdf, caps,
                                         allow_unreachable=allow_unreachable)
    if not failures:
        configs, out_samples, n_turns = _insert_join_turns(
            out_samples, configs, caps)
        if n_turns:
            print(f"   ↻ 段接缝原地转向过渡 {n_turns} 处（底盘航向/位置跳变 "
                  f"→ 相机静止窗口内斜坡过渡）")
    if failures:
        f0 = failures[0]
        shooting = not str(f0.get("segment") or "").startswith("arc")
        hint = ("拍摄段不可达 — build 的可达性硬门应已拦截此情形，请检查该段钳制/"
                "审计（严禁对拍摄段保持构型）" if shooting else
                "arc 过渡不可达 — 配置 jarvis.allow_unreachable: true 以驻足保持构型")
        raise ValueError(
            "jarvis export: %d/%d 个采样点不可达（首个 t=%.2fs pos=%s segment=%s）— %s"
            % (len(failures), len(out_samples), f0["t"],
               [round(v, 3) for v in f0["pos"]], f0.get("segment") or "?", hint))

    # exact verification on the configuration stream. Residual violations
    # (hold-resume jumps, reference-J vs true-J residue, look switches at
    # segment joins) are absorbed by PER-INTERVAL DWELL INSERTION: an interval
    # whose config step exceeds a cap gets its time gap widened (need_dt =
    # current_dt × ratio), shifting only the tail — camera positions never
    # move, the path shape is exact, and one pass fixes every interval
    # independently (each ratio depends only on its own step and gap).
    warns = check_trackable(out_samples, configs, caps)
    for _ in range(max_rounds):
        if not warns:
            break
        new_samples: list = [out_samples[0]]
        new_configs: list = [configs[0]]
        t_prev = float(configs[0]["t"])
        for k in range(1, len(configs)):
            cur_dt = float(configs[k]["t"]) - float(configs[k - 1]["t"])
            if cur_dt <= 1e-9:
                cur_dt = caps.command_period          # degenerate join
            syn_cur = dict(configs[k], t=t_prev + cur_dt)
            ratio = trackability_factor([new_configs[-1], syn_cur], caps)
            need_dt = cur_dt * ratio * 1.02
            t_prev = t_prev + max(cur_dt, need_dt)
            new_samples.append(dict(out_samples[k], t=round(t_prev, 6)))
            new_configs.append(dict(configs[k], t=round(t_prev, 6)))
        out_samples, configs = new_samples, new_configs
        warns = check_trackable(out_samples, configs, caps)
    if warns:                                     # pragma: no cover
        raise AssertionError("jarvis export: 配置流仍超出能力限制:\n  "
                             + "\n  ".join(warns[:10]))
    duration = float(out_samples[-1]["t"])
    scale = duration / max(float(samples[-1]["t"]), 1e-9)
    commands = config_commands(configs)

    exp = out_dir / "exports"
    exp.mkdir(parents=True, exist_ok=True)

    # -- retimed generic camera JSON (schema unchanged → /cam_desired_traj) -- #
    doc = {"scene_id": traj_final.get("scene_id"),
           "frame": {"up": "+Y", "units": "meters", "yaw_zero": "+Z",
                     "note": "world frame; yaw 0 → +Z, positive toward +X, roll 0"},
           "metrics": traj_final.get("metrics"),
           "retime": {"platform": "jarvis",
                      "duration_s": round(duration, 3),
                      "scale": round(scale, 3),
                      "caps": {"max_base_linear_vel": caps.max_base_linear_vel,
                               "max_base_angular_vel": caps.max_base_angular_vel,
                               "max_joint_vel": list(caps.max_joint_vel)}},
           "count": len(out_samples), "samples": out_samples}
    atomic_write_json(exp / "traj_samples_retimed.json", doc)

    # -- base/arm configuration stream --------------------------------------- #
    jdoc = {"scene_id": traj_final.get("scene_id"),
            "frame": {"up": "+Y", "units": "meters", "yaw_zero": "+Z",
                      "base": "floor plane (x, z, yaw), y = 0",
                      "q": "joint1..joint6 (rad, URDF jvs_mock boxes)"},
            "robot": {"source": "ros2_ws/jvs_mock @ yuelin/isaac-robot-controller",
                      "wheel_separation": 0.21, "wheel_radius": 0.07},
            "trackable": not warns,
            "held_configs": holds,
            "duration_s": round(duration, 3), "scale": round(scale, 3),
            "count": len(configs),
            "configs": [{"t": c["t"],
                         "base": [round(float(c["base"][0]), 4),
                                  round(float(c["base"][1]), 4),
                                  round(float(c["base"][2]), 4)],
                         "q": [round(float(v), 4) for v in c["q"]],
                         "cam": [round(float(s["x"]), 4), round(float(s["y"]), 4),
                                 round(float(s["z"]), 4)],
                         "yaw_deg": s["yaw_deg"], "pitch_deg": s["pitch_deg"],
                         "segment": s.get("segment") or ""}
                        for c, s in zip(configs, out_samples)]}
    atomic_write_json(exp / "jarvis_traj.json", jdoc)

    # -- feedforward setpoints CSV ------------------------------------------- #
    with open(exp / "jarvis_setpoints.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# jarvis feedforward (ZOH over [t_k, t_k+1]): t_s",
                    "base_x_m", "base_z_m", "base_yaw_deg",
                    "v_base_mps", "omega_base_radps",
                    "qdot1_radps", "qdot2_radps", "qdot3_radps",
                    "qdot4_radps", "qdot5_radps", "qdot6_radps", "segment"])
        for c, cmd in zip(configs, commands):
            w.writerow([c["t"], round(float(c["base"][0]), 4),
                        round(float(c["base"][1]), 4),
                        round(math.degrees(float(c["base"][2])), 3),
                        round(float(cmd["v_base"]), 4),
                        round(float(cmd["omega_base"]), 4),
                        *[round(float(v), 4) for v in cmd["qdot"]],
                        ""])
        # terminal row: parked base, zero rates
        last = configs[-1]
        w.writerow([last["t"], round(float(last["base"][0]), 4),
                    round(float(last["base"][1]), 4),
                    round(math.degrees(float(last["base"][2])), 3),
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""])

    # -- NED simulator CSV (camera stream, retimed = executable) -------------- #
    sim_dir = exp / "simulator_csv"
    sim_dir.mkdir(parents=True, exist_ok=True)
    entry = write_ned_csv(out_samples, None, sim_dir / "camera_traj.csv",
                          profile_name="jarvis_retimed",
                          extra_meta={"source": "cinema_traj jarvis retime",
                                      "retime_scale": round(scale, 6)})
    write_ned_summary(sim_dir / "summary.json", [entry],
                      notes="jarvis platform: camera stream (schema v1 unchanged); "
                            "base/arm configs in jarvis_traj.json")
    return {"exports_dir": str(exp), "samples": len(out_samples),
            "configs": len(configs), "held": holds,
            "duration": duration,
            "scale": scale, "ned_csv": str(sim_dir / "camera_traj.csv"),
            "jarvis_json": str(exp / "jarvis_traj.json"),
            "setpoints_csv": str(exp / "jarvis_setpoints.csv")}


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

    scene = load_scene(scene_path or ct.get("scene", ""))
    from cinema_traj.core.jarvis import build_base_sdf
    sdf = build_base_sdf(scene)          # base-placeability field (floor-free)
    jc = jarvis_caps_from_config(ct.get("jarvis", {}))
    rcfg = ct.get("retime", {})
    res = jarvis_retime_all(traj, out_dir, jc, sdf,
                            dt_out=float(rcfg["command_period"])
                            if rcfg.get("command_period") else None,
                            allow_unreachable=bool(
                                ct.get("jarvis", {}).get("allow_unreachable", False)))
    held = f"，保持构型 {res['held']} 点" if res["held"] else ""
    print(f"🤖 Jarvis 导出 → {res['exports_dir']}（{res['samples']} 个相机设定点 + "
          f"{res['configs']} 组底盘/臂构型，时长 {res['duration']:.1f}s "
          f"×{res['scale']:.2f}，check_trackable ✅{held}）")
    print(f"   构型流 → {res['jarvis_json']}")
    print(f"   前馈 CSV → {res['setpoints_csv']}")
    print(f"   NED simulator CSV（相机流，schema 不变）→ {res['ned_csv']}")
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

    ap = argparse.ArgumentParser(description="Jarvis 导出（构型流 + 可证可跟踪相机流）")
    ap.add_argument("--traj", help="traj_final.json 路径")
    ap.add_argument("--out-dir", help="输出目录（默认与 traj 同目录）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, traj_path=a.traj, out_dir=a.out_dir, force=a.force)
