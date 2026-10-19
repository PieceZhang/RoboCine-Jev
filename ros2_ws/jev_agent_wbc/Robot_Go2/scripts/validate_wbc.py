#!/usr/bin/env python3
"""Measured Isaac Sim rollout of upstream Go2-Piper TorchScript WBC policy.

The base remains a floating PhysX articulation. Only reset initializes its pose.
No Agent, base teleport, prescribed root velocity or external support is used.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import math
import sys
import signal
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
faulthandler.register(signal.SIGUSR1, all_threads=False)
POLICY = ROOT / "vendor/LeggedManip_Lab/mujoco/deploy/policy/go2_piper/wbc/policy.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--seconds", type=float, default=8.0, help="Simulation seconds per case, including 2 s settling.")
parser.add_argument("--cases", default="stand,forward,backward,left,right,diagonal_left,diagonal_right,yaw_left,yaw_right,coordinated")
parser.add_argument("--speed", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--snapshot", action="store_true", help="Render one diagnostic PNG and save final USD.")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if not args.gui:
    args.headless = True
if args.snapshot:
    args.enable_cameras = True
if args.device is None:
    args.device = "cpu"
OUTPUT_DIR = (args.output or ROOT / "results" / f"run_{time.time_ns()}").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
(OUTPUT_DIR / "validator_source.py").write_text(Path(__file__).read_text())
(OUTPUT_DIR / "arguments.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
torch.set_num_threads(1)
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, quat_from_euler_xyz, quat_mul
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.go2_piper.wbc_env_cfg import Go2PiperWBCEnvCfg


def main():
    output = OUTPUT_DIR
    cfg = Go2PiperWBCEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.wait_for_textures = False
    cfg.num_rerenders_on_reset = 0
    if args.headless:
        # No RGB sensors participate in the control loop. Render once at the end.
        cfg.sim.render_interval = 1_000_000
    cfg.scene.terrain.terrain_type = "usd"
    cfg.scene.terrain.usd_path = str(ROOT / "assets/ground.usda")
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=750.0)
    cfg.observations.policy.enable_corruption = False
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (100000.0, 100000.0)
    cfg.commands.base_velocity.resampling_time_range = (100000.0, 100000.0)
    # Deterministic nominal dynamics, without training pushes or property changes.
    for name in vars(cfg.events):
        if not name.startswith("_"):
            setattr(cfg.events, name, None)
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.envs import mdp
    cfg.events.reset_base = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {}, "velocity_range": {}})
    cfg.events.reset_robot_joints = EventTermCfg(func=mdp.reset_joints_by_scale, mode="reset", params={
        "position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})
    # Never let an automatic reset conceal a fall. Failures are measured below.
    for name in vars(cfg.terminations):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    for name in vars(cfg.curriculum):
        if not name.startswith("_"):
            setattr(cfg.curriculum, name, None)
    cfg.episode_length_s = 1e6
    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]
    policy = torch.jit.load(str(POLICY), map_location=env.device).eval()
    joint_expr = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "joint.*"]
    joint_ids, joint_names = robot.find_joints(joint_expr, preserve_order=True)
    ee = env.command_manager.get_term("ee_pose")
    velocity = env.command_manager.get_term("base_velocity")
    commands = {
        "stand": [0, 0, 0], "forward": [args.speed, 0, 0], "backward": [-args.speed, 0, 0],
        "left": [0, args.speed, 0], "right": [0, -args.speed, 0],
        "diagonal_left": [args.speed, args.speed, 0], "diagonal_right": [args.speed, -args.speed, 0],
        "yaw_left": [0, 0, args.speed], "yaw_right": [0, 0, -args.speed],
        "coordinated": [args.speed, args.speed / 2, args.speed / 2],
    }
    selected = args.cases.split(",")
    unknown = set(selected) - commands.keys()
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    results, measurements = [], []
    wall_start = time.monotonic()
    for case in selected:
        print("CASE_RESET_BEGIN " + case, flush=True)
        with torch.inference_mode():
            env.reset(seed=args.seed)
        print("CASE_RESET_DONE " + case, flush=True)
        previous_action = torch.zeros((1, 18), device=env.device)
        histories = [torch.zeros((1, 3 * n), device=env.device) for n in [3, 3, 18, 18, 18, 3, 7]]
        rows = []
        expected_steps = round(args.seconds / env.step_dt)
        for step in range(expected_steps):
            tick = time.monotonic()
            t = step * env.step_dt
            moving_t = max(0.0, t - 2.0)
            command = commands[case] if t >= 2.0 else [0, 0, 0]
            velocity.vel_command_b[:] = torch.tensor([command], device=env.device)
            velocity.is_standing_env[:] = False
            # Published WBC command: xy and quaternion in link0, z in WORLD.
            x, y, z = 0.50, 0.0, 0.50
            if case == "coordinated" and t >= 2.0:
                x += 0.04 * math.sin(0.8 * moving_t)
                y = 0.07 * math.sin(0.8 * moving_t)
                z += 0.06 * math.sin(0.8 * moving_t)
            ee.pose_command[:, :3] = torch.tensor([[x, y, z]], device=env.device)
            link0_z = float(robot.data.body_pos_w[0, ee.link0_idx, 2])
            pitch = -math.atan2(z - link0_z, math.hypot(x, y))
            ee.pose_command[:, 3:] = quat_from_euler_xyz(
                torch.zeros(1, device=env.device), torch.tensor([pitch], device=env.device),
                torch.tensor([math.atan2(y, x)], device=env.device))
            features = [robot.data.root_ang_vel_b * 0.2, robot.data.projected_gravity_b,
                        robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids],
                        (robot.data.joint_vel[:, joint_ids] - robot.data.default_joint_vel[:, joint_ids]) * 0.05,
                        previous_action, velocity.command, ee.command]
            if step == 0:
                # Isaac Lab CircularBuffer pads a new episode with its first observation.
                histories = [f.repeat(1, 3) for f in features]
            else:
                histories = [torch.cat([h[:, f.shape[1]:], f], dim=-1) for h, f in zip(histories, features)]
            obs = torch.cat(histories, dim=-1).clamp(-100, 100)
            with torch.inference_mode():
                action = policy(obs)
                env.step(action)
            previous_action = action
            ee._update_metrics()
            q = robot.data.root_quat_w[0].detach().cpu().numpy()
            qw, qx, qy, qz = q
            roll = math.atan2(2 * (qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
            pitch_actual = math.asin(float(np.clip(2*(qw*qy-qz*qx), -1, 1)))
            row = {
                "case": case, "t": t + env.step_dt, "command": command,
                "base_pos": robot.data.root_pos_w[0].detach().cpu().tolist(),
                "base_quat_wxyz": q.tolist(),
                "base_velocity_body": robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
                "base_angular_velocity_body": robot.data.root_ang_vel_b[0].detach().cpu().tolist(),
                "roll": roll, "pitch": pitch_actual,
                "ee_command_mixed": ee.command[0].detach().cpu().tolist(),
                "ee_world_pose_xyz_wxyz": robot.data.body_state_w[0, ee.body_idx, :7].detach().cpu().tolist(),
                "link0_world_pose_xyz_wxyz": robot.data.body_state_w[0, ee.link0_idx, :7].detach().cpu().tolist(),
                "ee_position_error_m": float(ee.metrics["position_error"][0]),
                "ee_orientation_error_rad": float(ee.metrics["orientation_error"][0]),
                "joint_positions": robot.data.joint_pos[0, joint_ids].detach().cpu().tolist(),
                "action": action[0].detach().cpu().tolist(),
            }
            rows.append(row)
            if step % 100 == 0:
                print(f"CASE_STEP {case} {step}/{expected_steps}", flush=True)
            # pace=1: never deliberately run slower than physical time.
            remaining = env.step_dt - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
            row["finite"] = bool(np.isfinite(
                row["base_pos"] + row["base_quat_wxyz"] + row["base_velocity_body"] +
                row["base_angular_velocity_body"] + row["joint_positions"] + row["action"] +
                row["ee_world_pose_xyz_wxyz"] + row["link0_world_pose_xyz_wxyz"] +
                [row["roll"], row["pitch"], row["ee_position_error_m"], row["ee_orientation_error_rad"]]
            ).all())
            if not row["finite"]:
                break
        settled = [r for r in rows if r["t"] >= 3.0] or rows
        v = np.array([[*r["base_velocity_body"][:2], r["base_angular_velocity_body"][2]] for r in settled])
        target = np.array(commands[case], float)
        heights = np.array([r["base_pos"][2] for r in rows])
        tilts = np.array([[r["roll"], r["pitch"]] for r in rows])
        ee_errors = np.array([r["ee_position_error_m"] for r in settled])
        orientation = np.array([r["ee_orientation_error_rad"] for r in settled])
        joints = np.array([r["joint_positions"] for r in settled])
        complete = len(rows) == expected_steps
        finite = all(r["finite"] for r in rows)
        stable = bool(complete and finite and np.min(heights) > 0.16 and np.max(np.abs(tilts)) < 0.65)
        mean_v = v.mean(axis=0)
        tracking = bool(np.linalg.norm(mean_v[:2]-target[:2]) < 0.15 and abs(mean_v[2]-target[2]) < 0.15)
        if case == "stand":
            tracking = bool(np.linalg.norm(mean_v[:2]) <= 0.08 and abs(mean_v[2]) <= 0.12)
        direction = bool(all(abs(c) < 0.05 or m*c > 0.04*abs(c) for m, c in zip(mean_v, target)))
        coordinated = bool(np.ptp(joints[:, 12:], axis=0).max() > 0.10 and np.ptp(joints[:, :12], axis=0).max() > 0.15)
        ee_pass = bool(np.sqrt(np.mean(ee_errors**2)) < 0.15 and np.sqrt(np.mean(orientation**2)) < 0.6)
        result = {"case": case, "command": target.tolist(), "stable": stable,
                  "complete": complete, "finite": finite, "expected_control_steps": expected_steps,
                  "measured_control_steps": len(rows),
                  "tracking": tracking, "correct_direction": direction,
                  "mean_measured_vx_vy_wz": mean_v.tolist(),
                  "min_base_height_m": float(np.min(heights)), "max_abs_roll_pitch_rad": float(np.max(np.abs(tilts))),
                  "ee_position_rmse_m": float(np.sqrt(np.mean(ee_errors**2))),
                  "ee_orientation_rmse_rad": float(np.sqrt(np.mean(orientation**2))),
                  "arm_max_joint_excursion_rad": float(np.ptp(joints[:, 12:], axis=0).max()),
                  "leg_max_joint_excursion_rad": float(np.ptp(joints[:, :12], axis=0).max()),
                  "simultaneous_leg_arm_motion": coordinated,
                  "pass": bool(stable and tracking and direction and (case != "coordinated" or (coordinated and ee_pass)))}
        results.append(result)
        measurements.extend(rows)
        (output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    wall_elapsed = time.monotonic() - wall_start
    if args.snapshot:
        try:
            import omni.usd
            from isaaclab.sim.utils.stage import save_stage
            save_stage(str(output / "diagnostic.usd"), save_and_reload_in_place=False)
            p = robot.data.root_pos_w[0].detach().cpu().numpy()
            import omni.replicator.core as rep
            camera = rep.create.camera(position=tuple(p + np.array([2.0, 2.0, 1.4])),
                                       look_at=tuple(p + np.array([0, 0, 0.15])))
            render_product = rep.create.render_product(camera, (960, 720))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(render_product)
            for _ in range(12):
                env.sim.render()
            from PIL import Image
            Image.fromarray(np.asarray(annotator.get_data())).save(output / "diagnostic.png")
        except Exception as exc:
            (output / "snapshot_error.txt").write_text(repr(exc))
    summary = {"robot": "Unitree Go2 + Agilex Piper", "task": "GO2-PIPER-WBC", "policy": str(POLICY),
               "policy_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest(),
               "joint_order": joint_names, "observation_dimension": 210, "action_dimension": 18,
               "physics_dt": cfg.sim.dt, "control_dt": env.step_dt, "device": env.device,
               "pace": 1, "simulated_seconds": len(measurements)*env.step_dt,
               "wall_seconds": wall_elapsed,
               "real_time_factor": len(measurements)*env.step_dt / wall_elapsed,
               "timing_pass": wall_elapsed <= len(measurements)*env.step_dt*1.05,
               "timing_tolerance_fraction": 0.05,
               "run_complete": len(results) == len(selected) and all(r["complete"] and r["finite"] for r in results),
               "all_pass": all(r["pass"] for r in results), "cases": results,
               "validator_sha256": hashlib.sha256((output / "validator_source.py").read_bytes()).hexdigest(),
               "source_manifest": json.loads((ROOT / "SOURCES.json").read_text()),
               "packages": {name: importlib.metadata.version(name) for name in ["isaacsim", "torch", "numpy", "isaaclab", "rsl-rl-lib", "LeggedManip_Lab"]},
               "scope": "Nominal flat-ground deterministic simulation only. Mixed-frame EE command, no Agent adaptation."}
    with (output / "measurements.jsonl").open("w") as f:
        for row in measurements:
            f.write(json.dumps(row) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULT " + str(output / "summary.json"), flush=True)
    env.close()
    return 0 if summary["all_pass"] else 2


try:
    status = main()
except BaseException as exc:
    traceback.print_exc()
    (OUTPUT_DIR / "error.json").write_text(json.dumps({"run_complete": False, "exception": repr(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
    status = 1
finally:
    from isaaclab.sim import SimulationContext
    context = SimulationContext.instance()
    if context is not None:
        context._disable_app_control_on_stop_handle = True
    app.close(wait_for_replicator=False)
sys.exit(status)
