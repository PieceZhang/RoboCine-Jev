#!/usr/bin/env python3
"""Standalone measured PhysX directional and 6D whole-body acceptance run."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / time.strftime("acceptance_%Y%m%d_%H%M%S"))
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--probe", action="store_true", help="5-second physical diagnostic; not a full acceptance")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true")
    display.add_argument("--headless", action="store_true")
    parser.add_argument("--pace", type=float, choices=[1.0], default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(python=sys.executable, sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__).resolve(), ROOT / "controllers/whole_body.py", ROOT / "run.sh", ROOT / "ASSET_MANIFEST.json"]},
                     runtime={p:version(p) for p in ["isaacsim", "numpy", "scipy", "torch"]})
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=not args.gui, renderer="RaytracedLighting", anti_aliasing=0,
                             multi_gpu=False, limit_cpu_threads=8))
    try:
        run(args, app)
    except Exception:
        import traceback
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        app.close()


def run(args, app):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from controllers.whole_body import WholeBodyController
    world = World(physics_dt=0.01, rendering_dt=0.02, stage_units_in_meters=1.0)
    stage = world.stage
    robot_path = "/World/Ridgeback"
    model = ROOT / "assets/Isaac/Robots/Clearpath/RidgebackUr/ridgeback_ur5.usd"
    add_reference_to_stage(str(model), robot_path)
    # The stock asset has a -20 m scene offset. Recenter the *fixed root anchor*
    # once before starting physics; all subsequent motion uses joint drives.
    fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath(robot_path + "/world/FixedJoint"))
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0., 0., 0.025930123))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1., 0., 0., 0.))
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(robot_path)).CreateSolverPositionIterationCountAttr(64)
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(robot_path)).CreateSolverVelocityIterationCountAttr(16)
    # The native planar articulation fixes height. Its non-rolling wheel
    # cylinders must not be forced into a redundant ground contact constraint.
    # Derive a 1 mm clearance beyond the explicit pair contact offsets from
    # the exact collision shell, rather than arbitrarily lowering the floor.
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
    collision_bounds = []
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(robot_path + "/base_link/") and prim.HasAPI(UsdPhysics.CollisionAPI):
            bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            collision_bounds.append(dict(path=str(prim.GetPath()), min=list(bounds.GetMin()), max=list(bounds.GetMax())))
    # A fixed-root articulation keeps its authored initial rigid-body transform;
    # FixedJoint.localPos0 does not translate every body by that vector. Use the
    # actual authored WORLD geometry bounds directly (verified base_z=0 in PhysX).
    min_collision_z = min(record["min"][2] for record in collision_bounds)
    contact_offset = .001
    clearance = .001
    ground_height = min_collision_z - 2*contact_offset - clearance
    # The convenience add_default_ground_plane() resolves a remote asset even
    # when a local closure already exists. Reference the downloaded Grid asset.
    ground_path = "/World/Ground"
    add_reference_to_stage(str(ROOT / "assets/Isaac/Environments/Grid/default_environment.usd"), ground_path)
    ground_material = PhysicsMaterial(prim_path="/World/GroundMaterial", static_friction=.5, dynamic_friction=.5, restitution=.8)
    ground = world.scene.add(GroundPlane(prim_path=ground_path, name="ground_plane", z_position=ground_height,
                                       physics_material=ground_material))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(contact_offset)
            api.CreateRestOffsetAttr(0.)
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(600.)
    robot = world.scene.add(SingleArticulation(prim_path=robot_path, name="ridgeback_ur5"))
    ee = world.scene.add(SingleRigidPrim(prim_path=robot_path + "/ur_arm_wrist_3_link", name="ee"))
    base = world.scene.add(SingleRigidPrim(prim_path=robot_path + "/base_link", name="base"))
    world.reset()
    # Recheck after PhysX reset, before executing a command. Fabric may keep
    # stage transforms static, so explicitly compare USD and physical base Z.
    post_reset_bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
    reset_bounds = []
    for record in collision_bounds:
        bounds = post_reset_bbox.ComputeWorldBound(stage.GetPrimAtPath(record["path"])).ComputeAlignedRange()
        reset_bounds.append(dict(path=record["path"], min=list(bounds.GetMin()), max=list(bounds.GetMax())))
    stage_base_z = float(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot_path + "/base_link")).ExtractTranslation()[2])
    reset_base_z = float(base.get_world_pose()[0][2])
    if abs(reset_base_z - stage_base_z) > 1e-5:
        raise ValueError("Physical base Z differs from the USD collision geometry; floor must be audited before running")
    min_collision_z = min(record["min"][2] for record in reset_bounds)
    ground_height = min_collision_z - 2*contact_offset - clearance
    ground.set_world_pose(position=np.array([0., 0., ground_height]))
    collision_bounds = reset_bounds
    expected = ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint",
                "ur_arm_shoulder_pan_joint", "ur_arm_shoulder_lift_joint", "ur_arm_elbow_joint",
                "ur_arm_wrist_1_joint", "ur_arm_wrist_2_joint", "ur_arm_wrist_3_joint"]
    if robot.dof_names != expected:
        raise ValueError(f"Unexpected policy joint order: {robot.dof_names}")
    home = np.array([0., 0., 0., 0., -1.1, 1.6, -0.5, 1.5707963268, 0.])
    kp = np.array([20000., 20000., 10000., 6000., 6000., 6000., 2500., 2500., 2500.])
    kd = np.array([3000., 3000., 1500., 400., 400., 400., 150., 150., 150.])
    effort = np.array([4800., 4800., 4800., 150., 150., 150., 28., 28., 28.])
    robot._articulation_view.set_gains(kps=kp[None, :], kds=kd[None, :])
    robot._articulation_view.set_max_efforts(effort[None, :])
    robot.set_joint_positions(home)
    robot.set_joint_velocities(np.zeros(9))
    controller = robot.get_articulation_controller()
    controller.apply_action(ArticulationAction(joint_positions=home.copy(), joint_velocities=np.zeros(9)))
    ee_index = robot._articulation_view.get_body_index("ur_arm_wrist_3_link")
    info = dict(joint_names=robot.dof_names, body_names=robot._articulation_view.body_names,
                ee_body_index=ee_index, jacobian_shape=list(robot._articulation_view.get_jacobians().shape),
                home=home.tolist(), kp=kp.tolist(), kd=kd.tolist(), max_effort=effort.tolist(),
                physics_hz=100, ground_height_m=ground_height, contact_offset_m=contact_offset, rest_offset_m=0.,
                collision_shell_min_world_z_m=min_collision_z, ground_clearance_m=2*contact_offset+clearance,
                base_collision_world_bounds_at_initial_pose=collision_bounds,
                initial_base_world_z_m=float(base.get_world_pose()[0][2]),
                simulation="Isaac Sim 5.1.0 / PhysX", base_model="native fixed-root planar x/y/yaw articulation; wheels have non-rolling collision cylinders", pace=1.0,
                asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    (args.output / "configuration.json").write_text(json.dumps(info, indent=2))
    print("CONFIGURATION", json.dumps(info), flush=True)
    start_wall = time.monotonic()
    steps = 0
    rows = []
    def step(qtarget, qvelocity, phase, reference=None):
        nonlocal steps
        controller.apply_action(ArticulationAction(joint_positions=qtarget, joint_velocities=qvelocity))
        world.step(render=False)
        steps += 1
        q = np.array(robot.get_joint_positions(), dtype=float)
        qd = np.array(robot.get_joint_velocities(), dtype=float)
        p, orientation = ee.get_world_pose()
        bp, bq = base.get_world_pose()
        if not all(np.isfinite(v).all() for v in (q, qd, p, orientation, bp, bq)):
            raise FloatingPointError("Nonfinite measured physical state")
        record = dict(t=steps*.01, phase=phase, q=q.tolist(), qd=qd.tolist(), ee_p=np.asarray(p).tolist(),
                      ee_qwxyz=np.asarray(orientation).tolist(), base_p=np.asarray(bp).tolist(), base_qwxyz=np.asarray(bq).tolist())
        if reference is not None:
            ref_p, ref_R = reference
            error_R = (ref_R * Rotation.from_quat(np.asarray(orientation)[[1,2,3,0]]).inv()).as_rotvec()
            record.update(ee_ref_p=ref_p.tolist(), ee_ref_qwxyz=ref_R.as_quat()[[3,0,1,2]].tolist(),
                          position_error=float(np.linalg.norm(ref_p - p)), orientation_error=float(np.linalg.norm(error_R)))
        rows.append(record)
        # pace=1 only caps faster-than-real-time execution; it never changes dt.
        remaining = start_wall + steps*.01 - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return q, qd, np.asarray(p), Rotation.from_quat(np.asarray(orientation)[[1,2,3,0]])
    qtarget = home.copy()
    for i in range(300):
        q, qd, p, rot = step(qtarget, np.zeros(9), "settle")
    print("SETTLED", q, p, flush=True)
    settled_base_p, settled_base_q = base.get_world_pose()
    settled_roll_pitch = Rotation.from_quat(np.asarray(settled_base_q)[[1,2,3,0]]).as_euler("xyz")[:2]
    if abs(float(settled_base_p[2]) - reset_base_z) > 1e-5 or np.max(np.abs(settled_roll_pitch)) > 1e-5:
        raise ValueError("Native planar base height or roll/pitch changed; collision-floor audit is invalid")
    info["post_settle_geometry_audit"] = dict(measured_base_z_m=float(settled_base_p[2]),
        measured_roll_pitch_rad=settled_roll_pitch.tolist(),
        measured_collision_shell_min_z_m=min_collision_z + float(settled_base_p[2]) - stage_base_z,
        measured_collision_floor_gap_m=min_collision_z + float(settled_base_p[2]) - stage_base_z - ground_height,
        basis="USD world collision bounds checked against PhysX measured Z and locked planar roll/pitch")
    (args.output / "configuration.json").write_text(json.dumps(info, indent=2))
    if args.inspect:
        (args.output / "samples.json").write_text(json.dumps(rows, indent=2))
        return
    if args.probe:
        initial = q.copy()
        for i in range(200):
            qtarget[0] += .15*.01
            q, qd, p, rot = step(qtarget.copy(), np.r_[.15,np.zeros(8)], "forward_probe")
        probe = dict(diagnostic_only=True, physical_delta=(q-initial).tolist(), forward_motion_pass=bool(q[0]-initial[0]>.25))
        (args.output / "probe.json").write_text(json.dumps(probe, indent=2))
        with (args.output / "samples.jsonl").open("w") as f:
            for row in rows: f.write(json.dumps(row) + "\n")
        print("PROBE", json.dumps(probe), flush=True)
        return
    direction_results = []
    directions = [("forward", [.15,0,0]), ("backward",[-.15,0,0]), ("left",[0,.15,0]), ("right",[0,-.15,0]),
                  ("diagonal_ne",[.10,.10,0]), ("diagonal_sw",[-.10,-.10,0]), ("yaw_ccw",[0,0,.20]), ("yaw_cw",[0,0,-.20]), ("stop",[0,0,0])]
    for label, velocity in directions:
        initial_q = q.copy()
        command = np.r_[velocity, np.zeros(6)]
        for i in range(500):
            # A smooth 0.5 s acceleration ramp at both ends of each direction.
            scale = min(1.0, (i+1)/50, (500-i)/50)
            qtarget[:3] += command[:3] * scale * .01
            qtarget[3:] = home[3:]
            q, qd, p, rot = step(qtarget.copy(), command*scale, label)
        measured = q[:3] - initial_q[:3]
        expected_delta = command[:3] * 4.51
        translation_error = float(np.linalg.norm((measured - expected_delta)[:2]))
        yaw_error = float(abs(measured[2] - expected_delta[2]))
        tail_linear_speed = float(np.linalg.norm(qd[:2]))
        tail_yaw_speed = float(abs(qd[2]))
        ok = translation_error < .045 and yaw_error < .045 and tail_linear_speed < .06 and tail_yaw_speed < .06
        direction_results.append(dict(name=label, commanded_velocity=velocity, expected_delta=expected_delta.tolist(),
                                      measured_delta=measured.tolist(), translation_error_m=translation_error, yaw_error_rad=yaw_error,
                                      final_linear_speed_mps=tail_linear_speed, final_yaw_rate_radps=tail_yaw_speed, passed=ok))
        print("DIRECTION", json.dumps(direction_results[-1]), flush=True)
    # Explicit braking/settle before the whole-body task.
    for i in range(100):
        q, qd, p, rot = step(qtarget.copy(), np.zeros(9), "pre_wbc")
    center, center_rot = p.copy(), rot
    wbc_initial_q = q.copy()
    wbc = WholeBodyController(home)
    period = 20.
    omega = 2*np.pi/period
    for i in range(2000):
        t = (i+1)*.01
        target_p = center + np.array([.28*np.sin(omega*t), .20*(1-np.cos(omega*t)), .07*np.sin(2*omega*t)])
        target_R = Rotation.from_rotvec([.12*np.sin(omega*t), .10*np.sin(2*omega*t), .16*np.sin(omega*t)]) * center_rot
        linear_ff = np.array([.28*omega*np.cos(omega*t), .20*omega*np.sin(omega*t), .14*omega*np.cos(2*omega*t)])
        angular_ff = np.array([.12*omega*np.cos(omega*t), .20*omega*np.cos(2*omega*t), .16*omega*np.cos(omega*t)])
        error_R = (target_R * rot.inv()).as_rotvec()
        jac = np.asarray(robot._articulation_view.get_jacobians())[0, ee_index-1, :, :]
        velocity = wbc.velocity(q, jac, target_p-p, error_R, np.r_[linear_ff, angular_ff])
        qtarget = qtarget + velocity*.01
        # Bound integral tracking error after actuator saturation/disturbances.
        qtarget = np.clip(qtarget, q-.04, q+.04)
        q, qd, p, rot = step(qtarget.copy(), velocity, "whole_body", (target_p, target_R))
    for i in range(200):
        q, qd, p, rot = step(qtarget.copy(), np.zeros(9), "final_stop")
    elapsed = time.monotonic() - start_wall
    wrows = [r for r in rows if r["phase"] == "whole_body"]
    position_errors = np.array([r["position_error"] for r in wrows])
    rotation_errors = np.array([r["orientation_error"] for r in wrows])
    wq = np.array([r["q"] for r in wrows])
    base_range = float(np.linalg.norm(np.ptp(wq[:,:2], axis=0)))
    arm_range = float(np.linalg.norm(np.ptp(wq[:,3:], axis=0)))
    final_rows = [r for r in rows if r["phase"] == "final_stop"][-100:]
    final_q_range = np.ptp(np.array([r["q"] for r in final_rows]), axis=0)
    final_ee_range = np.ptp(np.array([r["ee_p"] for r in final_rows]), axis=0)
    braking = dict(base_linear_speed_mps=float(np.linalg.norm(qd[:2])), base_yaw_rate_radps=float(abs(qd[2])),
                   arm_max_reported_joint_speed_radps=float(np.max(np.abs(qd[3:]))),
                   final_second_joint_position_range=final_q_range.tolist(), final_second_ee_position_range_m=final_ee_range.tolist(),
                   passed=bool(np.linalg.norm(qd[:2])<.02 and abs(qd[2])<.02 and np.max(final_q_range)<1e-4 and np.max(final_ee_range)<1e-4))
    result = dict(direction_tests=direction_results, whole_body=dict(duration_s=20, position_rmse_m=float(np.sqrt(np.mean(position_errors**2))),
                  position_max_m=float(position_errors.max()), orientation_rmse_rad=float(np.sqrt(np.mean(rotation_errors**2))),
                  orientation_max_rad=float(rotation_errors.max()), base_xy_range_m=base_range, arm_joint_range_norm_rad=arm_range,
                  passed=bool(position_errors.max()<.035 and rotation_errors.max()<.10 and base_range>.04 and arm_range>.08)),
                  simulated_duration_s=steps*.01, measured_wall_s=elapsed, realtime_ratio=steps*.01/elapsed,
                  braking=braking, timing_pass=bool(elapsed/(steps*.01)<1.10),
                  limitations=["Native planar base drives do not validate mecanum wheel-ground friction or vertical base balance.", "This acceptance covers unobstructed motion; no avoidance Agent is connected."])
    result["passed"] = bool(all(d["passed"] for d in direction_results) and result["whole_body"]["passed"] and braking["passed"])
    (args.output / "acceptance.json").write_text(json.dumps(result, indent=2))
    with (args.output / "samples.jsonl").open("w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    with (args.output / "measured.csv").open("w") as f:
        columns = ["t", "phase"]+[f"q{i}" for i in range(9)]+[f"qd{i}" for i in range(9)]+["ee_x","ee_y","ee_z","position_error","orientation_error"]
        writer = csv.writer(f);writer.writerow(columns)
        for r in rows:
            writer.writerow([r["t"], r["phase"], *r["q"], *r["qd"], *r["ee_p"], r.get("position_error",""),r.get("orientation_error","")])
    # A still diagnostic is generated after timing/physics acceptance ends.
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    final_base = np.asarray(base.get_world_pose()[0])
    set_camera_view(final_base + [2.2, 2.2, 1.7], final_base + [0.15, 0., .5])
    for _ in range(12):
        world.render()
    viewport = get_active_viewport()
    if viewport is not None:
        viewport.resolution = (960, 720)
        capture = capture_viewport_to_file(viewport, str(args.output / "diagnostic.png"))
        for _ in range(40):
            world.render()
            if (args.output / "diagnostic.png").exists():
                break
    stage.GetRootLayer().Export(str(args.output / "tested_scene.usda"))
    print("ACCEPTANCE", json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("Measured physical acceptance did not pass; inspect acceptance.json")


if __name__ == "__main__":
    main()
