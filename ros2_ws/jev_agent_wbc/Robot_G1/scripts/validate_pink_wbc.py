#!/usr/bin/env python3
"""Short actual PhysX validation of AGILE velocity-height plus genuine Pink.

This is a diagnostic flat-ground fixture, not an environment-constrained Agent
or continuous-camera certification. Original locomotion evidence is untouched.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(PACKAGE))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=6.)
    p.add_argument('--vx', type=float, default=0.)
    p.add_argument('--height', type=float, default=.72)
    p.add_argument('--rgb', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not (2. < a.seconds <= 30. and abs(a.vx) <= .2 and .5 <= a.height <= .72):
        raise ValueError('diagnostic requires 2<seconds<=30, |vx|<=.2, .5<=height<=.72')
    a.output = a.output.resolve()
    if not a.worker:
        import subprocess
        if a.output.exists():
            raise FileExistsError(a.output)
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'])
        result = a.output/'summary.json'
        summary = json.loads(result.read_text()) if result.is_file() else {}
        return 0 if child.returncode == 0 and summary.get('complete') and summary.get('diagnostic_pass') else 2
    a.output.mkdir(parents=True, exist_ok=False)
    frozen = a.output/'sources'; frozen.mkdir()
    sources = [Path(__file__), ROOT/'wbc_backend.py', ROOT/'config/height_policy_io.yaml',
               ROOT/'kinematics/g1_29dof.urdf', ROOT/'policies/unitree_g1_velocity_height_recurrent_student.pt',
               PACKAGE/'config/robot_sensor_mounts.json', PACKAGE/'jev_agent_wbc/robot_sensor_rig.py']
    for item in sources:
        shutil.copy2(item, frozen/item.name)
    (a.output/'provenance.json').write_text(json.dumps(dict(
        sources={str(item): hashlib.sha256((frozen/item.name).read_bytes()).hexdigest() for item in sources},
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        solver='pink.solve_ik with quadprog', base_variables_optimized=False,
        simulation_hz=200, policy_hz=50, pace_requested=1), indent=2)+'\n')
    import yaml
    rig_spec = importlib.util.spec_from_file_location('frozen_g1_sensor_rig', frozen/'robot_sensor_rig.py')
    a.sensor_module = importlib.util.module_from_spec(rig_spec); rig_spec.loader.exec_module(a.sensor_module)
    backend_spec = importlib.util.spec_from_file_location('g1_pink_process', frozen/'wbc_backend.py')
    backend = importlib.util.module_from_spec(backend_spec); backend_spec.loader.exec_module(backend)
    io = yaml.safe_load((frozen/'height_policy_io.yaml').read_text())
    sensor_spec, _ = a.sensor_module.load_config('g1', frozen/'robot_sensor_mounts.json')
    a.pink_worker = backend.PinkProcess(io['actions'][0]['joint_names'], io['articulations']['robot']['joint_names'],
        a.sensor_module.optical_transform(sensor_spec['camera']), frozen/'g1_29dof.urdf', source_path=frozen/'wbc_backend.py',
        stderr_path=a.output/'pink_worker.stderr.log')
    a.pink_worker.warmup(dict(zip(io['articulations']['robot']['joint_names'], io['articulations']['robot']['default_joint_pos'])))
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    os.environ.setdefault('ACCEPT_EULA', 'Y')
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=a.rgb, device='cpu')
    app = launcher.app
    try:
        return run(a)
    except BaseException:
        import traceback
        (a.output/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        a.pink_worker.close()
        from isaaclab.sim import SimulationContext
        if SimulationContext.instance() is not None:
            SimulationContext.instance()._disable_app_control_on_stop_handle = True
        app.close(wait_for_replicator=False)


def run(a):
    import numpy as np
    import torch
    import yaml
    from scipy.spatial.transform import Rotation
    from isaaclab.assets import Articulation
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from agile.rl_env.assets.robots.unitree_g1 import G1_29DOF
    RobotSensorRig, lab_body_poses = a.sensor_module.RobotSensorRig, a.sensor_module.lab_body_poses
    import omni.usd
    torch.set_num_threads(1); torch.manual_seed(42)
    frozen = a.output/'sources'
    backend_spec = importlib.util.spec_from_file_location('frozen_g1_pink', frozen/'wbc_backend.py')
    backend = importlib.util.module_from_spec(backend_spec); backend_spec.loader.exec_module(backend)
    io = yaml.safe_load((frozen/'height_policy_io.yaml').read_text())
    physics_dt, dt, decimation = .005, .02, 4
    sim = SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device='cpu', render_interval=decimation))
    ground = sim_utils.CuboidCfg(size=(20., 20., .1), collision_props=sim_utils.CollisionPropertiesCfg(),
                                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.))
    ground.func('/World/Ground', ground, translation=(0., 0., -.05))
    light = sim_utils.DomeLightCfg(intensity=1000.); light.func('/World/Light', light)
    cfg = G1_29DOF.copy(); cfg.prim_path = '/World/G1'
    cfg.spawn.usd_path = str(ROOT/'assets/G1/g1.usd'); cfg.init_state.pos = (0., 0., .78)
    robot = Articulation(cfg)
    stage = omni.usd.get_context().get_stage()
    rig = RobotSensorRig(stage, '/World/G1', 'g1', frozen/'robot_sensor_mounts.json')
    (a.output/'sensor_mounts.json').write_text(json.dumps(rig.manifest(), indent=2)+'\n')
    sim.reset()
    policy = torch.jit.load(str(frozen/'unitree_g1_velocity_height_recurrent_student.pt'), map_location='cpu').eval()
    policy.reset_flat()
    policy_names = io['actions'][0]['joint_names']
    leg_ids, names = robot.find_joints(policy_names, preserve_order=True)
    all_names = io['articulations']['robot']['joint_names']
    observation_ids = [robot.joint_names.index(n) for n in all_names]
    defaults = torch.tensor(io['articulations']['robot']['default_joint_pos']).reshape(1, 29)
    offsets = torch.tensor(io['actions'][0]['offset']).reshape(1, 12)
    scales = torch.tensor(io['actions'][0]['scale']).reshape(1, 12)
    upper = a.pink_worker
    target = robot.data.default_joint_pos.clone()
    action = torch.zeros(12)
    rows = []; reference = None; begin = time.monotonic(); failures = []
    maximum_fk_position = maximum_fk_angle = 0.
    fresh_commands = held_steps = 0
    original_limits = robot.data.joint_pos_limits[0]
    def measured():
        bodies = lab_body_poses(robot, rig.parent_names | {'pelvis'})
        joints = dict(zip(robot.joint_names, robot.data.joint_pos[0].numpy().tolist()))
        return bodies, joints, bodies[rig.camera_spec['parent']]@rig.camera_local
    with (a.output/'measured.jsonl').open('w', buffering=1) as log, torch.inference_mode():
        for step in range(round(a.seconds/dt)):
            stamp = step*dt
            bodies, joints, camera = measured()
            packet = upper.poll(stamp)
            if packet['ready']:
                predicted, acquisition = np.asarray(packet['predicted_camera']), np.asarray(packet['acquisition_camera'])
                fk_position = float(np.linalg.norm(predicted[:3, 3]-acquisition[:3, 3]))
                fk_angle = float(Rotation.from_matrix(predicted[:3, :3]@acquisition[:3, :3].T).magnitude())
                maximum_fk_position = max(maximum_fk_position, fk_position); maximum_fk_angle = max(maximum_fk_angle, fk_angle)
                if fk_position > .005 or fk_angle > .01:
                    raise RuntimeError(f'official URDF does not match measured USD camera FK: {fk_position}m / {fk_angle}rad')
            command = torch.tensor([[a.vx if stamp >= 2. and packet['ready'] else 0., 0., 0., a.height]])
            features = [command, robot.data.root_ang_vel_b, robot.data.projected_gravity_b,
                        robot.data.joint_pos[:, observation_ids]-defaults,
                        robot.data.joint_vel[:, observation_ids]*.1, action.reshape(1, 12)]
            observation = torch.cat(features, dim=-1).reshape(-1)
            if observation.shape != (80,):
                raise RuntimeError('incorrect height-policy observation layout')
            action = policy(observation).clamp(-6., 6.)
            target[:, leg_ids] = offsets+scales*action.reshape(1, 12)
            if stamp >= 2.:
                if reference is None:
                    reference = camera.copy()
                desired = reference.copy()
                phase = stamp-2.
                desired[:3, 3] += [.01*np.sin(phase), .01*np.sin(.7*phase), .01*np.sin(.5*phase)]
                desired[:3, :3] = Rotation.from_rotvec([0., .02*np.sin(phase), .02*np.sin(.7*phase)]).as_matrix()@reference[:3, :3]
                velocities = packet.get('velocity', {}) if packet['ready'] else {}
                fresh_commands += bool(velocities)
                held_steps += not bool(velocities)
                for name, speed in velocities.items():
                    index = robot.joint_names.index(name)
                    target[0, index] = torch.clamp(target[0, index]+speed*dt,
                                                  original_limits[index, 0], original_limits[index, 1])
            else:
                desired = camera.copy()
            upper.submit(joints, bodies['pelvis'], desired if stamp >= 2. else None, stamp, camera, dt)
            if upper.failure_count >= 3:
                failures.append('three consecutive stale/failed Pink results; upper targets held'); break
            for _ in range(decimation):
                robot.set_joint_position_target(target); robot.write_data_to_sim()
                sim.step(render=False); robot.update(physics_dt)
            bodies, joints, camera = measured()
            pelvis = bodies['pelvis']
            tilt = float(np.arccos(np.clip(pelvis[2, 2], -1., 1.)))
            row = dict(stamp_s=(step+1)*dt, phase='pink_camera' if stamp >= 2. else 'settle',
                       T_world_camera_optical=camera.tolist(), T_world_pelvis=pelvis.tolist(),
                       T_world_reference_optical=desired.tolist(), joint_positions=joints,
                       policy_command_vx_vy_wz_height=command[0].tolist(), policy_action=action.tolist(),
                       controller_runtime_status=packet['status'], controller_state_stamp_s=packet.get('stamp_s'),
                       position_error_m=float(np.linalg.norm(camera[:3, 3]-desired[:3, 3])),
                       orientation_error_rad=float(Rotation.from_matrix(desired[:3, :3]@camera[:3, :3].T).magnitude()),
                       base_tilt_rad=tilt, measured_base_height_m=float(pelvis[2, 3]))
            rows.append(row); log.write(json.dumps(row)+'\n')
            if not np.isfinite(camera).all() or pelvis[2, 3] < .5 or tilt > .6:
                failures.append('physical stability/finite-state gate'); break
            remaining = begin+(step+1)*dt-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    elapsed = time.monotonic()-begin
    active = [row for row in rows if row['phase'] == 'pink_camera']
    position_rmse = float(np.sqrt(np.mean([row['position_error_m']**2 for row in active]))) if active else None
    angle_rmse = float(np.sqrt(np.mean([row['orientation_error_rad']**2 for row in active]))) if active else None
    fresh_fraction = fresh_commands/len(active) if active else 0.
    if active and fresh_fraction < .9:
        failures.append('fewer than 90% fresh upper command steps')
    result = dict(schema='g1_agile_height_pink_diagnostic/v1', complete=len(rows) == round(a.seconds/dt),
                  diagnostic_pass=bool(active and not failures and len(rows) == round(a.seconds/dt)
                                       and position_rmse < .05 and angle_rmse < .15), failures=failures,
                  position_rmse_m=position_rmse, orientation_rmse_rad=angle_rmse,
                  maximum_measured_fk_position_error_m=maximum_fk_position, maximum_measured_fk_angle_error_rad=maximum_fk_angle,
                  joint_ownership=upper.owners, policy_observation_dimension=80, policy_action_dimension=12,
                  actual_pink=True, free_base_optimized=False, camera_feedback='measured_raw_optical_rigid_body_pose',
                  pink_runtime=upper.runtime, fresh_upper_commands=fresh_commands, held_upper_steps=held_steps,
                  fresh_upper_fraction=fresh_fraction, sensor_rig_import=str(frozen/'robot_sensor_rig.py'),
                  control_transport='one bounded asynchronous isolated Pink process; no physics wait or stale velocity reuse',
                  sim_seconds=len(rows)*dt, wall_seconds=elapsed, pace_requested=1,
                  realtime_factor=len(rows)*dt/elapsed, timing_pass=elapsed <= len(rows)*dt*1.05,
                  continuous_camera_tracking_verified=False, environment_constraints_verified=False,
                  representation_ingress_verified=False, sensors_avoidance_verified=False)
    if a.rgb:
        result['camera_render'] = rig.capture_camera(a.output, sim.render, len(rows)*dt)
    stage.Flatten().Export(str(a.output/'tested_scene.usda'))
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return 0 if result['diagnostic_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
