#!/usr/bin/env python3
"""Lockstep PhysX acceptance test using the same URDF, drives and WBC as ROS.

Run with jvs_isaac_bootstrap.py --runner this_file -- --xacro ... .
No kinematic substitute: measured poses/velocities come from Isaac PhysX.
"""
import json
import hashlib
from pathlib import Path
import sys
import tempfile

import jvs_isaac_sim as backend


def main():
    parser = backend.build_parser()
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--controller', choices=['stage1', 'stage2', 'no_preview', 'none', 'per_link', 'kc'], default='stage1')
    parser.add_argument('--initial-q', type=float, nargs=6, default=[0.]*6)
    args = parser.parse_args()
    backend.validate_args(args)
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    # Kit's full Python-extension teardown can segfault after a completed run.
    # Keep the backend's normal fast shutdown; the outer suite trusts the report,
    # never the Kit process's exit code alone.
    app = SimulationApp({'headless': args.headless})
    metrics = None
    audit = None
    collision_audit = None
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from jvs_mock.whole_body_qp import WholeBodyQP, whole_body_kinematics
        from jvs_mock.wbc_trajectory import CameraTrajectory
        from jvs_mock.wbc_metrics import Metrics, PhysicsAudit
        from jvs_mock.shot_predictive_qp import ShotPredictiveQP
        from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
        from jvs_mock.collision_audit import CollisionAudit
        source_root = Path(__file__).resolve().parents[1]/'jvs_mock'
        source_hashes = {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                         for name in ('whole_body_qp.py', 'shot_predictive_qp.py', 'wbc_trajectory.py',
                                      'kc_svsdf.py', 'obstacle_predictive_qp.py', 'collision_audit.py')}

        backend.enable_extensions(app)
        with tempfile.TemporaryDirectory(prefix='wbc_isaac_') as folder:
            urdf = Path(folder) / 'robot.urdf'
            backend.expand_xacro(args, urdf)
            stage = backend.prepare_stage(args, app)
            robot_path = backend.import_robot(urdf)
            app.update()
            backend.configure_articulation(stage, robot_path, args, urdf)
            backend.configure_physics_rate(stage, args.physics_hz, args.solver_type)
            sim = SimulationContext(physics_dt=1/args.physics_hz, rendering_dt=.02,
                                    stage_units_in_meters=1.)
            # SimulationContext applies its own default TGS after USD setup.
            sim.get_physics_context().set_solver_type(args.solver_type)
            sim.initialize_physics()
            sim.play()
            if sim.get_physics_context().get_solver_type() != args.solver_type:
                raise RuntimeError('Isaac did not retain the requested physics solver')
            robot = SingleArticulation(robot_path)
            robot.initialize()
            camera_path = str(Path(robot_path).parent / 'camera_link')
            camera = SingleRigidPrim(camera_path)
            camera.initialize()
            names = list(robot.dof_names)
            indices = np.array([names.index(n) for n in ['left_wheel_joint', 'right_wheel_joint'] + [f'joint{i}' for i in range(1, 7)]])
            controller = WholeBodyQP() if args.controller == 'stage1' else ShotPredictiveQP(preview=args.controller == 'stage2')
            if args.controller in ('none', 'per_link', 'kc'):
                controller = ObstaclePredictiveQP(scene=args.obstacles, collision_mode=args.controller)
            q_command = np.asarray(args.initial_q)
            robot.set_joint_positions(q_command, joint_indices=indices[2:])
            command = np.zeros(8)

            def apply():
                robot.apply_action(ArticulationAction(joint_positions=np.r_[0., 0., q_command],
                    joint_velocities=np.r_[controller.limits.wheels @ command, command[2:]],
                    joint_indices=indices))

            apply()
            # Contact settling is recorded separately from the controlled interval.
            for k in range(int(2*args.physics_hz)):
                sim.step(render=False)
                if k % (200 if args.headless else 20) == 0:
                    sim.render()
            dt = .02
            substeps = round(dt*args.physics_hz)
            if abs(substeps/args.physics_hz-dt) > 1e-9:
                raise ValueError('physics_hz must be divisible by the 50 Hz control rate')

            def observe(include_camera=True):
                bp, bq = robot.get_world_pose()
                br = Rotation.from_quat(np.roll(bq, -1)).as_matrix()
                yaw = float(np.arctan2(br[1, 0], br[0, 0]))
                q = robot.get_joint_positions(indices[2:])
                state = np.r_[bp[:2], yaw, q]
                velocities = robot.get_joint_velocities(indices)
                local_v = br.T @ robot.get_linear_velocity()
                measured = np.r_[local_v[0], robot.get_angular_velocity()[2], velocities[2:]]
                cp = cr = None
                if include_camera:
                    cp, cq = camera.get_world_pose()
                    cr = Rotation.from_quat(np.roll(cq, -1)).as_matrix()
                return state, bp, br, measured, velocities[:2], local_v[1], cp, cr

            state, bp, br, measured, wheels, lateral, cp, cr = observe()
            trajectory = CameraTrajectory(args.trajectory, cp, cr, br)
            if isinstance(controller, ShotPredictiveQP):
                controller.set_trajectory(trajectory)
            q_command = state[3:].copy()
            metrics = Metrics(args.output, controller.limits)
            audit = PhysicsAudit(args.output, controller.limits, 1/args.physics_hz)
            if args.obstacles:
                from jvs_mock.kc_svsdf import SweptCollision
                collision_audit = CollisionAudit(args.output, SweptCollision(args.obstacles))
                link_prims = [robot]
                for name in [f'link{i}' for i in range(1,7)]+['camera_link']:
                    prim = SingleRigidPrim(str(Path(robot_path).parent/name))
                    prim.initialize()
                    link_prims.append(prim)
                def audit_collision(t):
                    frames=[]
                    for prim in link_prims:
                        p,q=prim.get_world_pose()
                        frames.append((p,Rotation.from_quat(np.roll(q,-1)).as_matrix()))
                    collision_audit.record(t,frames)
                audit_collision(0.)
            audit.record(0., state[3:], measured, wheels, lateral)
            simulation_start = sim.current_time
            print('WBC_ISAAC_READY ' + str(robot_path), flush=True)
            for k in range(int((trajectory.duration+3.)/dt)+1):
                t = k*dt
                state, bp, br, measured, wheels, lateral, cp, cr = observe()
                tp, tr, twist = trajectory.sample(t)
                command, info = controller.solve(state, tp, tr, twist, dt,
                    base_z=bp[2], base_rotation=br, commanded_q=q_command,
                    brake=t > trajectory.duration+2.,
                    **({'trajectory_time': t} if isinstance(controller, ShotPredictiveQP) else {}))
                info.update(trajectory.shot_context(t))
                # Independent camera rigid-body ground truth, not controller FK.
                fp, fr, _ = whole_body_kinematics(state, bp[2], br)
                info.update(position_error=float(np.linalg.norm(tp-cp)),
                    orientation_error=float(np.linalg.norm(Rotation.from_matrix(tr @ cr.T).as_rotvec())),
                    fk_position_error=float(np.linalg.norm(fp-cp)),
                    fk_orientation_error=float(np.linalg.norm(Rotation.from_matrix(fr @ cr.T).as_rotvec())))
                for prefix, position, rotation in [('target_camera', tp, tr), ('actual_camera', cp, cr)]:
                    info.update({f'{prefix}_{axis}': float(v) for axis, v in zip('xyz', position)})
                    info.update({f'{prefix}_q{axis}': float(v) for axis, v in zip('xyzw', Rotation.from_matrix(rotation).as_quat())})
                metrics.record(t, dt, state, command, measured, wheels, lateral, info,
                    'tracking' if t <= trajectory.duration else 'stopping')
                q_command += dt*command[2:]
                apply()
                for j in range(substeps):
                    sim.step(render=False)
                    observed = observe(include_camera=False)
                    audit.record(t+(j+1)/args.physics_hz, observed[0][3:], observed[3], observed[4], observed[5])
                    if collision_audit is not None:
                        audit_collision(t+(j+1)/args.physics_hz)
                # Headless lockstep has no ROS graph or camera rendering input.
                # Kit event pumping can run at 5 Hz while physics/audit remain
                # 1 kHz and camera feedback/control remain 50 Hz.
                if not args.headless or k % 10 == 0:
                    sim.render()
                # PhysX dt is stored as float32; tolerate its accumulated rounding
                # (< 2 us over these runs), but never a missing/extra 1 ms step.
                clock_error = abs(sim.current_time-simulation_start-(k+1)*dt)
                if clock_error > 1e-4:
                    raise RuntimeError(f'Isaac physics clock differs by {clock_error:.9f}s')
                if k % 250 == 0:
                    print(f'WBC t={t:.1f} pos={info["position_error"]:.5f} rot={info["orientation_error"]:.5f}', flush=True)
            summary = metrics.finish(True)
            summary['physics_audit'] = audit.finish()
            if collision_audit is not None:
                summary['collision_audit'] = collision_audit.finish()
                collision_audit = None
            summary['verification_scope'] = 'all_physics_steps_after_settling'
            audit = None
            summary['clock_error_s'] = clock_error
            summary['stage1_pass'] = summary['sampled_checks_pass'] and summary['physics_audit']['pass_constraints']
            summary.pop('acceptance_note', None)
            summary['simulation'] = dict(physics_hz=args.physics_hz,
                source_sha256=source_hashes,
                controller=args.controller, initial_q=args.initial_q,
                obstacles=args.obstacles,
                solver_velocity_iterations=args.solver_velocity_iterations,
                arm_drive_stiffness=args.arm_drive_stiffness, arm_drive_damping=args.arm_drive_damping,
                actuator_armature=args.actuator_armature,
                solver_type=args.solver_type,
                wheel_drive_damping=args.wheel_drive_damping,
                trajectory=str(Path(args.trajectory).resolve()))
            (Path(args.output) / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print('WBC_RESULT ' + json.dumps(summary), flush=True)
            sim.stop()
            return 0 if summary['stage1_pass'] else 2
    except Exception as error:
        if metrics is not None:
            summary = metrics.finish(False, str(error))
            summary['simulation'] = dict(vars(args), source_sha256=source_hashes)
            if audit is not None:
                summary['physics_audit'] = audit.finish()
                audit = None
                summary['verification_scope'] = 'all_physics_steps_until_failure'
                (Path(args.output) / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            if collision_audit is not None:
                summary['collision_audit'] = collision_audit.finish()
                collision_audit = None
                (Path(args.output) / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(f'WBC_FAILURE {type(error).__name__}: {error}', flush=True)
        raise
    finally:
        if metrics is not None:
            metrics.close()
        if audit is not None:
            audit.finish()
        if collision_audit is not None:
            collision_audit.finish()
        app.close()


if __name__ == '__main__':
    raise SystemExit(main())
