#!/usr/bin/env python3
"""Headless, lockstep PhysX suite; one Kit startup, short independent shots.

Sparse dual LiDAR uses real PhysX closest-hit queries (not obstacle truth).
Robot/camera poses and contact audits come from PhysX. Reference RGB renders the
original static scene without movers/robot; execution RGB includes both. SAM is
run separately after Kit exits so the two large GPU consumers never overlap.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jvs_isaac_sim as backend


def arguments():
    parser = backend.build_parser()
    parser.set_defaults(headless=True, physics_hz=1000., enable_lidar=False,
                        actuator_armature=.05, wheel_drive_damping=20.,
                        xacro=str(ROOT/'urdf/jev_agent_wbc.urdf.xacro'))
    # The inherited xacro option was originally required; this package supplies its own.
    for action in parser._actions:
        if action.dest == 'xacro':
            action.required = False
    parser.add_argument('--case', default='all')
    parser.add_argument('--scenarios', type=Path, default=ROOT/'scenarios')
    parser.add_argument('--mode', choices=['jev', 'strict', 'rule', 'both'], default='jev')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='perplexity', help='explicit backend model ID')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000/v1/systemone')
    from jev_agent_wbc.camera_output import add_camera_arguments
    add_camera_arguments(parser)
    parser.add_argument('--virtual-stabilizer-ros', action='store_true', help='independent reference-aware virtual camera via timestamped ROS batches')
    parser.add_argument('--skip-reference-video', action='store_true', help='record raw/stabilized/overview only')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record-fps', type=float, default=8.)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--pace', type=float, default=1., help='maximum sim/wall rate; 0 means unpaced')
    parser.add_argument('--agent-period', type=float, default=.5)
    parser.add_argument('--activation-lead', type=float, default=4.)
    parser.add_argument('--candidate-budget', type=int, default=3)
    parser.add_argument('--prediction-horizon', type=float, default=8.)
    parser.add_argument('--prediction-step', type=float, default=.1)
    parser.add_argument('--audit-prediction', action='store_true', help='save bounded observation snapshots for offline predictor diagnostics')
    parser.add_argument('--no-preview', action='store_true', help='ablate inherited allocation MPC, retaining safety and bounded task planner')
    parser.add_argument('--allocation-preview', action='store_true', help='enable inherited shot-specific base-allocation MPC for ablation')
    parser.add_argument('--async-coordination', action='store_true',
                        help='run the cross-phase allocation in its live nonblocking worker instead of lockstep')
    parser.add_argument('--acceleration-reserve', type=float, default=.4,
                        help='operating acceleration / physical acceleration; evaluation limits stay fixed')
    parser.add_argument('--yaw-acceleration-reserve', type=float, default=.1,
                        help='optional separate operating fraction for chassis yaw acceleration')
    args = parser.parse_args()
    backend.validate_args(args)
    if args.physics_hz < 200 or abs(args.physics_hz/50-round(args.physics_hz/50)) > 1e-8:
        parser.error('physics-hz must be a multiple of 50 and at least 200')
    if not 0 < args.acceleration_reserve <= 1:
        parser.error('acceleration-reserve must be in (0,1]')
    if args.yaw_acceleration_reserve is not None and not 0 < args.yaw_acceleration_reserve <= 1:
        parser.error('yaw-acceleration-reserve must be in (0,1]')
    return args


def run_case(args, app, name, mode, output):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from omni.physx import get_physx_scene_query_interface, get_physx_simulation_interface
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    import omni.kit.app
    from jev_agent_wbc.scenarios import RuntimeScene
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.sensors import RuntimeLidar, RAYS, estimate_subject_from_points
    from jev_agent_wbc.engine import CinematicEngine
    from jev_agent_wbc.candidates import SearchBudget
    from jev_agent_wbc.evaluation import EVALUATION_PROTOCOL, save_json, trajectory_summary
    from jev_agent_wbc.dynamic_collision import DynamicCollision
    from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
    from jev_agent_wbc.wbc_metrics import PhysicsAudit
    from jev_agent_wbc.actuation import certify_control_ramp

    output.mkdir(parents=True, exist_ok=False)
    runtime_scene = RuntimeScene(args.scenarios/name/'runtime_scene.json')
    fixture = runtime_scene.document
    from copy import copy
    args = copy(args)
    args.spawn_x, args.spawn_y, args.spawn_yaw = fixture['initial_state'][:3]
    args.spawn_z = fixture['base_z']
    reference = OriginalReference(args.scenarios/name/'task.json', fixture['subject']['position'])
    save_json(output/'original_task.json', reference.task)
    save_json(output/'runtime_scene.json', fixture)
    save_json(output/'evaluation_protocol.json', EVALUATION_PROTOCOL)
    engine = audit = recorder = stabilizer_client = None
    from jev_agent_wbc.camera_output import CameraOutput
    camera_output = CameraOutput.from_args(args)
    camera_samples = []
    stabilized_pose = None
    rows, contacts, failure = [], [], ''
    prediction_snapshots = []
    recording_error = ''
    physical_summary = None
    begin = time.monotonic()
    dt = .02
    sim = None
    try:
        SimulationContext.clear_instance()
        stage = backend.prepare_stage(args, app)
        with tempfile.TemporaryDirectory(prefix='jev_robot_') as folder:
            urdf = Path(folder)/'robot.urdf'
            backend.expand_xacro(args, urdf)
            robot_path = backend.import_robot(urdf)
            save_json(output/'import_collision_audit.json',
                      backend.isolate_urdf_templates(stage, robot_path))
            app.update()
            backend.configure_articulation(stage, robot_path, args, urdf)
        backend.configure_physics_rate(stage, args.physics_hz, args.solver_type)
        robot_scope = str(Path(robot_path).parent)
        dome = UsdLux.DomeLight.Define(stage, '/World/SoftFill')
        dome.CreateIntensityAttr(700.)
        wall = UsdGeom.Cube.Define(stage, '/World/BackWall')
        wall.CreateSizeAttr(1.)
        wall.AddTranslateOp().Set(Gf.Vec3d(3.5, 0., .9))
        wall.AddScaleOp().Set(Gf.Vec3f(.08, 5., 1.8))
        wall.CreateDisplayColorAttr([Gf.Vec3f(.40, .45, .48)])
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
        subject_data = fixture['subject']
        subject = UsdGeom.Cylinder.Define(stage, '/World/BlueSubject')
        subject.CreateAxisAttr('Z')
        subject.CreateRadiusAttr(subject_data['radius'])
        subject.CreateHeightAttr(subject_data['height'])
        subject.AddTranslateOp().Set(Gf.Vec3d(*subject_data['position']))
        subject.CreateDisplayColorAttr([Gf.Vec3f(.025, .24, .85)])
        UsdPhysics.CollisionAPI.Apply(subject.GetPrim())
        top = UsdGeom.Sphere.Define(stage, '/World/SubjectTop')
        top.CreateRadiusAttr(.07)
        top_position = np.asarray(subject_data['position'])+[0., 0., subject_data['height']/2]
        top.AddTranslateOp().Set(Gf.Vec3d(*top_position))
        top.CreateDisplayColorAttr([Gf.Vec3f(.95, .68, .03)])
        # Visual subject identity: cylinder+cap. The fixed runtime association is not a SAM decision input.
        obstacle_ops = {}
        for obj in runtime_scene.obstacles(0.):
            root = UsdGeom.Xform.Define(stage, '/World/Moving/'+obj['id'])
            obstacle_ops[obj['id']] = root.AddTranslateOp()
            obstacle_ops[obj['id']].Set(Gf.Vec3d(*obj['center']))
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
            shape = UsdGeom.Sphere.Define(stage, str(root.GetPath())+'/Shape')
            shape.CreateRadiusAttr(obj['radius'])
            shape.CreateDisplayColorAttr([Gf.Vec3f(.9, .14, .035)])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)

        sim = SimulationContext(physics_dt=1/args.physics_hz, rendering_dt=.02, stage_units_in_meters=1.)
        sim.get_physics_context().set_solver_type(args.solver_type)
        sim.initialize_physics()
        sim.play()
        robot = SingleArticulation(robot_path)
        robot.initialize()
        camera_body = SingleRigidPrim(robot_scope+'/camera_link')
        camera_body.initialize()
        names = list(robot.dof_names)
        indices = np.array([names.index(n) for n in ['left_wheel_joint', 'right_wheel_joint']+[f'joint{i}' for i in range(1, 7)]])
        movers = RigidPrim('/World/Moving/*', name='jev_movers', reset_xform_properties=False)
        movers.initialize()
        targets = movers._physics_view.get_transforms().copy()
        mover_indices = np.arange(len(movers.prim_paths), dtype=np.uint32)
        mover_names = [str(p).split('/')[-1] for p in movers.prim_paths]

        def move_obstacles(t):
            objects = {o['id']: o for o in runtime_scene.obstacles(t)}
            targets[:, :3] = np.array([objects[n]['center'] for n in mover_names])
            movers._physics_view.set_kinematic_targets(targets, mover_indices)
            for key, op in obstacle_ops.items():
                op.Set(Gf.Vec3d(*objects[key]['center']))

        q_command = np.asarray(fixture['initial_state'][3:]).copy()
        robot.set_joint_positions(q_command, joint_indices=indices[2:])
        command = np.zeros(8)
        physical_limits = Limits()
        limits = Limits(acceleration=physical_limits.acceleration*args.acceleration_reserve,
                        wheel_acceleration=physical_limits.wheel_acceleration*args.acceleration_reserve,
                        velocity=physical_limits.velocity*.95, wheel_speed=physical_limits.wheel_speed*.95,
                        position_margin=.03, command_ramp=True)
        if args.yaw_acceleration_reserve is not None:
            limits.acceleration[1] = physical_limits.acceleration[1]*args.yaw_acceleration_reserve

        def apply():
            robot.apply_action(ArticulationAction(joint_positions=np.r_[0., 0., q_command],
                joint_velocities=np.r_[limits.wheels@command, command[2:]], joint_indices=indices))

        apply()
        for k in range(int(args.physics_hz*1.5)):
            sim.step(render=False)
            if k % 100 == 0:
                sim.render()

        def observe():
            bp, bq = robot.get_world_pose()
            br = Rotation.from_quat(np.roll(bq, -1)).as_matrix()
            yaw = float(np.arctan2(br[1, 0], br[0, 0]))
            state = np.r_[bp[:2], yaw, robot.get_joint_positions(indices[2:])]
            velocities = robot.get_joint_velocities(indices)
            local_v = br.T@robot.get_linear_velocity()
            measured = np.r_[local_v[0], robot.get_angular_velocity()[2], velocities[2:]]
            cp, cq = camera_body.get_world_pose()
            cr = Rotation.from_quat(np.roll(cq, -1)).as_matrix()
            return state, bp, br, measured, velocities[:2], local_v[1], cp, cr

        query = get_physx_scene_query_interface()
        lidar = RuntimeLidar()
        subject_track = None

        def scan(state, bp, stamp):
            nonlocal subject_track
            origins = lidar.origins(state, bp[2])
            returns, subject_returns = [], []
            self_returns = total_returns = 0
            audit_hits = {} if abs(stamp) < 1e-8 else None
            for origin in origins:
                points = []
                for direction in RAYS:
                    hit = query.raycast_closest(tuple(origin), tuple(direction), 4.5)
                    if hit.get('hit'):
                        total_returns += 1
                        if audit_hits is not None:
                            key = (str(hit.get('collision', '')), str(hit.get('rigidBody', '')))
                            record = audit_hits.setdefault(key, dict(collision=key[0], rigid_body=key[1], count=0, sample_points=[]))
                            record['count'] += 1
                            if len(record['sample_points']) < 3:
                                record['sample_points'].append(list(hit['position']))
                        # Known robot body identity is available to this
                        # simulator sensor adapter. Discard its first hit but
                        # do not raycast through the robot: occlusion remains.
                        # No external object identity, pose or velocity is used.
                        if any(str(hit.get(key, '')).startswith(robot_scope+'/') for key in ('collision', 'rigidBody')):
                            self_returns += 1
                            continue
                        point = hit['position']
                        points.append(point)
                        if 'BlueSubject' in str(hit.get('collision', '')):
                            subject_returns.append(point)
                returns.append(np.asarray(points).reshape(-1, 3))
            previous_subject = subject_track
            subject_track = dict(valid=False, orientation_valid=False, stamp_s=stamp,
                                 reason='no_current_subject_lidar_returns')
            if subject_returns:
                estimate = estimate_subject_from_points(np.asarray(subject_returns), subject_data['position'],
                    previous=previous_subject, stamp=stamp, symmetric=True)
                if estimate['valid']:
                    # This catalog's subject is stationary and rotationally symmetric.
                    # Its fixed frame is calibrated at startup; no rotating-subject claim is made.
                    estimate.update(rotation=np.eye(3).tolist(), orientation_valid=True,
                                    heading_source='stationary_subject_calibration', stationary=True)
                    subject_track = estimate
            snapshot = lidar.integrate(state, bp[2], stamp, returns, origins)
            snapshot['sensor_diagnostics'] = dict(self_returns_discarded=self_returns,
                total_returns=total_returns, self_filter='known_robot_collision_prim_prefix; first-hit occlusion retained')
            if audit_hits is not None:
                save_json(output/'initial_raycast_audit.json', dict(robot_scope=robot_scope,
                    returns=list(audit_hits.values()), role='offline sensor identity audit; not supplied to planner'))
            return snapshot

        # Acquire actual timestamped pre-roll observations. Giving a frozen
        # obstacle nine different timestamps would erase its initial velocity.
        for k in range(round(.8*args.physics_hz)+1):
            stamp = -.8+k/args.physics_hz
            move_obstacles(stamp)
            apply()
            sim.step(render=False)
            if k % round(args.physics_hz/10) == 0:
                state, bp, br, measured, wheels, lateral, cp, cr = observe()
                observed = scan(state, bp, float(stamp))
        state, bp, br, measured, wheels, lateral, cp, cr = observe()
        if not len(observed['surfel_centers']):
            raise RuntimeError('PhysX LiDAR startup produced no observed surface map')
        save_json(output/'initial_observed_scene.json', observed)
        engine = CinematicEngine(reference, observed, output, mode=mode, api_url=args.api_url, model=args.model, limits=limits,
                                 preview=args.allocation_preview and not args.no_preview,
                                 coordination_async=args.async_coordination,
                                 budget=SearchBudget(decision_period_s=args.agent_period,
                                     activation_lead_s=args.activation_lead, max_rollouts=args.candidate_budget,
                                     max_horizon_s=args.prediction_horizon, prediction_step_s=args.prediction_step),
                                 intrinsics=[.5/np.tan(np.deg2rad(27.5))*args.height/args.width,
                                             .5/np.tan(np.deg2rad(27.5)), .5, .5])
        audit = PhysicsAudit(str(output), physical_limits, 1/args.physics_hz)
        engine.wbc.measured_limits=physical_limits
        audit.record(0., state[3:], measured, wheels, lateral)
        physical_t = 0.
        if args.virtual_stabilizer_ros:
            from jev_agent_wbc.virtual_stabilizer_ros import StabilizerClient
            stabilizer_client = StabilizerClient(str(output.resolve()))

        def stabilize_batch(samples):
            nonlocal stabilized_pose
            if stabilizer_client:
                values = stabilizer_client.update(samples)
            else:
                values = []
                for sample in samples:
                    sp, sr, info = camera_output.filter.update(sample['t'], sample['rp'], sample['rr'], sample['ap'], sample['ar'])
                    values.append(dict(t=sample['t'], p=sp.tolist(), r=sr.tolist(), info=info))
            if len(values) != len(samples):
                raise RuntimeError('stabilizer dropped physics samples')
            for source, result in zip(samples, values):
                if abs(source['t']-result['t']) > 1e-10:
                    raise RuntimeError('stabilizer timestamp mismatch')
                camera_samples.append(dict(**source, stabilized=result))
                camera_output.accept(source['t'], source['ap'], source['ar'], result['p'], result['r'], result['info'])
            stabilized_pose = (np.array(values[-1]['p']), np.array(values[-1]['r']))

        def camera_sample(stamp, position, rotation):
            rp, rr, _ = engine.active.sample(stamp)
            return dict(t=float(stamp), rp=rp.tolist(), rr=rr.tolist(),
                        ap=position.tolist(), ar=rotation.tolist())

        stabilize_batch([camera_sample(0., cp, cr)])

        def on_contacts(headers, data):
            for h in headers:
                paths = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.actor0, h.actor1)]
                is_robot = any(p.startswith(robot_scope+'/') for p in paths)
                other = any(not p.startswith(robot_scope+'/') and
                            not p.startswith('/World/groundPlane/') for p in paths)
                if is_robot and other:
                    contacts.append(dict(t=physical_t, actors=paths, points=int(h.num_contact_data), kind=str(h.type)))
                    if len(contacts) <= 3:
                        print('CONTACT_DEBUG '+json.dumps(contacts[-1]), flush=True)

        subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
        if args.record or args.publish_camera_ros:
            recorder = RGBPair(stage, sim, args, output, robot_scope, subject_data)
        print(f'JEV_ISAAC_READY case={name} mode={mode} surfaces={len(observed["surfel_centers"])}', flush=True)
        run_start = time.monotonic()
        next_frame = 0.
        substeps = round(args.physics_hz*dt)
        previous_command = np.zeros(8)
        for k in range(round(reference.duration/dt)+1):
            t = k*dt
            state, bp, br, measured, wheels, lateral, cp, cr = observe()
            if k % 5 == 0:
                observed = scan(state, bp, t)
                if args.record:
                    sensor_folder = output/'observations'
                    sensor_folder.mkdir(exist_ok=True)
                    np.savez_compressed(sensor_folder/f'{k:05d}.npz',
                        **{key: np.asarray(value) for key, value in observed.items() if key.startswith('surfel_')},
                        observation_stamp=observed['observation_stamp'], t=t)
            if subject_track is not None:
                engine.perception.update_geometry(subject_track)
            if args.audit_prediction and k % 50 == 0 and len(prediction_snapshots) < 16:
                prediction_snapshots.append(dict(t=t, state=state.copy(), scene=observed,
                    previous=engine.wbc.previous.copy(), commanded_q=q_command.copy(),
                    base_z=float(bp[2]), base_rotation=br.copy(),
                    active_is_original=engine.active is reference))
            output_p, output_r = camera_output.pose(t)
            engine.set_camera_output(t, output_p, output_r, source=camera_output.source)
            try:
                command, info = engine.step(state, observed, t, dt, base_z=bp[2], base_rotation=br, commanded_q=q_command, measured_control=measured,
                                           measured_base_twist=np.r_[robot.get_linear_velocity(),robot.get_angular_velocity()])
            except Exception:
                from dataclasses import asdict
                save_json(output/'failure_snapshot.json', dict(t=t, dt=dt, state=state,
                    base_z=bp[2], base_rotation=br, commanded_q=q_command,
                    previous_command=previous_command, limits=asdict(limits), scene=observed))
                raise
            info['motor_ramp_sweep_lower_m'] = certify_control_ramp(observed, state, previous_command, command,
                t, dt, bp[2], steps=substeps, constant_certificate=info['command_sweep_lower_m'], base_rotation=br)
            truth = DynamicCollision(runtime_scene.truth_collision(t), t, uncertainty_rate=0.)
            clearance = float(truth.clearance(state, bp[2], br).min())
            fk, fr, _ = whole_body_kinematics(state, bp[2], br)
            info['fk_position_error_m'] = float(np.linalg.norm(fk-cp))
            rows.append(dict(t=t, state=state.tolist(), command=command.tolist(), measured=measured.tolist(),
                base_z=float(bp[2]), base_rotation=br.tolist(),
                commanded_q=q_command.tolist(),
                measured_wheels=wheels.tolist(), measured_lateral_velocity=float(lateral),
                **{key:value for key,value in camera_output.latest.items() if key != 't'},
                subject_track=subject_track, truth_clearance_m=clearance, info=info))
            rows[-1]['runtime_tracks'] = observed.get('tracks', [])
            rows[-1]['sensor_diagnostics'] = observed.get('sensor_diagnostics', {})
            if recorder and t+1e-8 >= next_frame:
                rp, rr, _ = reference.sample(t)
                recorder.capture(t, rp, rr, cp, cr, subject_track, stabilized_pose=stabilized_pose)
                next_frame += 1/args.record_fps
            if k % 50 == 0:
                print(json.dumps(dict(case=name, mode=mode, t=t, clearance=clearance, contacts=len(contacts),
                    jev_calls=engine.jev.decider.client.calls, installed=engine.installed_plans)), flush=True)
            if t >= reference.duration:
                break
            requested_command = command.copy()
            physics_camera_batch = []
            for j in range(substeps):
                physical_t = t+(j+1)/args.physics_hz
                # Interpolate the acceleration-bounded WBC command at physics rate.
                # Position targets integrate that SAME ramp, avoiding 50 Hz servo steps.
                command = previous_command+(j+.5)/substeps*(requested_command-previous_command)
                q_command += command[2:]/args.physics_hz
                apply()
                move_obstacles(physical_t)
                sim.step(render=False)
                measured_state = observe()
                physics_camera_batch.append(camera_sample(physical_t, measured_state[6], measured_state[7]))
                audit.record(physical_t, measured_state[0][3:], measured_state[3], measured_state[4], measured_state[5])
            stabilize_batch(physics_camera_batch)
            previous_command = requested_command
            if k % 10 == 0:
                sim.render()
            if args.pace > 0:
                slack = (k+1)*dt/args.pace-(time.monotonic()-run_start)
                if slack > 0:
                    time.sleep(slack)
        del subscription
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        import traceback
        traceback.print_exc()
    finally:
        if audit:
            physical_summary = audit.finish()
        if recorder:
            try:
                recorder.close()
            except Exception as error:
                recording_error = f'{type(error).__name__}: {error}'
                print(f'RECORDER_CLOSE_ERROR {error}', flush=True)
        summary = trajectory_summary(rows, reference, {} if engine is None else engine.summary(), failure, physical_summary)
        contact_count = sum(c['points'] > 0 for c in contacts)
        summary.update(backend='isaac_physx', physics_hz=args.physics_hz, wall_s=time.monotonic()-begin,
            mode=mode, case=name, observation_source='dual_physx_sparse_lidar_raycast', contacts=contact_count,
            contact_notifications=len(contacts),
            coordination_async=args.async_coordination,
            operating_acceleration_fraction=args.acceleration_reserve,
            operating_yaw_acceleration_fraction=(args.yaw_acceleration_reserve
                if args.yaw_acceleration_reserve is not None else args.acceleration_reserve),
            camera_output_source=camera_output.source,
            camera_output_role='primary Agent and visual evaluation input',
            raw_camera_role='intermediate physical diagnostic',
            sensor_limitation='PhysX first-hit raycasting; no RTX material/noise/motion-distortion simulation',
            measured_physical_constraints=physical_summary is not None, recording_error=recording_error)
        if contact_count:
            summary['safety_pass'] = summary['within_budget_success'] = False
        save_json(output/'summary.json', summary)
        save_json(output/'telemetry.json', rows)
        if prediction_snapshots:
            save_json(output/'prediction_snapshots.json', prediction_snapshots)
        save_json(output/'contacts.json', contacts)
        if camera_samples:
            save_json(output/'virtual_camera_physics.json', camera_samples)
        if stabilizer_client:
            stabilizer_client.close()
        if engine:
            engine.close()
        if sim:
            sim.stop()
        print('JEV_ISAAC_RESULT '+json.dumps(summary), flush=True)
    return summary


class RGBPair:
    def __init__(self, stage, sim, args, output, robot_scope, subject):
        import cv2
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        self.cv2, self.rep, self.stage, self.sim, self.args = cv2, rep, stage, sim, args
        rep.orchestrator.set_capture_on_play(False)
        self.recording = bool(getattr(args, 'record', True))
        self.root = output/'video'
        self.root.mkdir()
        self.frame_metadata = []
        from jev_agent_wbc.camera_output import CameraOutput
        self.camera_output = CameraOutput.from_args(args)
        self.output_rgb = None
        self.camera_publisher = None
        if getattr(args, 'publish_camera_ros', False):
            from jev_agent_wbc.camera_output_ros import CameraPublisher
            self.camera_publisher = CameraPublisher(self.camera_output.source)
        self.subject = subject
        self.hide_for_reference = [UsdGeom.Imageable(stage.GetPrimAtPath(path))
            for path in (robot_scope, '/World/Moving', '/World/MovingDisplay')
            if stage.GetPrimAtPath(path).IsValid()]
        self.cameras, self.products, self.annotators, self.writers = {}, {}, {}, {}
        self.camera_prims = {}
        keys = ['execution', 'overview']
        if self.recording and not getattr(args, 'skip_reference_video', False):
            keys.append('reference')
        keys.append('stabilized')
        for key in keys:
            folder = self.root/key
            folder.mkdir()
            cam = UsdGeom.Camera.Define(stage, '/World/Eval_'+key)
            cam.CreateHorizontalApertureAttr(24.*args.width/args.height)
            cam.CreateVerticalApertureAttr(24.)
            import math
            cam.CreateFocalLengthAttr(12./math.tan(math.radians(55.)/2))
            cam.CreateClippingRangeAttr(Gf.Vec2f(.05, 50.))
            translate, orient = cam.AddTranslateOp(), cam.AddOrientOp()
            self.cameras[key] = (translate, orient)
            self.camera_prims[key] = cam
            product = rep.create.render_product(str(cam.GetPath()), (args.width, args.height))
            self.products[key] = product
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach(product)
            self.annotators[key] = annotator
            if self.recording:
                writer = cv2.VideoWriter(str(self.root/f'{key}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.record_fps, (args.width, args.height))
                if not writer.isOpened():
                    raise RuntimeError('cannot open RGB video writer')
                self.writers[key] = writer
        source = self.camera_output.source
        (self.root/'camera').symlink_to(source, target_is_directory=True)
        if self.recording:
            (self.root/'camera.mp4').symlink_to(source+'.mp4')
        (self.root/'camera_output.json').write_text(json.dumps(dict(schema='camera_output/v1',
            source=source, image_directory=source, video=source+'.mp4' if self.recording else None,
            recorded=self.recording,
            comparison=None if not self.recording or getattr(args, 'skip_reference_video', False) else 'comparison.mp4',
            views=[view for view in ('reference','execution','stabilized') if view in self.cameras],
            raw_role='intermediate physical diagnostic', fps=args.record_fps))+'\n')
        from jev_agent_wbc.reference import look_at
        import numpy as np
        overview_position = np.array([2.6, -3.2, 2.1])
        self.pose('overview', overview_position, look_at(overview_position, np.array([.6, 0., .65])))
        for _ in range(3):
            sim.render()

    def pose(self, key, position, rotation):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from pxr import Gf
        # camera_link +X forward/+Y left/+Z up -> USD camera +X right/+Y up/-Z forward.
        quat = Rotation.from_matrix(rotation@np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])).as_quat()
        self.cameras[key][0].Set(Gf.Vec3d(*map(float, position)))
        self.cameras[key][1].Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))

    def capture(self, t, rp, rr, ap, ar, subject_track, stabilized_pose=None):
        import numpy as np
        from jev_agent_wbc.contracts import native
        from scipy.spatial.transform import Rotation
        if stabilized_pose is None:
            self.camera_output.update(t, rp, rr, ap, ar)
            value = self.camera_output.latest
            stabilized_pose = (np.array(value['stabilized_camera_position']), Rotation.from_quat(value['stabilized_camera_quaternion_xyzw']).as_matrix())
        capture_clock_before = float(self.sim.current_time)
        self.pose('execution', ap, ar)
        if stabilized_pose is not None:
            self.pose('stabilized', *stabilized_pose)
            from pxr import Gf
            # Preserve the physical camera's near-plane clearance when the
            # virtual viewpoint retreats into its visual housing. Intrinsics
            # and far clip remain identical; foreground closer than this
            # explicitly logged plane is excluded from the virtual image.
            virtual_near_m = .05+float(np.linalg.norm(stabilized_pose[0]-ap))+.01
            self.camera_prims['stabilized'].GetClippingRangeAttr().Set(Gf.Vec2f(virtual_near_m, 50.))
        # Keep all actors visible throughout execution. Repeatedly toggling USD
        # visibility between render products can leave a delayed reference image
        # containing the robot. Reference RGB is rendered in one isolated pass.
        raw_rgb = self.capture_rgb('execution', len(self.frame_metadata))
        self.capture_rgb('overview', len(self.frame_metadata), advance_render=False)
        if stabilized_pose is not None:
            stabilized_rgb = self.capture_rgb('stabilized', len(self.frame_metadata), advance_render=False)
        output_p, output_r = stabilized_pose if self.camera_output.source == 'stabilized' else (ap, ar)
        self.output_rgb = (stabilized_rgb if self.camera_output.source == 'stabilized' else raw_rgb)[:, :, ::-1].copy()
        if abs(float(self.sim.current_time)-capture_clock_before) > 1e-8:
            raise RuntimeError('RGB capture advanced the physical simulation')
        record = dict(frame=len(self.frame_metadata), t=t, physical_time_before_capture=capture_clock_before, physical_time_after_capture=float(self.sim.current_time),
            reference_camera_position=rp, actual_camera_position=ap,
            reference_camera_quaternion_xyzw=Rotation.from_matrix(rr).as_quat(),
            actual_camera_quaternion_xyzw=Rotation.from_matrix(ar).as_quat(), subject_track=subject_track,
            camera_output_source=self.camera_output.source,
            camera_position=output_p, camera_quaternion_xyzw=Rotation.from_matrix(output_r).as_quat(),
            visibility=self.visibility(output_p, output_r), actual_visibility=self.visibility(ap, ar))
        if stabilized_pose is not None:
            record['stabilized_camera_position'] = stabilized_pose[0]
            record['stabilized_camera_quaternion_xyzw'] = Rotation.from_matrix(stabilized_pose[1]).as_quat()
            record['capture_protocol'] = 'same_render_pass_same_simulation_timestamp'
            record['stabilized_near_clip_m'] = virtual_near_m
            record['raw_near_clip_m'] = .05
        if not self.frame_metadata:
            # A deterministic first-frame box from known evaluation fixture geometry,
            # used only to identify SAM's object, never as its measured mask.
            for key, p, r in [('reference', rp, rr), ('execution', ap, ar), ('stabilized', *stabilized_pose)]:
                if key not in self.cameras:
                    continue
                box = self.project_box(p, r)
                (self.root/key/'seed.json').write_text(json.dumps(dict(box_xyxy=box,
                    source='first_frame_projected_subject_fixture_bbox; offline_initialization_only'))+'\n')
        if self.camera_publisher is not None:
            intrinsics = [.5*self.args.height/self.args.width/np.tan(np.deg2rad(27.5)),
                          .5/np.tan(np.deg2rad(27.5)), .5, .5]
            self.camera_publisher.publish(self.output_rgb, raw_rgb[:, :, ::-1].copy(),
                record, capture_clock_before, intrinsics)
        self.frame_metadata.append(native(record))
        with (self.root/'frames.jsonl').open('a') as journal:
            journal.write(json.dumps(self.frame_metadata[-1], allow_nan=False)+'\n')

    def capture_rgb(self, key, index, advance_render=True):
        import numpy as np
        if advance_render:
            self.rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        rgba = np.asarray(self.annotators[key].get_data())
        if rgba.ndim != 3 or rgba.shape[:2] != (self.args.height, self.args.width):
            raise RuntimeError(f'RGB annotator is not ready: {key} {rgba.shape}')
        bgr = self.cv2.cvtColor(rgba[:, :, :3].copy(), self.cv2.COLOR_RGB2BGR)
        if self.recording:
            if not self.cv2.imwrite(str(self.root/key/f'{index:05d}.jpg'), bgr):
                raise RuntimeError('could not save paired RGB frame')
            self.writers[key].write(bgr)
        return bgr

    def project_box(self, p, r):
        import numpy as np
        center = np.asarray(self.subject['position'])
        radius, height = self.subject['radius'], self.subject['height']
        corners = np.array([center+[x, y, z] for x in (-radius, radius) for y in (-radius, radius)
                            for z in (-height/2, height/2+.07)])
        local = (corners-p)@r
        focal = .5*self.args.height/np.tan(np.deg2rad(55.)/2)
        x = .5*self.args.width-focal*local[:, 1]/local[:, 0]
        y = .5*self.args.height-focal*local[:, 2]/local[:, 0]
        return [float(np.clip(x.min()-3, 0, self.args.width-1)), float(np.clip(y.min()-3, 0, self.args.height-1)),
                float(np.clip(x.max()+3, 0, self.args.width-1)), float(np.clip(y.max()+3, 0, self.args.height-1))]

    def visibility(self, p, r):
        import numpy as np
        from omni.physx import get_physx_scene_query_interface
        center = np.asarray(self.subject['position'])
        delta = center-p
        local = r.T@delta
        focal = .5*self.args.height/np.tan(np.deg2rad(55.)/2)
        if local[0] <= 0:
            return dict(available=True, out_of_frame=True, occluded=False, visible_fraction=0.)
        u = .5*self.args.width-focal*local[1]/local[0]
        v = .5*self.args.height-focal*local[2]/local[0]
        out = not (0 <= u < self.args.width and 0 <= v < self.args.height)
        angle = np.arctan2(p[1]-center[1], p[0]-center[0])
        query = get_physx_scene_query_interface()
        visible, total = 0, 0
        for a in angle+np.linspace(-.7, .7, 5):
            for z in np.linspace(-.4, .4, 5)*self.subject['height']:
                target = center+np.array([self.subject['radius']*np.cos(a), self.subject['radius']*np.sin(a), z])
                direction = target-p
                distance = np.linalg.norm(direction)
                direction /= distance
                # Start outside the camera housing, matching the near clipping plane.
                hit = query.raycast_closest(tuple(p+.05*direction), tuple(direction), float(distance))
                visible += int(hit.get('hit', False) and 'BlueSubject' in str(hit.get('collision', '')))
                total += 1
        return dict(available=True, out_of_frame=out, occluded=visible < total and not out,
                    visible_fraction=visible/total, source='independent_physx_subject_surface_visibility')

    def close(self):
        from jev_agent_wbc.evaluation import save_json
        from scipy.spatial.transform import Rotation
        rendered = 0
        try:
            if not self.recording or getattr(self.args, 'skip_reference_video', False):
                return
            if self.frame_metadata:
                first = self.frame_metadata[0]
                self.pose('reference', first['reference_camera_position'],
                    Rotation.from_quat(first['reference_camera_quaternion_xyzw']).as_matrix())
            for obj in self.hide_for_reference:
                obj.MakeInvisible()
            # Flush visibility changes once, outside the shot/control clock.
            for _ in range(4):
                self.rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
            for row in self.frame_metadata:
                self.pose('reference', row['reference_camera_position'],
                          Rotation.from_quat(row['reference_camera_quaternion_xyzw']).as_matrix())
                self.capture_rgb('reference', row['frame'])
                rendered += 1
        finally:
            for obj in self.hide_for_reference:
                obj.MakeVisible()
            for writer in self.writers.values():
                writer.release()
            save_json(self.root/'frames.json', self.frame_metadata)
            save_json(self.root/'reference_rendering.json', dict(
                protocol='static_reference_separate_pass/v2',
                skipped=not self.recording or bool(getattr(self.args, 'skip_reference_video', False)),
                frames_expected=0 if not self.recording or getattr(self.args, 'skip_reference_video', False) else len(self.frame_metadata),
                frames_written=rendered,
                valid=None if not self.recording or getattr(self.args, 'skip_reference_video', False) else rendered == len(self.frame_metadata),
                scope='stationary subject and background fixture; original-time camera poses; not online perception'))
            try:
                if self.recording and not getattr(self.args, 'skip_reference_video', False):
                    from jev_agent_wbc.camera_output import write_three_camera_video
                    comparison = write_three_camera_video(self.root, self.frame_metadata, self.args.record_fps)
                    save_json(self.root/'comparison.json', comparison)
            finally:
                for annotator in self.annotators.values():
                    try:
                        annotator.detach()
                    except RuntimeError:
                        pass
                for product in self.products.values():
                    try:
                        product.destroy()
                    except RuntimeError:
                        pass
                if self.camera_publisher is not None:
                    self.camera_publisher.close()


def main():
    args = arguments()
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    os.environ.setdefault('ACCEPT_EULA', 'Y')
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=args.headless, width=args.width, height=args.height,
                             renderer='RaytracedLighting', anti_aliasing=0, multi_gpu=False,
                             limit_cpu_threads=8))
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
        if args.virtual_stabilizer_ros or args.publish_camera_ros:
            manager.set_extension_enabled_immediate('isaacsim.ros2.bridge', True)
        for _ in range(3):
            app.update()
        from jev_agent_wbc.scenarios import NAMES
        from jev_agent_wbc.evaluation import save_json
        from jev_agent_wbc.provenance import source_manifest
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output/'provenance.json', source_manifest(ROOT,
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}))
        names = list(NAMES) if args.case == 'all' else args.case.split(',')
        for name in names:
            if (Path(name).name != name or name in ('.', '..') or
                    not all((args.scenarios/name/file).is_file() for file in ('task.json', 'runtime_scene.json'))):
                raise ValueError(f'missing or invalid authored case: {name}')
        results = []
        for name in names:
            for mode in (['strict', 'jev'] if args.mode == 'both' else [args.mode]):
                results.append(run_case(args, app, name, mode, args.output/f'{name}_{mode}'))
                save_json(args.output/'runs.json', results)
        return 0 if all(r['completed'] for r in results) else 2
    except Exception as error:
        import traceback
        traceback.print_exc()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'runner_failure.json').write_text(json.dumps(dict(error=str(error), traceback=traceback.format_exc()))+'\n')
        return 2
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(main())
