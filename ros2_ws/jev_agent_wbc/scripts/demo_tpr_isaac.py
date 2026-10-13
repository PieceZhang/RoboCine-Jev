#!/usr/bin/env python3
"""PhysX version of the fixed-camera reconfiguration demo.

The controller receives the current tracked sphere pose and velocity only.
Robot state, camera pose and contacts are measured from PhysX. This isolates
WBC behavior from LiDAR tracking quality; it is not a perception benchmark.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jvs_isaac_sim as backend


def main():
    parser = backend.build_parser()
    parser.set_defaults(headless=True, physics_hz=1000., enable_lidar=False,
                        actuator_armature=.05, wheel_drive_damping=1000.,
                        arm_drive_stiffness=1800.,arm_drive_damping=400.,solver_velocity_iterations=64,
                        xacro=str(ROOT/'urdf/jev_agent_wbc.urdf.xacro'))
    for action in parser._actions:
        if action.dest == 'xacro':
            action.required = False
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=48.)
    parser.add_argument('--period', type=float, default=48.)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record-fps', type=float, default=12.)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--servo-frequency', type=float, default=20.)
    parser.add_argument('--velocity-window-ms', type=float, default=2.)
    parser.add_argument('--yaw-response-s', type=float, default=0.)
    parser.add_argument('--yaw-acceleration', type=float, default=.3)
    args = parser.parse_args()
    backend.validate_args(args)
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, width=args.width, height=args.height,
                             renderer='RaytracedLighting', anti_aliasing=0, multi_gpu=False))
    try:
        run(args, app)
    finally:
        app.close()


def run(args, app):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    from omni.physx import get_physx_simulation_interface
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from demo_tpr_wbc import FixedCamera, flight, scene, OBSTACLE_RADIUS
    from run_isaac import RGBPair
    from jev_agent_wbc.planning import ExecutionWBC
    from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
    from jev_agent_wbc.dynamic_collision import DynamicCollision
    from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER
    from jev_agent_wbc.evaluation import save_json
    from jev_agent_wbc.wbc_metrics import PhysicsAudit
    from jev_agent_wbc.reference import look_at
    from collections import deque
    from jev_agent_wbc.provenance import source_manifest

    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'provenance.json',source_manifest(ROOT,
        {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}))
    sim = controller = recorder = audit = None
    rows, contacts, failure = [], [], ''
    obstacle_position_error=0.;obstacle_samples=0
    started = time.monotonic()
    initial = np.r_[0.,0.,0.,0.,-.4,1.7,0.,-1.3,0.]
    physical_t = 0.
    try:
        args.spawn_x, args.spawn_y, args.spawn_yaw, args.spawn_z = 0., 0., 0., .002
        stage = backend.prepare_stage(args, app)
        with tempfile.TemporaryDirectory(prefix='tpr_robot_') as folder:
            urdf = Path(folder)/'robot.urdf'
            backend.expand_xacro(args, urdf)
            robot_path = backend.import_robot(urdf)
            backend.isolate_urdf_templates(stage, robot_path)
            app.update()
            backend.configure_articulation(stage, robot_path, args, urdf)
        backend.configure_physics_rate(stage, args.physics_hz, args.solver_type)
        robot_scope = str(Path(robot_path).parent)
        dome = UsdLux.DomeLight.Define(stage, '/World/SoftFill')
        dome.CreateIntensityAttr(800.)
        mover = UsdGeom.Xform.Define(stage, '/World/Moving/FlyingBall')
        mover_translation=mover.AddTranslateOp()
        mover_translation.Set(Gf.Vec3d(*flight(0., args.period)[0]))
        UsdPhysics.RigidBodyAPI.Apply(mover.GetPrim()).CreateKinematicEnabledAttr(True)
        ball = UsdGeom.Sphere.Define(stage, '/World/Moving/FlyingBall/Shape')
        ball.CreateRadiusAttr(OBSTACLE_RADIUS)
        ball.CreateDisplayColorAttr([Gf.Vec3f(.95,.18,.05)])
        UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
        # Kinematic tensor transforms are not mirrored to render USD. Writing
        # a USD pose back onto this collider can override the NEXT PhysX target.
        # Keep one collider and one non-colliding display sphere; the latter is
        # driven exclusively by the collider's measured pose.
        UsdGeom.Imageable(mover.GetPrim()).MakeInvisible()
        display_ball=UsdGeom.Sphere.Define(stage,'/World/Moving/FlyingBallDisplay')
        display_ball.CreateRadiusAttr(OBSTACLE_RADIUS)
        display_ball.CreateDisplayColorAttr([Gf.Vec3f(.95,.18,.05)])
        display_translation=display_ball.AddTranslateOp()
        display_translation.Set(Gf.Vec3d(*flight(0.,args.period)[0]))
        subject_data = dict(position=[1.6,0.,.85], radius=.12, height=.32)
        subject = UsdGeom.Cylinder.Define(stage, '/World/BlueSubject')
        subject.CreateRadiusAttr(.12); subject.CreateHeightAttr(.32)
        subject.AddTranslateOp().Set(Gf.Vec3d(*subject_data['position']))
        subject.CreateDisplayColorAttr([Gf.Vec3f(.03,.35,.9)])
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        sim = SimulationContext(physics_dt=1/args.physics_hz, rendering_dt=.02, stage_units_in_meters=1.)
        sim.initialize_physics(); sim.play()
        robot = SingleArticulation(robot_path); robot.initialize()
        camera = SingleRigidPrim(robot_scope+'/camera_link'); camera.initialize()
        names = list(robot.dof_names)
        indices = np.array([names.index(n) for n in ['left_wheel_joint','right_wheel_joint']+[f'joint{i}' for i in range(1,7)]])
        movers = RigidPrim('/World/Moving/FlyingBall', name='tpr_mover', reset_xform_properties=False); movers.initialize()
        transforms = movers._physics_view.get_transforms().copy()
        midx = np.array([0], dtype=np.uint32)
        qcmd = initial[3:].copy()
        robot.set_joint_positions(qcmd, joint_indices=indices[2:])
        command = np.zeros(8)
        physical_limits = Limits()
        limits = Limits(command_ramp=True, position_margin=.03,
                        velocity=physical_limits.velocity*.95, acceleration=physical_limits.acceleration*.4,
                        wheel_speed=physical_limits.wheel_speed*.95, wheel_acceleration=physical_limits.wheel_acceleration*.4)
        limits.acceleration[1] = args.yaw_acceleration
        limits.response_time[1]=args.yaw_response_s

        def apply(u):
            robot.apply_action(ArticulationAction(joint_positions=np.r_[0.,0.,qcmd],
                joint_velocities=np.r_[limits.wheels@u,u[2:]], joint_indices=indices))

        def observe():
            bp, bq = robot.get_world_pose()
            br = Rotation.from_quat(np.roll(bq,-1)).as_matrix()
            yaw = float(np.arctan2(br[1,0],br[0,0]))
            state = np.r_[bp[:2],yaw,robot.get_joint_positions(indices[2:])]
            v = robot.get_joint_velocities(indices)
            local = br.T@robot.get_linear_velocity()
            measured = np.r_[local[0],robot.get_angular_velocity()[2],v[2:]]
            cp, cq = camera.get_world_pose()
            cr = Rotation.from_quat(np.roll(cq,-1)).as_matrix()
            return state,bp,br,measured,v[:2],local[1],cp,cr

        apply(command)
        for k in range(round(1.5*args.physics_hz)):
            sim.step(render=False)
            if k%100 == 0:
                sim.render()
        state,bp,br,measured,wheels,lateral,cp,cr = observe()
        reference = FixedCamera(state,args.duration)
        reference.p, reference.r = cp.copy(), cr.copy()
        controller = ExecutionWBC(limits, scene=scene(0.,args.period))
        controller.measured_limits=physical_limits
        controller.tpr_servo_frequency=args.servo_frequency
        controller.set_trajectory(reference)
        window=max(1,round(args.velocity_window_ms*.001*args.physics_hz))
        velocity_samples=deque([measured.copy()],maxlen=window)
        base_samples=deque([np.r_[robot.get_linear_velocity(),robot.get_angular_velocity()]],maxlen=window)
        audit = PhysicsAudit(str(args.output),physical_limits,1/args.physics_hz)

        def on_contacts(headers, data):
            for h in headers:
                paths = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.actor0,h.actor1)]
                if any('/FlyingBall' in p for p in paths) and any(p.startswith(robot_scope+'/') for p in paths):
                    contacts.append(dict(t=physical_t,actors=paths,points=int(h.num_contact_data)))
        subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
        if args.record:
            recorder = RGBPair(stage,sim,args,args.output,robot_scope,subject_data)
            overview = np.array([1.45,-1.75,1.12])
            recorder.pose('overview',overview,look_at(overview,np.array([.05,0.,.46])))
            save_json(args.output/'video/overview_calibration.json',dict(position=overview,
                rotation=look_at(overview,np.array([.05,0.,.46])),width=args.width,height=args.height,vfov_deg=55.))
        next_frame=0.; dt=.02; substeps=round(args.physics_hz*dt)
        for tick in range(round(args.duration/dt)+1):
            t=tick*dt
            state,bp,br,measured,wheels,lateral,cp,cr=observe()
            obstacle_measured=movers._physics_view.get_transforms()[0,:3].copy()
            # PhysX tensor kinematic targets move the collider, but do not
            # write this kinematic actor's transform back to USD for rendering.
            # Copy the READ-BACK physical pose, never the requested flight pose.
            display_translation.Set(Gf.Vec3d(*map(float,obstacle_measured)))
            obstacle_usd=np.array(UsdGeom.Xformable(display_ball).ComputeLocalToWorldTransform(0).ExtractTranslation())
            observed_scene=scene(t,args.period)
            observed_scene['surfel_centers']=[obstacle_measured.tolist()]
            controller.update_collision_map(observed_scene);controller.set_time(t)
            # Causal encoder/odometry velocity estimate, never commanded
            # velocity. The independent audit retains every raw physics sample.
            controller.measured_control=np.mean(velocity_samples,axis=0)
            controller.measured_base_twist=np.mean(base_samples,axis=0)
            old=controller.previous.copy()
            u,info=controller.solve(state,*reference.sample(t),dt,base_z=bp[2],base_rotation=br,
                                    commanded_q=qcmd,trajectory_time=t)
            truth=DynamicCollision(observed_scene,t,uncertainty_rate=0.)
            fk,fr,_=whole_body_kinematics(state,bp[2],br)
            obstacle_requested,_=flight(t,args.period)
            rows.append(dict(t=t,state=state.tolist(),command=u.tolist(),measured=measured.tolist(),
                estimated_velocity=controller.measured_control.tolist(),velocity_window_ms=args.velocity_window_ms,
                obstacle=obstacle_measured.tolist(),obstacle_requested=obstacle_requested.tolist(),
                camera_position=cp.tolist(),camera_rotation=cr.tolist(),
                obstacle_physx_position=obstacle_measured.tolist(),obstacle_usd_position=obstacle_usd.tolist(),
                position_error_m=float(np.linalg.norm(cp-reference.p)),
                orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(cr@reference.r.T).as_rotvec())),
                clearance_m=float(truth.clearance(state,bp[2],br).min()),
                base_z=float(bp[2]),base_rotation=br.tolist(),fk_error_m=float(np.linalg.norm(fk-cp)),info=info))
            if recorder and t+1e-8>=next_frame:
                recorder.capture(t,reference.p,reference.r,cp,cr,None)
                next_frame+=1/args.record_fps
            if tick%100==0:
                print(f'TPR_PHYSX t={t:.2f} error={rows[-1]["position_error_m"]:.6f} clearance={rows[-1]["clearance_m"]:.4f} yaw={state[2]:.4f} omega_command={u[1]:.4f} status={info["tpr_status"]}',flush=True)
            if t>=args.duration:
                break
            for sub in range(substeps):
                physical_t=t+(sub+1)/args.physics_hz
                command=old+(sub+.5)/substeps*(u-old)
                qcmd+=command[2:]/args.physics_hz
                apply(command)
                transforms[0,:3]=flight(physical_t,args.period)[0]
                movers._physics_view.set_kinematic_targets(transforms,midx)
                sim.step(render=False)
                actual_obstacle=movers._physics_view.get_transforms()[0,:3].copy()
                obstacle_position_error=max(obstacle_position_error,
                    float(np.linalg.norm(actual_obstacle-flight(physical_t,args.period)[0])))
                obstacle_samples+=1
                if obstacle_position_error>1e-5:
                    raise RuntimeError('physical obstacle did not follow its kinematic target')
                ss=observe();audit.record(physical_t,ss[0][3:],ss[3],ss[4],ss[5])
                velocity_samples.append(ss[3].copy())
                base_samples.append(np.r_[robot.get_linear_velocity(),robot.get_angular_velocity()])
            if tick%10==0:
                sim.render()
        del subscription
    except Exception as error:
        import traceback
        traceback.print_exc()
        failure=f'{type(error).__name__}: {error}'
    finally:
        if controller:
            controller.close()
        physical=audit.finish() if audit else None
        if recorder:
            recorder.close()
        summary=dict(backend='PhysX_measured_camera_and_robot',duration_s=args.duration,
                     completed=bool(rows and rows[-1]['t']>=args.duration),failure=failure,
                     contacts=len(contacts),wall_s=time.monotonic()-started,physical_audit=physical,
                     observation='ideal current tracked sphere pose/velocity; no future flight path to WBC')
        summary['velocity_estimator_window_ms']=args.velocity_window_ms
        summary['obstacle_period_s']=args.period
        summary['obstacle_motion_audit']=dict(samples=obstacle_samples,
            physics_position_max_error_m=obstacle_position_error,
            render_pose_source='PhysX collider read-back copied to a non-colliding display sphere before capture')
        summary['received_camera_position']=reference.p.tolist() if rows else None
        summary['received_camera_rotation']=reference.r.tolist() if rows else None
        if rows:
            summary.update(position_max_m=max(r['position_error_m'] for r in rows),
                           orientation_max_rad=max(r['orientation_error_rad'] for r in rows),
                           minimum_clearance_m=min(r['clearance_m'] for r in rows),
                           fk_position_max_m=max(r['fk_error_m'] for r in rows),
                           degraded_steps=sum(r['info']['tpr_status']=='degraded' for r in rows))
            states=np.array([row['state'] for row in rows])
            summary['joint_excursion_rad']=np.ptp(states[:,3:],axis=0).tolist()
            summary['base_xy_excursion_m']=np.ptp(states[:,:2],axis=0).tolist()
            summary['base_yaw_excursion_rad']=float(np.ptp(states[:,2]))
            summary['obstacle_motion_audit']['usd_position_max_error_m']=float(max(
                np.linalg.norm(np.array(row['obstacle_usd_position'])-row['obstacle_physx_position']) for row in rows))
            summary['pass']=(summary['completed'] and not contacts and summary['minimum_clearance_m']>=.025
                and summary['position_max_m']<.001 and summary['orientation_max_rad']<.002
                and summary['degraded_steps']==0 and bool(physical and physical['pass_constraints']))
        save_json(args.output/'summary.json',summary)
        save_json(args.output/'telemetry.json',rows)
        save_json(args.output/'contacts.json',contacts)
        if recorder and rows:
            try:
                from audit_tpr_video import audit as audit_video
                motion=audit_video(args.output)
                summary['video_motion_audit']={k:v for k,v in motion.items() if k!='frames_checked'}
            except Exception as error:
                summary['video_motion_audit']=dict(pass_motion=False,error=str(error))
            summary['pass']=bool(summary.get('pass',False) and summary['video_motion_audit']['pass_motion'])
            save_json(args.output/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        if sim:
            sim.stop();SimulationContext.clear_instance()


if __name__=='__main__':
    main()
