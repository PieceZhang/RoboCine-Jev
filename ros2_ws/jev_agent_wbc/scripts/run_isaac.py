#!/usr/bin/env python3
"""Headless PhysX with nonblocking controller transport and actuator watchdog.

Sparse dual LiDAR uses real PhysX closest-hit queries (not obstacle truth).
Robot/camera poses and contact audits come from PhysX. The detached reference
camera follows the original path in the same live scene and render pass. The
raw physical camera supplies Agent camera feedback by default. SAM is run
separately after Kit exits so the two large GPU consumers never overlap.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NUMERIC_SITE=os.environ.get('JEV_PHYSX_NUMERIC_SITE')
if NUMERIC_SITE:
    sys.path.insert(0,str(Path(NUMERIC_SITE).resolve()))

import jvs_isaac_sim as backend


def resolve_default_xacro(args):
    """Find the installed ROS helper without importing its ABI into Isaac."""
    import shutil
    if args.xacro_executable!='xacro' or shutil.which('xacro') is not None:return
    root=Path('/opt/ros')/os.environ.get('ROS_DISTRO','humble')
    executable=root/'bin/xacro'
    if not executable.is_file():return
    args.xacro_executable=str(executable)
    paths=[str(p) for pattern in ('lib/python*/site-packages','local/lib/python*/dist-packages')
        for p in root.glob(pattern) if p.is_dir()]
    existing=os.environ.get('JVS_XACRO_PYTHONPATH',os.environ.get('PYTHONPATH',''))
    os.environ['JVS_XACRO_PYTHONPATH']=os.pathsep.join(filter(None,[existing,*paths]))


def arguments():
    parser = backend.build_parser()
    parser.set_defaults(headless=True, physics_hz=100., enable_lidar=False,
                        actuator_armature=0., arm_drive_stiffness=1800.,
                        arm_drive_damping=100000., wheel_drive_damping=100000.,
                        solver_velocity_iterations=128, caster_friction=0.,
                        xacro=str(ROOT/'urdf/jev_agent_wbc.urdf.xacro'))
    # The inherited xacro option was originally required; this package supplies its own.
    for action in parser._actions:
        if action.dest == 'xacro':
            action.required = False
    parser.add_argument('--case', default='all')
    parser.add_argument('--session-controller', action=argparse.BooleanOptionalAction, default=True,
                        help='use measured prepared-shot / continuous-combo CPU Agent driver')
    parser.add_argument('--startup-preplan', action=argparse.BooleanOptionalAction, default=True,
                        help='prepare a complete initial candidate; disable to match the CPU online random-scene protocol')
    parser.add_argument('--episode-time-limit',type=float,default=240.)
    parser.add_argument('--recovery-timeout',type=float,default=15.)
    parser.add_argument('--startup-planning-timeout',type=float,default=240.)
    parser.add_argument('--fault-script',type=Path,
                        help='explicit diagnostic controller delays/errors and LiDAR gaps; physics continues')
    parser.add_argument('--session-command-mode',choices=['velocity','acceleration','adaptive'],default='adaptive')
    parser.add_argument('--execution-aware',action=argparse.BooleanOptionalAction,default=True,
                        help='predict delayed WBC/drive execution and update empirical capability online')
    parser.add_argument('--execution-calibration',type=Path,default=ROOT/'config/jvs_mock_execution_model.json',
                        help='jvs_mock execution-model JSON with calibration provenance')
    parser.add_argument('--delay-aware-servo',action=argparse.BooleanOptionalAction,default=False,
                        help='diagnostic measured-delay gain adaptation; physical safety gates unchanged')
    parser.add_argument('--scenarios', type=Path, default=ROOT/'scenarios')
    parser.add_argument('--mode', choices=['jev', 'strict', 'rule', 'both'], default='jev')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='perplexity', help='explicit backend model ID')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000/v1/systemone')
    from jev_agent_wbc.camera_output import add_camera_arguments
    add_camera_arguments(parser)
    parser.set_defaults(camera_output='execution')
    parser.add_argument('--virtual-stabilizer-ros', action='store_true', help='independent reference-aware virtual camera via timestamped ROS batches')
    parser.add_argument('--skip-reference-video', action='store_true', help='omit the original-reference comparison view')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--save-replay', action='store_true',
                        help='save measured link poses for a separate offline render; no RGB work during execution')
    parser.add_argument('--record-fps', type=float, default=8.)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--render-subframes', type=int, choices=(1,2), default=1,
                        help='same-time render subframes; no material changes occur inside the shot')
    parser.add_argument('--pace', type=float, choices=(1.,), default=1., help='real-time target; slowdown is disabled')
    parser.add_argument('--command-timeout', type=float, default=.10)
    parser.add_argument('--sensor-timeout', type=float, default=.20)
    parser.add_argument('--planning-timeout', type=float, default=1.)
    parser.add_argument('--startup-command-timeout', type=float, default=.20)
    parser.add_argument('--single-candidate', action='store_true')
    parser.add_argument('--compiled-geometry',action=argparse.BooleanOptionalAction,default=False,
                        help='optional CPU-compiled exact primitive geometry; warm before arming')
    parser.add_argument('--profile-controller',action='store_true')
    parser.add_argument('--guard-recovery',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--certificate-recovery',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--cinegym-bridge',type=Path,
                        help='private local IPC descriptor for an external CineGym policy')
    parser.add_argument('--agent-period', type=float, default=.5)
    parser.add_argument('--activation-lead', type=float, default=.9)
    parser.add_argument('--candidate-budget', type=int, default=2)
    parser.add_argument('--candidate-proposals', type=int, default=6)
    parser.add_argument('--candidate-wall-budget', type=float, default=18.)
    parser.add_argument('--prediction-horizon', type=float, default=8.)
    parser.add_argument('--prediction-step', type=float, default=.1)
    parser.add_argument('--audit-prediction', action='store_true', help='save bounded observation snapshots for offline predictor diagnostics')
    parser.add_argument('--lidar-backend',choices=['native','scalar'],default='native')
    parser.add_argument('--lidar-query-threads',type=int,choices=range(1,9),default=8)
    parser.add_argument('--lidar-processing',choices=['async','sync'],default='async')
    parser.add_argument('--lidar-hz',type=float,default=10.,
                        help='requested live scan rate; one in-flight map, no queued backlog')
    parser.add_argument('--lidar-pipeline',action=argparse.BooleanOptionalAction,default=True,
                        help='overlap read-only ray queries with CPU supervision; join before any physical write/render')
    parser.add_argument('--lidar-static-input',choices=['cpu_fixture','full_raycast'],default='cpu_fixture',
                        help='CPU calibrated static background + real moving ray hits, or full observed surfaces')
    parser.add_argument('--motor-targets',choices=['cpu_hold','physical_slew'],default='physical_slew',
                        help='physical drive interface; shared WBC still uses the unmodified CPU endpoint command contract')
    parser.add_argument('--duration-limit',type=float,help='diagnostic executed prefix only; never a completed-shot result')
    parser.add_argument('--profile-loop', action='store_true', help='save cProfile data for the live loop (diagnostic overhead)')
    parser.add_argument('--validate-fast-paths', action='store_true',
                        help='diagnostic comparison with wrapper states and uncached raycasts')
    parser.add_argument('--fabric', action=argparse.BooleanOptionalAction, default=True,
                        help='render from PhysX Fabric without per-step USD transform writeback')
    parser.add_argument('--physics-threads', type=int, default=1,
                        help='optional PhysX CPU dispatcher thread count for single-scene benchmarks')
    parser.add_argument('--base-support',choices=['upright','free'],default='upright',
                        help='upright matches CPU planar-base roll/pitch assumptions; XYZ and yaw remain dynamic')
    parser.add_argument('--solver-position-iterations',type=int,default=255)
    parser.add_argument('--no-preview', action='store_true', help='ablate inherited allocation MPC, retaining safety and bounded task planner')
    parser.add_argument('--allocation-preview', action='store_true', help='enable inherited shot-specific base-allocation MPC for ablation')
    parser.add_argument('--async-coordination', action='store_true', default=True,
                        help='run the cross-phase allocation in its live nonblocking worker instead of lockstep')
    parser.add_argument('--acceleration-reserve', type=float, default=1.,
                        help='operating acceleration / physical acceleration; evaluation limits stay fixed')
    parser.add_argument('--yaw-acceleration-reserve', type=float, default=None,
                        help='optional separate operating fraction for chassis yaw acceleration')
    args = parser.parse_args()
    from jev_agent_wbc.runtime_fault_injection import FaultSchedule
    FaultSchedule.load(args.fault_script)
    from jev_agent_wbc.realtime import RuntimePolicy
    RuntimePolicy(args.command_timeout, args.sensor_timeout, args.planning_timeout, args.startup_command_timeout,
                  recover_decisions=args.session_controller,recovery_timeout_s=args.recovery_timeout)
    if any(not math.isfinite(value) or value<=0 for value in
           (args.episode_time_limit,args.startup_planning_timeout)):
        parser.error('episode and startup planning limits must be finite and positive')
    if args.virtual_stabilizer_ros:
        parser.error('blocking ROS stabilizer batches are incompatible with this realtime runner')
    if args.session_controller and args.startup_preplan and not args.single_candidate:
        args.single_candidate=True
        args.candidate_proposals=args.candidate_budget=1
    if args.motor_targets != 'physical_slew':
        parser.error('realtime watchdog requires the acceleration-limited physical_slew drive')
    if args.execution_aware and (args.physics_hz!=100. or args.base_support!='upright'):
        parser.error('execution prediction is calibrated for 100 Hz upright jvs_mock; other backends require --no-execution-aware')
    if (not math.isfinite(args.physics_hz) or args.physics_hz < 50
            or abs(args.physics_hz/50-round(args.physics_hz/50)) > 1e-8):
        parser.error('physics-hz must be a multiple of 50 and at least 50')
    resolve_default_xacro(args)
    backend.validate_args(args)
    if args.physics_threads is not None and args.physics_threads < 1:
        parser.error('physics-threads must be positive')
    if not 0 < args.lidar_hz <= 50:
        parser.error('lidar-hz must be in (0,50]')
    if args.arm_drive_stiffness<=0 or args.wheel_drive_damping<=0:
        parser.error('physical inverse-dynamics drive requires positive stiffness and damping')
    if not 1<=args.solver_position_iterations<=255:parser.error('solver position iterations must be in [1,255]')
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
    from jev_agent_wbc.sensors import RuntimeLidar, RAYS
    from physx_compat import install_worker_adapter, subject_from_returns, MotorTargets, configure_rolling_contacts, generalized_gravity
    os.environ['JEV_PHYSX_COMPILED_GEOMETRY']='1' if args.compiled_geometry else '0'
    install_worker_adapter()
    from jev_agent_wbc.realtime_engine import RealtimeEngine
    from jev_agent_wbc.realtime import RuntimePolicy, realtime_assessment
    from jev_agent_wbc.candidates import SearchBudget
    from jev_agent_wbc.evaluation import EVALUATION_PROTOCOL, save_json, trajectory_summary
    from jev_agent_wbc.dynamic_collision import DynamicCollision
    from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
    from jev_agent_wbc.wbc_metrics import PhysicsAudit
    from jev_agent_wbc.actuation import certify_control_ramp
    from jev_agent_wbc.camera_intent import POLICY, IntentSpec, visibility, progress_at
    from jev_agent_wbc.task_space_planning import camera_scene
    from jev_agent_wbc.contracts import native

    output.mkdir(parents=True, exist_ok=False)
    from jev_agent_wbc.runtime_fault_injection import FaultSchedule
    fault_schedule=FaultSchedule.load(args.fault_script)
    if args.fault_script is not None:
        save_json(output/'fault_injection.json',dict(events=fault_schedule.events,
            role='diagnostic perturbation; not an unperturbed performance run'))
    runtime_scene = RuntimeScene(args.scenarios/name/'runtime_scene.json')
    fixture = runtime_scene.document
    from copy import copy
    args = copy(args)
    args.spawn_x, args.spawn_y, args.spawn_yaw = fixture['initial_state'][:3]
    args.spawn_z = fixture['base_z']
    source_task=json.loads((args.scenarios/name/'task.json').read_text())
    from jev_agent_wbc.combo import gaps
    session_combo=bool(gaps(source_task))
    use_session=bool(args.session_controller and mode=='jev' and args.cinegym_bridge is None)
    if session_combo and not use_session:
        raise ValueError('combo phase gaps require --session-controller --mode jev')
    if session_combo:
        from jev_agent_wbc.continuous_combo import action_task
        reference=OriginalReference(action_task(source_task,source_task['shot_phase'][0]),fixture['subject']['position'])
    else:
        reference = OriginalReference(source_task, fixture['subject']['position'])
    save_json(output/'source_task.json',source_task)
    save_json(output/'original_task.json', reference.task)
    save_json(output/'runtime_scene.json', fixture)
    save_json(output/'evaluation_protocol.json', EVALUATION_PROTOCOL)
    engine = audit = recorder = stabilizer_client = lidar_worker = None
    lidar_failed = False
    native_ray = None
    restore_gc = False
    replay_poses = []
    execution_journal = None
    from jev_agent_wbc.camera_output import CameraOutput
    camera_output = CameraOutput.from_args(args)
    camera_samples = []
    stabilized_pose = None
    rows, contacts, failure = [], [], ''
    prediction_snapshots = []
    validation_failure_snapshot = None
    recording_error = ''
    physical_summary = None
    tick_times, tick_lags, tick_stages = [], [], []
    run_start = run_end = None
    loop_profile = None
    fast_path_validation = dict(state_samples=0, ray_samples=0, passed=None)
    physical_limits = limits = Limits()
    gravity_max = np.zeros(8)
    feedforward_max = np.zeros(8)
    mover_audit = dict(samples=0, max_target_error_m=0., max_display_error_m=0.)
    begin = time.monotonic()
    dt = .02
    sim = None
    try:
        SimulationContext.clear_instance()
        stage = backend.prepare_stage(args, app)
        if args.add_ground_plane:
            from isaacsim.core.api.objects import GroundPlane
            ground=GroundPlane(prim_path='/World/groundPlane',z_position=float(fixture['base_z']))
        with tempfile.TemporaryDirectory(prefix='jev_robot_') as folder:
            urdf = Path(folder)/'robot.urdf'
            backend.expand_xacro(args, urdf)
            robot_path = backend.import_robot(urdf)
            save_json(output/'import_collision_audit.json',
                      backend.isolate_urdf_templates(stage, robot_path))
            app.update()
            backend.configure_articulation(stage, robot_path, args, urdf)
            PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(robot_path)).CreateSolverPositionIterationCountAttr(args.solver_position_iterations)
            save_json(output/'wheel_contact_audit.json',configure_rolling_contacts(stage,robot_path))
        if args.base_support=='upright':
            support=UsdPhysics.Joint.Define(stage,'/World/UprightBaseSupport')
            support.CreateBody1Rel().SetTargets([robot_path])
            support.CreateExcludeFromArticulationAttr(True)
            for axis in ('rotX','rotY'):
                bound=UsdPhysics.LimitAPI.Apply(support.GetPrim(),axis)
                # USD/PhysX uses low > high for a locked degree of freedom.
                # Equal limits instead create two unsupported pyramid limits.
                bound.CreateLowAttr(1.)
                bound.CreateHighAttr(-1.)
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
        display_ops = {}
        for obj in runtime_scene.obstacles(0.):
            root = UsdGeom.Xform.Define(stage, '/World/Moving/'+obj['id'])
            root.AddTranslateOp().Set(Gf.Vec3d(*obj['center']))
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
            if obj.get('shape')=='box':
                shape=UsdGeom.Cube.Define(stage,str(root.GetPath())+'/Shape')
                shape.CreateSizeAttr(1.)
                shape.AddRotateZOp().Set(float(np.rad2deg(obj.get('yaw_rad',0.))))
                shape.AddScaleOp().Set(Gf.Vec3f(*obj['size']))
            else:
                shape = UsdGeom.Sphere.Define(stage, str(root.GetPath())+'/Shape')
                shape.CreateRadiusAttr(obj['radius'])
            shape.CreateDisplayColorAttr([Gf.Vec3f(.9, .14, .035)])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            # Never write USD transforms back to an active kinematic collider:
            # that can overwrite its next PhysX target. Render a separate,
            # non-colliding copy driven exclusively by measured physical poses.
            UsdGeom.Imageable(root.GetPrim()).MakeInvisible()
            if obj.get('shape')=='box':
                display=UsdGeom.Cube.Define(stage,'/World/MovingDisplay/'+obj['id'])
                display.CreateSizeAttr(1.)
            else:
                display = UsdGeom.Sphere.Define(stage, '/World/MovingDisplay/'+obj['id'])
                display.CreateRadiusAttr(obj['radius'])
            display.CreateDisplayColorAttr([Gf.Vec3f(.9, .14, .035)])
            display_ops[obj['id']] = display.AddTranslateOp()
            display_ops[obj['id']].Set(Gf.Vec3d(*obj['center']))
            if obj.get('shape')=='box':
                display.AddRotateZOp().Set(float(np.rad2deg(obj.get('yaw_rad',0.))))
                display.AddScaleOp().Set(Gf.Vec3f(*obj['size']))
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)

        sim = SimulationContext(physics_dt=1/args.physics_hz, rendering_dt=.02, stage_units_in_meters=1.)
        sim.get_physics_context().set_solver_type(args.solver_type)
        if args.physics_threads is not None:
            from omni.physx import get_physx_interface
            get_physx_interface().set_thread_count(args.physics_threads)
        from isaacsim.core.simulation_manager import SimulationManager
        SimulationManager.enable_fabric(args.fabric)
        sim.initialize_physics()
        sim.play()
        robot = SingleArticulation(robot_path)
        robot.initialize()
        camera_body = SingleRigidPrim(robot_scope+'/camera_link')
        camera_body.initialize()
        names = list(robot.dof_names)
        indices = np.array([names.index(n) for n in ['left_wheel_joint', 'right_wheel_joint']+[f'joint{i}' for i in range(1, 7)]])
        save_json(output/'physical_model.json',dict(dof_names=names,command_indices=indices,
            dof_properties={name:robot.dof_properties[name] for name in robot.dof_properties.dtype.names},
            gravity_raw=robot._articulation_view._physics_view.get_gravity_compensation_forces(),
            gravity_api=robot._articulation_view.get_generalized_gravity_forces(),
            mass_matrix=robot._articulation_view._physics_view.get_generalized_mass_matrices()))
        movers = RigidPrim('/World/Moving/*', name='jev_movers', reset_xform_properties=False)
        movers.initialize()
        targets = movers._physics_view.get_transforms().copy()
        mover_indices = np.arange(len(movers.prim_paths), dtype=np.uint32)
        mover_names = [str(p).split('/')[-1] for p in movers.prim_paths]

        def move_obstacles(t):
            objects = {o['id']: o for o in runtime_scene.obstacles(t)}
            targets[:, :3] = np.array([objects[n]['center'] for n in mover_names])
            movers._physics_view.set_kinematic_targets(targets, mover_indices)

        def audit_movers(t, display=False):
            measured_positions = movers._physics_view.get_transforms()[:, :3].copy()
            expected = {obj['id']: obj['center'] for obj in runtime_scene.obstacles(t)}
            error = max(float(np.linalg.norm(p-np.asarray(expected[name])))
                        for name, p in zip(mover_names, measured_positions))
            mover_audit['samples'] += 1
            mover_audit['max_target_error_m'] = max(mover_audit['max_target_error_m'], error)
            if error > .001:
                raise RuntimeError(f'PhysX obstacle target/readback mismatch: {error:.6f} m at {t:.3f} s')
            if display:
                for name, p in zip(mover_names, measured_positions):
                    display_ops[name].Set(Gf.Vec3d(*map(float, p)))
                    prim = stage.GetPrimAtPath('/World/MovingDisplay/'+name)
                    shown = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation())
                    mover_audit['max_display_error_m'] = max(mover_audit['max_display_error_m'],
                                                            float(np.linalg.norm(shown-p)))
            return measured_positions

        q_command = np.asarray(fixture['initial_state'][3:]).copy()
        robot.set_joint_positions(q_command, joint_indices=indices[2:])
        command = np.zeros(8)
        physical_limits = Limits()
        limits = Limits(acceleration=physical_limits.acceleration*args.acceleration_reserve,
                        wheel_acceleration=physical_limits.wheel_acceleration*args.acceleration_reserve)
        if args.yaw_acceleration_reserve is not None:
            limits.acceleration[1] = physical_limits.acceleration[1]*args.yaw_acceleration_reserve
        motors = MotorTargets(physical_limits)
        drive_velocity = np.zeros(8)
        gravity_max = np.zeros(8)
        previous_drive_velocity = np.zeros(8)
        previous_motor_control = np.zeros(8)
        physical_transition = None
        effort_caps=np.asarray(robot.dof_properties['maxEffort'][indices],float)

        physics_robot = robot._articulation_view._physics_view
        physics_camera = camera_body._rigid_prim_view._physics_view

        def apply(physical_state=None):
            nonlocal previous_drive_velocity, previous_motor_control
            raw_gravity=np.asarray(robot._articulation_view._physics_view.get_gravity_compensation_forces()[0])
            gravity=generalized_gravity(raw_gravity,len(names),indices)
            gravity_max[:]=np.maximum(gravity_max,np.abs(gravity))
            mass=np.asarray(robot._articulation_view._physics_view.get_generalized_mass_matrices()[0]).copy()
            if mass.shape!=(len(raw_gravity),len(raw_gravity)):
                raise RuntimeError('inconsistent PhysX floating-base mass/force dimensions')
            root_dofs=len(raw_gravity)-len(names)
            acceleration=np.zeros(len(raw_gravity))
            actual_joint_velocity=(physics_robot.get_dof_velocities()[0, indices]
                if physical_state is None else np.r_[physical_state[4],physical_state[3][2:]])
            acceleration[root_dofs+indices]=(drive_velocity-actual_joint_velocity)*args.physics_hz
            if root_dofs:
                rotation=(Rotation.from_quat(physics_robot.get_root_transforms()[0,3:7]).as_matrix()
                    if physical_state is None else physical_state[2])
                yaw=np.arctan2(rotation[1,0],rotation[0,0])
                forward=np.array([np.cos(yaw),np.sin(yaw),0.]);left=np.array([-forward[1],forward[0],0.])
                wl,wr=actual_joint_velocity[:2]
                wheel_control=np.r_[(wl+wr)*limits.wheel_radius/2,
                                   (wr-wl)*limits.wheel_radius/limits.wheel_separation,
                                   actual_joint_velocity[2:]]
                base_acc=(motors.control-wheel_control)*args.physics_hz
                acceleration[:3]=base_acc[0]*forward+motors.control[0]*motors.control[1]*left
                acceleration[5]=base_acc[1]
            # PhysX mass caches may omit the explicitly authored armature.
            controlled=root_dofs+indices
            if np.min(np.diag(mass)[controlled[2:]])<args.actuator_armature-1e-8:
                mass[controlled,controlled]+=args.actuator_armature
            coriolis=np.asarray(robot._articulation_view._physics_view.get_coriolis_and_centrifugal_compensation_forces()[0])
            generalized=raw_gravity+coriolis+mass@acceleration
            feedforward=generalized[controlled].copy()
            if root_dofs:
                traction=float(generalized[:3]@forward)*limits.wheel_radius/2
                turning=float(generalized[5])*limits.wheel_radius/limits.wheel_separation
                feedforward[:2]+=np.array([traction-turning,traction+turning])
            feedforward_max[:]=np.maximum(feedforward_max,np.abs(feedforward))
            feedforward=np.clip(feedforward,-effort_caps,effort_caps)
            previous_drive_velocity=drive_velocity.copy()
            previous_motor_control=motors.control.copy()
            # Feed forward dynamics through the position drive's single torque
            # budget. URDF effort caps remain on the SAME drive; no unbounded
            # external effort or state/velocity reset is applied.
            compensated_q=q_command+feedforward[2:]/args.arm_drive_stiffness
            compensated_velocity=drive_velocity.copy()
            compensated_velocity[:2]+=feedforward[:2]/args.wheel_drive_damping
            robot.apply_action(ArticulationAction(joint_positions=np.r_[0., 0., compensated_q],
                joint_velocities=compensated_velocity, joint_indices=indices))

        apply()
        for k in range(int(args.physics_hz*1.5)):
            sim.step(render=False)
            if k % 100 == 0:
                sim.render()
        # Finalize the authored reset after the importer/solver warm-up, before
        # timestamped pre-roll and before any Agent or runtime measurement.
        robot.set_joint_positions(q_command,joint_indices=indices[2:])
        robot.set_joint_velocities(np.zeros(len(names)))

        def observe():
            root_pose = physics_robot.get_root_transforms()[0].copy()
            bp = root_pose[:3]
            br = Rotation.from_quat(root_pose[3:7]).as_matrix()
            yaw = float(np.arctan2(br[1, 0], br[0, 0]))
            state = np.r_[bp[:2], yaw, physics_robot.get_dof_positions()[0,indices[2:]]]
            velocities = physics_robot.get_dof_velocities()[0,indices]
            twist = physics_robot.get_root_velocities()[0]
            local_v = br.T@twist[:3]
            measured = np.r_[local_v[0], twist[5], velocities[2:]]
            camera_pose = physics_camera.get_transforms()[0].copy()
            cp = camera_pose[:3]
            cr = Rotation.from_quat(camera_pose[3:7]).as_matrix()
            if args.validate_fast_paths:
                old_bp, old_bq = robot.get_world_pose()
                old_br = Rotation.from_quat(np.roll(old_bq,-1)).as_matrix()
                old_cp, old_cq = camera_body.get_world_pose()
                comparisons = [(bp,old_bp),(br,old_br),(cp,old_cp),
                    (cr,Rotation.from_quat(np.roll(old_cq,-1)).as_matrix()),
                    (state[3:],robot.get_joint_positions(indices[2:])),
                    (velocities,robot.get_joint_velocities(indices)),
                    (twist,np.r_[robot.get_linear_velocity(),robot.get_angular_velocity()])]
                for actual, expected in comparisons:
                    np.testing.assert_array_equal(actual,expected)
                fast_path_validation['state_samples'] += 1
            return state, bp, br, measured, velocities[:2], local_v[1], cp, cr

        query = get_physx_scene_query_interface()
        if args.lidar_backend == 'native':
            from physx_lidar import NativeRaycaster
            native_ray = NativeRaycaster(query,RAYS,robot_scope,output,threads=args.lidar_query_threads)
        import carb
        ray_directions = tuple(carb.Float3(*map(float, direction)) for direction in RAYS)
        raycast_closest = query.raycast_closest
        hit_classes = {}
        wall_points = np.array([[3.5,y,z] for y in np.linspace(-1.5,1.5,41)
                                for z in np.arange(.1,.8,.075)])
        lidar = RuntimeLidar()
        subject_track = None

        def scan(state, bp, stamp, batch=None):
            nonlocal subject_track
            origins = lidar.origins(state, bp[2])
            returns, subject_returns = [], []
            self_returns = total_returns = 0
            audit_hits = {} if abs(stamp) < 1e-8 else None
            if native_ray is not None:
                if batch is None:batch = native_ray.scan(origins,validate=abs(stamp)<1e-8)
                flags, hits = batch['flags'],batch['hit']
                self_returns = int(flags[:,:,0].sum())
                total_returns = int(hits.sum())
                for points, present, mask in zip(batch['positions'],hits,flags):
                    external = present & ~mask[:,0]
                    selected = external & (mask[:,1] if args.lidar_static_input=='cpu_fixture' else True)
                    cloud = points[selected].astype(float)
                    if args.lidar_static_input=='cpu_fixture':cloud=np.vstack([cloud,wall_points])
                    returns.append(cloud)
                    subject_returns.extend(points[external & mask[:,2]].astype(float))
                if abs(stamp)<1e-8:
                    audit_hits = {k:v for k,v in enumerate(batch['audit'])}
                    fast_path_validation['ray_samples'] += 2*len(RAYS)
            for origin in (origins if native_ray is None else []):
                points = []
                ray_origin = carb.Float3(*map(float, origin))
                for ray_index, direction in enumerate(ray_directions):
                    hit = raycast_closest(ray_origin, direction, 4.5)
                    if args.validate_fast_paths and abs(stamp) < 1e-8:
                        old_hit = query.raycast_closest(tuple(origin),tuple(RAYS[ray_index]),4.5)
                        assert hit['hit'] == old_hit['hit']
                        if hit['hit']:
                            # Compare every field consumed by the sensor adapter.
                            # protoIndex is unused (the scene has no point instancers).
                            np.testing.assert_array_equal(tuple(hit['position']),tuple(old_hit['position']))
                            for field in ('collision','rigidBody'):
                                assert hit[field] == old_hit[field], (field,hit[field],old_hit[field])
                        fast_path_validation['ray_samples'] += 1
                    if hit.get('hit'):
                        total_returns += 1
                        key = (hit.get('collision', ''), hit.get('rigidBody', ''))
                        classification = hit_classes.get(key)
                        if classification is None:
                            paths = tuple(map(str, key))
                            classification = (any(p.startswith(robot_scope+'/') for p in paths),
                                any(p.startswith('/World/Moving/') for p in paths),
                                'BlueSubject' in paths[0])
                            hit_classes[key] = classification
                        is_self, is_moving, is_subject = classification
                        if audit_hits is not None:
                            record = audit_hits.setdefault(key, dict(collision=key[0], rigid_body=key[1], count=0, sample_points=[]))
                            record['count'] += 1
                            if len(record['sample_points']) < 3:
                                record['sample_points'].append(list(hit['position']))
                        # Known robot body identity is available to this
                        # simulator sensor adapter. Discard its first hit but
                        # do not raycast through the robot: occlusion remains.
                        # No external object identity, pose or velocity is used.
                        if is_self:
                            self_returns += 1
                            continue
                        point = hit['position']
                        if args.lidar_static_input=='full_raycast' or is_moving:
                            points.append(point)
                        if is_subject:
                            subject_returns.append(point)
                points=np.asarray(points).reshape(-1,3)
                if args.lidar_static_input=='cpu_fixture':
                    # Exactly the static calibration used by RuntimeLidar.synthetic_scan.
                    # Moving position/velocity truth is never supplied to mapping.
                    points=np.vstack([points,wall_points])
                returns.append(points)
            acquired_subject = dict(valid=False, orientation_valid=False, stamp_s=stamp,
                                 reason='no_current_subject_lidar_returns')
            if subject_returns:
                estimate = subject_from_returns(np.asarray(subject_returns), subject_data, stamp)
                if estimate['valid']:
                    acquired_subject = estimate
            diagnostics = dict(self_returns_discarded=self_returns,
                total_returns=total_returns, static_input=args.lidar_static_input,
                self_filter='known_robot_collision_prim_prefix; first-hit occlusion retained')
            if audit_hits is not None:
                save_json(output/'initial_raycast_audit.json', dict(robot_scope=robot_scope,
                    returns=list(audit_hits.values()), role='offline sensor identity audit; not supplied to planner'))
            if lidar_worker is not None:
                lidar_worker.submit(state,bp[2],stamp,returns,origins,
                    dict(subject=acquired_subject,diagnostics=diagnostics))
                return None
            subject_track = acquired_subject
            snapshot = lidar.integrate(state,bp[2],stamp,returns,origins,acquisition_duration_s=0.)
            snapshot['sensor_diagnostics'] = diagnostics
            return snapshot

        # Acquire actual timestamped pre-roll observations. Giving a frozen
        # obstacle nine different timestamps would erase its initial velocity.
        for k in range(round(.8*args.physics_hz)+1):
            stamp = -.8+k/args.physics_hz
            move_obstacles(stamp)
            apply()
            sim.step(render=False)
            audit_movers(stamp)
            if k % round(args.physics_hz/10) == 0:
                state, bp, br, measured, wheels, lateral, cp, cr = observe()
                observed = scan(state, bp, float(stamp))
        state, bp, br, measured, wheels, lateral, cp, cr = observe()
        if not len(observed['surfel_centers']):
            raise RuntimeError('PhysX LiDAR startup produced no observed surface map')
        if args.validate_fast_paths:
            assert fast_path_validation['ray_samples'] == 2*len(RAYS)
            fast_path_validation['passed'] = True
        save_json(output/'initial_observed_scene.json', observed)
        if args.lidar_processing == 'async':
            from jev_agent_wbc.async_lidar import AsyncLidar
            # All rays complete against one unchanged physical scene. This is
            # an instantaneous acquisition, not a 1.8 s rolling scanner packet.
            # Float32 ray hits retain physical self-filter/range uncertainty.
            lidar_worker = AsyncLidar(lidar,acquisition_duration_s=0.)
        session_options={}
        if args.cinegym_bridge is not None:
            from CineGym.bridge import controller_process as bridge_process
            session_options=dict(worker_target=bridge_process,stream_on_fault=True,
                worker_options=dict(bridge_descriptor=str(args.cinegym_bridge)))
        elif use_session:
            from jev_agent_wbc.physical_session import controller_process as session_process
            session_options=dict(worker_target=session_process,stream_on_fault=True,
                startup_wait_s=args.startup_planning_timeout,worker_options=dict(
                    session_case=str(args.scenarios/name),session_combo=session_combo,session_command_mode=args.session_command_mode,
                    startup_preplan=args.startup_preplan,execution_aware=args.execution_aware,
                    execution_calibration=None if args.execution_calibration is None else str(args.execution_calibration.resolve()),
                    fault_events=fault_schedule.events,
                    delay_aware_servo=args.delay_aware_servo,profile_controller=args.profile_controller,
                    initial_snapshot=dict(state=state.copy(),base_z=float(bp[2]),base_rotation=br.copy(),
                        measured_control=measured.copy(),scene=observed)))
        engine = RealtimeEngine(reference, observed, output, **session_options, mode=mode, api_url=args.api_url, model=args.model, limits=limits,
                                 physical_limits=physical_limits,
                                 runtime_policy=RuntimePolicy(args.command_timeout,args.sensor_timeout,
                                     args.planning_timeout,args.startup_command_timeout,
                                     recover_decisions=use_session or args.cinegym_bridge is not None,recovery_timeout_s=args.recovery_timeout),
                                 subject_model=subject_data,
                                 preview=args.allocation_preview and not args.no_preview,
                                 coordination_async=args.async_coordination,
                                 budget=SearchBudget(max_proposals=args.candidate_proposals, decision_period_s=args.agent_period,
                                     activation_lead_s=args.activation_lead, max_rollouts=args.candidate_budget,
                                     max_horizon_s=args.prediction_horizon, prediction_step_s=args.prediction_step,
                                     candidate_wall_budget_s=args.candidate_wall_budget,
                                     single_candidate=args.single_candidate),
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
            if camera_output.source == 'execution':
                sample = samples[-1]
                camera_output.accept_raw(sample['t'], sample['ap'], sample['ar'])
                return
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
            if camera_output.source == 'execution':
                return dict(t=float(stamp), ap=position, ar=rotation)
            rp, rr, _ = engine.active.sample(stamp)
            return dict(t=float(stamp), rp=rp.tolist(), rr=rr.tolist(),
                        ap=position.tolist(), ar=rotation.tolist())

        stabilize_batch([camera_sample(0., cp, cr)])

        contact_paths = {}

        def on_contacts(headers, data):
            for h in headers:
                key = (h.actor0, h.actor1)
                cached = contact_paths.get(key)
                if cached is None:
                    paths = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in key]
                    is_robot = any(p.startswith(robot_scope+'/') for p in paths)
                    other = any(not p.startswith(robot_scope+'/') and
                                not p.startswith('/World/groundPlane/') for p in paths)
                    cached = contact_paths[key] = (paths, is_robot, other)
                paths, is_robot, other = cached
                if is_robot and other:
                    contacts.append(dict(t=physical_t, actors=paths, points=int(h.num_contact_data), kind=str(h.type)))
                    if h.num_contact_data and engine.watchdog is not None:
                        engine.escalate('measured_external_contact', physical_t)
                    if len(contacts) <= 3:
                        print('CONTACT_DEBUG '+json.dumps(contacts[-1]), flush=True)

        subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
        if args.record or args.publish_camera_ros:
            recorder = RGBPair(stage, sim, args, output, robot_scope, subject_data)
            # Initialize render products before arming the live controller.
            # This is a measured t=0 image, never preplanning a trajectory.
            rp, rr, _ = reference.sample(0.)
            recorder.capture(0., rp, rr, cp, cr, subject_track, stabilized_pose=stabilized_pose)
        print(f'JEV_ISAAC_READY case={name} mode={mode} surfaces={len(observed["surfel_centers"])}', flush=True)
        from jev_agent_wbc.async_journal import AsyncJournal
        execution_journal = AsyncJournal(output/'execution.jsonl')
        if args.profile_loop:
            import cProfile
            loop_profile = cProfile.Profile()
            loop_profile.enable()
        # The finite live segment retains its telemetry intentionally. Cyclic
        # collection can otherwise scan the entire Kit object graph mid-tick.
        # Reference counting stays active; restore collection after physical time.
        gc.collect()
        restore_gc = gc.isenabled()
        gc.disable()
        run_start = time.monotonic()
        engine.arm(run_start)
        next_frame = 1/args.record_fps if recorder else 0.
        substeps = round(args.physics_hz*dt)
        from jev_agent_wbc.kc_svsdf import CollisionGeometry
        physical_geometry = CollisionGeometry()
        from jev_agent_wbc.self_collision import SelfCollision
        physical_self_collision=SelfCollision()
        current_collision = None
        previous_command = np.zeros(8)
        command_brake_started = None
        next_scan_s = 1./args.lidar_hz
        deferred_scans = 0
        end_time = (args.episode_time_limit if session_combo else reference.duration+POLICY.terminal_hold_s+5.) if use_session else reference.duration+POLICY.terminal_hold_s
        if args.duration_limit is not None:end_time=min(end_time,args.duration_limit)
        for k in range(int(np.ceil(end_time/dt))+1):
            tick_started = time.monotonic()
            rays_pending = False
            ray_finish_wall = 0.
            t = k*dt
            # Display-only USD copies are needed at capture/UI times; collider
            # targets and readback audits still run at every physical substep.
            mover_positions = audit_movers(t, display=not args.headless or
                bool(recorder and t+1e-8 >= next_frame))
            observation_wall = time.monotonic()
            state, bp, br, measured, wheels, lateral, cp, cr = observe()
            # Match runner.py: a velocity command starts from the measured
            # joint position. Reset the DRIVE reference, never physical state.
            q_command=state[3:].copy()
            if lidar_worker is not None and not lidar_failed:
                try:
                    packet = lidar_worker.poll(t)
                    if packet is not None:
                        observed = packet['snapshot']
                        observed['sensor_diagnostics'] = packet['extra']['diagnostics']
                        subject_track = packet['extra']['subject']
                except RuntimeError as error:
                    engine.trip(str(error),t)
                    lidar_failed = True
            scan_due = t+1e-9 >= next_scan_s
            for event in fault_schedule.due('lidar_gap',t):
                engine._write('diagnostic_fault_injected',t,fault=event,physics_continues=True)
            scan_due = scan_due and not fault_schedule.active('lidar_gap',t)
            mapping_busy = lidar_worker is not None and lidar_worker.pending
            if scan_due and mapping_busy:
                deferred_scans += 1
            if scan_due and not mapping_busy and not lidar_failed:
                next_scan_s = max(next_scan_s+1./args.lidar_hz,t)
                # Pre-roll already acquired t=0 after the last physical step.
                # Startup rendering does not advance time. Reuse that exact
                # measurement instead of blocking first dispatch on a duplicate.
                if k != 0 or abs(float(observed['observation_stamp'])-t) > 1e-10:
                    try:
                        if args.lidar_pipeline and native_ray is not None and lidar_worker is not None:
                            native_ray.begin(lidar.origins(state,bp[2]))
                            rays_pending=True
                            acquired=None
                        else:
                            acquired = scan(state,bp,t)
                        if acquired is not None:observed=acquired
                    except RuntimeError as error:
                        if lidar_worker is None:raise
                        engine.trip(str(error),t)
                        lidar_failed=True
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
            sensor_done = time.monotonic()
            try:
                command, info = engine.step(state, observed, t, dt, base_z=bp[2], base_rotation=br, commanded_q=q_command, measured_control=measured,
                                           observation_wall=observation_wall,
                                           applied_command=previous_command,physical_transition=physical_transition,
                                           measured_base_twist=np.r_[robot.get_linear_velocity(),robot.get_angular_velocity()])
            except Exception:
                from dataclasses import asdict
                save_json(output/'failure_snapshot.json', dict(t=t, dt=dt, state=state,
                    base_z=bp[2], base_rotation=br, commanded_q=q_command,
                    previous_command=previous_command, measured_control=measured,
                    measured_base_twist=np.r_[robot.get_linear_velocity(),robot.get_angular_velocity()],
                    limits=asdict(limits), scene=observed))
                raise
            info['agent_camera_input']={key:engine.actual_feedback[key] for key in
                ('camera_output_source','stamp_s','camera_position','camera_quaternion_xyzw')}
            info['lidar_deferred_ticks'] = deferred_scans
            # A worker certificate refers to an older measurement. Recheck the
            # issued command on CURRENT geometry before it reaches the drives.
            if current_collision is None or current_collision.scene is not observed:
                current_collision = DynamicCollision(observed, t, geometry=physical_geometry)
            else:
                current_collision.set_time(t)
            proposals=[command]
            validation_error=None
            ramp_lower=None
            brake_index=None
            recovery_index=None
            for proposal_index,proposal in enumerate(proposals):
                command=proposal
                proof = current_collision.sweep(state, command, dt, bp[2], max_samples=33, base_rotation=br)
                lower = float(proof['lower'].min())
                try:
                    if not np.isfinite(lower) or lower < current_collision.margin:
                        raise RuntimeError('current_state_sweep_failed')
                    ramp_lower = certify_control_ramp(observed,state,measured,command,t,dt,bp[2],
                        steps=substeps,constant_certificate=lower,base_rotation=br,collision=current_collision)
                    if ((use_session and args.session_command_mode!='velocity') or args.cinegym_bridge is not None
                            or proposal_index==recovery_index or engine.watchdog.certificate_waiting is not None):
                        self_lower=physical_self_collision.sweep(state,measured,command,dt)
                        info['current_self_sweep_lower_m']=self_lower
                        if self_lower<physical_self_collision.margin:
                            raise RuntimeError('current_self_sweep_failed')
                except RuntimeError as error:
                    validation_error=error
                    if (use_session and proposal_index==len(proposals)-1 and brake_index is None
                            and engine.watchdog.fault is None and engine.watchdog.packet is not None):
                        # Try an attainable deceleration endpoint, never an
                        # instantaneous stop. It must satisfy the same current
                        # actuator bounds and BOTH nonlinear sweep gates.
                        braking=MotorTargets(physical_limits)
                        braking.step(np.zeros(8),dt,measured,state[3:],wheels)
                        bounds=engine.transport_bounds
                        bounds.previous=measured.copy()
                        A,lo,hi=bounds.bounds(state[3:],dt,commanded_q=state[3:])
                        if max(np.max(lo-A@braking.control),np.max(A@braking.control-hi))<=1e-6:
                            brake_index=len(proposals)
                            proposals.append(braking.control.copy())
                            info['command_validation_retry_reason']=str(error)
                    if (args.guard_recovery and proposal_index==len(proposals)-1 and recovery_index is None
                            and engine.watchdog.fault is None and engine.watchdog.waiting is None
                            and engine.watchdog.sensor_waiting is None and engine.watchdog.packet is not None):
                        from jev_agent_wbc.physical_guard import recovery_proposal
                        try:
                            alternative,guard_info=recovery_proposal(current_collision,state,measured,
                                physical_limits,dt,bp[2],br)
                        except RuntimeError as recovery_error:
                            info['guard_recovery_rejected']=str(recovery_error)
                        else:
                            recovery_index=len(proposals);proposals.append(alternative)
                            info['guard_recovery_proposal']=guard_info
                    continue
                validation_error=None
                waiting=engine.watchdog.certificate_waiting
                fresh_tracks=[track['id'] for track in observed.get('tracks',[]) if
                    waiting is not None and track['age_s']<=engine.policy.sensor_timeout_s and
                    float(observed['observation_stamp'])-track['age_s']>waiting['sensor_stamp']+1e-9]
                if engine.watchdog.certificate_passed(t,time.monotonic(),float(observed['observation_stamp']),fresh_tracks):
                    engine._write('physical_certificate_recovered',t,source_sequence=engine.watchdog.sequence,
                        fresh_observation_required=True,all_current_sweep_gates_passed=True)
                if proposal_index and engine.watchdog.packet is not None:
                    engine.watchdog.packet['command']=command.copy()
                if proposal_index==brake_index:
                    command_brake_started=t if command_brake_started is None else command_brake_started
                    engine.watchdog.packet.update(control_acceleration=None,guard_brake=True)
                    info.update(command_override_source='measured_safety_brake',command_override_time_s=t,
                        brake_interval_certified_s=dt)
                    engine._write('command_validation_retry',t,reason=info['command_validation_retry_reason'],
                        source_sequence=engine.watchdog.sequence,original_command_lease_preserved=True)
                elif proposal_index==recovery_index:
                    command_brake_started=t if command_brake_started is None else command_brake_started
                    engine.watchdog.packet.update(control_acceleration=None,guard_brake=True)
                    info.update(command_override_source='measured_safety_recovery',command_override_time_s=t,
                        recovery_interval_certified_s=dt)
                    engine._write('command_validation_recovery',t,source_sequence=engine.watchdog.sequence,
                        original_command_lease_preserved=True,camera_task_preserved=False)
                elif engine.watchdog.packet is not None and not engine.watchdog.packet.get('guard_brake'):
                    command_brake_started=None
                if command_brake_started is not None and t-command_brake_started>engine.policy.recovery_timeout_s:
                    validation_error=RuntimeError('command_validation_recovery_exhausted')
                break
            if validation_error is not None:
                if validation_failure_snapshot is None:
                    validation_failure_snapshot=dict(t=t,dt=dt,
                        state=state.copy(),base_z=bp[2],base_rotation=br.copy(),measured_control=measured.copy(),
                        measured_wheels=wheels.copy(),proposals=proposals,scene=observed,
                        reason=str(validation_error),lower_m=lower,ramp_lower_m=ramp_lower,
                        command_source_time_s=None if engine.watchdog.packet is None else engine.watchdog.packet['t'])
                if args.certificate_recovery and str(validation_error) in (
                        'current_state_sweep_failed','current_self_sweep_failed',
                        'interpolated motor command fails dynamic clearance certificate'):
                    was_waiting=engine.watchdog.certificate_waiting is not None
                    obstacle_indices=proof.get('obstacle_indices',np.empty(0,dtype=int))
                    bad=np.min(proof['lower'],axis=1)<current_collision.margin if len(obstacle_indices) else []
                    required=current_collision.track_ids[obstacle_indices][bad] if len(obstacle_indices) else []
                    engine.watchdog.wait_for_certificate(str(validation_error),t,time.monotonic(),
                        float(observed['observation_stamp']),required)
                    if not was_waiting:engine._write('physical_certificate_waiting',t,
                        reason=str(validation_error),braking_certified=False)
                else:engine.escalate(str(validation_error),t)
                command=np.zeros(8)
            info['command_sweep_lower_m'] = lower
            info['motor_ramp_sweep_lower_m'] = ramp_lower
            info['runtime_fault'] = engine.watchdog.fault
            info['waiting_for_certificate'] = engine.watchdog.certificate_waiting
            if engine.watchdog.certificate_waiting is not None:command=np.zeros(8)
            info['safety_brake'] = engine.watchdog.fault is not None or engine.watchdog.certificate_waiting is not None
            info['brake_collision_certified'] = False if info['safety_brake'] else None
            # This instantaneous endpoint sweep does not certify the actual
            # acceleration-limited braking path; retain measured physical audit.
            truth = DynamicCollision(runtime_scene.truth_collision(t), t, uncertainty_rate=0.,geometry=physical_geometry)
            clearance = float(truth.clearance(state, bp[2], br).min())
            if args.cinegym_bridge is not None:
                prior=getattr(engine,'external_evaluation',None) or {}
                engine.external_evaluation=dict(stamp_s=t,truth_clearance_m=clearance,
                    min_clearance_m=min(clearance,prior.get('min_clearance_m',float('inf'))),
                    collision_samples=prior.get('collision_samples',0)+int(clearance<0.),
                    physical_contacts=sum(c['points']>0 for c in contacts))
            info['emergency_stop_required'] = engine.watchdog.emergency_stop_required
            fk, fr, _ = whole_body_kinematics(state, bp[2], br)
            info['fk_position_error_m'] = float(np.linalg.norm(fk-cp))
            measured_scene = camera_scene(dict(observed, subject_model=subject_data,
                                               intrinsics=engine.intrinsics), t, reference.duration)
            rows.append(dict(t=t, state=state.tolist(), command=command.tolist(), measured=measured.tolist(),
                actuation_transition=physical_transition,
                base_z=float(bp[2]), base_rotation=br.tolist(),
                commanded_q=q_command.tolist(),
                measured_wheels=wheels.tolist(), measured_lateral_velocity=float(lateral),
                **{key:value for key,value in camera_output.latest.items() if key != 't'},
                subject_track=subject_track, truth_clearance_m=clearance, info=info,
                progress_s=progress_at(engine.active, t),
                visual=visibility(output_p, output_r, measured_scene, t, subject_data, engine.intrinsics),
                measured_obstacles=dict(zip(mover_names, mover_positions.tolist()))))
            rows[-1]['runtime_tracks'] = observed.get('tracks', [])
            rows[-1]['sensor_diagnostics'] = observed.get('sensor_diagnostics', {})
            # Until this barrier, all main-thread PhysX access is read-only.
            # The CPU geometry/Agent transport above consumes the last completed
            # map, exactly as in non-pipelined asynchronous mapping. Never update
            # collider targets, drive tensors, simulation or RGB while rays run.
            if rays_pending:
                ray_finish_started=time.monotonic()
                try:
                    scan(state,bp,t,batch=native_ray.finish())
                except Exception as error:
                    engine.trip('lidar_query_failed: '+str(error),t)
                    lidar_failed=True
                    rows[-1]['info'].update(runtime_fault=engine.watchdog.fault,safety_brake=True)
                ray_finish_wall=time.monotonic()-ray_finish_started
            execution_journal.write(json.dumps(native(rows[-1]), allow_nan=False)+'\n')
            if args.save_replay:
                replay_poses.append(robot._articulation_view._physics_view.get_link_transforms()[0].copy())
            control_done = time.monotonic()
            if recorder and t+1e-8 >= next_frame:
                rp, rr, _ = reference.sample(t)
                recorder.capture(t, rp, rr, cp, cr, subject_track, stabilized_pose=stabilized_pose)
                next_frame += 1/args.record_fps
            render_done = time.monotonic()
            if k % 50 == 0:
                print(json.dumps(dict(case=name, mode=mode, t=t, clearance=clearance, contacts=len(contacts),
                    jev_calls=engine.jev.decider.client.calls, installed=engine.installed_plans)), flush=True)
            if use_session and info.get('session_done'):
                result=info['session_result']
                end_time=(t if result.get('completed') else
                    max(t,float(source_task['reference_camera_trajectory']['samples'][-1]['t'])+POLICY.terminal_hold_s))
            if t >= end_time:
                run_end = time.monotonic()
                break
            requested_command = command.copy()
            physics_camera_batch = []
            motor_feedback=measured.copy()
            motor_wheel_feedback=wheels.copy()
            measured_state=(state,bp,br,measured,wheels,lateral,cp,cr)
            for j in range(substeps):
                physical_t = t+(j+1)/args.physics_hz
                # Enforce command/sensor leases at the drive rate too, including
                # wall time consumed by RGB capture or a stalled main callback.
                requested_command = engine.watchdog.command(physical_t,time.monotonic(),
                    float(observed['observation_stamp']),alive=engine.process.is_alive())
                if not use_session and physical_t >= reference.duration:
                    requested_command = np.zeros(8)
                # Match the CPU command contract: constant endpoint velocity
                # throughout the control interval when command_ramp is false.
                command = (previous_command+(j+.5)/substeps*(requested_command-previous_command)
                           if limits.command_ramp else requested_command)
                motor_positions=measured_state[0][3:]
                drive_velocity = (motors.step(command,1/args.physics_hz,motor_feedback,motor_positions,motor_wheel_feedback) if args.motor_targets=='physical_slew'
                                  else np.r_[limits.wheels@command,command[2:]])
                q_command = motor_positions.copy()+drive_velocity[2:]/args.physics_hz
                apply(measured_state)
                move_obstacles(physical_t)
                sim.step(render=False)
                audit_movers(physical_t)
                measured_state = observe()
                physical_transition=dict(stamp_s=physical_t,dt_s=1/args.physics_hz,
                    previous_velocity=motor_feedback.copy(),drive_target=motors.control.copy(),
                    measured_velocity=measured_state[3].copy())
                motor_feedback=measured_state[3].copy()
                motor_wheel_feedback=measured_state[4].copy()
                physics_camera_batch.append(camera_sample(physical_t, measured_state[6], measured_state[7]))
                audit.record(physical_t, measured_state[0][3:], measured_state[3], measured_state[4], measured_state[5])
            stabilize_batch(physics_camera_batch)
            physics_done = time.monotonic()
            previous_command = requested_command
            if not args.headless and recorder is None and k % 10 == 0:
                sim.render()
            tick_times.append(time.monotonic()-tick_started)
            tick_stages.append(dict(t=t, sensor_s=sensor_done-tick_started,
                supervision_and_audit_s=control_done-sensor_done-ray_finish_wall,
                lidar_finish_s=ray_finish_wall,rgb_s=render_done-control_done,
                physics_and_stabilizer_s=physics_done-render_done,
                display_s=time.monotonic()-physics_done))
            tick_lags.append(max(0.,time.monotonic()-run_start-(k+1)*dt))
            if args.pace > 0:
                slack = (k+1)*dt/args.pace-(time.monotonic()-run_start)
                if slack > 0:
                    time.sleep(slack)
        del subscription
    except Exception as error:
        if args.validate_fast_paths and isinstance(error, AssertionError):
            fast_path_validation['passed'] = False
        failure = f'{type(error).__name__}: {error}'
        if engine:
            engine._write('run_failed', len(rows)*dt, failure=failure,
                          diagnostics=getattr(error, 'diagnostics', None))
        import traceback
        traceback.print_exc()
    finally:
        if validation_failure_snapshot is not None:
            save_json(output/'validation_failure_snapshot.json',validation_failure_snapshot)
        if loop_profile is not None:
            loop_profile.disable()
            loop_profile.dump_stats(str(output/'loop_profile.pstats'))
        if run_start is not None and run_end is None:
            run_end = time.monotonic()
        if restore_gc:
            gc.enable()
            restore_gc = False
        if execution_journal is not None:
            execution_journal.close()
        if replay_poses:
            replay_dir=output/'replay'
            replay_dir.mkdir(exist_ok=False)
            np.savez_compressed(replay_dir/'measured_links.npz',
                transforms=np.asarray(replay_poses),times=np.asarray([r['t'] for r in rows]),
                paths=np.asarray(robot._articulation_view._physics_view.link_paths[0],dtype=str))
            stage.Flatten().Export(str(replay_dir/'scene.usdc'))
            save_json(replay_dir/'manifest.json',dict(protocol='measured_link_poses/v1',
                measured=True,role='offline presentation only; not a second physical execution',
                samples=len(replay_poses),rgb_work_during_live_loop=False,
                source_clock='telemetry.t',positions='PhysX articulation link transforms',
                recording_does_not_change_agent_or_physics=True))
        if engine and engine.watchdog and engine.watchdog.fault and not failure:
            failure = 'runtime_fail_safe: '+engine.watchdog.fault['reason']
        if lidar_worker is not None:
            lidar_worker.close()
            save_json(output/'lidar_timing.json',lidar_worker.samples)
        if native_ray is not None:
            native_ray.close()
            save_json(output/'native_lidar_timing.json',native_ray.timings)
        if audit:
            physical_summary = audit.finish()
        if recorder:
            try:
                recorder.close()
            except Exception as error:
                recording_error = f'{type(error).__name__}: {error}'
                print(f'RECORDER_CLOSE_ERROR {error}', flush=True)
        summary = trajectory_summary(rows, reference, {} if engine is None else engine.summary(), failure, physical_summary)
        if camera_output.source == 'execution' and summary.get('physical_camera_tracking'):
            summary['physical_camera_tracking']['role'] = 'primary Agent and visual evaluation input'
        contact_count = sum(c['points'] > 0 for c in contacts)
        summary.update(backend='isaac_physx', physics_hz=args.physics_hz, wall_s=time.monotonic()-begin,
            compiled_geometry=bool(args.compiled_geometry),certificate_recovery=bool(args.certificate_recovery),
            execution_aware=bool(args.execution_aware),
            pace=args.pace, physics_waits_for_agent=False,
            fabric_enabled=args.fabric,
            lidar_backend=args.lidar_backend,lidar_processing=args.lidar_processing,
            lidar_query_threads=args.lidar_query_threads,
            lidar_requested_hz=args.lidar_hz,
            lidar_pipeline=args.lidar_pipeline,
            native_rays_validated=0 if native_ray is None else native_ray.validated_rays,
            physics_threads=args.physics_threads,
            render_subframes=args.render_subframes,
            fast_path_validation=fast_path_validation if args.validate_fast_paths else None,
            mode=mode, case=name, observation_source='dual_physx_sparse_lidar_raycast', contacts=contact_count,
            contact_notifications=len(contacts),
            coordination_async=args.async_coordination,
            operating_acceleration_fraction=args.acceleration_reserve,
            operating_yaw_acceleration_fraction=(args.yaw_acceleration_reserve
                if args.yaw_acceleration_reserve is not None else args.acceleration_reserve),
            camera_output_source=camera_output.source,
            camera_output_role='primary Agent and visual evaluation input',
            raw_camera_role=('primary Agent input' if camera_output.source == 'execution' else 'intermediate physical diagnostic'),
            sensor_limitation='PhysX first-hit raycasting; no RTX material/noise/motion-distortion simulation',
            lidar_static_input=args.lidar_static_input, motor_targets=args.motor_targets,
            diagnostic_duration_limit_s=args.duration_limit,
            base_support=args.base_support,
            measured_physical_constraints=physical_summary is not None, recording_error=recording_error)
        if contact_count:
            summary['safety_pass'] = summary['within_budget_success'] = False
        summary['obstacle_pose_audit'] = mover_audit
        from dataclasses import asdict
        summary['control_limits'] = native(asdict(limits))
        summary['gravity_feedforward_max_abs_nm']=gravity_max.tolist() if sim else None
        summary['dynamics_feedforward_max_abs_nm']=feedforward_max.tolist() if sim else None
        summary['cpu_parameter_parity'] = all(
            np.array_equal(np.asarray(value), np.asarray(asdict(Limits())[key]))
            for key,value in asdict(limits).items())
        if engine:
            from dataclasses import replace
            runtime_limits=asdict(engine.wbc.limits)
            expected=asdict(replace(Limits(),deadline_s=reference.duration))
            summary['runtime_control_limits']=native(runtime_limits)
            summary['cpu_runtime_parameter_parity']=all(
                np.array_equal(np.asarray(value),np.asarray(expected[key])) for key,value in runtime_limits.items())
        import osqp
        summary['numeric_runtime']=dict(osqp_version=osqp.__version__,osqp_file=osqp.__file__,numeric_site=NUMERIC_SITE)
        if rows and engine:
            measured_intent = IntentSpec(reference, subject_data, engine.intrinsics).score(
                np.array([r['t'] for r in rows]), [r['camera_position'] for r in rows],
                [Rotation.from_quat(r['camera_quaternion_xyzw']).as_matrix() for r in rows],
                [r['visual'] for r in rows], [r['progress_s'] for r in rows])
            summary.update(executed_camera_intent=measured_intent,
                terminal_confirmation_observed_s=max(0., rows[-1]['t']-reference.duration),
                intent_success=bool(not failure and summary['safety_pass'] and measured_intent['intent_complete']),
                intent_measurement='PhysX camera poses; observed-LiDAR geometric visibility, not SAM')
            from jev_agent_wbc.shot_acceptance import evaluate_shot, attach_acceptance
            acceptance=evaluate_shot(reference,subject_data,
                [r['t'] for r in rows],[r['camera_position'] for r in rows],
                [Rotation.from_quat(r['camera_quaternion_xyzw']).as_matrix() for r in rows],
                [r['visual'] for r in rows],[r['progress_s'] for r in rows],
                safety_pass=summary['safety_pass'],execution_ok=not failure,
                intrinsics=engine.intrinsics,motion_end_s=engine.active.duration,
                legacy_intent=measured_intent,safety_source=f'PhysX contacts and raw {args.physics_hz:g}Hz actuator audit')
            attach_acceptance(summary,acceptance)
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
        if rows:
            # User-requested PhysX reporting exclusions are applied only after
            # execution and worker shutdown. Raw audits and control limits stay
            # untouched; CPU and shared Agent acceptance remain unchanged.
            from physx_acceptance import assess_physx
            api_log=output/'jev_api.jsonl'
            api_records=([json.loads(line) for line in api_log.read_text().splitlines() if line]
                         if api_log.exists() else [])
            summary=assess_physx(summary,rows,api_records)
            timing=dict(pace=args.pace, executed_sim_s=rows[-1]['t'],
                execution_wall_s=run_end-run_start,
                sim_wall_ratio=rows[-1]['t']/max(run_end-run_start,1e-9),
                max_tick_wall_s=max(tick_times,default=0.),
                p99_tick_wall_s=float(np.percentile(tick_times,99)) if tick_times else 0.,
                ticks_over_20ms=sum(x>.02 for x in tick_times),
                max_lag_s=max(tick_lags,default=0.),
                scope='live physics, sensors, watchdog, audit and RGB; startup/shutdown excluded')
            if use_session:
                from jev_agent_wbc.physical_session_audit import assess_session
                summary=assess_session(summary,rows,source_task,subject_data,engine.intrinsics,output)
            summary['realtime']=realtime_assessment(timing,
                None if engine.watchdog is None else engine.watchdog.fault, summary['shot_acceptance']['accepted'])
            save_json(output/'realtime_timing.json',dict(tick_wall_s=tick_times,lag_s=tick_lags,
                stages=tick_stages,**summary['realtime']))
            save_json(output/'summary.json',summary)
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
        self.reference_frames_written = 0
        self.cameras, self.products, self.annotators, self.writers = {}, {}, {}, {}
        self.camera_prims = {}
        keys = ['execution', 'overview']
        if self.recording and not getattr(args, 'skip_reference_video', False):
            keys.append('reference')
        if self.camera_output.source == 'stabilized':
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
            comparison=('comparison.mp4' if self.recording and source == 'stabilized'
                        and not getattr(args, 'skip_reference_video', False) else None),
            views=[view for view in ('reference','execution','stabilized') if view in self.cameras],
            delivery_views=['overview','reference',source],
            reference_source='original_reference_floating_camera_same_scene_time',
            raw_role='primary Agent input' if source == 'execution' else 'intermediate physical diagnostic', fps=args.record_fps))+'\n')
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
        if stabilized_pose is None and self.camera_output.source == 'stabilized':
            self.camera_output.update(t, rp, rr, ap, ar)
            value = self.camera_output.latest
            stabilized_pose = (np.array(value['stabilized_camera_position']), Rotation.from_quat(value['stabilized_camera_quaternion_xyzw']).as_matrix())
        capture_clock_before = float(self.sim.current_time)
        self.pose('execution', ap, ar)
        if 'reference' in self.cameras:
            # A render-only camera is unconstrained by the robot. Its pose is
            # the original reference, never the installed repair or physical FK.
            self.pose('reference', rp, rr)
        if stabilized_pose is not None:
            self.pose('stabilized', *stabilized_pose)
            from pxr import Gf
            # Preserve the physical camera's near-plane clearance when the
            # virtual viewpoint retreats into its visual housing. Intrinsics
            # and far clip remain identical; foreground closer than this
            # explicitly logged plane is excluded from the virtual image.
            virtual_near_m = .05+float(np.linalg.norm(stabilized_pose[0]-ap))+.01
            self.camera_prims['stabilized'].GetClippingRangeAttr().Set(Gf.Vec2f(virtual_near_m, 50.))
        # All cameras observe the same current actors and simulation timestamp.
        # Never toggle actor visibility or render reference images at a later
        # scene time: that would corrupt the dynamic reference comparison.
        raw_rgb = self.capture_rgb('execution', len(self.frame_metadata))
        self.capture_rgb('overview', len(self.frame_metadata), advance_render=False)
        if 'reference' in self.cameras:
            self.capture_rgb('reference', len(self.frame_metadata), advance_render=False)
            self.reference_frames_written += 1
        if stabilized_pose is not None:
            stabilized_rgb = self.capture_rgb('stabilized', len(self.frame_metadata), advance_render=False)
        output_p, output_r = stabilized_pose if self.camera_output.source == 'stabilized' else (ap, ar)
        self.output_rgb = (stabilized_rgb if self.camera_output.source == 'stabilized' else raw_rgb)[:, :, ::-1].copy()
        if abs(float(self.sim.current_time)-capture_clock_before) > 1e-8:
            raise RuntimeError('RGB capture advanced the physical simulation')
        record = dict(frame=len(self.frame_metadata), t=t, physical_time_before_capture=capture_clock_before, physical_time_after_capture=float(self.sim.current_time),
            capture_protocol='same_render_pass_same_simulation_timestamp',
            reference_camera_position=rp, actual_camera_position=ap,
            reference_camera_source='original_reference_floating_camera',
            reference_scene_time_s=float(t),
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
            seeds = [('reference', rp, rr), ('execution', ap, ar)]
            if stabilized_pose is not None:
                seeds.append(('stabilized', *stabilized_pose))
            for key, p, r in seeds:
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
            self.rep.orchestrator.step(rt_subframes=self.args.render_subframes, delta_time=0.0, pause_timeline=False)
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
        try:
            skipped=not self.recording or bool(getattr(self.args,'skip_reference_video',False))
            if not skipped and self.reference_frames_written!=len(self.frame_metadata):
                raise RuntimeError('live original-reference RGB frame count mismatch')
        finally:
            for writer in self.writers.values():
                writer.release()
            save_json(self.root/'frames.json', self.frame_metadata)
            save_json(self.root/'reference_rendering.json', dict(
                protocol='original_reference_floating_camera_same_scene_time/v1',
                skipped=not self.recording or bool(getattr(self.args, 'skip_reference_video', False)),
                frames_expected=0 if not self.recording or getattr(self.args, 'skip_reference_video', False) else len(self.frame_metadata),
                frames_written=self.reference_frames_written,
                valid=None if not self.recording or getattr(self.args, 'skip_reference_video', False) else self.reference_frames_written == len(self.frame_metadata),
                camera_trajectory='original_reference',physics_body=False,
                same_render_pass=True,actor_visibility_modified=False,
                scope='detached original-reference camera in current physical scene; comparison only, not Agent input'))
            try:
                if self.recording and not getattr(self.args, 'skip_reference_video', False) and 'stabilized' in self.cameras:
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
    # SimulationApp overwrites OPENBLAS_NUM_THREADS with its Kit thread limit.
    # Spawned servo/map/planning processes must retain the CPU benchmark's
    # single-threaded small-matrix numerics. PhysX and native ray thread counts
    # remain independently configured above.
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[variable]='1'
    exit_code = 2
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
        names = (list(NAMES)+sorted(p.name for p in args.scenarios.glob('combo_*') if p.is_dir())) if args.case == 'all' else args.case.split(',')
        for name in names:
            if (Path(name).name != name or name in ('.', '..') or
                    not all((args.scenarios/name/file).is_file() for file in ('task.json', 'runtime_scene.json'))):
                raise ValueError(f'missing or invalid authored case: {name}')
        results = []
        for name in names:
            for mode in (['strict', 'jev'] if args.mode == 'both' else [args.mode]):
                results.append(run_case(args, app, name, mode, args.output/f'{name}_{mode}'))
                save_json(args.output/'runs.json', results)
        exit_code = 0 if all(r.get('shot_acceptance',{}).get('accepted',False) for r in results) else 2
        return exit_code
    except Exception as error:
        import traceback
        traceback.print_exc()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'runner_failure.json').write_text(json.dumps(dict(error=str(error), traceback=traceback.format_exc()))+'\n')
        return 2
    finally:
        # Fast Kit shutdown can terminate natively before Python returns. Give
        # Kit the actual result before releasing the framework as well.
        omni.kit.app.get_app().post_quit(exit_code)
        try:
            app.close()
        except SystemExit:
            # Kit's normal shutdown may call sys.exit(0); retain the actual
            # experiment result instead of turning failed episodes into exit 0.
            pass


if __name__ == '__main__':
    raise SystemExit(main())
