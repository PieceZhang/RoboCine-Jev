#!/usr/bin/env python3
"""TIAGo physical acceptance using the isolated genuine Giskard framework.

Adapted from the historical scene/measurement runner; no historical SLSQP
controller is imported or executed. World/Goals/Statechart/Executor is the default;
the explicitly named core-only component remains a separate diagnostic option.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time

ROBOT = Path(__file__).resolve().parent
PACKAGE = ROBOT.parent
sys.path[:0] = [str(ROBOT),str(ROBOT/"wbc")]
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'scripts')]


def independent_joint_fixture(joints):
    amplitude=[min(.12,.4*min(j['home']-j['lower'],j['upper']-j['home'])) for j in joints]
    authored=[]
    for i in range(801):
        t=i*.01;angle=2*math.pi*t/8.
        authored.append(dict(t_s=t,position=[j['home']+v*math.sin(angle) for j,v in zip(joints,amplitude)],
            velocity=[v*(2*math.pi/8.)*math.cos(angle) for v in amplitude]))
    return dict(schema='tiago_independent_joint_fixture/v1',joint_names=[j['name'] for j in joints],
        duration_s=8.,amplitude=amplitude,authoring='q=original_home+A*sin(2*pi*t/8); full return then measured closed-loop home hold',samples=authored)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--wbc-only', action='store_true', help='Short WBC/stop diagnostic; skips direction and independent-arm acceptance.')
    p.add_argument('--direction-probe',choices=['forward','backward','left','right','diagonal_ne','diagonal_sw'],
        help='One original direction plus its fixed5s stop; diagnostic only, no camera acceptance.')
    p.add_argument('--arm-only',action='store_true',help='Original8s independent arm fixture, fixed5s measured home return and fixed5s zero-command strict stop; diagnostic only.')
    p.add_argument('--initial-heading-rad', type=float, default=0., help='Initial placement only, before physics; useful for WBC diagnostics.')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--sensor-audit', action='store_true')
    p.add_argument('--sensor-rgb', action='store_true')
    p.add_argument('--tgs-force-every-iteration',type=int,choices=[0,1],default=0,
        help='Explicit numerical TGS profile; default preserves the original force integration.')
    p.add_argument('--tgs-velocity-iterations',type=int,choices=[0,1,16],default=16,
        help='Explicit numerical TGS velocity iterations; default preserves the original16.')
    p.add_argument('--diagnostic-forces',action='store_true',help='Record actual projected joint constraint forces; includes reactions, not direct motor torque.')
    p.add_argument('--sensor-config', type=Path)
    p.add_argument('--task',type=Path,help='Immutable public TimedTask JSON; observation binding and absolute clock remain strict.')
    p.add_argument('--backend',choices=['giskard-executor','giskard-core'],default='giskard-executor')
    p.add_argument('--trajectory', type=Path, help='JSON samples: t_s, position_m, quaternion_xyzw; world optical frame.')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    a.robot_dir, a.output = a.robot_dir.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((a.robot_dir/'robot.json').read_text())
    files = [Path(__file__), ROBOT/'wbc/physical_control.py', a.robot_dir/'robot.json',
             a.sensor_config or a.robot_dir/spec['sensors'], PACKAGE/'jev_agent_wbc/robot_sensor_rig.py']
    if a.trajectory:
        files.append(a.trajectory)
    if a.task:
        files.append(a.task)
    frozen = a.output/'sources'
    frozen.mkdir()
    (frozen/'tiago_physics.urdf').write_bytes((a.robot_dir/'urdf/tiago_physics.urdf').read_bytes())
    for index, path in enumerate(files):
        (frozen/f'{index}_{path.name}').write_bytes(path.read_bytes())
    if not a.probe and not a.wbc_only:
        arm_input=independent_joint_fixture(spec['joints'])
        arm_path=frozen/'independent_arm_reference.json'
        arm_path.write_text(json.dumps(arm_input,indent=2)+'\n')
        a.arm_reference=json.loads(arm_path.read_text())
    if a.task:
        task_index=files.index(a.task)
        a.task=frozen/f'{task_index}_{a.task.name}'
    a.native_sources=frozen/'native_worker'
    a.native_sources.mkdir()
    worker_files=[ROBOT/'wbc'/name for name in ('native_solver.py','collision_model.py','worker.py','async_client.py','full_framework.py','export_collision.py','source_state.py','floor_model.py')]+[ROBOT/'wbc_backend.py']
    for path in worker_files:
        (a.native_sources/path.name).write_bytes(path.read_bytes())
    (a.native_sources/'FRAMEWORK_SOURCE_LOCK.json').write_bytes((ROBOT/'wbc/FRAMEWORK_SOURCE_LOCK.json').read_bytes())
    shared=frozen/'shared_package/jev_agent_wbc'
    shared.mkdir(parents=True)
    (shared/'__init__.py').write_text('')
    shared_files=[PACKAGE/'jev_agent_wbc'/name for name in ('representation.py','wbc_protocol.py','wbc_physical.py')]
    for path in shared_files:
        (shared/path.name).write_bytes(path.read_bytes())
    sys.path[:0]=[str(a.native_sources),str(shared.parent)]
    import importlib.util
    def frozen_module(index, module_name):
        path = frozen/f'{index}_{files[index].name}'
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module
    a.control_module = frozen_module(1, 'frozen_mobile_control')
    a.rig_module = frozen_module(4, 'frozen_robot_sensor_rig')
    a.sensor_config = frozen/f'3_{files[3].name}'
    if a.trajectory:
        a.trajectory_data = json.loads((frozen/f'5_{files[5].name}').read_text())
        a.control_module.validate_trajectory(a.trajectory_data)
    provenance = dict(created_unix=time.time(), python=sys.executable, isaacsim=version('isaacsim'),
                      sources={str(f): hashlib.sha256((frozen/f'{i}_{f.name}').read_bytes()).hexdigest() for i, f in enumerate(files)},
                      native_sources={str(f):hashlib.sha256((a.native_sources/f.name).read_bytes()).hexdigest() for f in worker_files},
                      shared_sources={str(f):hashlib.sha256((shared/f.name).read_bytes()).hexdigest() for f in shared_files},
                      frozen_sha256={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*') if p.is_file()},
                      physics_hz=100,measurement_hz=100,requested_pace=1., base_execution='wheel velocity drives; free floating physical base',
                      controller=('genuine upstream Giskard World/CartesianPose/Statechart/Executor/QPController plus explicit independent Giskard/FCL safety projection'
                          if a.backend=='giskard-executor' else 'genuine upstream Giskard explicit-QP component adapter; not full World/Executor'))
    (a.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    # Warm the real framework BEFORE Kit startup and before physical time exists.
    # The live stage must then match these exact body-local collision shapes.
    import subprocess
    from async_client import AsyncGiskardClient
    metadata_path=a.output/'collision_geometry_prepared.json'
    subprocess.run([sys.executable,str(a.native_sources/'export_collision.py'),'--output',str(metadata_path),
        '--robot-dir',str(a.robot_dir)],check=True)
    frozen_metadata=frozen/'collision_geometry_prepared.json';frozen_metadata.write_bytes(metadata_path.read_bytes())
    a.native=AsyncGiskardClient(frozen_metadata,a.output/'giskard_worker.log',robot_dir=a.robot_dir,
        module_dir=a.native_sources,spec_path=a.output/'sources/2_robot.json',backend=a.backend,mesh_cache=frozen/'native_meshes',sensor_config=a.sensor_config)
    try:
        warm_start=time.monotonic()
        a.native.prewarm()
        initialization=dict(a.native.initialization,prewarm_wall_s=time.monotonic()-warm_start,physical_time_started=False)
        (a.output/'native_initialization.json').write_text(json.dumps(initialization,indent=2)+'\n')
        (frozen/'native_initialization.json').write_text(json.dumps(initialization,indent=2)+'\n')
        provenance['frozen_sha256']={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*') if p.is_file() and p.suffix!='.pyc'}
        (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    except BaseException:
        import traceback
        (a.output/'failure.txt').write_text(traceback.format_exc())
        (a.output/'acceptance.json').write_text(json.dumps(dict(passed=False,complete=False,reason='Before-physics native framework prewarm failed.'),indent=2)+'\n')
        a.native.close()
        raise
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    os.environ.setdefault('ACCEPT_EULA', 'Y')
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=not a.gui, renderer='RaytracedLighting', anti_aliasing=0,
                             multi_gpu=False, limit_cpu_threads=8))
    try:
        return run(a, spec)
    except BaseException:
        import traceback
        (a.output/'failure.txt').write_text(traceback.format_exc())
        (a.output/'acceptance.json').write_text(json.dumps(dict(passed=False, complete=False,
            reason='See failure.txt and streaming samples.jsonl; no successful acceptance implied.'), indent=2)+'\n')
        raise
    finally:
        app.close(wait_for_replicator=False)
        a.native.close()


def run(a, spec):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    RobotSensorRig, SensorAudit = a.rig_module.RobotSensorRig, a.rig_module.SensorAudit
    pose_transform, set_usd_transform, transform = a.rig_module.pose_transform, a.rig_module.set_usd_transform, a.rig_module.transform
    from async_client import AsyncGiskardClient
    from collision_model import extract_collision_geometry,same_collision_geometry
    from jev_agent_wbc.wbc_physical import MeasuredColliderGeometry,LiveTaskSession,audit_stop,pose as public_pose
    from jev_agent_wbc.wbc_protocol import Capabilities,CameraTrajectory,TimedTask,semantic_camera
    CameraController = a.control_module.CameraController
    camera_jacobian, wheel_speeds = a.control_module.camera_jacobian, a.control_module.wheel_speeds
    shift_jacobian = a.control_module.spatial_jacobian_at_offset

    dt, root = .01, '/World/Robot'
    world = World(physics_dt=dt, rendering_dt=.02, stage_units_in_meters=1.)
    stage = world.stage
    physics_scene=world.get_physics_context().prim_path
    force_integration=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(physics_scene))
    model = a.robot_dir/spec['usd']
    if not model.is_file():
        raise FileNotFoundError(f'{model}: fetch assets / import the resolved URDF first')
    add_reference_to_stage(str(model), root)
    # Place the authentic free base above contact ground ONCE before reset.
    bodies = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bodies:
        raise RuntimeError('asset has no rigid bodies')
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.IsA(UsdPhysics.FixedJoint):
            j = UsdPhysics.FixedJoint(p)
            if not j.GetBody0Rel().GetTargets() or not j.GetBody1Rel().GetTargets():
                raise RuntimeError(f'Fixed-to-world model is incompatible with wheel validation: {p.GetPath()}')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    bottom = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():
            bounds = cache.ComputeWorldBound(p).ComputeAlignedRange()
            if bounds.IsEmpty():
                raise RuntimeError(f'collision has empty bounds: {p.GetPath()}')
            value = float(bounds.GetMin()[2])
            if not np.isfinite(value) or abs(value) > 100.:
                raise RuntimeError(f'invalid initial collision bounds: {p.GetPath()}: {value}')
            bottom.append(value)
    if not bottom:
        raise RuntimeError('no collision geometry for wheel-ground physics')
    old = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T.copy()
    if not np.isfinite(a.initial_heading_rad):
        raise ValueError('initial heading must be finite')
    old[:3, :3] = Rotation.from_euler('z', a.initial_heading_rad).as_matrix()@old[:3, :3]
    old[2, 3] += .003-min(bottom)
    set_usd_transform(stage.GetPrimAtPath(root), old)
    material = PhysicsMaterial('/World/GroundMaterial', static_friction=.8, dynamic_friction=.6, restitution=0.)
    world.scene.add(GroundPlane('/World/Ground', name='ground', size=40., physics_material=material))
    from floor_model import configure_single_ground,audit_ground
    floor_report=configure_single_ground(stage,'/World/Ground','/World/GroundMaterial')
    (a.output/'single_floor_static_gate.json').write_text(json.dumps(floor_report,indent=2)+'\n')
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(800.)
    for p in bodies:
        PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateMaxDepenetrationVelocityAttr(1.)
    articulation_suffix = spec.get('articulation_root', '').strip('/')
    robot = world.scene.add(SingleArticulation(prim_path=root+('/'+articulation_suffix if articulation_suffix else ''), name='robot'))
    rigid = {}
    rig = RobotSensorRig(stage, root, spec['robot_key'], a.sensor_config or a.robot_dir/spec['sensors'])
    if len(rig.lidars) > 2:
        raise ValueError('at most one main + one auxiliary LiDAR is permitted for these platforms')
    scan_keys = ('horizontal_samples', 'elevation_deg', 'min_range_m', 'max_range_m')
    if any(any(x['spec'][k] != rig.lidars[0]['spec'][k] for k in scan_keys) for x in rig.lidars):
        raise ValueError('multiple LiDARs on one robot must use identical scan parameters')
    # All collision owners obtain actual PhysX transforms, never live USD poses.
    collision_metadata=extract_collision_geometry(stage,root)
    metadata_path=a.output/'collision_geometry.json'
    metadata_path.write_text(json.dumps(collision_metadata,indent=2)+'\n')
    prepared=json.loads((a.output/'collision_geometry_prepared.json').read_text())
    if prepared['asset_sha256']!=hashlib.sha256((a.robot_dir/spec['usd']).read_bytes()).hexdigest():
        raise ValueError('Asset changed after CPU collision preparation')
    equivalence=same_collision_geometry(prepared,collision_metadata)
    (a.output/'collision_geometry_equivalence.json').write_text(json.dumps(equivalence,indent=2)+'\n')
    import xml.etree.ElementTree as ET
    tree=ET.parse(a.robot_dir/'urdf/tiago_physics.urdf').getroot()
    manipulator_bodies={'torso_lift_link'}
    for _ in range(len(bodies)):
        for joint in tree.findall('joint'):
            if joint.find('parent').get('link') in manipulator_bodies:
                manipulator_bodies.add(joint.find('child').get('link'))
    geometry=MeasuredColliderGeometry.from_stage(stage,root,manipulator_bodies)
    (a.output/'measured_geometry_manifest.json').write_text(json.dumps(geometry.manifest(),indent=2)+'\n')
    for name in set(geometry.body_names) | {x['body'] for x in collision_metadata['shapes']} | rig.parent_names | {spec['base_link'], spec['ee_link']}:
        matches = [p for p in bodies if p.GetName() == name]
        if len(matches) != 1:
            raise ValueError(f'expected unique rigid body {name}: {matches}')
        rigid[name] = world.scene.add(SingleRigidPrim(prim_path=str(matches[0].GetPath()), name=name))
    audit = SensorAudit(rig, a.output, a.sensor_audit)
    # A numerical TGS force-integration change, applied before physics exists.
    # Save exact authored mass/inertia, material/contact and drive-limit values
    # on both sides, so the precision option cannot conceal a physical change.
    def physical_inventory():
        selected={'physics:mass','physics:density','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes',
            'physics:staticFriction','physics:dynamicFriction','physics:restitution','physxMaterial:frictionCombineMode',
            'drive:angular:physics:maxForce','drive:linear:physics:maxForce','physics:collisionEnabled',
            'physxArticulation:solverPositionIterationCount','physxArticulation:solverVelocityIterationCount'}
        return {str(p.GetPath()):{str(attr.GetName()):str(attr.Get()) for attr in p.GetAttributes() if str(attr.GetName()) in selected}
            for p in Usd.PrimRange(stage.GetPrimAtPath('/World'),Usd.TraverseInstanceProxies())
            if any(str(attr.GetName()) in selected for attr in p.GetAttributes())}
    precision_before=physical_inventory()
    flag_before=force_integration.GetEnableExternalForcesEveryIterationAttr().Get()
    articulation_precision=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(robot.prim_path))
    original_velocity_iterations=articulation_precision.GetSolverVelocityIterationCountAttr().Get()
    articulation_precision.CreateSolverVelocityIterationCountAttr().Set(a.tgs_velocity_iterations)
    force_integration.CreateEnableExternalForcesEveryIterationAttr().Set(bool(a.tgs_force_every_iteration))
    precision_after=physical_inventory()
    def physical_only(inventory):
        return {path:{name:value for name,value in attrs.items() if name!='physxArticulation:solverVelocityIterationCount'}
            for path,attrs in inventory.items()}
    if (physical_only(precision_before)!=physical_only(precision_after)
            or force_integration.GetEnableExternalForcesEveryIterationAttr().Get()!=bool(a.tgs_force_every_iteration)):
        raise RuntimeError('Force integration precision modified physical parameters or failed to enable')
    native=a.native
    session=LiveTaskSession(Capabilities('genuine Giskard World/Executor physical backend' if a.backend=='giskard-executor'
        else 'genuine Giskard explicit-QP component physical adapter',True,frozenset({'timed_camera_se3'})))
    feedback_file=(a.output/'public_feedback.jsonl').open('w',buffering=1)
    world.reset()
    floor_readback=audit_ground(stage,'/World/Ground','/World/GroundMaterial')
    (a.output/'single_floor_after_reset.json').write_text(json.dumps(floor_readback,indent=2)+'\n')
    if not floor_readback['passed']:raise RuntimeError('Reset reintroduced duplicate or unbound ground collider')
    names = robot.dof_names
    arm_ids = np.array([names.index(j['name']) for j in spec['joints']])
    left = np.array([names.index(n) for n in spec['left_wheels']])
    right = np.array([names.index(n) for n in spec['right_wheels']])
    wheel_ids = np.r_[left, right]
    home = np.asarray(robot.get_joint_positions()).copy()
    for i, j in zip(arm_ids, spec['joints']):
        home[i] = j['home']
    for name, value in spec.get('passive_joint_positions', {}).items():
        home[names.index(name)] = value
    kp, kd, efforts = np.full(len(names), 50.), np.full(len(names), 5.), np.full(len(names), 30.)
    kp[wheel_ids], kd[wheel_ids], efforts[wheel_ids] = 0., spec.get('wheel_kd', 100.), spec.get('wheel_effort', 100.)
    for i, j in zip(arm_ids, spec['joints']):
        kp[i], kd[i], efforts[i] = j['kp'], j['kd'], j['effort']
    for name, gains in spec.get('passive_joint_gains', {}).items():
        i = names.index(name)
        kp[i], kd[i], efforts[i] = gains['kp'], gains['kd'], gains['effort']
    view = robot._articulation_view
    view.set_gains(kps=kp[None], kds=kd[None])
    view.set_max_efforts(efforts[None])
    precision_report=dict(schema='tiago_tgs_force_precision/v1',scene_path=physics_scene,
        original_force_every_iteration=flag_before,
        requested_force_every_iteration=bool(a.tgs_force_every_iteration),
        stage_readback_after_reset=force_integration.GetEnableExternalForcesEveryIterationAttr().Get(),
        solver_type=force_integration.GetSolverTypeAttr().Get(),physics_dt_s=world.get_physics_dt(),
        physical_properties_before=precision_before,physical_properties_after=precision_after,
        authored_mass_material_effort_position_iterations_unchanged=physical_only(precision_before)==physical_only(precision_after),
        numerical_velocity_iterations=dict(original=original_velocity_iterations,requested=a.tgs_velocity_iterations,
            actual=int(np.asarray(view.get_solver_velocity_iteration_counts())[0])),
        actual_link_names=view.body_names,actual_link_masses_kg=np.asarray(view.get_body_masses())[0].tolist(),
        original_profile_effort_caps=efforts.tolist(),actual_drive_effort_caps=np.asarray(view.get_max_efforts())[0].tolist(),
        source_asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        source='https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/RigidBodyDynamics.html#constraint-solver',
        limitation='Numerical precision hypothesis; does not itself prove physical stability')
    precision_report['passed']=bool(precision_report['stage_readback_after_reset']==bool(a.tgs_force_every_iteration) and
        precision_report['solver_type']=='TGS' and abs(precision_report['physics_dt_s']-.01)<1e-12 and
        precision_report['numerical_velocity_iterations']['actual']==a.tgs_velocity_iterations and
        np.allclose(precision_report['actual_drive_effort_caps'],efforts,atol=1e-5,rtol=0))
    (a.output/'physics_precision_profile.json').write_text(json.dumps(precision_report,indent=2)+'\n')
    if not precision_report['passed']:raise RuntimeError('TGS precision configuration readback failed')
    robot.set_joint_positions(home)
    robot.set_joint_velocities(np.zeros(len(names)))
    controller = robot.get_articulation_controller()
    wbc = CameraController(spec['joints'], spec['wheel_radius_m'], spec['wheel_separation_m'], dt)
    parking = a.control_module.WheelParkingServo()
    active_wheel_kp=0.
    full_dof_guard=a.control_module.PhysicalDofGuard(names,tree.findall('joint'),spec['left_wheels']+spec['right_wheels'])
    target = home.copy()
    rows, results = [], []
    steps, start_wall = 0, time.monotonic()
    last_vw = np.zeros(2)
    last_joint_speed = np.zeros(len(arm_ids))
    base0 = None
    peak_tilt = 0.
    arm_lower, arm_upper = wbc.lower, wbc.upper
    measured_limits = np.asarray(view.get_dof_limits())[0, arm_ids]
    linear_names = {p.GetName() for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.IsA(UsdPhysics.PrismaticJoint)}
    stop_joint_tolerance = np.array([.002 if j['name'] in linear_names else .01 for j in spec['joints']])
    if np.any(arm_lower < measured_limits[:, 0]-1e-4) or np.any(arm_upper > measured_limits[:, 1]+1e-4):
        raise ValueError(f'profile limits exceed physical model: {measured_limits}')
    state_file = (a.output/'samples.jsonl').open('w', buffering=1)
    configuration = dict(spec=spec, dof_names=names, body_names=view.body_names, home=home.tolist(),
        raw_full_dof_guard=full_dof_guard.manifest(),
        physical_joint_limits=measured_limits.tolist(), kp=kp.tolist(), kd=kd.tolist(), effort=efforts.tolist(),
        controlled_joint_units={j['name']: ('m' if j['name'] in linear_names else 'rad') for j in spec['joints']},
        asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), physical_root=str(robot.prim_path),
        jacobian_shape=list(view.get_jacobians().shape), floor_height_m=0., base_relocated_once_before_reset=True,
        enable_external_forces_every_iteration=bool(force_integration.GetEnableExternalForcesEveryIterationAttr().Get()),
        tgs_force_integration_source='https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/RigidBodyDynamics.html#constraint-solver',
        measurement_scope='one cached actual articulation tensor acquisition per physical steps counter; all 37 links directly mapped',
        base_linear_velocity_reference='measured base_link origin, COM velocity shifted using actual angular velocity and original COM offset')
    configuration['full_suite_stop_policy']=dict(fixed_hold_s=5.,window_s=.5,
        selection='only final complete0.5s after each predetermined5s hold; no favorable-window search')
    configuration['wheel_parking_servo']=parking.manifest()
    configuration['independent_joint_policy']=dict(authored_motion_s=8.,home_return_s=5.,zero_command_stop_s=5.,
        home_return='P from actual measured q toward original physical home through native RateTask and unchanged issued speed/acceleration; actual measured home tolerance required',
        drive_bias='gravity-compensation bias may remain, logged explicitly; no position target teleport or forced target-home convergence',
        measured_home_tolerances=stop_joint_tolerance.tolist(),
        stop='original native zero-velocity request; fixed drive position target, predetermined final0.5s of fixed5s hold')
    (a.output/'configuration.json').write_text(json.dumps(configuration, indent=2)+'\n')
    print('CONFIGURATION', json.dumps(configuration), flush=True)

    body_names=list(rigid)
    body_indices=np.array([view.get_body_index(name) for name in body_names])
    if len(set(body_indices))!=len(body_names) or np.any(body_indices<0):
        raise RuntimeError('Every measured rigid body must map to one actual articulation link')
    base_index=view.get_body_index(spec['base_link'])
    link_com=np.asarray(view.get_body_coms()[0])[0].copy()
    floating=configuration['jacobian_shape'][-1]==len(names)+6
    columns=(6 if floating else 0)+arm_ids
    jacobian_indices=body_indices if floating else body_indices-1
    measurement_cache=[-1,None]
    timing_rows=[]
    from isaacsim.core.simulation_manager import SimulationManager
    SimulationManager.get_physics_sim_view().update_articulations_kinematic()

    def acquire():
        if measurement_cache[0]==steps:
            return measurement_cache[1]
        started=time.perf_counter()
        links=np.asarray(view._physics_view.get_link_transforms())[0]
        all_poses=np.zeros((len(links),4,4))
        all_poses[:,:3,:3]=Rotation.from_quat(links[:,3:7]).as_matrix()
        all_poses[:,:3,3]=links[:,:3];all_poses[:,3,3]=1.
        poses={name:all_poses[index] for name,index in zip(body_names,body_indices)}
        q=np.asarray(robot.get_joint_positions());qd=np.asarray(robot.get_joint_velocities())
        camera=poses[rig.camera_spec['parent']]@rig.camera_local
        bp=poses[spec['base_link']]
        velocity=np.asarray(view._physics_view.get_link_velocities())[0,base_index]
        angular=velocity[3:].copy()
        com_world=bp[:3,:3]@link_com[base_index]+bp[:3,3]
        linear=velocity[:3]+np.cross(angular,bp[:3,3]-com_world)
        jacobians=np.asarray(view.get_jacobians())[0]
        body_js=np.zeros((len(body_names),6,10))
        body_js[:,:3,0]=bp[:3,0]
        body_js[:,:3,1]=np.cross(bp[:3,2],all_poses[body_indices,:3,3]-bp[:3,3])
        body_js[:,3:,1]=bp[:3,2]
        # PhysX dense Jacobian linear rows refer to each link COM. FCL point
        # Jacobians use the actor origin: v_origin=v_COM+omega x(p_origin-p_COM).
        origin_from_com=-np.einsum('bij,bj->bi',all_poses[body_indices,:3,:3],link_com[body_indices])
        body_js[:,:,2:]=shift_jacobian(jacobians[jacobian_indices][:,:,columns],origin_from_com)
        body_jacobians={name:body_js[i] for i,name in enumerate(body_names)}
        camera_J=body_jacobians[spec['ee_link']].copy()
        camera_J[:3]+=np.cross(camera_J[3:].T,camera[:3,3]-poses[spec['ee_link']][:3,3]).T
        result=dict(q=q,qd=qd,poses=poses,camera=camera,linear=linear,angular=angular,
            body_jacobians=body_jacobians,camera_jacobian=camera_J,readback_wall_s=time.perf_counter()-started)
        measurement_cache[:]=[steps,result]
        return result

    def measured():
        state=acquire()
        return state['q'],state['poses'],state['camera']

    # Compare batching to the original per-body PhysX query once, outside pacing.
    initial_batch=acquire();position_error=orientation_error=0.
    for name,body in rigid.items():
        original=pose_transform(*body.get_world_pose());batched=initial_batch['poses'][name]
        position_error=max(position_error,float(np.linalg.norm(original[:3,3]-batched[:3,3])))
        orientation_error=max(orientation_error,float(Rotation.from_matrix(original[:3,:3]@batched[:3,:3].T).magnitude()))
    batch_equivalence=dict(body_count=len(body_names),maximum_position_error_m=position_error,
        maximum_orientation_error_rad=orientation_error,
        passed=position_error<1e-5 and orientation_error<1e-5,
        source='original SingleRigidPrim PhysX queries versus same-tick articulation tensor links')
    (a.output/'batched_measurement_equivalence.json').write_text(json.dumps(batch_equivalence,indent=2)+'\n')
    if not batch_equivalence['passed']:raise RuntimeError('Batched physical measurement differs from original source')

    # Independent multiple-posture reference-point check BEFORE the first paced
    # physics step. These initial-state placements are never trajectory samples.
    # Compare corrected PhysX tensors against BOTH original-URDF analytic FK/J
    # and finite differences of actual PhysX actor transforms, including base
    # tangent translation/yaw, proximal links, and the end effector.
    from source_state import source_state
    kinematic_root=initial_batch['poses'][spec['base_link']].copy()
    diagnostic_samples=[np.asarray([j['home'] for j in spec['joints']]),
        np.array([.10,.65,-.35,-.8,1.35,.25,.2,-.3]),np.array([.28,.2,-.9,-.2,1.85,-.3,.75,.4])]
    jacobian_checks=[]
    epsilon=.002
    def initial_placement(joint_values,base_transform):
        robot.set_joint_positions(joint_values)
        robot.set_joint_velocities(np.zeros(len(names)))
        robot.set_world_pose(position=base_transform[:3,3],
            orientation=Rotation.from_matrix(base_transform[:3,:3]).as_quat()[[3,0,1,2]])
        SimulationManager.get_physics_sim_view().update_articulations_kinematic()
        measurement_cache[0]=-1
        return acquire()
    try:
        for posture_index,q8 in enumerate(diagnostic_samples):
            q14=home.copy();q14[arm_ids]=q8
            base=kinematic_root.copy()
            base[:3,:3]=kinematic_root[:3,:3]@Rotation.from_euler('xyz',[.07*posture_index,-.05*posture_index,.43*posture_index]).as_matrix()
            baseline=initial_placement(q14,base)
            source_poses,source_js=source_state(spec,q8,a.output/'sources/tiago_physics.urdf')
            analytic_error=position_error=orientation_error=0.
            for name in body_names:
                J=source_js[name].copy();J[:3]=base[:3,:3]@J[:3];J[3:]=base[:3,:3]@J[3:]
                analytic_error=max(analytic_error,float(np.max(np.abs(J-baseline['body_jacobians'][name]))))
                actual,expected=baseline['poses'][name],base@source_poses[name]
                position_error=max(position_error,float(np.linalg.norm(actual[:3,3]-expected[:3,3])))
                orientation_error=max(orientation_error,float(Rotation.from_matrix(actual[:3,:3]@expected[:3,:3].T).magnitude()))
            linear_error=angular_error=0.
            for column in range(10):
                perturbed=[]
                for sign in [-1.,1.]:
                    q_perturbed=q14.copy();base_perturbed=base.copy()
                    if column==0:base_perturbed[:3,3]+=base[:3,0]*epsilon*sign
                    elif column==1:base_perturbed[:3,:3]=base[:3,:3]@Rotation.from_euler('z',epsilon*sign).as_matrix()
                    else:q_perturbed[arm_ids[column-2]]+=epsilon*sign
                    perturbed.append(initial_placement(q_perturbed,base_perturbed)['poses'])
                for name in body_names:
                    minus,plus=perturbed[0][name],perturbed[1][name]
                    linear=(plus[:3,3]-minus[:3,3])/(2*epsilon)
                    angular=Rotation.from_matrix(plus[:3,:3]@minus[:3,:3].T).as_rotvec()/(2*epsilon)
                    J=baseline['body_jacobians'][name]
                    linear_error=max(linear_error,float(np.max(np.abs(linear-J[:3,column]))))
                    angular_error=max(angular_error,float(np.max(np.abs(angular-J[3:,column]))))
            jacobian_checks.append(dict(posture_index=posture_index,q8=q8.tolist(),T_world_base=base.tolist(),
                body_count=len(body_names),tested_columns=10,maximum_source_position_error_m=position_error,
                maximum_source_orientation_error_rad=orientation_error,maximum_source_jacobian_error=analytic_error,
                maximum_actual_central_difference_linear_error=linear_error,maximum_actual_central_difference_angular_error=angular_error,
                passed=bool(position_error<1e-5 and orientation_error<1e-5 and analytic_error<2e-5 and linear_error<5e-4 and angular_error<5e-4)))
    finally:
        restored=initial_placement(home,kinematic_root)
    restore_report=dict(maximum_joint_position_error=float(np.max(np.abs(restored['q']-home))),
        base_position_error_m=float(np.linalg.norm(restored['poses'][spec['base_link']][:3,3]-kinematic_root[:3,3])),
        base_orientation_error_rad=float(Rotation.from_matrix(restored['poses'][spec['base_link']][:3,:3]@kinematic_root[:3,:3].T).magnitude()),
        maximum_raw_joint_velocity=float(np.max(np.abs(restored['qd']))))
    restore_report['passed']=bool(restore_report['maximum_joint_position_error']<2e-6 and
        restore_report['base_position_error_m']<2e-6 and restore_report['base_orientation_error_rad']<2e-6 and
        restore_report['maximum_raw_joint_velocity']<1e-7)
    jacobian_verification=dict(schema='tiago_physical_jacobian_reference/v1',passed=all(v['passed'] for v in jacobian_checks),
        actual_physics_advances=0,finite_difference_epsilon=epsilon,body_names=body_names,checks=jacobian_checks,
        initial_state_restoration=restore_report,
        source='independent original URDF analytic origin Jacobians and central differences of actual PhysX actor transforms',
        physx_reference='https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/docs/Articulations.html',
        spatial_reference='world [linear;angular], COM shifted to actor origin; synthetic differential-base tangent at actual base origin')
    (a.output/'jacobian_reference_verification.json').write_text(json.dumps(jacobian_verification,indent=2)+'\n')
    if not jacobian_verification['passed'] or not restore_report['passed']:
        raise RuntimeError('Independent actual PhysX/URDF actor-origin Jacobian validation or initial restoration failed')
    start_wall=time.monotonic()

    def step(vw, joint_speed, phase, ref=None,joint_ref=None,park=False):
        nonlocal steps, peak_tilt, last_vw, last_joint_speed,active_wheel_kp
        tick_start=time.perf_counter()
        state=acquire()
        measured_q,measured_poses,measured_camera=state['q'],state['poses'],state['camera']
        measured_body_js=state['body_jacobians']
        base_T=measured_poses[spec['base_link']]
        desired=np.r_[vw,joint_speed]
        measured_qd=state['qd']
        full_dof_motion=full_dof_guard.check(measured_q,measured_qd)
        packet=dict(sample_s=steps*dt,phase=phase,q=measured_q[arm_ids],previous=np.r_[last_vw,last_joint_speed],
            poses=measured_poses,body_jacobians=measured_body_js,measured_camera=measured_camera,mode='actuator',desired_rates=desired,
            measured_rates=np.r_[base_T[:3,0]@state['linear'],base_T[:3,2]@state['angular'],measured_qd[arm_ids]])
        if ref is not None and wbc.camera_request is not None:
            packet.update(mode='camera',**wbc.camera_request)
        measured_motion=wbc.measured_motion_check(measured_q[arm_ids],packet['measured_rates'])
        before_native=time.perf_counter()
        native_command,native_status=native.command(packet)
        after_native=time.perf_counter()
        safe_command=wbc.guard(measured_q[arm_ids],native_command)
        vw,joint_speed=safe_command[:2],safe_command[2:]
        wbc.previous=safe_command.copy()
        vw = np.clip(np.asarray(vw, dtype=float), last_vw-np.array([.45, .65])*dt, last_vw+np.array([.45, .65])*dt)
        last_vw = vw.copy()
        joint_speed = np.clip(np.asarray(joint_speed, dtype=float), last_joint_speed-wbc.accel[2:]*dt,
                              last_joint_speed+wbc.accel[2:]*dt)
        last_joint_speed = joint_speed.copy()
        velocities = np.zeros(len(names))
        rates = wheel_speeds(*vw, spec['wheel_radius_m'], spec['wheel_separation_m'])
        for ids, rate in [(left, rates[0]), (right, rates[1])]:
            for i in ids:
                velocities[i] = rate*spec.get('wheel_signs', {}).get(names[i], 1.)
        velocities[arm_ids] = joint_speed
        parking_state=parking.update(steps*dt,park,rates,measured_q[wheel_ids],measured_qd[wheel_ids])
        if parking_state['kp']!=active_wheel_kp:
            kp[wheel_ids]=parking_state['kp'];view.set_gains(kps=kp[None],kds=kd[None])
            active_wheel_kp=parking_state['kp']
            actual_caps=np.asarray(view.get_max_efforts())[0,wheel_ids]
            if not np.allclose(actual_caps,efforts[wheel_ids],rtol=0.,atol=1e-6):
                raise RuntimeError('Parking changed original wheel effort cap')
            parking.events[-1].update(drive_effort_caps_Nm=actual_caps.tolist(),external_actuation_forces_Nm=
                np.asarray(view._physics_view.get_dof_actuation_forces())[0,wheel_ids].tolist())
            if np.max(np.abs(parking.events[-1]['external_actuation_forces_Nm']))>1e-7:
                raise RuntimeError('Unexpected external wheel actuation force in parking mode')
        target[arm_ids] = np.clip(target[arm_ids]+np.asarray(joint_speed)*dt, arm_lower+.002, arm_upper-.002)
        # Independent indexed actions avoid sending position targets to wheels.
        hold_ids = np.array([i for i in range(len(names)) if i not in wheel_ids])
        controller.apply_action(ArticulationAction(joint_positions=target[hold_ids], joint_velocities=velocities[hold_ids], joint_indices=hold_ids))
        controller.apply_action(ArticulationAction(joint_positions=parking_state['target_rad'],
            joint_velocities=velocities[wheel_ids], joint_indices=wheel_ids))
        before_physics=time.perf_counter()
        world.step(render=False)
        after_physics=time.perf_counter()
        steps += 1
        post=acquire();q,poses,camera=post['q'],post['poses'],post['camera']
        after_readback=time.perf_counter()
        bp = poses[spec['base_link']]
        # Bind feedback to this row's POST-step measurement, not the preceding
        # state. Keep a violating raw sample/explicit public failure as evidence
        # before terminating; never advance another physical tick after it.
        post_guard_error=None
        try:
            post_full_guard=full_dof_guard.check(q,post['qd'])
            post_motion_guard=wbc.measured_motion_check(q[arm_ids],np.r_[
                bp[:3,0]@post['linear'],bp[:3,2]@post['angular'],post['qd'][arm_ids]])
        except RuntimeError as exc:
            post_guard_error=str(exc)
            post_full_guard=dict(raw_physical_dofs_checked=len(q),passed=False,error=post_guard_error)
            post_motion_guard=dict(passed=False,error=post_guard_error)
        tilt = np.arccos(np.clip(bp[2, 2], -1., 1.))
        peak_tilt = max(peak_tilt, float(tilt))
        if (not np.isfinite(q).all() or not np.isfinite(camera).all() or tilt > .35
                or np.any(q[arm_ids] < arm_lower-.015) or np.any(q[arm_ids] > arm_upper+.015)):
            raise RuntimeError('physical state/tilt/joint-limit safety check failed')
        row = dict(t_s=steps*dt, phase=phase, q=q.tolist(), qd=post['qd'].tolist(),
                   T_world_base=bp.tolist(), T_world_camera=camera.tolist(), command_vw=list(map(float, vw)),
                   command_joint_speed=np.asarray(joint_speed).tolist(), base_tilt_rad=float(tilt),
                   controlled_joint_position_targets=target[arm_ids].tolist(),
                   native=native_status,
                   wheel_parking=dict(active=parking_state['active'],kp=parking_state['kp'],
                       target_rad=None if parking_state['target_rad'] is None else parking_state['target_rad'].tolist()),
                   measured_motion_guard=measured_motion,
                   measured_full_dof_guard=full_dof_motion,
                   post_step_measured_motion_guard=post_motion_guard,
                   post_step_full_dof_guard=post_full_guard,
                   base_linear_velocity=post['linear'].tolist(),base_angular_velocity=post['angular'].tolist())
        observation_wall=feedback_wall=0.
        if session.task_id is not None:
            before_observation=time.perf_counter()
            obs=geometry.observe(steps*dt,bp,camera,poses)
            after_observation=time.perf_counter();observation_wall=after_observation-before_observation
            configuration_ok=bool(np.all(q>=full_dof_guard.lower) and np.all(q<=full_dof_guard.upper))
            reserve=float(np.clip(2*np.min(np.minimum(q[arm_ids]-arm_lower,arm_upper-q[arm_ids])/(arm_upper-arm_lower)),0.,1.))
            collision=native_status.get('collision') or {}
            clearance=collision.get('minimum_distance_lower_bound_m')
            constraints=dict(configuration=dict(status='pass' if configuration_ok else 'fail',stamp_s=steps*dt,
                scope='measured_native_limits',reserve_fraction=reserve),
                actuation=dict(status='fail' if post_guard_error else ('pass' if native_status['fresh_native_command'] and native_status.get('hard_constraint_violation',1.)<1e-6 else 'unknown'),
                    stamp_s=steps*dt,scope='measured_native_limits'),
                self_collision=dict(status='pass' if native_status['fresh_native_command'] and clearance is not None and clearance>=0. else 'unknown',
                    stamp_s=native_status['command_sample_s'],scope='measured_collision_model',clearance_m=clearance),
                environment=dict(status='unknown',stamp_s=None,scope='not_available'),
                policy_domain=dict(status='not_applicable',stamp_s=None,scope='not_applicable'))
            feedback=session.feedback(obs,feasible=False if post_guard_error else None,
                reason='actuation_limit' if post_guard_error else ('accepted' if native_status['fresh_native_command'] else 'command_stale'),
                hold_terminal=True,constraints=constraints,now_s=steps*dt)
            feedback_wall=time.perf_counter()-after_observation
            feedback.update(native_command_fresh=native_status['fresh_native_command'],environment_evidence='unknown',
                constraints_verified=False,full_world_executor=bool(native_status.get('full_world_executor')),observation=obs.to_dict())
            feedback_file.write(json.dumps(feedback)+'\n')
            row['task_feedback']=feedback
        row.update(stamp_s=steps*dt,case=phase,joint_positions=q.tolist(),
            joint_names=list(names),
            joint_units=['m' if name in linear_names else 'rad' for name in names],
            T_world_camera_optical=camera.tolist(),T_world_reference_optical=ref.tolist() if ref is not None else None,
            requested_body_velocity=[float(desired[0]),0.,float(desired[1])],
            applied_body_velocity=[float(vw[0]),0.,float(vw[1])],
            measured_body_velocity=[float(bp[:3,0]@np.asarray(row['base_linear_velocity'])),
                float(bp[:3,1]@np.asarray(row['base_linear_velocity'])),float(bp[:3,2]@np.asarray(row['base_angular_velocity']))],
            base_linear_velocity_world=row['base_linear_velocity'],base_angular_velocity_world=row['base_angular_velocity'])
        if a.probe or a.diagnostic_forces:
            applied = controller.get_applied_action()
            row['applied_velocity_targets'] = np.asarray(applied.joint_velocities).tolist()
            row['physical_gains_kp'] = np.asarray(view.get_gains()[0])[0].tolist()
            row['physical_gains_kd'] = np.asarray(view.get_gains()[1])[0].tolist()
            row['measured_joint_efforts'] = np.asarray(robot.get_measured_joint_efforts()).tolist()
            row['measured_joint_effort_scope']='projected joint constraint force/torque including reactions; not direct motor torque'
            row['external_joint_actuation_forces']=np.asarray(view._physics_view.get_dof_actuation_forces())[0].tolist()
        if ref is not None:
            delta = Rotation.from_matrix(ref[:3, :3]@camera[:3, :3].T).as_rotvec()
            row.update(reference=ref.tolist(), position_error_m=float(np.linalg.norm(ref[:3, 3]-camera[:3, 3])), orientation_error_rad=float(np.linalg.norm(delta)))
        if joint_ref is not None:
            row['authored_joint_reference']=np.asarray(joint_ref).tolist()
        before_serialization=time.perf_counter()
        state_file.write(json.dumps(row)+'\n')
        rows.append(row)
        if a.sensor_audit:
            audit.record(steps*dt, poses, phase)
        timing_rows.append(dict(stamp_s=steps*dt,phase=phase,pre_control_s=before_native-tick_start,
            native_transport_s=after_native-before_native,apply_s=before_physics-after_native,
            physics_s=after_physics-before_physics,readback_s=after_readback-after_physics,
            observe_s=observation_wall,feedback_s=feedback_wall,serialize_s=time.perf_counter()-before_serialization,
            total_active_step_s=time.perf_counter()-tick_start))
        if post_guard_error:
            raise RuntimeError('Post-step actual physical safety guard: '+post_guard_error)
        remaining = start_wall+steps*dt-time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return q, poses, camera

    def hold(seconds, label):
        for _ in range(round(seconds/dt)):
            state = step([0., 0.], np.zeros(len(arm_ids)), label,park=True)
        return state

    def strict_hold(label):
        """Fixed5s physics hold; assess only its predetermined final0.5s."""
        first=steps
        for i in range(500):
            state=step([0.,0.],np.zeros(len(arm_ids)),label,park=True)
        tail=rows[-51:]
        report=audit_stop([r['t_s'] for r in tail],[r['T_world_base'] for r in tail],
            [r['T_world_camera'] for r in tail],np.array([r['q'] for r in tail]),
            full_dof_guard.units,tail[-1]['base_linear_velocity'],tail[-1]['base_angular_velocity'])
        results.append(dict(phase=label,hold_s=(steps-first)*dt,window_selection='predetermined final0.5s of fixed5s hold',**report))
        (a.output/'partial_phase_results.json').write_text(json.dumps(results,indent=2)+'\n')
        if not report['passed']:
            raise RuntimeError(label+': original strict measured stop failed at end of fixed5s physical hold')
        return state

    try:
        q, poses, camera = hold(2., 'settle')
        base0 = poses[spec['base_link']].copy()
        if a.probe:
            for i in range(200):
                ramp = min(1., i/50., (199-i)/50.)
                q, poses, camera = step([.15*ramp, 0.], np.zeros(len(arm_ids)), 'forward_probe')
            hold(3., 'stop')
            delta = poses[spec['base_link']][:3, 3]-base0[:3, 3]
            results.append(dict(phase='forward_probe', displacement_m=delta.tolist(), passed=bool(delta@base0[:3, 0] > .12)))
            strict_stop=audit_stop([r['t_s'] for r in rows],
                [r['T_world_base'] for r in rows],[r['T_world_camera'] for r in rows],
                np.array([r['q'] for r in rows]),['m' if name in linear_names else 'rad' for name in names],
                rows[-1]['base_linear_velocity'],rows[-1]['base_angular_velocity'])
            results.append(dict(phase='measured_stop',**strict_stop))
        else:
            if not a.wbc_only:
                # Turn then drive: no artificial lateral/prismatic base actuators.
                for label, angle, sign in [('forward', 0., 1.), ('backward', 0., -1.),
                                           ('left', np.pi/2, 1.), ('right', -np.pi/2, 1.),
                                           ('diagonal_ne', np.pi/4, 1.), ('diagonal_sw', -3*np.pi/4, 1.)]:
                    if a.arm_only or (a.direction_probe and label!=a.direction_probe):continue
                    yaw_target = angle
                    yaw_integral = 0.
                    heading_steps = round(float(spec.get('heading_timeout_s', 60.))/dt)
                    heading_speed = float(spec.get('base_max_yaw_radps', .3))
                    for i in range(heading_steps):
                        _, poses, _ = measured()
                        R = poses[spec['base_link']][:3, :3]
                        yaw = np.arctan2(R[1, 0], R[0, 0])
                        error = np.arctan2(np.sin(yaw_target-yaw), np.cos(yaw_target-yaw))
                        yaw_integral = np.clip(yaw_integral+error*dt, -1.2, 1.2)
                        omega = np.clip(2.*error+.5*yaw_integral, -heading_speed, heading_speed)
                        q, poses, camera = step([0., omega], np.zeros(len(arm_ids)), label+'_turn')
                        if abs(error) < .015 and i > 50:
                            break
                    if abs(error) > .04:
                        raise RuntimeError(f'{label}: wheel-driven turn did not reach heading ({error})')
                    before = poses[spec['base_link']][:3, 3].copy()
                    for i in range(300):
                        R = poses[spec['base_link']][:3, :3]
                        yaw = np.arctan2(R[1, 0], R[0, 0])
                        error = np.arctan2(np.sin(angle-yaw), np.cos(angle-yaw))
                        ramp = min(1., i/50., (299-i)/50.)
                        q, poses, camera = step([sign*.15*ramp, np.clip(error, -.15, .15)], np.zeros(len(arm_ids)), label)
                    delta = poses[spec['base_link']][:3, 3]-before
                    along = delta[:2]@np.array([np.cos(angle), np.sin(angle)])*sign
                    across = abs(delta[:2]@np.array([-np.sin(angle), np.cos(angle)]))
                    results.append(dict(phase=label, displacement_m=delta.tolist(), along_m=float(along), lateral_error_m=float(across), passed=bool(along > .25 and across < .08)))
                    q,poses,camera=strict_hold(label+'_measured_stop')
                if not a.direction_probe:
                    # Execute a frozen authored joint-position sinusoid in both
                    # directions and explicitly return to the original home pose.
                    before_base = poses[spec['base_link']].copy()
                    arm_start=np.array([j['home'] for j in spec['joints']])
                    arm_states = []
                    amplitude=np.asarray(a.arm_reference['amplitude'])
                    for i in range(800):
                        authored=a.arm_reference['samples'][i+1]
                        joint_reference=np.asarray(authored['position'])
                        speed=np.clip(np.asarray(authored['velocity'])+2.*(joint_reference-q[arm_ids]),-wbc.speed[2:],wbc.speed[2:])
                        q, poses, camera = step([0., 0.], speed, 'independent_arm',joint_ref=joint_reference,park=True)
                        arm_states.append(q[arm_ids].copy())
                    # Returning home is a motion phase, not a stationary test.
                    # Reach the original measured home through the genuine
                    # rate task and unchanged issued caps. A small low-level
                    # gravity-compensation position bias is retained, not
                    # snapped away. The following stop phase requests zero.
                    for i in range(500):
                        current_q,_,_=measured()
                        speed=np.clip(2.*(arm_start-current_q[arm_ids]),-wbc.speed[2:],wbc.speed[2:])
                        q,poses,camera=step([0.,0.],speed,'independent_arm_home_return',joint_ref=arm_start,park=True)
                    home_error=np.abs(q[arm_ids]-arm_start)
                    target_error=np.abs(target[arm_ids]-arm_start)
                    home_return=dict(phase='independent_arm_home_return',duration_s=5.,
                        measured_home_error=home_error.tolist(),home_tolerance=stop_joint_tolerance.tolist(),
                        drive_target_bias_to_home=target_error.tolist(),
                        passed=bool(np.all(home_error<stop_joint_tolerance)))
                    results.append(home_return)
                    (a.output/'partial_phase_results.json').write_text(json.dumps(results,indent=2)+'\n')
                    if not home_return['passed']:raise RuntimeError('Fixed5s independent-arm home return failed original measured tolerance or drive-target convergence')
                    q,poses,camera=strict_hold('independent_arm_measured_stop')
                    base_drift = float(np.linalg.norm(poses[spec['base_link']][:2, 3]-before_base[:2, 3]))
                    arm_span = np.ptp(arm_states, axis=0)
                    return_error=np.abs(q[arm_ids]-arm_start)
                    results.append(dict(phase='independent_arm', base_drift_m=base_drift, measured_joint_span=arm_span.tolist(),
                                        authored_motion_s=8.,home_return_s=5.,strict_zero_command_stop_s=5.,
                                        home_return_error=return_error.tolist(),home_return_tolerance=stop_joint_tolerance.tolist(),
                                        passed=bool(base_drift < .03 and np.all(arm_span > amplitude*.5) and np.all(return_error<stop_joint_tolerance))))
            else:
                q, poses, camera = measured()
            if not a.direction_probe and not a.arm_only:
                initial = camera.copy()
                observation=geometry.observe(steps*dt,poses[spec['base_link']],camera,poses)
                trajectory, duration = None, 16.
                if a.trajectory:
                    trajectory = a.control_module.validate_trajectory(a.trajectory_data)
                    duration = trajectory[0][-1]
                # Author a new acceptance task, or admit the caller's unchanged task.
                # Public axes and absolute simulation clock remain in the live session.
                if a.task:
                    task=TimedTask.from_dict(json.loads(a.task.read_text()))
                    duration=task.trajectory.end_s-steps*dt
                else:
                    authored_times=np.linspace(0.,duration,321)
                    authored_poses=[]
                    for relative_t in authored_times:
                        ref=initial.copy()
                        if trajectory:
                            ts,ps,slerp=trajectory
                            ref[:3,3]=[np.interp(relative_t,ts,ps[:,k]) for k in range(3)]
                            ref[:3,:3]=slerp(relative_t).as_matrix()
                        else:
                            s=2*np.pi*relative_t/duration
                            ref[:3,3]+=[.22*(1-np.cos(s)),.10*np.sin(s),.06*np.sin(s)]
                            ref[:3,:3]=Rotation.from_rotvec([.08*np.sin(s),.10*np.sin(s),.15*np.sin(s)]).as_matrix()@initial[:3,:3]
                        authored_poses.append(semantic_camera(public_pose(ref)))
                    if trajectory:
                        # Preserve the caller's original knots and exact linear/SLERP curve.
                        authored_times=trajectory[0]
                        authored_poses=[]
                        for relative_t in authored_times:
                            ref=initial.copy();ref[:3,3]=[np.interp(relative_t,trajectory[0],trajectory[1][:,k]) for k in range(3)]
                            ref[:3,:3]=trajectory[2](relative_t).as_matrix()
                            authored_poses.append(semantic_camera(public_pose(ref)))
                    camera_path=CameraTrajectory(steps*dt,tuple(map(float,authored_times)),tuple(authored_poses))
                    task=TimedTask('tiago_giskard_physical_camera',steps*dt,observation.id,camera_path,.03,.08)
                admitted=session.accept(task,observation,steps*dt)
                (a.output/'public_task.json').write_text(json.dumps(task.to_dict(),indent=2)+'\n')
                (a.output/'timed_tasks.json').write_text(json.dumps([task.to_dict()],indent=2)+'\n')
                (a.output/'public_initial_observation.json').write_text(json.dumps(observation.to_dict(),indent=2)+'\n')
                (a.output/'accepted_observations.json').write_text(json.dumps([observation.to_dict()],indent=2)+'\n')
                (a.output/'public_admission.json').write_text(json.dumps(admitted,indent=2)+'\n')
                if not admitted['supported']:
                    raise RuntimeError('Public task unsupported/rejected: '+repr(admitted))
                whole_rows=[]
                for i in range(round(duration/dt)):
                    absolute_t=min((steps+1)*dt,task.trajectory.end_s)
                    ref,feedforward=session.reference(absolute_t)
                    q,poses,camera=measured()
                    J=acquire()['camera_jacobian']
                    error=np.r_[ref[:3,3]-camera[:3,3],Rotation.from_matrix(ref[:3,:3]@camera[:3,:3].T).as_rotvec()]
                    wbc.camera_request=dict(jacobian=J,error=error,feedforward=feedforward,reference=ref)
                    q,poses,camera=step(np.zeros(2),np.zeros(len(arm_ids)),'whole_body_camera',ref)
                    whole_rows.append(rows[-1])
                errors = np.array([r['position_error_m'] for r in whole_rows])
                angles = np.array([r['orientation_error_rad'] for r in whole_rows])
                bases = np.array([np.asarray(r['T_world_base'])[:2, 3] for r in whole_rows])
                arms = np.array([np.asarray(r['q'])[arm_ids] for r in whole_rows])
                outside_position=errors>task.position_tolerance_m
                outside_orientation=angles>task.orientation_tolerance_rad
                outside=outside_position|outside_orientation
                run=longest=0;longest_end=None
                for index,flag in enumerate(outside):
                    run=run+1 if flag else 0
                    if run>longest:longest=run;longest_end=index
                pointwise=dict(position_tolerance_m=task.position_tolerance_m,
                    orientation_tolerance_rad=task.orientation_tolerance_rad,
                    position_outside_samples=int(outside_position.sum()),orientation_outside_samples=int(outside_orientation.sum()),
                    either_outside_samples=int(outside.sum()),longest_consecutive_outside_samples=longest,
                    longest_consecutive_first_stamp_s=None if longest_end is None else whole_rows[longest_end-longest+1]['stamp_s'],
                    longest_consecutive_last_stamp_s=None if longest_end is None else whole_rows[longest_end]['stamp_s'],
                    sampling_interval_s=dt,scope='diagnostic of original instantaneous Task tolerances; original RMS+terminal gates unchanged')
                results.append(dict(phase='whole_body_camera', position_rmse_m=float(np.sqrt(np.mean(errors**2))),
                    position_max_m=float(errors.max()), orientation_rmse_rad=float(np.sqrt(np.mean(angles**2))),
                    orientation_max_rad=float(angles.max()),pointwise_tolerance_diagnostic=pointwise,
                    base_span_m=float(np.linalg.norm(np.ptp(bases, axis=0))), joint_span=np.ptp(arms, axis=0).tolist(),
                    passed=bool(np.sqrt(np.mean(errors**2)) < .03 and np.sqrt(np.mean(angles**2)) < .08
                                and np.linalg.norm(np.ptp(bases, axis=0)) > .01 and np.linalg.norm(np.ptp(arms, axis=0)) > .03)))
                q,poses,camera=strict_hold('measured_stop')
        elapsed = time.monotonic()-start_wall
        sensor = audit.finish(steps*dt, world.render if a.sensor_rgb else None)
        if a.sensor_rgb:
            # A detached diagnostic photograph; no physics step or new observation.
            try:
                import omni.replicator.core as rep
                from PIL import Image
                centre = poses[spec['base_link']][:3, 3]+np.array([0., 0., .6])
                eye = centre+np.array([2.2, -2.2, 1.4])
                forward = (centre-eye)/np.linalg.norm(centre-eye)
                right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
                up = np.cross(right, forward)
                overview = UsdGeom.Camera.Define(stage, '/World/DiagnosticOverview')
                set_usd_transform(overview.GetPrim(), transform(eye, np.column_stack([right, up, -forward])))
                overview.CreateFocalLengthAttr(24.)
                product = rep.create.render_product(str(overview.GetPath()), (960, 720))
                rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach(product)
                try:
                    for _ in range(12):
                        world.render()
                    pixels = np.asarray(rgb.get_data()).copy()
                    if not pixels.size:
                        raise RuntimeError('empty overview RGB')
                    Image.fromarray(pixels).save(a.output/'overview.png')
                finally:
                    rgb.detach(product); product.destroy()
            except Exception as exc:
                (a.output/'overview_error.txt').write_text(repr(exc)+'\n')
        stage.Flatten().Export(str(a.output/'tested_scene.usda'))
        timing_pass=bool(elapsed<=steps*dt*1.05)
        fresh_fraction=sum(bool(row['native']['fresh_native_command']) for row in rows)/len(rows)
        worker_times=[row['native']['worker_wall_s'] for row in rows if row['native'].get('worker_wall_s') is not None]
        result = dict(schema='mobile_manipulator_acceptance/v1', robot=spec['name'], complete=True,
            diagnostic_only=bool(a.probe or a.wbc_only or a.direction_probe or a.arm_only), passed=bool(all(x['passed'] for x in results) and timing_pass and fresh_fraction>.9), phases=results,
            simulated_seconds=steps*dt, wall_seconds=elapsed, timing_pass=bool(elapsed <= steps*dt*1.05),
            native_fresh_command_fraction=fresh_fraction,
            native_fresh_command_fraction_scope='valid native command lease at each100Hz physical step; not unique solver update rate',
            native_unique_result_count=len({row['native']['command_sample_s'] for row in rows if row['native']['fresh_native_command']}),
            native_effective_result_hz=len({row['native']['command_sample_s'] for row in rows if row['native']['fresh_native_command']})/(steps*dt),
            native_worker_median_wall_s=float(np.median(worker_times)) if worker_times else None,
            native_worker_p99_wall_s=float(np.quantile(worker_times,.99)) if worker_times else None,
            peak_base_tilt_rad=peak_tilt, sensors=sensor, full_motion_range_verified=False,
            avoidance_verified=False, native_backend=a.backend,
            full_world_executor=a.backend=='giskard-executor',representation_ingress_verified=bool(session.task_id is not None),
            native_worker_recovery_events=native.events,limitations=['finite flat-ground tests', 'camera/lidar mounts are massless prototypes',
            'hard native self-collision convex-distance velocity barriers; environment perception/planning remains unknown',
            'separate coverage validator required; motion success alone is not sensor acceptance'])
        (a.output/'acceptance.json').write_text(json.dumps(result, indent=2)+'\n')
        print('ACCEPTANCE', json.dumps(result), flush=True)
        return 0 if result['passed'] else 2
    finally:
        # Always command brake, including after solver/safety exceptions.
        controller.apply_action(ArticulationAction(joint_velocities=np.zeros(len(wheel_ids)), joint_indices=wheel_ids))
        controller.apply_action(ArticulationAction(joint_positions=np.asarray(robot.get_joint_positions())[arm_ids], joint_velocities=np.zeros(len(arm_ids)), joint_indices=arm_ids))
        state_file.close()
        feedback_file.close()
        (a.output/'wheel_parking_events.json').write_text(json.dumps(dict(events=parking.events,policy=parking.manifest()),indent=2)+'\n')
        (a.output/'partial_phase_results.json').write_text(json.dumps(results,indent=2)+'\n')
        (a.output/'timing_breakdown.json').write_text(json.dumps(dict(rows=timing_rows,
            scope='every physical tick; no change to measurement/control frequency or task reference'),indent=2)+'\n')
        native.close()


if __name__ == '__main__':
    if '--worker' in sys.argv or '--help' in sys.argv or '-h' in sys.argv:
        raise SystemExit(main())
    # Kit can terminate Python with exit 0 during shutdown after an exception.
    # This independent parent accepts only the completed measured JSON result.
    import subprocess
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path)
    args, _ = parser.parse_known_args()
    if args.output and args.output.exists():
        raise SystemExit(f'Output already exists; choose a new directory: {args.output}')
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'])
    result_file = args.output/'acceptance.json' if args.output else None
    if result_file is None or not result_file.is_file():
        raise SystemExit(child.returncode or 2)
    result = json.loads(result_file.read_text())
    raise SystemExit(0 if child.returncode == 0 and result.get('complete') and result.get('passed') else 2)
