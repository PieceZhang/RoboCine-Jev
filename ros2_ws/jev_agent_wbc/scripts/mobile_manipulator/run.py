#!/usr/bin/env python3
"""Measured wheel/arm/WBC acceptance for Husky+UR5 and single-arm TIAGo."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'scripts')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--wbc-only', action='store_true', help='Short WBC/stop diagnostic; skips direction and independent-arm acceptance.')
    p.add_argument('--initial-heading-rad', type=float, default=0., help='Initial placement only, before physics; useful for WBC diagnostics.')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--sensor-audit', action='store_true')
    p.add_argument('--sensor-rgb', action='store_true')
    p.add_argument('--sensor-config', type=Path)
    p.add_argument('--trajectory', type=Path, help='JSON samples: t_s, position_m, quaternion_xyzw; world optical frame.')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    a.robot_dir, a.output = a.robot_dir.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((a.robot_dir/'robot.json').read_text())
    files = [Path(__file__), Path(__file__).with_name('control.py'), a.robot_dir/'robot.json',
             a.sensor_config or a.robot_dir/spec['sensors'], PACKAGE/'jev_agent_wbc/robot_sensor_rig.py']
    if a.trajectory:
        files.append(a.trajectory)
    frozen = a.output/'sources'
    frozen.mkdir()
    for index, path in enumerate(files):
        (frozen/f'{index}_{path.name}').write_bytes(path.read_bytes())
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
                      physics_hz=100, requested_pace=1., base_execution='wheel velocity drives; free floating physical base',
                      controller='standalone constrained camera resolved-rate QP; no Agent/planner')
    (a.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
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
    CameraController = a.control_module.CameraController
    camera_jacobian, wheel_speeds = a.control_module.camera_jacobian, a.control_module.wheel_speeds

    dt, root = .01, '/World/Robot'
    world = World(physics_dt=dt, rendering_dt=.02, stage_units_in_meters=1.)
    stage = world.stage
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
    for name in rig.parent_names | {spec['base_link'], spec['ee_link']}:
        matches = [p for p in bodies if p.GetName() == name]
        if len(matches) != 1:
            raise ValueError(f'expected unique rigid body {name}: {matches}')
        rigid[name] = world.scene.add(SingleRigidPrim(prim_path=str(matches[0].GetPath()), name=name))
    audit = SensorAudit(rig, a.output, a.sensor_audit)
    world.reset()
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
    robot.set_joint_positions(home)
    robot.set_joint_velocities(np.zeros(len(names)))
    controller = robot.get_articulation_controller()
    wbc = CameraController(spec['joints'], spec['wheel_radius_m'], spec['wheel_separation_m'], dt)
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
        physical_joint_limits=measured_limits.tolist(), kp=kp.tolist(), kd=kd.tolist(), effort=efforts.tolist(),
        controlled_joint_units={j['name']: ('m' if j['name'] in linear_names else 'rad') for j in spec['joints']},
        asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), physical_root=str(robot.prim_path),
        jacobian_shape=list(view.get_jacobians().shape), floor_height_m=0., base_relocated_once_before_reset=True)
    (a.output/'configuration.json').write_text(json.dumps(configuration, indent=2)+'\n')
    print('CONFIGURATION', json.dumps(configuration), flush=True)

    def measured():
        poses = {name: pose_transform(*body.get_world_pose()) for name, body in rigid.items()}
        return np.asarray(robot.get_joint_positions()), poses, poses[rig.camera_spec['parent']]@rig.camera_local

    def step(vw, joint_speed, phase, ref=None):
        nonlocal steps, peak_tilt, last_vw, last_joint_speed
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
        target[arm_ids] = np.clip(target[arm_ids]+np.asarray(joint_speed)*dt, arm_lower+.002, arm_upper-.002)
        # Independent indexed actions avoid sending position targets to wheels.
        hold_ids = np.array([i for i in range(len(names)) if i not in wheel_ids])
        controller.apply_action(ArticulationAction(joint_positions=target[hold_ids], joint_velocities=velocities[hold_ids], joint_indices=hold_ids))
        controller.apply_action(ArticulationAction(joint_velocities=velocities[wheel_ids], joint_indices=wheel_ids))
        world.step(render=False)
        steps += 1
        q, poses, camera = measured()
        bp = poses[spec['base_link']]
        tilt = np.arccos(np.clip(bp[2, 2], -1., 1.))
        peak_tilt = max(peak_tilt, float(tilt))
        if (not np.isfinite(q).all() or not np.isfinite(camera).all() or tilt > .35
                or np.any(q[arm_ids] < arm_lower-.015) or np.any(q[arm_ids] > arm_upper+.015)):
            raise RuntimeError('physical state/tilt/joint-limit safety check failed')
        row = dict(t_s=steps*dt, phase=phase, q=q.tolist(), qd=np.asarray(robot.get_joint_velocities()).tolist(),
                   T_world_base=bp.tolist(), T_world_camera=camera.tolist(), command_vw=list(map(float, vw)),
                   command_joint_speed=np.asarray(joint_speed).tolist(), base_tilt_rad=float(tilt))
        if a.probe:
            applied = controller.get_applied_action()
            row['applied_velocity_targets'] = np.asarray(applied.joint_velocities).tolist()
            row['physical_gains_kp'] = np.asarray(view.get_gains()[0])[0].tolist()
            row['physical_gains_kd'] = np.asarray(view.get_gains()[1])[0].tolist()
            row['measured_joint_efforts'] = np.asarray(robot.get_measured_joint_efforts()).tolist()
        if ref is not None:
            delta = Rotation.from_matrix(ref[:3, :3]@camera[:3, :3].T).as_rotvec()
            row.update(reference=ref.tolist(), position_error_m=float(np.linalg.norm(ref[:3, 3]-camera[:3, 3])), orientation_error_rad=float(np.linalg.norm(delta)))
        state_file.write(json.dumps(row)+'\n')
        rows.append(row)
        if a.sensor_audit:
            audit.record(steps*dt, poses, phase)
        remaining = start_wall+steps*dt-time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return q, poses, camera

    def hold(seconds, label):
        for _ in range(round(seconds/dt)):
            state = step([0., 0.], np.zeros(len(arm_ids)), label)
        return state

    try:
        q, poses, camera = hold(2., 'settle')
        base0 = poses[spec['base_link']].copy()
        if a.probe:
            for i in range(200):
                ramp = min(1., i/50., (199-i)/50.)
                q, poses, camera = step([.15*ramp, 0.], np.zeros(len(arm_ids)), 'forward_probe')
            hold(1., 'stop')
            delta = poses[spec['base_link']][:3, 3]-base0[:3, 3]
            results.append(dict(phase='forward_probe', displacement_m=delta.tolist(), passed=bool(delta@base0[:3, 0] > .12)))
        else:
            if not a.wbc_only:
                # Turn then drive: no artificial lateral/prismatic base actuators.
                for label, angle, sign in [('forward', 0., 1.), ('backward', 0., -1.),
                                           ('left', np.pi/2, 1.), ('right', -np.pi/2, 1.),
                                           ('diagonal_ne', np.pi/4, 1.), ('diagonal_sw', -3*np.pi/4, 1.)]:
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
                    hold(.5, label+'_stop')
                # Drive a smooth arm-only joint motion; wheel commands stay zero.
                before_base = poses[spec['base_link']].copy()
                arm_start = q[arm_ids].copy()
                arm_states = []
                amplitude = np.minimum(.12, np.minimum(arm_start-arm_lower, arm_upper-arm_start)*.4)
                for i in range(800):
                    speed = amplitude*(2*np.pi/8)*np.sin(2*np.pi*i*dt/8)
                    q, poses, camera = step([0., 0.], speed, 'independent_arm')
                    arm_states.append(q[arm_ids].copy())
                base_drift = float(np.linalg.norm(poses[spec['base_link']][:2, 3]-before_base[:2, 3]))
                arm_span = np.ptp(arm_states, axis=0)
                results.append(dict(phase='independent_arm', base_drift_m=base_drift, measured_joint_span=arm_span.tolist(),
                                    passed=bool(base_drift < .03 and np.all(arm_span > amplitude*.5))))
                q, poses, camera = hold(1., 'arm_stop')
            else:
                q, poses, camera = measured()
            initial = camera.copy()
            trajectory, duration = None, 16.
            if a.trajectory:
                trajectory = a.control_module.validate_trajectory(a.trajectory_data)
                duration = trajectory[0][-1]
            wbc.previous[:] = 0.
            whole_rows = []
            for i in range(round(duration/dt)):
                t = min((i+1)*dt, duration)
                ref = initial.copy()
                if trajectory:
                    ts, ps, slerp = trajectory
                    ref[:3, 3] = [np.interp(t, ts, ps[:, k]) for k in range(3)]
                    ref[:3, :3] = slerp(t).as_matrix()
                    t0, t1 = max(0., t-dt/2), min(duration, t+dt/2)
                    p0 = np.array([np.interp(t0, ts, ps[:, k]) for k in range(3)])
                    p1 = np.array([np.interp(t1, ts, ps[:, k]) for k in range(3)])
                    angular = (slerp(t1)*slerp(t0).inv()).as_rotvec()/(t1-t0)
                    feedforward = np.r_[(p1-p0)/(t1-t0), angular]
                else:
                    s = 2*np.pi*t/duration
                    ref[:3, 3] += [.22*(1-np.cos(s)), .10*np.sin(s), .06*np.sin(s)]
                    ref[:3, :3] = Rotation.from_rotvec([.08*np.sin(s), .10*np.sin(s), .15*np.sin(s)]).as_matrix()@initial[:3, :3]
                    rate = 2*np.pi/duration
                    feedforward = rate*np.r_[.22*np.sin(s), .10*np.cos(s), .06*np.cos(s),
                                             .08*np.cos(s), .10*np.cos(s), .15*np.cos(s)]
                q, poses, camera = measured()
                link = poses[spec['ee_link']]
                js = np.asarray(view.get_jacobians())[0]
                index = view.get_body_index(spec['ee_link'])
                # Floating articulations include base six columns and root body.
                floating = js.shape[-1] == len(names)+6
                link_j = js[index if floating else index-1, :, (6 if floating else 0)+arm_ids].T
                if link_j.shape != (6, len(arm_ids)):
                    raise RuntimeError(f'Unexpected physical Jacobian shape: {link_j.shape}')
                J = camera_jacobian(poses[spec['base_link']], camera[:3, 3], link_j, camera[:3, 3]-link[:3, 3])
                error = np.r_[ref[:3, 3]-camera[:3, 3], Rotation.from_matrix(ref[:3, :3]@camera[:3, :3].T).as_rotvec()]
                u = wbc.velocity(q[arm_ids], J, error, feedforward)
                q, poses, camera = step(u[:2], u[2:], 'whole_body_camera', ref)
                whole_rows.append(rows[-1])
            errors = np.array([r['position_error_m'] for r in whole_rows])
            angles = np.array([r['orientation_error_rad'] for r in whole_rows])
            bases = np.array([np.asarray(r['T_world_base'])[:2, 3] for r in whole_rows])
            arms = np.array([np.asarray(r['q'])[arm_ids] for r in whole_rows])
            results.append(dict(phase='whole_body_camera', position_rmse_m=float(np.sqrt(np.mean(errors**2))),
                position_max_m=float(errors.max()), orientation_rmse_rad=float(np.sqrt(np.mean(angles**2))),
                base_span_m=float(np.linalg.norm(np.ptp(bases, axis=0))), joint_span=np.ptp(arms, axis=0).tolist(),
                passed=bool(np.sqrt(np.mean(errors**2)) < .03 and np.sqrt(np.mean(angles**2)) < .08
                            and np.linalg.norm(np.ptp(bases, axis=0)) > .01 and np.linalg.norm(np.ptp(arms, axis=0)) > .03)))
            hold(2., 'final_stop')
            stopped = rows[-51:]
            stopped_base = np.array([np.asarray(r['T_world_base'])[:3, 3] for r in stopped])
            stopped_arm = np.array([np.asarray(r['q'])[arm_ids] for r in stopped])
            stopped_camera = np.array([np.asarray(r['T_world_camera'])[:3, 3] for r in stopped])
            base_span = float(np.linalg.norm(np.ptp(stopped_base, axis=0)))
            joint_span = np.ptp(stopped_arm, axis=0)
            camera_span = float(np.linalg.norm(np.ptp(stopped_camera, axis=0)))
            def rotation_span(key):
                rotations = Rotation.from_matrix(np.array([np.asarray(r[key])[:3, :3] for r in stopped]))
                return float(np.max((rotations*rotations[0].inv()).magnitude()))
            base_rotation_span = rotation_span('T_world_base')
            camera_rotation_span = rotation_span('T_world_camera')
            results.append(dict(phase='measured_stop', final_half_second_base_span_m=base_span,
                joint_span=joint_span.tolist(), joint_tolerances=stop_joint_tolerance.tolist(),
                final_half_second_camera_span_m=camera_span,
                final_half_second_base_rotation_span_rad=base_rotation_span,
                final_half_second_camera_rotation_span_rad=camera_rotation_span,
                passed=bool(base_span < .005 and np.all(joint_span < stop_joint_tolerance) and camera_span < .005
                            and base_rotation_span < .01 and camera_rotation_span < .01)))
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
        result = dict(schema='mobile_manipulator_acceptance/v1', robot=spec['name'], complete=True,
            diagnostic_only=bool(a.probe or a.wbc_only), passed=bool(all(x['passed'] for x in results)), phases=results,
            simulated_seconds=steps*dt, wall_seconds=elapsed, timing_pass=bool(elapsed <= steps*dt*1.05),
            peak_base_tilt_rad=peak_tilt, sensors=sensor, full_motion_range_verified=False,
            avoidance_verified=False, limitations=['finite flat-ground tests', 'camera/lidar mounts are massless prototypes',
            'no self-collision distance constraints or environment path planner in this local kinematic WBC',
            'separate coverage validator required; motion success alone is not sensor acceptance'])
        (a.output/'acceptance.json').write_text(json.dumps(result, indent=2)+'\n')
        print('ACCEPTANCE', json.dumps(result), flush=True)
        return 0 if result['passed'] else 2
    finally:
        # Always command brake, including after solver/safety exceptions.
        controller.apply_action(ArticulationAction(joint_velocities=np.zeros(len(wheel_ids)), joint_indices=wheel_ids))
        controller.apply_action(ArticulationAction(joint_positions=np.asarray(robot.get_joint_positions())[arm_ids], joint_velocities=np.zeros(len(arm_ids)), joint_indices=arm_ids))
        state_file.close()


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
