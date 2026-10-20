#!/usr/bin/env python3
"""Independent native OCS2/SLQ measured PhysX execution; historical runners stay intact."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

PACKAGE=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker-binary',type=Path)
    parser.add_argument('--wbc-only',action='store_true')
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--task-file',type=Path)
    parser.add_argument('--sensor-audit',action='store_true')
    parser.add_argument('--sensor-rgb',action='store_true')
    parser.add_argument('--gui',action='store_true')
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.robot_dir=args.robot_dir.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    frozen=args.output/'sources';frozen.mkdir()
    source_files=[Path(__file__),PACKAGE/'scripts/ocs2_wheeled_safety.py',PACKAGE/'scripts/ocs2_wheeled_scene.py']
    source_files+=list((args.robot_dir/'ocs2/include').glob('*.h'))+list((args.robot_dir/'ocs2/src').glob('*.cpp'))
    source_files += [args.robot_dir/'ocs2/backend.json',args.robot_dir/'ocs2/model.json']
    spec=json.loads((args.robot_dir/'ocs2/backend.json').read_text())
    source_files += [args.robot_dir/spec['sensor_config']]
    if (args.robot_dir/'robot.json').is_file():source_files.append(args.robot_dir/'robot.json')
    if args.task_file:source_files.append(args.task_file)
    binary=args.worker_binary or args.robot_dir/'ocs2/.cache/backend/camera_mpc_worker'
    if not binary.is_file():raise FileNotFoundError('Build the actual native OCS2 camera_mpc_worker first')
    binary=binary.resolve();shutil.copy2(binary,frozen/'camera_mpc_worker');args.worker_binary=frozen/'camera_mpc_worker'
    guard_library=binary.with_name('libmeasured_collision_guard.so')
    if not guard_library.is_file():raise FileNotFoundError('Build the native measured_collision_guard shared library')
    shutil.copy2(guard_library,frozen/'libmeasured_collision_guard.so');args.guard_library=frozen/'libmeasured_collision_guard.so'
    snapshot={}
    for i,path in enumerate(source_files):
        dest=frozen/f'{i}_{path.name}';shutil.copy2(path,dest);snapshot[str(path.resolve())]=dict(snapshot=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    pkg=frozen/'python/jev_agent_wbc';pkg.mkdir(parents=True);(pkg/'__init__.py').write_text('')
    for name in ['representation.py','wbc_protocol.py','wbc_physical.py','robot_sensor_rig.py']:
        src=PACKAGE/'jev_agent_wbc'/name;shutil.copy2(src,pkg/name);snapshot[str(src)]=dict(snapshot='python/jev_agent_wbc/'+name,sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    sys.path[:0]=[str(frozen/'python'),str(PACKAGE/'scripts'),str(PACKAGE)]
    m=importlib.util.spec_from_file_location('frozen_ocs2_safety',frozen/'1_ocs2_wheeled_safety.py');safety=importlib.util.module_from_spec(m);m.loader.exec_module(safety);args.safety=safety
    m=importlib.util.spec_from_file_location('frozen_ocs2_scene',frozen/'2_ocs2_wheeled_scene.py');scene=importlib.util.module_from_spec(m);m.loader.exec_module(scene);args.scene=scene
    args.sensor_config=frozen/f'{source_files.index(args.robot_dir/spec["sensor_config"])}_{Path(spec["sensor_config"]).name}'
    model_index=source_files.index(args.robot_dir/'ocs2/model.json')
    args.spec=spec;args.model=json.loads((frozen/f'{model_index}_model.json').read_text())
    profile_index=source_files.index(args.robot_dir/'robot.json') if (args.robot_dir/'robot.json').is_file() else None
    args.profile=json.loads((frozen/f'{profile_index}_robot.json').read_text()) if profile_index is not None else None
    (args.output/'provenance.json').write_text(json.dumps(dict(controller='actual native OCS2 GaussNewtonDDP_MPC/SLQ input-rate augmentation',rate_state_semantics='last_applied_velocity_command',positions_and_camera_source='fresh measured PhysX SE3 and joints',actual_velocity_source='separately recorded unfiltered PhysX velocities',sources=snapshot,worker_sha256=hashlib.sha256(args.worker_binary.read_bytes()).hexdigest(),physics_hz=100,requested_pace=1,old_qp_or_dls_controller_used=False),indent=2)+'\n')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES');os.environ.setdefault('ACCEPT_EULA','Y');sys.argv=[sys.argv[0]]
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=not args.gui,renderer='RaytracedLighting',anti_aliasing=0,multi_gpu=False,limit_cpu_threads=8))
    try:
        return run(args)
    except BaseException:
        import traceback
        (args.output/'failure.txt').write_text(traceback.format_exc())
        if not (args.output/'acceptance.json').is_file():(args.output/'acceptance.json').write_text(json.dumps(dict(passed=False,complete=False,framework='OCS2_SLQ',reason='see failure.txt and streaming measurement/worker logs'),indent=2)+'\n')
        raise
    finally:app.close(wait_for_replicator=False)


def run(a):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdLux
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import SingleArticulation,SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from jev_agent_wbc.robot_sensor_rig import RobotSensorRig,SensorAudit,pose_transform,set_usd_transform
    from jev_agent_wbc.wbc_protocol import CameraTrajectory,TimedTask,Capabilities,semantic_camera,optical_camera
    from jev_agent_wbc.wbc_physical import MeasuredColliderGeometry,LiveTaskSession,audit_stop,pose,rigid
    spec,profile=a.spec,a.profile;omni=spec['base_model']=='omnidirectional';nu=9 if omni else 8
    world=World(physics_dt=.01,rendering_dt=.02,stage_units_in_meters=1.);stage=world.stage;root='/World/Robot'
    model=a.robot_dir/spec['usd'];add_reference_to_stage(str(model),root)
    prims=list(Usd.PrimRange(stage.GetPrimAtPath(root)))
    body_prims={p.GetName():p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if len(body_prims)!=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims):raise ValueError('duplicate physical body names')
    joint_prims=[p for p in prims if p.IsA(UsdPhysics.Joint)]
    joints=[];parents={};arm_children={}
    for p in joint_prims:
        j=UsdPhysics.Joint(p);zero=j.GetBody0Rel().GetTargets();one=j.GetBody1Rel().GetTargets()
        if not zero or not one:continue
        b0=stage.GetPrimAtPath(zero[0]).GetName();b1=stage.GetPrimAtPath(one[0]).GetName()
        joints.append((b0,b1,p.IsA(UsdPhysics.FixedJoint)));parents[b1]=(b0,p.GetName())
        if p.GetName() in spec['arm_joints']:arm_children[b1]=spec['arm_joints'].index(p.GetName())+1
    def prefix(name):
        seen=set();result=0
        while name not in seen:
            seen.add(name);result=max(result,arm_children.get(name,0))
            if name not in parents:break
            name=parents[name][0]
        return result
    manipulator={name for name in body_prims if prefix(name)>0}
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
    collision_prims=[p for p in Usd.PrimRange(stage.GetPrimAtPath(root),Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    ranges=[bbox.ComputeWorldBound(p).ComputeAlignedRange() for p in collision_prims]
    if not ranges or any(b.IsEmpty() for b in ranges):raise ValueError('missing/empty actual collision geometry')
    if omni:
        fixed=UsdPhysics.FixedJoint(stage.GetPrimAtPath(root+'/world/FixedJoint'))
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.,0.,.025930123));fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.,0.,0.,0.))
        base_ranges=[bbox.ComputeWorldBound(p).ComputeAlignedRange() for p in collision_prims if str(p.GetPath()).startswith(root+'/base_link/')]
        floor=min(float(b.GetMin()[2]) for b in base_ranges)-.003
    else:
        transform=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T.copy()
        transform[2,3]+=.003-min(float(b.GetMin()[2]) for b in ranges);set_usd_transform(stage.GetPrimAtPath(root),transform);floor=0.
    material=PhysicsMaterial('/World/GroundMaterial',static_friction=.8 if not omni else .5,dynamic_friction=.6 if not omni else .5,restitution=0.)
    world.scene.add(GroundPlane('/World/Ground',name='ground',z_position=floor,size=40.,physics_material=material))
    ground_report=a.scene.repair_single_ground(stage,'/World/Ground',material.material,floor)
    (a.output/'ground_audit_before_reset.json').write_text(json.dumps(ground_report,indent=2)+'\n')
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(800.)
    articulation_root=root if omni else root+'/'+profile.get('articulation_root','').strip('/')
    articulation=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(articulation_root));articulation.CreateEnabledSelfCollisionsAttr(True)
    # Retain the Husky asset's previously validated 32/16 solver setting and
    # authored/default contact offsets. Ridgeback's planar model uses 64/16.
    articulation.CreateSolverPositionIterationCountAttr(64 if omni else 32);articulation.CreateSolverVelocityIterationCountAttr(16)
    if omni:
        # The stock fixed-height planar asset cannot roll/settle vertically.
        # Restore its previously validated explicit pair offsets: the 3mm floor
        # gap must exceed BOTH 1mm offsets, avoiding a redundant contact joint.
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.CollisionAPI):
                api=PhysxSchema.PhysxCollisionAPI.Apply(p);api.CreateContactOffsetAttr(.001);api.CreateRestOffsetAttr(0.)
    else:
        for p in body_prims.values():PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateMaxDepenetrationVelocityAttr(1.)
    robot=world.scene.add(SingleArticulation(prim_path=articulation_root,name='robot'))
    rig=RobotSensorRig(stage,root,spec['robot_key'],a.sensor_config);sensor_audit=SensorAudit(rig,a.output,a.sensor_audit)
    geometry=MeasuredColliderGeometry.from_stage(stage,root,manipulator,body_prims)
    a.safety.add_convex_vertices(stage,geometry)
    collision=a.safety.CollisionGuard(geometry.shapes,joints)
    # Exact authored body kinematics for the native collision model and for
    # explicitly rigid-fixed bodies merged by PhysX, never live USD substitution.
    xcache=UsdGeom.XformCache();authored={name:np.asarray(xcache.GetLocalToWorldTransform(p)).T.copy() for name,p in body_prims.items()}
    invbase=np.linalg.inv(authored[spec['base_link']]);live_model=dict(a.model)
    live_model.update(body_kinematics=[dict(name=name,prefix=prefix(name),T_base_body=(invbase@T).tolist()) for name,T in authored.items()],collision_shapes=geometry.shapes,collision_pairs=collision.pairs)
    live_model_path=a.output/'sources/live_model.json';live_model_path.write_text(json.dumps(live_model,indent=2)+'\n')
    provenance_path=a.output/'provenance.json';provenance=json.loads(provenance_path.read_text());provenance['frozen_sha256']={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.output/'sources').rglob('*') if p.is_file()};provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    measured_collision=a.safety.NativeCollisionGuard(a.guard_library,live_model_path)
    world.reset()
    (a.output/'ground_audit.json').write_text(json.dumps(a.scene.audit_single_ground(stage,'/World/Ground',material.material,floor),indent=2)+'\n')
    names=robot.dof_names;linear_names={p.GetName() for p in joint_prims if p.IsA(UsdPhysics.PrismaticJoint)};joint_units=['m' if n in linear_names else 'rad' for n in names];view=robot._articulation_view;arm_ids=np.array([names.index(n) for n in spec['arm_joints']])
    direct_names=set(view.body_names)
    link_indices={name:i for i,name in enumerate(view.body_names)}
    rigid_bodies={name:link_indices[name] for name in body_prims if name in direct_names}
    link_com=np.asarray(view.get_body_coms()[0])[0]
    body_sources={}
    for name in body_prims:
        if name in rigid_bodies:body_sources[name]=(name,np.eye(4))
        else:
            candidates=[host for host in rigid_bodies if collision.groups.get(host)==collision.groups.get(name)]
            if not candidates:raise ValueError('no direct PhysX or rigid-fixed measurement source for '+name)
            host=candidates[0];body_sources[name]=(host,np.linalg.inv(authored[host])@authored[name])
    home=np.asarray(robot.get_joint_positions()).copy();home[arm_ids]=spec['initial_arm']
    if omni:
        base_ids=np.array([names.index(n) for n in ['dummy_base_prismatic_x_joint','dummy_base_prismatic_y_joint','dummy_base_revolute_z_joint']]);home[base_ids]=0.
        kp=np.array([20000.,20000.,10000.,6000.,6000.,6000.,2500.,2500.,2500.]);kd=np.array([3000.,3000.,1500.,400.,400.,400.,150.,150.,150.]);effort=np.array([4800.,4800.,4800.,150.,150.,150.,28.,28.,28.]);wheels=np.array([],dtype=int)
    else:
        left=np.array([names.index(n) for n in profile['left_wheels']]);right=np.array([names.index(n) for n in profile['right_wheels']]);wheels=np.r_[left,right]
        kp=np.full(len(names),50.);kd=np.full(len(names),5.);effort=np.full(len(names),30.);kp[wheels]=0.;kd[wheels]=profile.get('wheel_kd',100.);effort[wheels]=profile.get('wheel_effort',100.)
        for i,j in zip(arm_ids,profile['joints']):kp[i],kd[i],effort[i]=j['kp'],j['kd'],j['effort']
    original_velocity_caps=np.asarray(view.get_joint_max_velocities())[0].copy()
    operating_velocity_caps=np.full(len(names),8.);operating_velocity_caps[arm_ids]=.65
    if omni:operating_velocity_caps[base_ids]=[.25,.25,.45]
    operating_velocity_caps=np.minimum(operating_velocity_caps,original_velocity_caps)
    view.set_gains(kps=kp[None],kds=kd[None]);view.set_max_efforts(effort[None]);view.set_max_joint_velocities(operating_velocity_caps[None]);robot.set_joint_positions(home);robot.set_joint_velocities(np.zeros(len(names)))
    runtime_effort_caps=np.asarray(view.get_max_efforts())[0].copy();runtime_velocity_caps=np.asarray(view.get_joint_max_velocities())[0].copy()
    if not np.allclose(runtime_effort_caps,effort,rtol=1e-6) or not np.allclose(runtime_velocity_caps,operating_velocity_caps,rtol=1e-6):raise RuntimeError('physical drive caps do not match configured bounded actuator limits')
    actuator=dict(schema='ocs2_physical_actuator_audit/v1',joint_names=names,joint_units=joint_units,physical_limits=[[float(v) if np.isfinite(v) else None for v in row] for row in np.asarray(view.get_dof_limits())[0]],effort_caps=runtime_effort_caps.tolist(),velocity_caps=runtime_velocity_caps.tolist(),original_velocity_caps=[float(v) if np.isfinite(v) else None for v in original_velocity_caps],effort_readback='PhysX get_max_efforts after set_max_efforts',velocity_readback='PhysX get_joint_max_velocities after conservative operating-cap configuration',source_joint_position_limits_unchanged=True,effort_measurement_scope='projected joint reaction efforts recorded; not asserted identical to implicit drive torque')
    (a.output/'actuator_audit.json').write_text(json.dumps(actuator,indent=2)+'\n')
    from isaacsim.core.simulation_manager import SimulationManager
    SimulationManager.get_physics_sim_view().update_articulations_kinematic()
    control=robot.get_articulation_controller();target=home.copy()
    native_limits=np.asarray(view.get_dof_limits())[0];limits=native_limits[arm_ids];lower=np.array([j['lower'] for j in live_model['joints']]);upper=np.array([j['upper'] for j in live_model['joints']])
    if np.any(lower<limits[:,0]-1e-4) or np.any(upper>limits[:,1]+1e-4):raise ValueError('native joint limits exceed actual physical model')
    guard=a.safety.HardCommandGuard(omni,lower,upper)
    skid_servo=None if omni else a.safety.SkidSteerVelocityServo(profile['wheel_radius_m'],profile['wheel_separation_m'])
    (a.output/'configuration.json').write_text(json.dumps(dict(robot=spec,physical_joint_limits=actuator['physical_limits'],arm_model_limits=limits.tolist(),joint_names=names,joint_units=joint_units,kp=kp.tolist(),kd=kd.tolist(),max_effort=effort.tolist(),floor_height_m=floor,stop_budget_s=5.,measurement_scope='one shared fresh acquisition per 100Hz physics step',skid_mapping=None if omni else dict(type='measured-body-yaw PI to bounded physical wheel velocities',native_reference_yaw_cap_radps=.35,native_reference_yaw_slew_radps2=.65,actual_body_yaw_cap_radps=.45,wheel_cap_radps=8.,wheel_slew_radps2=skid_servo.wheel_slew,virtual_actuator_yaw_cap=4.,virtual_yaw_is_body_reference=False,manual_turn_operating_wheel_cap_radps=1.,manual_turn_brake_enter_radps=.22,manual_turn_brake_release_radps=.12),camera_manipulator_bodies=sorted(manipulator),physical_body_sources={name:dict(host=host,T_host_body=offset.tolist()) for name,(host,offset) in body_sources.items()},collision_geometry=geometry.manifest(),collision_pairs=collision.pairs,collision_scope='conservative actual convex hulls; SAT broad phase; fixed-component and adjacent-joint exclusions',asset_sha256=hashlib.sha256(model.read_bytes()).hexdigest()),indent=2)+'\n')
    worker=a.safety.NativeWorker(a.worker_binary,a.output/'sources'/next(p.name for p in (a.output/'sources').glob('*_backend.json')),live_model_path,a.output/'native_mpc.jsonl')
    rows=[];phases=[];safety_violations=[];timings=[];stop_hold_s=5.;steps=0;measurement_cache=[-1,None];wall=time.monotonic();session=None;active_policy=None;serial=0;last_submit=-1.;recovery_since=None;failed_solutions=0;native_failure=None;native_infeasible=False
    state_stream=(a.output/'samples.jsonl').open('w',buffering=1);feedback_stream=(a.output/'feedback.jsonl').open('w',buffering=1)

    def measured():
        # The solver, guard and logger share the same acquisition for this
        # physics tick. A new world.step always invalidates it through `steps`.
        # No measurement is carried across a physics step.
        if measurement_cache[0]==steps:return measurement_cache[1]
        links=np.asarray(view._physics_view.get_link_transforms())[0]
        rotations=Rotation.from_quat(links[:,3:7]).as_matrix()
        hosts={name:np.block([[rotations[index],links[index,:3,None]],[np.zeros((1,3)),np.ones((1,1))]]) for name,index in rigid_bodies.items()}
        poses={name:hosts[host]@offset for name,(host,offset) in body_sources.items()}
        q=np.asarray(robot.get_joint_positions());qd=np.asarray(robot.get_joint_velocities());base=poses[spec['base_link']];camera=poses[rig.camera_spec['parent']]@rig.camera_local
        host=body_sources[spec['base_link']][0];index=rigid_bodies[host];velocity_tensor=np.asarray(view._physics_view.get_link_velocities())[0,index]
        angular=velocity_tensor[3:].copy();com_world=hosts[host][:3,:3]@link_com[index]+hosts[host][:3,3]
        linear=velocity_tensor[:3]+np.cross(angular,base[:3,3]-com_world)
        yaw=np.arctan2(base[1,0],base[0,0]);c,s=np.cos(yaw),np.sin(yaw);vbody=np.array([c*linear[0]+s*linear[1],-s*linear[0]+c*linear[1]])
        velocity=np.r_[vbody,angular[2],qd[arm_ids]] if omni else np.r_[vbody[0],angular[2],qd[arm_ids]]
        x=np.r_[base[:2,3],yaw,q[arm_ids],velocity]
        result=(q,qd,poses,camera,x,linear,angular);measurement_cache[:]=[steps,result]
        return result

    def step(desired,phase,*,brake=False):
        nonlocal steps
        tick=time.perf_counter()
        q,qd,poses,camera,x,linear,angular=measured()
        before_guard=time.perf_counter()
        if not np.isfinite(x).all() or np.arccos(np.clip(poses[spec['base_link']][2,2],-1,1))>.35:raise RuntimeError('nonfinite/tilted physical robot')
        distance,pair=measured_collision.check(poses)
        if distance<.005:raise RuntimeError('conservative self collision guard rejected '+str((distance,pair)))
        body_unsafe=bool(np.linalg.norm(linear[:2])>.25+1e-5 or abs(angular[2])>.45+1e-5)
        if body_unsafe:safety_violations.append(dict(stamp_s=steps*.01,case=phase,actual_body_speed_ok=False))
        command=guard.command(desired if omni else (np.r_[np.zeros(2),np.asarray(desired)[-6:]] if session is not None and phase=='final_stop' else desired),q[arm_ids],qd[arm_ids],brake=brake or body_unsafe or (not omni and session is not None and phase=='final_stop' and steps*.01>=session.end_s+4.-1e-8))
        execution_brake=bool(brake or body_unsafe)
        if not all(guard.last_audit[k] for k in ['configuration_ok','actual_velocity_ok','actual_braking_ok']):
            safety_violations.append(dict(stamp_s=steps*.01,case=phase,**guard.last_audit))
            target[arm_ids]=q[arm_ids];execution_brake=True
        velocities=np.zeros(len(names));velocities[arm_ids]=command[-6:]
        target[arm_ids]=np.clip(target[arm_ids]+command[-6:]*.01,np.maximum(lower+.002,q[arm_ids]-.04),np.minimum(upper-.002,q[arm_ids]+.04))
        if omni:
            yaw=x[2];c,s=np.cos(yaw),np.sin(yaw);velocities[base_ids]=[c*command[0]-s*command[1],s*command[0]+c*command[1],command[2]];target[base_ids]+=velocities[base_ids]*.01
            target[base_ids]=np.clip(target[base_ids],q[base_ids]-.04,q[base_ids]+.04)
            control.apply_action(ArticulationAction(joint_positions=target.copy(),joint_velocities=velocities))
        else:
            left_target,right_target=skid_servo.command(command[:2],angular[2],brake=execution_brake or (session is not None and phase=='final_stop'),turn_in_place=phase.endswith('_turn'))
            velocities[left]=left_target;velocities[right]=right_target
            for i in wheels:velocities[i]*=profile.get('wheel_signs',{}).get(names[i],1.)
            hold=np.array([i for i in range(len(names)) if i not in wheels]);control.apply_action(ArticulationAction(joint_positions=target[hold],joint_velocities=velocities[hold],joint_indices=hold));control.apply_action(ArticulationAction(joint_velocities=velocities[wheels],joint_indices=wheels))
        before_physics=time.perf_counter();world.step(render=False);steps+=1;after_physics=time.perf_counter()
        q,qd,poses,camera,x,linear,angular=measured();stamp=steps*.01
        after_measurement=time.perf_counter()
        yaw=x[2];measured_body=[np.cos(yaw)*linear[0]+np.sin(yaw)*linear[1],-np.sin(yaw)*linear[0]+np.cos(yaw)*linear[1],angular[2]]
        requested_body=list(np.asarray(desired)[:3]) if omni else [float(desired[0]),0.,float(desired[1])]
        commanded_body=command[:3].tolist() if omni else [float(command[0]),0.,float(command[1])]
        projected_effort=np.asarray(view.get_measured_joint_efforts())[0]
        config_ok=bool(np.all(q>=native_limits[:,0]) and np.all(q<=native_limits[:,1]) and np.all(q[arm_ids]>=lower) and np.all(q[arm_ids]<=upper))
        actual_speed_ratio=float(np.max(np.abs(qd)/runtime_velocity_caps))
        body_speed_ok=bool(np.linalg.norm(linear[:2])<=(.25 if not omni else .25)+1e-5 and abs(angular[2])<=.45+1e-5)
        if not config_ok or actual_speed_ratio>1.00001 or not body_speed_ok:safety_violations.append(dict(stamp_s=stamp,case=phase,configuration_ok=config_ok,maximum_actual_velocity_ratio=actual_speed_ratio,actual_body_speed_ok=body_speed_ok))
        record=dict(measured_configuration_ok=config_ok,maximum_actual_velocity_ratio=actual_speed_ratio,measured_arm_guard=guard.last_audit,joint_projected_efforts=projected_effort.tolist(),applied_user_efforts=np.asarray(view.get_applied_joint_efforts())[0].tolist(),t_s=stamp,stamp_s=stamp,case=phase,phase='stop' if phase in ['settle','stop','pre_wbc'] or phase.endswith('_stop') else 'move',phase_label=phase,q=q.tolist(),qd=qd.tolist(),joint_positions=q.tolist(),joint_names=names,joint_units=joint_units,
                    requested_body_velocity=requested_body,commanded_body_velocity=commanded_body,measured_body_velocity=list(map(float,measured_body)),base_linear_velocity_world=linear.tolist(),base_angular_velocity_world=angular.tolist(),T_world_camera_optical=camera.tolist(),T_world_base=poses[spec['base_link']].tolist(),T_world_camera=camera.tolist(),base_linear_velocity=linear.tolist(),base_angular_velocity=angular.tolist(),command=command.tolist(),self_separation_lower_bound_m=distance,self_pair=pair)
        if skid_servo is not None:record['skid_actuator_mapping']=skid_servo.last_audit
        if session is not None:
            observation=geometry.observe(stamp,poses[spec['base_link']],camera,poses)
            current_distance,current_pair=measured_collision.check(poses)
            constraints={
                'configuration':dict(status='pass' if config_ok else 'fail',stamp_s=stamp,scope='measured_native_limits'),
                'actuation':dict(status='fail' if actual_speed_ratio>1.00001 else 'unknown',stamp_s=stamp,scope='measured_native_limits' if actual_speed_ratio>1.00001 else 'not_available'),
                'self_collision':dict(status='pass' if current_distance>=.005 else 'fail',stamp_s=stamp,scope='measured_collision_model',clearance_m=current_distance),
                'environment':dict(status='unknown',stamp_s=stamp,scope='not_available'),
                'policy_domain':dict(status='not_applicable',stamp_s=None,scope='not_applicable')}
            result=session.feedback(observation,feasible=False if native_infeasible else None,reason=native_failure or 'not_evaluated',hold_terminal=stamp>session.end_s,constraints=constraints,now_s=stamp)
            feedback_stream.write(json.dumps(result,allow_nan=False)+'\n');record['task_feedback']=result;record['T_world_reference_optical']=session.reference(min(stamp,session.end_s))[0].tolist();record.update(position_error_m=result['position_error_m'],orientation_error_rad=result['orientation_error_rad'],camera_within_tolerance=result['camera_within_tolerance'])
        after_feedback=time.perf_counter();rows.append(record);state_stream.write(json.dumps(record,allow_nan=False)+'\n');after_logging=time.perf_counter()
        if a.sensor_audit:sensor_audit.record(stamp,{name:poses[name] for name in rig.parent_names},phase)
        timings.append(dict(stamp_s=stamp,case=phase,measure_before_s=before_guard-tick,guard_and_command_s=before_physics-before_guard,physics_s=after_physics-before_physics,measure_after_s=after_measurement-after_physics,feedback_and_geometry_s=after_feedback-after_measurement,logging_s=after_logging-after_feedback,total_before_sleep_s=time.perf_counter()-tick,wall_lag_s=time.monotonic()-(wall+stamp)))
        remaining=wall+stamp-time.monotonic()
        if remaining>0:time.sleep(remaining)
        return q,qd,poses,camera,x,linear,angular

    def hold(seconds,phase):
        for _ in range(round(seconds/.01)):state=step(np.zeros(nu),phase,brake=True)
        return state

    def strict_hold(label):
        hold(stop_hold_s,label)
        selected=[r for r in rows if r['case']==label]
        report=audit_stop([r['t_s'] for r in selected],[r['T_world_base'] for r in selected],[r['T_world_camera'] for r in selected],[r['q'] for r in selected],joint_units,selected[-1]['base_linear_velocity'],selected[-1]['base_angular_velocity'])
        phases.append(dict(phase=label,**report))
        return report

    try:
        q,qd,poses,camera,x,linear,angular=hold(3.,'settle')
        initial_observation=geometry.observe(steps*.01,poses[spec['base_link']],camera,poses)
        (a.output/'initial_measured_observation.json').write_text(json.dumps(initial_observation.to_dict(),indent=2)+'\n')
        # FK is tied to physical measurements, including full base height/tilt.
        from scipy.linalg import expm
        predicted=np.eye(4)
        for value,j in zip(q[arm_ids],live_model['joints']):
            omega=np.asarray(j['screw_omega']);v=np.asarray(j['screw_v']);hat=np.zeros((4,4));hat[:3,:3]=[[0,-omega[2],omega[1]],[omega[2],0,-omega[0]],[-omega[1],omega[0],0]];hat[:3,3]=v;predicted=predicted@expm(hat*(value-j['q_at_authored_pose']))
        predicted=poses[spec['base_link']]@predicted@np.asarray(live_model['T_base_ee_at_authored_pose'])@np.asarray(live_model['T_ee_camera_optical'])
        fk_error=float(np.linalg.norm(predicted[:3,3]-camera[:3,3]));fk_angle=float(Rotation.from_matrix(predicted[:3,:3]@camera[:3,:3].T).magnitude())
        if fk_error>.003 or fk_angle>.01:raise RuntimeError(f'USD/native optical FK differs from physical camera {fk_error}m/{fk_angle}rad')
        if a.probe:
            start=poses[spec['base_link']][:3,3].copy()
            for i in range(200):
                u=np.zeros(nu);u[0]=.15*min(1.,i/50.,(199-i)/50.);state=step(u,'forward_probe')
            hold(3.,'probe_stop');delta=state[2][spec['base_link']][:3,3]-start;phases.append(dict(phase='forward_probe',displacement_m=delta.tolist(),passed=bool(delta[0]>.12)))
        else:
            if not a.wbc_only:
                if omni:
                    for label,vector in [('forward',[.15,0,0]),('backward',[-.15,0,0]),('left',[0,.15,0]),('right',[0,-.15,0]),('diagonal_ne',[.1,.1,0]),('diagonal_sw',[-.1,-.1,0]),('yaw_ccw',[0,0,.2]),('yaw_cw',[0,0,-.2])]:
                        before=measured()[0][base_ids].copy()
                        for i in range(500):u=np.r_[np.asarray(vector)*min(1.,(i+1)/50.,(500-i)/50.),np.zeros(6)];state=step(u,label)
                        delta=state[0][base_ids]-before;expected=np.asarray(vector)*4.51;error=delta-expected;tail_linear=float(np.linalg.norm(state[1][base_ids][:2]));tail_yaw=float(abs(state[1][base_ids][2]));phases.append(dict(phase=label,displacement=delta.tolist(),translation_error_m=float(np.linalg.norm(error[:2])),yaw_error_rad=float(abs(error[2])),final_linear_speed_mps=tail_linear,final_yaw_rate_radps=tail_yaw,passed=bool(np.linalg.norm(error[:2])<.045 and abs(error[2])<.045 and tail_linear<.06 and tail_yaw<.06)));strict_hold(label+'_stop')
                else:
                    for label,angle,sign in [('forward',0.,1.),('backward',0.,-1.),('left',np.pi/2,1.),('right',-np.pi/2,1.),('diagonal_ne',np.pi/4,1.),('diagonal_sw',-3*np.pi/4,1.)]:
                        integral=0.
                        for i in range(6000):
                            yaw=measured()[4][2];error=np.arctan2(np.sin(angle-yaw),np.cos(angle-yaw));integral=np.clip(integral+error*.01,-1.2,1.2);u=np.zeros(nu);u[1]=np.clip(2*error+.5*integral,-.45,.45);state=step(u,label+'_turn')
                            if abs(error)<.015 and i>50:break
                        if abs(error)>.04:raise RuntimeError('physical wheel turn did not reach required heading')
                        before=measured()[2][spec['base_link']][:3,3].copy()
                        for i in range(300):
                            yaw=measured()[4][2];error=np.arctan2(np.sin(angle-yaw),np.cos(angle-yaw));u=np.zeros(nu);u[:2]=[sign*.15*min(1.,i/50.,(299-i)/50.),np.clip(error,-.15,.15)];state=step(u,label)
                        delta=state[2][spec['base_link']][:3,3]-before;along=delta[:2]@np.array([np.cos(angle),np.sin(angle)])*sign;across=abs(delta[:2]@np.array([-np.sin(angle),np.cos(angle)]));phases.append(dict(phase=label,along_m=float(along),lateral_error_m=float(across),passed=bool(along>.25 and across<.08)));strict_hold(label+'_stop')
                before=measured();arm_start=before[0][arm_ids].copy();amplitude=np.minimum(.12,np.minimum(arm_start-lower,upper-arm_start)*.4);arm_samples=[]
                for i in range(800):u=np.r_[np.zeros(nu-6),amplitude*(2*np.pi/8)*np.cos(2*np.pi*i*.01/8)];state=step(u,'independent_arm');arm_samples.append(state[0][arm_ids].copy())
                drift=float(np.linalg.norm(state[2][spec['base_link']][:2,3]-before[2][spec['base_link']][:2,3]));span=np.ptp(arm_samples,axis=0);phases.append(dict(phase='independent_arm',base_drift_m=drift,arm_span_rad=span.tolist(),passed=bool(drift<.03 and np.all(span>amplitude*.5))));strict_hold('arm_stop')
            q,qd,poses,camera,x,linear,angular=measured();stamp=steps*.01;observation=geometry.observe(stamp,poses[spec['base_link']],camera,poses)
            if a.task_file:
                task=TimedTask.from_dict(json.loads(a.task_file.read_text()))
            else:
                times=np.arange(0,21.0001,.05);references=[]
                for t in times:
                    phase=np.clip(t-1.,0.,20.);w=2*np.pi/20;T=camera.copy();T[:3,3]+=np.array([.28*np.sin(w*phase),.20*(1-np.cos(w*phase)),.07*np.sin(2*w*phase)]);T[:3,:3]=Rotation.from_rotvec([.12*np.sin(w*phase),.10*np.sin(2*w*phase),.16*np.sin(w*phase)]).as_matrix()@camera[:3,:3];references.append(semantic_camera(pose(T)))
                task=TimedTask('ocs2_physical_camera',stamp,observation.id,CameraTrajectory(stamp,tuple(times),tuple(references)),.03 if not omni else .035,.08 if not omni else .10)
            session=LiveTaskSession(Capabilities('actual OCS2 SLQ input-rate augmentation',True,frozenset({'timed_camera_se3'})),.1)
            admitted=session.accept(task,observation,stamp);(a.output/'task.json').write_text(json.dumps(task.to_dict(),indent=2)+'\n');(a.output/'timed_tasks.json').write_text(json.dumps([task.to_dict()],indent=2)+'\n');(a.output/'task_ingress.json').write_text(json.dumps(dict(observation=observation.to_dict(),negotiation=admitted),indent=2)+'\n')
            (a.output/'measured_observation.json').write_text(json.dumps(observation.to_dict(),indent=2)+'\n')
            (a.output/'accepted_observations.json').write_text(json.dumps([observation.to_dict()],indent=2)+'\n')
            if not admitted['supported']:raise RuntimeError('public TimedTask rejected '+str(admitted))
            optical=[rigid(optical_camera(p)) for p in task.trajectory.poses];optical_serialized=[T.tolist() for T in optical];absolute=np.array([task.trajectory.start_s+t for t in task.trajectory.times_s]);wbc_start=len(rows)
            if omni:
                while steps*.01<task.trajectory.end_s-1e-8:
                    stamp=steps*.01;q,qd,poses,camera,x,linear,angular=measured();answer=worker.poll()
                    if answer is not None:
                        if answer.get('ok') is True and stamp-answer['state_stamp_s']<=.2 and answer.get('transport_wall_s',1e9)<=.2:
                            active_policy=answer;failed_solutions=0;native_failure=None
                        else:
                            failed_solutions+=1;active_policy=None;text=str(answer).lower()
                            native_failure='self_collision' if 'collision' in text else ('actuation_limit' if 'velocity/acceleration' in text else ('configuration_limit' if 'joint limit' in text else 'solver_failure'))
                    horizon=min(1.,task.trajectory.end_s-stamp)
                    if not worker.busy and stamp-last_submit>=.05-1e-8 and horizon>=.2-1e-8:
                        model_state=np.r_[x[:9],guard.previous]
                        reference_times,reference_poses,indices=a.safety.reference_window(absolute,optical_serialized,stamp,horizon)
                        serial+=1;worker.submit(dict(request_id=serial,state_stamp_s=stamp,horizon_s=horizon,state=model_state.tolist(),rate_state_semantics='last_applied_velocity_command',measured_velocity=x[9:].tolist(),T_world_base=poses[spec['base_link']].tolist(),frame='world_optical',camera_path_id=session.camera_path_id,original_reference_slice=list(indices),absolute_times_s=reference_times,camera_poses=reference_poses));last_submit=stamp
                    try:
                        if active_policy is None:raise ValueError('no fresh native policy')
                        u=a.safety.policy_velocity(active_policy,stamp,nu,ttl_s=.2);recovery_since=None
                    except ValueError:
                        u=np.zeros(nu)
                        if recovery_since is None:recovery_since=stamp
                        if stamp-recovery_since>2. or failed_solutions>=5:
                            native_infeasible=True
                            report=strict_hold('solver_recovery_failure_stop');(a.output/'failure_stop.json').write_text(json.dumps(report,indent=2)+'\n')
                            raise RuntimeError('bounded native-solver recovery exhausted while physics kept advancing')
                    state=step(u,'ocs2_camera',brake=active_policy is None)
            else:
                while steps*.01<task.trajectory.end_s+stop_hold_s-1e-8:
                    stamp=steps*.01;q,qd,poses,camera,x,linear,angular=measured();answer=worker.poll()
                    if answer is not None:
                        if answer.get('ok') is True and stamp-answer['state_stamp_s']<=.2 and answer.get('transport_wall_s',1e9)<=.2:
                            active_policy=answer;failed_solutions=0;native_failure=None
                        else:
                            failed_solutions+=1;active_policy=None;text=str(answer).lower()
                            native_failure='self_collision' if 'collision' in text else ('actuation_limit' if 'velocity/acceleration' in text else ('configuration_limit' if 'joint limit' in text else 'solver_failure'))
                    horizon=1.
                    if not worker.busy and stamp-last_submit>=.05-1e-8 and horizon>=.2-1e-8:
                        model_state=np.r_[x[:9],guard.previous]
                        reference_times,reference_poses,indices,terminal_hold=a.safety.terminal_reference_window(absolute,optical_serialized,stamp,horizon)
                        terminal_hold['chassis_brake_applied']=bool(stamp>=task.trajectory.end_s-1e-8)
                        terminal_hold['all_actuators_brake_applied']=bool(stamp>=task.trajectory.end_s+4.-1e-8)
                        terminal_hold['all_actuator_brake_offset_s']=4.
                        serial+=1;worker.submit(dict(request_id=serial,state_stamp_s=stamp,horizon_s=horizon,state=model_state.tolist(),rate_state_semantics='last_applied_velocity_command',measured_velocity=x[9:].tolist(),T_world_base=poses[spec['base_link']].tolist(),frame='world_optical',camera_path_id=session.camera_path_id,original_reference_slice=list(indices),absolute_times_s=reference_times,camera_poses=reference_poses,terminal_hold=terminal_hold));last_submit=stamp
                    try:
                        if active_policy is None:raise ValueError('no fresh native policy')
                        u=a.safety.policy_velocity(active_policy,stamp,nu,ttl_s=.2);recovery_since=None
                    except ValueError:
                        u=np.zeros(nu)
                        if recovery_since is None:recovery_since=stamp
                        if stamp-recovery_since>2. or failed_solutions>=5:
                            native_infeasible=True
                            report=strict_hold('solver_recovery_failure_stop');(a.output/'failure_stop.json').write_text(json.dumps(report,indent=2)+'\n')
                            raise RuntimeError('bounded native-solver recovery exhausted while physics kept advancing')
                    state=step(u,'ocs2_camera' if stamp<task.trajectory.end_s-1e-8 else 'final_stop',brake=active_policy is None)
            wrows=rows[wbc_start:]
            if not omni:wrows=[r for r in wrows if r['case']=='ocs2_camera']
            position=np.array([r['position_error_m'] for r in wrows]);orientation=np.array([r['orientation_error_rad'] for r in wrows]);base_span=float(np.linalg.norm(np.ptp(np.array([r['T_world_base'] for r in wrows])[:,:2,3],axis=0)));arm_span=float(np.linalg.norm(np.ptp(np.array([r['q'] for r in wrows])[:,arm_ids],axis=0)))
            tracking=bool(np.sqrt(np.mean(position**2))<task.position_tolerance_m and np.sqrt(np.mean(orientation**2))<task.orientation_tolerance_rad and base_span>.01 and arm_span>.03)
            if omni:tracking=bool(position.max()<task.position_tolerance_m and orientation.max()<task.orientation_tolerance_rad and base_span>.04 and arm_span>.08)
            phases.append(dict(phase='ocs2_camera',framework='actual native OCS2 SLQ',position_rmse_m=float(np.sqrt(np.mean(position**2))),position_max_m=float(position.max()),orientation_rmse_rad=float(np.sqrt(np.mean(orientation**2))),orientation_max_rad=float(orientation.max()),base_span_m=base_span,arm_span_norm_rad=arm_span,native_requests=serial,passed=tracking))
            if not omni:
                selected=[r for r in rows if r['case']=='final_stop']
                report=audit_stop([r['t_s'] for r in selected],[r['T_world_base'] for r in selected],[r['T_world_camera'] for r in selected],[r['q'] for r in selected],joint_units,selected[-1]['base_linear_velocity'],selected[-1]['base_angular_velocity'])
                phases.append(dict(phase='final_stop',**report))
        if omni or a.probe:strict_hold('final_stop')
        q,qd,poses,camera,x,linear,angular=measured();final_observation=geometry.observe(steps*.01,poses[spec['base_link']],camera,poses);(a.output/'final_measured_observation.json').write_text(json.dumps(final_observation.to_dict(),indent=2)+'\n')
        elapsed=time.monotonic()-wall
        if session is not None:
            metrics=a.safety.camera_task_metrics(position,orientation,rows[-1]['position_error_m'],rows[-1]['orientation_error_rad'],task.position_tolerance_m,task.orientation_tolerance_rad)
            phases.append(dict(phase='original_task_final_tolerance',**metrics,passed=metrics['final_task_tolerance_pass']))
        sensors=sensor_audit.finish(steps*.01,world.render if a.sensor_rgb else None)
        stage.Flatten().Export(str(a.output/'tested_scene.usda'))
        (a.output/'safety_violations.json').write_text(json.dumps(safety_violations,indent=2)+'\n')
        result=dict(schema='ocs2_wheeled_physical_acceptance/v1',framework='actual native OCS2 SLQ',robot=spec['robot'],complete=True,diagnostic_only=bool(a.probe or a.wbc_only),passed=bool(all(p['passed'] for p in phases) and not safety_violations and elapsed<=steps*.01*1.05),stop_hold_s=stop_hold_s,measured_safety_pass=not safety_violations,measured_safety_violations=len(safety_violations),phases=phases,simulation_s=steps*.01,wall_s=elapsed,timing_pass=bool(elapsed<=steps*.01*1.05),optical_fk_position_error_m=fk_error,optical_fk_orientation_error_rad=fk_angle,representation_ingress_verified=bool(session),sensors=sensors,avoidance_verified=False)
        (a.output/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print('ACCEPTANCE',json.dumps(result),flush=True);return 0 if result['passed'] else 2
    finally:
        # A failed run cannot leave a velocity command latched during shutdown.
        (a.output/'safety_violations.json').write_text(json.dumps(safety_violations,indent=2)+'\n')
        (a.output/'timing_breakdown.json').write_text(json.dumps(dict(scope='physical loop wall-clock measurements; solver transport remains in native_mpc.jsonl',stop_hold_s=stop_hold_s,rows=timings),indent=2)+'\n')
        measured_q=np.asarray(robot.get_joint_positions());control.apply_action(ArticulationAction(joint_positions=measured_q,joint_velocities=np.zeros(len(names))))
        worker.close();measured_collision.close();state_stream.close();feedback_stream.close()


if __name__=='__main__':
    if '--child' in sys.argv or '--help' in sys.argv or '-h' in sys.argv:raise SystemExit(main())
    import subprocess
    original=sys.argv[1:];output=Path(original[original.index('--output')+1])
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),*original,'--child'])
    result=json.loads((output/'acceptance.json').read_text()) if (output/'acceptance.json').is_file() else {}
    raise SystemExit(0 if child.returncode==0 and result.get('complete') is True and result.get('passed') is True else 2)
