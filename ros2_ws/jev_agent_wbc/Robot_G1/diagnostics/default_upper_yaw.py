#!/usr/bin/env python3
"""Diagnostic only: native AGILE yaw with bounded source-default upper PD.

Never a Pink camera/WBC acceptance. Original timed camera Tasks are read-only
error references, and formal readiness gates are never changed by this script.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import shutil, sys
from types import SimpleNamespace
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
BASELINE='adaptive_ground_sameprofile_01'
BASELINE_RUNNER_SHA='85bfd95d4c629ea590aaf89cfaa04135cf47ce2ac84e0df8246fc31267404d2c'
FORMAL_GATES=('backend_runtime_ready','continuous_camera_tracking_verified','representation_ingress_verified')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def bounded_home_reference(measured,home,limits,max_lead_rad=.05):
    measured,home,limits=map(lambda x:np.asarray(x,float),(measured,home,limits))
    if (measured.ndim!=1 or home.shape!=measured.shape or limits.shape!=(len(home),2)
            or not all(np.isfinite(x).all() for x in (measured,home,limits))
            or np.any(limits[:,0]>=limits[:,1]) or np.any(home<limits[:,0]) or np.any(home>limits[:,1])
            or not np.isfinite(max_lead_rad) or max_lead_rad<=0):
        raise ValueError('finite source-default goal within original bounds required')
    low=np.maximum(limits[:,0],measured-max_lead_rad);high=np.minimum(limits[:,1],measured+max_lead_rad)
    if np.any(low>high):raise ValueError('measured upper state outside original physical bounds')
    target=np.clip(home,low,high)
    return target,dict(diagnostic_only=True,source_default_goal=home.tolist(),maximum_home_error_rad=float(np.max(abs(home-measured))),
        maximum_reference_lead_rad=float(np.max(abs(target-measured))),max_lead_rad=float(max_lead_rad),
        dynamic_pink_frame_task_control=False,reference_kind='source_default_goal_bounded_by_original_reference_lead')


class DefaultUpperDiagnosticSession:
    """Validate/freeze a reference for logging, while explicitly refusing control."""
    def __init__(self,capabilities,max_observation_age_s=.1):
        if capabilities.runtime_ready:raise ValueError('diagnostic cannot advertise camera runtime ready')
        self.max_age=float(max_observation_age_s);self._task=None;self._last=-1.
    def accept(self,task,observation,now_s):
        from jev_agent_wbc.wbc_protocol import TimedTask
        from jev_agent_wbc.representation import Observation,number
        now=number(now_s,'now_s',0.);candidate=TimedTask.from_dict(task.to_dict());obs=Observation.from_dict(observation.to_dict())
        if (obs.id!=candidate.observation_id or obs.source!='measured_fk' or not 0<=now-obs.stamp_s<=self.max_age
                or candidate.issued_at_s>now or now>candidate.trajectory.end_s):
            raise ValueError('fresh measured observation and original Task binding required')
        self._task=candidate;self._last=obs.stamp_s
        return dict(supported=False,feasible=None,reason='unsupported_request',diagnostic_reference_only=True,
                    camera_control_supported=False,task_id=candidate.task_id,camera_path_id=candidate.trajectory.id)
    def reference(self,stamp_s):
        from jev_agent_wbc.wbc_protocol import optical_camera
        from jev_agent_wbc.wbc_physical import rigid
        if self._task is None:raise ValueError('no frozen diagnostic reference')
        pose,twist=self._task.trajectory.sample_with_twist(stamp_s)
        return rigid(optical_camera(pose)),twist.copy()
    def feedback(self,observation,*,feasible=None,reason=None,hold_terminal=False,constraints=None,now_s=None,**kwargs):
        from jev_agent_wbc.wbc_protocol import measured_feedback
        from jev_agent_wbc.wbc_physical import constraint_feedback
        if self._task is None:raise ValueError('no frozen diagnostic reference')
        now=observation.stamp_s if now_s is None else float(now_s)
        if observation.stamp_s<=self._last or not 0<=now-observation.stamp_s<=self.max_age:
            raise ValueError('strictly increasing fresh diagnostic observations required')
        result=measured_feedback(self._task,observation,feasible=None,reason='unsupported_request',hold_terminal=hold_terminal)
        result['constraints']=constraint_feedback(constraints,now,self.max_age)
        result.update(supported=False,diagnostic_only=True,camera_control_supported=False)
        self._last=observation.stamp_s;return result


def save_startup_physical(robot,stage,output,source_usd):
    """Read-only native tensors before timed physics; no inference of merged frames."""
    import inspect
    view=robot.root_physx_view
    # Read the installed pose cache once: it refreshes articulation FK after
    # initialization-only joint writes without advancing physical time.
    robot.data.body_link_pose_w
    def tensor(value):
        value=value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
        return dict(shape=list(value.shape),dtype=str(value.dtype),values=value.tolist())
    api=Path(inspect.getfile(type(view)));layers=[]
    for layer in stage.GetUsedLayers():
        path=Path(layer.realPath) if layer.realPath else None
        layers.append(dict(identifier=layer.identifier,real_path=str(path) if path else None,
            sha256=digest(path) if path and path.is_file() else hashlib.sha256(layer.ExportToString().encode()).hexdigest(),
            hash_kind='file_bytes' if path and path.is_file() else 'layer_export_text'))
    result=dict(schema='g1_default_upper_startup_physx/v1',diagnostic_only=True,stamp_s=0.,before_timed_loop=True,
        body_names=list(robot.body_names),joint_names=list(robot.joint_names),
        masses=tensor(view.get_masses()),coms=tensor(view.get_coms()),inertias=tensor(view.get_inertias()),
        actor_world_poses_xyzw=tensor(view.get_link_transforms()),joint_positions=tensor(robot.data.joint_pos),
        source_default_joint_positions=tensor(robot.data.default_joint_pos),
        source_usd=dict(path=str(source_usd),sha256=digest(source_usd)),layers=layers,
        conventions=dict(masses_axes='articulation, body_names; kg',
            coms_axes='articulation, body_names, xyz/qx/qy/qz/qw; COM principal-axes frame relative to rigid-body-prim actor frame',
            inertias_axes='articulation, body_names,9 column-major components; tensor at COM expressed in rigid-body-prim actor frame, not assumed principal diagonal',
            actor_poses_axes='articulation, body_names,xyz/qx/qy/qz/qw; native global frame (World scene)',
            comparison_warning='Body fusion and COM principal axes must be resolved before comparing USD/URDF tensors; direct json equality is not a mass-model equivalence proof'),
        native_tensor_api_source=dict(path=str(api),sha256=digest(api)))
    try:
        jac=view.get_jacobians();shape=list(jac.shape);ndof=len(robot.joint_names)
        result['dense_link_jacobian']=tensor(jac)
        result['dense_link_jacobian_convention']=dict(frame='global link velocity per installed native tensor API',
            rows='body block,6 velocity components; raw order retained',
            columns=('6 floating-root global-velocity columns then joint_names DOFs' if shape[-1]==ndof+6 else 'joint_names DOFs' if shape[-1]==ndof else 'unresolved'),
            linear_angular_order='not verified by finite differences in this diagnostic',
            point_of_velocity='COM versus actor origin not independently verified; do not label this a verified COM Jacobian',
            used_for_gravity_feedforward=False,finite_difference_verified=False)
    except Exception as exc:result['dense_link_jacobian_unknown']=repr(exc)
    (Path(output)/'startup_physx_mass_properties.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('frozen source anchor absent or ambiguous: '+old[:70])
    return source.replace(old,new)


def candidate_sources(runner,backend):
    runner=replace_once(runner,"    start_wall=time.monotonic(); case_index=-1;", "    save_startup_physical(robot,stage,a.output,cfg.spawn.usd_path)\n    start_wall=time.monotonic(); case_index=-1;")
    runner=replace_once(runner,"session=LiveTaskSession(Capabilities(f'{a.robot}_native_adaptive',True,frozenset(['timed_camera_se3'])))", "session=DefaultUpperDiagnosticSession(Capabilities('g1_default_upper_diagnostic',False,frozenset()))")
    runner=replace_once(runner,"if not admission['supported']:","if not admission.get('diagnostic_reference_only',False):")
    runner=replace_once(runner,"                    reference,upper_reference_report=upper_reference.update(robot.data.joint_pos[0,upper_ids].numpy(),upper_v,dt)","                    reference,upper_reference_report=bounded_home_reference(robot.data.joint_pos[0,upper_ids].numpy(),robot.data.default_joint_pos[0,upper_ids].numpy(),robot.data.joint_pos_limits[0,upper_ids].numpy())")
    runner=replace_once(runner,"                    desired_camera if session else None,stamp,camera,dt,measured_bodies=bodies if a.fine_collision else None,", "                    None,stamp,camera,dt,measured_bodies=bodies if a.fine_collision else None,")
    runner=replace_once(runner,"    geometry = MeasuredColliderGeometry.from_stage(stage,'/World/Robot',manipulator_bodies,robot.body_names)","    geometry = MeasuredColliderGeometry.from_stage(stage,'/World/Robot',manipulator_bodies,robot.body_names)\n    if len(geometry.shapes)!=35:raise RuntimeError('diagnostic requires all35 original collision shapes')")
    runner=replace_once(runner,"    if a.rgb:result['camera_render']=rig.capture_camera(a.output,sim.render,len(rows)*dt)","    result.update(diagnostic_only=True,all_pass=False,wbc_acceptance_pass=False,camera_control_supported=False,public_timed_task_actual_ingress=False,physical_policy='native AGILE height80/12; bounded source-default upper PD plus original Pin gravity only',diagnostic_yaw_velocity_pass=bool(len(results)==2 and all(r['velocity_tracking_pass'] for r in results)),dynamic_pink_frame_task_control=False,formal_readiness_gates_unchanged=True)\n    if a.rgb:result['camera_render']=rig.capture_camera(a.output,sim.render,len(rows)*dt)")
    runner=replace_once(runner,"    return 0 if result['all_pass'] else 2", "    return 0 if result['run_complete'] and not failures and result['timing_pass'] else 2")
    backend=replace_once(backend,"                    result['pink_collision_report']=solver.last_collision_report","                    result['pink_collision_report']=None if request['target'] is None else solver.last_collision_report")
    compile(runner,'diagnostic_runner.py','exec');compile(backend,'gravity_only_backend.py','exec')
    return runner,backend


def verify_false_gates():
    d=json.loads((PACKAGE/'Robot_G1/WBC_READINESS.json').read_text())
    if any(d.get(k) is not False for k in FORMAL_GATES):raise ValueError('diagnostic must not upgrade or overwrite formal readiness')
    return {k:False for k in FORMAL_GATES}


def prepare(bundle):
    baseline=PACKAGE/'Robot_G1/results'/BASELINE;source=baseline/'sources'
    if digest(source/'adaptive_runner.py')!=BASELINE_RUNNER_SHA:raise ValueError('immutable baseline runner changed')
    gates=verify_false_gates();bundle.mkdir(parents=True,exist_ok=False);shutil.copytree(source,bundle/'sources',ignore=shutil.ignore_patterns('__pycache__'))
    runner,backend=candidate_sources((source/'adaptive_runner.py').read_text(),(source/'g1_backend.py').read_text())
    (bundle/'candidate_runner.py').write_text(runner);(bundle/'candidate_g1_backend.py').write_text(backend);shutil.copy2(Path(__file__),bundle/'default_upper_yaw.py')
    args=json.loads((baseline/'provenance.json').read_text())['arguments'];args.update(cases='yaw_left,yaw_right',move_seconds=5.,stop_seconds=5.,yaw_speed=.12,g1_velocity_profile='baseline',ground_material='source_multiply',rgb=False)
    for k in ('output','root','frozen','worker'):args.pop(k,None)
    contract=dict(schema='g1_default_upper_yaw_predeclared/v1',diagnostic_only=True,baseline=BASELINE,arguments=args,simulation_s=22.,policy_hz=50,physics_hz=200,physics_steps_per_policy=4,pacing='pace1; originalwall<=1.05sim',height_command_m=.72,default_goal_max_reference_lead_rad=.05,upper_control='original explicit bounded PD plus same fresh Pin gravity; source-default goal; no dynamic FrameTask',camera='original WORLD Task only reference/error logging; unsupported for control; overall WBCpass alwaysfalse',collisions='all35 original shapes, all original FCL pairs and caps retained',startup='read-only mass/COM/inertia/actorT/q/DOF/source-layerSHA/rawJac before timer; no FF replacement',formal_gates=gates,source_sha256={})
    contract['source_sha256']={str(p.relative_to(bundle)):digest(p) for p in bundle.rglob('*') if p.is_file()}
    (bundle/'declaration.json').write_text(json.dumps(contract,indent=2)+'\n');print(bundle/'declaration.json')


def run_prepared(bundle,output,rgb=False):
    declaration=json.loads((bundle/'declaration.json').read_text());verify_false_gates()
    for rel,expected in declaration['source_sha256'].items():
        if digest(bundle/rel)!=expected:raise ValueError('prepared diagnostic source changed: '+rel)
    if digest(Path(__file__))!=declaration['source_sha256']['default_upper_yaw.py']:raise ValueError('launcher changed after preparation')
    output.mkdir(parents=True,exist_ok=False);shutil.copytree(bundle/'sources',output/'sources');frozen=output/'sources'
    shutil.copy2(frozen/'adaptive_runner.py',frozen/'baseline_adaptive_runner.py');shutil.copy2(bundle/'candidate_runner.py',frozen/'adaptive_runner.py');shutil.copy2(bundle/'candidate_g1_backend.py',frozen/'g1_backend.py');shutil.copy2(bundle/'default_upper_yaw.py',frozen/'default_upper_yaw.py');shutil.copy2(bundle/'declaration.json',output/'declaration.json')
    sys.path.insert(0,str(frozen));support=load('frozen_default_upper_support',frozen/'default_upper_yaw.py');module=load('frozen_default_upper_runner',frozen/'adaptive_runner.py');module.PACKAGE=PACKAGE;module.DefaultUpperDiagnosticSession=support.DefaultUpperDiagnosticSession;module.bounded_home_reference=support.bounded_home_reference;module.save_startup_physical=support.save_startup_physical
    a=SimpleNamespace(**declaration['arguments']);a.output=output;a.frozen=frozen;a.root=PACKAGE/'Robot_G1';a.worker=True;a.rgb=bool(rgb);a.rig_module=load('default_upper_rig',frozen/'jev_agent_wbc/robot_sensor_rig.py');a.pink=None;a.collision=None
    import yaml
    io=yaml.safe_load((frozen/'height_policy_io.yaml').read_text());sensor,_=a.rig_module.load_config('g1',frozen/'robot_sensor_mounts.json');backend=load('default_upper_backend',frozen/'g1_backend.py')
    a.pink=backend.PinkProcess(io['actions'][0]['joint_names'],io['articulations']['robot']['joint_names'],a.rig_module.optical_transform(sensor['camera']),frozen/'g1_29dof.urdf',source_path=frozen/'g1_backend.py',stderr_path=output/'pink.stderr.log',solver_options=dict(posture_home=dict(zip(io['articulations']['robot']['joint_names'],io['articulations']['robot']['default_joint_pos'])),posture_cost=.03,damping_cost=.02,waist_posture_cost=a.waist_posture_cost))
    a.pink.warmup(dict(zip(io['articulations']['robot']['joint_names'],io['articulations']['robot']['default_joint_pos'])))
    (output/'provenance.json').write_text(json.dumps(dict(diagnostic_only=True,declaration_sha256=digest(output/'declaration.json'),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('rig_module','pink','collision')},frozen_sha256={str(p.relative_to(frozen)):digest(p) for p in frozen.rglob('*') if p.is_file()},pace=1),indent=2)+'\n')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES');os.environ.setdefault('ACCEPT_EULA','Y')
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,enable_cameras=a.rgb,device='cpu');app=launcher.app
    try:return module.run(a)
    except BaseException:
        import traceback
        (output/'error.txt').write_text(traceback.format_exc());raise
    finally:
        if a.pink:a.pink.close()
        if a.collision:a.collision.close()
        from isaaclab.sim import SimulationContext
        if SimulationContext.instance() is not None:SimulationContext.instance()._disable_app_control_on_stop_handle=True
        app.close(wait_for_replicator=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path);parser.add_argument('--rgb',action='store_true');args=parser.parse_args()
    if args.prepare:
        if args.output or args.rgb:raise ValueError('preparation cannot run physics or rendering')
        prepare(args.bundle.resolve());return 0
    if not args.output:raise ValueError('new output directory required')
    return run_prepared(args.bundle.resolve(),args.output.resolve(),args.rgb)
if __name__=='__main__':raise SystemExit(main())
