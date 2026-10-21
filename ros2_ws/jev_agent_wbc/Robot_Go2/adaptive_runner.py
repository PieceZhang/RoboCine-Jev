#!/usr/bin/env python3
"""Measured velocity/camera task suite for native G1-height/Pink and Go2-Piper.

No root stabilization or learned joint-action corrections. All authored camera
samples remain immutable WORLD TimedTask samples, including final hold.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(PACKAGE/'scripts'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def six_axis_excursion(t,duration,nontrivial):
    """Authored local perturbation; no measured state enters this function."""
    import numpy as np
    envelope=np.sin(np.pi*t/duration)**2
    if nontrivial:
        phase=2*np.pi*t/duration
        return (envelope*np.array([.10*np.sin(phase),.07*np.cos(phase),.08*np.sin(phase/2)]),
                envelope*np.array([.36*np.sin(phase),.30*np.cos(phase),.34*np.sin(phase/2)]))
    return (envelope*np.array([.012*np.sin(t),.012*np.sin(.8*t),.015*np.sin(.6*t)]),
            envelope*np.array([.025*np.sin(.6*t),.025*np.sin(.8*t),.025*np.sin(t)]))


def native_posture_enabled(profile,preferred_velocity):
    import numpy as np
    if profile not in ('none','x58_moving_v1','x58_all_v1'):
        raise ValueError('unknown native posture profile')
    return profile=='x58_all_v1' or (profile=='x58_moving_v1' and np.linalg.norm(preferred_velocity)>0)


def finalize_native_feedback(controller,feedback_check,actual_check,override_reasons):
    """Keep the actual domain check and roll back I on any later native veto."""
    report=feedback_check.get('world_z_feedback')
    if controller is not None and report is not None:
        if override_reasons:
            controller.rollback_last_update(','.join(override_reasons),update_id=report['update_id'])
        actual_check['world_z_feedback']=report
    return actual_check


def consume_policy_with_seed_confirmation(invoke_policy,controller,actual_check,actual_native_feature):
    """Commit an admission seed only after the real float32 policy call succeeds."""
    report=actual_check['world_z_feedback']
    try:
        result=invoke_policy()
    except BaseException:
        controller.rollback_last_update('seed_policy_consumption_failed',update_id=report['update_id'])
        raise
    if not controller.finish_seed_update(report['update_id'],applied_native=actual_native_feature):
        raise RuntimeError('admission seed was not confirmed by its current actual policy input')
    return result


def finalize_active_stand_brake(context,candidate,support,feedback_overrides,veto_reasons):
    issued=context.finalize(candidate,veto_reasons=veto_reasons)
    if context.manifest()['veto_reasons']:
        support.update(feasible=False,reason='tracking_limit',camera_supported=False,base_velocity_feasible=False)
        feedback_overrides.append('active_brake_safety_veto')
    return issued


def make_native_z_feedback(feedback_module,seed_module=None,*,seed_profile='none'):
    """Select the explicit initialization scope without changing pinned P+I."""
    if seed_profile=='none':
        return feedback_module.WorldZFeedback(enabled=True,ki_per_s=.2)if feedback_module is not None else None
    if (seed_profile not in ('fresh_steady_native_z_v1','fresh_actual_output_continuity_v2')
            or feedback_module is None or seed_module is None):
        raise ValueError('explicit native Z initialization requires its bounded feedback and wrapper')
    return seed_module.BumplessWorldZFeedback(enabled=True,ki_per_s=.2,initialization_profile=seed_profile)


def save_native_plant_readback(robot,stage,output,source_usd):
    """Read existing native tensors once, before timing; no draws or mutations."""
    import inspect
    import numpy as np
    view=robot.root_physx_view
    robot.data.body_link_pose_w
    def tensor(value):
        value=value.detach().cpu().numpy()if hasattr(value,'detach')else np.asarray(value)
        return dict(shape=list(value.shape),dtype=str(value.dtype),values=value.tolist())
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    actuators={}
    for name,actuator in robot.actuators.items():
        actuators[name]=dict(joint_names=list(actuator.joint_names),stiffness=tensor(actuator.stiffness),
            damping=tensor(actuator.damping),effort_limit=tensor(actuator.effort_limit),
            actual_delay_steps={key:tensor(getattr(actuator,key).time_lags)for key in
                ['positions_delay_buffer','velocities_delay_buffer','efforts_delay_buffer']if hasattr(actuator,key)})
    layers=[]
    for layer in stage.GetUsedLayers():
        path=Path(layer.realPath)if layer.realPath else None
        layers.append(dict(identifier=layer.identifier,real_path=str(path)if path else None,
            sha256=digest(path)if path and path.is_file()else hashlib.sha256(layer.ExportToString().encode()).hexdigest(),
            hash_kind='file_bytes'if path and path.is_file()else'layer_export_text'))
    api=Path(inspect.getfile(type(view)))
    report=dict(schema='go2_startup_native_plant/v1',before_timed_loop=True,read_only=True,stamp_s=0.,
        body_names=list(robot.body_names),joint_names=list(robot.joint_names),
        masses=tensor(view.get_masses()),coms=tensor(view.get_coms()),inertias=tensor(view.get_inertias()),
        actor_world_poses_xyzw=tensor(view.get_link_transforms()),joint_positions=tensor(robot.data.joint_pos),
        source_default_joint_positions=tensor(robot.data.default_joint_pos),actual_explicit_actuators=actuators,
        source_usd=dict(path=str(source_usd),sha256=digest(source_usd)),layers=layers,
        conventions=dict(masses_axes='articulation,body_names; kg',
            coms_axes='articulation,body_names,xyz/qx/qy/qz/qw; COM principal-axes frame relative to rigid-body actor frame',
            inertias_axes='articulation,body_names,9 column-major components; tensor atCOM in actor frame, not assumed principal diagonal',
            actor_poses_axes='articulation,body_names,xyz/qx/qy/qz/qw; native world',
            delay_axes='articulation environment,physics-step count; no delay resampling performed',
            frame_warning='Resolve body fusion and COM principal axes before any source-model comparison.'),
        native_tensor_api_source=dict(path=str(api),sha256=digest(api)))
    (Path(output)/'startup_native_plant.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def camera_support_with_terminal_rejection(coordinator,stamp,sample,link,body,command,measured,
        previous,dt,horizon,forecast,terminal_error=None,deadline_failed=False):
    """The actual loop's rejected-context branch must not retry invalid history."""
    if terminal_error is not None:
        return coordinator.reject_without_forecast(sample(stamp),link,previous,command_dt=dt,
            reason='tracking_limit'if deadline_failed else'command_stale')
    return coordinator.command(stamp,sample,link,body,command,measured,max_lookahead_s=horizon,**forecast)


def final_terminal_body_command(command,previous,dt,context=None,terminal_error=None,deadline_failed=False):
    """Unknown forecast or failed deadline keeps final body command braking.

    This runs after native-domain recovery and FCL. Their legal native7 result
    remains active, while recovery cannot restart body motion under a rejected
    context. A fresh successful context may resume within the original budget.
    """
    import numpy as np
    command=np.asarray(command,float);previous=np.asarray(previous,float)
    if (command.shape!=(3,)or previous.shape!=(3,)or not np.isfinite(np.r_[command,previous,dt]).all()or dt<=0):
        raise ValueError('finite actual body command,previous and interval required')
    if terminal_error is not None or deadline_failed:
        delta=np.array([.5,.5,.8])*dt
        return np.clip(np.zeros(3),previous-delta,previous+delta),True
    return (context.guard_applied(command)if context is not None else command.copy()),False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=['g1', 'go2'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', default='forward,backward,left,right,diagonal_left,diagonal_right,yaw_left,yaw_right,camera_shot')
    p.add_argument('--move-seconds', type=float, default=5.)
    p.add_argument('--stop-seconds', type=float, default=3.)
    p.add_argument('--speed', type=float, default=.10)
    p.add_argument('--yaw-speed', type=float, default=.12)
    p.add_argument('--upper-damping-scale',type=float,default=3.,help='G1 upper implicit PD damping; original effort/position caps preserved')
    p.add_argument('--upper-velocity-feedforward',action='store_true',help='Diagnostic native PD velocity feedforward')
    p.add_argument('--upper-mode',choices=['position_integral','velocity_servo','bounded_integral'],default='velocity_servo')
    p.add_argument('--park-damping-scale',type=float,default=3.,help='Smooth native lower-body damping during stop; learned targets unchanged')
    p.add_argument('--park-upper-damping-scale',type=float,default=1.,help='Go2 native arm damping multiplier during declared stop')
    p.add_argument('--parking-camera-hold',action='store_true',help='Optional native-input hold diagnostic; not a validated improvement')
    p.add_argument('--parking-anchor-reset',action='store_true',help='Optional measured brake-pose anchor diagnostic')
    p.add_argument('--parking-profile',choices=['legacy','brake_settle_v1'],default='legacy',help='G1 native-input stand transition candidate')
    p.add_argument('--park-posture-mode',choices=['home','measured'],default='home',help='G1 soft PostureTask preference after measured stand entry')
    p.add_argument('--waist-posture-cost',type=float,default=None,help='G1 soft home preference; camera and collision tasks retain waist authority')
    p.add_argument('--fine-collision',action='store_true',help='Audit original geometry with an isolated existing FCL worker')
    p.add_argument('--g1-velocity-profile',choices=['baseline','low_gain'],default='baseline')
    p.add_argument('--go2-response-profile',choices=['legacy','measured_backward_v1'],default='legacy')
    p.add_argument('--go2-native-posture',choices=['none','x58_moving_v1','x58_all_v1'],default='none')
    p.add_argument('--go2-prediction-profile',choices=['legacy','persistent_bias_12_v1','bounded_ramp_history_v2'],default='legacy')
    p.add_argument('--go2-world-z-feedback',choices=['none','bounded_integral_v1'],default='none')
    p.add_argument('--go2-world-z-seed',choices=['none','fresh_steady_native_z_v1','fresh_actual_output_continuity_v2'],default='none')
    p.add_argument('--go2-terminal-native-hold',choices=['none','end2_actual_v1'],default='none')
    p.add_argument('--go2-active-stand-brake',choices=['none','vy_p05_v1','vxy_p05_v1'],default='none')
    p.add_argument('--go2-terminal-profile',choices=['none','endpoint_xy58_15_v1','endpoint_xy58_15_yaw0_v1'],default='none')
    p.add_argument('--camera-excursion',choices=['diagnostic','nontrivial'],default='diagnostic')
    p.add_argument('--physics-accuracy',choices=['native','tgs_forces','tgs_forces_velocity1'],default='native',
        help='Explicit PhysX precision comparison; preserves dt, mass, friction, caps and learned actions')
    p.add_argument('--ground-material',choices=['native_average','source_multiply'],default='native_average',
        help='Compare the pinned source ground combine mode without changing any friction coefficient')
    p.add_argument('--rgb', action='store_true')
    p.add_argument('--velocity-audit', action='store_true',help='Go2-only read-only 200 Hz root/link consistency diagnostic')
    p.add_argument('--velocity-audit-tail-seconds',type=float,default=0.,help='Record only a predeclared final stop tail; zero records every substep')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    from jev_agent_wbc.multi_robot_runtime import add_arguments
    add_arguments(p)
    a = p.parse_args(); a.output = a.output.resolve()
    if a.avoidance_case:
        a.cases=a.avoidance_case;a.move_seconds=8.;a.rgb=False
    if a.velocity_audit and a.robot!='go2':
        raise ValueError('velocity-audit is a Go2-only diagnostic')
    if a.parking_profile!='legacy' and a.robot!='g1':
        raise ValueError('brake_settle_v1 is a G1-only candidate')
    if a.go2_native_posture!='none' and a.robot!='go2':
        raise ValueError('native EE operating preference is a Go2-only candidate')
    if a.go2_prediction_profile!='legacy' and a.robot!='go2':
        raise ValueError('persistent input prediction is a Go2-only candidate')
    if a.go2_world_z_feedback!='none' and a.robot!='go2':
        raise ValueError('world-Z native-input feedback is a Go2-only candidate')
    if a.go2_world_z_seed!='none' and (a.robot!='go2' or a.go2_world_z_feedback!='bounded_integral_v1'):
        raise ValueError('admission-only native Z seed requires Go2 bounded world-Z feedback')
    if a.go2_terminal_native_hold!='none' and (a.robot!='go2' or a.go2_terminal_profile=='none'
            or a.stop_seconds!=5. or a.parking_camera_hold or not a.fine_collision
            or a.go2_world_z_feedback!='bounded_integral_v1'):
        raise ValueError('fixed terminal native hold requires original5s Go2 terminal/FCL/integral and no legacy hold')
    if a.go2_active_stand_brake!='none' and (a.robot!='go2' or a.go2_terminal_profile=='none'
            or a.stop_seconds!=5. or a.parking_camera_hold or not a.fine_collision
            or a.go2_terminal_native_hold!='none' or a.go2_prediction_profile!='bounded_ramp_history_v2'):
        raise ValueError('active stand braking requires original5s Go2 terminal/FCL and no native7 hold')
    if a.go2_terminal_profile!='none' and (a.robot!='go2' or a.stop_seconds<5. or a.parking_camera_hold):
        raise ValueError('terminal allocation requires Go2, original stop>=5s and no native7 parking hold')
    if a.waist_posture_cost is not None and (a.robot!='g1' or not .03<=a.waist_posture_cost<=1.):
        raise ValueError('bounded positive G1 waist soft posture cost required')
    if not 0<=a.velocity_audit_tail_seconds<=a.stop_seconds or (a.velocity_audit_tail_seconds and not a.velocity_audit):
        raise ValueError('velocity audit tail requires enabled probe and a bounded declared stop window')
    if not (3. <= a.move_seconds <= 20. and 3. <= a.stop_seconds <= 10.
            and 0 < a.speed <= .15 and 0 < a.yaw_speed <= .2 and 1.<=a.upper_damping_scale<=5.
            and 1.<=a.park_damping_scale<=5. and 1.<=a.park_upper_damping_scale<=5.):
        raise ValueError('bounded diagnostic durations and native velocity requests required')
    if not a.worker:
        if a.output.exists():
            raise FileExistsError(a.output)
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'])
        summary = json.loads((a.output/'summary.json').read_text()) if (a.output/'summary.json').is_file() else {}
        return 0 if child.returncode == 0 and summary.get('run_complete') and summary.get('all_pass') else 2
    a.output.mkdir(parents=True, exist_ok=False); frozen = a.output/'sources'; frozen.mkdir()
    shutil.copytree(PACKAGE/'jev_agent_wbc', frozen/'jev_agent_wbc', ignore=shutil.ignore_patterns('__pycache__'))
    paths = [Path(__file__), PACKAGE/'Robot_G1/velocity_servo.py', PACKAGE/'Robot_G1/wbc_backend.py',
             PACKAGE/'Robot_Go2/wbc_backend.py', PACKAGE/'Robot_Go2/camera_coordinator.py',
             PACKAGE/'config/robot_sensor_mounts.json']
    names = ['adaptive_runner.py', 'velocity_servo.py', 'g1_backend.py', 'go2_backend.py',
             'camera_coordinator.py', 'robot_sensor_mounts.json']
    if a.robot=='go2':
        paths.append(PACKAGE/'Robot_Go2/optical_sampler.py');names.append('optical_sampler.py')
        if a.go2_world_z_feedback!='none':
            paths.append(PACKAGE/'Robot_Go2/world_z_feedback.py');names.append('world_z_feedback.py')
            if a.go2_world_z_seed!='none':
                paths.append(PACKAGE/'Robot_Go2/world_z_bumpless.py');names.append('world_z_bumpless.py')
        if a.go2_prediction_profile=='bounded_ramp_history_v2':
            paths.append(PACKAGE/'Robot_Go2/bounded_body_prediction.py');names.append('bounded_body_prediction.py')
        if a.go2_terminal_profile!='none':
            paths.append(PACKAGE/'Robot_Go2/terminal_allocator.py');names.append('terminal_allocator.py')
            if a.go2_terminal_profile=='endpoint_xy58_15_yaw0_v1':
                paths.append(PACKAGE/'Robot_Go2/terminal_yaw_allocator.py');names.append('terminal_yaw_allocator.py')
        if a.go2_terminal_native_hold!='none':
            paths.append(PACKAGE/'Robot_Go2/terminal_native_hold.py');names.append('terminal_native_hold.py')
        if a.go2_active_stand_brake!='none':
            paths.append(PACKAGE/'Robot_Go2/lateral_stand_brake.py');names.append('lateral_stand_brake.py')
    for source, name in zip(paths, names):
        shutil.copy2(source, frozen/name)
    shutil.copy2(PACKAGE/'Robot_G1/legged_collision.py',frozen/'legged_collision.py')
    shutil.copy2(PACKAGE/'Robot_G1/pink_collision.py',frozen/'pink_collision.py')
    shutil.copy2(PACKAGE/'Robot_G1/upper_reference.py',frozen/'upper_reference.py')
    shutil.copy2(PACKAGE/'Robot_G1/contact_materials.py',frozen/'contact_materials.py')
    if a.velocity_audit:
        shutil.copy2(PACKAGE/'Robot_Go2/velocity_audit.py',frozen/'velocity_audit.py')
    if a.parking_profile!='legacy':
        shutil.copy2(PACKAGE/'Robot_G1/parking_supervisor.py',frozen/'parking_supervisor.py')
    root = PACKAGE/('Robot_G1' if a.robot == 'g1' else 'Robot_Go2')
    a.root, a.frozen = root, frozen
    if a.robot == 'g1':
        for source in [root/'config/height_policy_io.yaml', root/'kinematics/g1_29dof.urdf',
                       root/'policies/unitree_g1_velocity_height_recurrent_student.pt']:
            shutil.copy2(source, frozen/source.name)
    else:
        shutil.copy2(root/'vendor/LeggedManip_Lab/mujoco/deploy/policy/go2_piper/wbc/policy.pt', frozen/'policy.pt')
        shutil.copy2(root/'actuator_limits.py',frozen/'go2_actuator_limits.py')
        shutil.copy2(root/'vendor/LeggedManip_Lab/mujoco/robots/go2_piper/go2piper.xml',frozen/'go2piper.xml')
        mdp = root/'vendor/LeggedManip_Lab/source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab'
        for relative in ['config/go2_piper/wbc_env_cfg.py','mdp/pose_command_wbc.py']:
            destination=frozen/'go2_upstream'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(mdp/relative,destination)
    # Import common measured geometry/protocol from the exact frozen package.
    sys.path.insert(0, str(frozen))
    a.rig_module = load('frozen_adaptive_rig', frozen/'jev_agent_wbc/robot_sensor_rig.py')
    a.pink = None;a.collision=None
    if a.robot == 'g1':
        import yaml
        io = yaml.safe_load((frozen/'height_policy_io.yaml').read_text())
        sensor, _ = a.rig_module.load_config('g1', frozen/'robot_sensor_mounts.json')
        backend = load('frozen_adaptive_g1', frozen/'g1_backend.py')
        a.pink = backend.PinkProcess(io['actions'][0]['joint_names'], io['articulations']['robot']['joint_names'],
            a.rig_module.optical_transform(sensor['camera']), frozen/'g1_29dof.urdf',
            source_path=frozen/'g1_backend.py', stderr_path=a.output/'pink.stderr.log',
            solver_options=dict(posture_home=dict(zip(io['articulations']['robot']['joint_names'],
                io['articulations']['robot']['default_joint_pos'])),posture_cost=.03,damping_cost=.02,
                waist_posture_cost=a.waist_posture_cost))
        a.pink.warmup(dict(zip(io['articulations']['robot']['joint_names'], io['articulations']['robot']['default_joint_pos'])))
    (a.output/'provenance.json').write_text(json.dumps(dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('rig_module','pink','collision')},
        frozen_sha256={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in frozen.rglob('*') if p.is_file()}, pace=1), indent=2)+'\n')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES'); os.environ.setdefault('ACCEPT_EULA', 'Y')
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=a.rgb, device='cpu'); app = launcher.app
    try:
        return run(a)
    except BaseException:
        import traceback
        (a.output/'error.txt').write_text(traceback.format_exc()); raise
    finally:
        if getattr(a,'avoidance_runtime',None) is not None:a.avoidance_runtime.finish()
        if a.pink:
            a.pink.close()
        if a.collision:
            a.collision.close()
        from isaaclab.sim import SimulationContext
        if SimulationContext.instance() is not None:
            SimulationContext.instance()._disable_app_control_on_stop_handle = True
        app.close(wait_for_replicator=False)


def run(a):
    import math
    import re
    import numpy as np
    import torch
    import yaml
    from scipy.spatial.transform import Rotation
    from isaaclab.assets import Articulation
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from jev_agent_wbc.representation import Pose
    from jev_agent_wbc.wbc_protocol import CameraTrajectory, TimedTask, Capabilities, semantic_camera
    from jev_agent_wbc.wbc_physical import MeasuredColliderGeometry, LiveTaskSession, audit_stop
    import omni.usd
    torch.set_num_threads(1); torch.manual_seed(42)
    servo_module = load('frozen_adaptive_servo', a.frozen/'velocity_servo.py')
    dt, physics_dt = .02, .005
    physx_cfg=sim_utils.PhysxCfg()
    if a.physics_accuracy!='native':
        if physx_cfg.solver_type!=1:raise ValueError('TGS precision candidate requires native TGS solver')
        physx_cfg.enable_external_forces_every_iteration=True
        if a.physics_accuracy=='tgs_forces_velocity1':physx_cfg.min_velocity_iteration_count=1
    sim = SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device='cpu', render_interval=4,physx=physx_cfg,enable_scene_query_support=bool(a.avoidance_case)))
    ground_combine='multiply' if a.ground_material=='source_multiply' else 'average'
    ground = sim_utils.CuboidCfg(size=(40.,40.,.1), collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.,
            friction_combine_mode=ground_combine,restitution_combine_mode=ground_combine))
    ground.func('/World/Ground', ground, translation=(0.,0.,-.05))
    light = sim_utils.DomeLightCfg(intensity=1000.); light.func('/World/Light',light)
    if a.robot == 'g1':
        from agile.rl_env.assets.robots.unitree_g1 import G1_29DOF, G1_29DOF_DELAYED_DC_MOTOR
        cfg = G1_29DOF.copy(); cfg.spawn.usd_path = str(a.root/'assets/G1/g1.usd')
        # Conservative historical operating cap: this height asset/config has
        # 139Nm hip-roll source caps; the prior tested operating cap is 88Nm.
        cfg.actuators['legs'].effort_limit_sim['.*_hip_roll_joint'] = G1_29DOF_DELAYED_DC_MOTOR.actuators['legs'].effort_limit_sim['.*_hip_roll_joint']
        # Damping addresses physical upper-body ringing; never increase torque
        # caps or alter the lower-body policy output to obtain tracking speed.
        for group in ['arms','waist']:
            cfg.actuators[group].damping={k:v*a.upper_damping_scale for k,v in cfg.actuators[group].damping.items()}
        if a.upper_mode in ('velocity_servo','bounded_integral'):
            from isaaclab.actuators import IdealPDActuatorCfg
            for group in ['arms','waist']:
                original=cfg.actuators[group]
                # One explicit sum tau=Kp*error+Kd*velocity_error+g, clipped
                # inside the original motor caps. No extra torque is stacked
                # on a separate saturated implicit drive.
                cfg.actuators[group]=IdealPDActuatorCfg(joint_names_expr=list(original.joint_names_expr),
                    stiffness=original.stiffness,damping=original.damping,armature=original.armature,
                    effort_limit=original.effort_limit_sim,effort_limit_sim=original.effort_limit_sim,
                    velocity_limit=original.velocity_limit_sim,velocity_limit_sim=original.velocity_limit_sim)
        cfg.init_state.pos = (0.,0.,.78)
        io = yaml.safe_load((a.frozen/'height_policy_io.yaml').read_text())
        policy = torch.jit.load(str(a.frozen/'unitree_g1_velocity_height_recurrent_student.pt'),map_location='cpu').eval();policy.reset_flat()
        policy_names = io['actions'][0]['joint_names']; observation_names = io['articulations']['robot']['joint_names']
        offsets = torch.tensor(io['actions'][0]['offset']).reshape(1,12)
        scales = torch.tensor(io['actions'][0]['scale']).reshape(1,12)
        defaults = torch.tensor(io['articulations']['robot']['default_joint_pos']).reshape(1,29)
        command_limits = (.5,.5,1.)
    else:
        from LeggedManip_Lab.assets.go2_piper.go2_piper_articulation_cfg import GO2_PIPER_CFG
        cfg = GO2_PIPER_CFG.copy()
        policy = torch.jit.load(str(a.frozen/'policy.pt'), map_location='cpu').eval()
        policy_names = ['FR_hip_joint','FR_thigh_joint','FR_calf_joint','FL_hip_joint','FL_thigh_joint','FL_calf_joint',
            'RR_hip_joint','RR_thigh_joint','RR_calf_joint','RL_hip_joint','RL_thigh_joint','RL_calf_joint',
            'joint1','joint2','joint3','joint4','joint5','joint6']
        source_limits=load('frozen_go2_motor_limits',a.frozen/'go2_actuator_limits.py').load_motor_limits(a.frozen/'go2piper.xml')
        (a.output/'declared_motor_limits.json').write_text(json.dumps(source_limits,indent=2)+'\n')
        for actuator in cfg.actuators.values():
            caps={n:source_limits['effort_limits_nm'][n] for n in policy_names
                  if any(re.fullmatch(pattern,n) for pattern in actuator.joint_names_expr)}
            if not caps:raise ValueError('native actuator lacks a pinned motor cap')
            actuator.effort_limit=dict(caps);actuator.effort_limit_sim=dict(caps)
        command_limits = (.2,.2,.2)
        backend = load('frozen_adaptive_go2', a.frozen/'go2_backend.py')
        # The outer-domain checker reads pinned source metadata; no inferred
        # checkpoint training-domain certification is attached to that metadata.
        backend.UPSTREAM_MDP = a.frozen/'go2_upstream'
        coordinate_module = load('frozen_adaptive_coordinator',a.frozen/'camera_coordinator.py')
        optical_module=load('frozen_adaptive_optical_sampler',a.frozen/'optical_sampler.py')
    cfg.prim_path = '/World/Robot'
    from pxr import Usd,UsdPhysics
    original_stage=Usd.Stage.Open(cfg.spawn.usd_path)
    source_drives={p.GetName():UsdPhysics.DriveAPI(p,'angular').GetMaxForceAttr().Get()
        for p in original_stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    (a.output/'source_usd_drive_caps.json').write_text(json.dumps(source_drives,indent=2)+'\n')
    robot = Articulation(cfg); stage = omni.usd.get_context().get_stage()
    rig = a.rig_module.RobotSensorRig(stage,'/World/Robot',a.robot,a.frozen/'robot_sensor_mounts.json')
    avoidance=None
    if a.avoidance_case:
        from jev_agent_wbc.multi_robot_runtime import AvoidanceRuntime
        avoidance=AvoidanceRuntime(a,PACKAGE,stage,rig,None,'pelvis' if a.robot=='g1' else 'base',a.robot,
                                   native_platform_limits=command_limits)
        a.avoidance_runtime=avoidance
    sim.reset()
    import inspect
    precision_source=dict(profile=a.physics_accuracy,physics_dt_s=physics_dt,control_dt_s=dt,
        solver_type=physx_cfg.solver_type,enable_external_forces_every_iteration=physx_cfg.enable_external_forces_every_iteration,
        min_position_iteration_count=physx_cfg.min_position_iteration_count,max_position_iteration_count=physx_cfg.max_position_iteration_count,
        min_velocity_iteration_count=physx_cfg.min_velocity_iteration_count,max_velocity_iteration_count=physx_cfg.max_velocity_iteration_count,
        native_articulation_properties=cfg.spawn.articulation_props.to_dict(),
        actual_scene_attributes={str(prim.GetPath()):{attr.GetName():attr.Get() for attr in prim.GetAttributes()
            if any(word in attr.GetName().lower() for word in ('solver','iteration','externalforce'))}
            for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)},
        native_simulation_source_sha256={inspect.getfile(t):hashlib.sha256(Path(inspect.getfile(t)).read_bytes()).hexdigest()
            for t in (type(physx_cfg),type(sim))},
        source_note='Force-only candidate changes only TGS force integration; velocity1 is an explicitly separate compound candidate.')
    (a.output/'physics_precision.json').write_text(json.dumps(precision_source,indent=2,default=str)+'\n')
    force_flags=[value for attributes in precision_source['actual_scene_attributes'].values() for key,value in attributes.items()
        if key.lower().endswith('enableexternalforceseveryiteration')]
    if a.physics_accuracy!='native' and (not force_flags or any(value is not True for value in force_flags)):
        raise RuntimeError('requested TGS force integration was not authored on the measured physical scene')
    if a.physics_accuracy=='tgs_forces_velocity1':
        velocity_floors=[value for attributes in precision_source['actual_scene_attributes'].values()
            for key,value in attributes.items() if key.lower().endswith('minvelocityiterationcount')]
        if not velocity_floors or any(value!=1 for value in velocity_floors):
            raise RuntimeError('requested minimum velocity iteration count was not authored on the measured physical scene')
    material_module=load('frozen_contact_materials',a.frozen/'contact_materials.py')
    material_audit=material_module.audit_stage(stage)
    material_audit.update(profile=a.ground_material,
        native_articulation_shape_material_properties=robot.root_physx_view.get_material_properties()[0].tolist(),
        native_material_property_axes=['static_friction','dynamic_friction','restitution'],
        native_shape_capacity=robot.root_physx_view.max_shapes,
        source_matching_scope='Only ground combine modes; existing coefficients/default material retained, source randomization not recreated')
    (a.output/'contact_materials.json').write_text(json.dumps(material_audit,indent=2)+'\n')
    if not material_audit['all_collision_materials_explicitly_resolved'] or any(
            x['friction_combine']!=ground_combine or x['restitution_combine']!=ground_combine
            or x['static_friction']!=1. or x['dynamic_friction']!=1.
            for x in material_audit['ground_materials']):
        raise RuntimeError('physical contact material readback differs from declared ground profile')
    # A single physical initial condition, before measured time starts.
    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone())
    robot.reset(); robot.update(physics_dt)
    ids = [robot.joint_names.index(n) for n in policy_names]
    observation_ids = [robot.joint_names.index(n) for n in observation_names] if a.robot=='g1' else ids
    target = robot.data.default_joint_pos.clone(); target_velocity=torch.zeros_like(target)
    target_effort=torch.zeros_like(target)
    action = torch.zeros(12 if a.robot=='g1' else 18)
    histories = None
    manipulator_bodies = {n for n in robot.body_names if ('shoulder' in n or 'elbow' in n or 'wrist' in n or n=='torso_link')}
    if a.robot == 'go2':
        manipulator_bodies = {n for n in robot.body_names if re.fullmatch(r'link[0-9]+',n) or 'effector' in n or 'gripper' in n}
    geometry = MeasuredColliderGeometry.from_stage(stage,'/World/Robot',manipulator_bodies,robot.body_names)
    if avoidance is not None:avoidance.attach_geometry(geometry)
    (a.output/'collider_geometry.json').write_text(json.dumps(geometry.manifest(),indent=2)+'\n')
    (a.output/'sensor_mounts.json').write_text(json.dumps(rig.manifest(),indent=2)+'\n')
    if a.fine_collision:
        collision_module=load('frozen_legged_collision',a.frozen/'legged_collision.py')
        metadata=collision_module.extract_collision_geometry(stage,'/World/Robot')
        metadata_path=a.output/'fine_collision_geometry.json';metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
        a.collision=collision_module.DistanceProcess(PACKAGE/'Robot_TIAGo/wbc/.venv/bin/python',
            a.frozen/'legged_collision.py',metadata_path,a.output/'collision.stderr.log')
        if a.robot=='g1':
            measured_initial=a.rig_module.lab_body_poses(robot,{s['body'] for s in metadata['shapes']}|{'pelvis'})
            initialization=a.pink.configure_collisions(metadata,dict(zip(robot.joint_names,robot.data.joint_pos[0].tolist())),
                measured_initial['pelvis'],measured_initial)
            (a.output/'pink_collision_initialization.json').write_text(json.dumps(initialization,indent=2)+'\n')
    effort = robot.data.joint_effort_limits[0].numpy()
    (a.output/'actuator_audit.json').write_text(json.dumps(dict(joint_names=robot.joint_names,
        effort_limits=effort.tolist(),position_limits=robot.data.joint_pos_limits[0].tolist(),
        velocity_limits=robot.data.joint_vel_limits[0].tolist(),initialization_only_root_write=True,
        native_pd_damping={n:x.damping.tolist() for n,x in robot.actuators.items()},park_damping_scale=a.park_damping_scale,
        native_actuator_effort_limits={n:x.effort_limit.tolist() for n,x in robot.actuators.items()},
        native_actuator_joint_names={n:list(x.joint_names) for n,x in robot.actuators.items()},
        native_actuator_models={n:('implicit' if x.is_implicit_model else 'explicit') for n,x in robot.actuators.items()},
        historical_hip_roll_operating_cap_88nm=(all(abs(effort[robot.joint_names.index(n)]-88.)<1e-3
            for n in ['left_hip_roll_joint','right_hip_roll_joint']) if a.robot=='g1' else None)),indent=2)+'\n')
    if a.robot=='go2':
        coordinator = coordinate_module.CameraCoordinator(rig.camera_local,backend=backend,response_profile=a.go2_response_profile,
            prediction_profile=a.go2_prediction_profile)
        native_parking=coordinate_module.NativeCameraParking()
        domain_recovery=coordinate_module.NativeDomainRecovery(coordinator.domain,backend)
        feedback_module=(load('frozen_world_z_feedback',a.frozen/'world_z_feedback.py')
            if a.go2_world_z_feedback!='none' else None)
        seed_module=(load('frozen_world_z_bumpless',a.frozen/'world_z_bumpless.py')if a.go2_world_z_seed!='none'else None)
        world_z_feedback=make_native_z_feedback(feedback_module,seed_module,seed_profile=a.go2_world_z_seed)
        terminal_module=(load('frozen_terminal_allocator',a.frozen/('terminal_yaw_allocator.py'
            if a.go2_terminal_profile=='endpoint_xy58_15_yaw0_v1' else'terminal_allocator.py'))
            if a.go2_terminal_profile!='none' else None)
        terminal_hold_module=(load('frozen_terminal_native_hold',a.frozen/'terminal_native_hold.py')
            if a.go2_terminal_native_hold!='none'else None)
        active_brake_module=(load('frozen_lateral_stand_brake',a.frozen/'lateral_stand_brake.py')
            if a.go2_active_stand_brake!='none'else None)
    lower_groups=['legs','feet'] if a.robot=='g1' else ['base_legs']
    damping_groups=lower_groups if a.robot=='g1' else list(robot.actuators)
    native_damping={n:robot.actuators[n].damping.clone() for n in damping_groups}
    damping_scales={n:1. for n in damping_groups}
    leg_damping_scale=1.
    upper_ids=[robot.joint_names.index(n) for n in a.pink.owners['pink']] if a.robot=='g1' else []
    upper_reference=None
    if a.robot=='g1' and a.upper_mode=='bounded_integral':
        upper_module=load('frozen_upper_reference',a.frozen/'upper_reference.py')
        upper_reference=upper_module.UpperReference(a.pink.owners['pink'],robot.data.joint_pos[0,upper_ids].numpy(),
            robot.data.joint_pos_limits[0,upper_ids].numpy())
    base_name = 'pelvis' if a.robot=='g1' else 'base'
    velocity_audit = load('frozen_velocity_audit',a.frozen/'velocity_audit.py').VelocityAudit(robot,a.output,base_name) if a.velocity_audit else None
    measured_names=set(geometry.body_names)|rig.parent_names|{base_name}
    if a.robot=='go2':measured_names.add('link0')
    def measure(stamp):
        bodies = a.rig_module.lab_body_poses(robot,measured_names)
        camera = bodies[rig.camera_spec['parent']]@rig.camera_local
        return bodies,camera,geometry.observe(stamp,bodies[base_name],camera,bodies)
    def matrix_pose(T):
        return Pose.from_rt(T[:3,:3],T[:3,3])
    cases = dict(forward=[a.speed,0.,0.],backward=[-a.speed,0.,0.],left=[0.,a.speed,0.],right=[0.,-a.speed,0.],
        diagonal_left=[a.speed,a.speed,0.],diagonal_right=[a.speed,-a.speed,0.],
        yaw_left=[0.,0.,a.yaw_speed],yaw_right=[0.,0.,-a.yaw_speed],camera_shot=[a.speed,0.,0.],
        camera_6d=[0.,0.,0.],
        static_camera_conflict=[a.speed,0.,0.])
    if avoidance is not None:cases.update(orbit=[0.,0.,0.],movein=[0.,0.,0.])
    selected = a.cases.split(',')
    if not selected or any(n not in cases for n in selected):
        raise ValueError('unknown or empty case list')
    velocity_profile=servo_module.g1_profile(a.g1_velocity_profile) if a.robot=='g1' else servo_module.go2_profile(a.go2_response_profile)
    servo = servo_module.VelocityServo(limits=command_limits,**velocity_profile)
    (a.output/'velocity_servo_config.json').write_text(json.dumps(dict(limits=command_limits,parameters=velocity_profile,
        profile=a.g1_velocity_profile if a.robot=='g1' else a.go2_response_profile),indent=2)+'\n')
    session = None; rows=[];results=[];case_rows=[];all_tasks=[];failures=[]
    optical_sampler=None;optical_sampling_manifests=[]
    terminal_native_hold=None;terminal_hold_manifests=[]
    fk_max_position=fk_max_angle=0.;body_fk_max_position=body_fk_max_angle=0.
    preparation_s=2.+(a.avoidance_lead_s if avoidance is not None else 0.)
    steps = round((preparation_s+len(selected)*(a.move_seconds+a.stop_seconds))/dt)
    if a.robot=='go2':save_native_plant_readback(robot,stage,a.output,cfg.spawn.usd_path)
    start_wall=time.monotonic(); case_index=-1; stop_anchor=None; parking_mode=False;previous_policy_command=np.zeros(3)
    parking_posture=None;parking_posture_stamp=None
    terminal_endpoint=None;terminal_deadline_failed=False
    parking_supervisor=None
    parking_module=load('frozen_parking_supervisor',a.frozen/'parking_supervisor.py') if a.parking_profile!='legacy' else None
    limits = robot.data.joint_pos_limits[0].numpy()
    current_measurement=measure(0.)
    timing_totals=np.zeros(5)
    with (a.output/'measurements.jsonl').open('w',buffering=1) as log,torch.inference_mode():
        for step in range(steps):
            tick_start=time.monotonic()
            stamp=step*dt; bodies,camera,observation=current_measurement
            measured=np.r_[robot.data.root_lin_vel_b[0,:2].numpy(),robot.data.root_ang_vel_b[0,2].item()]
            if avoidance is not None and avoidance.scene_start is None and stamp>=2.-1e-8:
                from jev_agent_wbc.multi_robot_geometry import NativeKinematicModel
                active_names=list(a.pink.owners['pink']) if a.robot=='g1' else ['joint1','joint2','joint3','joint4','joint5','joint6']
                active_ids=[robot.joint_names.index(n) for n in active_names]
                model=NativeKinematicModel.from_stage(stage,'/World/Robot',bodies,base_name,active_names,
                    robot.data.joint_pos[0,active_ids].numpy(),limits[active_ids,0],limits[active_ids,1],
                    np.minimum(.65,robot.data.joint_vel_limits[0,active_ids].numpy()),geometry,rig,omnidirectional=True)
                avoidance.bind(stamp,bodies,camera,model)
            in_case=stamp>=preparation_s; new_index=min(len(selected)-1,int((stamp-preparation_s+1e-8)/(a.move_seconds+a.stop_seconds))) if in_case else -1
            if new_index!=case_index:
                if case_rows:
                    results.append(summarize(selected[case_index],case_rows,a,audit_stop));case_rows=[]
                case_index=new_index;servo.reset();stop_anchor=None;parking_mode=False
                parking_posture=None;parking_posture_stamp=None
                if parking_module:parking_supervisor=parking_module.ParkingSupervisor()
                if a.robot=='go2':native_parking=coordinate_module.NativeCameraParking()
                name=selected[case_index];case_start=stamp;reference=camera.copy();root_reference=bodies[base_name].copy()
                velocity=np.asarray(cases[name]);knots=np.linspace(0.,a.move_seconds,round(a.move_seconds/.1)+1)
                poses=[]
                for t in knots:
                    # A separately authored WORLD camera follows the nominal
                    # base path plus a six-axis smooth move. Never retarget to
                    # measured robot drift or modify this trajectory later.
                    T=reference.copy();yaw=velocity[2]*t
                    yaw_rotation=Rotation.from_euler('z',yaw).as_matrix()
                    heading=math.atan2(root_reference[1,0],root_reference[0,0])
                    # Author locomotion in the gravity-horizontal world plane;
                    # physical base roll/pitch are feedback, not a commanded
                    # vertical camera ramp. Failed earlier authored tasks stay
                    # immutable in their own result directories.
                    displacement=Rotation.from_euler('z',heading).as_matrix()@np.r_[velocity[:2]*t,0.]
                    if name=='static_camera_conflict':
                        displacement[:]=0.;yaw_rotation=np.eye(3)
                    T[:3,3]=root_reference[:3,3]+displacement+yaw_rotation@(reference[:3,3]-root_reference[:3,3])
                    T[:3,:3]=yaw_rotation@reference[:3,:3]
                    if name in ('camera_shot','camera_6d'):
                        delta_p,delta_r=six_axis_excursion(t,a.move_seconds,
                            name=='camera_6d' or a.camera_excursion=='nontrivial')
                        T[:3,3]+=delta_p
                        T[:3,:3]=Rotation.from_rotvec(delta_r).as_matrix()@T[:3,:3]
                    poses.append(semantic_camera(matrix_pose(T)))
                task=TimedTask(f'{a.robot}_{name}_{step}',stamp,observation.id,
                    CameraTrajectory(stamp,tuple(knots),tuple(poses)),.05,.15)
                if avoidance is not None:task=avoidance.task(stamp,bodies,camera)
                task_end=task.trajectory.end_s
                session=LiveTaskSession(Capabilities(f'{a.robot}_native_adaptive',True,frozenset(['timed_camera_se3'])))
                admission=session.accept(task,observation,stamp)
                if not admission['supported']:
                    raise RuntimeError(admission)
                if a.robot=='go2':
                    optical_sampler=optical_module.OpticalSampler(task)
                    terminal_endpoint=optical_sampler.sample(task_end)
                    terminal_deadline_failed=False
                    prior_terminal_hold=terminal_native_hold.manifest()if terminal_native_hold is not None else None
                    if prior_terminal_hold is not None:terminal_hold_manifests.append(prior_terminal_hold)
                    if a.go2_world_z_seed!='none':
                        world_z_feedback.admit(backend,task,now_s=stamp,observation_stamp_s=observation.stamp_s,
                            reference_optical=optical_sampler.sample(stamp),measured_optical=camera,
                            link0=bodies['link0'],extrinsic=rig.camera_local,
                            previous_applied_native=ee[0].numpy().copy()if case_index>0 else None,
                            previous_issue_stamp_s=stamp-dt if case_index>0 else None,
                            history_rows=rows[-26:],domain=coordinator.domain,previous_terminal_hold=prior_terminal_hold)
                    terminal_native_hold=(terminal_hold_module.TerminalNativeHold(task,a.stop_seconds)
                        if terminal_hold_module is not None else None)
                    optical_sampling_manifests.append(optical_sampler.manifest())
                    (a.output/'optical_sampling.json').write_text(json.dumps(dict(
                        scope='Go2 coordinator pose-only lookahead; original public admission and feedback retained',
                        task_manifests=optical_sampling_manifests),indent=2)+'\n')
                all_tasks.append(task.to_dict());(a.output/'timed_tasks.json').write_text(json.dumps(all_tasks,indent=2)+'\n')
            if avoidance is not None and session is not None:
                revision=avoidance.revision(session,stamp,bodies,camera)
                if revision is not None:
                    task=revision
                    if a.robot=='g1':current_task_context=solver_context.task_context(task.to_dict())
                    if a.robot=='go2':optical_sampler=optical_module.OpticalSampler(task)
                    all_tasks.append(task.to_dict())
                    (a.output/'timed_tasks.json').write_text(json.dumps(all_tasks,indent=2)+'\n')
            moving=in_case and stamp-case_start<a.move_seconds-1e-8
            if avoidance is not None and in_case and stamp>=task_end-1e-8:
                avoidance.execution_complete=True
            desired_velocity=np.asarray(cases[selected[case_index]]) if moving else np.zeros(3)
            if avoidance is not None and moving:
                # Piper carries most of the shot yaw; rotating the short-legged
                # base spends scarce lateral velocity on the camera lever arm.
                # The native coordinator may still allocate yaw to retain its
                # original workspace bounds.
                desired_velocity=avoidance.platform_velocity(stamp,bodies,camera,
                    yaw_share=0. if a.robot=='go2' else 1.,yaw_feedback_gain=1.6 if a.robot=='go2' else 0.)
            if in_case and not moving and stop_anchor is None:
                stop_anchor=bodies[base_name].copy();servo.reset()
                if parking_module and a.park_posture_mode=='measured':
                    # A subordinate soft preference starts with the declared
                    # stop, while the immutable WORLD camera and real collision
                    # barriers retain all upper-body control authority.
                    parking_posture={n:float(robot.data.joint_pos[0,robot.joint_names.index(n)]) for n in a.pink.owners['pink']}
                    parking_posture_stamp=stamp
            anchor_error=servo_module.stop_pose_error(stop_anchor,bodies[base_name]) if stop_anchor is not None else None
            policy_command=servo.update(desired_velocity,measured,dt,pose_error_body=anchor_error)
            for group in damping_groups:
                requested_damping=(a.park_damping_scale if group in lower_groups else a.park_upper_damping_scale) if stop_anchor is not None else 1.
                new_damping_scale=damping_scales[group]+float(np.clip(requested_damping-damping_scales[group],-4*dt,4*dt))
                if abs(new_damping_scale-damping_scales[group])>1e-8:
                    actuator=robot.actuators[group]
                    actuator.damping[:]=native_damping[group]*new_damping_scale
                    if actuator.is_implicit_model:
                        robot.write_joint_damping_to_sim(actuator.damping,joint_ids=actuator.joint_indices)
                    damping_scales[group]=new_damping_scale
            leg_damping_scale=damping_scales[lower_groups[0]]
            parking_transition=None
            if stop_anchor is not None:
                # A learned standing mode needs a literal zero velocity input;
                # small continuous PI corrections can keep its walking cycle
                # alive. Enter only after measured braking, resume on drift.
                if parking_supervisor:
                    policy_command,parking_transition=parking_supervisor.update(stamp,bodies[base_name],measured,
                        policy_command,previous_policy_command,dt)
                    parking_mode=parking_transition['mode']=='PARK'
                    if parking_transition['entered']:stop_anchor=parking_supervisor.anchor.copy()
                    if parking_transition['previous_mode']!=parking_transition['mode']:servo.reset()
                    servo.applied(policy_command)
                else:
                    parking_mode,stop_anchor,parking_transition=servo_module.parking_transition(stop_anchor,bodies[base_name],
                        servo.filtered,stamp-case_start-a.move_seconds,parking_mode,reset_anchor_on_entry=a.parking_anchor_reset)
                    if parking_transition['entered'] or parking_transition['exited']:servo.reset()
                if a.robot=='g1' and a.park_posture_mode=='measured' and not parking_supervisor:
                    if parking_transition['entered'] and not parking_transition['exited']:
                        parking_posture={n:float(robot.data.joint_pos[0,robot.joint_names.index(n)]) for n in a.pink.owners['pink']}
                        parking_posture_stamp=stamp
                    if parking_transition['exited']:parking_posture=None;parking_posture_stamp=None
                if parking_mode:
                    policy_command[:]=0.
            desired_camera=camera.copy();support={'reason':'accepted','feasible':None}
            terminal_context=None;terminal_report=None;terminal_error=None;active_brake_context=None
            if session:
                if a.robot=='go2':desired_camera=optical_sampler.sample(min(stamp,task_end))
                else:desired_camera,_=session.reference(min(stamp,task_end))
            if avoidance is not None and avoidance.should_brake(stamp,bodies):
                desired_camera=camera.copy()
                support=dict(reason='environment_evidence_hold',feasible=False)
            collision_evidence=None
            if a.collision:
                collision_evidence=a.collision.poll(stamp)
                a.collision.submit(stamp,bodies)
                if collision_evidence['status']!='pass' and in_case:
                    policy_command[:]=0.;servo.applied(policy_command)
                if collision_evidence['status']=='fail':
                    (a.output/'constraint_abort.json').write_text(json.dumps(dict(stamp_s=stamp,collision=collision_evidence),indent=2)+'\n')
                    failures.append('fine original-shape self clearance below5mm');break
            packet=None;control_begin=time.monotonic()
            requested_policy_command=policy_command.copy();policy_velocity_guard=None
            if a.robot=='g1':
                target_velocity.zero_();target_effort.zero_()
                packet=a.pink.poll(stamp)
                if packet['ready']:
                    predicted=np.asarray(packet['predicted_camera']);acquired=np.asarray(packet['acquisition_camera'])
                    fk_max_position=max(fk_max_position,float(np.linalg.norm(predicted[:3,3]-acquired[:3,3])))
                    fk_max_angle=max(fk_max_angle,float(Rotation.from_matrix(predicted[:3,:3]@acquired[:3,:3].T).magnitude()))
                    if fk_max_position>.005 or fk_max_angle>.01:
                        failures.append('physical camera FK differs from official Pink model');break
                    if packet.get('predicted_collision_bodies'):
                        for n,predicted_body in packet['predicted_collision_bodies'].items():
                            predicted_body=np.asarray(predicted_body);physical=np.asarray(packet['acquisition_bodies'][n])
                            body_fk_max_position=max(body_fk_max_position,float(np.linalg.norm(predicted_body[:3,3]-physical[:3,3])))
                            body_fk_max_angle=max(body_fk_max_angle,float(Rotation.from_matrix(predicted_body[:3,:3]@physical[:3,:3].T).magnitude()))
                        if body_fk_max_position>.001 or body_fk_max_angle>.001:
                            failures.append('actual collision-body FK differs from calibrated pinned model');break
                if not packet['ready']:
                    policy_command=np.clip(np.zeros(3),previous_policy_command-servo.slew*dt,
                        previous_policy_command+servo.slew*dt)
                    collision_veto=bool(a.collision and in_case and collision_evidence['status']!='pass')
                    if collision_veto:policy_command[:]=0.
                    servo.applied(policy_command,transient_hold=not collision_veto)
                    policy_velocity_guard=dict(reason='pink_missing_fresh_command_slew_brake',
                        upper_reference_held=True,upper_velocity_target_zero=True,
                        integral_frozen=not collision_veto,collision_veto=collision_veto)
                if avoidance is not None:
                    policy_command=avoidance.constrain_platform(stamp,bodies,policy_command,previous_policy_command,dt)
                    servo.applied(policy_command)
                command=torch.tensor([[*policy_command,.72]],dtype=torch.float32)
                features=[command,robot.data.root_ang_vel_b,robot.data.projected_gravity_b,
                    robot.data.joint_pos[:,observation_ids]-defaults,robot.data.joint_vel[:,observation_ids]*.1,action.reshape(1,12)]
                action=policy(torch.cat(features,dim=-1).reshape(-1)).clamp(-6.,6.)
                target[:,ids]=offsets+scales*action.reshape(1,12)
                upper_reference_report=None
                if upper_reference is not None and packet['ready']:
                    upper_v=np.array([packet.get('velocity',{}).get(n,0.) if session else 0. for n in a.pink.owners['pink']])
                    reference,upper_reference_report=upper_reference.update(robot.data.joint_pos[0,upper_ids].numpy(),upper_v,dt)
                    target[0,upper_ids]=torch.tensor(reference,dtype=torch.float32)
                    target_velocity[0,upper_ids]=torch.tensor(upper_v,dtype=torch.float32)
                    for n in a.pink.owners['pink']:target_effort[0,robot.joint_names.index(n)]=float(np.clip(packet['gravity'][n],-effort[robot.joint_names.index(n)],effort[robot.joint_names.index(n)]))
                for n,v in (packet.get('velocity',{}) if packet['ready'] and session and upper_reference is None else {}).items():
                    i=robot.joint_names.index(n)
                    position=(robot.data.joint_pos[0,i]+v*.20) if a.upper_mode=='velocity_servo' else (target[0,i]+v*dt)
                    target[0,i]=torch.clamp(position,*robot.data.joint_pos_limits[0,i])
                    target_velocity[0,i]=v
                    if a.upper_mode=='velocity_servo':
                        target_effort[0,i]=float(np.clip(packet['gravity'][n],-effort[i],effort[i]))
                a.pink.submit(dict(zip(robot.joint_names,robot.data.joint_pos[0].tolist())),bodies['pelvis'],
                    desired_camera if session else None,stamp,camera,dt,measured_bodies=bodies if a.fine_collision else None,
                    posture_reference=parking_posture)
                if a.pink.failure_count>=3:
                    (a.output/'pink_runtime_abort.json').write_text(json.dumps(dict(stamp_s=stamp,
                        packet=packet,failure_count=a.pink.failure_count),indent=2)+'\n')
                    failures.append('Pink worker deadline/repeated failure');break
            else:
                if session:
                    end=task_end
                    sample=lambda t:optical_sampler.sample(min(t,end))
                    forecast={}
                    if a.go2_response_profile=='measured_backward_v1':
                        gains=np.where(desired_velocity<0,servo.negative_feedforward_gain,servo.feedforward_gain)
                        forecast=dict(future_policy_velocity=np.clip(desired_velocity*gains,-np.asarray(command_limits),command_limits),
                            previous_applied=previous_policy_command,command_dt=dt)
                    if native_posture_enabled(a.go2_native_posture,cases[selected[case_index]]):
                        forecast.update(native_posture_x_m=.58,
                            posture_weight=coordinate_module.operating_posture_weight(stamp,case_start,task_end))
                    support_horizon=max(0.,end-stamp)
                    if terminal_module is not None and not moving:
                        remaining=max(0.,task_end+a.stop_seconds-stamp)
                        support_horizon=min(coordinator.horizon_s,remaining)
                        sample=lambda t:terminal_endpoint.copy()
                        forecast.pop('native_posture_x_m',None);forecast.pop('posture_weight',None)
                        try:
                            if active_brake_module is not None and stamp>=task_end+2-1e-9:
                                active_brake_context=active_brake_module.LateralStandBrake(backend,stamp_s=stamp,
                                    measurement_stamp_s=observation.stamp_s,task_end_s=task_end,stop_seconds=a.stop_seconds,
                                    body=bodies['base'],link=bodies['link0'],measured_velocity=measured,
                                    previous_applied=previous_policy_command,input_history=coordinator.input_history,
                                    prediction_intervals=coordinator.prediction_intervals(support_horizon),
                                    profile=a.go2_active_stand_brake)
                                policy_command=active_brake_context.first.copy()
                                terminal_report=active_brake_context.manifest()
                                forecast.update(prediction_provider=active_brake_context.predict_link,locked_input=True,
                                    previous_applied=previous_policy_command,command_dt=dt)
                            else:
                                terminal_context=terminal_module.TerminalContext(backend,stamp_s=stamp,task_end_s=task_end,
                                    stop_seconds=a.stop_seconds,endpoint_optical=terminal_endpoint,extrinsic=rig.camera_local,
                                    body=bodies['base'],link=bodies['link0'],measured_velocity=measured,
                                    previous_applied=previous_policy_command,input_history=coordinator.input_history,
                                    target_xy=(.58,.15),prediction_intervals=coordinator.prediction_intervals(support_horizon))
                                policy_command=terminal_context.solve()
                                terminal_report=terminal_context.manifest()
                                forecast.update(prediction_provider=terminal_context.predict_link,
                                    locked_input=terminal_context.phase!='allocate',
                                    previous_applied=previous_policy_command,command_dt=dt)
                        except (ValueError,np.linalg.LinAlgError)as error:
                            # Continue physical time with the ORIGINAL slew
                            # brake. An impossible deadline remains a failed
                            # Task, never a jump to zero or a retimed reference.
                            terminal_error=str(error)
                            if isinstance(error,terminal_module.TerminalDeadlineInfeasible):terminal_deadline_failed=True
                            terminal_context=None
                            policy_command=np.clip(np.zeros(3),previous_policy_command-np.array([.5,.5,.8])*dt,
                                previous_policy_command+np.array([.5,.5,.8])*dt)
                            support_horizon=0.
                            terminal_report=dict(profile=a.go2_terminal_profile,stamp_s=stamp,reference_stamp_s=task_end,
                                reference_mode='explicit_original_endpoint_hold',error=terminal_error,
                                deadline_failure_latched=terminal_deadline_failed,safe_original_slew_brake=True,
                                original_world_task_modified=False,all18_learned_actions_unchanged=True)
                    support=camera_support_with_terminal_rejection(coordinator,stamp,sample,bodies['link0'],bodies['base'],
                        policy_command,measured,previous_policy_command,dt,support_horizon,forecast,
                        terminal_error,terminal_deadline_failed)
                    if terminal_error is not None or terminal_deadline_failed:
                        support.update(feasible=False,reason='tracking_limit'if terminal_deadline_failed else'command_stale',
                            camera_supported=False,base_velocity_feasible=False)
                    policy_command=np.asarray(support['base_command']);servo.applied(policy_command)
                    requested_native=np.asarray(support['native_command']).copy()
                    requested_domain=coordinator.domain.check(requested_native)
                    feedback_overrides=[]
                    if world_z_feedback is None:
                        native,corrected=coordinate_module.feedback_native_command(backend,desired_camera,camera,bodies['link0'],rig.camera_local,
                            domain=coordinator.domain)
                    else:
                        native,corrected=world_z_feedback.feedback_native_command(backend,desired_camera,camera,bodies['link0'],rig.camera_local,
                            task_id=session.task_id,stamp_s=observation.stamp_s,now_s=stamp,dt_s=dt,sample_seq=step,
                            fresh=True,integration_allowed=bool(support.get('camera_supported',False)
                                and requested_domain['status']=='inside_outer_domain_unverified'
                                and (not a.collision or collision_evidence['status']=='pass')
                                and not (terminal_native_hold is not None and terminal_native_hold.in_window(stamp))),domain=coordinator.domain)
                    feedback_check=corrected
                    native,parking_input= native_parking.update(native,desired_camera,camera,bodies['base'],measured,
                        parking=a.parking_camera_hold and parking_mode and not moving
                            and requested_domain['status']=='inside_outer_domain_unverified')
                    if parking_input['active']:feedback_overrides.append('native_parking_hold')
                    terminal_hold_report=None
                    if terminal_native_hold is not None:
                        native,terminal_hold_report=terminal_native_hold.select(native,stamp_s=stamp,
                            measurement_stamp_s=observation.stamp_s,previous_consumed_native=ee[0].numpy(),
                            previous_issue_stamp_s=stamp-dt,previous_consumption_seq=step-1,
                            previous_body_input=previous_policy_command,issued_body_input=policy_command,
                            domain=coordinator.domain,collision_evidence=collision_evidence,
                            original_support_ok=bool(support.get('feasible')is not False and
                                requested_domain['status']=='inside_outer_domain_unverified'),
                            world_position_error_m=parking_input['world_position_error_m'],
                            world_orientation_error_rad=parking_input['world_orientation_error_rad'])
                        if terminal_hold_report['selected']:feedback_overrides.append('terminal_native_hold')
                    rejected=corrected.get('rejected_feedback_correction')
                    recovery_candidate=requested_native if requested_domain['status']!='inside_outer_domain_unverified' else native
                    native,recovery_command,recovery_report=domain_recovery.update(recovery_candidate,bodies['link0'],bodies['base'],
                        stamp,previous_policy_command,dt)
                    if recovery_command is not None:
                        feedback_overrides.append('native_domain_recovery')
                        policy_command=np.asarray(recovery_command);servo.applied(policy_command)
                        support.update(feasible=False,reason='out_of_policy_domain',camera_supported=False,
                            base_velocity_feasible=False,domain_recovery=recovery_report)
                    corrected=coordinator.domain.check(native)
                    if rejected is not None:corrected['rejected_feedback_correction']=rejected
                    if corrected['status']!='inside_outer_domain_unverified':
                        finalize_native_feedback(world_z_feedback,feedback_check,corrected,['actual_native_domain_failure'])
                        raise RuntimeError('domain failure escaped legal native-input hold guard')
                    if a.collision and collision_evidence['status']!='pass':
                        feedback_overrides.append('collision_freshness_veto')
                        # Geometry freshness veto has priority over task-domain
                        # projection and recovery; never override this brake.
                        policy_command=np.clip(np.zeros(3),previous_policy_command-np.array([.5,.5,.8])*dt,
                            previous_policy_command+np.array([.5,.5,.8])*dt)
                        servo.applied(policy_command)
                    if active_brake_context is not None:
                        brake_vetoes=[]
                        if support.get('feasible')is False:brake_vetoes.append('original_forecast_rejected')
                        if a.collision and collision_evidence['status']!='pass':brake_vetoes.append('collision_freshness_veto')
                        if recovery_command is not None:brake_vetoes.append('native_domain_recovery')
                        if terminal_deadline_failed:brake_vetoes.append('prior_terminal_deadline_failure')
                        policy_command=finalize_active_stand_brake(active_brake_context,policy_command,
                            support,feedback_overrides,brake_vetoes)
                        servo.applied(policy_command)
                    elif terminal_context is not None or terminal_error is not None or terminal_deadline_failed:
                        issued,forced_brake=final_terminal_body_command(policy_command,previous_policy_command,dt,
                            terminal_context,terminal_error,terminal_deadline_failed)
                        terminal_report.update(final_body_input=issued.tolist(),actual_standing_input=bool(np.all(issued==0)),
                            later_body_veto_changed=not np.allclose(issued,policy_command,atol=1e-12,rtol=0),
                            rejected_forecast_final_brake=forced_brake,
                            deadline_failure_latched=terminal_deadline_failed)
                        if forced_brake:feedback_overrides.append('terminal_rejected_forecast_body_brake')
                        policy_command=issued;servo.applied(policy_command)
                    if support.get('feasible') is False:feedback_overrides.append('original_camera_support_rejected')
                    corrected=finalize_native_feedback(world_z_feedback,feedback_check,corrected,feedback_overrides)
                    ee=torch.tensor(native,dtype=torch.float32).reshape(1,7)
                else:
                    z=float(bodies['link0'][2,3]);pitch=-math.atan2(.5-z,.5)
                    ee=torch.tensor([[.5,0.,.5,math.cos(pitch/2),0.,math.sin(pitch/2),0.]])
                    domain_recovery.seed(ee[0].numpy())
                if avoidance is not None:
                    policy_command=avoidance.constrain_platform(stamp,bodies,policy_command,previous_policy_command,dt)
                    servo.applied(policy_command)
                features=[robot.data.root_ang_vel_b*.2,robot.data.projected_gravity_b,
                    robot.data.joint_pos[:,ids]-robot.data.default_joint_pos[:,ids],
                    (robot.data.joint_vel[:,ids]-robot.data.default_joint_vel[:,ids])*.05,
                    action.reshape(1,18),torch.tensor(policy_command,dtype=torch.float32).reshape(1,3),ee]
                histories=[f.repeat(1,3) for f in features] if histories is None else [torch.cat([h[:,f.shape[1]:],f],dim=-1) for h,f in zip(histories,features)]
                if in_case and corrected.get('world_z_feedback',{}).get('seed_pending_commit'):
                    action=consume_policy_with_seed_confirmation(
                        lambda:policy(torch.cat(histories,dim=-1).clamp(-100.,100.)).reshape(-1).clamp(-10.,10.),
                        world_z_feedback,corrected,ee[0].numpy())
                else:
                    action=policy(torch.cat(histories,dim=-1).clamp(-100.,100.)).reshape(-1).clamp(-10.,10.)
                if in_case and terminal_native_hold is not None and terminal_hold_report is not None:
                    terminal_native_hold.confirm(terminal_hold_report,actual_native=ee[0].numpy(),
                        actual_body_input=policy_command,stamp_s=stamp,consumption_seq=step,override_reasons=feedback_overrides)
                target[:,ids]=robot.data.default_joint_pos[:,ids]+.25*action.reshape(1,18)
            previous_policy_command=policy_command.copy()
            if a.robot=='go2':coordinator.observe_applied_input(stamp,policy_command)
            physics_begin=time.monotonic();effort_peak=np.zeros(len(robot.joint_names));velocity_peak=effort_peak.copy()
            substep_joint_margin=np.inf
            for substep in range(4):
                robot.set_joint_position_target(target)
                if a.robot=='g1' and (a.upper_velocity_feedforward or a.upper_mode in ('velocity_servo','bounded_integral')):
                    robot.set_joint_velocity_target(target_velocity)
                    robot.set_joint_effort_target(target_effort)
                robot.write_data_to_sim()
                effort_peak=np.maximum(effort_peak,np.abs(robot.data.applied_torque[0].numpy()))
                if avoidance is not None:avoidance.update_scene(stamp+(substep+1)*physics_dt)
                sim.step(render=False);robot.update(physics_dt)
                if velocity_audit and (not a.velocity_audit_tail_seconds or (in_case and not moving and
                        stamp+(substep+1)*physics_dt>task_end+a.stop_seconds-a.velocity_audit_tail_seconds+1e-9)):
                    velocity_audit.record(stamp+(substep+1)*physics_dt)
                velocity_peak=np.maximum(velocity_peak,np.abs(robot.data.joint_vel[0].numpy()))
                sampled_q=robot.data.joint_pos[0].numpy()
                substep_joint_margin=min(substep_joint_margin,float(np.min(np.minimum(sampled_q-limits[:,0],limits[:,1]-sampled_q))))
            measurement_begin=time.monotonic()
            current_measurement=measure((step+1)*dt)
            if avoidance is not None:
                avoidance.update((step+1)*dt,*current_measurement,
                    native_joint_positions=dict(zip(robot.joint_names,robot.data.joint_pos[0].tolist())))
            bodies,camera,observation=current_measurement;base=bodies[base_name]
            feedback_begin=time.monotonic()
            if session:
                if a.robot=='go2':desired_camera=optical_sampler.sample(min((step+1)*dt,task_end))
                else:desired_camera,_=session.reference(min((step+1)*dt,task_end))
            q=robot.data.joint_pos[0].numpy();tilt=float(np.arccos(np.clip(base[2,2],-1.,1.)))
            velocity_world=robot.data.root_lin_vel_w[0].tolist();angular_world=robot.data.root_ang_vel_w[0].tolist()
            row=dict(stamp_s=(step+1)*dt,case=selected[case_index] if in_case else 'settle',phase='move' if moving else 'stop',
                T_world_base=base.tolist(),T_world_camera_optical=camera.tolist(),T_world_reference_optical=desired_camera.tolist(),
                joint_positions=q.tolist(),joint_names=robot.joint_names,requested_body_velocity=desired_velocity.tolist(),
                measured_body_velocity=np.r_[robot.data.root_lin_vel_b[0,:2].numpy(),robot.data.root_ang_vel_b[0,2].item()].tolist(),
                applied_policy_velocity=policy_command.tolist(),base_linear_velocity_world=velocity_world,base_angular_velocity_world=angular_world,
                native_standing_mode=(bool(not moving and np.all(policy_command==0))
                    if a.robot=='go2'and a.go2_terminal_profile!='none'else parking_mode),
                parking_transition=parking_transition,
                native_lower_damping_scale=leg_damping_scale,native_policy_action=action.tolist(),
                native_pd_damping_scales=dict(damping_scales),
                applied_joint_position_target=target[0].tolist(),measured_joint_velocity=robot.data.joint_vel[0].tolist(),
                applied_joint_effort_estimate=robot.data.applied_torque[0].tolist(),
                applied_joint_effort_peak_abs=effort_peak.tolist(),
                measured_joint_velocity_peak_abs=velocity_peak.tolist(),minimum_substep_joint_margin_rad=substep_joint_margin,
                position_error_m=float(np.linalg.norm(camera[:3,3]-desired_camera[:3,3])),
                orientation_error_rad=float(Rotation.from_matrix(desired_camera[:3,:3]@camera[:3,:3].T).magnitude()),
                original_joint_margin_min_rad=float(np.min(np.minimum(q-limits[:,0],limits[:,1]-q))),
                base_height_m=float(base[2,3]),base_tilt_rad=tilt,measured_observation=observation.to_dict(),
                controller_status=packet['status'] if packet else support['reason'])
            if a.robot=='g1':
                row['requested_policy_velocity']=requested_policy_command.tolist()
                row['policy_velocity_guard']=policy_velocity_guard
                row['pink_runtime_recovery']={k:packet[k] for k in ('seq','stamp_s','acquisition_stamp_s',
                    'discarded_results','replacement_allowed','outstanding_requests') if k in packet}
                row['pink_upper_velocity']=packet.get('velocity',{}) if packet['ready'] else {}
                row['upper_gravity_feedforward']=packet.get('gravity',{}) if packet['ready'] else {}
                row['upper_effort_peak_fraction']=float(np.max(effort_peak[upper_ids]/effort[upper_ids]))
                row['pink_collision_report']=packet.get('pink_collision_report') if packet['ready'] else None
                row['pink_parking_posture']=dict(stamp_s=parking_posture_stamp,targets=parking_posture,
                    kind='soft_measured_posture_preference',camera_frame_task_unchanged=True,collision_barriers_unchanged=True)
                row['pink_posture_report']=packet.get('pink_posture_report') if packet['ready'] else None
                row['upper_reference_report']=upper_reference_report
                row['measured_upper_velocity_error_rad_s']={n:float(robot.data.joint_vel[0,robot.joint_names.index(n)])-packet.get('velocity',{}).get(n,0.) for n in a.pink.owners['pink']}
                if a.upper_mode in ('velocity_servo','bounded_integral') and row['upper_effort_peak_fraction']>1.00001:
                    failures.append('total explicit upper effort exceeds original cap');break
            if session:
                q_margin=row['original_joint_margin_min_rad']
                q_range=limits[:,1]-limits[:,0]
                q_reserve=float(np.clip(np.min(np.minimum(q-limits[:,0],limits[:,1]-q)/np.maximum(q_range/2,1e-9)),0.,1.))
                native_domain_pass=bool(np.all(np.abs(policy_command)<=np.asarray(command_limits)+1e-8))
                if a.robot=='go2':
                    native_domain_pass=bool(native_domain_pass and corrected['status']=='inside_outer_domain_unverified')
                constraints={
                    'configuration':dict(status='pass' if q_margin>=-1e-6 else 'fail',stamp_s=observation.stamp_s,
                        scope='measured_native_limits',reserve_fraction=q_reserve),
                    # Implicit PD drive torque estimates and zero contact force
                    # are not certified measurements of all actuation/contacts.
                    'actuation':dict(status='unknown',stamp_s=observation.stamp_s,scope='not_available'),
                    'self_collision':dict(status='unknown',stamp_s=observation.stamp_s,scope='not_available'),
                    'environment':(avoidance.environment(observation.stamp_s,bodies) if avoidance is not None else dict(status='unknown',stamp_s=observation.stamp_s,scope='not_available')),
                    'policy_domain':dict(status='pass' if native_domain_pass else 'fail',stamp_s=observation.stamp_s,
                        scope='policy_config_outer_domain')}
                effort_fraction=float(np.max(effort_peak/effort))
                speed_fraction=float(np.max(velocity_peak/robot.data.joint_vel_limits[0].numpy()))
                row['native_effort_peak_fraction']=effort_fraction
                row['native_speed_peak_fraction']=speed_fraction
                if all(not x.is_implicit_model for x in robot.actuators.values()):
                    # Explicit actuator applied_torque is the actual bounded
                    # command issued to PhysX, including saturation. This is
                    # a command/model audit, not a joint torque sensor.
                    constraints['actuation']=dict(status='pass' if max(effort_fraction,speed_fraction)<=1.00001 else 'fail',
                        stamp_s=observation.stamp_s,scope='measured_native_limits',
                        reserve_fraction=float(np.clip(1-max(effort_fraction,speed_fraction),0.,1.)))
                if collision_evidence and collision_evidence['status'] in ('pass','fail'):
                    constraints['self_collision']=dict(status=collision_evidence['status'],stamp_s=collision_evidence['stamp_s'],
                        scope='measured_collision_model',clearance_m=collision_evidence['minimum_clearance_lower_bound_m'])
                    row['fine_collision_evidence']=collision_evidence
                row['task_feedback']=session.feedback(observation,feasible=support.get('feasible'),reason=support['reason'],
                    hold_terminal=True,constraints=constraints,now_s=observation.stamp_s)
            if a.robot=='go2' and session:
                row['camera_support']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in support.items()}
                row['actual_native_command']=native.tolist()
                row['actual_native_domain']={k:v for k,v in corrected.items() if k!='rejected_feedback_correction'}
                row['rejected_feedback_correction']=corrected.get('rejected_feedback_correction')
                row['native_camera_parking']=parking_input
                row['requested_native_command']=requested_native.tolist()
                row['requested_native_domain']=requested_domain
                row['native_domain_recovery']=recovery_report
                if terminal_native_hold is not None:row['terminal_native_hold']=terminal_hold_report
                if a.go2_terminal_profile!='none':
                    row['terminal_allocation']=terminal_report
                    row['parking_controller_mode']=parking_mode
            rows.append(row)
            if in_case:case_rows.append(row)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            timing_totals+=np.diff([tick_start,control_begin,physics_begin,measurement_begin,feedback_begin,time.monotonic()])
            if not np.isfinite(q).all() or substep_joint_margin < -1e-6 or base[2,3] < (.5 if a.robot=='g1' else .20) or tilt > (.6 if a.robot=='g1' else .65):
                failures.append('original joint/finite/balance safety gate');break
            if step%100==0:print(f'{a.robot} STEP {step}/{steps} {row["case"]} {row["phase"]}',flush=True)
            delay=start_wall+(step+1)*dt-time.monotonic()
            if delay>0:time.sleep(delay)
    elapsed=time.monotonic()-start_wall
    if velocity_audit:
        velocity_audit.close()
    if case_rows:results.append(summarize(selected[case_index],case_rows,a,audit_stop))
    if avoidance is not None and failures:avoidance.execution_failure='; '.join(failures)
    physical_pass=bool(len(rows)==steps and not failures and all(r['passed'] for r in results))
    timing_pass=elapsed<=len(rows)*dt*1.05
    result=dict(schema='legged_adaptive_physical/v1',robot=a.robot,run_complete=len(rows)==steps,
        all_pass=bool(physical_pass and timing_pass),physical_tracking_and_stop_pass=physical_pass,failures=failures,cases=results,
        sim_seconds=len(rows)*dt,wall_seconds=elapsed,realtime_factor=len(rows)*dt/elapsed,timing_pass=timing_pass,
        physical_policy='AGILE height80/12 + genuine Pink15' if a.robot=='g1' else 'LeggedManip Go2-Piper WBC210/18',
        camera_reference_modified=False,public_timed_task_actual_ingress=True,measured_two_part_collider_feedback=True,
        maximum_measured_fk_position_error_m=fk_max_position,maximum_measured_fk_angle_error_rad=fk_max_angle,
        maximum_collision_body_fk_position_error_m=body_fk_max_position,maximum_collision_body_fk_angle_error_rad=body_fk_max_angle,
        mean_control_phase_wall_ms=dict(zip(['task_servo','native_controller','physics','measurement_geometry','feedback_log'],
            (timing_totals/max(len(rows),1)*1000).tolist())),
        environment_constraints_verified=False,sensors_avoidance_verified=False)
    collision_rows=[r for r in rows if r['case']!='settle']
    collision_status=[r.get('task_feedback',{}).get('constraints',{}).get('self_collision',{}).get('status','unknown') for r in collision_rows]
    result['fine_collision']=dict(enabled=bool(a.fine_collision),active_samples=len(collision_rows),
        status_counts={s:collision_status.count(s) for s in sorted(set(collision_status))},
        all_active_samples_fresh_pass=bool(collision_status and all(s=='pass' for s in collision_status)),
        future_swept_collision_certified=False)
    result['constraint_status_counts']={family:{status:sum(r.get('task_feedback',{}).get('constraints',{}).get(family,{}).get('status','unknown')==status for r in collision_rows)
        for status in ('pass','fail','unknown','not_applicable')}
        for family in ('configuration','actuation','self_collision','environment','policy_domain')}
    result['known_constraint_failure_free']=not any(counts['fail'] for counts in result['constraint_status_counts'].values())
    result['task_infeasible_samples']=sum(r.get('task_feedback',{}).get('feasible') is False for r in collision_rows)
    if a.robot=='go2':result['requested_native_domain_failure_samples']=sum(r.get('requested_native_domain',{}).get('status')!='inside_outer_domain_unverified' for r in collision_rows)
    result['all_pass']=bool(result['all_pass'] and result['known_constraint_failure_free'] and
        result['task_infeasible_samples']==0 and (not a.fine_collision or result['fine_collision']['all_active_samples_fresh_pass']))
    if a.robot=='go2'and a.go2_world_z_seed!='none':
        (a.output/'world_z_seed_calibrations.json').write_text(json.dumps(dict(
            schema='go2_fresh_admission_z_seed/v1',scope='admission-only measured calibration; original Task and clocks unchanged',
            records=world_z_feedback.calibrations),indent=2,allow_nan=False)+'\n')
    if a.robot=='go2'and terminal_native_hold is not None:
        (a.output/'terminal_native_hold_manifests.json').write_text(json.dumps(dict(
            scope='Original WORLD Task unchanged; learned18 active; only actually consumed nativeEE7 held afterend+2.',
            records=terminal_hold_manifests+[terminal_native_hold.manifest()]),indent=2,allow_nan=False)+'\n')
    if a.rgb:result['camera_render']=rig.capture_camera(a.output,sim.render,len(rows)*dt)
    stage.Flatten().Export(str(a.output/'tested_scene.usda'))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return 0 if result['all_pass'] else 2


def summarize(name,rows,a,audit_stop):
    import numpy as np
    from scipy.spatial.transform import Rotation
    moving=[r for r in rows if r['phase']=='move']
    tail=moving[round(len(moving)/2):]
    mean=np.mean([r['measured_body_velocity'] for r in tail],axis=0)
    desired=np.asarray(tail[0]['requested_body_velocity'])
    tracking=bool(np.all(np.abs(mean-desired)<=np.maximum(.025,.25*np.abs(desired))))
    errors=np.array([[r['position_error_m'],r['orientation_error_rad']] for r in moving])
    rms=np.sqrt(np.mean(errors**2,axis=0));camera_pass=bool(rms[0]<.05 and rms[1]<.15
        and rows[-1]['position_error_m']<.05 and rows[-1]['orientation_error_rad']<.15)
    last=rows[-1]
    stop=audit_stop([r['stamp_s'] for r in rows],[r['T_world_base'] for r in rows],
        [r['T_world_camera_optical'] for r in rows],[r['joint_positions'] for r in rows],
        ['rad']*len(last['joint_positions']),last['base_linear_velocity_world'],last['base_angular_velocity_world'])
    result=dict(case=name,measured_mean_vx_vy_wz=mean.tolist(),requested_vx_vy_wz=desired.tolist(),
        velocity_tracking_pass=tracking,camera_position_rmse_m=float(rms[0]),camera_orientation_rmse_rad=float(rms[1]),
        camera_tracking_pass=camera_pass,strict_stop=stop,passed=bool(tracking and camera_pass and stop['passed']))
    if name=='camera_6d' or (name=='camera_shot' and a.camera_excursion=='nontrivial'):
        reference=np.asarray([r['T_world_reference_optical'] for r in moving]);actual=np.asarray([r['T_world_camera_optical'] for r in moving])
        initial=reference[0]
        hold_position=np.linalg.norm(reference[:,:3,3]-initial[:3,3],axis=1)
        hold_rotation=Rotation.from_matrix(reference[:,:3,:3]@initial[:3,:3].T).magnitude()
        baseline=np.sqrt(np.mean(np.c_[hold_position,hold_rotation]**2,axis=0))
        rotation_span=lambda matrices:float(max(np.max(Rotation.from_matrix(matrices@R.T).magnitude()) for R in matrices))
        ref_span=float(np.linalg.norm(np.ptp(reference[:,:3,3],axis=0)));ref_angle=rotation_span(reference[:,:3,:3])
        ratios=rms/np.maximum(baseline,1e-12)
        joint_span=np.ptp(np.asarray([r['joint_positions'] for r in moving]),axis=0)
        arm_ids=[i for i,n in enumerate(rows[0]['joint_names']) if ('shoulder_' in n or 'elbow_joint' in n or 'wrist_' in n or n=='waist_yaw_joint'
            or (n.startswith('joint') and n[5:].isdigit()))]
        arm_span=float(np.linalg.norm(joint_span[arm_ids]))
        response=bool(ref_span>=.10 and ref_angle>=.30 and np.all(ratios<=.65) and arm_span>=.10)
        full_base=np.asarray([r['T_world_base'] for r in rows]);moving_base=np.asarray([r['T_world_base'] for r in moving])
        relative_camera=np.linalg.inv(moving_base)@actual
        result['six_axis_response']=dict(reference_position_span_m=ref_span,reference_rotation_span_rad=ref_angle,
            measured_camera_position_span_m=float(np.linalg.norm(np.ptp(actual[:,:3,3],axis=0))),
            measured_camera_rotation_span_rad=rotation_span(actual[:,:3,:3]),
            full_case_base_position_span_m=float(np.linalg.norm(np.ptp(full_base[:,:3,3],axis=0))),
            full_case_base_rotation_span_rad=rotation_span(full_base[:,:3,:3]),
            moving_base_position_span_m=float(np.linalg.norm(np.ptp(moving_base[:,:3,3],axis=0))),
            moving_base_rotation_span_rad=rotation_span(moving_base[:,:3,:3]),
            measured_camera_in_base_position_span_m=float(np.linalg.norm(np.ptp(relative_camera[:,:3,3],axis=0))),
            measured_camera_in_base_rotation_span_rad=rotation_span(relative_camera[:,:3,:3]),
            arm_joint_names=[rows[0]['joint_names'][i] for i in arm_ids],arm_joint_span_rad=joint_span[arm_ids].tolist(),
            reference_position_axis_span_m=np.ptp(reference[:,:3,3],axis=0).tolist(),
            measured_position_axis_span_m=np.ptp(actual[:,:3,3],axis=0).tolist(),
            initial_pose_hold_baseline_position_rmse_m=float(baseline[0]),initial_pose_hold_baseline_orientation_rmse_rad=float(baseline[1]),
            rmse_over_hold_baseline=ratios.tolist(),measured_arm_joint_span_norm_rad=arm_span,passed=response)
        result['passed']=bool(result['passed'] and response)
    return result


if __name__=='__main__':
    raise SystemExit(main())
