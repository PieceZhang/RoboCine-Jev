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


class PreparedSimulationGarbage:
    """Keep the initialized simulator graph out of cyclic scans during one run.

    Automatic collection stays enabled for newly allocated runtime objects.
    Preparation happens before the physical loop, and the prior freeze state
    is restored on every exit. Existing external freeze ownership is preserved.
    """
    def __init__(self,enabled):
        self.enabled=enabled;self.owned=False;self.threshold=None;self.report=dict(enabled=enabled)

    def __enter__(self):
        if self.enabled:
            import gc
            self.report.update(automatic_gc_enabled=gc.isenabled(),
                               previous_frozen_objects=gc.get_freeze_count())
            self.threshold=gc.get_threshold()
            # This context is used only for the finite 8 s avoidance shot plus
            # pre-roll/stop. Keep young-generation collection and refcounting;
            # defer scanning the accumulating immutable audit history until exit.
            gc.set_threshold(self.threshold[0],self.threshold[1],1000000)
            self.report.update(previous_threshold=list(self.threshold),
                               runtime_threshold=list(gc.get_threshold()),
                               full_history_scan_deferred_until_exit=True)
            if not gc.get_freeze_count():
                started=time.monotonic()
                self.report['startup_collected_objects']=gc.collect()
                gc.freeze();self.owned=True
                self.report.update(frozen_objects=gc.get_freeze_count(),
                                   preparation_wall_s=time.monotonic()-started)
        return self

    def __exit__(self,*unused):
        if self.threshold is not None:
            import gc
            gc.set_threshold(*self.threshold)
        if self.owned:
            import gc
            gc.unfreeze()
            self.report['restored']=gc.get_freeze_count()==0


class GarbageCollectionTiming:
    """Observe existing collections; never trigger one or change GC policy."""
    def __init__(self,enabled):
        self.enabled=enabled;self.events=[];self.active={}

    def __enter__(self):
        if self.enabled:
            import gc
            self.settings=dict(enabled=gc.isenabled(),threshold=list(gc.get_threshold()),
                               frozen_objects=gc.get_freeze_count())
            gc.callbacks.append(self._observe)
        return self

    def _observe(self,phase,info):
        now=time.monotonic();generation=info['generation']
        if phase=='start':self.active[generation]=now
        elif phase=='stop':
            started=self.active.pop(generation,None)
            self.events.append(dict(generation=generation,start_wall_s=started,end_wall_s=now,
                elapsed_wall_s=None if started is None else now-started,
                collected=info.get('collected'),uncollectable=info.get('uncollectable')))

    def __exit__(self,*unused):
        if self.enabled:
            import gc
            gc.callbacks.remove(self._observe)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def native_g1_velocity_parameters(servo_module,profile):
    """Phase-only G1 overrides retain the actual native baseline factory."""
    factory_profile='baseline' if profile in ('yaw_gain2','yaw_phase_2_3_v1',
        'yaw_pi08_2_ff1_v1','yaw_phase_symmetric2_v1','yaw_phase_symmetric2_ki4_v1') else profile
    return servo_module.g1_profile(factory_profile)


def current_state_gravity_support(model,*,joint_names,joint_positions,pelvis_rotation,
        stamp_s,physics_state_seq):
    """Calculate static support from this physical state, never a solver packet.

    The caller keeps the last committed position reference and zero velocity.
    This function neither accepts a motion lease nor clamps/executes torque.
    """
    import math
    import numpy as np
    if (isinstance(stamp_s,bool) or not isinstance(stamp_s,(int,float))
            or not math.isfinite(stamp_s) or stamp_s<0 or isinstance(physics_state_seq,bool)
            or not isinstance(physics_state_seq,int) or physics_state_seq<0):
        raise ValueError('current physical state stamp and sequence required')
    names=list(joint_names)
    if len(names)!=29 or len(set(names))!=29 or set(names)!=set(model.joint_names):
        raise ValueError('complete measured physical29 names required')
    q=np.asarray(joint_positions)
    if q.shape!=(29,) or q.dtype.kind not in 'fiu' or not np.isfinite(q).all():
        raise ValueError('finite current measured physical29 positions required')
    rotation=np.asarray(pelvis_rotation)
    gravity=model.upper_torque(dict(zip(names,q.tolist())),rotation)
    if set(gravity)!=set(model.upper_joint_names) or not np.isfinite(list(gravity.values())).all():
        raise ValueError('current static model must return exactly the original upper15')
    return gravity,dict(stamp_s=float(stamp_s),physics_state_seq=physics_state_seq,
        measurement_source='current_measured_physx',joint_names=names,
        joint_positions_rad=q.tolist(),pelvis_rotation_world=rotation.tolist(),
        computed_gravity_nm=dict(gravity),trajectory_reused=False,
        worker_packet_gravity_reused=False)


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


def include_terminal_abort(result, terminal_abort=None, evidence_sha256=None):
    """Keep an abort between measurements separate from the sampled prefix.

    Do not synthesize a physical row or change any row counts. A passing prefix
    cannot hide the final asynchronous native safety result.
    """
    collision=result['fine_collision']
    sampled_pass=collision['all_active_samples_fresh_pass']
    collision['completed_active_samples_fresh_pass']=sampled_pass
    collision['all_active_samples_fresh_pass']=bool(sampled_pass and result['run_complete'] and not terminal_abort)
    result['constraint_abort_detected']=terminal_abort is not None
    result['terminal_constraint_abort']=terminal_abort
    result['constraint_abort_evidence']=(dict(path='constraint_abort.json',sha256=evidence_sha256)
        if terminal_abort is not None else None)
    result['task_infeasible_terminal_aborts']=int(bool(terminal_abort and
        terminal_abort.get('public_task_status',{}).get('feasible') is False))
    if terminal_abort is not None:
        result['known_constraint_failure_free']=False
        result['physical_tracking_and_stop_pass']=False
        result['all_pass']=False
        result['final_public_task_status']=terminal_abort.get('public_task_status')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=['g1', 'go2'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', default='forward,backward,left,right,diagonal_left,diagonal_right,yaw_left,yaw_right,camera_shot')
    p.add_argument('--move-seconds', type=float, default=5.)
    p.add_argument('--stop-seconds', type=float, default=3.)
    p.add_argument('--speed', type=float, default=.10)
    p.add_argument('--yaw-speed', type=float, default=.12)
    p.add_argument('--upper-damping-scale',type=float,default=3.,help='G1 native upper PD damping multiplier; original effort/position caps preserved')
    p.add_argument('--g1-arm-stiffness-scale',type=float,choices=[1.,2.,4.],default=1.,help='Explicit G1 arm-only PD comparison; waist and all original physical caps retained')
    p.add_argument('--g1-park-yaw-assist',action='store_true',help='Optional PARK-only measured native yaw input; XYzero, original parking drift exit retained')
    p.add_argument('--g1-refresh-posture-on-park',action='store_true',help='Once per Task, refresh the soft15joint posture preference at the original first measured PARK admission')
    p.add_argument('--upper-velocity-feedforward',action='store_true',help='Diagnostic native PD velocity feedforward')
    p.add_argument('--upper-mode',choices=['position_integral','velocity_servo','bounded_integral'],default='velocity_servo')
    p.add_argument('--park-damping-scale',type=float,default=3.,help='Smooth native lower-body damping during stop; learned targets unchanged')
    p.add_argument('--park-upper-damping-scale',type=float,default=1.,help='Go2 native arm damping multiplier during declared stop')
    p.add_argument('--parking-camera-hold',action='store_true',help='Optional native-input hold diagnostic; not a validated improvement')
    p.add_argument('--parking-anchor-reset',action='store_true',help='Optional measured brake-pose anchor diagnostic')
    p.add_argument('--parking-profile',choices=['legacy','brake_settle_v1'],default='legacy',help='G1 native-input stand transition candidate')
    p.add_argument('--park-posture-mode',choices=['home','measured'],default='home',help='G1 soft PostureTask preference after measured stand entry')
    p.add_argument('--waist-posture-cost',type=float,default=None,help='G1 soft home preference; camera and collision tasks retain waist authority')
    p.add_argument('--walking-posture-cost',type=float,default=None,help='G1 native soft home preference during locomotion tasks only')
    p.add_argument('--terminal-damping-cost',type=float,default=None,help='G1 native soft zero-velocity preference after locomotion tasks only')
    p.add_argument('--fine-collision',action='store_true',help='Audit original geometry with an isolated existing FCL worker')
    p.add_argument('--g1-following-reserve',action='store_true',help='Conditional measured following reserve; actual braking remains unverified')
    p.add_argument('--g1-velocity-profile',choices=['baseline','low_gain','yaw_gain2','yaw_phase_2_3_v1','yaw_pi08_2_ff1_v1','yaw_phase_symmetric2_v1','yaw_phase_symmetric2_ki4_v1'],default='baseline')
    p.add_argument('--g1-gravity-profile',choices=['urdf','actual_physx_v1'],default='urdf',
        help='Separate measured mass/COM gravity-only model; leaves original IK, collision and physical plant unchanged')
    p.add_argument('--g1-gravity-hold-profile',choices=['zero','fresh_current_state_v1'],default='zero',
        help='Optional static measured-state support while upper motion has no fresh command; never extends its lease')
    p.add_argument('--g1-collision-gain',type=float,default=.1,
        help='Native Pink class-K convergence rate; original 6.5mm barrier, 5mm physical guard and joint caps retained')
    p.add_argument('--go2-response-profile',choices=['legacy','measured_backward_v1'],default='legacy')
    p.add_argument('--go2-native-posture',choices=['none','x58_moving_v1'],default='none')
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
    if a.g1_gravity_profile!='urdf' and (a.robot!='g1' or a.upper_mode not in ('velocity_servo','bounded_integral')):
        raise ValueError('actual G1 gravity requires its single explicit PD-plus-gravity capped execution path')
    if a.g1_gravity_hold_profile!='zero' and (a.robot!='g1' or a.g1_gravity_profile!='actual_physx_v1'
            or a.upper_mode!='bounded_integral' or a.g1_following_reserve):
        raise ValueError('fresh G1 static hold requires the original actual-gravity capped path without conditional following reserve')
    if a.g1_following_reserve and (a.robot!='g1' or not a.fine_collision or a.upper_mode!='bounded_integral'):
        raise ValueError('G1 following reserve requires real collision geometry and bounded-integral execution')
    if a.g1_arm_stiffness_scale!=1. and (a.robot!='g1' or a.upper_mode not in ('velocity_servo','bounded_integral')):
        raise ValueError('arm stiffness comparison requires the G1 single explicit capped PD path')
    if a.g1_park_yaw_assist and (a.robot!='g1' or a.parking_profile!='brake_settle_v1' or not a.fine_collision):
        raise ValueError('G1 PARK yaw assist requires the original measured parking supervisor and fine collision guard')
    if a.g1_refresh_posture_on_park and (a.robot!='g1' or a.parking_profile!='brake_settle_v1' or a.park_posture_mode!='measured'):
        raise ValueError('G1 soft posture refresh requires the original measured parking supervisor and measured preference')
    if not .1<=a.g1_collision_gain<=1. or (a.g1_collision_gain!=.1 and a.robot!='g1'):
        raise ValueError('bounded G1 native collision class-K rate required')
    if a.parking_profile!='legacy' and a.robot!='g1':
        raise ValueError('brake_settle_v1 is a G1-only candidate')
    if a.go2_native_posture!='none' and a.robot!='go2':
        raise ValueError('native EE operating preference is a Go2-only candidate')
    if a.waist_posture_cost is not None and (a.robot!='g1' or not .03<=a.waist_posture_cost<=1.):
        raise ValueError('bounded positive G1 waist soft posture cost required')
    if any(v is not None for v in (a.walking_posture_cost,a.terminal_damping_cost)):
        if a.robot!='g1':raise ValueError('native Pink soft costs are G1-only')
        load('validate_pink_profile',PACKAGE/'Robot_G1/pink_profile.py').task_cost_overrides(
            True,True,a.walking_posture_cost,a.terminal_damping_cost)
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
    if a.robot=='g1':
        shutil.copy2(PACKAGE/'Robot_G1/pink_profile.py',frozen/'pink_profile.py')
        if a.g1_park_yaw_assist:
            shutil.copy2(PACKAGE/'Robot_G1/stand_yaw_assist.py',frozen/'stand_yaw_assist.py')
        shutil.copy2(PACKAGE/'Robot_G1/solver_context.py',frozen/'solver_context.py')
        shutil.copy2(PACKAGE/'Robot_G1/solver_feedback.py',frozen/'solver_feedback.py')
        if a.g1_following_reserve:
            shutil.copy2(PACKAGE/'Robot_G1/following_reserve.py',frozen/'following_reserve.py')
            shutil.copy2(PACKAGE/'Robot_G1/following_execution.py',frozen/'following_execution.py')
        if a.g1_gravity_profile!='urdf':
            shutil.copy2(PACKAGE/'Robot_G1/gravity_model.py',frozen/'gravity_model.py')
            shutil.copy2(PACKAGE/'Robot_G1/config/actual_gravity_model.json',frozen/'actual_gravity_model.json')
        if a.g1_gravity_hold_profile!='zero':
            shutil.copy2(PACKAGE/'Robot_G1/gravity_hold.py',frozen/'gravity_hold.py')
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
                waist_posture_cost=a.waist_posture_cost,collision_gain=a.g1_collision_gain,
                actual_gravity_metadata=(json.loads((frozen/'actual_gravity_model.json').read_text())
                    if a.g1_gravity_profile!='urdf' else None)))
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
        source_arms=cfg.actuators['arms'].copy()
        # Conservative historical operating cap: this height asset/config has
        # 139Nm hip-roll source caps; the prior tested operating cap is 88Nm.
        cfg.actuators['legs'].effort_limit_sim['.*_hip_roll_joint'] = G1_29DOF_DELAYED_DC_MOTOR.actuators['legs'].effort_limit_sim['.*_hip_roll_joint']
        # Damping addresses physical upper-body ringing; never increase torque
        # caps or alter the lower-body policy output to obtain tracking speed.
        for group in ['arms','waist']:
            cfg.actuators[group].damping={k:v*a.upper_damping_scale for k,v in cfg.actuators[group].damping.items()}
        cfg.actuators['arms'].stiffness={k:v*a.g1_arm_stiffness_scale for k,v in source_arms.stiffness.items()}
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
    if a.robot=='g1' and a.g1_gravity_profile!='urdf':
        actual_gravity_metadata=json.loads((a.frozen/'actual_gravity_model.json').read_text())
        if hashlib.sha256(Path(cfg.spawn.usd_path).read_bytes()).hexdigest()!=actual_gravity_metadata['source_usd_sha256']:
            raise RuntimeError('actual gravity metadata does not match the current native USD')
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
    if a.robot=='g1' and a.upper_mode in ('velocity_servo','bounded_integral'):
        actuator=robot.actuators['arms'];arm_names=list(actuator.joint_names)
        arm_ids=[robot.joint_names.index(n) for n in arm_names]
        def source_arm_values(value):
            return [float(next(v for pattern,v in value.items() if re.fullmatch(pattern,n)))
                for n in arm_names] if isinstance(value,dict) else [float(value)]*len(arm_names)
        pd_readback=load('frozen_arm_PD_preflight',a.frozen/'pink_profile.py').arm_pd_preflight(
            joint_names=arm_names,stiffness_scale=a.g1_arm_stiffness_scale,
            source_stiffness=source_arm_values(source_arms.stiffness),source_damping=source_arm_values(source_arms.damping),
            source_armature=source_arm_values(source_arms.armature),source_effort_caps=source_arm_values(source_arms.effort_limit_sim),
            source_velocity_caps=source_arm_values(source_arms.velocity_limit_sim),
            actual_stiffness=actuator.stiffness[0].tolist(),actual_damping=actuator.damping[0].tolist(),
            actual_armature=robot.root_physx_view.get_dof_armatures()[0,arm_ids].tolist(),
            actual_effort_caps=robot.root_physx_view.get_dof_max_forces()[0,arm_ids].tolist(),
            actual_velocity_caps=robot.root_physx_view.get_dof_max_velocities()[0,arm_ids].tolist(),
            engine_stiffness=robot.root_physx_view.get_dof_stiffnesses()[0,arm_ids].tolist(),
            engine_damping=robot.root_physx_view.get_dof_dampings()[0,arm_ids].tolist(),
            upper_damping_scale=a.upper_damping_scale,physics_dt_s=physics_dt)
        pd_readback['native_actuator_model']='implicit' if actuator.is_implicit_model else 'explicit'
        if actuator.is_implicit_model:
            pd_readback['passed']=False;pd_readback['failures'].append('actuator_not_explicit')
        (a.output/'arm_PD_preflight.json').write_text(json.dumps(pd_readback,indent=2)+'\n')
        if not pd_readback['passed']:raise RuntimeError('G1 explicit arm PD readback or scalar stability screen failed')
    if a.robot=='g1' and a.g1_gravity_profile!='urdf':
        measured_gravity_bodies=a.rig_module.lab_body_poses(robot,set(robot.body_names))
        gravity_verification=a.pink.verify_actual_gravity(robot.body_names,
            robot.root_physx_view.get_masses()[0].numpy(),robot.root_physx_view.get_coms()[0,:,:3].numpy(),
            measured_gravity_bodies,dict(zip(robot.joint_names,robot.data.joint_pos[0].tolist())),measured_gravity_bodies['pelvis'])
        gravity_verification.update(profile=a.g1_gravity_profile,
            source_usd_sha256=actual_gravity_metadata['source_usd_sha256'],
            calibration_dump_sha256=actual_gravity_metadata['native_dump_sha256'],
            total_effort_implementation='single explicit tauPD + gravity then original joint cap clamp',
            original_ik_collision_model_unchanged=True)
        (a.output/'actual_gravity_verification.json').write_text(json.dumps(gravity_verification,indent=2)+'\n')
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
        native_pd_stiffness={n:x.stiffness.tolist() for n,x in robot.actuators.items()},upper_damping_scale=a.upper_damping_scale,
        g1_arm_stiffness_scale=a.g1_arm_stiffness_scale if a.robot=='g1' else None,
        native_pd_damping={n:x.damping.tolist() for n,x in robot.actuators.items()},park_damping_scale=a.park_damping_scale,
        native_actuator_effort_limits={n:x.effort_limit.tolist() for n,x in robot.actuators.items()},
        native_actuator_joint_names={n:list(x.joint_names) for n,x in robot.actuators.items()},
        native_actuator_models={n:('implicit' if x.is_implicit_model else 'explicit') for n,x in robot.actuators.items()},
        historical_hip_roll_operating_cap_88nm=(all(abs(effort[robot.joint_names.index(n)]-88.)<1e-3
            for n in ['left_hip_roll_joint','right_hip_roll_joint']) if a.robot=='g1' else None)),indent=2)+'\n')
    if a.robot=='go2':
        coordinator = coordinate_module.CameraCoordinator(rig.camera_local,backend=backend,response_profile=a.go2_response_profile)
        native_parking=coordinate_module.NativeCameraParking()
        domain_recovery=coordinate_module.NativeDomainRecovery(coordinator.domain,backend)
    lower_groups=['legs','feet'] if a.robot=='g1' else ['base_legs']
    damping_groups=lower_groups if a.robot=='g1' else list(robot.actuators)
    native_damping={n:robot.actuators[n].damping.clone() for n in damping_groups}
    damping_scales={n:1. for n in damping_groups}
    leg_damping_scale=1.
    upper_ids=[robot.joint_names.index(n) for n in a.pink.owners['pink']] if a.robot=='g1' else []
    gravity_hold_model=None;gravity_hold_binding_id=None;gravity_hold_names=[];gravity_hold_ids=[]
    last_upper_written=None;gravity_pelvis_body_id=None
    if a.g1_gravity_hold_profile!='zero':
        gravity_hold_module=load('frozen_current_state_gravity',a.frozen/'gravity_hold.py')
        gravity_hold_model=gravity_hold_module.GravityHold(a.frozen/'g1_29dof.urdf',a.frozen/'actual_gravity_model.json')
        gravity_hold_names=list(gravity_hold_model.upper_joint_names)
        if set(gravity_hold_names)!=set(a.pink.owners['pink']):
            raise RuntimeError('static gravity and original Pink owners differ')
        gravity_hold_ids=[robot.joint_names.index(n) for n in gravity_hold_names]
        gravity_pelvis_body_id=robot.body_names.index('pelvis')
        last_upper_written=target[0,gravity_hold_ids].tolist()
        binding=dict(gravity_hold_module_sha256=hashlib.sha256((a.frozen/'gravity_hold.py').read_bytes()).hexdigest(),
            actual_metadata_file_sha256=hashlib.sha256((a.frozen/'actual_gravity_model.json').read_bytes()).hexdigest(),
            source_urdf_sha256=hashlib.sha256((a.frozen/'g1_29dof.urdf').read_bytes()).hexdigest(),
            native_dump_sha256=actual_gravity_metadata['native_dump_sha256'],
            canonical_upper_joint_names=gravity_hold_names,plant_joint_names=list(robot.joint_names))
        gravity_hold_binding_id=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        (a.output/'gravity_hold_initialization.json').write_text(json.dumps(dict(
            schema='g1_current_state_gravity_hold_initialization/v1',profile=a.g1_gravity_hold_profile,
            model_binding_id=gravity_hold_binding_id,model_binding=binding,model=gravity_hold_model.manifest(),
            runtime_connected_profile=a.g1_gravity_hold_profile,
            initial_upper_position_target15=list(last_upper_written),
            initial_reference_source='native_reset_joint_positions_before_first_timed_write',
            pelvis_pose_source='physx_sdk_body_link_pose_w_wxyz',
            upper_gravity_feedforward_field_scope='fresh worker-motion packet only; local static support is logged per physical write',
            worker_motion_lease_unchanged=True,held_reference_velocity_zero=True,
            physical_parameters_changed=False,effort_composition='existing_explicit_pd_plus_g_single_clamp'),indent=2)+'\n')
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
    pink_profile=load('frozen_pink_profile',a.frozen/'pink_profile.py') if a.robot=='g1' else None
    yaw_assist_module=load('frozen_stand_yaw_assist',a.frozen/'stand_yaw_assist.py') if a.g1_park_yaw_assist else None
    solver_context=load('frozen_solver_context',a.frozen/'solver_context.py') if a.robot=='g1' else None
    solver_feedback=load('frozen_solver_feedback',a.frozen/'solver_feedback.py') if a.robot=='g1' else None
    following_module=load('frozen_following_reserve',a.frozen/'following_reserve.py') if a.g1_following_reserve else None
    following_execution=load('frozen_following_execution',a.frozen/'following_execution.py') if a.g1_following_reserve else None
    following_names=list(a.pink.optimized_joint_names) if a.g1_following_reserve else []
    following_ids=[robot.joint_names.index(n) for n in following_names]
    following_owner_order=[list(a.pink.owners['pink']).index(n) for n in following_names] if a.g1_following_reserve else []
    committed_upper_reference=None;physical_reference_write_seq=0
    def reference_snapshot(stamp):
        if committed_upper_reference is None:return None
        return following_module.committed_reference_snapshot(**committed_upper_reference,snapshot_stamp_s=stamp)
    current_task_context=None
    velocity_profile=native_g1_velocity_parameters(servo_module,a.g1_velocity_profile) if a.robot=='g1' else servo_module.go2_profile(a.go2_response_profile)
    if a.robot=='g1':velocity_profile.update(pink_profile.velocity_servo_overrides(a.g1_velocity_profile))
    servo = servo_module.VelocityServo(limits=command_limits,**velocity_profile)
    (a.output/'velocity_servo_config.json').write_text(json.dumps(dict(limits=command_limits,parameters=velocity_profile,
        profile=a.g1_velocity_profile if a.robot=='g1' else a.go2_response_profile),indent=2)+'\n')
    session = None; rows=[];results=[];case_rows=[];all_tasks=[];failures=[];terminal_abort=None
    optical_sampler=None;optical_sampling_manifests=[]
    fk_max_position=fk_max_angle=0.;body_fk_max_position=body_fk_max_angle=0.
    preparation_s=2.+(a.avoidance_lead_s if avoidance is not None else 0.)
    steps = round((preparation_s+len(selected)*(a.move_seconds+a.stop_seconds))/dt)
    start_wall=time.monotonic(); case_index=-1; stop_anchor=None; parking_mode=False;previous_policy_command=np.zeros(3)
    parking_posture=None;parking_posture_stamp=None;parking_posture_refreshed=False;parking_posture_refresh=None
    parking_supervisor=None
    parking_module=load('frozen_parking_supervisor',a.frozen/'parking_supervisor.py') if a.parking_profile!='legacy' else None
    limits = robot.data.joint_pos_limits[0].numpy()
    current_measurement=measure(0.)
    timing_totals=np.zeros(5)
    with PreparedSimulationGarbage(a.robot=='g1' and avoidance is not None) as prepared_gc,\
            (a.output/'measurements.jsonl').open('w',buffering=1) as log,torch.inference_mode(),\
            (a.output/'control_wall_timing.jsonl').open('w',buffering=1) as wall_log,\
            GarbageCollectionTiming(a.robot=='g1') as gc_timing:
        start_wall=time.monotonic()
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
                parking_posture=None;parking_posture_stamp=None;parking_posture_refreshed=False;parking_posture_refresh=None
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
                if a.robot=='g1':current_task_context=solver_context.task_context(task.to_dict())
                if a.robot=='go2':
                    optical_sampler=optical_module.OpticalSampler(task)
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
            if avoidance is not None and moving:desired_velocity=avoidance.platform_velocity(stamp,bodies,camera)
            applied_feedforward=None
            if a.robot=='g1':
                applied_feedforward=pink_profile.phase_velocity_servo_overrides(a.g1_velocity_profile,bool(moving),float(desired_velocity[2]))
                if applied_feedforward is not None:
                    servo.feedforward_gain=np.asarray(applied_feedforward['feedforward_gain'])
                    servo.negative_feedforward_gain=np.asarray(applied_feedforward['negative_feedforward_gain'])
                    if 'kp' in applied_feedforward:servo.kp=np.asarray(applied_feedforward['kp'])
                    if 'ki' in applied_feedforward:servo.ki=np.asarray(applied_feedforward['ki'])
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
            parking_transition=None;park_yaw_assist_report=None
            if stop_anchor is not None:
                # A learned standing mode needs a literal zero velocity input;
                # small continuous PI corrections can keep its walking cycle
                # alive. Enter only after measured braking, resume on drift.
                if parking_supervisor:
                    policy_command,parking_transition=parking_supervisor.update(stamp,bodies[base_name],measured,
                        policy_command,previous_policy_command,dt)
                    parking_mode=parking_transition['mode']=='PARK'
                    if parking_transition['entered']:stop_anchor=parking_supervisor.anchor.copy()
                    if a.g1_refresh_posture_on_park and parking_transition['entered'] and not parking_posture_refreshed:
                        parking_posture_refresh=pink_profile.first_park_posture_refresh(enabled=True,entered=True,
                            already_refreshed=parking_posture_refreshed,owner_names=a.pink.owners['pink'],
                            measured_joints=dict(zip(robot.joint_names,robot.data.joint_pos[0].tolist())),
                            stamp_s=stamp,task_context=current_task_context)
                        parking_posture_refresh.update(previous_capture_stamp_s=parking_posture_stamp,
                            previous_targets=dict(parking_posture),applies_to_new_requests_from_s=stamp,
                            pending_previous_preference_remains_subject_to_original_context_and_lease=True)
                        parking_posture=parking_posture_refresh['targets'];parking_posture_stamp=stamp
                        parking_posture_refreshed=True
                    if parking_transition['previous_mode']!=parking_transition['mode']:servo.reset()
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
                    if yaw_assist_module:
                        policy_command,park_yaw_assist_report=yaw_assist_module.command(policy_command,
                            previous_policy_command,mode=parking_transition['mode'],
                            anchor_yaw_error_rad=float(parking_transition['anchor_error_body'][2]),
                            measured_yaw_rate_rad_s=float(measured[2]),dt_s=dt)
                        park_yaw_assist_report.update(acquisition_stamp_s=stamp,
                            task_context=current_task_context,downstream_vetoes_authoritative=True)
                        parking_transition['supervisor_literal_zero_input']=parking_transition['literal_zero_input']
                        parking_transition['literal_zero_input']=bool(np.max(np.abs(policy_command))<=1e-12)
                        parking_transition['active_stand_yaw_assist']=True
                if parking_supervisor:servo.applied(policy_command)
            desired_camera=camera.copy();support={'reason':'accepted','feasible':None}
            if session:
                desired_camera,_=session.reference(min(stamp,task_end))
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
                    support=(solver_feedback.public_support(None,collision_evidence) if a.robot=='g1'
                        else dict(reason='self_collision',feasible=False))
                    terminal_abort=dict(stamp_s=stamp,collision=collision_evidence,public_task_status=support)
                    (a.output/'constraint_abort.json').write_text(json.dumps(terminal_abort,indent=2)+'\n')
                    failures.append('fine original-shape self clearance below5mm');break
            packet=None;control_begin=time.monotonic()
            following_application=None;following_authorization=None;following_evidence=None
            following_deferred=None;following_writes=[];following_write_checks=[];following_submitted=False
            requested_policy_command=policy_command.copy();policy_velocity_guard=None
            if a.robot=='g1':
                upper_position_before_tick=target[0,upper_ids].clone()
                target_velocity.zero_();target_effort.zero_()
                packet=a.pink.poll(stamp,current_context=current_task_context,execution_horizon_s=dt)
                if avoidance is not None and ('deadline_exceeded' in packet['status'] or a.pink.process.poll() is not None):
                    (a.output/'pink_runtime_abort.json').write_text(json.dumps(dict(stamp_s=stamp,
                        packet=packet,failure_count=a.pink.failure_count,public_task_status=support),indent=2)+'\n')
                    failures.append('Pink worker bounded recovery deadline');break
                if a.g1_following_reserve and session and packet['ready']:
                    try:
                        if packet.get('following_reserve_report') is None:
                            raise RuntimeError('current guarded reply is missing its native Limit report')
                        following_application=following_module.validate_application_manifest(packet['following_reserve_report'],
                            task_id=session.task_id,task_sha256=current_task_context['task_sha256'],
                            current_reference=reference_snapshot(stamp),now_s=stamp,execution_horizon_s=dt)
                    except following_module.FollowingReserveRejected as error:
                        packet.update(ready=False,status='following_reference_hold_upper_targets',
                            following_application_rejection=str(error))
                        a.pink.needs_reset=True
                    except RuntimeError as error:
                        packet.update(ready=False,status='following_reference_hold_upper_targets',
                            error=str(error),failure_kind='transport_failure')
                        a.pink.needs_reset=True
                support=solver_feedback.public_support(packet,collision_evidence)
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
                upper_reference_report=None
                if upper_reference is not None and packet['ready']:
                    integrator_before=upper_reference.reference.copy()
                    upper_v=np.array([packet.get('velocity',{}).get(n,0.) if session else 0. for n in a.pink.owners['pink']])
                    reference,upper_reference_report=upper_reference.update(robot.data.joint_pos[0,upper_ids].numpy(),upper_v,dt)
                    target[0,upper_ids]=torch.tensor(reference,dtype=torch.float32)
                    target_velocity[0,upper_ids]=torch.tensor(upper_v,dtype=torch.float32)
                    for n in a.pink.owners['pink']:target_effort[0,robot.joint_names.index(n)]=float(np.clip(packet['gravity'][n],-effort[robot.joint_names.index(n)],effort[robot.joint_names.index(n)]))
                    if a.g1_following_reserve and session:
                        try:
                            following_authorization=following_execution.authorize_native_step(task_context=current_task_context,
                                packet_seq=packet['seq'],acquisition_stamp_s=packet['stamp_s'],application_stamp_s=stamp,
                                committed_reference=reference_snapshot(stamp),integrator_before=integrator_before[following_owner_order],
                                velocity=upper_v[following_owner_order],proposed_reference=reference[following_owner_order],
                                actual_float32_target=target[0,following_ids].numpy(),
                                antiwindup_active=upper_reference_report['antiwindup_active'],dt_s=dt)
                        except following_execution.FollowingExecutionRejected as error:
                            target[0,upper_ids]=upper_position_before_tick;upper_reference.reference=integrator_before
                            target_velocity[0,upper_ids]=0.;target_effort[0,upper_ids]=0.
                            packet.update(ready=False,status='reference_step_clipped_hold_upper_targets',
                                following_execution_rejection=str(error))
                            a.pink.needs_reset=True;support=solver_feedback.public_support(packet,collision_evidence)
                            policy_command=np.clip(np.zeros(3),previous_policy_command-servo.slew*dt,
                                previous_policy_command+servo.slew*dt)
                            servo.applied(policy_command,transient_hold=True)
                            policy_velocity_guard=dict(reason='native_reference_step_rejected_slew_brake',
                                upper_reference_held=True,upper_velocity_target_zero=True,integral_frozen=True,collision_veto=False)
                for n,v in (packet.get('velocity',{}) if packet['ready'] and session and upper_reference is None else {}).items():
                    i=robot.joint_names.index(n)
                    position=(robot.data.joint_pos[0,i]+v*.20) if a.upper_mode=='velocity_servo' else (target[0,i]+v*dt)
                    target[0,i]=torch.clamp(position,*robot.data.joint_pos_limits[0,i])
                    target_velocity[0,i]=v
                    if a.upper_mode=='velocity_servo':
                        target_effort[0,i]=float(np.clip(packet['gravity'][n],-effort[i],effort[i]))
                if avoidance is not None:
                    policy_command=avoidance.constrain_platform(stamp,bodies,policy_command,previous_policy_command,dt)
                    servo.applied(policy_command)
                command=torch.tensor([[*policy_command,.72]],dtype=torch.float32)
                features=[command,robot.data.root_ang_vel_b,robot.data.projected_gravity_b,
                    robot.data.joint_pos[:,observation_ids]-defaults,robot.data.joint_vel[:,observation_ids]*.1,action.reshape(1,12)]
                action=policy(torch.cat(features,dim=-1).reshape(-1)).clamp(-6.,6.)
                target[:,ids]=offsets+scales*action.reshape(1,12)
                soft_costs=pink_profile.task_cost_overrides(bool(in_case and np.linalg.norm(cases[selected[case_index]])>0),
                    bool(moving),a.walking_posture_cost,a.terminal_damping_cost)
                submit_allowed=True
                if a.g1_following_reserve and session:
                    committed=reference_snapshot(stamp)
                    if (committed is None or committed['task_id']!=session.task_id
                            or committed['task_sha256']!=current_task_context['task_sha256']):
                        submit_allowed=False;following_deferred='await_first_actual_reference_write_in_current_Task'
                    else:
                        following_evidence=dict(schema='g1_following_evidence/v1',task_id=session.task_id,
                            task_sha256=current_task_context['task_sha256'],acquisition_stamp_s=stamp,
                            joint_names=following_names,measured_q_rad=robot.data.joint_pos[0,following_ids].tolist(),
                            measured_qd_rad_s=robot.data.joint_vel[0,following_ids].tolist(),velocity_stamp_s=stamp,
                            measurement_source='measured_physx',committed_reference=committed,
                            planned_advance_bound=following_module.planned_advance_bound(committed,
                                target[0,following_ids].tolist(),task_id=session.task_id,request_stamp_s=stamp,write_dtype='float32'))
                if submit_allowed:
                    following_submitted=a.pink.submit(dict(zip(robot.joint_names,robot.data.joint_pos[0].tolist())),bodies['pelvis'],
                        desired_camera if session else None,stamp,camera,dt,measured_bodies=bodies if a.fine_collision else None,
                        posture_reference=parking_posture,task_context=current_task_context,following_evidence=following_evidence,**soft_costs)
                # In avoidance runs use the worker's original 180 ms recovery
                # deadline. A counter of three expired replies is not a clock.
                # Each resumed command still has the unchanged 60 ms lease.
                if avoidance is None and a.pink.failure_count>=3:
                    (a.output/'pink_runtime_abort.json').write_text(json.dumps(dict(stamp_s=stamp,
                        packet=packet,failure_count=a.pink.failure_count,public_task_status=support),indent=2)+'\n')
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
                    if a.go2_native_posture=='x58_moving_v1' and np.linalg.norm(cases[selected[case_index]])>0:
                        forecast.update(native_posture_x_m=.58,
                            posture_weight=coordinate_module.operating_posture_weight(stamp,case_start,task_end))
                    support=coordinator.command(stamp,sample,bodies['link0'],bodies['base'],policy_command,measured,
                        max_lookahead_s=max(0.,end-stamp),**forecast)
                    policy_command=np.asarray(support['base_command']);servo.applied(policy_command)
                    native,corrected=coordinate_module.feedback_native_command(backend,desired_camera,camera,bodies['link0'],rig.camera_local,
                        domain=coordinator.domain)
                    requested_native=np.asarray(support['native_command']).copy()
                    requested_domain=coordinator.domain.check(requested_native)
                    native,parking_input= native_parking.update(native,desired_camera,camera,bodies['base'],measured,
                        parking=a.parking_camera_hold and parking_mode and not moving
                            and requested_domain['status']=='inside_outer_domain_unverified')
                    rejected=corrected.get('rejected_feedback_correction')
                    recovery_candidate=requested_native if requested_domain['status']!='inside_outer_domain_unverified' else native
                    native,recovery_command,recovery_report=domain_recovery.update(recovery_candidate,bodies['link0'],bodies['base'],
                        stamp,previous_policy_command,dt)
                    if recovery_command is not None:
                        policy_command=np.asarray(recovery_command);servo.applied(policy_command)
                        support.update(feasible=False,reason='out_of_policy_domain',camera_supported=False,
                            base_velocity_feasible=False,domain_recovery=recovery_report)
                    corrected=coordinator.domain.check(native)
                    if rejected is not None:corrected['rejected_feedback_correction']=rejected
                    if corrected['status']!='inside_outer_domain_unverified':
                        raise RuntimeError('domain failure escaped legal native-input hold guard')
                    if a.collision and collision_evidence['status']!='pass':
                        # Geometry freshness veto has priority over task-domain
                        # projection and recovery; never override this brake.
                        policy_command=np.clip(np.zeros(3),previous_policy_command-np.array([.5,.5,.8])*dt,
                            previous_policy_command+np.array([.5,.5,.8])*dt)
                        servo.applied(policy_command)
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
                action=policy(torch.cat(histories,dim=-1).clamp(-100.,100.)).reshape(-1).clamp(-10.,10.)
                target[:,ids]=robot.data.default_joint_pos[:,ids]+.25*action.reshape(1,18)
            previous_policy_command=policy_command.copy()
            if a.robot=='go2':coordinator.observe_applied_input(stamp,policy_command)
            physics_begin=time.monotonic();effort_peak=np.zeros(len(robot.joint_names));velocity_peak=effort_peak.copy()
            substep_joint_margin=np.inf
            upper_execution_lease=None
            upper_static_gravity_support=(dict(schema='g1_upper_static_gravity_support/v1',
                model_binding_id=gravity_hold_binding_id,physics_dt_s=physics_dt,writes=[])
                if gravity_hold_model is not None else None)
            if a.robot=='g1' and packet['ready']:
                upper_execution_lease=dict(accepted_wall_age_s=packet['accepted_wall_age_s'],checks=[],expired=False,
                    original_requested_velocity=packet.get('velocity',{}))
            for substep in range(4):
                if a.robot=='g1' and packet['ready']:
                    check=solver_feedback.execution_lease_status(packet,stamp_s=stamp+substep*physics_dt,
                        wall_s=time.monotonic(),horizon_s=physics_dt)
                    upper_execution_lease['checks'].append(check)
                    if not check['valid']:
                        upper_execution_lease.update(expired=True,first_expired_substep=substep)
                        if substep==0:
                            # No physical write used this newly integrated
                            # target. Hold the last actually committed target.
                            target[0,upper_ids]=upper_position_before_tick
                            if upper_reference is not None:upper_reference.reference=upper_position_before_tick.numpy().astype(float)
                            upper_reference_report=dict(fresh_velocity_integrated=False,held_original_reference=True)
                            upper_execution_lease['discarded_before_first_physical_write']=True
                        packet.update(ready=False,status='command_expired_during_execution_hold_upper_targets')
                        target_velocity[0,upper_ids]=0.;target_effort[0,upper_ids]=0.
                        a.pink.needs_reset=True
                        if a.pink.pending is not None:a.pink.pending['held_since_submit']=True
                        support=solver_feedback.public_support(packet,collision_evidence)
                if a.g1_following_reserve:
                    now_stamp=stamp+substep*physics_dt
                    current_commit=reference_snapshot(now_stamp)
                    if current_commit is not None:
                        try:
                            if following_authorization is not None and packet['ready']:
                                following_write_checks.append(following_execution.validate_authorized_write(
                                    following_authorization,task_context=current_task_context,current_reference=current_commit,
                                    target_to_write=target[0,following_ids].tolist(),now_s=now_stamp,horizon_s=physics_dt))
                            elif not packet['ready']:
                                following_write_checks.append(following_execution.validate_expired_reference_hold(
                                    current_reference=current_commit,target_to_write=target[0,following_ids].tolist(),
                                    velocity_to_write=target_velocity[0,following_ids].tolist(),
                                    gravity_to_write=target_effort[0,following_ids].tolist()))
                        except following_execution.FollowingExecutionRejected as error:
                            # Emergency hold of a real committed target is not
                            # a new projected/certified QP displacement.
                            target[0,following_ids]=torch.tensor(current_commit['positions_rad'],dtype=torch.float32)
                            if upper_reference is not None:
                                by_name=dict(zip(following_names,current_commit['positions_rad']))
                                upper_reference.reference=np.array([by_name[n] for n in upper_reference.names])
                            target_velocity[0,upper_ids]=0.;target_effort[0,upper_ids]=0.
                            packet.update(ready=False,status='reference_step_clipped_hold_upper_targets',
                                following_execution_rejection=str(error))
                            a.pink.needs_reset=True;support=solver_feedback.public_support(packet,collision_evidence)
                            following_write_checks.append(dict(valid=False,reason=str(error),emergency_actual_reference_hold=True))
                if gravity_hold_model is not None:
                    # Static support is calculated independently after every
                    # motion veto, from the just-updated SDK state. It never
                    # revives a packet, changes qref, or adds a velocity target.
                    zeros=[0.]*len(gravity_hold_names)
                    write_stamp=float(stamp+substep*physics_dt)
                    write=dict(substep=substep,physics_state_seq=step*4+substep,
                        measured_stamp_s=write_stamp,write_stamp_s=write_stamp,
                        motion_ready=bool(packet['ready']),position_target15=target[0,gravity_hold_ids].tolist(),
                        held_reference15=list(last_upper_written),velocity_target15=target_velocity[0,gravity_hold_ids].tolist(),
                        worker_gravity15=zeros.copy(),local_gravity15=zeros.copy(),applied_gravity15=zeros.copy(),
                        computed_gravity15=None,joint_names=None,measured_q29=None,pelvis_R=None,sample_token=None,
                        trajectory_reused=False,effort_composition='existing_explicit_pd_plus_g_single_clamp')
                    if packet['ready']:
                        write.update(source='worker_motion',status='worker',reason='original_fresh_motion_lease',
                            worker_gravity15=target_effort[0,gravity_hold_ids].tolist())
                    else:
                        target_velocity[0,upper_ids]=0.;target_effort[0,upper_ids]=0.
                        try:
                            if target[0,gravity_hold_ids].tolist()!=last_upper_written:
                                target[0,gravity_hold_ids]=torch.tensor(last_upper_written,dtype=torch.float32)
                                if upper_reference is not None:
                                    held_by_name=dict(zip(gravity_hold_names,last_upper_written))
                                    upper_reference.reference=np.array([held_by_name[n] for n in upper_reference.names])
                                packet.update(ready=False,status='reference_step_clipped_hold_upper_targets',
                                    static_gravity_hold_reference_rejected=True)
                                support=solver_feedback.public_support(packet,collision_evidence)
                                write['position_target15']=list(last_upper_written)
                                raise ValueError('uncommitted upper reference refused; restored last physical write with zero velocity/gravity')
                            measured_q=robot.data.joint_pos[0].numpy().copy()
                            pelvis_pose=robot.data.body_link_pose_w[0,gravity_pelvis_body_id].numpy().copy()
                            pelvis_R=Rotation.from_quat(pelvis_pose[[4,5,6,3]]).as_matrix()
                            gravity,acquisition=current_state_gravity_support(gravity_hold_model,
                                joint_names=robot.joint_names,joint_positions=measured_q,pelvis_rotation=pelvis_R,
                                stamp_s=write_stamp,physics_state_seq=step*4+substep)
                            target_effort[0,gravity_hold_ids]=torch.tensor([gravity[n] for n in gravity_hold_names],dtype=torch.float32)
                            sample=dict(physics_state_seq=step*4+substep,stamp_s=write_stamp,
                                joint_names=acquisition['joint_names'],positions_rad=acquisition['joint_positions_rad'],
                                pelvis_R=acquisition['pelvis_rotation_world'])
                            write.update(source='local_static_model',status='computed',reason='no_fresh_motion_static_support',
                                joint_names=sample['joint_names'],measured_q29=sample['positions_rad'],pelvis_R=sample['pelvis_R'],
                                measurement_source='physx_sdk_current_after_update',
                                sample_token=hashlib.sha256(json.dumps(sample,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                                computed_gravity15=[gravity[n] for n in gravity_hold_names],
                                local_gravity15=target_effort[0,gravity_hold_ids].tolist())
                        except (ValueError,TypeError,KeyError,IndexError) as error:
                            target_effort[0,upper_ids]=0.
                            write.update(source='zero_fallback',status='rejected',reason=str(error))
                        write['velocity_target15']=target_velocity[0,gravity_hold_ids].tolist()
                    write['applied_gravity15']=target_effort[0,gravity_hold_ids].tolist()
                    upper_static_gravity_support['writes'].append(write)
                robot.set_joint_position_target(target)
                if a.robot=='g1' and (a.upper_velocity_feedforward or a.upper_mode in ('velocity_servo','bounded_integral')):
                    robot.set_joint_velocity_target(target_velocity)
                    robot.set_joint_effort_target(target_effort)
                robot.write_data_to_sim()
                if gravity_hold_model is not None:
                    last_upper_written=target[0,gravity_hold_ids].tolist()
                if a.g1_following_reserve:
                    physical_reference_write_seq+=1
                    committed_upper_reference=dict(task_id=None if current_task_context is None else current_task_context['task_id'],
                        task_sha256=None if current_task_context is None else current_task_context['task_sha256'],
                        joint_names=following_names,positions_rad=target[0,following_ids].tolist(),
                        commit_seq=physical_reference_write_seq,commit_stamp_s=stamp+substep*physics_dt)
                    following_writes.append(reference_snapshot(stamp+substep*physics_dt))
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
            if upper_execution_lease is not None:
                upper_execution_lease.update(execution_end_wall_age_s=time.monotonic()-packet['command_sent_wall_s'],
                    execution_end_sim_age_s=(step+1)*dt-packet['stamp_s'])
                if packet['ready']:
                    end_check=solver_feedback.execution_lease_status(packet,stamp_s=(step+1)*dt,wall_s=time.monotonic())
                    upper_execution_lease['execution_end_check']=end_check
                    if not end_check['valid']:
                        upper_execution_lease.update(expired=True,first_expired_substep=4,expired_at_measurement=True)
                        packet.update(ready=False,status='command_expired_during_execution_hold_upper_targets')
                        target_velocity[0,upper_ids]=0.;target_effort[0,upper_ids]=0.
                        a.pink.needs_reset=True
                        if a.pink.pending is not None:a.pink.pending['held_since_submit']=True
                        support=solver_feedback.public_support(packet,collision_evidence)
            feedback_begin=time.monotonic()
            if session:
                desired_camera,_=session.reference(min((step+1)*dt,task_end))
            q=robot.data.joint_pos[0].numpy();tilt=float(np.arccos(np.clip(base[2,2],-1.,1.)))
            velocity_world=robot.data.root_lin_vel_w[0].tolist();angular_world=robot.data.root_ang_vel_w[0].tolist()
            row=dict(stamp_s=(step+1)*dt,case=selected[case_index] if in_case else 'settle',phase='move' if moving else 'stop',
                T_world_base=base.tolist(),T_world_camera_optical=camera.tolist(),T_world_reference_optical=desired_camera.tolist(),
                joint_positions=q.tolist(),joint_names=robot.joint_names,requested_body_velocity=desired_velocity.tolist(),
                measured_body_velocity=np.r_[robot.data.root_lin_vel_b[0,:2].numpy(),robot.data.root_ang_vel_b[0,2].item()].tolist(),
                applied_policy_velocity=policy_command.tolist(),base_linear_velocity_world=velocity_world,base_angular_velocity_world=angular_world,
                native_standing_mode=parking_mode,
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
                row['park_yaw_assist']=(dict(park_yaw_assist_report,
                    actual_applied_policy_velocity=policy_command.tolist(),
                    downstream_command_changed=not np.array_equal(policy_command,park_yaw_assist_report['requested_native_input']))
                    if park_yaw_assist_report else None)
                row['requested_policy_velocity']=requested_policy_command.tolist()
                row['policy_velocity_guard']=policy_velocity_guard
                row['pink_runtime_recovery']={k:packet[k] for k in ('seq','stamp_s','acquisition_stamp_s',
                    'discarded_results','replacement_allowed','outstanding_requests','task_context','task_context_status',
                    'reply_task_context','current_task_context','failure_kind','dropped_seq','dropped_stamp_s',
                    'error','failed_request','qp_failure','accepted_wall_age_s','execution_horizon_s',
                    'command_sent_wall_s','lease_deadline_wall_s','lease_deadline_sim_s',
                    'wall_age_s','sim_age_s','recovery_clock','poll_interval_wall_s','deadline_records',
                    'pipe_readable','buffered_complete_reply','continuous_recovery_wall_s') if k in packet}
                row['current_solver_task_context']=current_task_context
                row['pink_velocity_feedforward']=dict(profile=a.g1_velocity_profile,
                    phase_schedule=applied_feedforward,actual_positive_gain=servo.feedforward_gain.tolist(),
                    actual_negative_gain=servo.negative_feedforward_gain.tolist(),actual_kp=servo.kp.tolist(),actual_ki=servo.ki.tolist())
                row['upper_execution_lease']=upper_execution_lease
                if upper_static_gravity_support is not None:
                    row['upper_static_gravity_support']=upper_static_gravity_support
                    row['upper_gravity_feedforward_scope']='fresh worker-motion packet only; see upper_static_gravity_support for actual per-write support'
                if a.g1_following_reserve:
                    row['following_reference_execution']=dict(optimized_joint_names=following_names,
                        request_submitted=following_submitted,request_evidence=following_evidence,
                        deferred_reason=following_deferred,before_first_use=following_application,
                        native_report=packet.get('following_reserve_report'),authorization=following_authorization,
                        physical_write_checks=following_write_checks,physical_reference_writes=following_writes,
                        final_reference_snapshot=reference_snapshot((step+1)*dt),
                        rejection=packet.get('following_execution_rejection',packet.get('following_application_rejection')),
                        actual_braking_certified=False)
                row['pink_upper_velocity']=packet.get('velocity',{}) if packet['ready'] else {}
                row['upper_gravity_feedforward']=packet.get('gravity',{}) if packet['ready'] else {}
                row['upper_effort_peak_fraction']=float(np.max(effort_peak[upper_ids]/effort[upper_ids]))
                row['pink_collision_report']=packet.get('pink_collision_report') if packet['ready'] else None
                row['pink_parking_posture']=dict(stamp_s=parking_posture_stamp,targets=parking_posture,
                    kind='soft_measured_posture_preference',camera_frame_task_unchanged=True,collision_barriers_unchanged=True)
                row['pink_parking_posture_refresh']=parking_posture_refresh
                row['pink_posture_report']=packet.get('pink_posture_report') if packet['ready'] else None
                row['pink_requested_soft_costs']=soft_costs
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
            rows.append(row)
            if in_case:case_rows.append(row)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            tick_log_end=time.monotonic()
            phase_wall=np.diff([tick_start,control_begin,physics_begin,measurement_begin,feedback_begin,tick_log_end])
            timing_totals+=phase_wall
            wall_log.write(json.dumps(dict(schema='g1_control_wall_timing/v1',step=step,stamp_s=(step+1)*dt,
                tick_start_wall_s=tick_start,measurement_log_end_wall_s=tick_log_end,
                tick_work_wall_s=tick_log_end-tick_start,
                phase_wall_ms=dict(zip(['task_servo','native_controller','physics','measurement_geometry','feedback_log'],
                    (phase_wall*1000).tolist())),
                start_lateness_wall_s=max(0.,tick_start-(start_wall+step*dt)),
                end_lateness_wall_s=max(0.,tick_log_end-(start_wall+(step+1)*dt)),
                own_timing_record_write_excluded=True),allow_nan=False)+'\n')
            if not np.isfinite(q).all() or substep_joint_margin < -1e-6 or base[2,3] < (.5 if a.robot=='g1' else .20) or tilt > (.6 if a.robot=='g1' else .65):
                failures.append('original joint/finite/balance safety gate');break
            if step%100==0:print(f'{a.robot} STEP {step}/{steps} {row["case"]} {row["phase"]}',flush=True)
            delay=start_wall+(step+1)*dt-time.monotonic()
            if delay>0:time.sleep(delay)
    elapsed=time.monotonic()-start_wall
    if gc_timing.enabled:
        (a.output/'gc_timing_events.json').write_text(json.dumps(dict(schema='g1_observed_gc_timing/v1',
            start_wall_s=start_wall,physical_loop_elapsed_wall_s=elapsed,settings=gc_timing.settings,
            observer_only=True,collections_forced=False,pre_loop_preparation=prepared_gc.report,
            events=gc_timing.events),indent=2)+'\n')
    if velocity_audit:
        velocity_audit.close()
    if avoidance is not None and failures:avoidance.execution_failure='; '.join(failures)
    if case_rows:results.append(summarize(selected[case_index],case_rows,a,audit_stop))
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
    include_terminal_abort(result,terminal_abort,(hashlib.sha256((a.output/'constraint_abort.json').read_bytes()).hexdigest()
        if terminal_abort is not None else None))
    if a.robot=='go2':result['requested_native_domain_failure_samples']=sum(r.get('requested_native_domain',{}).get('status')!='inside_outer_domain_unverified' for r in collision_rows)
    result['all_pass']=bool(result['all_pass'] and result['known_constraint_failure_free'] and
        result['task_infeasible_samples']==0 and (not a.fine_collision or result['fine_collision']['all_active_samples_fresh_pass']))
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
    stopped=[r for r in rows if r['phase']=='stop']
    if len(stopped)<2 or stopped[-1]['stamp_s']-stopped[0]['stamp_s']<.5-1e-8:
        stop=dict(passed=False,status='unverified_incomplete_measured_stop_window')
    else:
        stop=audit_stop([r['stamp_s'] for r in stopped],[r['T_world_base'] for r in stopped],
            [r['T_world_camera_optical'] for r in stopped],[r['joint_positions'] for r in stopped],
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
