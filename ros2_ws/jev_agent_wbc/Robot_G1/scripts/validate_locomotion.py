#!/usr/bin/env python3
"""Physical G1 rollout with pinned official AGILE motor config and TorchScript.

All root state writes occur exactly once at initialization. Motion is produced
only by the 14 policy position targets and the official DC motor controller.
"""
from pathlib import Path
import argparse,json,time,hashlib,sys,shutil,importlib.metadata
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--gui',action='store_true')
p.add_argument('--output',type=Path,default=None)
p.add_argument('--commands',type=Path,default=ROOT/'config/commands.json')
p.add_argument('--seconds',type=float,default=None,help='Short diagnostic prefix only.')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--snapshot',action='store_true')
a=p.parse_args()
out=a.output or ROOT/'results'/time.strftime('%Y%m%d_%H%M%S')
out.mkdir(parents=True,exist_ok=False)
def record_unhandled(exc_type,exc,tb):
    import traceback
    (out/'error.txt').write_text(''.join(traceback.format_exception(exc_type,exc,tb)))
    sys.__excepthook__(exc_type,exc,tb)
sys.excepthook=record_unhandled
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=not a.gui,enable_cameras=a.snapshot,device='cpu')
app=launcher.app
import torch,numpy as np,yaml
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils import CircularBuffer
from isaaclab.utils.math import euler_xyz_from_quat
from agile.rl_env.assets.robots.unitree_g1 import G1_29DOF_DELAYED_DC_MOTOR
from pxr import Usd

def main():
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    io=yaml.safe_load((ROOT/'config/policy_io.yaml').read_text())
    physics_dt=io['scene']['physics_dt']; decimation=io['scene']['decimation']; dt=physics_dt*decimation
    sim=SimulationContext(sim_utils.SimulationCfg(dt=physics_dt,device='cpu',render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0,dynamic_friction=1.0,restitution=0.0)))
    sim.set_camera_view(eye=(3.5,3.5,2.2),target=(0,0,0.7))
    ground=sim_utils.CuboidCfg(size=(200.,200.,0.1),collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3,0.32,0.36)))
    ground.func('/World/Ground',ground,translation=(0.,0.,-0.05))
    light=sim_utils.DomeLightCfg(intensity=1500.,color=(0.8,0.85,0.9));light.func('/World/Light',light)
    cfg=G1_29DOF_DELAYED_DC_MOTOR.copy()
    cfg.prim_path='/World/G1';cfg.spawn.usd_path=str(ROOT/'assets/G1/g1.usd')
    cfg.init_state.pos=(0.,0.,0.78)
    robot=Articulation(cfg)
    policy_path=ROOT/'policies/unitree_g1_velocity_history.pt'
    policy=torch.jit.load(str(policy_path),map_location='cpu').eval()
    sim.reset()
    robot.write_root_pose_to_sim(robot.data.default_root_state[:,:7])
    robot.write_root_velocity_to_sim(robot.data.default_root_state[:,7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos,robot.data.default_joint_vel)
    robot.reset()
    ids, names=robot.find_joints(io['actions'][0]['joint_names'],preserve_order=True)
    assert names==io['actions'][0]['joint_names'],(names,io['actions'][0]['joint_names'])
    default=torch.tensor(io['actions'][0]['offset']).unsqueeze(0)
    action=torch.zeros(1,14)
    history=[CircularBuffer(5,1,device='cpu') for _ in io['observations']['policy']]
    commands=json.loads(a.commands.read_text()); total=sum(x['seconds'] for x in commands)
    if a.seconds: total=min(total,a.seconds)
    segments=[]; boundary=0
    for c in commands:
        segments.append((boundary,boundary+c['seconds'],c));boundary+=c['seconds']
    rows=[];failures=[];wall=time.monotonic(); cmd=torch.zeros(1,3); previous_name=None
    metadata={'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'io_sha256':hashlib.sha256((ROOT/'config/policy_io.yaml').read_bytes()).hexdigest(),'commands_sha256':hashlib.sha256(a.commands.read_bytes()).hexdigest(),'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'joint_order':robot.joint_names,
        'controlled_joint_order':names,'physics_dt':physics_dt,'decimation':decimation,'seed':a.seed,
        'root_writes':'initialization only','policy_observations':'per-term history oldest to newest, term order from official IO YAML',
        'sim_device':'cpu','policy_device':'cpu','pace_requested':1,'commands':commands,
        'runtime':{'torch':torch.__version__,'numpy':np.__version__,'warp':__import__('warp').__version__,'warp_file':__import__('warp').__file__},
        'motor_source_sha256':hashlib.sha256((ROOT/'upstream/WBC-AGILE/agile/rl_env/assets/robots/unitree_g1.py').read_bytes()).hexdigest(),
        'actuators_source_sha256':hashlib.sha256((ROOT/'upstream/WBC-AGILE/agile/rl_env/mdp/actuators/actuators.py').read_bytes()).hexdigest(),
        'assets_manifest_sha256':hashlib.sha256((ROOT/'config/assets.manifest.json').read_bytes()).hexdigest(),
        'environment_freeze_sha256':hashlib.sha256((ROOT/'config/environment.freeze.txt').read_bytes()).hexdigest(),
        'package_versions':{name:importlib.metadata.version(name) for name in ['isaacsim','torch','numpy','isaaclab','isaaclab-assets','isaaclab-tasks','agile','rsl-rl-lib','warp-lang','pin','datasets','fsspec','pyarrow']}}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    source_dir=out/'source';source_dir.mkdir()
    for source in [Path(__file__),ROOT/'scripts/supervise.py',ROOT/'run.sh',ROOT/'config/policy_io.yaml',a.commands,
                   ROOT/'config/environment.freeze.txt',ROOT/'config/constraints.txt',ROOT/'config/sources.json',ROOT/'config/assets.manifest.json',
                   ROOT/'upstream/WBC-AGILE/agile/rl_env/assets/robots/unitree_g1.py',
                   ROOT/'upstream/WBC-AGILE/agile/rl_env/mdp/actuators/actuators.py',
                   ROOT/'upstream/WBC-AGILE/agile/rl_env/mdp/actuators/actuators_cfg.py']:
        shutil.copy2(source,source_dir/source.name)
    wall=time.monotonic()
    with torch.inference_mode():
      for step in range(round(total/dt)):
        if not app.is_running(): failures.append({'step':step,'reason':'simulation app closed'});break
        t=step*dt
        start,end,c=next(s for s in segments if s[0]<=t<s[1])
        if c['name']!=previous_name: print('SEGMENT',c,flush=True);previous_name=c['name']
        goal=torch.tensor(c['velocity']).reshape(1,3)
        max_change=torch.tensor([[0.35,0.35,0.6]])*dt
        cmd+=torch.clamp(goal-cmd,min=-max_change,max=max_change)
        terms=[robot.data.root_ang_vel_b*0.2,robot.data.projected_gravity_b,cmd,
               robot.data.joint_pos[:,ids]-default,robot.data.joint_vel[:,ids]*0.05,action]
        obs_parts=[]
        for h,v in zip(history,terms): h.append(v);obs_parts.append(h.buffer.flatten(1))
        observation=torch.cat(obs_parts,dim=-1)
        assert observation.shape==(1,255)
        action=policy(observation)
        target=robot.data.default_joint_pos.clone()
        target[:,ids]=default+0.5*torch.clamp(action,-10.,10.)
        for _ in range(decimation):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(physics_dt)
        if a.gui or a.snapshot: sim.render()
        state=robot.data.root_state_w[0].numpy().copy()
        r,pitch,yaw=euler_xyz_from_quat(robot.data.root_quat_w)
        vel=robot.data.root_lin_vel_b[0].numpy().copy();wz=float(robot.data.root_ang_vel_b[0,2])
        finite=bool(torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(action).all())
        row={'step':step,'time':t+dt,'segment':c['name'],'segment_elapsed':t-start,
             'command':cmd[0].tolist(),'goal':c['velocity'],'velocity_body':[float(vel[0]),float(vel[1]),wz],
             'base_position':state[:3].tolist(),'rpy':[float(r[0]),float(pitch[0]),float(yaw[0])],
             'finite':finite,'joint_position':robot.data.joint_pos[0].tolist()}
        rows.append(row)
        if not finite or state[2]<0.45 or abs(float(r[0]))>0.9 or abs(float(pitch[0]))>0.9:
            failures.append({'step':step,'time':t+dt,'reason':'nonfinite' if not finite else 'fallen','state':row});break
        # pace=1: no time dilation; physics and wall clock pacing reported separately.
        remaining=wall+(step+1)*dt-time.monotonic()
        if remaining>0: time.sleep(remaining)
    wall_elapsed=time.monotonic()-wall
    with (out/'trajectory.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    reports=[]
    for _,_,c in segments:
        sample=[r for r in rows if r['segment']==c['name'] and r['segment_elapsed']>=2.]
        if not sample:continue
        v=np.array([r['velocity_body'] for r in sample]);goal=np.array(c['velocity']);rp=np.array([r['rpy'][:2] for r in sample])
        mean=v.mean(axis=0);err=np.sqrt(((v-goal)**2).mean(axis=0));height=[r['base_position'][2] for r in sample]
        is_stop=c['name'] in ('stand','stop')
        linear_tolerance=0.08 if is_stop else 0.16
        yaw_tolerance=0.12 if is_stop else 0.2
        tracking=bool(np.linalg.norm(mean[:2]-goal[:2])<linear_tolerance and abs(mean[2]-goal[2])<yaw_tolerance)
        reports.append({'name':c['name'],'samples':len(sample),'goal':c['velocity'],'mean_velocity_body':mean.tolist(),
                        'rmse_velocity_body':err.tolist(),'height_min':min(height),'height_max':max(height),
                        'abs_roll_pitch_max':np.abs(rp).max(axis=0).tolist(),'linear_tolerance':linear_tolerance,'yaw_tolerance':yaw_tolerance,'tracking_pass':tracking})
    complete=len(rows)==round(total/dt)
    all_rp=np.abs(np.array([r['rpy'][:2] for r in rows])) if rows else np.array([[float('inf'),float('inf')]])
    all_heights=[r['base_position'][2] for r in rows]
    stable=bool(rows and min(all_heights)>=0.55 and all_rp.max()<=0.5)
    canonical_commands=json.loads((ROOT/'config/commands.json').read_text())
    full_suite=bool(a.seconds is None and commands==canonical_commands)
    selected_commands_pass=bool(complete and stable and not failures and len(reports)==len(commands) and all(x['tracking_pass'] for x in reports))
    passed=bool(full_suite and selected_commands_pass)
    summary={'stable_multidirection_pass':passed,'complete':complete,'full_suite':full_suite,'diagnostic_prefix':a.seconds is not None,'selected_commands_pass':selected_commands_pass,
        'sim_seconds':len(rows)*dt,'wall_seconds':wall_elapsed,'real_time_factor':len(rows)*dt/wall_elapsed,
        'resets_after_start':0,'all_time_abs_roll_pitch_max':all_rp.max(axis=0).tolist(),'all_time_height_min':min(all_heights) if rows else None,'stability_pass':stable,'failures':failures,'segments':reports}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import omni.usd
    omni.usd.get_context().get_stage().GetRootLayer().Export(str(out/'scene.usda'))
    if a.snapshot:
        from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
        viewport=get_active_viewport()
        if viewport:
            capture_viewport_to_file(viewport,str(out/'diagnostic.png'))
            for _ in range(12):app.update()
        else:
            (out/'snapshot_error.txt').write_text('No active viewport in headless IsaacLab experience; no PNG was produced. Physical measurements are retained.\n')
    print(json.dumps(summary,indent=2),flush=True)
    print('RESULT_DIR',out,flush=True)
    return 0 if selected_commands_pass or (a.seconds and complete and stable and not failures and all(x['tracking_pass'] for x in reports)) else 2
try:
    code=main()
except BaseException as e:
    import traceback
    (out/'error.txt').write_text(traceback.format_exc());raise
finally:
    context=SimulationContext.instance()
    if context is not None:context._disable_app_control_on_stop_handle=True
    (out/'run_complete.json').write_text(json.dumps({'returncode':locals().get('code',1)})+'\n')
    app.close(wait_for_replicator=False)
sys.exit(code)
