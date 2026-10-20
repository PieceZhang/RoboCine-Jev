"""Current upstream Giskard World/Statechart/Executor integration experiments.

Original physical URDF/assets stay untouched. Control-only URDF normalizes STL
encoding and freezes non-commanded joints at their physical hold setpoints.
"""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import timedelta

import numpy as np
from scipy.spatial.transform import Rotation

from wbc_backend import ROOT,ARM_JOINTS,add_pinned_core_paths,model_urdf


def control_urdf(cache_dir,sensor_config=None):
    import trimesh
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    document=ET.fromstring(model_urdf())
    if sensor_config is not None:
        camera=json.loads(Path(sensor_config).read_text())['robots']['tiago']['camera']
        forward,up=np.asarray(camera['forward_parent']),np.asarray(camera['up_parent'])
        optical=np.column_stack([np.cross(forward,up),-up,forward])
        if not np.allclose(optical.T@optical,np.eye(3),atol=1e-10) or not np.isclose(np.linalg.det(optical),1.):
            raise ValueError('Frozen physical camera axes must be an orthonormal optical frame')
        joint=document.find("joint[@name='tiago_camera_optical_fixed']")
        joint.find('parent').set('link',camera['parent'])
        joint.find('origin').set('xyz',' '.join(map(str,camera['translation_m'])))
        joint.find('origin').set('rpy',' '.join(map(str,Rotation.from_matrix(optical).as_euler('xyz'))))
    for link in document.findall('link'):
        for visual in list(link.findall('visual')):link.remove(visual)
    profile=json.loads((ROOT/'robot.json').read_text())
    fixed_values=dict(profile.get('passive_joint_positions',{}))
    fixed_values.update({name:0. for name in profile['left_wheels']+profile['right_wheels']})
    normalized=[]
    frozen_meshes=[]
    for mesh in document.findall('.//mesh'):
        source=Path(mesh.get('filename'))
        data=source.read_bytes()
        original_sha=hashlib.sha256(data).hexdigest()
        if source.suffix.lower()=='.stl':
            binary=len(data)>=84 and 84+50*int.from_bytes(data[80:84],'little')==len(data)
            if not binary:
                geometry=trimesh.load(source,force='mesh',process=False)
                target=cache_dir/(hashlib.sha256(data).hexdigest()+'.stl')
                if not target.exists():target.write_bytes(geometry.export(file_type='stl'))
                mesh.set('filename',str(target.resolve()))
                normalized.append(dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(data).hexdigest(),binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),triangle_count=len(geometry.faces)))
            else:
                target=cache_dir/(original_sha+source.suffix.lower())
                if not target.exists():target.write_bytes(data)
                mesh.set('filename',str(target.resolve()))
        else:
            raise ValueError('Control collision mesh format needs explicit native conversion: '+str(source))
        frozen_meshes.append(dict(source=str(source.relative_to(ROOT)),source_sha256=original_sha,
            executed_path=mesh.get('filename'),executed_sha256=hashlib.sha256(Path(mesh.get('filename')).read_bytes()).hexdigest()))
    frozen=[]
    for joint in document.findall('joint'):
        if joint.get('type')=='fixed' or joint.get('name') in ARM_JOINTS:
            continue
        value=fixed_values.get(joint.get('name'))
        if value is None:raise ValueError('Unmapped passive joint '+joint.get('name'))
        origin=joint.find('origin')
        if origin is None:origin=ET.SubElement(joint,'origin',xyz='0 0 0',rpy='0 0 0')
        position=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        rotation=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
        axis=np.fromstring(joint.find('axis').get('xyz','1 0 0'),sep=' ')
        if joint.get('type')=='prismatic':position+=rotation.apply(axis*value)
        else:rotation=rotation*Rotation.from_rotvec(axis*value)
        origin.set('xyz',' '.join(map(str,position)));origin.set('rpy',' '.join(map(str,rotation.as_euler('xyz'))))
        frozen.append(dict(joint=joint.get('name'),position=value))
        joint.set('type','fixed')
        for node in list(joint):
            if node.tag not in ('origin','parent','child'):joint.remove(node)
    executed=ET.tostring(document,encoding='unicode')
    (cache_dir/'control_world.urdf').write_text(executed)
    return executed,dict(stl_binary_normalization=normalized,collision_meshes=frozen_meshes,
        executed_control_urdf_sha256=hashlib.sha256(executed.encode()).hexdigest(),
        source_urdf_sha256=hashlib.sha256((ROOT/'urdf/tiago_physics.urdf').read_bytes()).hexdigest(),
        frozen_camera_config_sha256=hashlib.sha256(Path(sensor_config or ROOT/'sensors.json').read_bytes()).hexdigest(),
        control_only_visuals_omitted=True,control_only_frozen_joints=frozen,original_physical_model_modified=False)


def build_world(cache_dir,sensor_config=None):
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    add_pinned_core_paths()
    from giskardpy.model.world_config import WorldWithDiffDriveRobot
    from semantic_digital_twin.world_description.connections import DifferentialDrive
    urdf,manifest=control_urdf(cache_dir,sensor_config)
    config=WorldWithDiffDriveRobot(urdf=urdf)
    with config.world.modify_world():
        config.setup_world()
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    robots=config.world.get_semantic_annotations_by_type(AbstractRobot)
    if len(robots)!=1:raise RuntimeError('Expected one world-owned robot annotation after merge')
    config.robot=robots[0]
    drives=[c for c in config.world.connections if isinstance(c,DifferentialDrive)]
    if len(drives)!=1:raise RuntimeError('Expected real DifferentialDrive connection')
    drives[0].has_hardware_interface=True
    manifest.update(body_count=len(config.world.bodies),active_dof_names=[str(d.name) for d in config.world.active_degrees_of_freedom],
        localization_active_dof_count=len(config.localization.active_dofs),localization_passive_dof_count=len(config.localization.passive_dofs))
    if len(config.world.active_degrees_of_freedom)!=10:
        raise RuntimeError('Control world must contain precisely 10 active coordinates')
    return config,drives[0],manifest


class FullGiskardExecutor:
    """Actual current World -> CartesianPose/Task -> Statechart -> Executor.

    The additional FCL projection is an independent, explicitly named safety
    layer. It does not replace this upstream task/controller solve.
    """
    def __init__(self,spec,metadata,cache_dir,derivative_mode='jerk_issued_velocity',braking_time_s=.05,sensor_config=None):
        add_pinned_core_paths()
        from dataclasses import dataclass,field
        import krrood.symbolic_math.symbolic_math as sm
        from giskardpy.motion_statechart.context import MotionStatechartContext
        from giskardpy.motion_statechart.graph_node import Task,NodeArtifacts
        from giskardpy.motion_statechart.motion_statechart import MotionStatechart
        from giskardpy.motion_statechart.goals.collision_avoidance import SelfCollisionAvoidance
        from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
        from giskardpy.qp.qp_controller_config import QPControllerConfig
        from giskardpy.executor import Executor
        from semantic_digital_twin.collision_checking.collision_rules import AvoidSelfCollisions,AllowCollisionRule
        from semantic_digital_twin.collision_checking.collision_matrix import CollisionCheck
        from semantic_digital_twin.spatial_types.derivatives import Derivatives
        from semantic_digital_twin.datastructures.joint_state import JointState
        from native_solver import NativeGiskardSolver
        from wbc_backend import GiskardCameraGoalBinding,CameraReferenceSample,CAMERA_FRAME
        self.spec=spec
        self.config,self.drive,self.model_manifest=build_world(cache_dir,sensor_config)
        world=self.config.world
        self.world=world
        arm={str(d.name).split('/')[-1]:d for d in world.active_degrees_of_freedom if str(d.name).split('/')[-1] in ARM_JOINTS}
        self.dofs=[self.drive.x_velocity,self.drive.yaw]+[arm[name] for name in ARM_JOINTS]
        self.camera=world.get_body_by_name(CAMERA_FRAME)
        # Undo MinimalRobot's generic 1rad/s tightening; retain official limits.
        for dof,joint in zip(self.dofs[2:],spec['joints']):
            dof.limits.lower.position=joint['lower'];dof.limits.upper.position=joint['upper']
            dof.limits.lower.velocity=-joint['velocity'];dof.limits.upper.velocity=joint['velocity']
            # URDF declares no acceleration limit. The unchanged issued-command
            # acceleration and stopping policy is enforced by the independent
            # native safety QP at every actual command, rather than inventing a
            # full-horizon triangular-jerk assumption at maximum hardware speed.
            dof.limits.lower.acceleration=None
            dof.limits.upper.acceleration=None
            world.state[dof.id].position=joint['home']
        for dof,speed,acceleration in zip(self.dofs[:2],[.22,.35],[.45,.65]):
            dof.limits.lower.velocity=-speed;dof.limits.upper.velocity=speed
            dof.limits.lower.acceleration=None;dof.limits.upper.acceleration=None
        world.notify_state_change()
        if derivative_mode not in ('velocity','jerk_fd_diagnostic','jerk_zero_estimate','jerk_command_estimate','jerk_issued_state','jerk_issued_velocity'):
            raise ValueError('Unsupported upstream derivative configuration')
        self.derivative_mode=derivative_mode
        context=MotionStatechartContext(world=world,qp_controller_config=QPControllerConfig(target_frequency=100.,prediction_horizon=50,
            max_derivative=Derivatives.velocity if derivative_mode=='velocity' else Derivatives.jerk,
            braking_time=timedelta(seconds=braking_time_s),verbose=False))
        self.model_manifest.update(controller_derivative_mode=derivative_mode,controller_frequency_hz=100.,prediction_horizon=50,
            native_braking_time_s=braking_time_s,unmeasured_acceleration_estimate={
                'jerk_zero_estimate':'zero; not a physical acceleration measurement',
                'jerk_command_estimate':'issued-command finite difference; not a physical acceleration measurement',
                'jerk_issued_state':'issued-command finite difference; velocity is issued-command state, q/fullSE3 remain measured',
                'jerk_issued_velocity':'zero acceleration estimate with issued-command velocity; q/fullSE3 remain measured',
                'jerk_fd_diagnostic':'raw measured-rate finite difference; failed historical diagnostic',
                'velocity':'unused higher-order estimate; upstream configuration unsupported at pinned version'}[derivative_mode],
            issued_acceleration_and_braking_policy='unchanged independent genuine-Giskard safety QP plus command guard')
        self.context=context
        self.camera_enabled=sm.FloatVariable('tiago_camera_enabled')
        context.float_variable_data.register_expression(self.camera_enabled)
        context.float_variable_data.set_value(self.camera_enabled,0.)
        # Native QP cost preference; physical joint ranges and hard guards are
        # unchanged. Penalize approaching a limit before the hard brake becomes
        # active, so the redundant mobile/torso/arm model can choose another path.
        self.joint_costs=[sm.FloatVariable(f'tiago_native_joint_cost_{i}') for i in range(8)]
        for variable in self.joint_costs:
            context.float_variable_data.register_expression(variable)
            context.float_variable_data.set_value(variable,.002)
        for dof,cost,speed in zip(self.dofs[:2],[.06,.03],[.22,.35]):
            context.qp_controller_config.set_dof_weight(dof.name,Derivatives.velocity,cost*speed**2)
        for dof,variable,joint in zip(self.dofs[2:],self.joint_costs,spec['joints']):
            context.qp_controller_config.set_dof_weight(dof.name,Derivatives.velocity,variable*joint['velocity']**2)
        self.posture=JointPositionList(name='native_camera_home_preference',
            goal_state=JointState.from_str_dict({j['name']:j['home'] for j in spec['joints']},world),
            weight=.003*self.camera_enabled,max_velocity=.5)
        self.model_manifest.update(native_posture_goal='giskardpy JointPositionList, weight .003 only during camera task',
            native_limit_preference='effective velocity cost .002+.5*pressure; quadratic pressure within min(.25,.2*range), no range reduction',
            native_base_effective_velocity_cost=[.06,.03])
        initial=world.compute_forward_kinematics_np(world.root,self.camera)
        self.binding=GiskardCameraGoalBinding(context,CameraReferenceSample(-.01,initial,np.zeros(6)))
        self.binding.goal.weight=self.camera_enabled
        self.rate_targets=[sm.FloatVariable(f'tiago_rate_target_{i}') for i in range(10)]
        for variable in self.rate_targets:
            context.float_variable_data.register_expression(variable)
            context.float_variable_data.set_value(variable,0.)
        @dataclass(eq=False,repr=False)
        class ActuatorRateTask(Task):
            expressions:list=field(kw_only=True)
            targets:list=field(kw_only=True)
            enabled:object=field(kw_only=True)
            def build_artifacts(self,context):
                result=NodeArtifacts(observation=sm.Scalar.const_true())
                for i,(expression,target) in enumerate(zip(self.expressions,self.targets)):
                    result.constraints.add_velocity_constraint(name=f'physical_rate_{i}',
                        task_expression=expression,lower_velocity_limit=target,upper_velocity_limit=target,
                        quadratic_weight=100.*self.enabled,velocity_limit=1.)
                return result
        self.rate_task=ActuatorRateTask(name='physical_actuator_tracking',expressions=[d.variables.position for d in self.dofs],
            targets=self.rate_targets,enabled=1.-self.camera_enabled)
        manager=world.collision_manager
        bodies={body.name.name:body for body in world.bodies_with_collision}
        @dataclass
        class AllowedNamedBodyPairs(AllowCollisionRule):
            pairs:list=field(default_factory=list)
            def _update(self,world):
                self.allowed_collision_pairs={CollisionCheck.create_and_validate(body_a=a,body_b=b) for a,b in self.pairs}
        allowed=[(bodies[a],bodies[b]) for a,b in metadata['allowed_pairs'] if a in bodies and b in bodies]
        with world.modify_world():
            manager.add_default_rule(AvoidSelfCollisions(robot=self.config.robot,buffer_zone_distance=.05,violated_distance=0.))
            manager.add_ignore_collision_rule(AllowedNamedBodyPairs(pairs=allowed))
        statechart=MotionStatechart()
        statechart.add_node(self.binding.goal);statechart.add_node(self.rate_task)
        statechart.add_node(self.posture)
        statechart.add_node(SelfCollisionAvoidance(name='official_filtered_self_collision',robot=self.config.robot,cancel_if_collision_violated=True))
        self.executor=Executor(context)
        self.executor.compile(statechart)
        self.safety=NativeGiskardSolver(spec,metadata)
        self.model_manifest['independent_projection_piqp_settings']=self.safety.projection_settings
        self.commit=self.safety.commit
        self.last_sample=-np.inf
        self.last_measured_rates=None
        self.last_issued_rates=None

    def synchronize(self,request):
        from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
        base=np.asarray(request['poses'][self.spec['base_link']])
        world=self.world
        # Localization is passive: writing its measured full SE3 cannot become
        # a solver command. Reset the differential chart at this measured pose.
        self.config.localization.origin=HomogeneousTransformationMatrix(base,reference_frame=world.root)
        for dof in (self.drive.x,self.drive.y,self.drive.roll,self.drive.pitch,self.drive.yaw,self.drive.x_velocity):
            world.state[dof.id].position=0.
        measured=np.asarray(request.get('measured_rates',request['previous']))
        delta=float(request['sample_s']-self.last_sample)
        # Physical joint-state feedback measures position/velocity. Raw numerical
        # acceleration is retained only for replaying the failed jerk diagnostic.
        # The operational jerk controller uses issued velocity as its command
        # prediction state and explicitly estimates unobserved acceleration as
        # zero. Measured q/full base SE3 still close the Cartesian feedback loop;
        # raw physical qd is independently checked by the Isaac-side guards.
        acceleration=np.zeros(10)
        if delta>0:
            if self.derivative_mode=='jerk_fd_diagnostic' and self.last_measured_rates is not None:
                acceleration=(measured-self.last_measured_rates)/delta
            elif self.derivative_mode in ('jerk_command_estimate','jerk_issued_state') and self.last_issued_rates is not None:
                acceleration=(np.asarray(request['previous'])-self.last_issued_rates)/delta
        for i,dof in enumerate(self.dofs):
            if i>=2:world.state[dof.id].position=float(request['q'][i-2])
            world.state[dof.id].velocity=float(request['previous'][i] if self.derivative_mode in ('jerk_issued_state','jerk_issued_velocity') else measured[i])
            world.state[dof.id].acceleration=float(acceleration[i])
            world.state[dof.id].jerk=0.
        world.notify_state_change()
        self.last_sample=float(request['sample_s']);self.last_measured_rates=measured.copy()
        self.last_issued_rates=np.asarray(request['previous']).copy()

    def solve(self,request):
        from wbc_backend import CameraReferenceSample
        import time
        self.synchronize(request)
        measured_camera=request.get('measured_camera')
        camera_consistency=None
        if measured_camera is not None:
            modeled=self.world.compute_forward_kinematics_np(self.world.root,self.camera)
            measured_camera=np.asarray(measured_camera)
            camera_consistency=dict(position_m=float(np.linalg.norm(modeled[:3,3]-measured_camera[:3,3])),
                orientation_rad=float(Rotation.from_matrix(modeled[:3,:3]@measured_camera[:3,:3].T).magnitude()))
            if camera_consistency['position_m']>.01 or camera_consistency['orientation_rad']>.02:
                raise RuntimeError('Native source camera disagrees with actual measured optical frame: '+repr(camera_consistency))
        data=self.context.float_variable_data
        camera_mode=request['mode']=='camera'
        data.set_value(self.camera_enabled,float(camera_mode))
        q=np.asarray(request['q'])
        lower=np.array([j['lower'] for j in self.spec['joints']]);upper=np.array([j['upper'] for j in self.spec['joints']])
        zone=np.minimum(.25,.2*(upper-lower))
        pressure=np.clip(1.-np.minimum(q-lower,upper-q)/zone,0.,1.)**2
        for variable,value in zip(self.joint_costs,.002+.5*pressure):data.set_value(variable,float(value))
        if camera_mode:
            self.binding.update(CameraReferenceSample(request['sample_s'],np.asarray(request['reference']),np.asarray(request['feedforward'])))
        else:
            for variable,value in zip(self.rate_targets,request['desired_rates']):data.set_value(variable,float(value))
        started=time.perf_counter()
        self.executor.tick()
        proposal=np.array([self.world.state[dof.id].velocity for dof in self.dofs])
        framework_wall=time.perf_counter()-started
        # Explicit independent safety projection under unchanged measured bounds
        # and actual-USD FCL distance barriers, using genuine upstream solver.
        safety_request=dict(request,mode='actuator',desired_rates=proposal)
        result=self.safety.solve(safety_request)
        result.update(full_world_executor=True,upstream_task_controller='CartesianPose/ActuatorRateTask -> MotionStatechart -> Executor -> QPController',
            upstream_proposal=proposal,framework_tick_wall_s=framework_wall,
            projection_norm=float(np.linalg.norm(result['command']-proposal)),upstream_derivative_mode=self.derivative_mode,
            measured_native_camera_consistency=camera_consistency)
        return result
