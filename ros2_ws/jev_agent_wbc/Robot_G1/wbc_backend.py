"""Real Pink upper-body IK beside the official 12-leg AGILE height policy.

All transforms here are physical optical transforms (x right, y down, z forward).
The pelvis pose is measured and is never an optimization/execution variable.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
_CONTEXT_HELPER=None
_FOLLOWING_HELPER=None


def context_helper():
    global _CONTEXT_HELPER
    if _CONTEXT_HELPER is None:
        import importlib.util
        spec=importlib.util.spec_from_file_location('pink_solver_task_context',Path(__file__).with_name('solver_context.py'))
        _CONTEXT_HELPER=importlib.util.module_from_spec(spec);spec.loader.exec_module(_CONTEXT_HELPER)
    return _CONTEXT_HELPER


def following_helper():
    global _FOLLOWING_HELPER
    if _FOLLOWING_HELPER is None:
        import importlib.util
        spec=importlib.util.spec_from_file_location('pink_following_reserve',Path(__file__).with_name('following_reserve.py'))
        _FOLLOWING_HELPER=importlib.util.module_from_spec(spec);spec.loader.exec_module(_FOLLOWING_HELPER)
    return _FOLLOWING_HELPER


def ownership(policy_joints, robot_joints):
    legs = tuple(policy_joints)
    upper = tuple(n for n in robot_joints if n == 'waist_yaw_joint' or any(
        token in n for token in ('shoulder_', 'elbow_joint', 'wrist_')))
    held = tuple(n for n in robot_joints if n not in legs and n not in upper)
    if len(legs) != 12 or len(upper) != 15 or set(legs) & set(upper):
        raise ValueError('G1 height/Pink requires disjoint 12-leg and 15-upper ownership')
    if set(held) != {'waist_roll_joint', 'waist_pitch_joint'}:
        raise ValueError(f'unexpected unowned G1 joints: {held}')
    return dict(policy=legs, pink=upper, hold=held)


def rigid(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1])
            or not np.allclose(value[:3, :3].T@value[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-6)):
        raise ValueError('expected a finite rigid 4x4 transform')
    return value


class PinkUpperBody:
    """Fixed-pelvis, reduced-joint model; genuine pink.solve_ik/FrameTask.

    Legs and waist roll/pitch are locked at measured configuration before each
    solve. Only waist yaw and the two arms are returned to physical drives.
    Optional actual-USD geometry supplies genuine Pink upper collision barriers.
    Leg contact optimization is not part of this fixed-pelvis upper-body solver.
    """
    def __init__(self, policy_joints, robot_joints, T_wrist_camera_optical, urdf=None,
                 speed_limit=.5, acceleration_limit=1., posture_home=None, posture_cost=1e-3,
                 damping_cost=0.,waist_posture_cost=None,actual_gravity_metadata=None,collision_gain=.1):
        import pinocchio as pin
        import pink
        self.pin, self.pink = pin, pink
        self.model = pin.buildModelFromUrdf(str(urdf or ROOT/'kinematics/g1_29dof.urdf'))
        if self.model.nq != 29 or self.model.nv != 29:
            raise ValueError('expected the fixed-base 29-DoF G1; free-flyer IK is forbidden')
        self.owners = ownership(policy_joints, robot_joints)
        model_names = list(self.model.names)[1:]
        if set(model_names) != set(robot_joints):
            raise ValueError('physical and URDF joint sets differ')
        wrist_id = self.model.getFrameId('right_wrist_yaw_link')
        wrist = self.model.frames[wrist_id]
        camera = pin.SE3(rigid(T_wrist_camera_optical))
        self.model.addFrame(pin.Frame('cine_camera_optical', wrist.parent, wrist_id,
                                     wrist.placement*camera, pin.FrameType.OP_FRAME))
        self.speed_limit, self.acceleration_limit = speed_limit, acceleration_limit
        self.previous = {name: 0. for name in self.owners['pink']}
        self.posture_home, self.posture_cost, self.damping_cost = posture_home, posture_cost, damping_cost
        if waist_posture_cost is not None and (not np.isfinite(waist_posture_cost) or waist_posture_cost<=0):
            raise ValueError('positive finite soft waist posture cost required')
        self.waist_posture_cost=waist_posture_cost
        if isinstance(collision_gain,bool) or not np.isfinite(collision_gain) or not .1<=collision_gain<=1.:
            raise ValueError('bounded positive native collision class-K gain required')
        self.collision_gain=float(collision_gain)
        self.last_posture_report=None
        self.collision_geometry=None;self.collision_offsets={};self.last_collision_report=None
        self.last_qp_failure=None;self.last_following_report=None
        self.actual_gravity=None
        if actual_gravity_metadata is not None:
            import importlib.util
            spec=importlib.util.spec_from_file_location('pink_actual_gravity_only',Path(__file__).with_name('gravity_model.py'))
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            self.actual_gravity=helper.ActualGravity(urdf or ROOT/'kinematics/g1_29dof.urdf',actual_gravity_metadata)

    def configure_collisions(self,metadata,measured_joints,T_world_pelvis,body_poses):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pink_actual_collision',Path(__file__).with_name('pink_collision.py'))
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        self.collision_helper=helper
        self.collision_offsets=helper.frame_offsets(self.model,metadata,measured_joints,T_world_pelvis,body_poses)
        self.collision_geometry=helper.geometry_model(self.model,metadata,self.collision_offsets)
        return dict(shape_count=len(metadata['shapes']),all_nonfiltered_pairs=len(self.collision_geometry.collisionPairs),
                    physical_body_frame_offsets=self.collision_offsets,
                    source='actual native USD geometry; measured one-time frame calibration; original source filters')

    def configuration(self, measured_joints):
        pin = self.pin
        full = pin.neutral(self.model)
        for name in list(self.model.names)[1:]:
            full[self.model.joints[self.model.getJointId(name)].idx_q] = measured_joints[name]
        locked = [self.model.getJointId(n) for n in self.owners['policy']+self.owners['hold']]
        collision=None
        if self.collision_geometry is None:model=pin.buildReducedModel(self.model,locked,full)
        else:model,collision=self.collision_helper.upper_collision_model(self.model,self.collision_geometry,locked,full)
        if model.nq != 15 or model.nv != 15:
            raise ValueError('Pink reduced model must contain only the 15 owned upper joints')
        names = list(model.names)[1:]
        q = np.array([measured_joints[n] for n in names], dtype=float)
        return self.pink.Configuration(model, model.createData(), q,collision_model=collision,
            collision_data=collision.createData() if collision is not None else None), names

    def predicted_camera(self, measured_joints, T_world_pelvis):
        # FK does not require constructing/rechecking the reduced collision
        # model. The actual QP still performs that complete update once.
        full=self.pin.neutral(self.model)
        for name in list(self.model.names)[1:]:
            full[self.model.joints[self.model.getJointId(name)].idx_q]=measured_joints[name]
        data=self.model.createData();self.pin.forwardKinematics(self.model,data,full);self.pin.updateFramePlacements(self.model,data)
        return rigid(T_world_pelvis)@data.oMf[self.model.getFrameId('cine_camera_optical')].np

    def upper_gravity(self, measured_joints, T_world_pelvis):
        """Fixed-pelvis model gravity in the measured pelvis orientation.

        This is feedforward from the pinned inertial model, not a measurement
        of actual actuator effort or a source of floating-base commands.
        """
        if self.actual_gravity is not None:
            all_torques=self.actual_gravity.torque(measured_joints,T_world_pelvis)
            return {name:all_torques[name] for name in self.owners['pink']}
        full=self.pin.neutral(self.model)
        for name in list(self.model.names)[1:]:
            full[self.model.joints[self.model.getJointId(name)].idx_q]=measured_joints[name]
        self.model.gravity.linear=rigid(T_world_pelvis)[:3,:3].T@np.array([0.,0.,-9.81])
        torque=self.pin.computeGeneralizedGravity(self.model,self.model.createData(),full)
        return {name:float(torque[self.model.joints[self.model.getJointId(name)].idx_v]) for name in self.owners['pink']}

    def velocity(self, measured_joints, T_world_pelvis, T_world_camera_target_optical, dt,posture_reference=None,
                 posture_cost_override=None,damping_cost_override=None,following_evidence=None,following_task_context=None,
                 following_request_stamp_s=None):
        from pink.tasks import FrameTask, PostureTask, DampingTask
        from pink.limits import ConfigurationLimit, VelocityLimit
        self.last_qp_failure=None;self.last_following_report=None
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        for value,lower,upper in [(posture_cost_override,.03,.15),(damping_cost_override,.02,.5)]:
            if value is not None and (isinstance(value,bool) or not np.isfinite(value) or not lower<=value<=upper):
                raise ValueError('finite bounded native soft-task cost required')
        base_posture_cost=self.posture_cost if posture_cost_override is None else float(posture_cost_override)
        damping_cost=self.damping_cost if damping_cost_override is None else float(damping_cost_override)
        configuration, names = self.configuration(measured_joints)
        target = np.linalg.inv(rigid(T_world_pelvis)) @ rigid(T_world_camera_target_optical)
        # Physical PD drives and the asynchronous link add latency: do not
        # assume the entire Cartesian error can disappear in a single tick.
        frame = FrameTask('cine_camera_optical', position_cost=1., orientation_cost=.3,
                          lm_damping=1e-3, gain=.1)
        frame.set_target(self.pin.SE3(target))
        preferred=self.posture_home if posture_reference is None else posture_reference
        preferred_q=configuration.q.copy() if preferred is None else np.array([preferred[n] for n in names],float)
        if not np.isfinite(preferred_q).all():raise ValueError('finite posture preference required')
        posture_cost=base_posture_cost
        if self.waist_posture_cost is not None:
            posture_cost=np.full(len(names),base_posture_cost)
            index=names.index('waist_yaw_joint');posture_cost[index]=self.waist_posture_cost
            if self.posture_home is not None:preferred_q[index]=self.posture_home['waist_yaw_joint']
            self.last_posture_report=dict(waist_yaw_soft_cost=self.waist_posture_cost,
                waist_yaw_target_rad=float(preferred_q[index]),other_posture_cost=base_posture_cost,
                damping_cost=damping_cost,optimized_upper_dofs=len(names),
                waist_yaw_locked=False,camera_frame_task_unchanged=True,collision_barriers_unchanged=True)
        posture = PostureTask(cost=posture_cost, gain=.1 if preferred is not None else 1.)
        posture.set_target(preferred_q)
        previous = np.array([self.previous[n] for n in names])
        lo = np.maximum(-self.speed_limit, previous-self.acceleration_limit*dt)
        hi = np.minimum(self.speed_limit, previous+self.acceleration_limit*dt)
        class CommandLimit:
            def compute_qp_inequalities(inner, q, step):
                eye = np.eye(len(names))
                return np.vstack([eye, -eye]), np.r_[hi, -lo]*step
        limits = [ConfigurationLimit(configuration.model), VelocityLimit(configuration.model), CommandLimit()]
        limit_names=['native_configuration','native_velocity','owned_velocity_acceleration']
        tasks=[frame,posture]
        if damping_cost>0:
            tasks.append(DampingTask(cost=damping_cost))
        barriers=[]
        if configuration.collision_model is not None:
            from pink.barriers import SelfCollisionBarrier
            count=len(configuration.collision_model.collisionPairs)
            if count:
                barriers=[SelfCollisionBarrier(min(20,count),gain=self.collision_gain,safe_displacement_gain=0.,d_min=.0065)]
                self.last_collision_report=dict(upper_controlled_pairs=count,
                    minimum_upper_distance_m=float(min(x.min_distance for x in configuration.collision_data.distanceResults)),
                    pink_barrier_minimum_m=.0065,pink_barrier_gain_per_s=self.collision_gain,barrier_type='pink.barriers.SelfCollisionBarrier',
                    continuous_swept_certificate=False)
        if following_evidence is not None:
            if not barriers or not following_task_context:
                raise ValueError('following reserve requires actual geometry and a current full Task context')
            if (isinstance(following_request_stamp_s,bool) or not isinstance(following_request_stamp_s,(int,float))
                    or not np.isfinite(following_request_stamp_s)
                    or abs(following_evidence['acquisition_stamp_s']-following_request_stamp_s)>1e-9):
                raise ValueError('following evidence acquisition must equal the outer measured request stamp')
            reserve=following_helper().FollowingReserveLimit.from_native_barrier(configuration,barriers[0],following_evidence,
                task_id=following_task_context['task_id'],task_sha256=following_task_context['task_sha256'],
                reference_token=following_evidence['committed_reference']['token'],now_s=following_request_stamp_s)
            limits.append(reserve);limit_names.append('native_following_reserve')
            self.last_following_report=reserve.manifest()
        try:
            velocity = self.pink.solve_ik(configuration, tasks, dt, solver='quadprog',
                                          limits=limits,barriers=barriers,safety_break=True)
        except Exception as error:
            # No retry and no change to constraints. Persist the actual request
            # and previous QP command. Pink keeps the exact failed Problem in
            # its exception frame. Capture that object when available, without
            # touching the solver call; a reconstruction is explicitly labelled.
            self.last_qp_failure=dict(exception=repr(error),joint_names=names,dt_s=float(dt),
                previous_owned_velocity=previous.tolist(),owned_velocity_lower=lo.tolist(),owned_velocity_upper=hi.tolist(),
                task_costs=dict(posture=base_posture_cost,damping=damping_cost),
                collision_class_K_gain_per_s=self.collision_gain,collision_minimum_distance_m=.0065,
                execution_retried=False)
            try:
                problem=None;trace=error.__traceback__
                while trace is not None:
                    if trace.tb_frame.f_code is self.pink.solve_ik.__code__:
                        problem=trace.tb_frame.f_locals.get('problem')
                        break
                    trace=trace.tb_next
                self.last_qp_failure['reconstructed_after_solver_failure']=problem is None
                self.last_qp_failure['qp_capture']='native_pink_exception_frame' if problem is not None else 'reconstructed_same_state'
                if problem is None:problem=self.pink.build_ik(configuration,tasks,dt,limits=limits,barriers=barriers)
                self.last_qp_failure['qp']={name:(None if getattr(problem,name) is None else np.asarray(getattr(problem,name)).tolist())
                    for name in ('P','q','G','h','A','b')}
                groups=[];offset=0
                for name,limit in zip(limit_names,limits):
                    rows=limit.compute_qp_inequalities(configuration.q,dt)
                    count=0 if rows is None else len(rows[1])
                    group=dict(name=name,row_start=offset,row_count=count)
                    if name=='native_following_reserve':group['pairs']=self.last_following_report['pairs']
                    groups.append(group);offset+=count
                for barrier in barriers:
                    distances=np.array([r.min_distance for r in configuration.collision_data.distanceResults])
                    indices=np.argpartition(-distances,-barrier.dim)[-barrier.dim:]
                    pairs=[]
                    for index in indices:
                        pair=configuration.collision_model.collisionPairs[int(index)]
                        pairs.append(dict(index=int(index),distance_m=float(distances[index]),
                            shapes=[configuration.collision_model.geometryObjects[k].name for k in (pair.first,pair.second)]))
                    groups.append(dict(name='native_pink_self_collision',row_start=offset,row_count=len(indices),pairs=pairs))
                    offset+=len(indices)
                self.last_qp_failure['inequality_groups']=groups
            except Exception as diagnostic_error:
                self.last_qp_failure['diagnostic_error']=repr(diagnostic_error)
            raise
        if (not np.isfinite(velocity).all() or np.any(velocity < lo-1e-6) or np.any(velocity > hi+1e-6)):
            raise RuntimeError('Pink returned a command outside upper-body bounds')
        self.previous.update(zip(names, map(float, velocity)))
        return dict(zip(names, map(float, velocity)))


class PinkProcess:
    """Prestarted isolated Pink worker; the physical loop never waits for it.

    One live request and at most one cancelled request may be in flight. A
    missing response invalidates that request and permits a fresh measured-state
    replacement. Cancelled velocities never resume, even within the old lease.
    """
    def __init__(self, policy_joints, robot_joints, T_wrist_camera_optical, urdf,
                 source_path=None, stderr_path=None, max_age_s=.06, solver_options=None):
        import os
        import subprocess
        import sys
        self._seq = 0; self.pending = None; self.cancelled = None; self.max_age_s = max_age_s
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        self.owners = ownership(policy_joints, robot_joints)
        self.needs_reset = False; self.failure_count = 0
        self._recovery_started_wall_s=None;self._last_state_stamp=-np.inf
        self._recovery_started_stamp_s=None;self._recovery_started_context=None
        self.stderr = open(stderr_path, 'w') if stderr_path else subprocess.DEVNULL
        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        environment.pop('PYTHONPATH', None)
        self._receive_buffer=b''
        self.process = subprocess.Popen([sys.executable, str(source_path or Path(__file__).resolve()), '--pink-worker'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                        text=False, bufsize=0, env=environment)
        self._send(dict(op='init', policy_joints=policy_joints, robot_joints=robot_joints,
                        T_wrist_camera_optical=np.asarray(T_wrist_camera_optical).tolist(), urdf=str(urdf),
                        solver_options=solver_options or {}))
        result = self._receive(15.)  # Initialization only, before physical time starts.
        if not result or not result.get('ok'):
            self.close(); raise RuntimeError(f'isolated Pink initialization failed: {result}')
        self.runtime = result['runtime']

    def _send(self, request):
        import json
        import time
        request['seq'] = self._seq; self._seq += 1
        self.process.stdin.write((json.dumps(request)+'\n').encode()); self.process.stdin.flush()
        self.pending = dict(request=request, sent_wall_s=time.monotonic())

    def _receive(self, wait_s=0.):
        import json
        import os
        import select
        import time
        deadline=time.monotonic()+wait_s
        while b'\n' not in self._receive_buffer:
            if not select.select([self.process.stdout],[],[],max(0.,deadline-time.monotonic()))[0]:
                return None
            data=os.read(self.process.stdout.fileno(),65536)
            if not data:return dict(ok=False,error='Pink worker exited')
            self._receive_buffer+=data
        line,self._receive_buffer=self._receive_buffer.split(b'\n',1)
        return json.loads(line)

    def warmup(self, measured_joints):
        """Warm FK and first QP before SimulationApp/physical clock startup."""
        self.pending = None
        self._send(dict(op='step', joints=measured_joints, T_world_pelvis=np.eye(4).tolist(),
                        target=None, stamp_s=0., dt=.02, reset_command=True))
        result = self._receive(15.)
        if not result or not result.get('ok'):
            raise RuntimeError(f'Pink warmup failed: {result}')
        self.pending = None
        self._send(dict(op='step', joints=measured_joints, T_world_pelvis=np.eye(4).tolist(),
                        target=result['predicted_camera'], stamp_s=0., dt=.02, reset_command=True))
        result = self._receive(15.)
        if not result or not result.get('ok'):
            raise RuntimeError(f'Pink first QP warmup failed: {result}')
        self.optimized_joint_names=tuple(result['velocity'])
        if len(self.optimized_joint_names)!=15 or set(self.optimized_joint_names)!=set(self.owners['pink']):
            raise RuntimeError('real Pink reduced joint order unavailable from warmup')
        self.pending = None; self.needs_reset = True

    def configure_collisions(self,metadata,measured_joints,T_world_pelvis,body_poses):
        if self.pending is not None:raise RuntimeError('collision initialization requires idle pre-physics worker')
        self._send(dict(op='collision_init',metadata=metadata,joints=measured_joints,
            T_world_pelvis=rigid(T_world_pelvis).tolist(),body_poses={n:np.asarray(T).tolist() for n,T in body_poses.items()}))
        response=self._receive(20.)
        if not response or not response.get('ok'):raise RuntimeError(f'Pink actual-geometry initialization failed: {response}')
        self.pending=None;self.needs_reset=True
        return response['collision_initialization']

    def verify_actual_gravity(self,body_names,masses,centers_actor,body_poses,joints,T_world_pelvis):
        """Verify native measurements in the isolated process before timing.

        Never import the Pin binary into the Kit process. This synchronous call
        is only allowed at initialization while there is no physical request.
        """
        if self.pending is not None or getattr(self,'cancelled',None) is not None:
            raise RuntimeError('gravity initialization requires an idle pre-physics worker')
        self._send(dict(op='gravity_verify',body_names=list(body_names),masses=np.asarray(masses).tolist(),
            centers_actor=np.asarray(centers_actor).tolist(),body_poses={n:np.asarray(T).tolist() for n,T in body_poses.items()},
            joints=joints,T_world_pelvis=rigid(T_world_pelvis).tolist()))
        response=self._receive(15.)
        if not response or not response.get('ok'):
            raise RuntimeError(f'actual gravity native verification failed: {response}')
        self.pending=None;self.needs_reset=True
        return response['gravity_verification']

    def submit(self, measured_joints, T_world_pelvis, target, stamp_s, measured_camera, dt=.02,measured_bodies=None,posture_reference=None,
               posture_cost_override=None,damping_cost_override=None,task_context=None,following_evidence=None):
        import copy
        if self.pending is not None:
            return False
        if (not np.isfinite(stamp_s) or stamp_s<=getattr(self,'_last_state_stamp',-np.inf)
                or not np.isfinite(dt) or dt<=0):
            raise ValueError('each Pink submission requires a strictly newer finite measured-state stamp and positive dt')
        if self.process.poll() is not None:
            self.needs_reset = True; self.failure_count += 1
            return False
        self._send(dict(op='step', joints=measured_joints, T_world_pelvis=rigid(T_world_pelvis).tolist(),
                        target=rigid(target).tolist() if target is not None else None,
                        stamp_s=float(stamp_s), dt=float(dt), reset_command=self.needs_reset,
                        posture_reference=posture_reference,posture_cost_override=posture_cost_override,
                        damping_cost_override=damping_cost_override,task_context=context_helper().snapshot(task_context),
                        following_evidence=copy.deepcopy(following_evidence)))
        self.pending['measured_camera'] = rigid(measured_camera).copy()
        self.pending['measured_bodies']=None if measured_bodies is None else {n:np.asarray(T).tolist() for n,T in measured_bodies.items()}
        self.pending['recovery_request']=bool(getattr(self,'_recovery_started_wall_s',None) is not None
            or getattr(self,'_task_transition_wait',False))
        self._task_transition_wait=False
        self._last_state_stamp=float(stamp_s)
        self.needs_reset = False
        return True

    def poll(self, current_stamp_s,current_context=None,execution_horizon_s=0.):
        import time
        import select
        current_context=context_helper().snapshot(current_context)
        if (isinstance(execution_horizon_s,bool) or not np.isfinite(execution_horizon_s)
                or not 0<=execution_horizon_s<=self.max_age_s):
            raise ValueError('finite bounded command execution horizon required')
        now=time.monotonic();previous_poll=getattr(self,'_last_poll_wall_s',None);self._last_poll_wall_s=now
        def clock_report(event,sampled_wall_s=None):
            # Diagnostic only: expose the exact clock already used by the
            # existing180ms recovery gate, before a successful reply clears it.
            started=getattr(self,'_recovery_started_wall_s',None)
            sampled=max(now,started) if sampled_wall_s is None and started is not None else (
                now if sampled_wall_s is None else sampled_wall_s)
            return dict(schema='g1_module_recovery_clock/v1',event=event,poll_wall_s=now,
                poll_stamp_s=float(current_stamp_s),sample_wall_s=sampled,
                poll_interval_wall_s=None if previous_poll is None else now-previous_poll,
                fault_detected_wall_s=started,recovery_started_wall_s=started,
                recovery_started_stamp_s=getattr(self,'_recovery_started_stamp_s',None),
                recovery_started_context=getattr(self,'_recovery_started_context',None),
                elapsed_wall_s=None if started is None else sampled-started,
                budget_s=3*self.max_age_s,original_motion_lease_s=self.max_age_s)
        def traced(packet,event,sampled_wall_s=None):
            packet['recovery_clock']=clock_report(event,sampled_wall_s)
            return packet
        def deadline_packet(reason):
            outstanding=[x for x in (getattr(self,'cancelled',None),self.pending) if x is not None]
            stream=getattr(self.process,'stdout',None)
            readable=bool(select.select([stream],[],[],0.)[0]) if stream is not None else None
            self.process.terminate();self.needs_reset=True;self.failure_count=3
            return traced(dict(ready=False,status=reason,
                deadline_records=[dict(seq=x['request']['seq'],stamp_s=x['request'].get('stamp_s'),
                    wall_age_s=now-x['sent_wall_s'],
                    sim_age_s=current_stamp_s-x['request'].get('stamp_s',current_stamp_s)) for x in outstanding],
                poll_interval_wall_s=None if previous_poll is None else now-previous_poll,
                pipe_readable=readable,buffered_complete_reply=b'\n' in getattr(self,'_receive_buffer',b''),
                continuous_recovery_wall_s=None if recovery_started is None else now-recovery_started),'deadline')
        cancelled=getattr(self,'cancelled',None)
        outstanding=[x for x in (cancelled,self.pending) if x is not None]
        if not outstanding:
            return traced(dict(ready=False, status='idle'),'idle')
        recovery_started=getattr(self,'_recovery_started_wall_s',None)
        discarded=[]
        for _ in range(len(outstanding)):
            result=self._receive(0.)
            if result is None:break
            cancelled=getattr(self,'cancelled',None)
            if cancelled is not None and result.get('seq')==cancelled['request']['seq']:
                discarded.append(dict(seq=result.get('seq'),stamp_s=result.get('stamp_s'),
                    reason=cancelled.get('cancellation_reason','cancelled_after_missing_response'),ok=bool(result.get('ok'))))
                self.cancelled=None
                if not result.get('ok') and cancelled.get('cancellation_reason')!='superseded_task':self.failure_count+=1
                continue
            request=self.pending;self.pending=None
            task_status=context_helper().reply_status(request['request'].get('task_context') if request else None,
                result.get('task_context'),current_context)
            if task_status!='same_task':
                self.needs_reset=True
                if task_status=='superseded_task':
                    self._task_transition_wait=True
                    return traced(dict(ready=False,status='superseded_task_hold_upper_targets',
                        discarded_results=discarded,dropped_seq=result.get('seq'),task_context_status=task_status,
                        reply_task_context=result.get('task_context'),current_task_context=current_context),'superseded_task')
                self.failure_count+=1
                if getattr(self,'_recovery_started_wall_s',None) is None:
                    self._recovery_started_wall_s=now
                    self._recovery_started_stamp_s=float(current_stamp_s);self._recovery_started_context=current_context
                return traced(dict(ready=False,status=task_status,failure_kind='transport_failure',
                    discarded_results=discarded,dropped_seq=result.get('seq'),task_context_status=task_status),'failure_detected')
            if recovery_started is not None and now-recovery_started>3*self.max_age_s:
                # Drain first, but never resume after the bounded recovery
                # budget. This is distinct from a worker with no response.
                return deadline_packet('recovery_deadline_exceeded_hold_upper_targets')
            wall_age=None if request is None else time.monotonic()-request['sent_wall_s']
            following_transport_failure=False
            if request is not None and request['request'].get('following_evidence') is not None and result.get('ok'):
                import hashlib
                import json
                expected_evidence_sha=hashlib.sha256(json.dumps(request['request']['following_evidence'],
                    sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                report=result.get('following_reserve_report')
                binding=report.get('worker_request_binding',{}) if isinstance(report,dict) else {}
                following_transport_failure=not (isinstance(report,dict)
                    and isinstance(binding,dict)
                    and report.get('evidence_sha256')==expected_evidence_sha
                    and report.get('acquisition_stamp_s')==request['request']['stamp_s']
                    and binding.get('seq')==request['request']['seq']
                    and binding.get('stamp_s')==request['request']['stamp_s']
                    and binding.get('task_sha256')==request['request']['task_context']['task_sha256']
                    and binding.get('evidence_sha256')==expected_evidence_sha)
                if following_transport_failure:
                    result.update(ok=False,error='following native report differs from the original outer request',
                        failed_request=request['request'])
            valid=(request is not None and result.get('ok') and not request.get('held_since_submit',False)
                and result.get('seq')==request['request']['seq']
                and result.get('stamp_s')==request['request'].get('stamp_s')
                and 0<=current_stamp_s-result['stamp_s']
                and current_stamp_s-result['stamp_s']+execution_horizon_s<=self.max_age_s+1e-9
                and 0<=wall_age and wall_age+execution_horizon_s<=self.max_age_s+1e-9)
            if not valid:
                self.needs_reset=True;self.failure_count+=1
                if getattr(self,'_recovery_started_wall_s',None) is None:
                    self._recovery_started_wall_s=now
                    self._recovery_started_stamp_s=float(current_stamp_s);self._recovery_started_context=current_context
                return traced(dict(ready=False,status='stale_or_failed_result_dropped',error=result.get('error'),
                    failure_kind='transport_failure' if following_transport_failure else (
                        'solver_failure' if not result.get('ok') else 'stale_or_mismatched_transport'),
                    discarded_results=discarded,dropped_seq=result.get('seq'),dropped_stamp_s=result.get('stamp_s'),
                    failed_request=result.get('failed_request'),qp_failure=result.get('qp_failure'),
                    following_reserve_report=result.get('following_reserve_report'),
                    wall_age_s=None if request is None else now-request['sent_wall_s'],
                    sim_age_s=None if request is None else current_stamp_s-request['request'].get('stamp_s',current_stamp_s)),
                    'failure_detected')
            self.failure_count=0
            self.needs_reset=False
            result.update(ready=True,status='fresh_command',acquisition_camera=request['measured_camera'].tolist(),
                acquisition_bodies=request.get('measured_bodies'),discarded_results=discarded,
                acquisition_stamp_s=request['request']['stamp_s'],task_context_status='same_task',
                accepted_wall_age_s=time.monotonic()-request['sent_wall_s'],execution_horizon_s=float(execution_horizon_s),
                command_sent_wall_s=request['sent_wall_s'],lease_deadline_wall_s=request['sent_wall_s']+self.max_age_s,
                lease_deadline_sim_s=request['request']['stamp_s']+self.max_age_s)
            result=traced(result,'resumed' if getattr(self,'_recovery_started_wall_s',None) is not None else 'fresh',
                request['sent_wall_s']+result['accepted_wall_age_s'])
            self._recovery_started_wall_s=None
            self._recovery_started_stamp_s=None;self._recovery_started_context=None
            return result
        outstanding=[x for x in (getattr(self,'cancelled',None),self.pending) if x is not None]
        # A completed but expired reply above was discarded, not mistaken for a
        # hung worker. Truly unresponsive work retains the original180ms bound.
        if any(now-x['sent_wall_s']>3*self.max_age_s for x in outstanding):
            return deadline_packet('worker_deadline_exceeded_hold_upper_targets')
        if recovery_started is not None and now-recovery_started>3*self.max_age_s:
            return deadline_packet('recovery_deadline_exceeded_hold_upper_targets')
        self.needs_reset=True
        superseded=bool(self.pending is not None and self.pending['request'].get('task_context')!=current_context)
        if superseded:self._task_transition_wait=True
        if not superseded and getattr(self,'_recovery_started_wall_s',None) is None:
            self._recovery_started_wall_s=time.monotonic()
            self._recovery_started_stamp_s=float(current_stamp_s);self._recovery_started_context=current_context
        if self.pending is not None:
            recovery_fresh=bool(not superseded and self.pending.get('recovery_request')
                and 0<=current_stamp_s-self.pending['request']['stamp_s']<=self.max_age_s
                and now-self.pending['sent_wall_s']<=self.max_age_s)
            if not recovery_fresh:
                self.pending['held_since_submit']=True
                if superseded:self.pending['cancellation_reason']='superseded_task'
                if getattr(self,'cancelled',None) is None:
                    self.cancelled=self.pending;self.pending=None
        return traced(dict(ready=False,status='superseded_task_hold_upper_targets' if superseded else 'waiting_hold_upper_targets',
            discarded_results=discarded,
            replacement_allowed=self.pending is None,
            outstanding_requests=int(self.pending is not None)+int(getattr(self,'cancelled',None) is not None)),
            'superseded_task' if superseded else 'waiting')

    def close(self):
        if getattr(self, 'process', None) is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.)
            except Exception:
                self.process.kill(); self.process.wait(timeout=2.)
        if hasattr(getattr(self, 'stderr', None), 'close'):
            self.stderr.close()


def _pink_worker():
    import importlib.metadata
    import json
    import sys
    import traceback
    solver = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            if request['op'] == 'init':
                solver = PinkUpperBody(request['policy_joints'], request['robot_joints'], request['T_wrist_camera_optical'],
                    request['urdf'], **request.get('solver_options',{}))
                result = dict(runtime={name: importlib.metadata.version(name) for name in ['pin-pink', 'pin', 'quadprog']})
            elif request['op']=='collision_init':
                result=dict(collision_initialization=solver.configure_collisions(request['metadata'],request['joints'],
                    request['T_world_pelvis'],request['body_poses']))
                # First geometry update and QP happen before physical time.
                target=solver.predicted_camera(request['joints'],request['T_world_pelvis'])
                solver.velocity(request['joints'],request['T_world_pelvis'],target,.02)
                solver.previous=dict.fromkeys(solver.previous,0.)
            elif request['op']=='gravity_verify':
                if solver.actual_gravity is None:raise ValueError('actual gravity verification requires the measured model')
                result=dict(gravity_verification=solver.actual_gravity.verify_native(request['body_names'],
                    request['masses'],request['centers_actor'],request['body_poses'],request['joints'],request['T_world_pelvis']))
            else:
                solver.last_qp_failure=None
                if request.get('reset_command'):
                    solver.previous = dict.fromkeys(solver.previous, 0.)
                result = dict(stamp_s=request['stamp_s'], predicted_camera=solver.predicted_camera(
                    request['joints'], request['T_world_pelvis']).tolist(), velocity={},
                    gravity=solver.upper_gravity(request['joints'],request['T_world_pelvis']))
                if request['target'] is not None:
                    result['velocity'] = solver.velocity(request['joints'], request['T_world_pelvis'], request['target'], request['dt'],
                        posture_reference=request.get('posture_reference'),posture_cost_override=request.get('posture_cost_override'),
                        damping_cost_override=request.get('damping_cost_override'),following_evidence=request.get('following_evidence'),
                        following_task_context=request.get('task_context'),following_request_stamp_s=request['stamp_s'])
                    result['pink_posture_report']=solver.last_posture_report
                    result['following_reserve_report']=solver.last_following_report
                    if solver.last_following_report is not None:
                        result['following_reserve_report']['worker_request_binding']=dict(seq=request['seq'],
                            stamp_s=request['stamp_s'],task_sha256=request['task_context']['task_sha256'],
                            evidence_sha256=solver.last_following_report['evidence_sha256'])
                if solver.collision_geometry is not None:
                    result['predicted_collision_bodies']=solver.collision_helper.predicted_body_poses(solver.model,
                        request['joints'],request['T_world_pelvis'],solver.collision_offsets)
                    result['pink_collision_report']=solver.last_collision_report
            result.update(ok=True, seq=request['seq'],task_context=request.get('task_context'))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = dict(ok=False, seq=request['seq'], stamp_s=request.get('stamp_s'), error=repr(exc),
                task_context=request.get('task_context'))
            if request.get('op')=='step':
                result.update(failed_request=request,qp_failure=getattr(solver,'last_qp_failure',None),
                    following_reserve_report=getattr(solver,'last_following_report',None))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    _pink_worker()
