"""Measured-state adapter for the same prepared-shot/continuous-combo drivers.

Only this controller process waits for observations. The PhysX process advances
independently, checks every command on current geometry, and owns the watchdog.
No CPU integration result is ever returned as a physical robot observation.
"""
import json
import os
from pathlib import Path
import pickle
from queue import Empty
import time

import numpy as np

from .contracts import native
from .realtime_engine import offer


def phase_evidence(perception,offset):
    """Change clock origin, never acquisition age or freshness allowance."""
    if abs(offset)<1e-9:return perception
    from copy import deepcopy
    result=deepcopy(perception)
    seen=set()
    records=[result.geometry,result.visual,result.last_trusted_visual,
             *result.geometry_history,*result.visual_history]
    for record in records:
        if record is not None and id(record) not in seen:
            seen.add(id(record));record['stamp_s']-=offset
    return result


def compute_connector_preview(search,controller,state,goal,target,t,z):
    command=search.preference(controller,state,goal,target,t,z)
    return search,command


class AsyncConnectorPreview:
    """Run the unchanged connector forecast away from the measured servo.

    Returned forecasts remain preferences. Every issued velocity still passes
    the WBC and independent current-state physical certificates.
    """
    def __init__(self,search,pool):
        self.search,self.pool=search,pool
        self.pending=None;self.info={};self.goal=None;self.submitted=-float('inf')

    def preference(self,controller,state,goal,target,t,z):
        from types import SimpleNamespace
        if self.pending is not None and self.pending.done():
            try:self.search,_=self.pending.result()
            except Exception as error:self.info=dict(active=False,failure=str(error),retry=True)
            else:self.info=dict(self.search.info)
            self.pending=None
        if self.pending is None and t-self.submitted>=.2:
            from copy import copy
            snapshot=SimpleNamespace(collision=copy(controller.collision),limits=controller.limits,
                previous=controller.previous.copy(),self_collision=controller.self_collision)
            self.pending=self.pool.submit(compute_connector_preview,self.search,snapshot,
                state.copy(),goal.copy(),target.copy(),t,z)
            self.submitted=t;self.goal=goal.copy()
        self.info=dict(self.search.info,active=False,asynchronous=True)
        if self.search.info.get('active'):
            smooth=self.search.smooth
            if (self.search.info.get('method')=='whole_body_smooth_preview' and
                    smooth.controls is not None and t+.3<=smooth.times[-1] and
                    np.linalg.norm(goal-smooth.goal)<.02):
                self.info['active']=True
                return np.array([np.interp(t+.3,smooth.times,smooth.controls[:,j]) for j in range(8)])
            lattice=self.search.lattice
            if (lattice.selected is not None and t-lattice.updated<.2 and
                    np.linalg.norm(goal-lattice.goal)<.02):
                self.info['active']=True
                return np.r_[lattice.selected,target[2:]]
        return target


class MeasuredExecution:
    def __init__(self,inbox,outbox,stop,initial,limits,policy,command_mode='adaptive'):
        self.inbox,self.outbox,self.stop=inbox,outbox,stop
        self.snapshot=initial
        self.limits,self.policy=limits,policy
        self.command_mode=command_mode
        self.engines=[]
        self.armed=False
        self.completed=False
        self.last_info={}
        self.control_retries=0
        self.control_stamps={}
        self.connector_pool=None
        from .runtime_fault_injection import FaultSchedule
        self.fault_schedule=FaultSchedule()
        self.delay_aware_servo=False
        self.control_latency_history=[]
        self.profile=None
        self.execution_observer=None
        self.last_model_feedback_s=-float('inf')
        self.numeric_runtime=dict(environment={key:os.environ.get(key) for key in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
        try:
            from threadpoolctl import threadpool_info
            self.numeric_runtime['libraries']=[{k:r.get(k) for k in ('internal_api','prefix','num_threads')}
                for r in threadpool_info()]
        except ImportError:
            import ctypes
            libraries=[]
            maps=Path('/proc/self/maps')
            for path in sorted({line.split()[-1] for line in (maps.read_text().splitlines() if maps.exists() else [])
                    if 'openblas' in line and '.so' in line}):
                try:library=ctypes.CDLL(path)
                except OSError:continue
                for symbol in ('openblas_get_num_threads64_','openblas_get_num_threads',
                               'scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads'):
                    if hasattr(library,symbol):
                        function=getattr(library,symbol);function.restype=ctypes.c_int
                        libraries.append(dict(library=Path(path).name,num_threads=function()))
                        break
            self.numeric_runtime['libraries']=libraries

    def initial(self):
        s=self.snapshot
        return s['state'].copy(),float(s['base_z']),s['measured_control'].copy(),s['scene']

    def summary(self):
        return dict(mode='jev',installed_plans=sum(e.installed_plans for e in self.engines),
            jev_calls=sum(e.jev.decider.client.calls for e in self.engines),
            controller='shared CPU task driver with measured PhysX execution',session_done=self.completed,
            control_retries=self.control_retries,controller_numeric_runtime=self.numeric_runtime)

    def ready(self,engines):
        self.engines=engines
        if len(engines)>1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            from .shot_preview_worker import initialize_worker
            self.connector_pool=ProcessPoolExecutor(max_workers=1,
                mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker)
            self.connector_pool.submit(initialize_worker).result(timeout=30.)
        import gc
        gc.collect()
        gc.disable()
        offer(self.outbox,pickle.dumps(dict(ready=True,summary=self.summary(),limits=self.limits)))
        self.armed=True
        if self.profile is not None:self.profile.enable()
        self.receive(-1.)

    def bind_execution(self,engine,state=None,local_t=0.):
        if self.execution_observer is None:return
        from dataclasses import replace
        s=self.snapshot
        model=replace(self.execution_observer.model,snapshot_time_s=float(local_t),
            body_aware=bool(getattr(engine.wbc,'body_aware',False)))
        engine.execution_model=model.to_dict()
        engine.execution_capability=model.public(s['state'] if state is None else state,
            s['measured_control'],self.limits,s['base_z'],s.get('base_rotation'))
        engine.wbc.feedback_latency_s=model.delay_s+model.delay_jitter_s
        engine.wbc.expected_command_delay_s=model.delay_s
        engine.wbc.execution_safety_delay_s=min(self.policy.command_timeout_s,model.delay_s+model.delay_jitter_s)

    def prepare(self,function,engine,*args,**kwargs):
        self.bind_execution(engine)
        if self.execution_observer is not None:
            # Later combo actions are prepared at a planned stopped arrival,
            # not at the current measured configuration of the first action.
            # Expose that condition rather than attaching a current-state lease
            # to a capability computed for the wrong phase.
            import inspect
            bound=inspect.signature(function).bind(engine,*args,**kwargs).arguments
            capability=self.execution_observer.model.public(bound['state'],np.zeros(8),
                self.limits,bound['base_z'])
            capability.update(state_scope='planned_stopped_phase_start; revalidate_at_measured_arrival',
                state_stamp_s=None,valid_until_s=None,expected_command_effect_s=None)
            engine.execution_capability=capability

        # Bounded task preparation happens before arming, exactly as the CPU
        # success baseline. Live replanning never uses this blocking helper.
        errors=[]
        engine.resumable_preparation=True
        for attempt in range(self.policy.max_retries+1):
            try:return function(engine,*args,**kwargs)
            except (RuntimeError,ValueError,TimeoutError) as error:
                errors.append(str(error))
                engine.preparation_feedback=(getattr(engine,'preparation_feedback',[])+[str(error)])[-4:]
                engine._write('preparation_retry',0.,attempt=attempt+1,reason=str(error))
                if attempt==self.policy.max_retries:raise

    def report_execution_feedback(self,engine,evidence,local_t):
        stamp=self.snapshot['t']
        if evidence is None or stamp-self.last_model_feedback_s<.5:return
        # Rate-limit on the continuous physical clock. A new combo phase resets
        # its local clock; it must not suppress model feedback until that clock
        # catches up with the end of the preceding phase.
        evidence=dict(evidence,first_anomaly_s=local_t,physical_time_s=stamp,
            time_frame='phase_local_with_physical_time')
        engine.pending_feedback.append(dict(error=evidence['symptom'],time_s=local_t,failure=evidence))
        engine.pending_feedback=engine.pending_feedback[-4:]
        engine._write('execution_model_feedback',local_t,failure=evidence,
            capability=engine.execution_capability)
        self.last_model_feedback_s=stamp

    def transfer_step(self,free,state,scene,t,previous):
        free.wbc.previous=np.asarray(previous).copy()
        if self.execution_observer is not None:
            free.wbc.feedback_latency_s=self.execution_observer.model.delay_s+self.execution_observer.model.delay_jitter_s
            free.wbc.expected_command_delay_s=self.execution_observer.model.delay_s
            free.wbc.execution_safety_delay_s=min(self.policy.command_timeout_s,
                self.execution_observer.model.delay_s+self.execution_observer.model.delay_jitter_s)
        if self.connector_pool is not None and not isinstance(free.motion_preview,AsyncConnectorPreview):
            free.motion_preview=AsyncConnectorPreview(free.motion_preview,self.connector_pool)
        try:
            elapsed=self.control_dt(free)
            result=free.step(state,scene,t,.02)
            result[1]['measured_observation_interval_s']=elapsed
            result[1]['measured_control_interval_s']=.02
            self.control_retries=0
            return result
        except RuntimeError as error:
            self.control_retries+=1
            if self.control_retries>self.policy.max_retries:raise
            return np.zeros(8),dict(controller_retry=str(error),loop_wall_s=0.)

    def control_dt(self,controller):
        stamp=float(self.snapshot['t'])
        before=self.control_stamps.get(id(controller),stamp-.02)
        self.control_stamps[id(controller)]=stamp
        # A latest-observation queue can skip ticks. Rate constraints must use
        # the measured command interval, not silently halve available control
        # acceleration when a 40 ms observation interval is labelled 20 ms.
        return min(.1,max(.02,stamp-before))

    def receive(self,after):
        while not self.stop.is_set():
            try:s=pickle.loads(self.inbox.get(timeout=.1))
            except Empty:continue
            if s['t']>after+1e-9:
                self.snapshot=s
                self.snapshot_received_wall=time.monotonic()
                if s.get('control_status',{}).get('emergency_stop_required'):
                    raise RuntimeError('physical_safety_controller_latched')
                return s
        raise RuntimeError('physical_session_stopped')

    def step(self,engine,state,scene,t,dt,z):
        s=self.snapshot
        engine.wbc.previous=np.asarray(s['measured_control']).copy()
        if s.get('control_status',{}).get('waiting_for_certificate') is not None:
            # The actuator alone can resume after a fresh map and every current
            # sweep gate. Keep fresh sensor-bound brake requests flowing; do
            # not repeatedly solve an uncertifiable moving objective meanwhile.
            from .planning import joint_margin
            from .whole_body_qp import whole_body_kinematics
            target=engine.active.sample(t)[0]
            actual=whole_body_kinematics(state,z,s.get('base_rotation'))[0]
            return np.zeros(8),dict(waiting_for_physical_certificate=True,
                target_position=target.tolist(),position_error=float(np.linalg.norm(actual-target)),
                qp_violation=0.,controller_timing_missing=True,
                joint_margin_rad=joint_margin(state),loop_wall_s=0.,
                measured_control_interval_s=.02)
        engine.wbc.measured_limits=self.limits
        if self.execution_observer is not None:
            transport=s.get('execution_transport',{})
            evidence=self.execution_observer.observe(s['t'],state,s['measured_control'],
                transport.get('applied_command',s['measured_control']),self.limits,
                command_delay_s=None if transport.get('waiting') else transport.get('command_delay_s',transport.get('command_age_s')),
                command_sequence=transport.get('command_sequence'),transition=s.get('physical_transition'))
            self.bind_execution(engine,state,t)
            self.report_execution_feedback(engine,evidence,t)

        engine.perception=phase_evidence(s['perception'],s['t']-t)
        engine.set_camera_output(t,s['camera']['position'],s['camera']['rotation'],source='execution')
        engine.runtime_policy=self.policy
        elapsed=self.control_dt(engine)
        # The encoders already include every skipped physical tick. Bounds
        # and sweep certificates describe the NEXT 20 ms command interval,
        # not the elapsed observation gap. Using that gap as a future horizon
        # both spends extra acceleration authority and grows solve latency.
        dt=.02
        started=time.monotonic()
        if self.delay_aware_servo:
            compute_estimate=float(np.quantile(self.control_latency_history,.9)) if self.control_latency_history else .02
            engine.wbc.feedback_latency_s=max(0.,started-s['observation_wall'])+compute_estimate+.02
        try:
            for event in self.fault_schedule.due('controller_delay',s['t']):
                engine._write('diagnostic_fault_injected',t,fault=event,physics_continues=True)
                time.sleep(event['duration_s'])
            for event in self.fault_schedule.due('controller_error',s['t']):
                engine._write('diagnostic_fault_injected',t,fault=event,physics_continues=True)
                raise RuntimeError('injected_transient_controller_error')
            command,info=engine.step(state,scene,t,dt,base_z=z,
                base_rotation=s['base_rotation'],commanded_q=state[3:],
                measured_control=s['measured_control'],measured_base_twist=s.get('measured_base_twist'))
            self.control_retries=0
            self.control_latency_history=(self.control_latency_history+[time.monotonic()-started])[-20:]
            if self.execution_observer is not None:
                info['execution_capability']=engine.execution_capability
            info['measured_observation_interval_s']=elapsed
            info['measured_control_interval_s']=dt
            return command,info
        except RuntimeError as error:
            diagnostics=getattr(error,'diagnostics',{})
            deadline=engine.wbc.limits.deadline_s
            if deadline is not None and deadline-.4<=t<=deadline+.4+1e-8:
                # A late physical velocity cannot become zero instantly at a
                # cinematic deadline, including its terminal observation hold.
                # Request a physically feasible brake with
                # identical actuator/collision bounds. The shot deadline and
                # terminal-frame acceptance remain unchanged.
                from dataclasses import replace
                from .safety import solve_with_certificate
                from .planning import joint_margin
                original_limits=engine.wbc.limits
                try:
                    engine.wbc.limits=replace(original_limits,deadline_s=None)
                    target,rotation,_=engine.active.sample(t)
                    command,info=solve_with_certificate(engine.wbc,state,target,rotation,np.zeros(6),dt,
                        base_z=z,base_rotation=s['base_rotation'],commanded_q=state[3:],trajectory_time=t,brake=True)
                except RuntimeError:
                    pass
                else:
                    engine._write('physical_deadline_braking',t,original_deadline_s=deadline,
                        task_deadline_extended=False,physical_limits_relaxed=False)
                    info.update(physical_deadline_braking=True,target_position=target.tolist(),
                        joint_margin_rad=joint_margin(state),measured_control_interval_s=dt,loop_wall_s=time.monotonic()-started)
                    self.control_retries=0
                    return command,info
                finally:
                    engine.wbc.limits=original_limits
            # Keep the physical clock moving; a later snapshot may resolve an
            # empty local numerical solve. Never suppress contact/sensor faults.
            self.control_retries+=1
            from .execution_prediction import failure_evidence
            engine._write('controller_retry',t,reason=str(error),attempt=self.control_retries,
                failure=failure_evidence(str(error),t,source='measured_state_control_check',measured=True),
                diagnostics=diagnostics)
            if self.control_retries>self.policy.max_retries:raise
            from .planning import joint_margin
            from .whole_body_qp import whole_body_kinematics
            target=engine.active.sample(t)[0]
            return np.zeros(8),dict(target_position=engine.active.sample(t)[0].tolist(),
                position_error=float(np.linalg.norm(whole_body_kinematics(state,z)[0]-target)),
                joint_margin_rad=joint_margin(state),qp_violation=0.,
                loop_wall_s=0.,controller_retry=str(error),controller_timing_missing=True)

    def exchange(self,command,info,stage,index,local_t,reference_p,reference_r):
        s=self.snapshot
        info=dict(info,session_stage=stage,phase_index=index,phase_time_s=float(local_t),phase_start_s=s['t']-local_t,
            session_reference_position=np.asarray(reference_p).tolist(),
            session_reference_rotation=np.asarray(reference_r).tolist())
        info['worker_snapshot_compute_wall_s']=time.monotonic()-getattr(self,'snapshot_received_wall',time.monotonic())
        info['worker_command_age_wall_s']=time.monotonic()-s['observation_wall']
        if self.engines:
            self.engines[min(index,len(self.engines)-1)]._write('physical_command_ready',local_t,
                source_time_s=s['t'],stage=stage,sequence=s['sequence'],
                worker_compute_wall_s=info['worker_snapshot_compute_wall_s'],
                command_age_wall_s=info['worker_command_age_wall_s'],
                control_compute_wall_s=info.get('loop_wall_s'))
        info.setdefault('loop_wall_s',info.get('iteration_compute_wall_s',0.))
        info.setdefault('target_position',np.asarray(reference_p).tolist())
        self.last_info=info
        packet=dict(t=s['t'],observation_wall=s['observation_wall'],sequence=s['sequence'],
            sensor_stamp=float(s['scene']['observation_stamp']),
            command=np.asarray(command),info=info,active=None,feedback=dict(source='measured_physical_camera',time_s=s['t']),summary=self.summary())
        accelerate=(self.command_mode=='acceleration' or
            self.command_mode=='adaptive' and
            (stage=='main_action' or info.get('body_anticipation',{}).get('active',False)))
        if accelerate and not info.get('controller_retry') and not info.get('waiting_for_physical_certificate'):
            packet['control_acceleration']=(np.asarray(command)-s['measured_control'])/info.get('measured_control_interval_s',.02)
        offer(self.outbox,pickle.dumps(packet,protocol=pickle.HIGHEST_PROTOCOL))
        self.receive(s['t'])
        state,z,previous,scene=self.initial()
        return state,z,previous,scene,self.snapshot['t']

    def finish(self,result):
        if not self.armed:
            raise RuntimeError('session preparation failed: '+result.get('failure','unknown'))
        self.completed=True
        # Stay alive until the actuator consumes completion. The transport
        # process exiting must not be mistaken for a successful task.
        while not self.stop.is_set():
            s=self.snapshot
            info=dict(self.last_info,session_done=True,session_result=result,loop_wall_s=0.)
            offer(self.outbox,pickle.dumps(dict(t=s['t'],observation_wall=s['observation_wall'],
                sensor_stamp=float(s['scene']['observation_stamp']),
                sequence=s['sequence'],command=np.zeros(8),info=info,active=None,
                feedback={},summary=self.summary()),protocol=pickle.HIGHEST_PROTOCOL))
            try:self.receive(s['t'])
            except RuntimeError:return


def controller_process(inbox,outbox,stop,original,scene,output,options,physical_limits):
    os.setsid()
    from physx_compat import install_worker_adapter
    install_worker_adapter()
    execution=MeasuredExecution(inbox,outbox,stop,options['initial_snapshot'],physical_limits,options['runtime_policy'],options['session_command_mode'])
    from .runtime_fault_injection import FaultSchedule
    execution.fault_schedule=FaultSchedule(options.get('fault_events',()))
    execution.delay_aware_servo=options.get('delay_aware_servo',False)
    if options.get('execution_aware',False):
        from .execution_prediction import ExecutionModel, ExecutionObserver
        profile={}
        if options.get('execution_calibration'):
            data=json.loads(Path(options['execution_calibration']).read_text())
            profile=data.get('model',data)
        profile.update(command_mode=options['session_command_mode'],delay_aware=True)
        execution.execution_observer=ExecutionObserver(ExecutionModel(**profile))
    if options.get('profile_controller'):
        import cProfile
        execution.profile=cProfile.Profile()
    case=Path(options['session_case'])
    try:
        if options['session_combo']:
            from .continuous_combo import run
            result=run(case,case.parent/'video_combo',output/'session',execution=execution)
        else:
            from .runner import run_case
            result=run_case(case,output/'session',mode='jev',api_url=options['api_url'],
                model=options['model'],budget=options['budget'],preplan=options.get('startup_preplan',True),execution=execution)
        execution.finish(result)
    except BaseException as error:
        import traceback
        (output/'session_error.json').write_text(json.dumps(dict(error=str(error),traceback=traceback.format_exc())))
        offer(outbox,pickle.dumps(dict(error=f'{type(error).__name__}: {error}')))
    finally:
        if execution.profile is not None:
            execution.profile.disable();execution.profile.dump_stats(str(output/'controller_profile.pstats'))
        if execution.connector_pool is not None:
            execution.connector_pool.shutdown(wait=False,cancel_futures=True)
