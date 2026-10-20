"""Continuous CPU combo: main shots and unconstrained-camera safe connectors.

One state, actuator history, sensor map, and obstacle clock for the entire run.
Main-shot programs are selected before the physical clock starts, using only
initial observations. Connector API calls are asynchronous during execution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .scenarios import RuntimeScene
from .sensors import RuntimeLidar
from .reference import OriginalReference
from .whole_body_qp import Limits,whole_body_kinematics
from .kc_svsdf import advance
from .dynamic_collision import DynamicCollision
from .engine import CinematicEngine
from .candidates import SearchBudget
from .phase_program import prepare
from .free_transfer import FreeTransferAgent
from .transfer_geometry import nearby_goal,fresh_reference,capture_reference,prepare_arrival
from .kinematic_feasibility import KinematicSearch,summary as kinematic_summary
from .shot_acceptance import evaluate_shot
from .camera_intent import POLICY
from .jev import SystemOneClient
from .evaluation import save_json


def action_task(task,phase):
    a,b=phase['start_s'],phase['end_s'];value=deepcopy(task)
    value['shot_phase']=[dict(phase,start_s=0.,end_s=b-a)]
    value['reference_camera_trajectory']['samples']=[dict(row,t=round(row['t']-a,8))
        for row in task['reference_camera_trajectory']['samples'] if a<=row['t']<=b]
    return value


def local_scene(observation,offset):
    return dict(observation,observation_stamp=observation['observation_stamp']-offset)


def run(case,nominal_root,output,watchdog=35.,model='jev-1.14.0',execution=None):
    case=Path(case);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    task=json.loads((case/'task.json').read_text());runtime=RuntimeScene(case/'runtime_scene.json');scene=runtime.document
    source=scene.get('source_combo',case.name)
    nominal=json.loads((Path(nominal_root)/(source+'_nominal.json')).read_text())
    state=np.array(scene['initial_state'],float);previous=np.array(scene.get('initial_control',np.zeros(8)),float);z=scene['base_z']
    lidar=RuntimeLidar()
    for stamp in np.linspace(-.8,0.,9):observed=lidar.synthetic_scan(runtime,state,z,float(stamp))
    if execution is not None:
        state,z,previous,observed=execution.initial()
    save_json(output/'source_task.json',task);save_json(output/'runtime_scene.json',scene)
    intrinsics=[.5/np.tan(np.deg2rad(27.5))*288/384,.5/np.tan(np.deg2rad(27.5)),.5,.5]
    events=(output/'events.jsonl').open('w')
    def journal(value):events.write(json.dumps(value)+'\n');events.flush()
    client=SystemOneClient(url='https://api.typesafe.ai/v1/systemone',model=model,timeout_s=15.,journal=output/'jev_api.jsonl',
        policy='Choose a short safe whole-body connector using current observations. The camera is free during connectors. Never alter physical safety limits.')
    engines=[];programs=[];goals=[];phases=[];transfers=[];rows=[];failure='';completed=False;clock=0.;begin=time.monotonic();free=None
    if execution is None:
        pool=ThreadPoolExecutor(max_workers=1)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from .shot_preview_worker import initialize_worker
        pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker)
        pool.submit(initialize_worker).result(timeout=30.)
    prepared_goals={}
    refresh_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='arrival-refresh')
    try:
        # No simulated obstacle motion has begun yet. This is task preparation,
        # not a pause between shots. No future scene/keyframe data enters Jev.
        for index,phase in enumerate(task['shot_phase']):
            original=OriginalReference(action_task(task,phase),scene['subject']['position'])
            goal=np.asarray(nominal['actions'][index]['states'][0],float)
            initial_boundary=None
            if index==0:
                measured_p,measured_r,_=whole_body_kinematics(state,z);original_p,original_r,_=original.sample(0.)
                if np.linalg.norm(measured_p-original_p)>.001 or Rotation.from_matrix(measured_r@original_r.T).magnitude()>.003:
                    initial_boundary=state.copy();goal=state.copy()
                    goal_record=dict(selected='measured_initial_boundary',state=state.tolist(),
                        scope='smooth initial-pose residual over original action; original direction, terminal frame, duration and strict checks unchanged')
                else:goal,goal_record=nearby_goal(state,goal,original,observed,z)
                save_json(output/'initial_goal.json',goal_record)
            engine=CinematicEngine(original,local_scene(observed,0.),output/f'phase_{index+1:02d}',
                mode='jev',api_url='https://api.typesafe.ai/v1/systemone',model=model,subject_model=scene['subject'],intrinsics=intrinsics,
                coordination_async=execution is not None,
                budget=SearchBudget(max_proposals=3,max_rollouts=2,candidate_wall_budget_s=6.,activation_lead_s=.8))
            engines.append(engine);goals.append(goal)
            programs.append(prepare(engine,local_scene(observed,0.),goal,z,initial_boundary=initial_boundary)
                if execution is None else execution.prepare(prepare,engine,local_scene(observed,0.),goal,z,initial_boundary=initial_boundary))
            save_json(output/f'phase_{index+1:02d}/original_task.json',original.task)
            save_json(output/f'phase_{index+1:02d}/phase_program.json',programs[-1])
            journal(dict(event='main_program_prepared',time_s=0.,phase_index=index,selection=programs[-1]['shapes'],source='official_jev_before_physical_start'))
        if execution is not None:
            for prepared_engine in engines:prepared_engine.warm_workers()
            execution.ready(engines)
            state,z,previous,observed=execution.initial()
        begin=time.monotonic()
        def tick(command,info,stage,index,local_t,reference_p,reference_r):
            nonlocal state,previous,clock,observed,z
            p,r,_=whole_body_kinematics(state,z)
            truth=DynamicCollision(runtime.truth_collision(clock),clock,uncertainty_rate=0.)
            info['target_position']=np.asarray(info.get('target_position',reference_p)).tolist()
            rows.append(dict(t=clock,state=state.tolist(),command=np.asarray(command).tolist(),camera_position=p.tolist(),
                camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),reference_position=np.asarray(reference_p).tolist(),
                reference_quaternion_xyzw=Rotation.from_matrix(reference_r).as_quat().tolist(),stage=stage,phase_index=index,local_time_s=local_t,
                truth_clearance_m=float(truth.clearance(state,z).min()),info=info,runtime_tracks=observed.get('tracks',[]),observation_stamp=observed['observation_stamp']))
            if execution is not None:
                state,z,previous,observed,clock=execution.exchange(command,info,stage,index,local_t,reference_p,reference_r)
                return
            state=advance(state,command,.02);previous=np.asarray(command).copy();clock=round(clock+.02,8)
            wait=begin+clock-time.monotonic()
            if wait>0:time.sleep(wait)
            if round(clock/.02)%5==0:observed=lidar.synthetic_scan(runtime,state,z,clock)
        for index,engine in enumerate(engines):
            # The first phase may also need alignment after a perturbed reset.
            cp,cr,_=whole_body_kinematics(state,z);gp,gr,_=engine.active.sample(0.)
            alignment=index>0 or np.linalg.norm(cp-gp)>.006 or Rotation.from_matrix(gr@cr.T).magnitude()>.015
            if alignment:
                started=clock;start_row=len(rows);ready_since=None;arrived=False
                goal_future=prepared_goals.pop(index,None)
                if goal_future is None:goal_future=pool.submit(nearby_goal,state.copy(),goals[index],engine.active,deepcopy(observed),z)
                if goal_future.done():goals[index]=goal_future.result()[0]
                # Keep executing a safe configuration controller while a nearby
                # goal is solved; never stop the physical clock for planning.
                free=FreeTransferAgent(state,goals[index] if goal_future.done() else state.copy(),observed,client.ask,z,journal,previous,startup_async=True,initial_time=clock)
                goal_installed=False;last_goal_search=clock;goal_revision=0;revising=False;last_arrival_check=-np.inf;arrival_plan=None
                refresh_future=None;refresh_started=None;refresh_finished=False;refreshed_path=None
                while clock-started<watchdog:
                    if goal_future.done() and (not goal_installed or revising):
                        from .transfer_geometry import configuration_distance
                        goal,record=goal_future.result()
                        useful=(not goal_installed or configuration_distance(state,goal)<.75*configuration_distance(state,free.goal))
                        if useful:
                            free.goal=goal;free.route=None;free.install_numerical_route(state,observed,clock)
                            goal_revision+=1
                            body_reference=engine.active
                            if record.get('prepared_geometry') is not None:
                                from .intent_paths import IntentPath
                                body_reference=IntentPath(engine.active.original,engine.active.previous,0.,record['prepared_program'],
                                    engine.active.model,engine.active.intrinsics,geometry=record['prepared_geometry'])
                            arrival_plan=pool.submit(prepare_arrival,body_reference,goal.copy(),z)
                            save_json(output/f'transfer_{index:02d}_goal.json',record)
                            save_json(output/f'transfer_{index:02d}_goal_revision_{goal_revision:02d}.json',record)
                            journal(dict(event='nearby_goal_selected',time_s=clock,phase_index=index,goal=record,revision=goal_revision))
                        elif revising:
                            journal(dict(event='nearby_goal_revision_declined',time_s=clock,phase_index=index,goal=record))
                        goal_installed=True;revising=False
                    if (goal_installed and not revising and ready_since is None and clock-last_goal_search>3. and
                            (np.linalg.norm(state[:2]-free.goal[:2])>.20 or clock-free.last_improvement>2.5)):
                        # Escape may leave the robot beside a different valid
                        # redundant arrival. Re-anchor to that short solution
                        # instead of retracing the old detour to a fixed reset.
                        goal_future=pool.submit(nearby_goal,state.copy(),free.goal.copy(),engine.active,deepcopy(observed),z)
                        revising=True;last_goal_search=clock
                    before=time.monotonic()
                    command,info=(free.step(state,observed,clock) if execution is None else
                                  execution.transfer_step(free,state,observed,clock,previous))
                    ready=goal_installed and free.ready(state,previous)
                    ready_since=clock if ready and ready_since is None else ready_since if ready else None
                    if ready and arrival_plan is not None and arrival_plan.done() and not refresh_finished:
                        from .arrival_refresh import without_detour,propose
                        if refresh_future is None:
                            if without_detour(engine.active) is None:refresh_finished=True
                            else:
                                refresh_started=clock
                                save_json(output/f'phase_{index+1:02d}/arrival_refresh_observation.json',observed)
                                refresh_future=refresh_pool.submit(propose,engine.active,state.copy(),previous.copy(),deepcopy(observed),z,clock,client.ask)
                                journal(dict(event='arrival_detour_refresh_requested',time_s=clock,phase_index=index))
                    if not refresh_finished and refresh_future is not None and refresh_future.done():
                        try:refreshed_path,refresh_record=refresh_future.result()
                        except Exception as error:
                            refresh_record=dict(selected=False,reason='refresh_error',error=str(error))
                        refresh_finished=True
                        save_json(output/f'phase_{index+1:02d}/arrival_refresh.json',refresh_record)
                        journal(dict(event='arrival_detour_refresh_result',time_s=clock,phase_index=index,**refresh_record))
                    if not refresh_finished and refresh_started is not None and clock-refresh_started>6.:
                        # Bounded asynchronous proposal: no physical-clock pause,
                        # and a late result cannot replace the latched fallback.
                        refresh_finished=True
                        journal(dict(event='arrival_detour_refresh_expired',time_s=clock,phase_index=index))
                    if (ready_since is not None and clock-ready_since>=.3 and clock-last_arrival_check>=.3
                            and arrival_plan is not None and arrival_plan.done() and refresh_finished):
                        last_arrival_check=clock
                        from .cinematic_geometry import curve_quality
                        from .floating_camera import camera_motion_limits
                        from .intent_paths import poses_at
                        prepared,body_report=arrival_plan.result()
                        shortening=None
                        if refreshed_path is not None:
                            from .arrival_refresh import screen,path_length
                            shortened=capture_reference(refreshed_path,state,previous,z,keep_seed=True)
                            shorter_proof,current_evidence=screen(shortened,state,previous,observed,z,clock)
                            save_json(output/f'phase_{index+1:02d}/arrival_refresh_activation.json',current_evidence)
                            if shorter_proof is not None:
                                prepared=shortened
                                body_report=dict(kinematics=kinematic_summary(shorter_proof),source='fresh_shorter_path_allocation')
                                shortening=dict(event='obsolete_main_detour_removed',time_s=clock,phase_index=index,
                                    old_length_m=path_length(engine.active),new_length_m=path_length(shortened),
                                    observation_stamp_s=observed['observation_stamp'])
                            else:
                                refreshed_path=None
                                journal(dict(event='arrival_detour_refresh_rejected',time_s=clock,phase_index=index,reason=current_evidence['reason']))
                        if shortening is not None:
                            # Reuse the checks just performed on this exact
                            # measured state/map, not the worker's old proof.
                            candidate=shortened;witness=shorter_proof
                            quality=current_evidence['quality'];motion_limits=current_evidence['camera_limits']
                            lease=current_evidence['observed_motion_lease']
                        else:
                            candidate=capture_reference(prepared,state,previous,z,keep_seed=True)
                            quality=curve_quality(candidate)
                            check_times=np.linspace(0.,candidate.duration,401);pp,rr=poses_at(candidate,check_times)
                            motion_limits=camera_motion_limits(list(zip(pp,rr)),check_times)
                            witness=KinematicSearch(state,previous,0.,z,Limits(),measured_control=previous).lift(candidate,candidate.duration)
                            lease=None
                            if witness['feasible']:
                                lease=KinematicSearch(state,previous,0.,z,Limits(),measured_control=previous,
                                    scene=local_scene(observed,clock)).lift(candidate,min(1.5,candidate.duration))
                        if (witness['feasible'] and lease and lease['feasible'] and quality['direction_pass']
                                and quality['strict_full_curve_pass'] and motion_limits['camera_limits_pass']):
                            # All prediction seeds must describe this actual
                            # arrival, not the independently authored reset.
                            from .intent_paths import IntentPath
                            path=candidate
                            geometry=dict(path.geometry,joint_seed=witness['trace'],joint_seed_preference_only=True)
                            engine.active=IntentPath(path.original,path.previous,path.activation_s,path.program,
                                path.model,path.intrinsics,geometry=geometry)
                            engine.wbc.set_trajectory(engine.active)
                            engine.wbc.kinematic_reference=witness
                            engine.wbc.arrival_reanchored=True
                            if shortening is not None:journal(shortening)
                            save_json(output/f'phase_{index+1:02d}/arrival_kinematic_seed.json',witness)
                            save_json(output/f'phase_{index+1:02d}/arrival_boundary.json',dict(geometry=path.geometry,
                                program=path.program,quality=quality,camera_limits=motion_limits,observed_lease=kinematic_summary(lease),
                                body_allocation=body_report))
                            arrived=True;break
                    info['iteration_compute_wall_s']=time.monotonic()-before
                    tick(command,info,'initial_alignment' if index==0 else 'free_transfer',index,clock-started,gp,gr)
                segment=rows[start_row:];positions=np.array([row['camera_position'] for row in segment]);states=np.array([row['state'] for row in segment])
                record=dict(index=index,start_s=started,end_s=clock,duration_s=clock-started,completed=arrived,
                    camera_metrics_required=False,deadline_s=None,
                    path_metrics=dict(platform_length_m=float(np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1).sum()),
                        platform_total_turn_rad=float(abs(np.diff(states[:,2])).sum()),camera_length_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())) if len(segment)>1 else None,
                    next_shot_kinematics=kinematic_summary(witness) if arrived else None)
                transfers.append(record);journal(dict(event='transfer_completed' if arrived else 'transfer_failed',time_s=clock,**record))
                free.close();free=None
                if not arrived:raise RuntimeError('free_transfer_watchdog_exhausted')
            started=clock;start_row=len(rows);engine.wbc.previous=previous.copy()
            # Re-anchor only the proof, never the state or velocity.
            engine.wbc.update_collision_map(local_scene(observed,started))
            journal(dict(event='main_phase_started',time_s=clock,phase_index=index,motion=task['shot_phase'][index]['motion']))
            local_failure=''
            try:
                count=round((engine.original.duration+POLICY.terminal_hold_s)/.02)+1
                for step in range(count):
                    local_t=step*.02 if execution is None else clock-started
                    if local_t>engine.original.duration+POLICY.terminal_hold_s+1e-8:break
                    before=time.monotonic()
                    if local_t>=engine.original.duration:
                        # Terminal framing holds the CAMERA, not every motor.
                        # Keep redundant body escape available during that
                        # confirmation window under the same physical limits.
                        engine.wbc.limits.deadline_s=None
                        if getattr(engine.wbc,'terminal_camera_pose',None) is None:
                            from .subject_state import frame_state,frame_error
                            from .cinematic_geometry import inside_terminal_band
                            p,r,_=whole_body_kinematics(state,z);tp,tr,_=engine.original.sample(engine.original.duration)
                            actual=frame_state(p,r,scene['subject'],intrinsics);expected=frame_state(tp,tr,scene['subject'],intrinsics)
                            error=frame_error(actual,expected);error['silhouette_axis_required']=expected.get('silhouette_axis_rad') is not None
                            if inside_terminal_band(error):engine.wbc.terminal_camera_pose=(p.copy(),r.copy())
                    if index+1<len(engines) and index+1 not in prepared_goals and local_t>=engine.original.duration-1.2:
                        prepared_goals[index+1]=pool.submit(nearby_goal,state.copy(),goals[index+1],engines[index+1].active,deepcopy(observed),z)
                    if execution is None:
                        command,info=engine.step(state,local_scene(observed,started),local_t,.02,base_z=z,commanded_q=state[3:],measured_control=previous)
                    else:
                        command,info=execution.step(engine,state,local_scene(observed,started),local_t,.02,z)
                    info['iteration_compute_wall_s']=time.monotonic()-before
                    p,r,_=engine.original.sample(local_t)
                    tick(command,info,'main_action',index,local_t,p,r)
            except Exception as error:
                local_failure=f'{type(error).__name__}: {error}'
                raise
            finally:
                history=engine.camera_history
                acceptance=None
                if len(history)>1:
                    acceptance=evaluate_shot(engine.original,scene['subject'],[h['t'] for h in history],
                        [h['camera_position'] for h in history],[Rotation.from_quat(h['camera_quaternion_xyzw']).as_matrix() for h in history],
                        [h['visual'] for h in history],[h['progress_s'] for h in history],safety_pass=not local_failure,
                        execution_ok=not local_failure,intrinsics=intrinsics,motion_end_s=engine.active.duration)
                record=dict(index=index,motion=task['shot_phase'][index]['motion'],start_s=started,end_s=clock,
                    completed=not local_failure,failure=local_failure,shot_acceptance=acceptance,program=programs[index],
                    executed_program=engine.active.program)
                phases.append(record);save_json(output/f'phase_{index+1:02d}/summary.json',record)
                save_json(output/f'phase_{index+1:02d}/camera_history.json',history)
                save_json(output/f'phase_{index+1:02d}/telemetry.json',rows[start_row:])
                journal(dict(event='main_phase_finished',time_s=clock,phase_index=index,accepted=bool(acceptance and acceptance['accepted']),failure=local_failure))
            engine.close()
        completed=True
    except Exception as error:
        failure=f'{type(error).__name__}: {error}';journal(dict(event='run_failed',time_s=clock,failure=failure,
            diagnostics=getattr(error,'diagnostics',None)))
    finally:
        # Record the last integrated state as well as every command interval.
        # This also preserves the final safe prefix if the next solve fails.
        if rows and clock>rows[-1]['t']:
            terminal=dict(rows[-1]);p,r,_=whole_body_kinematics(state,z)
            terminal.update(t=clock,state=state.tolist(),command=previous.tolist(),camera_position=p.tolist(),
                camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
                runtime_tracks=observed.get('tracks',[]),observation_stamp=observed['observation_stamp'],
                info=dict(rows[-1]['info'],terminal_observation=True,command_not_executed=True))
            rows.append(terminal)
        if free is not None and not any(q['index']==index for q in transfers):
            segment=rows[start_row:];q=np.asarray([r['state'] for r in segment]);p=np.asarray([r['camera_position'] for r in segment])
            transfers.append(dict(index=index,start_s=started,end_s=clock,duration_s=clock-started,completed=False,failure=failure,
                camera_metrics_required=False,deadline_s=None,path_metrics=dict(platform_length_m=float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1).sum()),
                    platform_total_turn_rad=float(abs(np.diff(q[:,2])).sum()),camera_length_m=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()))))
        client.closed=True
        if free is not None:free.close()
        pool.shutdown(wait=False,cancel_futures=True)
        refresh_pool.shutdown(wait=False,cancel_futures=True)
        for engine in engines:engine.close()
        events.close()
        accepted=bool(completed and all(p['shot_acceptance'] and p['shot_acceptance']['accepted'] for p in phases))
        failed=[f"phase_{p['index']+1}:{','.join(p['shot_acceptance']['failed_checks'])}" for p in phases if p['shot_acceptance'] and not p['shot_acceptance']['accepted']]
        if failure:failed.append(failure)
        result=dict(mode='jev',semantic_model=model,backend='kinematic_integration',completed=completed,failure=failure,
            elapsed_shot_s=rows[-1]['t'] if rows else 0.,wall_s=time.monotonic()-begin,installed_plans=sum(e.installed_plans for e in engines),
            phases=phases,transfers=transfers,obstacle_avoidance_enabled=True,representation_boxes=3,
            shot_acceptance=dict(accepted=accepted,failed_checks=failed),safety_pass=bool(completed and not failure),
            scope='continuous CPU combo; no state, actuator-history, LiDAR-map or physical-obstacle-clock resets between phases',pace=1)
        if execution is not None:
            result.update(backend='physx_measured_controller',scope='controller stage audit; authoritative measured safety/acceptance in parent')
        save_json(output/'summary.json',result);save_json(output/'telemetry.json',rows)
        save_json(output/'original_task.json',dict(task,shot_phase=[dict(name='continuous_combo',motion='continuous_combo',start_s=0.,end_s=max(clock,.02))]))
        save_json(output/'stage_timeline.json',dict(phases=phases,transfers=transfers))
        print(json.dumps(dict(case=case.name,completed=completed,accepted=accepted,failure=failure,phases=[(p['motion'],bool(p['shot_acceptance'] and p['shot_acceptance']['accepted'])) for p in phases])),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--scenarios',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--watchdog',type=float,default=35.)
    a=p.parse_args();r=run(a.scenarios/a.case,a.scenarios/'video_combo',a.output/a.case/(a.case+'_jev'),a.watchdog)
    return 0 if r['shot_acceptance']['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
