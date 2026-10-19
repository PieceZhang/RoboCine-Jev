"""Jev-directed configuration transfers with free camera motion and no deadline.

A transfer never installs a camera shot or changes its acceptance thresholds.
The same actuator, whole-body/self-collision and swept-command checks apply.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .jev import SystemOneClient,choice
from .planning import ExecutionWBC
from .whole_body_qp import Limits,whole_body_kinematics
from .jvs_kinematics import JOINT_LOWER,JOINT_UPPER
from .evaluation import save_json
from .kc_svsdf import advance


def wrap(angle):return float(np.arctan2(np.sin(angle),np.cos(angle)))


def route_waypoints(start,goal,route):
    start=np.asarray(start[:2]);end=np.asarray(goal[:2]);delta=end-start
    side=np.array([-delta[1],delta[0]])/max(np.linalg.norm(delta),1e-9)
    return ([.5*(start+end)+(.25 if route=='left' else -.25)*side]
        if route!='direct' and np.linalg.norm(delta)>.05 else [])+[end.copy()]


def configuration_control(state,goal,waypoints,drive_sign=None,navigation=None):
    while len(waypoints)>1 and np.linalg.norm(waypoints[0]-state[:2])<.06:waypoints.pop(0)
    delta=waypoints[0]-state[:2];distance=float(np.linalg.norm(delta))
    if navigation is not None:
        if navigation.get('align',False) and distance>.022:
            from .transfer_geometry import approach_sign
            navigation['align']=False;navigation['sign']=approach_sign(state,goal,waypoints[0])
        if len(waypoints)==1 and distance<.010:navigation['align']=True
        drive_sign=navigation.get('sign',drive_sign)
    approach=distance>.010 if navigation is None else not navigation.get('align',False)
    if approach:
        angle=wrap(np.arctan2(delta[1],delta[0])-state[2]);sign=1. if drive_sign is None else drive_sign
        if drive_sign is None and abs(angle)>np.pi/2:sign=-1.
        if sign<0:angle=wrap(angle+np.pi)
        drive=sign*min(.23,1.5*distance)*max(0.,np.cos(angle))**3
        turn=float(np.clip(2.*angle,-.75,.75))
    else:
        direction=np.array([np.cos(state[2]),np.sin(state[2])])
        drive=float(np.clip(1.5*(delta@direction),-.04,.04))
        turn=float(np.clip(2.*wrap(goal[2]-state[2]),-.75,.75))
    return np.r_[drive,turn,np.clip(2.5*(goal[3:]-state[3:]),-.65,.65)]


def route_cards(state,goal,scene,z=0.):
    """Compare configuration routes against current observed occupancy only.

    These sampled, frozen-map probes describe route geometry; they cannot
    certify future obstacle motion or replace the execution safety filter.
    """
    from .dynamic_collision import DynamicCollision
    collision=DynamicCollision(scene);cards={};limits=Limits()
    for route in ('direct','left','right'):
        x=np.asarray(state).copy();old=np.zeros(8);waypoints=route_waypoints(x,goal,route)
        from .transfer_geometry import approach_sign
        sign=approach_sign(x,goal,waypoints[0]);last_waypoint=waypoints[0].copy();navigation=dict(sign=sign,align=False)
        states=[x.copy()];distance=0.;rotation=0.
        for _ in range(220):
            if not np.array_equal(last_waypoint,waypoints[0]):
                sign=approach_sign(x,goal,waypoints[0]);last_waypoint=waypoints[0].copy();navigation=dict(sign=sign,align=False)
            target=configuration_control(x,goal,waypoints,sign,navigation)
            u=np.clip(target,old-.1*limits.acceleration,old+.1*limits.acceleration)
            u[2:]=np.clip(u[2:],(JOINT_LOWER+.01-x[3:])/.1,(JOINT_UPPER-.01-x[3:])/.1)
            x=advance(x,u,.1);old=u;states.append(x.copy())
            distance+=abs(u[0])*.1;rotation+=abs(u[1])*.1
            if np.linalg.norm(x[:2]-goal[:2])<.012 and abs(wrap(x[2]-goal[2]))<.03 and np.max(abs(x[3:]-goal[3:]))<.03:break
        values=collision.geometry.distances_sequence(np.array(states),collision.centers[None],z)-collision.radii[None,:,None]
        # A short lease uses only measured object velocities. Long route
        # geometry remains explicitly frozen rather than extrapolated forever.
        count=min(len(states),26);times=np.arange(count)*.1
        future_centers=collision.centers[None]+times[:,None,None]*collision.velocities[None]
        future_radii=collision.radii[None]+times[:,None]*collision.radius_rates[None]
        dynamic_values=collision.geometry.distances_sequence(np.array(states[:count]),future_centers,z)-future_radii[:,:,None]
        cards[route]=dict(platform_travel_m=float(distance),platform_rotation_rad=float(rotation),
            frozen_observed_map_sampled_clearance_m=float(values.min()),
            observed_motion_lease_clearance_m=float(dynamic_values.min()),motion_lease_s=float(times[-1]),
            scope='sampled configuration geometry in current frozen observed occupancy; NOT a future-motion or safety certificate')
    return cards


def nondominated_routes(cards):
    """Saturate a geometric comfort reserve; reject gratuitous body loops.

    This selects proposals, not executable commands. A larger clearance in a
    frozen snapshot cannot prove that a longer detour will avoid a future mover.
    """
    reserve={key:min(.10,card['frozen_observed_map_sampled_clearance_m'],card.get('observed_motion_lease_clearance_m',np.inf)) for key,card in cards.items()}
    exposure={key:card['platform_travel_m']+.05*card['platform_rotation_rad'] for key,card in cards.items()}
    # Safety evidence precedes travel preference. When every probe is at
    # risk, retain only the least exposed proposals; none becomes certified.
    best=max(reserve.values());minimum=.05 if best>=.05 else best-.02
    eligible=[key for key in cards if reserve[key]>=minimum]
    return [key for key in eligible if not any(other!=key and reserve[other]>=reserve[key]-1e-6
        and exposure[other]<exposure[key]-.02 for other in eligible)]


class FreeTransferAgent:
    def __init__(self,state,goal,scene,choose,base_z=0.,journal=None,initial_control=None,startup_async=False,initial_time=0.):
        self.goal=np.asarray(goal,float).copy();self.z=base_z;self.choose=choose;self.journal=journal
        if self.goal.shape!=(9,) or not np.isfinite(self.goal).all() or np.any(self.goal[3:]<=JOINT_LOWER) or np.any(self.goal[3:]>=JOINT_UPPER):
            raise ValueError('transfer goal must be a finite interior whole-body configuration')
        self.wbc=ExecutionWBC(Limits(),scene=scene)
        from .free_motion import FreeMotionSearch
        self.motion_preview=FreeMotionSearch()
        if initial_control is not None:
            initial_control=np.asarray(initial_control,float)
            if initial_control.shape!=(8,) or not np.isfinite(initial_control).all():raise ValueError('invalid initial measured control')
            self.wbc.previous=initial_control.copy()
        self.pool=ThreadPoolExecutor(max_workers=1);self.pending=None;self.calls=0
        self.last_improvement=initial_time;self.best_error=np.inf;self.last_request=-np.inf;self.last_risk_probe=-np.inf
        self.last_shortcut=-np.inf;self.last_branch_change=initial_time
        if startup_async:
            self.install_numerical_route(state,scene,initial_time)
            self.pending=self.pool.submit(self._ask,np.asarray(state).copy(),scene,initial_time)
        else:
            answer=self._ask(state,scene,initial_time);self._install(answer['route'],state,initial_time)

    def install_numerical_route(self,state,scene,t):
        cards=route_cards(state,self.goal,scene,self.z);eligible=nondominated_routes(cards)
        selected=min(eligible,key=lambda k:cards[k]['platform_travel_m']+.05*cards[k]['platform_rotation_rad'])
        self._install(selected,state,t)
        if self.journal:self.journal(dict(event='transfer_numerical_preference',time_s=t,route=selected,
            scope='bounded observed-geometry initialization while official Jev runs asynchronously; not safety approval'))

    def _ask(self,state,scene,t):
        self.calls+=1
        goal=self.goal.copy()
        cards=route_cards(state,goal,scene,self.z);eligible=nondominated_routes(cards)
        evidence=dict(stage='free_configuration_transfer',time_s=t,state=np.asarray(state).tolist(),
            goal_state=goal.tolist(),observed_objects=scene.get('object_observations',[]),
            observed_tracks=scene.get('tracks',[]),observation_stamp=scene['observation_stamp'],
            route_geometry=cards,eligible_routes=eligible,
            route_filter='First prefer sampled clearance >= 0.05 m if available, otherwise within 0.02 m of the best probe. Then Pareto screen on clearance capped at 0.10 m and platform travel + 0.05 * absolute turn. These are proposals, never safety certificates; physical checks remain unchanged.',
            contract='Reach the next shot initial configuration safely. Camera composition, curve shape and elapsed transfer time are not objectives. Never bypass actuator or collision checks. The choice is a navigation preference, not a safety certificate.')
        options=dict(direct='Approach the goal directly.',left='Use a 0.25 m left-side intermediate platform waypoint.',right='Use a 0.25 m right-side intermediate platform waypoint.')
        answer=self.choose(evidence,{'route':choice('Choose a platform approach side from current observations. Prefer direct when its observed configuration corridor has adequate clearance; unnecessary lateral loops add body exposure and steering reversals. Use a side route when the measured geometry or moving tracks justify it. All routes share the same next-shot configuration goal; sampled frozen-map clearance is not a safety certificate.',
            {key:options[key] for key in eligible})})['route']['choice']
        if answer not in eligible:raise ValueError('API selected a screened-out route')
        return dict(route=answer,goal=goal,time_s=t)

    def accept_route(self,proposal,state,scene,t):
        difference=proposal['goal']-self.goal;difference[2]=wrap(difference[2])
        if np.max(abs(difference))>.01 or t-proposal['time_s']>1.5:
            if self.journal:self.journal(dict(event='transfer_route_expired',time_s=t,reason='goal_changed_or_observation_lease_expired'))
            return False
        answer=proposal['route']
        if answer!=self.route:
            cards=route_cards(state,self.goal,scene,self.z)
            if answer not in nondominated_routes(cards):return False
            def clearance(k):return min(cards[k]['frozen_observed_map_sampled_clearance_m'],cards[k]['observed_motion_lease_clearance_m'])
            # Changing side repeatedly recreates long complementary arcs.
            # Commit while making progress unless new measured evidence is
            # materially better or the old corridor has become blocked.
            improvement=clearance(answer)-clearance(self.route)
            if t-self.last_improvement<2. and improvement<.04 and clearance(self.route)>.025:return False
        self._install(answer,state,t);return True

    def _install(self,answer,state,t):
        if answer not in ('direct','left','right'):raise ValueError('invalid transfer route')
        if getattr(self,'route',None)==answer and getattr(self,'waypoints',None):
            if self.journal:self.journal(dict(event='transfer_route_retained',time_s=t,route=answer))
            return
        self.waypoints=route_waypoints(state,self.goal,answer)
        from .transfer_geometry import approach_sign
        self.drive_sign=approach_sign(state,self.goal,self.waypoints[0]);self.last_waypoint=self.waypoints[0].copy()
        self.navigation=dict(sign=self.drive_sign,align=False)
        self.route=answer;self.best_error=np.inf;self.last_improvement=t
        if self.journal:self.journal(dict(event='transfer_route_selected',time_s=t,route=answer,waypoints=[p.tolist() for p in self.waypoints]))

    def errors(self,state):
        p,r,_=whole_body_kinematics(state,self.z);gp,gr,_=whole_body_kinematics(self.goal,self.z)
        return dict(base_position_m=float(np.linalg.norm(np.asarray(state[:2])-self.goal[:2])),
            base_yaw_rad=abs(wrap(state[2]-self.goal[2])),joint_max_rad=float(np.max(abs(np.asarray(state[3:])-self.goal[3:]))),
            camera_position_m=float(np.linalg.norm(p-gp)),camera_orientation_rad=float(Rotation.from_matrix(gr@r.T).magnitude()))

    def ready(self,state,measured_control=None):
        e=self.errors(state)
        # A redundant body pose need not equal the first IK solution. The
        # caller must still verify the complete next action from this state.
        measured=self.wbc.previous if measured_control is None else np.asarray(measured_control)
        # solve() has already replaced wbc.previous with the NEXT command.
        # A newly computed stop is not evidence that the current body stopped.
        return (e['camera_position_m']<.005 and e['camera_orientation_rad']<.01
            and np.max(abs(measured))<.001 and np.max(abs(self.wbc.previous))<.001)

    def step(self,state,scene,t,dt=.02):
        if self.pending is not None and self.pending.done():
            try:self.accept_route(self.pending.result(),state,scene,t)
            except Exception as error:
                if self.journal:self.journal(dict(event='transfer_replan_failed',time_s=t,error=str(error)))
            self.pending=None
        if not np.array_equal(self.last_waypoint,self.waypoints[0]):
            from .transfer_geometry import approach_sign
            self.drive_sign=approach_sign(state,self.goal,self.waypoints[0]);self.last_waypoint=self.waypoints[0].copy()
            self.navigation=dict(sign=self.drive_sign,align=False)
        if self.route!='direct' and t-self.last_shortcut>.5:
            self.last_shortcut=t
            remaining=np.vstack([state[:2],*self.waypoints]);old_length=float(np.linalg.norm(np.diff(remaining,axis=0),axis=1).sum())
            direct_distance=float(np.linalg.norm(self.goal[:2]-state[:2]))
            if old_length>1.25*direct_distance+.05:
                cards=route_cards(state,self.goal,scene,self.z);card=cards['direct']
                if min(card['frozen_observed_map_sampled_clearance_m'],card['observed_motion_lease_clearance_m'])>=.10:
                    self._install('direct',state,t)
                    if self.journal:self.journal(dict(event='transfer_obsolete_detour_removed',time_s=t,
                        old_remaining_length_m=old_length,direct_distance_m=direct_distance))
        if not self.navigation.get('align') and t-self.last_branch_change>1.:
            direction=np.arctan2(self.waypoints[0][1]-state[1],self.waypoints[0][0]-state[0])
            def turn_cost(sign):
                heading=direction+(np.pi if sign<0 else 0.)
                return abs(wrap(heading-state[2]))+abs(wrap(self.goal[2]-heading))
            sign=self.navigation['sign']
            if turn_cost(-sign)+.35<turn_cost(sign):
                self.drive_sign=-sign;self.navigation['sign']=-sign;self.last_branch_change=t
                if self.journal:self.journal(dict(event='transfer_shorter_drive_branch',time_s=t,drive_sign=-sign))
        control=configuration_control(state,self.goal,self.waypoints,self.drive_sign,self.navigation)
        # Finish the camera boundary precisely using redundant arm motion.
        # Millimetric lateral base error must not trigger another car-like loop.
        if (np.linalg.norm(state[:2]-self.goal[:2])<.15 and abs(wrap(state[2]-self.goal[2]))<.8
                and np.max(abs(state[3:]-self.goal[3:]))<.8):
            from scipy.optimize import lsq_linear
            p,r,J=whole_body_kinematics(state,self.z);gp,gr,_=whole_body_kinematics(self.goal,self.z)
            desired=3.*np.r_[gp-p,Rotation.from_matrix(gr@r.T).as_rotvec()]
            low=np.r_[-.05,-.15,np.maximum(-.65,3.*(JOINT_LOWER+.015-state[3:]))]
            high=np.r_[.05,.15,np.minimum(.65,3.*(JOINT_UPPER-.015-state[3:]))]
            # Solve inside joint bounds rather than clipping a pseudoinverse:
            # clipping can pin one shoulder at its stop while the wrist has
            # enough redundant rotation to finish the same camera pose.
            control=lsq_linear(np.vstack([J,.002*np.eye(8)]),
                np.r_[desired,.002*control],
                bounds=(low,high),tol=1e-8,max_iter=50).x
        # A settled transfer ends at its measured camera pose. Avoid chasing
        # submillimetric IK residuals into the first tangent of the next shot.
        # This is only a preference: preview and physical barriers can still
        # move the robot when an observed obstacle approaches.
        boundary=self.errors(state)
        if (boundary['camera_position_m']<.003 and boundary['camera_orientation_rad']<.006
                and np.max(abs(self.wbc.previous))<.01):
            control=np.zeros(8)
        goal_distance=float(np.linalg.norm(self.goal[:2]-state[:2]))
        error=goal_distance+.1*abs(wrap(self.goal[2]-state[2]))+.1*np.linalg.norm(state[3:]-self.goal[3:])
        if error<self.best_error-.01:self.best_error=error;self.last_improvement=t
        risk=False
        if self.pending is None and t-self.last_risk_probe>.3 and t-self.last_request>1.:
            self.last_risk_probe=t
            if any(np.linalg.norm(row['velocity'])>.04 for row in scene.get('tracks',[])):
                cards=route_cards(state,self.goal,scene,self.z)
                current=cards[self.route]['observed_motion_lease_clearance_m']
                risk=current<.10
        if self.pending is None and (risk or (t-self.last_improvement>3. and t-self.last_request>4.)) and not self.ready(state):
            self.pending=self.pool.submit(self._ask,np.asarray(state).copy(),scene,t);self.last_request=t
            if self.journal:self.journal(dict(event='transfer_replan_requested',time_s=t,reason='observed_motion_corridor_risk' if risk else 'progress_stalled'))
        self.wbc.update_collision_map(scene);self.wbc.set_time(t)
        control=self.motion_preview.preference(self.wbc,state,self.goal,control,t,self.z)
        self.wbc.free_configuration_preview=bool(self.motion_preview.info.get('active'))
        p,r,_=whole_body_kinematics(state,self.z)
        command,info=self.wbc.solve(state,p,r,np.zeros(6),dt,base_z=self.z,trajectory_time=t,free_transfer_control=control)
        info.update(transfer_route=self.route,transfer_goal_error=self.errors(state),transfer_replan_pending=self.pending is not None)
        info['free_configuration_preview']=self.motion_preview.info
        return command,info

    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True);self.wbc.close()


def run_case(case,output,model='jev-1.14.0',api_url='https://api.typesafe.ai/v1/systemone',watchdog_s=45.):
    from .scenarios import RuntimeScene
    from .sensors import RuntimeLidar
    from .reference import OriginalReference
    from .kinematic_feasibility import KinematicSearch,summary as kinematic_summary
    from .provenance import source_manifest
    case=Path(case);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    task=json.loads((case/'task.json').read_text());scene=RuntimeScene(case/'runtime_scene.json');runtime=scene.document
    state=np.asarray(runtime['initial_state'],float);z=runtime['base_z'];lidar=RuntimeLidar()
    history=runtime.get('initial_observation_history') or [dict(t=float(stamp),state=state.tolist()) for stamp in np.linspace(-.8,0.,9)]
    for sample in history:observed=lidar.synthetic_scan(scene,np.asarray(sample['state']),z,sample['t'])
    for name,doc in [('source_task.json',task),('runtime_scene.json',runtime),('next_task.json',json.loads((case/'next_task.json').read_text())),
        ('original_task.json',dict(shot_phase=[dict(name=task['name'],motion='free_transfer',start_s=0.,end_s=watchdog_s)],
            deadline_s=None,test_watchdog_s=watchdog_s))]:save_json(output/name,doc)
    save_json(output/'provenance.json',source_manifest(Path(__file__).resolve().parents[1],dict(mode='free_transfer',pace=1,model=model,api_url=api_url)))
    events=(output/'events.jsonl').open('w')
    def journal(value):events.write(json.dumps(value)+'\n');events.flush()
    client=SystemOneClient(url=api_url,model=model,timeout_s=15.,journal=output/'jev_api.jsonl',
        policy='Choose only a bounded approach side for a free-camera robot configuration transfer using current observations. All numerical safety checks are immutable.')
    rows=[];failure='';ready_since=None;reached=False;agent=None;begin=time.monotonic()
    try:
        from .transfer_geometry import nearby_goal
        next_ref=OriginalReference(json.loads((case/'next_task.json').read_text()),runtime['subject']['position'])
        goal,goal_report=nearby_goal(state,task['goal_state'],next_ref,observed,z)
        save_json(output/'goal_selection.json',goal_report)
        agent=FreeTransferAgent(state,goal,observed,client.ask,z,journal,runtime.get('initial_control'))
        begin=time.monotonic()
        for i in range(int(watchdog_s/.02)+1):
            t=i*.02;started=time.monotonic()
            observed=lidar.synthetic_scan(scene,state,z,t)
            measured=agent.wbc.previous.copy()
            command,info=agent.step(state,observed,t)
            info['iteration_compute_wall_s']=time.monotonic()-started
            p,r,_=whole_body_kinematics(state,z)
            rows.append(dict(t=t,state=state.tolist(),command=command.tolist(),camera_position=p.tolist(),
                camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),info=info))
            ready=agent.ready(state,measured)
            ready_since=t if ready and ready_since is None else ready_since if ready else None
            if ready_since is not None and t-ready_since>=.4:reached=True;break
            state=advance(state,command,.02)
            wait=begin+(i+1)*.02-time.monotonic()
            if wait>0:time.sleep(wait)
        if not reached:failure='goal_not_reached_within_test_watchdog; no shot deadline applied'
    except Exception as error:failure=f'{type(error).__name__}: {error}'
    finally:
        client.closed=True
        if agent is not None:agent.close()
        events.close()
    next_ready=None
    if reached:
        ref=OriginalReference(json.loads((case/'next_task.json').read_text()),runtime['subject']['position'])
        next_ready=KinematicSearch(state,agent.wbc.previous,0.,z,Limits(),measured_control=agent.wbc.previous).lift(ref,ref.duration)
    accepted=bool(reached and next_ready and next_ready['feasible'])
    states=np.array([r['state'] for r in rows]);positions=np.array([r['camera_position'] for r in rows])
    result=dict(mode='jev_free_transfer',completed=reached,failure=failure,elapsed_shot_s=rows[-1]['t'] if rows else 0.,
        elapsed_transfer_s=rows[-1]['t'] if rows else 0.,wall_s=time.monotonic()-begin,
        installed_plans=0,safety_pass=bool(rows and not failure),goal_error=None if agent is None else agent.errors(state),
        path_metrics=dict(platform_length_m=float(np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1).sum()),
            platform_total_turn_rad=float(abs(np.diff(states[:,2])).sum()),camera_length_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())) if len(rows)>1 else None,
        next_shot_kinematics=None if next_ready is None else kinematic_summary(next_ready),
        transfer_acceptance=dict(accepted=accepted,goal_reached=reached,next_shot_geometric_start_verified=bool(next_ready and next_ready['feasible']),
            failed_checks=([] if accepted else ['goal_not_reached'] if not reached else ['next_start_kinematics']),
            camera_metrics_required=False,deadline_s=None,test_watchdog_s=watchdog_s,physical_audit='pending independent audit'),
        scope=task['scope'])
    save_json(output/'telemetry.json',rows);save_json(output/'summary.json',result)
    print(json.dumps(dict(case=task['name'],completed=reached,accepted=accepted,failure=failure)),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--scenarios',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='jev-1.14.0');p.add_argument('--api-url',default='https://api.typesafe.ai/v1/systemone')
    p.add_argument('--watchdog',type=float,default=45.);a=p.parse_args()
    r=run_case(a.scenarios/a.case,a.output/a.case/(a.case+'_jev'),a.model,a.api_url,a.watchdog)
    return 0 if r['transfer_acceptance']['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
