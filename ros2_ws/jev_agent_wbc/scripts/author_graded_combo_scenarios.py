#!/usr/bin/env python3
"""Author six incomplete combo designs and fit independent grounded obstacle lanes.

Run into a new directory, prepare per-action nominal previews with the existing
presentation script, then run --fit-obstacles. This never fills transition gaps
or changes the Agent. Clearance audits are offline geometric design evidence.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from jev_agent_wbc.reference import look_at, smoothstep
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from jev_agent_wbc.kc_svsdf import CollisionGeometry
from record_scenario_trajectories import digest

CASES = [
    ('combo_01_easy', '简单', '推进 → 左移', 0.,
     [('movein',7.,[.35,0.,.025]), ('truck_left',7.,[0.,.40,0.])], [-.09], [.80]),
    ('combo_02_easy', '简单', '右移 → 升高', -1.1,
     [('truck_right',7.,[0.,-.40,0.]), ('pedestal_up',7.,[.18,0.,.10])], [.09], [.85]),
    ('combo_03_medium', '中等', '推进 → 环绕 → 升高', 0.,
     [('movein',7.,[.40,0.,.025]), ('orbit',8.,-.30), ('pedestal_up',7.,[.18,0.,.10])],[-.09,-.08],[.40,.46]),
    ('combo_04_medium', '中等', '左移 → 推进 → 下降', 1.1,
     [('truck_left',7.,[0.,.45,0.]), ('movein',7.,[.30,0.,.025]), ('pedestal_down',7.,[.18,0.,-.10])],[-.09,.08],[.38,.44,.50]),
    ('combo_05_hard', '困难', '环绕 → 推进 → 右移', .9,
     [('orbit',8.,-.40), ('movein',7.,[.30,0.,.025]), ('truck_right',7.,[0.,-.40,0.])],[-.08,.09],[.14,.17,.20,.23]),
    ('combo_06_hard', '困难', '升高 → 左移 → 下降', 0.,
     [('pedestal_up',7.,[.20,0.,.12]), ('truck_left',7.,[0.,.45,0.]), ('pedestal_down',7.,[.20,0.,-.12])],[-.09,.08],[.14,.165,.19,.215,.24]),
]


def write(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def author(root):
    root.mkdir(parents=True, exist_ok=True)
    catalog=dict(version='graded_combo/v2', cases=[c[0] for c in CASES],
        description='六个递增难度的组合场景；净距指障碍物表面至整机碰撞几何表面，而非中心距。',
        main_action_limit_s=10., transition_duration_s=2., transition_endpoint_distance_limit_m=.20,
        reference_sample_interval_s=.2,
        transition_contract=dict(authoring='shot_phase 只有主动作；衔接窗口没有阶段定义和内部轨迹样本。',
            agent_responsibility='由 Agent 自行设计 arc；不指定弧面、圆心、半径、控制点或方向。',
            loader_behavior='当前完整任务校验要求阶段连续。此组合文件是待补全的设计输入；逐段预览可读取，不能直接作为完整闭环任务。'),
        obstacle_design='地面移动道具的球形几何代理，沿独立固定路线平滑经过；不跟随机器人、不穿过腕部，不在空中悬浮。',
        difficulty_definition=dict(easy=dict(obstacle_counts=[1,1],target_nearest_surface_clearance_m=[.80,.85]),
            medium=dict(obstacle_counts=[2,3],target_nearest_surface_clearance_m=[.40,.38]),
            hard=dict(obstacle_counts=[4,5],target_nearest_surface_clearance_m=[.14,.14])),
        validation_scope='名义逐段轨迹的离线整机几何净距与障碍物相互距离；不代表 Agent arc 或 PhysX 执行验证。',
        scenarios=[])
    for number,(name,level,title,yaw,actions,gaps,targets) in enumerate(CASES,1):
        folder=root/name; folder.mkdir(exist_ok=False)
        initial=np.array([0.,0.,yaw,-yaw,-.9,2.1,0.,-1.2,0.])
        p,_,_=whole_body_kinematics(initial,.002);subject=p+[1.55,0.,0.]
        samples=[];phases=[];main=[];transitions=[];clock=0.
        for i,(motion,duration,change) in enumerate(actions):
            start=p.copy();end=clock+duration;phase=f'action_{i+1:02d}_{motion}'
            phases.append(dict(name=phase,motion=motion,start_s=clock,end_s=end))
            for local in np.linspace(0,duration,round(duration/.2)+1):
                f=smoothstep(local/duration)
                p=subject+Rotation.from_euler('z',change*f).apply(start-subject) if motion=='orbit' else start+f*np.array(change)
                samples.append(dict(t=round(clock+local,8),position=p.tolist(),quaternion_xyzw=Rotation.from_matrix(look_at(p,subject)).as_quat().tolist()))
            main.append(dict(phase=phase,motion=motion,start_s=clock,end_s=end,duration_s=duration,start_position_m=start.tolist(),end_position_m=p.tolist()))
            clock=end
            if i<len(gaps):
                next_p=subject+Rotation.from_euler('z',gaps[i]).apply(p-subject)
                transitions.append(dict(id=f'gap_{i+1:02d}',role='unfilled_transition_window',start_s=clock,end_s=clock+2.,
                    from_position_m=p.tolist(),to_position_m=next_p.tolist(),endpoint_gap_m=float(np.linalg.norm(next_p-p)),interior_samples_authored=False))
                p=next_p;clock+=2.
        task=dict(reference_camera_trajectory=dict(frame='world_z_up_camera_link',samples=samples),shot_phase=phases,
            subject=dict(id='blue_subject',description='A small matte blue sculptural cylinder with a yellow top'),
            intent='Perform these main camera actions in order: '+', then '.join(f"{a['motion']} during {a['start_s']:g}-{a['end_s']:g} s" for a in main)+
            '. Keep the subject recognizable and centered with a level horizon. Preserve approach, lateral parallax, directed orbit and vertical revelation as applicable. '+
            'The 2-second gaps between actions are deliberately empty: no phase, motion type, interior pose or placeholder orbit is authored. '+
            'Independently design a smooth arc in each gap from the previous endpoint to the next start using current obstacle observations. Choose the arc geometry and timing yourself within the window. Do not skip or reorder the main actions.')
        scene=dict(name=name,initial_state=initial.tolist(),base_z=.002,pre_roll_motion='constant_initial_velocity',
            subject=dict(id='blue_subject',position=subject.tolist(),orientation_xyzw=[0.,0.,0.,1.],radius=.13,height=.36,source='runtime_subject_initial_calibration_and_rendering'),
            obstacles=[],description=f'{number:02d} {level}：{title}；独立地面道具路线，衔接待 Agent 设计。',
            evaluation=dict(progress_correspondence='original_elapsed_time',capture_fps=8,width=384,height=288,vfov_deg=55.))
        write(folder/'task.json',task);write(folder/'runtime_scene.json',scene)
        catalog['scenarios'].append(dict(number=number,name=name,difficulty=name.rsplit('_',1)[1],difficulty_zh=level,
            title=f'{number:02d} {level} · {title}',main_action_count=len(actions),obstacle_count=len(targets),
            target_obstacle_surface_clearances_m=targets,duration_s=clock,main_actions=main,transitions=transitions))
    write(root/'combined_catalog.json',catalog)


def samples_with_gap_boundaries(cache, step):
    times=[];states=[];kinds=[]
    for i,action in enumerate(cache['actions']):
        ts=np.array(action['times_s']);xs=np.array(action['states'])
        ts2=np.linspace(ts[0],ts[-1],round((ts[-1]-ts[0])/step)+1)
        xs2=np.column_stack([np.interp(ts2,ts,xs[:,j]) for j in range(9)])
        times.extend(ts2);states.extend(xs2);kinds.extend(['main']*len(ts2))
        if i+1<len(cache['actions']):
            following=cache['actions'][i+1]
            gap=np.linspace(ts[-1],following['times_s'][0],round((following['times_s'][0]-ts[-1])/step)+1)
            for x in (xs[-1],np.array(following['states'][0])):
                times.extend(gap);states.extend([x]*len(gap));kinds.extend(['gap_endpoint_envelope']*len(gap))
    return np.array(times),np.array(states),np.array(kinds)


def fit_obstacles(root):
    catalog=json.loads((root/'combined_catalog.json').read_text());geometry=CollisionGeometry();reports=[]
    for case in catalog['scenarios']:
        name=case['name'];folder=root/name;scene=json.loads((folder/'runtime_scene.json').read_text())
        cache_path=root/'video_combo'/f'{name}_nominal.json';cache=json.loads(cache_path.read_text())
        ts,xs,kinds=samples_with_gap_boundaries(cache,.10)
        frame_p,frame_r=geometry.frame_arrays(xs,scene['base_z'])
        duration=case['duration_s'];subject=np.array(scene['subject']['position'])
        full_times=np.linspace(0,duration,round(duration/.02)+1)
        chosen=[];events=[];curves=[]
        # Tangential lanes around the working area, rather than per-shot
        # wrist-centered crossings. Vary trial angles to maintain spacing.
        count=case['obstacle_count'];angles=np.linspace(-150,150,count,endpoint=False) if count>1 else np.array([-85. if case['number']==1 else -35.])
        for j,target in enumerate(case['target_obstacle_surface_clearances_m']):
            action=cache['actions'][j%len(cache['actions'])]
            event_t=(action['times_s'][0]+action['times_s'][-1])/2
            anchor=np.array(action['states'][len(action['states'])//2])[:2]
            radius=.18
            begin=max(0.,event_t-4.5);end=min(duration,event_t+4.5)
            accepted=None
            for trial in range(300):
                turn=trial%100
                angle=np.deg2rad(angles[j]+(turn//2+1)*7*(1 if turn%2==0 else -1))
                normal=np.array([np.cos(angle),np.sin(angle)])
                tangent=np.array([-normal[1],normal[0]])*(1 if j%2==0 else -1)
                travel=(.65,.45,.30)[trial//100]+.05*(j%3)
                def make(distance):
                    center=anchor+distance*normal
                    start=np.r_[center-.5*travel*tangent,radius+.002]
                    finish=np.r_[center+.5*travel*tangent,radius+.002]
                    keys={0.:start,begin:start,end:finish,duration:finish}
                    kt=sorted(keys);kp=np.array([keys[t] for t in kt]);kv=np.zeros_like(kp)
                    return CubicHermiteSpline(kt,kp,kv),[dict(t=t,position=p.tolist(),velocity=v.tolist()) for t,p,v in zip(kt,kp,kv)]
                def clearance(distance):
                    curve,_=make(distance)
                    return float(geometry._primitive_arrays(frame_p,frame_r,curve(ts)[:,None,:]).min()-radius)
                try:
                    if clearance(0.)>target:continue
                    distance=brentq(lambda d:clearance(d)-target,0.,4.,xtol=2e-5)
                except ValueError:continue
                curve,keys=make(distance);points=curve(full_times)
                # Clearance between ground props and the actual blue cylinder,
                # including its bottom height (no infinite-cylinder fiction).
                from jev_agent_wbc.kc_svsdf import primitive_sdf
                subject_clearance=primitive_sdf(points-subject,'cylinder',[scene['subject']['radius'],scene['subject']['height']])-radius
                if subject_clearance.min()<.20:continue
                pair_min=min((float((np.linalg.norm(points-c(full_times),axis=1)-2*radius).min()) for c in curves),default=99.)
                if pair_min<(.12 if case['difficulty']=='hard' else .20):continue
                if np.max(np.abs(points[:,:2]))>2.5:continue
                accepted=(curve,keys,distance,pair_min,float(subject_clearance.min()));break
            if accepted is None:raise RuntimeError(f'No independent lane found: {name}/{j}')
            curve,keys,distance,pair_min,subject_min=accepted
            curves.append(curve);obj_id=f'floor_prop_{j+1:02d}'
            chosen.append(dict(id=obj_id,radius=radius,color=[.92,.38+.07*(j%3),.08],keyframes=keys))
            events.append(dict(obstacle_id=obj_id,moving_interval_s=[begin,end],nominal_pass_time_s=event_t,
                target_surface_clearance_m=target,travel_m=travel,geometry='grounded sphere proxy for moving floor prop',
                purpose='独立地面路线缓慢经过；启动和停止速度为零，与机器人无反馈耦合。'))
            print(f'LANE {name}/{obj_id}: target {target:.3f} m, pair clearance {pair_min:.3f} m',flush=True)
        scene['obstacles']=chosen;write(folder/'runtime_scene.json',scene)
        # Robot rollout is independent of obstacle geometry. Preserve all states
        # and refresh only the runtime-scene fingerprint after authoring lanes.
        cache['source_sha256'][str(folder/'runtime_scene.json')]=digest(folder/'runtime_scene.json')
        cache['obstacle_independent_nominal']=True;write(cache_path,cache)
        case['obstacle_events']=events
        fine_t,fine_x,fine_kind=samples_with_gap_boundaries(cache,.02)
        centers=np.stack([c(fine_t) for c in curves],axis=1)
        clearances=geometry.distances_sequence(fine_x,centers,scene['base_z'])-.18
        per_object=clearances.min(axis=(0,2));main_mask=fine_kind=='main'
        main_min=float(clearances[main_mask].min());gap_min=float(clearances[~main_mask].min())
        assert np.all(per_object>=np.array(case['target_obstacle_surface_clearances_m'])-.002)
        # Conservative intersample bound for linearly interpolated nominal
        # state playback only. Separate each action to exclude hidden IK resets.
        interval_lower=[]
        for action in cache['actions']:
            times=np.array(action['times_s']);states=np.array(action['states']);dt=np.diff(times)
            points=np.stack([c(times) for c in curves],axis=1)
            d=geometry.distances_sequence(states,points,scene['base_z'])-.18
            robot_motion=np.array([geometry.displacement_bound(dx) for dx in np.diff(states,axis=0)])
            max_obstacle_speed=max(float(np.linalg.norm(c(full_times,1),axis=1).max()) for c in curves)
            interval_lower.append(float((d[:-1].min(axis=(1,2))-robot_motion-max_obstacle_speed*dt).min()))
        pair_min=min((float((np.linalg.norm(a(full_times)-b(full_times),axis=1)-.36).min())
            for i,a in enumerate(curves) for b in curves[i+1:]),default=None)
        report=dict(case=name,difficulty=case['difficulty'],obstacle_count=count,sample_step_s=.02,
            nominal_main_min_surface_clearance_m=main_min,
            nominal_playback_interval_lower_bound_m=min(interval_lower),
            gap_endpoint_envelope_min_surface_clearance_m=gap_min,
            per_obstacle_min_surface_clearance_m=per_object.tolist(),
            obstacle_pair_min_surface_clearance_m=pair_min,
            max_obstacle_speed_m_s=max(float(np.linalg.norm(c(full_times,1),axis=1).max()) for c in curves),
            scope='full 8-body CollisionGeometry; offline per-action nominal WBC; chassis/casters conservative; no gap motion constructed',
            gap_scope='both fixed adjacent endpoint robot configurations checked throughout each empty gap; actual future Agent arc is not evaluated')
        assert report['nominal_playback_interval_lower_bound_m']>0.
        case['clearance_audit']=deepcopy(report);reports.append(report)
        print(f'AUDIT {name}: min {main_min:.3f} m, interval bound {min(interval_lower):.3f} m',flush=True)
    write(root/'combined_catalog.json',catalog)
    write(root/'video_combo/clearance_audit.json',dict(version='graded_combo_clearance/v2',scenarios=reports))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fit-obstacles',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    fit_obstacles(root) if args.fit_obstacles else author(root)
