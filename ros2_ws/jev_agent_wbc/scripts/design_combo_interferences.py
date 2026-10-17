#!/usr/bin/env python3
"""Fit low/middle/high moving obstacle crossings against timed nominal robot motion.

Only scene data is authored. Camera actions and empty transitions are unchanged.
Negative signed clearance is required separately for every obstacle, during a
main action, with a numerical witness and a nonzero interference duration.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'scripts')]
from jev_agent_wbc.kc_svsdf import CollisionGeometry, primitive_sdf
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from record_scenario_trajectories import digest
from author_graded_combo_scenarios import samples_with_gap_boundaries, write

POLICIES = {
    'easy': dict(penetration_m=.025, crossing_speed_m_s=.34, overlap=False),
    'medium': dict(penetration_m=.055, crossing_speed_m_s=.27, overlap=False),
    'hard': dict(penetration_m=.085, crossing_speed_m_s=.20, overlap=True),
}
LEVELS = [('low', '低', .16, [0], [.97,.60,.08]),
          ('middle', '中', .48, [2,3,4], [.94,.25,.06]),
          ('high', '高', None, [5,6,7], [.66,.16,.84])]


def event_schedule(case):
    actions=case['main_actions']
    if case['difficulty']=='easy':
        return [(0,actions[0]['start_s']+2.3),(0,actions[0]['start_s']+5.1),
                (1,actions[1]['start_s']+3.5)]
    if case['difficulty']=='medium':
        return [(i,(a['start_s']+a['end_s'])/2) for i,a in enumerate(actions)]
    # Two simultaneous constraints on the base and middle arm, followed by an
    # upper-arm/camera crossing in the last action. No contacts in empty gaps.
    a=actions[1];b=actions[2]
    return [(1,(a['start_s']+a['end_s'])/2-.20),
            (1,(a['start_s']+a['end_s'])/2+.20),
            (2,(b['start_s']+b['end_s'])/2)]


def main(root):
    catalog=json.loads((root/'combined_catalog.json').read_text());geometry=CollisionGeometry();reports=[]
    for case in catalog['scenarios']:
        name=case['name'];folder=root/name;scene=json.loads((folder/'runtime_scene.json').read_text())
        cache_path=root/'video_combo'/f'{name}_nominal.json';cache=json.loads(cache_path.read_text())
        task_hash=digest(folder/'task.json')
        ts,xs,kinds=samples_with_gap_boundaries(cache,.04)
        main_mask=kinds=='main';fp,fr=geometry.frame_arrays(xs,scene['base_z'])
        duration=case['duration_s'];policy=POLICIES[case['difficulty']]
        scan_t=np.linspace(0,duration,round(duration/.02)+1)
        selected=[];curves=[];events=[]
        for j,((level,label,z,bodies,color),(action_index,event_t)) in enumerate(zip(LEVELS,event_schedule(case))):
            action=cache['actions'][action_index];at=np.array(action['times_s']);states=np.array(action['states'])
            state=np.array([np.interp(event_t,at,states[:,k]) for k in range(9)])
            camera,rotation,_=whole_body_kinematics(state,scene['base_z'])
            if z is None:z=float(max(.74,camera[2]-.035))
            frames,_,_=geometry.frames(state,scene['base_z'])
            candidates=[]
            for body,_,_,_,offset,_ in geometry.shapes:
                if body in bodies:
                    bp,br=frames[body];point=bp+br@offset
                    candidates.append((abs(point[2]-z),point))
            anchor=min(candidates,key=lambda a:a[0])[1].copy();anchor[2]=z
            radius=.12;speed=policy['crossing_speed_m_s'];target=-policy['penetration_m']
            accepted=None
            # Vary orientation of the *fixed* crossing line. Neither the lane
            # nor its clock follows robot feedback during the demonstration.
            for attempt in range(120):
                theta=np.deg2rad((j*73+case['number']*19)+attempt*23)
                direction=np.array([np.cos(theta),np.sin(theta),0.])
                normal=np.array([-direction[1],direction[0],0.])
                if attempt%2:normal=-normal
                lead=1.1;start=max(0.,event_t-3.8);end=min(duration,event_t+3.8)
                remote=1.05+.10*(attempt//40)
                def make(offset):
                    center=anchor+offset*normal
                    rows={0.:(center-remote*direction,np.zeros(3)),
                          start:(center-remote*direction,np.zeros(3)),
                          event_t-lead:(center-speed*lead*direction,speed*direction),
                          event_t+lead:(center+speed*lead*direction,speed*direction),
                          end:(center+remote*direction,np.zeros(3)),
                          duration:(center+remote*direction,np.zeros(3))}
                    kt=sorted(rows);kp=np.array([rows[t][0] for t in kt]);kv=np.array([rows[t][1] for t in kt])
                    return CubicHermiteSpline(kt,kp,kv),[dict(t=float(t),position=p.tolist(),velocity=v.tolist()) for t,p,v in zip(kt,kp,kv)]
                window=(np.abs(ts-event_t)<1.7)&main_mask
                def distances(offset):
                    curve,_=make(offset)
                    return geometry._primitive_arrays(fp,fr,curve(ts)[:,None,:]).min(axis=(1,2))-radius
                try:
                    # Fit penetration in the intended time window, then reject
                    # a lane if another time causes an unintended deeper hit.
                    if distances(0.)[window].min()>target:continue
                    distance=brentq(lambda d:distances(d)[window].min()-target,0.,.65,xtol=2e-5)
                except ValueError:continue
                curve,keys=make(distance);ds=distances(distance)
                contact=(ds<-.003)&main_mask
                if not contact.any() or np.any(contact&~window):continue
                if ds[main_mask].min()<target-.004:continue
                # All action boundaries and the whole initial/final parked
                # portion must be clear; contact belongs to the moving pass.
                boundary=np.array([any(abs(t-a[k])<1e-7 for a in case['main_actions'] for k in ('start_s','end_s')) for t in ts])
                if ds[boundary].min()<.08 or ds[~main_mask].min()<.08:continue
                if min(ds[np.isclose(ts,0.)].min(),ds[np.isclose(ts,duration)].min())<.35:continue
                points=curve(scan_t)
                if np.linalg.norm(curve(scan_t,1),axis=1).max()>.90:continue
                if any((np.linalg.norm(points-c(scan_t),axis=1)-2*radius).min()<.045 for c in curves):continue
                subject=np.array(scene['subject']['position'])
                subject_d=primitive_sdf(points-subject,'cylinder',[scene['subject']['radius'],scene['subject']['height']])-radius
                if subject_d.min()<.08:continue
                accepted=(curve,keys,float(ds[main_mask].min()));break
            if accepted is None:raise RuntimeError(f'No timed interference lane found: {name}/{level}')
            curve,keys,minimum=accepted;curves.append(curve);obj_id=f'{level}_crossing'
            selected.append(dict(id=obj_id,radius=radius,color=color,keyframes=keys))
            events.append(dict(obstacle_id=obj_id,height_level=level,height_label=label,center_height_m=z,
                target_action=case['main_actions'][action_index]['phase'],intended_crossing_time_s=event_t,
                moving_interval_s=[start,end],target_penetration_m=policy['penetration_m'],
                crossing_speed_m_s=speed,description='固定高度独立横穿路线；与同一时刻的名义整机轨迹发生几何干涉。'))
            print(f'FITTED {name}/{level}: z={z:.3f} m, clearance={minimum:.3f} m',flush=True)
        fine_t,fine_x,fine_kind=samples_with_gap_boundaries(cache,.01)
        centers=np.stack([c(fine_t) for c in curves],axis=1)
        ds=geometry.distances_sequence(fine_x,centers,scene['base_z'])-.12
        main_mask=fine_kind=='main';main_t=fine_t[main_mask];main_ds=ds[main_mask]
        contacts=main_ds.min(axis=2)<-.003
        obstacle_reports=[]
        for j,event in enumerate(events):
            time_idx,body_idx=np.unravel_index(np.argmin(main_ds[:,j,:]),main_ds[:,j,:].shape)
            hit_times=main_t[contacts[:,j]];assert len(hit_times)>=20
            assert main_ds[:,j,:].min()<-.015
            obstacle_reports.append(dict(**event,min_surface_clearance_m=float(main_ds[:,j,:].min()),
                witness_time_s=float(main_t[time_idx]),witness_body_index=int(body_idx),
                witness_obstacle_center_m=curves[j](main_t[time_idx]).tolist(),
                interference_interval_s=[float(hit_times.min()),float(hit_times.max())],
                interference_duration_s=float(len(hit_times)*.01),interference_verified=True))
        simultaneous=int(contacts.sum(axis=1).max())
        assert simultaneous==(2 if case['difficulty']=='hard' else 1), (name,simultaneous)
        pair_min=min(float((np.linalg.norm(a(scan_t)-b(scan_t),axis=1)-.24).min()) for i,a in enumerate(curves) for b in curves[i+1:])
        report=dict(case=name,difficulty=case['difficulty'],obstacle_count=3,sample_step_s=.01,
            nominal_main_min_surface_clearance_m=float(main_ds.min()),
            all_obstacles_interfere=True,verified_obstacle_count=3,max_simultaneous_interfering_obstacles=simultaneous,
            obstacle_pair_min_surface_clearance_m=pair_min,
            gap_endpoint_envelope_min_surface_clearance_m=float(ds[~main_mask].min()),
            initial_min_surface_clearance_m=float(ds[np.isclose(fine_t,0.)].min()),
            final_min_surface_clearance_m=float(ds[np.isclose(fine_t,duration)].min()),
            max_obstacle_speed_m_s=max(float(np.linalg.norm(c(scan_t,1),axis=1).max()) for c in curves),
            obstacles=obstacle_reports,
            metric='signed obstacle-sphere surface clearance to full 8-body robot collision geometry; negative means geometric interference',
            scope='offline nominal per-action WBC at matching physical timestamps; no avoidance or PhysX collision response; gaps excluded from hit evidence')
        scene['obstacles']=selected
        scene['description']=f"{case['number']:02d} {case['difficulty_zh']}：低/中/高三个固定高度横穿障碍物，三者均与名义整机运动产生干涉；衔接留空。"
        write(folder/'runtime_scene.json',scene)
        cache['source_sha256']={str(folder/f):digest(folder/f) for f in ('task.json','runtime_scene.json')}
        cache['obstacle_independent_nominal']=True;write(cache_path,cache)
        assert digest(folder/'task.json')==task_hash
        case.pop('target_obstacle_surface_clearances_m',None)
        case['obstacle_count']=3;case['obstacle_events']=obstacle_reports
        case['interference_audit']=report;case.pop('clearance_audit',None);reports.append(report)
        print(f"VERIFIED {name}: 3/3; max overlap {simultaneous}; durations {[round(o['interference_duration_s'],2) for o in obstacle_reports]}",flush=True)
    catalog.update(version='graded_combo/v3_height_interference',
        description='六个组合均有低、中、高三个障碍物；每个障碍物必须与同一时刻的名义整机运动发生实际几何干涉。',
        obstacle_design='三个固定高度的独立移动球形代理；横穿底盘、中部机械臂、上部机械臂/相机；起止与衔接窗口不安排接触。',
        difficulty_definition={k:dict(obstacle_count=3,**v) for k,v in POLICIES.items()},
        validation_scope='10 ms 同时刻整机 signed-distance 干涉证据；每个障碍物有见证时间、部件、持续时间。不是 Agent 闭环避障验证。')
    write(root/'combined_catalog.json',catalog)
    write(root/'video_combo/interference_audit.json',dict(version='combo_interference/v3',scenarios=reports))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scenarios',type=Path,required=True)
    main(parser.parse_args().scenarios.resolve())
