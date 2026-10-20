"""Grounded cuboids that retain a constructive whole-shot escape route."""
from copy import deepcopy
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from jev_agent_wbc.kc_svsdf import CollisionGeometry,primitive_sdf
from jev_agent_wbc.runtime_obstacles import local_cover,box_rotation
from .solvability import safe_clock_indices


def add_ground_boxes(rng,scene,times,states,objects,audit,pool,config):
    if pool is None:raise ValueError('mixed scenes require an offline avoidance witness')
    objects=deepcopy(objects);audit=deepcopy(audit)
    survivors=audit['timing_witness_indices'];geometry=CollisionGeometry()
    frames=geometry.frame_arrays(states,scene['base_z'])
    curves=[CubicHermiteSpline([k['t'] for k in o['keyframes']],
        [k['position'] for k in o['keyframes']],[k['velocity'] for k in o['keyframes']]) for o in objects]
    nominal_xy=states[:,:2];direction=nominal_xy[-1]-nominal_xy[0]
    heading=np.arctan2(direction[1],direction[0]) if np.linalg.norm(direction)>.03 else states[0,2]
    pair_min=audit['obstacle_pair_min_clearance_m']
    for number in range(2 if config.difficulty!='hard' else 3):
        accepted=None
        for attempt in range(100):
            height=float(rng.uniform(*((.16,.38) if number%2==0 else (.55,1.05))))
            size=np.array([rng.uniform(.22,.62),rng.uniform(.20,.55),height])
            angle=heading+(1 if number%2==0 else -1)*np.pi/2+rng.uniform(-.6,.6)
            center=np.r_[nominal_xy[int(rng.integers(len(times)))]+rng.uniform(.65,1.35)*np.array([np.cos(angle),np.sin(angle)]),height/2]
            obj=dict(id=f'ground_box_{number+1:02d}',shape='box',size=size.tolist(),
                yaw_rad=float(rng.uniform(-np.pi,np.pi)),radius=float(np.linalg.norm(size)/2),
                color=[.2,.55,.7] if number%2 else [.6,.45,.2],
                keyframes=[dict(t=float(t),position=center.tolist(),velocity=[0.,0.,0.]) for t in (0.,times[-1])])
            offsets,radii=local_cover(obj);rotation=box_rotation(obj)
            coarse=(geometry._primitive_arrays(frames[0][::10],frames[1][::10],(center+offsets)[None]).min(axis=-1)-radii).min()
            # Static boxes constrain nearby escape space while leaving the
            # authored route open. Dynamic actors still force avoidance.
            if not .14<=coarse<=.50:continue
            pair=100.
            for other,curve in zip(objects,curves):
                if other.get('shape')=='box':
                    gap=np.linalg.norm(center-curve(0.))-obj['radius']-other['radius']
                else:
                    gap=(primitive_sdf((curve(times)-center)@rotation,'box',size)-other['radius']).min()
                pair=min(pair,float(gap))
            if pair<.04:continue
            subject=scene['subject'];subject_radius=np.hypot(subject['radius'],subject['height']/2)+.07
            if primitive_sdf((np.array(subject['position'])-center)@rotation,'box',size)[0]-subject_radius<.06:continue
            curve=CubicHermiteSpline([0.,times[-1]],[center,center],np.zeros((2,3)))
            remaining=safe_clock_indices(pool,survivors,curve,obj['radius'],0.,obstacle=obj)
            if not remaining:continue
            distances=(geometry._primitive_arrays(*frames,(center+offsets)[None]).min(axis=-1)-radii).min(axis=1)
            if distances.min()<.12 or min(distances[0],distances[-1])<config.initial_clearance_m:continue
            report=dict(obstacle_id=obj['id'],shape='box',size_m=size.tolist(),yaw_rad=obj['yaw_rad'],
                height_level='low' if number%2==0 else 'high',motion='static_ground_box',radius_m=obj['radius'],
                interaction_role='corridor_boundary',motion_model='stationary_grounded_cuboid',
                continuous_motion_bounds=dict(speed_m_s=0.,acceleration_m_s2=0.,jerk_m_s3=0.),
                center_height_range_m=[height/2,height/2],ground_bottom_m=0.,
                min_surface_clearance_m=float(distances.min()),initial_clearance_m=float(distances[0]),
                final_clearance_m=float(distances[-1]),global_penetration_cap_m=0.,
                max_speed_m_s=0.,max_acceleration_m_s2=0.,interference_sample_count=0,
                first_conflict_s=None,conflict_duration_s=0.,
                collision_model='conservative full-volume cuboid ball cover',collision_proxy_count=len(offsets))
            accepted=obj,report,curve,remaining;pair_min=min(pair_min,pair);break
        if accepted is None:raise ValueError('could not place grounded boxes while retaining a mixed-scene escape witness')
        obj,report,curve,survivors=accepted
        objects.append(obj);curves.append(curve);audit['obstacles'].append(report)
    audit.update(timing_witness_indices=survivors,ground_box_count=sum(o.get('shape')=='box' for o in objects),
        obstacle_pair_min_clearance_m=pair_min,
        ground_box_route_policy='all boxes retain at least one common full-body timing witness with every dynamic actor')
    return objects,audit
