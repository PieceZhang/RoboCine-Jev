"""Compact public consequences; retain full native traces in the execution journal."""
from copy import deepcopy


def intent_summary(report):
    """Remove repeated shape descriptions and bin traces, never a scored outcome."""
    result=deepcopy(report)
    for segment in result.get('segments',[]):
        segment.pop('coverage_bin_dwell_s',None)
    frame_fields=('valid','centroid','unoccluded_area','bbox_size','view_direction',
                  'distance_m','elevation_rad','silhouette_axis_rad','body_orientation','source')
    def trim_frames(value):
        if isinstance(value,dict):
            value.pop('target',None)
            measured=value.pop('predicted_or_actual',None)
            if isinstance(measured,dict):
                value['endpoint_state']={k:v for k,v in measured.items() if k in frame_fields}
            for child in value.values():trim_frames(child)
        elif isinstance(value,list):
            for child in value:trim_frames(child)
    trim_frames(result)
    result['detail_scope']='all_scored_outcomes_retained; full_shape_and_bin_traces_in_native_screen_report'
    return result


def deformation_program(path):
    """Explicit camera-space meaning for the existing geometric candidate view."""
    from .contracts import PARAMETERS
    return dict(kind='returning_camera_deformation',relative_to='immutable_original',
        offsets={key:float(value) for key,value in zip(PARAMETERS,path.values) if value!=0.},
        blend_s=path.envelope.blend_s,return_s=path.return_s,
        protected=list(path.envelope.protected),terminal=path.envelope.terminal)


def native_search_feedback(events,shot_s,activation_s):
    """Describe only this bounded search; never infer global infeasibility."""
    reasons={};failures=[];eligible=screened=valid=0
    for event in events:
        if event['event']=='camera_proposal_screen':
            intent=event['intent']
            eligible+=int(intent['geometric_complete'] and intent['motion_trend']['pass_check']
                          and intent['camera_limits_pass'])
        if event['event']!='candidate':continue
        screened+=1;card=event['card'];report=card['screen'];valid+=int(card['physical_valid'])
        program=deepcopy(card['trajectory'].get('program',{}))
        if 'route' not in program:
            program['route']=', '.join(f'{key} {value:+.3f} m' for key,value in
                program.get('offsets',{}).items() if key.endswith('_m')) or 'returning_camera_deformation'
        for key,reason in [('tracking_screen_pass','tracking_unreachable'),
                           ('observed_collision_screen_pass','observed_collision_clearance'),
                           ('nonholonomic_screen_pass','native_nonholonomic_constraint'),
                           ('native_platform_screen_pass','native_platform_velocity_constraint')]:
            if not report.get(key,False):
                reasons[reason]=reasons.get(reason,0)+1
                failures.append(dict(reason=reason,program=program))
    return dict(time_s=float(shot_s),activation_s=float(activation_s),reasons=reasons,
        failures=failures[:6],search_exhausted=valid==0,
        untested_proposals=max(0,eligible-screened),
        scope='bounded_native_search_only_not_global_infeasibility')


def reference_context(original,observation,snapshot,start_s):
    """Keep all relative reference coordinates in the current mount snapshot."""
    import numpy as np
    from .representation import Pose
    from .task_space_planning import camera_scene
    from .floating_camera import evaluate_path
    relative_stamp=observation.stamp_s-start_s
    scene=camera_scene(snapshot,observation.stamp_s,start_s+original.duration)
    scene=dict(scene,observation_stamp=scene['observation_stamp']-start_s)
    nominal,risk=evaluate_path(original,scene,max(0.,relative_stamp),original.duration,.06,step=.1)
    inverse=observation.world_from_mount.inverse();samples=[]
    for t in sorted(set((0.,float(np.clip(relative_stamp,0.,original.duration)),original.duration/2.,original.duration))):
        position,rotation,_=original.sample(t)
        samples.append(dict(shot_s=t,camera=inverse.compose(Pose.from_rt(rotation,position)).to_dict()))
    return dict(original_reference=dict(frame='mount_snapshot',observation_id=observation.id,
            samples=samples,subject_center_m=inverse.apply([original.subject])[0].tolist()),
        nominal_camera_forecast=dict(score=nominal.to_dict(),observation_id=observation.id,
            acquisition_stamp_s=snapshot['observation_stamp'],projection_stamp_s=observation.stamp_s,
            scope='observed_map_optical_center_only_not_whole_body_or_visibility_certificate')),risk
