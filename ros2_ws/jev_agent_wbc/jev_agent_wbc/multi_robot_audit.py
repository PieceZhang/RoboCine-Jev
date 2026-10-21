"""Independent measured execution/intent audit; never feeds future scene truth back."""
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .camera_intent import IntentSpec, progress_at
from .contracts import native
from .reference import OriginalReference
from .representation import Pose
from .task_space_planning import camera_scene


def camera_intrinsics(camera):
    fx=.5/np.tan(np.deg2rad(camera['horizontal_fov_deg'])/2)
    width,height=camera['resolution']
    return [float(fx),float(fx*width/height),.5,.5]


def visibility_scene(snapshot,stamp,end):
    """Exclude identified target returns only from external-occlusion scoring."""
    scene=camera_scene(snapshot,stamp,end)
    labels=snapshot.get('surfel_subject_mask')
    if labels is None:
        return dict(scene,valid=False,visibility_reason='subject_return_labels_unavailable')
    labels=np.asarray(labels,bool)
    if labels.shape!=(len(scene['centers']),):
        raise ValueError('subject return labels must bind every observed surfel')
    tracks=np.asarray(snapshot.get('surfel_track_ids',np.zeros(len(labels))),int)
    keep=~(labels & (tracks==0))
    return dict(scene,**{key:np.asarray(scene[key])[keep] for key in
        ('centers','radii','velocities','radius_rates')},
        visibility_label_source=snapshot.get('subject_label_source'))


def classify_agent_trial(mode,installed,events,api_records):
    """Transport-only missing decisions are unverified, never planner failures."""
    if mode!='jev':
        return 'diagnostic_not_agent'
    if installed:
        return 'agent_plan_installed_requires_physical_audit'
    # A later exact selected plan that misses installation is more informative
    # than an earlier retry with no numerical candidate or semantic refusal.
    if any(event['event']=='late_plan_rejected' for event in events):
        return 'failed_installation'
    reasons=[event.get('check',{}).get('reason') for event in events]
    if 'no_native_model_screened_candidate' in reasons:
        return 'no_feasible_native_candidate'
    if 'semantic_refusal' in reasons:
        return 'semantic_refusal'
    if any(row.get('status') in ('invalid_protocol_reply','request_failed') for row in api_records):
        return 'invalid_api_reply_or_request'
    if any(row.get('status')=='transport_unverified_retry' for row in api_records):
        return 'unverified_network'
    return 'unverified_no_agent_plan'


def tiago_guard_audit(rows):
    """Require recorded pre/post physical checks; absence never means pass."""
    failures=[]
    for row in rows:
        for key in ('measured_full_dof_guard','post_step_full_dof_guard'):
            guard=row.get(key,{})
            if (guard.get('raw_physical_dofs_checked')!=len(row.get('q',[]))
                    or not row.get('q') or guard.get('error')
                    or guard.get('max_position_excess',float('inf'))>1e-5
                    or guard.get('max_velocity_excess',float('inf'))>1e-4):
                failures.append(dict(stamp_s=row.get('t_s'),guard=key))
        for key in ('measured_motion_guard','post_step_measured_motion_guard'):
            guard=row.get(key,{})
            if (guard.get('raw_measured_velocity_limits_checked') is not True
                    or guard.get('max_speed_excess',float('inf'))>1e-4
                    or guard.get('minimum_joint_stopping_reserve_mixed_units',-float('inf')) < -1e-6):
                failures.append(dict(stamp_s=row.get('t_s'),guard=key))
    return dict(pass_check=bool(rows and not failures),samples=len(rows),
                failures=failures,scope='original_raw_full_DOF_velocity_position_and_stopping_guards')


def restore_execution_evidence(original, initial_record, selected_record, installed_events, start_s):
    """Bind progress and executable knots to the logged immutable selection."""
    from .candidates import restore_path,path_hash
    from .representation import Observation
    from .multi_robot_scenarios import timed_path
    initial=original
    if initial_record is not None:
        initial=restore_path(original,initial_record['trajectory'])
        if path_hash(initial)!=initial_record['path_id']:
            raise ValueError('initial execution path hash mismatch')
    if not installed_events:
        return initial,None,None
    if len(installed_events)!=1 or selected_record is None:
        raise ValueError('installed path requires one exact recorded selection')
    card=selected_record['card'];report=selected_record['lift']
    selected=restore_path(original,card['trajectory'])
    identity=path_hash(selected)
    if not identity==card['id']==report['path_id']==installed_events[0]['path_id']:
        raise ValueError('selected path identity mismatch')
    observation=Observation.from_dict(card['visual']['geometry']['observation'])
    execution=timed_path(selected,observation,start_s)
    trajectory=execution.trajectory
    if 'camera_trajectory' in selected_record:
        from .wbc_protocol import CameraTrajectory
        recorded=CameraTrajectory.from_dict(selected_record['camera_trajectory'])
        # The exact transmitted knots are immutable evidence. A restored
        # scalar curve can differ from batched evaluation at the last bit;
        # compare its geometry independently without changing the stored hash.
        if recorded.start_s!=trajectory.start_s or recorded.times_s!=trajectory.times_s:
            raise ValueError('selected executable camera knot clock mismatch')
        for actual,expected in zip(recorded.poses,trajectory.poses):
            if (not np.allclose(actual.translation_m,expected.translation_m,rtol=0.,atol=1e-12)
                    or not np.allclose(actual.rotation,expected.rotation,rtol=0.,atol=1e-12)):
                raise ValueError('selected executable camera knots disagree with selected curve')
        trajectory=recorded
    if not trajectory.id==report['camera_path_id']==card['visual']['geometry']['request']['camera_path_id']:
        raise ValueError('selected executable camera knots mismatch')
    return initial,selected,float(installed_events[0]['stamp_s'])


def audit_run(directory,filename='independent_audit.json'):
    directory=Path(directory);root=directory/'avoidance'
    inputs=['summary.json','measurements.jsonl','observed_maps.jsonl','events.jsonl',
            'original_task.json','runtime_scene.json']
    data={f:(root/f).read_bytes() for f in inputs}
    summary=json.loads(data['summary.json'])
    rows=[json.loads(line) for line in data['measurements.jsonl'].splitlines()]
    maps=[json.loads(line) for line in data['observed_maps.jsonl'].splitlines()]
    events=[json.loads(line) for line in data['events.jsonl'].splitlines()]
    bound=next((r for r in events if r['event']=='scenario_bound'),None)
    result=dict(schema='five_robot_independent_acceptance/v1',robot=summary['robot'],
        case=summary['case'],mode=summary['mode'],complete=summary['complete'],
        exact_agent_plan_installed=summary['exact_plan_installed'],successful=False,
        agent_trial_status=summary.get('agent_trial_status','historical_not_classified'),
        input_sha256={f:hashlib.sha256(value).hexdigest() for f,value in data.items()},
        truth_scope='actual moving-obstacle poses and conservative collider OBBs at control acquisitions',
        visibility_scope='calibrated subject geometry and causal observed LiDAR; no RGB/SAM certificate',
        physical_summary=summary)
    native_file=directory/('summary.json' if summary['robot'] in ('g1','go2') else 'acceptance.json')
    controller=json.loads(native_file.read_text()) if native_file.exists() else {}
    if native_file.exists():
        result['input_sha256']['../'+native_file.name]=hashlib.sha256(native_file.read_bytes()).hexdigest()
    result['timing_pass']=controller.get('timing_pass',False)
    result['simulation_s']=controller.get('sim_seconds',controller.get('simulation_s',controller.get('simulated_seconds')))
    result['wall_s']=controller.get('wall_seconds',controller.get('wall_s'))
    if summary['robot'] in ('g1','go2'):
        counts=controller.get('constraint_status_counts',{})
        result['native_constraint_status_counts']=counts
        result['native_hard_constraints_pass']=bool(counts and all(
            counts.get(family,{}).get('fail',1)==0 and counts.get(family,{}).get('unknown',1)==0
            for family in ('configuration','actuation','self_collision','environment')))
        result['stop_pass']=bool(controller.get('cases') and all(
            c.get('strict_stop',{}).get('passed',False) for c in controller['cases']))
    elif summary['robot']=='tiago':
        samples_file=directory/'samples.jsonl'
        raw=samples_file.read_bytes() if samples_file.exists() else b''
        samples=[json.loads(line) for line in raw.splitlines()]
        result['input_sha256']['../samples.jsonl']=hashlib.sha256(raw).hexdigest()
        result['measured_native_guards']=tiago_guard_audit(samples)
        task_samples=[r for r in samples if r.get('task_feedback')]
        counts={family:dict(Counter(r['task_feedback'].get('constraints',{}).get(family,{}).get('status','unknown')
                                   for r in task_samples))
                for family in ('configuration','actuation','self_collision','environment')}
        result['native_constraint_status_counts']=counts
        result['native_hard_constraints_pass']=bool(controller.get('complete') and
            result['measured_native_guards']['pass_check'] and task_samples and
            all(c.get('fail',0)==0 and c.get('unknown',0)==0 for c in counts.values()))
        stops=[p for p in controller.get('phases',[]) if 'stop' in p.get('phase','')]
        result['stop_pass']=bool(stops and all(p.get('passed',False) for p in stops))
    else:
        phases=controller.get('phases',[])
        result['native_hard_constraints_pass']=bool(controller.get('complete') and
            controller.get('measured_safety_pass',False) and not summary['execution_failure'])
        stops=[p for p in phases if 'stop' in p.get('phase','')]
        result['stop_pass']=bool(stops and all(p.get('passed',False) for p in stops))
    result['native_hard_constraints_status']='pass' if result['native_hard_constraints_pass'] else 'unknown'
    if (any(c.get('fail',0)>0 for c in result.get('native_constraint_status_counts',{}).values())
            or result.get('measured_native_guards',{}).get('pass_check') is False
            or controller.get('measured_safety_violations',0)>0):
        result['native_hard_constraints_status']='fail'
    if bound and len(rows)>2:
        start=bound['shot_start_s']
        selected=[];last=-1.
        for row in rows:
            t=row['stamp_s']-start
            if t>8.4+1e-8:break
            if t-last>=.099-1e-8:
                selected.append(row);last=t
        if len(selected)>2:
            scene=json.loads(data['runtime_scene.json'])
            original=OriginalReference(json.loads(data['original_task.json']),scene['subject']['position'])
            path_records=[]
            for evidence_name in ('initial_execution_path.json','selected_plan.json'):
                file=root/evidence_name
                raw=file.read_bytes() if file.exists() else None
                if raw is not None:
                    result['input_sha256'][evidence_name]=hashlib.sha256(raw).hexdigest()
                path_records.append(json.loads(raw) if raw is not None else None)
            installation=[event for event in events if event['event']=='exact_plan_installed']
            try:
                initial,chosen,installed_at=restore_execution_evidence(original,*path_records,installation,start)
                result['execution_path_binding_pass']=bool(
                    summary['exact_plan_installed']==bool(installation))
            except (KeyError,TypeError,ValueError) as error:
                result['execution_path_binding_pass']=False
                result['execution_path_binding_error']=str(error)
                initial,chosen,installed_at=original,None,None
            sensor_bytes=(directory/'sensor_mounts.json').read_bytes()
            result['input_sha256']['../sensor_mounts.json']=hashlib.sha256(sensor_bytes).hexdigest()
            sensors=json.loads(sensor_bytes)
            specification=IntentSpec(original,scene['subject'],camera_intrinsics(sensors['camera']))
            consumed=[m['consumed_s'] for m in maps]
            times=[];positions=[];rotations=[];scenes=[];progress=[]
            for row in selected:
                now=row['stamp_s'];t=now-start;index=bisect_right(consumed,now)-1
                if index>=0 and 0<=now-maps[index]['snapshot']['observation_stamp']<=.25:
                    observed=visibility_scene(maps[index]['snapshot'],now,now)
                    observed['observation_stamp']-=start
                else:
                    observed=dict(valid=False)
                pose=Pose.from_dict(row['actual_camera'])
                times.append(t);positions.append(pose.translation_m);rotations.append(pose.rotation)
                path=chosen if chosen is not None and now>=installed_at else initial
                scenes.append(observed);progress.append(progress_at(path,min(t,path.duration)))
            result['intent']=specification.measure(times,positions,rotations,scenes,progress)
        result['maximum_scene_tracking_error_m']=max((r.get('scene_tracking_error_m',float('inf')) for r in rows),default=None)
    result['physical_acceptance']=bool(summary.get('physical_shot_pass') and result['native_hard_constraints_pass']
        and result['stop_pass'] and result.get('intent',{}).get('intent_complete')
        and result.get('execution_path_binding_pass',False)
        and result.get('maximum_scene_tracking_error_m',float('inf'))<.001)
    result['successful']=bool(result['physical_acceptance'] and result['exact_agent_plan_installed'])
    result['realtime_end_to_end_pass']=bool(result['successful'] and result['timing_pass'])
    destination=root/filename
    if destination.exists():
        raise FileExistsError('audit artifacts are immutable; choose a new filename')
    destination.write_text(json.dumps(native(result),indent=2,allow_nan=False)+'\n')
    return result
