"""Allowlisted text view at the Agent boundary, not the numerical event journal."""
from copy import deepcopy

CONFIGURATION_NAMES = dict(local='compact',base='platform_follow',timing='timing',
                           explore='wider_search',executable='backend_realizable')
CONFIGURATION_DESCRIPTIONS = dict(
    local='Search small returning camera paths; prefer a compact camera-manipulator envelope. The backend may propose realizable alternatives; inspect their explicit shot deviations.',
    base='Prefer the mobile platform to follow the camera (0.12 m / 0.3 rad deadband). The backend searches realizable camera alternatives under this soft preference; inspect the exact proposed camera path and shot consequences.',
    timing='Prefer timing changes with the same deadline and small camera displacements; no additional body preference.',
    explore='Explore wider camera detours after failed searches; no additional body preference.',
    executable='Ask the backend to generate realizable camera trajectories when exact tracking failed. Prefer complete original intent and terminal recovery; compare all explicit deviations. A realizable path alone is not shot success; no additional body preference.')


def geometric_card(card):
    geometry = card.get('visual',{}).get('geometry')
    if not geometry:
        return None
    request = deepcopy(geometry['request'])
    # Exact camera path is kept opaque; the existing candidate alias binds it.
    from .representation import Pose
    anchor = Pose.from_dict(geometry['observation']['world_from_mount']).inverse()
    full = geometry['samples']
    selected = sorted(set([0,len(full)//2,len(full)-1])) if full else []
    samples = []
    for i in selected:
        sample = full[i]; snapshot = sample['geometry']
        displacement = anchor.compose(Pose.from_dict(snapshot['world_from_mount']))
        samples.append(dict(stamp_s=sample['stamp_s'],
            mount_in_initial_mount=displacement.to_dict(),
            mount_from_camera=snapshot['mount_from_camera'],
            part_bounds_mount={part:value['bounds'] for part,value in snapshot['parts'].items()},
            preference_residual_norm=sample['preference_residual_norm'],
            execution_uncertainty_m=sample.get('execution_uncertainty_m'),
            safety_supported=sample.get('safety_supported')))
    return dict(request=request,predicted_geometry=samples,evidence_kind='predicted',
                backend_status='screened' if card['physical_valid'] else 'unverified',
                scope=geometry['scope'])


def failure_category(reason):
    reason = str(reason).lower()
    if any(word in reason for word in ('expired','deadline','late','budget')):
        return 'insufficient_time_or_expired_candidate'
    if any(word in reason for word in ('collision','clearance')):
        return 'occupancy_conflict'
    if any(word in reason for word in ('intent','frame','coverage','visibility')):
        return 'camera_intent_mismatch'
    if any(word in reason for word in ('stale','sensor','observation')):
        return 'observation_unavailable_or_stale'
    if 'singular' in reason:
        return 'camera_motion_freedom_low'
    if 'tracking_unreachable' in reason:
        return 'camera_path_not_realized'
    if any(word in reason for word in ('braking','acceleration','qp_infeasible','constraint')):
        return 'backend_motion_conflict'
    return 'backend_motion_unverified'


def search_outcome(row):
    """Retain actionable time/search evidence while hiding robot internals."""
    counts = {}
    for reason,count in row.get('reasons',{}).items():
        category = failure_category(reason)
        counts[category] = counts.get(category,0)+count
    failures = []
    for item in row.get('failures',[])[:6]:
        program = item.get('program') or {}
        at = item.get('first_failure_s')
        activation = row.get('activation_s')
        failures.append(dict(status=failure_category(item.get('reason','')),
            route=program.get('seed_route',program.get('route')),
            phase_rate=program.get('phase_rate',0.), first_failure_s=at,
            failure_before_change=(at < activation if at is not None and activation is not None else None)))
    return dict(time_s=row.get('time_s'),outcomes=counts,
        search_exhausted=row.get('search_exhausted'),untested_proposals=row.get('untested_proposals'),
        activation_s=row.get('activation_s'),failures=failures,
        interpretation='bounded_search_outcome_not_global_infeasibility')


def public_evidence(state):
    """Use positive field selection for backend diagnostics, not key-name guessing."""
    if not state.get('robot_geometry'):
        return state  # Preserve standalone floating-camera and legacy protocols.
    out = deepcopy(state)
    feedback = out['robot_geometry'].get('execution_feedback')
    if feedback:
        feedback.pop('geometry',None)
    out.pop('actual_robot_feedback',None)
    out.pop('robot_capability',None)
    difficulty = out.get('difficulty',{})
    out['difficulty'] = {k:v for k,v in difficulty.items() if k in (
        'first_risk_s','min_clearance_m','intent','accepted_intent_degradation_unchanged','execution_prediction')}
    out['difficulty']['issues'] = [
        'backend_motion_unverified' if ('kinematic' in str(x) or 'joint' in str(x)) else x
        for x in difficulty.get('issues',[])]
    out['execution_history'] = [{k:v for k,v in row.items() if k in (
        'time_s','original_position_error_m','original_orientation_error_rad','target_tracking_residual_m',
        'view_angle_proxy_rad','area_scale_proxy_ratio','visual_proxy_source','clearance_m',
        'actual_parameters','remaining_budget_s')} for row in out.get('execution_history',[])]
    out['validation_feedback'] = [dict(time_s=row.get('time_s'),status=failure_category(row.get('error','')),
                                      failure=row.get('failure'))
                                  for row in out.get('validation_feedback',[])]
    if 'search_options' in out:
        options = out['search_options']
        failures = [search_outcome(row) for row in options.get('latest_search_feedback', [])]
        out['search_options'] = dict(selected_family='combined' if options['selected_family']=='joint' else options['selected_family'],
            configurations={CONFIGURATION_NAMES[k]:dict(description=CONFIGURATION_DESCRIPTIONS[k])
                            for k in options['configurations']},
            recent_backend_outcomes=failures,
            semantics='preferences_are_soft_camera_task_and_backend_safety_remain_primary')
    # Semantic cards normally arrive prebuilt. Also protect direct uses of the
    # client/compactor with raw candidate cards (e.g. debugging and replay).
    if 'candidates' in out:
        from .jev import semantic_card
        out['candidates'] = [semantic_card(f'c{i}',card) if 'trajectory' in card else card
                             for i,card in enumerate(out['candidates'])]
    return out


def bound_geometry_preview(state, byte_budget=11000):
    """Bound optional previews; never truncate intent, candidates or exact requests.

    Byte size is a conservative heuristic for this local text service, not a
    tokenizer promise. Full time-indexed geometry remains in candidate journals.
    """
    import json
    if not state.get('robot_geometry'):
        return state
    def size():
        return len(json.dumps(state,ensure_ascii=False,separators=(',',':')).encode())
    if size() <= byte_budget:
        return state
    for card in state.get('candidates',[]):
        geometry=card.get('geometry')
        if geometry and len(geometry['predicted_geometry'])>1:
            geometry['predicted_geometry']=geometry['predicted_geometry'][-1:]
            geometry['preview_scope']='endpoint_only_full_timed_geometry_in_journal'
    if size() > byte_budget:
        comparison=state.get('subject_state_comparison',{})
        state['subject_state_comparison']={k:v for k,v in comparison.items() if k in (
            'errors','measurement_status','immutable_terminal_target','observation_stamp_s','actual_measurement')}
        state['execution_history']=state.get('execution_history',[])[-1:]
        # Keep all failure kinds and measurements, referencing the existing
        # metric groups instead of repeating them inside each violation.
        def deduplicate(value):
            if isinstance(value,dict):
                if 'intent_complete' in value:
                    for violation in value.get('intent_violations',[]):
                        group={'terminal_frame_mismatch':'terminal_frame',
                               'motion_trend_violation':'motion_trend'}.get(violation.get('kind'))
                        measured=violation.get('measurements')
                        if group and measured and all(value.get(group,{}).get(k)==v for k,v in measured.items()):
                            violation.pop('measurements')
                            violation['measurements_ref']=group
                    value.pop('endpoint_state',None)
                for child in value.values():deduplicate(child)
            elif isinstance(value,list):
                for child in value:deduplicate(child)
        deduplicate(state)
        geometry=state.get('robot_geometry',{})
        for part in geometry.get('parts',{}).values():
            if part.get('regions'):part.pop('bounds',None)
        obstacles=geometry.get('obstacles',{})
        nearest={}
        for relation in obstacles.get('relations',[]):
            key=(relation['part'],relation['obstacle_id'])
            if key not in nearest or relation['proxy_clearance_m']<nearest[key]['proxy_clearance_m']:
                nearest[key]=relation
        if nearest:
            obstacles['relations']=list(nearest.values())
            obstacles['text_scope']='nearest_proxy_per_observed_track_and_part'
    if size()>byte_budget:
        for card in state.get('candidates',[]):
            geometry=card.get('geometry')
            if geometry:
                geometry['predicted_geometry']=[]
                geometry['preview_scope']='omitted_for_text_budget_full_timed_geometry_in_journal'
    return state
