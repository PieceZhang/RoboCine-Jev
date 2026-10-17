"""PhysX-only reporting policy requested by the user on 2026-10-17.

Acceleration excess is diagnostic. A trial with only network timeouts and no
model response is unverified and excluded from the failure denominator. Small
final camera speeds and solver infeasibility confined to the end of the shot
are not failures. Commands, solver behavior and CPU acceptance stay unchanged.
"""
from copy import deepcopy
import math


POLICY_VERSION='physx_user_acceptance/2026-10-17-terminal-v2'
ACCELERATION_KEYS=('acceleration_violation','wheel_acceleration_violation')
PHYSICAL_THRESHOLDS=dict(position_violation=.001,velocity_violation=.001,
                         wheel_speed_violation=.001,lateral_velocity=.005)
TERMINAL_SPEED_LIMIT_M_S=.10
SOLVER_END_GRACE_S=.10
SOLVER_INFEASIBILITY_MESSAGES=(
    'tpr hard constraint intersection is empty',
    'tpr task-conditioned safety set empty',
    'tpr hard safety constraints infeasible')


def is_network_timeout(record):
    message=str(record.get('error','')).lower()
    return (('timed out' in message or 'timeout' in message)
            and any(token in message for token in ('urlerror','timeouterror','ssl','socket','connection')))


def _gate(value,**evidence):
    return dict(status='unverified' if value is None else 'pass' if value else 'fail',
                pass_check=value,**evidence)


def _finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def _final_camera_speed(summary,rows):
    if len(rows)>=2:
        previous,current=rows[-2:]
        t0,t1=previous.get('t'),current.get('t')
        p0,p1=previous.get('camera_position'),current.get('camera_position')
        if (_finite(t0) and _finite(t1) and t1>t0
            and isinstance(p0,(list,tuple)) and isinstance(p1,(list,tuple))
            and len(p0)==len(p1)==3 and all(_finite(v) for v in (*p0,*p1))):
            return math.dist(p0,p1)/(t1-t0),'last_two_measured_camera_positions'
    speed=summary.get('terminal_camera_speed_m_s')
    if _finite(speed) and speed>=0.:
        return speed,'recorded_terminal_camera_speed_m_s'
    return None,'missing_final_camera_speed'


def _terminal_gate(summary,rows,gate,protocol):
    result=deepcopy(gate)
    measured=result.get('measurements')
    if not isinstance(measured,dict):
        # A previously accepted strict gate already implies the relaxed speed
        # limit; do not invent missing frame measurements to relax a failure.
        return result
    speed,source=_final_camera_speed(summary,rows)
    limits=protocol.get('terminal',{})
    defaults=dict(centroid_error=.02,log_area_error=.05,bbox_log_error=.06,
                  view_angle_error_rad=math.radians(3.),
                  silhouette_axis_error_rad=math.radians(3.))
    geometry=[]
    target=measured.get('target') or {}
    for key,default in defaults.items():
        if key=='silhouette_axis_error_rad' and target.get('silhouette_axis_rad') is None:
            continue
        value=measured.get(key)
        geometry.append(None if not _finite(value) else value<=limits.get(key,default))
    duration=measured.get('confirmed_window_s')
    geometry.append(None if not _finite(duration) else duration>=limits.get('hold_s',.4)-.051)
    speed_pass=None if speed is None else speed<=TERMINAL_SPEED_LIMIT_M_S+1e-9
    evidence=geometry+[speed_pass]
    accepted=False if False in evidence else None if None in evidence else True
    measured.update(final_speed_m_s=speed,final_speed_source=source,
        final_speed_limit_m_s=TERMINAL_SPEED_LIMIT_M_S,pass_check=accepted,
        window_max_speed_role='diagnostic_only; final measured speed decides speed acceptance')
    result.update(status='unverified' if accepted is None else 'pass' if accepted else 'fail',
        pass_check=accepted,measurements=measured,
        reason='missing_terminal_evidence' if accepted is None else None)
    return result


def _terminal_solver_warning(summary,rows):
    message=str(summary.get('failure','')).lower()
    recognized=any(token in message for token in SOLVER_INFEASIBILITY_MESSAGES)
    end=summary.get('original_duration_s')
    last=rows[-1].get('t') if rows else summary.get('elapsed_shot_s')
    near_end=bool(_finite(end) and end>0. and _finite(last)
                  and end-SOLVER_END_GRACE_S-1e-8<=last<=end+.4+1e-8)
    return dict(ignored_as_failure=bool(recognized and near_end),
        recognized_solver_infeasibility=recognized,near_shot_end=near_end,
        last_recorded_time_s=last,planned_motion_end_s=end,
        end_grace_s=SOLVER_END_GRACE_S,recorded_failure=summary.get('failure',''))


def assess_physx(summary,rows,api_records):
    """Return a derived summary; source summaries and raw audits are immutable."""
    if summary.get('backend')!='isaac_physx':
        raise ValueError('the user exclusions apply to PhysX recordings only')
    if summary.get('physx_reporting_policy',{}).get('version')==POLICY_VERSION:
        return deepcopy(summary)
    result=deepcopy(summary)
    result['recorded_acceptance_before_exclusions']=deepcopy(summary.get('recorded_acceptance_before_exclusions')) or {
        key:deepcopy(summary.get(key)) for key in
        ('shot_acceptance','intent_success','within_budget_success','safety_pass')}
    physical=summary.get('physical_audit') or {}
    maxima=physical.get('maxima') or {}
    physical_checks={}
    for key,threshold in PHYSICAL_THRESHOLDS.items():
        value=maxima.get(key)
        known=value is not None and math.isfinite(float(value)) and physical.get('samples',0)>=2
        physical_checks[key]=_gate(None if not known else float(value)<=threshold,
            maximum=value,threshold=threshold)
    geometric_checks={
        'external_clearance':_gate(None if not rows else
            all(float(row['truth_clearance_m'])>=.025 for row in rows),threshold_m=.025),
        'joint_margin':_gate(None if not rows else
            all(float(row['info']['joint_margin_rad'])>=0. for row in rows),threshold_rad=0.),
        'command_constraint_residual':_gate(None if not rows else
            all(float(row['info'].get('qp_violation',0.))<=1e-6 for row in rows),threshold=1e-6),
        'external_contacts':_gate(None if summary.get('contacts') is None else
            int(summary['contacts'])==0,recorded_contacts=summary.get('contacts'))}
    values=[c['pass_check'] for c in (*physical_checks.values(),*geometric_checks.values())]
    safety=False if False in values else None if None in values else True
    timeouts=[record for record in api_records if is_network_timeout(record)]
    responses=[record for record in api_records if record.get('response')]
    network_invalid=bool(api_records and not responses and len(timeouts)==len(api_records))
    policy=dict(version=POLICY_VERSION,
        acceleration_excluded_from_failure=True,
        diagnostic_acceleration_maxima={key:maxima.get(key) for key in ACCELERATION_KEYS},
        original_raw_constraints_pass=physical.get('pass_constraints'),
        physical_checks_without_acceleration=physical_checks,geometric_checks=geometric_checks,
        network_timeout_calls=len(timeouts),model_response_calls=len(responses),
        network_invalid_trial=network_invalid,included_in_failure_denominator=not network_invalid,
        network_rule='all calls are transport timeouts and no model response: unverified, not failed',
        terminal_speed_limit_m_s=TERMINAL_SPEED_LIMIT_M_S,
        terminal_speed_rule='last measured camera speed; nonzero speed is permitted',
        terminal_solver_warning=_terminal_solver_warning(summary,rows),
        commands_and_limits_modified=False)
    acceptance=deepcopy(summary.get('shot_acceptance'))
    if acceptance is None:
        acceptance=dict(checks={},failed_checks=[],unverified_checks=['missing_shot_acceptance'])
    acceptance.setdefault('protocol',{})['physx_reporting_policy']=POLICY_VERSION
    acceptance['checks']['safety']=_gate(safety,
        source='recorded geometry, contacts and raw physical checks excluding acceleration',
        physics_checks=physical_checks,geometry_checks=geometric_checks)
    if 'terminal_frame' in acceptance['checks']:
        acceptance['checks']['terminal_frame']=_terminal_gate(
            summary,rows,acceptance['checks']['terminal_frame'],acceptance['protocol'])
    acceptance['protocol'].setdefault('terminal',{}).update(
        speed_m_s=TERMINAL_SPEED_LIMIT_M_S,
        speed_measurement='last measured camera speed; terminal-window maximum is diagnostic')
    failed=[key for key,value in acceptance['checks'].items() if value['status']=='fail']
    unknown=[key for key,value in acceptance['checks'].items() if value['status']=='unverified']
    if summary.get('failure') and not policy['terminal_solver_warning']['ignored_as_failure']:
        failed.append('execution_error')
    if 'missing_shot_acceptance' in acceptance.get('unverified_checks',[]):
        unknown.append('missing_shot_acceptance')
    if network_invalid:
        acceptance['network_invalid_prefix_diagnostics']=dict(
            failed_checks=failed,unverified_checks=unknown,checks=deepcopy(acceptance['checks']))
        acceptance['checks']={key:_gate(None,reason='network_invalid_trial',
            recorded_prefix_status=value['status']) for key,value in acceptance['checks'].items()}
        failed=[];unknown=['network_connection']
    acceptance.update(failed_checks=failed,unverified_checks=unknown,
        status='fail' if failed else 'unverified' if unknown else 'pass',
        accepted=not failed and not unknown,
        motion_complete=bool(acceptance['checks'].get('direction',{}).get('pass_check')
                             and acceptance['checks'].get('terminal_frame',{}).get('pass_check')))
    result.update(physx_reporting_policy=policy,shot_acceptance=acceptance,
        safety_pass=safety,intent_success=acceptance['accepted'],
        within_budget_success=bool(acceptance['accepted'] and summary.get('time_overrun_s',0.)<=.400001))
    if policy['terminal_solver_warning']['ignored_as_failure'] and acceptance['accepted']:
        # Completion under the user's reporting rule, not an invented control
        # interval. Keep the original stop, failure and samples for inspection.
        result['recorded_execution_completed']=summary.get('recorded_execution_completed',summary.get('completed'))
        result['completed']=True
        policy['completion_basis']='accepted_shot_with_terminal_solver_warning'
    return result
