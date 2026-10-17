"""User-requested PhysX exclusions never modify execution or CPU rules."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from physx_acceptance import assess_physx, is_network_timeout


def fixture():
    checks={key:dict(status='pass',pass_check=True) for key in
            ('safety','direction','natural_curve','composition','terminal_frame')}
    checks['safety']=dict(status='fail',pass_check=False)
    summary=dict(backend='isaac_physx',contacts=0,failure='',time_overrun_s=0.,
        safety_pass=False,intent_success=False,within_budget_success=False,
        physical_audit=dict(samples=200,pass_constraints=False,maxima=dict(
            acceleration_violation=5.,wheel_acceleration_violation=10.,
            position_violation=0.,velocity_violation=0.,wheel_speed_violation=0.,lateral_velocity=0.)),
        shot_acceptance=dict(protocol=dict(version='shot_acceptance/v1'),checks=checks,
            accepted=False,status='fail',failed_checks=['safety'],unverified_checks=[]))
    rows=[dict(truth_clearance_m=.04,info=dict(joint_margin_rad=.1,qp_violation=0.))]
    return summary,rows


def test_acceleration_is_diagnostic_without_mutating_raw_evidence():
    summary,rows=fixture();before=copy.deepcopy((summary,rows))
    result=assess_physx(summary,rows,[dict(response=dict(model='jev-1.14.0'))])
    assert result['intent_success'] and result['safety_pass']
    assert result['physical_audit']['pass_constraints'] is False
    assert result['physical_audit']['maxima']['acceleration_violation']==5.
    assert (summary,rows)==before


@pytest.mark.parametrize('key,value',[('position_violation',.002),('velocity_violation',.002),
                                    ('wheel_speed_violation',.002),('lateral_velocity',.006)])
def test_other_physical_violations_still_fail(key,value):
    summary,rows=fixture();summary['physical_audit']['maxima'][key]=value
    result=assess_physx(summary,rows,[])
    assert not result['safety_pass'] and 'safety' in result['shot_acceptance']['failed_checks']


def test_network_only_trial_is_unverified_and_not_counted_as_failure():
    summary,rows=fixture();summary['failure']='TPR hard constraint intersection is empty'
    calls=[dict(error='URLError: <urlopen error _ssl.c:999: The handshake operation timed out>')]
    result=assess_physx(summary,rows,calls)
    assert result['shot_acceptance']['status']=='unverified'
    assert result['shot_acceptance']['failed_checks']==[]
    assert result['physx_reporting_policy']['included_in_failure_denominator'] is False
    assert result['failure']==summary['failure']


def test_successful_response_with_interrupted_execution_is_still_failure():
    summary,rows=fixture();summary['failure']='TPR hard constraint intersection is empty'
    calls=[dict(error='URLError: connection timed out'),dict(response=dict(model='jev-1.14.0'))]
    result=assess_physx(summary,rows,calls)
    assert result['physx_reporting_policy']['included_in_failure_denominator']
    assert result['shot_acceptance']['failed_checks']==['execution_error']


def test_semantic_protocol_error_is_not_a_network_timeout():
    assert not is_network_timeout(dict(error='JevError: configuration: option probabilities do not sum to one'))
    summary,rows=fixture();summary['shot_acceptance']['checks']['natural_curve']['status']='fail'
    result=assess_physx(summary,rows,[])
    assert result['shot_acceptance']['failed_checks']==['natural_curve']


def test_missing_physical_evidence_is_not_a_pass():
    summary,rows=fixture();summary['physical_audit']['maxima'].pop('velocity_violation')
    result=assess_physx(summary,rows,[])
    assert result['shot_acceptance']['status']=='unverified'
    assert result['safety_pass'] is None


def test_cpu_summary_is_rejected_without_changes():
    summary,rows=fixture();summary['backend']='kinematic_integration'
    with pytest.raises(ValueError):assess_physx(summary,rows,[])


def test_recorded_new_policy_is_idempotent_during_video_rendering():
    summary,rows=fixture();calls=[dict(response=dict(model='jev-1.14.0'))]
    result=assess_physx(summary,rows,calls)
    assert assess_physx(result,rows,calls)==result
    assert result['recorded_acceptance_before_exclusions']['safety_pass'] is False


def terminal_fixture(speed=.001,stop=7.96):
    summary,rows=fixture()
    summary.update(original_duration_s=8.,elapsed_shot_s=stop,completed=False,
        terminal_camera_speed_m_s=speed,
        failure='RuntimeError: TPR hard constraint intersection is empty')
    terminal=dict(centroid_error=.001,log_area_error=.01,bbox_log_error=.01,
        view_angle_error_rad=.01,silhouette_axis_error_rad=.001,
        confirmed_window_s=.4,max_speed_m_s=.2,pass_check=False,
        target=dict(silhouette_axis_rad=1.57))
    summary['shot_acceptance']['checks']['terminal_frame']=dict(
        status='fail',pass_check=False,measurements=terminal)
    rows=[dict(rows[0],t=stop-.02,camera_position=[0.,0.,0.]),
          dict(rows[0],t=stop,camera_position=[speed*.02,0.,0.])]
    calls=[dict(response=dict(model='jev-1.14.0'))]
    return summary,rows,calls


@pytest.mark.parametrize('speed',[0.,.001,.05,.10])
def test_small_nonzero_end_speed_and_terminal_solver_infeasibility_are_not_failures(speed):
    summary,rows,calls=terminal_fixture(speed)
    original=copy.deepcopy((summary,rows,calls))
    result=assess_physx(summary,rows,calls)
    assert result['intent_success'] and result['completed']
    assert result['recorded_execution_completed'] is False
    assert result['failure']==summary['failure']
    assert result['shot_acceptance']['checks']['terminal_frame']['measurements']['max_speed_m_s']==.2
    assert result['physx_reporting_policy']['terminal_solver_warning']['ignored_as_failure']
    assert assess_physx(result,rows,calls)==result
    assert (summary,rows,calls)==original


def test_large_final_speed_still_fails_even_when_last_solver_error_is_ignored():
    summary,rows,calls=terminal_fixture(.2)
    result=assess_physx(summary,rows,calls)
    assert result['shot_acceptance']['failed_checks']==['terminal_frame']
    assert not result['intent_success'] and not result['completed']


def test_early_solver_infeasibility_is_still_an_execution_failure():
    summary,rows,calls=terminal_fixture(stop=6.)
    result=assess_physx(summary,rows,calls)
    assert result['shot_acceptance']['failed_checks']==['execution_error']
    assert not result['physx_reporting_policy']['terminal_solver_warning']['ignored_as_failure']


@pytest.mark.parametrize('stop',[7.92,7.96,8.,8.4])
def test_solver_warning_applies_to_shot_tail_and_terminal_hold(stop):
    summary,rows,calls=terminal_fixture(stop=stop)
    assert assess_physx(summary,rows,calls)['intent_success']


def test_end_frame_outside_band_is_not_excused_by_terminal_solver_warning():
    summary,rows,calls=terminal_fixture()
    summary['shot_acceptance']['checks']['terminal_frame']['measurements']['centroid_error']=.03
    result=assess_physx(summary,rows,calls)
    assert result['shot_acceptance']['failed_checks']==['terminal_frame']


def test_unrelated_end_exception_is_not_excused_as_solver_infeasibility():
    summary,rows,calls=terminal_fixture()
    summary['failure']='RuntimeError: RGB recorder failed'
    assert assess_physx(summary,rows,calls)['shot_acceptance']['failed_checks']==['execution_error']


def test_final_speed_without_measured_evidence_is_unverified():
    summary,rows,calls=terminal_fixture()
    summary.pop('terminal_camera_speed_m_s')
    for row in rows:row.pop('camera_position')
    result=assess_physx(summary,rows,calls)
    assert result['shot_acceptance']['status']=='unverified'
    assert result['shot_acceptance']['unverified_checks']==['terminal_frame']


def test_previous_policy_can_be_upgraded_without_losing_original_acceptance():
    summary,rows,calls=terminal_fixture()
    recorded=dict(safety_pass=False,intent_success=False,shot_acceptance=dict(status='fail'))
    summary.update(physx_reporting_policy=dict(version='physx_user_acceptance/2026-10-17'),
                   recorded_acceptance_before_exclusions=copy.deepcopy(recorded))
    result=assess_physx(summary,rows,calls)
    assert result['intent_success']
    assert result['recorded_acceptance_before_exclusions']==recorded
