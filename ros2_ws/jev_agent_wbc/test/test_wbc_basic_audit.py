"""Basic qualification never upgrades precision, sensing or unsafe trials."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('basic_audit', SCRIPTS/'audit_wbc_basic.py')
BASIC = importlib.util.module_from_spec(spec); spec.loader.exec_module(BASIC)


def fixture():
    physical = dict(audit_passed=True, run_complete=True, timing_pass=True,
        native_reported_safety_failure=False, constraint_failure_samples={},
        physical_acceptance_pass=False, tasks=[])
    summary = dict(robot='go2', fine_collision=dict(all_active_samples_fresh_pass=True))
    tasks, rows = [], []
    commands = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(1,1,0),(1,-1,0),(0,0,1),(0,0,-1),(1,0,0)]
    names = ['forward','backward','left','right','diagonal_left','diagonal_right','yaw_left','yaw_right','camera_shot']
    for index,(name,command) in enumerate(zip(names,commands)):
        task_id = str(index); start = index*10.
        tasks.append(dict(task_id=task_id, camera_trajectory=dict(start_s=start, samples=[dict(t_s=5.)])))
        stop = {key:cap*.5 for key,cap in BASIC.STOP.items() if key != 'all_revolute_joint_span_rad'}
        stop.update(joint_span=[.025]*18, measured_window_s=.5, passed=False)
        physical['tasks'].append(dict(task_id=task_id, strict_stop=stop, camera_tracking_pass=True))
        for stamp in [start+1., start+4.]:
            rows.append(dict(stamp_s=stamp, case=name, task_feedback=dict(task_id=task_id),
                requested_body_velocity=[x*.1 for x in command],
                measured_body_velocity=[x*.04 for x in command]))
    return physical, dict(passed=True), dict(passed=True), summary, tasks, rows


def test_basic_direction_and_coarse_stop_do_not_claim_precision_or_avoidance():
    result = BASIC.qualify(*fixture())
    assert result['basic_control_pass'] and len(result['cases']) == 9
    assert result['original_precision_stop_pass_count'] == 0
    for key in ('original_precision_physical_pass', 'precision_velocity_tracking_qualified',
                'environment_constraints_verified', 'sensors_avoidance_verified',
                'runtime_recovery_wall_bound_certified'):
        assert result[key] is False


@pytest.mark.parametrize('bad', ['incomplete','physical_error','physical_limit','feedback',
    'collision','abort','pace','camera','direction','empty_request','missing_direction',
    'joint','camera_stop','linear_stop','angular_stop','incomplete_window','unsupported'])
def test_basic_never_excuses_hard_failure_or_missing_coverage(bad):
    args = list(fixture()); physical,limits,feedback,summary,tasks,rows = args
    if bad == 'incomplete': physical['run_complete'] = False
    elif bad == 'physical_error': physical['audit_passed'] = False
    elif bad == 'physical_limit': limits['passed'] = False
    elif bad == 'feedback': feedback['passed'] = False
    elif bad == 'collision': summary['fine_collision']['all_active_samples_fresh_pass'] = False
    elif bad == 'abort': physical['native_reported_safety_failure'] = True
    elif bad == 'pace': physical['timing_pass'] = False
    elif bad == 'camera': physical['tasks'][0]['camera_tracking_pass'] = False
    elif bad == 'direction': rows[1]['measured_body_velocity'] = [-.04,0.,0.]
    elif bad == 'empty_request':
        for r in rows[:2]: r['requested_body_velocity'] = [0.,0.,0.]
    elif bad == 'missing_direction': tasks.pop()
    elif bad == 'joint': physical['tasks'][0]['strict_stop']['joint_span'][0] = .051
    elif bad == 'camera_stop': physical['tasks'][0]['strict_stop']['camera_position_span_m'] = .021
    elif bad == 'linear_stop': physical['tasks'][0]['strict_stop']['final_linear_speed_mps'] = .031
    elif bad == 'angular_stop': physical['tasks'][0]['strict_stop']['final_angular_speed_radps'] = .051
    elif bad == 'incomplete_window': physical['tasks'][0]['strict_stop']['measured_window_s'] = .49
    else: rows[0]['task_feedback'].update(feasible=False, reason='unsupported_request')
    assert not BASIC.qualify(*args)['basic_control_pass']


def test_transient_failure_stays_public_and_is_not_a_wall_recovery_certificate():
    args = list(fixture()); args[-1][0]['task_feedback'].update(feasible=False, reason='solver_failure')
    result = BASIC.qualify(*args)
    assert result['basic_control_pass'] and result['transient_module_failure_samples'] == 1
    assert result['runtime_recovery_wall_bound_certified'] is False
