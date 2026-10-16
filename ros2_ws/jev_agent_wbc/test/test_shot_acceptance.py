"""User requirements, independent failures and immutable legacy diagnostics."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.reference import OriginalReference, look_at
from jev_agent_wbc.shot_acceptance import evaluate_shot, occlusion_diagnostics, attach_acceptance

ROOT=Path(__file__).resolve().parents[1]/'scenarios'


def fixture(name='truck_left'):
    scene=json.loads((ROOT/name/'runtime_scene.json').read_text())
    original=OriginalReference(ROOT/name/'task.json',scene['subject']['position'])
    t=np.arange(421)*.02
    p=np.array([original.sample(float(x))[0] for x in t])
    r=np.array([original.sample(float(x))[1] for x in t])
    v=[dict(visibility=1.,in_frame=1.) for _ in t]
    return original,scene['subject'],t,p,r,v,np.minimum(t,original.duration)


def evaluate(data,**kwargs):
    return evaluate_shot(*data,**dict(dict(safety_pass=True),**kwargs))


@pytest.mark.parametrize('name',['orbit','movein','truck_left','truck_right','pedestal_up','pedestal_down'])
def test_reference_acceptance(name):
    result=evaluate(fixture(name))
    assert result['accepted'],result['failed_checks']


def test_occlusion_is_reported_without_turning_an_acceptable_shot_into_failure():
    data=list(fixture())
    data[5]=[dict(visibility=0. if 2.<t<6. else 1.,in_frame=1.) for t in data[2]]
    result=evaluate(data)
    assert result['accepted']
    assert not result['diagnostics']['reference_fidelity']['legacy_intent']['intent_complete']
    d=result['diagnostics']['occlusion']
    assert d['max_occluded_fraction']==1.
    assert d['longest_severe_occlusion_s']==pytest.approx(4.)
    assert d['severe_occlusion_intervals']==[dict(start_s=2.,end_s=6.,duration_s=4.)]


def test_unknown_visibility_is_not_fabricated_clear_visibility():
    data=list(fixture());data[5]=[dict(visibility=None,in_frame=None) for _ in data[2]]
    result=evaluate(data)
    assert result['accepted']
    d=result['diagnostics']['occlusion']
    assert d['visibility_min'] is None and d['unknown_samples']==421
    assert d['unknown_interval_s']==pytest.approx(8.4)


def test_terminal_band_replaces_exact_reference_distance_and_clock_endpoint():
    data=list(fixture())
    original=data[0];p0=data[3][0].copy()
    data[3]=p0+.942*(data[3]-p0)
    data[4]=np.array([look_at(p,original.subject) for p in data[3]])
    ref_t=np.linspace(0.,8.,401);ref_y=original.translation(ref_t)[:,1]
    data[6]=np.interp(data[3][:,1],ref_y,ref_t)
    result=evaluate(data)
    assert result['accepted'] and result['motion_complete']
    old=result['diagnostics']['reference_fidelity']['legacy_intent']
    assert not old['phase_order_valid'] and not old['geometric_complete']
    assert old['segments'][0]['end_error_fraction']>.035


def test_large_regular_detour_is_not_failed_for_reference_process_offset():
    data=list(fixture());u=np.minimum(data[2]/8.,1.)
    data[3][:,0]-=.36*16*u**2*(1-u)**2
    data[4]=np.array([look_at(p,data[0].subject) for p in data[3]])
    result=evaluate(data)
    assert result['accepted'],result['failed_checks']
    old=result['diagnostics']['reference_fidelity']['legacy_intent']
    assert old['motion_trend']['companion_displacement_max_m']>.24
    assert old['motion_trend']['companion_travel_m']>.52
    assert not old['intent_complete']


def test_safety_failure_and_unknown_safety_are_not_hidden_by_good_camera_geometry():
    failed=evaluate(fixture(),safety_pass=False)
    assert failed['failed_checks']==['safety'] and not failed['accepted']
    unknown=evaluate(fixture(),safety_pass=None)
    assert unknown['status']=='unverified' and unknown['unverified_checks']==['safety']


def test_out_of_frame_is_still_failure_even_when_final_frame_recovers():
    data=list(fixture())
    data[4][(data[2]>2)&(data[2]<3)]=Rotation.from_euler('z',np.pi).as_matrix()
    result=evaluate(data)
    assert result['checks']['terminal_frame']['pass_check']
    assert 'composition' in result['failed_checks']


def test_terminal_mismatch_still_fails():
    data=list(fixture());data[3][:,2]+=.2
    data[4]=np.array([look_at(p,data[0].subject) for p in data[3]])
    result=evaluate(data)
    assert 'terminal_frame' in result['failed_checks']


def test_reversal_remains_failure_with_monotone_reported_clock_and_valid_endpoint():
    data=list(fixture());u=np.minimum(data[2]/8.,1.)
    data[3][:,1]+=.3*np.sin(2*np.pi*u)
    data[4]=np.array([look_at(p,data[0].subject) for p in data[3]])
    result=evaluate(data)
    assert 'direction' in result['failed_checks']
    assert result['checks']['terminal_frame']['pass_check']


def test_midshot_wiggles_are_not_excused_by_endpoint_or_direction():
    data=list(fixture());u=np.minimum(data[2]/8.,1.)
    data[3][:,0]+=.04*np.sin(8*np.pi*u)*np.sin(np.pi*u)**2
    data[4]=np.array([look_at(p,data[0].subject) for p in data[3]])
    result=evaluate(data)
    assert 'natural_curve' in result['failed_checks']
    assert result['checks']['direction']['pass_check']
    assert result['checks']['terminal_frame']['pass_check']


def test_stationary_terminal_pose_cannot_fake_motion_completion():
    data=list(fixture());data[3][:]=data[3][-1];data[4][:]=data[4][-1]
    result=evaluate(data)
    assert 'direction' in result['failed_checks'] and not result['motion_complete']


def test_frame_truncation_is_not_counted_as_obstacle_occlusion():
    result=occlusion_diagnostics([0.,1.,2.],[dict(visibility=1.,in_frame=0.)]*3)
    assert result['severe_occlusion_total_s']==0.
    assert result['partial_occlusion_total_s']==0.


def test_future_summary_preserves_legacy_results_and_publishes_new_acceptance():
    summary=dict(intent_success=False,within_budget_success=False,time_overrun_s=.4)
    attach_acceptance(summary,evaluate(fixture()))
    assert summary['intent_success'] and summary['within_budget_success']
    assert summary['legacy_reference_intent_success'] is False
    assert summary['legacy_within_budget_success'] is False


def test_malformed_samples_do_not_silently_pass():
    data=list(fixture());data[2][30]=data[2][29]
    with pytest.raises(ValueError):evaluate(data)
