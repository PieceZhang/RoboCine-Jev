"""Subject-state feedback, immutable final framing, and bounded motion trends."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.camera_intent import IntentSpec, POLICY
from jev_agent_wbc.candidates import SearchBudget, path_spec, restore_path, path_hash
from jev_agent_wbc.floating_camera import CAMERA_RAYS, FloatingAgent, camera_candidates, intent_prediction
from jev_agent_wbc.intent_paths import IntentPath
from jev_agent_wbc.perception import mask_evidence, SubjectEvidence
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.subject_state import (frame_state, frame_error, simulated_subject_returns,
    fit_subject_points, SubjectStateObserver, compare_subject_evidence)

ROOT=Path(__file__).resolve().parents[1]/'scenarios'

def fixture(name='movein'):
    model=json.loads((ROOT/name/'runtime_scene.json').read_text())['subject']
    original=OriginalReference(ROOT/name/'task.json',model['position'])
    spec=IntentSpec(original,model)
    scene=dict(valid=True,observation_stamp=0.,centers=[],radii=[],velocities=[],subject_model=model)
    return original,spec,scene


def test_mask_features_are_text_numbers_not_invented_body_pose_or_visibility():
    mask=np.zeros((80,100),bool);mask[10:70,30:50]=True
    v=mask_evidence(mask,'subject',1.,identity_valid=True,confidence=.95)
    assert v['centroid']==pytest.approx([.4,.5])
    assert v['bbox_size']==pytest.approx([.2,.75])
    assert v['silhouette_axis_rad']==pytest.approx(np.pi/2)
    assert v['body_orientation']=='unknown'
    assert v['unoccluded_area'] is None and v['external_visibility_fraction'] is None
    assert not v['scale_measurement_valid']
    json.dumps(v,allow_nan=False)


def test_lidar_fit_uses_surface_geometry_not_partial_cloud_centroid():
    original,spec,_=fixture();p,r,_=original.sample(0.)
    points=simulated_subject_returns(p,CAMERA_RAYS,spec.model,[])
    record=fit_subject_points(points,spec.model,0.)
    assert record['valid'] and record['measured']
    np.testing.assert_allclose(record['position'],original.subject,atol=1e-8)
    assert np.linalg.norm(points.mean(axis=0)-original.subject)>.05
    assert not record['orientation_valid'] and record['body_orientation']=='unknown'
    assert record['position_components'][2]=='calibrated_prior'
    moved=dict(spec.model,position=(original.subject+np.array([.02,.01,0.])).tolist())
    moved_points=simulated_subject_returns(p,CAMERA_RAYS,moved,[])
    measured=fit_subject_points(moved_points,spec.model,.1)
    np.testing.assert_allclose(measured['position'],moved['position'],atol=1e-8)


def test_lidar_occlusion_and_freshness_invalidate_measurement():
    original,spec,_=fixture();p,r,_=original.sample(0.)
    points=simulated_subject_returns(p,CAMERA_RAYS,spec.model,[])
    observer=SubjectStateObserver(spec.model,spec.intrinsics)
    assert observer.observe(points,p,r,1.)['status']=='valid'
    assert observer.snapshot(1.31)['status']=='unknown'
    sphere=dict(center=(p+original.subject)/2,radius=.3)
    hidden=simulated_subject_returns(p,CAMERA_RAYS,spec.model,[sphere])
    assert len(hidden)>0  # foreground obstacle returns are not removed by a truth ID
    assert not fit_subject_points(hidden,spec.model,1.4)['valid']
    assert observer.observe(hidden,p,r,1.4)['status']=='unknown'
    assert not observer.snapshot(1.4)['geometry']['valid']


@pytest.mark.parametrize('timing,extension',[('early_slow',.8),('early_slow',2.4),('early_fast',-.8),('early_fast',-1.6)])
def test_timing_changes_do_not_reverse_or_reset_shot_and_replay_exactly(timing,extension):
    original,spec,scene=fixture('orbit')
    path=IntentPath(original,original,1.5,dict(route='timing',amplitude_m=0.,extension_s=extension,timing_profile=timing),spec.model)
    times=np.linspace(0,path.duration,801)
    assert np.min(np.diff([path.progress_time(t) for t in times]))>=-1e-10
    assert path.progress_time(path.duration)==pytest.approx(original.duration)
    assert intent_prediction(spec,path,scene,0.)['intent_complete']
    replay=restore_path(original,path_spec(path));assert path_hash(replay)==path_hash(path)
    for t in times[::40]:
        np.testing.assert_array_equal(replay.sample(t)[0],path.sample(t)[0])
    for t in [1.5,1.50001]:
        np.testing.assert_allclose(path.sample(t)[0],original.sample(t)[0],atol=1e-8)


def test_small_different_terminal_pose_is_accepted_as_frame_set():
    original,spec,scene=fixture()
    path=IntentPath(original,original,1.5,dict(route='above',amplitude_m=.01,extension_s=0.,profile='hold',rise_s=1.2),spec.model)
    m=intent_prediction(spec,path,scene,0.)
    assert np.linalg.norm(path.sample(8)[0]-original.sample(8)[0])>.005
    assert m['terminal_frame']['pass_check'] and m['intent_complete']


def test_large_midshot_detour_cannot_be_hidden_by_perfect_terminal_frame():
    original,spec,scene=fixture()
    path=IntentPath(original,original,1.5,dict(route='above',amplitude_m=.65,extension_s=0.),spec.model)
    m=intent_prediction(spec,path,scene,0.)
    assert m['terminal_frame']['pass_check'] and m['range_complete']
    assert not m['motion_trend']['pass_check'] and not m['intent_complete']


def test_roll_and_camera_pan_do_not_pass_terminal_frame():
    original,spec,_=fixture()
    p,r,_=original.sample(8.)
    target=spec.reference_frame(8.)
    roll=frame_state(p,r@Rotation.from_euler('x',.15).as_matrix(),spec.model,spec.intrinsics)
    pan=frame_state(p,r@Rotation.from_euler('z',.10).as_matrix(),spec.model,spec.intrinsics)
    assert frame_error(roll,target)['silhouette_axis_error_rad']>POLICY.terminal_silhouette_axis_tolerance_rad
    assert frame_error(pan,target)['centroid_error']>POLICY.terminal_centroid_tolerance


def test_stale_subject_evidence_cannot_authorize_candidate():
    original,spec,scene=fixture()
    scene['subject_observation']=dict(status='unknown')
    result=camera_candidates(original,original,'timing',scene,0.,SearchBudget(),.06)
    assert result['error']=='subject_measurement_unknown_or_stale'


def test_original_terminal_not_replaced_by_active_path_or_observed_subject():
    original,spec,_=fixture()
    active=IntentPath(original,original,1.5,dict(route='above',amplitude_m=.35,extension_s=0.,profile='hold',rise_s=1.2),spec.model)
    state=compare_subject_evidence(original,active,{},spec.model,spec.intrinsics,8.)
    assert state['immutable_terminal_target']==spec.reference_frame(8.)
    assert frame_error(state['candidate_prediction_at_capture'],state['immutable_terminal_target'])['view_angle_error_rad']>.1
    assert not state['terminal_verified']


def test_sam_measurement_mismatch_is_feedback_without_wbc():
    from jev_agent_wbc.engine import CinematicEngine
    original,spec,_=fixture()
    p,r,_=original.sample(4.)
    visual=dict(status='valid',stamp_s=4.,occluded=False,truncated=False,
        centroid=[.7,.5],visible_area=.02,bbox_size=[.1,.2],source='sam2.1_rgb')
    snapshot=dict(visual=visual,diagnosis='observed',scale_measurement_valid=False)
    engine=object.__new__(CinematicEngine)
    engine.original=engine.active=original;engine.subject_model=spec.model;engine.intrinsics=spec.intrinsics
    engine.perception=SimpleNamespace(snapshot=lambda t:snapshot)
    engine.last_visual_stamp=None;engine.active_candidate=SimpleNamespace(consequences=dict(intent={}))
    feedback=[];engine._write=lambda *a,**k:None
    engine._feedback=lambda cause,t,**k:feedback.append(cause)
    engine._monitor_visual(4.1)
    assert feedback==['measured_subject_frame_mismatch']
    comparison=engine._subject_comparison(4.1)
    assert comparison['capture_stamp_s']==4.
    assert comparison['errors']['candidate']['log_area_error'] is None
    assert comparison['body_orientation']=='unknown'


def test_measurement_identity_time_and_calibration_are_checked_again():
    from jev_agent_wbc.subject_state import subject_observation_error
    original,spec,_=fixture()
    obs=dict(subject_id=spec.model['id'],status='valid',stamp_s=1.,
        geometry=dict(valid=True,position=original.subject.tolist()))
    assert not subject_observation_error(obs,original,1.1)
    assert subject_observation_error(obs,original,.9)=='subject_measurement_unknown_or_stale'
    assert subject_observation_error(obs,original,1.31)=='subject_measurement_unknown_or_stale'
    assert subject_observation_error(dict(obs,subject_id='other'),original,1.)=='subject_measurement_identity_mismatch'
    obs['geometry']['position']=(original.subject+np.array([.1,0,0])).tolist()
    assert subject_observation_error(obs,original,1.)=='subject_calibration_mismatch_requires_reconstruction'


def test_lidar_only_three_way_comparison_uses_measured_pose_and_subject():
    original,spec,_=fixture()
    p,r,_=original.sample(4.)
    snapshot=dict(geometry=dict(status='valid',valid=True,stamp_s=4.,position=original.subject.tolist(),source='lidar'))
    feedback=dict(time_s=4.,camera_position=(p+np.array([0.,.1,0.])).tolist(),
        camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist())
    result=compare_subject_evidence(original,original,snapshot,spec.model,spec.intrinsics,4.,feedback)
    assert result['actual_measurement']['valid']
    assert result['errors']['candidate']['centroid_error']>.05
    assert result['immutable_terminal_target']==spec.reference_frame(8.)
    feedback['time_s']=3.
    assert not compare_subject_evidence(original,original,snapshot,spec.model,spec.intrinsics,4.,feedback)['actual_measurement']['valid']


def test_confirmation_window_is_not_shortened_by_monitor_grid_offset():
    original,spec,scene=fixture('truck_right')
    history=[]
    for t in np.arange(0.,2.78,.02):
        p,r,_=original.sample(t)
        history.append(dict(t=float(t),camera_position=p.tolist(),
            camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),progress_s=float(t),
            visual=dict(visibility=1.,in_frame=1.)))
    m=intent_prediction(spec,original,scene,2.78,history,step=.1)
    assert m['terminal_frame']['confirmed_window_s']==pytest.approx(.4)
    assert m['terminal_frame']['pass_check'] and m['intent_complete']
    assert m['camera_limits_pass']
