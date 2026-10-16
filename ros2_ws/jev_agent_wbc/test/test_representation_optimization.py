"""Task timing, recoverable camera geometry, and private backend feedback."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from jev_agent_wbc.representation import TaskContext
from jev_agent_wbc.representation_agent import search_outcome
from jev_agent_wbc.intent_paths import IntentPath
from jev_agent_wbc.candidates import path_spec,restore_path
from jev_agent_wbc.camera_intent import projected_area
from test_camera_intent import fixture


def test_task_context_roundtrip_and_unchangeable_prefix():
    context=TaskContext('immutable',.6,8.,1.8,1.2,True)
    value=context.to_dict()
    assert value['unchanged_prefix_at_risk'] is True
    assert TaskContext.from_dict(json.loads(json.dumps(value)))==context
    value['unchanged_prefix_at_risk']=False
    with pytest.raises(ValueError):TaskContext.from_dict(value)
    with pytest.raises(ValueError):TaskContext('id',2.,8.,1.)
    assert TaskContext('id',7.,8.,9.).to_dict()['earliest_change_s']>8.


def test_task_context_json_schema():
    jsonschema=pytest.importorskip('jsonschema')
    schema=json.loads((Path(__file__).resolve().parents[1]/'schemas/geometry-v1.schema.json').read_text())
    jsonschema.validate(TaskContext('reference',.5,8.,1.7).to_dict(),schema)


def test_search_feedback_retains_unknown_search_and_failure_time_without_internals():
    result=search_outcome(dict(time_s=1.,activation_s=2.,untested_proposals=15,search_exhausted=False,
        reasons={'kinematic_gate:kinematic_tracking_unreachable':1},
        failures=[dict(reason='kinematic_gate:kinematic_tracking_unreachable',first_failure_s=1.8,
                       program=dict(route='above',phase_rate=.5,numerical_model='joint6'))]))
    assert result['untested_proposals']==15 and not result['search_exhausted']
    assert result['failures'][0]['failure_before_change'] is True
    assert result['failures'][0]['status']=='camera_path_not_realized'
    assert 'joint' not in json.dumps(result) and 'kinematic' not in json.dumps(result)


@pytest.mark.parametrize('rate,split',[(1.8,.25),(.9,.35),(-.9,.65),(-.9,.75),(.9,.5)])
def test_asymmetric_clock_keeps_deadline_and_strict_progress(rate,split):
    original,scene,_=fixture()
    path=IntentPath(original,original,1.,dict(route='timing',amplitude_m=0.,extension_s=0.,
        phase_rate=rate,phase_split=split),scene['subject_model'])
    times=np.linspace(1.,8.,1001)
    progress=np.array([path.parent_time(t) for t in times])
    assert progress[0]==pytest.approx(1.) and progress[-1]==pytest.approx(8.)
    assert np.min(np.diff(progress))>0
    assert (path.parent_time(1.00001)-1.)/1e-5==pytest.approx(1.,abs=1e-6)
    assert (8.-path.parent_time(7.99999))/1e-5==pytest.approx(1.,abs=1e-6)
    with pytest.raises(ValueError):
        IntentPath(original,original,1.,dict(route='timing',amplitude_m=0.,extension_s=0.,phase_rate=2.,phase_split=.6))


def test_recovery_returns_to_immutable_frame_with_exact_prefix_and_replay():
    original,scene,_=fixture('orbit')
    active=IntentPath(original,original,.5,dict(route='above',amplitude_m=.15,extension_s=0.,profile='hold',rise_s=1.),scene['subject_model'])
    recovered=IntentPath(original,active,2.,dict(route='timing',amplitude_m=0.,extension_s=0.,restore_original=True),scene['subject_model'])
    replay=restore_path(original,path_spec(recovered))
    assert np.linalg.norm(active.sample(8.)[0]-original.sample(8.)[0])>.1
    for t in np.linspace(0.,2.,21):
        np.testing.assert_allclose(recovered.sample(t)[0],active.sample(t)[0],atol=1e-10)
    for i in (0,1):np.testing.assert_allclose(recovered.sample(8.)[i],original.sample(8.)[i],atol=1e-9)
    for t in np.linspace(0.,8.,31):
        for a,b in zip(recovered.sample(t),replay.sample(t)):np.testing.assert_allclose(a,b,atol=1e-10)


@pytest.mark.parametrize('name,route',[('pedestal_up','left'),('pedestal_down','right'),('truck_left','above'),('orbit','above')])
def test_area_compensation_keeps_motion_coordinate(name,route):
    original,scene,spec=fixture(name)
    path=IntentPath(original,original,.5,dict(route=route,amplitude_m=.18,extension_s=0.,profile='arch',
        preserve_projected_area=True),scene['subject_model'])
    poses=[path.sample(float(t)) for t in np.linspace(.5,8.,41)]
    nominal=[original.sample(float(t)) for t in np.linspace(.5,8.,41)]
    segment=spec.segments[0]
    np.testing.assert_allclose(spec.features(segment,[p[0] for p in poses],[p[1] for p in poses]),
        spec.features(segment,[p[0] for p in nominal],[p[1] for p in nominal]),atol=2e-5)
    ratios=[projected_area(p,r,spec.model,spec.intrinsics)/projected_area(op,orr,spec.model,spec.intrinsics)
            for (p,r,_),(op,orr,_) in zip(poses,nominal)]
    assert np.max(np.abs(np.log(ratios)))<.005


def test_verified_body_preparation_can_start_before_camera_change():
    from test_tpr_wbc import controller,STATE
    from jev_agent_wbc.whole_body_qp import whole_body_kinematics
    c=controller();c.trajectory.activation_s=1.
    p,r,j=whole_body_kinematics(STATE)
    desired=(np.eye(8)-np.linalg.pinv(j)@j)@np.array([.03,0.,0.,0.,0.,0.,0.,0.])
    c.kinematic_reference=dict(feasible=True,trace=[dict(t=t,state=STATE.tolist(),control=desired.tolist()) for t in [0.,1.]])
    c.reconfiguration.update=lambda *args:np.zeros(8)
    u,info=c.solve(STATE,p,r,np.zeros(6),.02,commanded_q=STATE[3:],trajectory_time=0.)
    assert info['tpr_report']['kinematic_preference_active']
    assert np.linalg.norm(u)>1e-5
    np.testing.assert_allclose(j@u,0.,atol=1e-8)


def test_compaction_keeps_every_violation_without_repeating_full_subject_geometry():
    from jev_agent_wbc.jev import compact_evidence
    blob=dict(unoccluded_area=.1,view_direction=[1,0,0],landmarks=list(range(300)))
    violation=dict(kind='terminal_frame_mismatch',interval_s=[8.,8.4],recovery='not_verified',
                   measurements=dict(centroid_error=.2,pass_check=False,target=blob,predicted_or_actual=blob))
    intent=dict(intent_complete=False,intent_violations=[violation],terminal_frame=dict(centroid_error=.2,target=blob,predicted_or_actual=blob))
    cleaned=compact_evidence(dict(objective='visible_intent_completion',intent=intent))['intent']
    assert cleaned['intent_violations'][0]['measurements']==dict(centroid_error=.2,pass_check=False)
    assert cleaned['intent_violations'][0]['recovery']=='not_verified'
    assert cleaned['terminal_frame']['centroid_error']==.2
    assert 'landmarks' not in json.dumps(cleaned)


def test_nominal_view_forecast_does_not_relax_collision_or_claim_future_certainty():
    from jev_agent_wbc.floating_camera import intent_prediction,evaluate_path
    original,scene,spec=fixture('orbit')
    scene['radius_rates']=[5.]
    # The observed object is far away; its future uncertainty envelope is huge.
    forecast=intent_prediction(spec,original,scene,0.)
    assert forecast['intent_complete']
    assert not forecast['visibility_forecast']['guarantee']
    assert forecast['visibility_forecast']['max_future_radius_growth_m']>40.
    assert not evaluate_path(original,scene,0.,8.4,.06)[0].feasible
    assert scene['radius_rates']==[5.]
    scene['valid']=False
    assert not intent_prediction(spec,original,scene,0.)['intent_complete']
