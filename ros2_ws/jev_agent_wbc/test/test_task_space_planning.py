"""Measured-state task-space planning must not silently become ideal execution."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from jev_agent_wbc import task_space_planning as planner
from jev_agent_wbc.candidates import Approval, SearchBudget, path_hash, installation_guard
from jev_agent_wbc.contracts import digest
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.whole_body_qp import Limits, WholeBodyQP, whole_body_kinematics

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def setup():
    runtime=json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'scenarios/movein/task.json',runtime['subject']['position'])
    scene=dict(observation_stamp=0.,surfel_centers=[[8.,8.,8.]],surfel_radius=.1,
               surfel_velocities=[[0.,0.,0.]],minimum_clearance_m=.025,subject_model=runtime['subject'])
    return original,np.array(runtime['initial_state']),runtime['base_z'],scene


def test_virtual_prediction_is_seeded_by_measured_pose(setup):
    original,state,z,scene=setup
    state[0]+=.25;state[2]+=.15
    virtual=planner.MeasuredCameraPrediction(original,state,z,0.,1.5)
    p,r,_=whole_body_kinematics(state,z)
    np.testing.assert_allclose(virtual.sample(0.)[0],p,atol=1e-12)
    np.testing.assert_allclose(virtual.sample(0.)[1],r,atol=1e-12)
    np.testing.assert_allclose(virtual.sample(1.5)[0],original.sample(1.5)[0],atol=1e-12)
    np.testing.assert_allclose(virtual.sample(1.5)[1],original.sample(1.5)[1],atol=1e-12)


def test_camera_screen_catches_collision_at_actual_not_nominal_pose(setup):
    original,state,z,scene=setup
    state[0]+=.5
    scene['surfel_centers']=[whole_body_kinematics(state,z)[0].tolist()]
    value,_=planner.prediction(original,scene,state,z,0.,2.,1.5)
    assert not value.feasible and value.min_clearance_m<0


def test_candidate_and_installation_use_measured_feedback_without_wbc_rollout(setup,monkeypatch):
    original,state,z,scene=setup;budget=SearchBudget(activation_lead_s=1.5)
    monkeypatch.setattr(WholeBodyQP,'solve',lambda *a,**k:pytest.fail('WBC prediction invoked'))
    result=planner.build(original,original,'view',scene,state,np.zeros(8),0.,z,Limits(),budget=budget)
    assert not result['error'] and result['candidates']
    candidate=result['candidates'][0]
    approval=Approval(candidate.id,digest(candidate.card()),path_hash(original),original.sha256,None,1)
    checked=planner.verify(candidate,approval,original,scene,state,np.zeros(8),0.,z,Limits(),budget=budget)
    assert not checked['error']
    assert checked['geometry_guidance'].request.camera_path_id == candidate.id
    assert checked['geometry_guidance'].camera_path is candidate.path
    from copy import deepcopy
    tampered = deepcopy(candidate)
    tampered.consequences['geometry']['request']['end_s'] -= .1
    rejected = planner.verify(tampered,approval,original,scene,state,np.zeros(8),0.,z,Limits(),budget=budget)
    assert rejected['error'] == 'approval_binding_mismatch'
    assert installation_guard(checked,original,scene,state,0.,budget,previous=np.zeros(8)) is None
    assert installation_guard(checked,original,scene,state,0.,budget)=='missing_latest_velocity_for_kinematic_gate'
    moved=state.copy();moved[0]+=.5
    assert installation_guard(checked,original,scene,moved,0.,budget)=='camera_left_verified_state_neighborhood'
    stale=planner.verify(candidate,approval,original,scene,state,np.zeros(8),.8,z,Limits(),budget=budget)
    assert stale['error']=='scene_stale'


def test_same_map_motion_uncertainty_and_staleness(setup):
    _,_,_,scene=setup
    scene.update(surfel_track_ids=[1],surfel_ages=[.2])
    value=planner.camera_scene(scene,0.,2.)
    assert value['radii'][0]==pytest.approx(.1+.035*.2)
    assert value['radius_rates'][0]==pytest.approx(.035)
    assert not planner.camera_scene(scene,.7,2.)['valid']


def test_uncertainty_grows_at_prediction_time_not_from_horizon_end():
    from jev_agent_wbc.floating_camera import predicted_clearances
    scene=dict(valid=True,observation_stamp=0.,centers=[[1.,0.,0.]],velocities=[[0.,0.,0.]],
               radii=[.1],radius_rates=[.035])
    values=predicted_clearances([[0.,0.,0.],[0.,0.,0.]],[0.,8.],scene,.06)
    np.testing.assert_allclose(values,[.84,.56],atol=1e-12)


def test_camera_valid_candidates_are_rejected_by_joint_acceleration_gate(setup):
    original,state,z,scene=setup
    limits=Limits(acceleration=np.full(8,.001),wheel_acceleration=.01)
    result=planner.build(original,original,'view',scene,state,np.zeros(8),0.,z,limits,
        budget=SearchBudget(activation_lead_s=1.5),measured_control=np.zeros(8))
    assert result['error'] and not result['candidates']
    assert result['evaluations']
    numeric=[e for e in result['evaluations'] if 'kinematics' in e]
    assert numeric  # cooperative wall budget can stop after the first failed lift
    for evaluation in numeric:
        assert evaluation['physical']['feasible']  # camera-only screening passes
        assert evaluation['rejected']=='kinematic_gate:kinematic_tracking_unreachable'


def test_latest_velocity_is_rechecked_and_jev_does_not_receive_fk_internals(setup):
    from jev_agent_wbc.jev import semantic_card
    original,state,z,scene=setup;budget=SearchBudget(activation_lead_s=1.5)
    result=planner.build(original,original,'view',scene,state,np.zeros(8),0.,z,Limits(),budget=budget)
    candidate=result['candidates'][0]
    assert candidate.card()['kinematic_feasibility']['feasible']
    card=semantic_card('c0',candidate.card())
    assert 'kinematic_feasibility' not in card
    assert not any(k in json.dumps(card) for k in ('effective_jacobian','control_matrix','singular_value_floor'))
    approval=Approval(candidate.id,digest(candidate.card()),path_hash(original),original.sha256,None,1)
    checked=planner.verify(candidate,approval,original,scene,state,np.zeros(8),0.,z,Limits(),
        budget=budget,measured_control=np.full(8,2.))
    assert checked['error']=='kinematic_gate:initial_velocity_limit'
    assert checked['checks']['kinematic_feasibility']=='fail'
