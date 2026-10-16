"""Safety and intent tradeoffs must remain distinct across the new stages."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.candidates import SearchBudget
from jev_agent_wbc.camera_intent import IntentSpec
from jev_agent_wbc.floating_camera import camera_candidates, evaluate_path
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.kinematic_feasibility import KinematicSearch
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
from jev_agent_wbc.configuration_preparation import preparation, ConfigurationMonitor
from jev_agent_wbc.jev import CandidateDecider, compact_evidence
from jev_agent_wbc.search_configuration import offers, capability, profile

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    runtime=json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'scenarios/movein/task.json',runtime['subject']['position'])
    return original,np.array(runtime['initial_state']),runtime['base_z'],runtime


def test_incomplete_past_coverage_is_retained_for_semantic_tradeoff(fixture):
    original,state,z,runtime=fixture
    scene=dict(valid=True,observation_stamp=1.,centers=[[9.,9.,9.]],radii=[.1],
               velocities=[[0.,0.,0.]],subject_model=runtime['subject'])
    history=[]
    for t in np.arange(0.,1.,.05):
        p,r,_=original.sample(float(t))
        history.append(dict(t=float(t),camera_position=p.tolist(),camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
            progress_s=float(t),visual=dict(visibility=0.,in_frame=1.,projected_area=None)))
    result=camera_candidates(original,original,'view',scene,1.,
        SearchBudget(max_proposals=2,max_rollouts=2,activation_lead_s=.5),.06,history)
    assert result['candidates']
    card=result['candidates'][0].card()
    assert card['physical_valid']
    assert not card['visual']['intent']['intent_complete']
    violations=card['visual']['intent']['intent_violations']
    assert any(v['kind']=='incomplete_visible_intent' for v in violations)
    assert violations[0]['recovery']=='not_verified'
    clean=compact_evidence(dict(objective='visible_intent_completion',intent=card['visual']['intent']))
    assert clean['intent']['intent_violations']


def test_camera_clearance_cannot_hide_base_collision(fixture):
    original,state,z,runtime=fixture
    # A body at the base does not intersect the camera path.
    center=[state[0],state[1],z+.12]
    scene=dict(format='jvs_lidar_surfels/v1',observation_stamp=0.,surfel_centers=[center],
        surfel_radius=.1,surfel_velocities=[[0.,0.,0.]],minimum_clearance_m=.025)
    camera_scene=dict(valid=True,observation_stamp=0.,centers=[center],radii=[.1],velocities=[[0.,0.,0.]])
    assert evaluate_path(original,camera_scene,0.,.5,.06)[0].feasible
    result=KinematicSearch(state,np.zeros(8),0.,z,Limits(),scene=scene).lift(original,.5)
    assert not result['feasible']
    assert result['reason']=='whole_body_collision'
    assert result['failure_class']=='physical_collision'


def test_configuration_is_a_bounded_followup_with_new_numeric_evidence(fixture):
    original,state,z,_=fixture
    search=KinematicSearch(state,np.zeros(8),0.,z,Limits())
    numeric=capability(search,original.subject)
    assert len(numeric['probes'])==6
    options=offers('view',numeric,[dict(reason='unverified_numerical_search')])
    class Client:
        def ask(self,evidence,questions):
            assert evidence['search_options']['robot_capability']==numeric
            assert set(questions['configuration']['criteria'])=={'local','base','timing','explore','executable'}
            assert 'effective_jacobian' not in json.dumps(compact_evidence(evidence))
            return dict(configuration=dict(choice='base'))
    decision=CandidateDecider(Client()).decide(dict(stage='configuration',search_options=options))
    assert decision['strategy']=='view' and decision['search_config']=='base'
    assert profile('base')['amplitudes']!=profile('local')['amplitudes']
    with pytest.raises(ValueError):profile('lower_collision_margin')


def test_earliest_preparation_accounts_for_future_velocity_and_neighborhood(fixture):
    _,state,_,_=fixture;limits=Limits()
    controls=[np.r_[limits.velocity[0]*.99,np.zeros(7)],np.zeros(8),np.zeros(8)]
    report=dict(feasible=True,step_s=.04,trace=[dict(t=t,state=state.tolist(),control=u.tolist())
        for t,u in zip([0.,.04,.08],controls)])
    result=preparation(report,0.,limits)
    assert result['status']=='verified' and result['duration_s']==pytest.approx(.04)
    assert result['camera_clock']=='unchanged'
    # A large excursion followed by a return cannot masquerade as nearby prep.
    report['trace'][1]['state']=state.copy().tolist()
    report['trace'][1]['state'][3]+=.8
    assert preparation(report,0.,limits)['status']=='not_found'


def test_predicted_preparation_is_never_reported_as_measured_success(fixture):
    _,state,_,_=fixture;limits=Limits()
    report=dict(feasible=True,step_s=.04,trace=[dict(t=t,state=state.tolist(),control=np.zeros(8).tolist()) for t in [1.,1.04]])
    report['preparation']=preparation(report,1.,limits)
    monitor=ConfigurationMonitor();monitor.reset('exact-hash',report)
    degraded=monitor.update(state,np.zeros(8),1.,dict(tpr_status='degraded',position_error=.1,orientation_error=.1))
    assert degraded['measured_reached_s'] is None
    normal=monitor.update(state,np.zeros(8),1.02,dict(tpr_status='normal',position_error=0.,orientation_error=0.))
    assert normal['status']=='maintaining'
    assert normal['measured_duration_s']==pytest.approx(.02)
    monitor.reset('new-hash',None)
    assert monitor.update(state,np.zeros(8),1.04,{})['status']=='inactive'


def test_arrival_tolerance_does_not_waive_measured_reserve(fixture):
    from jev_agent_wbc.jvs_kinematics import JOINT_UPPER
    _,state,_,_=fixture
    target=state.copy();target[3]=JOINT_UPPER[0]-.09
    report=dict(feasible=True,step_s=.04,trace=[dict(t=t,state=target.tolist(),control=np.zeros(8).tolist()) for t in [0.,.04]])
    report['preparation']=preparation(report,0.,Limits())
    state[3]=JOINT_UPPER[0]-.06  # within arrival tolerance, outside comfort set
    monitor=ConfigurationMonitor();monitor.reset('candidate',report)
    result=monitor.update(state,np.zeros(8),0.,dict(tpr_status='normal',position_error=0.,orientation_error=0.),Limits())
    assert result['measured_reached_s'] is None
    assert not result['measured_reserve_sufficient']


def test_nonzero_preparation_preserves_a_moving_camera_and_nearby_state(fixture):
    from jev_agent_wbc.jvs_kinematics import JOINT_UPPER
    from jev_agent_wbc.reference import smoothstep
    _,state,z,_=fixture
    state[3]=JOINT_UPPER[0]-.06
    p,r,_=whole_body_kinematics(state,z)
    class MovingCamera:
        duration=4.
        def sample(self,t):
            u=np.clip(t/4.,0.,1.)
            return p+smoothstep(u)*r[:,0]*.04,r.copy(),np.zeros(6)
    scene=dict(format='jvs_lidar_surfels/v1',observation_stamp=0.,surfel_centers=[[8.,8.,8.]],
               surfel_radius=.1,surfel_velocities=[[0.,0.,0.]],minimum_clearance_m=.025)
    result=KinematicSearch(state,np.zeros(8),0.,z,Limits(),scene=scene,search_config='local').lift(MovingCamera(),4.4)
    assert result['feasible']
    prep=result['preparation']
    assert prep['status']=='verified' and 0.<prep['duration_s']<1.
    assert prep['max_joint_change_rad']<=prep['joint_neighborhood_rad']
    assert prep['predicted_tail_margin_rad']>=prep['required_tail_joint_margin_rad']
    assert result['max_position_error_m']<.001
    assert result['max_orientation_error_rad']<.003
    assert np.linalg.norm(np.array(result['trace'][-1]['position'])-p)>.035


def test_fk_projection_is_explicit_bound_and_strictly_rechecked(fixture):
    from jev_agent_wbc.intent_paths import IntentPath
    from jev_agent_wbc.fk_camera_path import FKCameraPath
    from jev_agent_wbc.task_space_planning import MeasuredCameraPrediction
    from jev_agent_wbc.candidates import path_hash,restore_path
    original,state,z,runtime=fixture
    scene=dict(format='jvs_lidar_surfels/v1',observation_stamp=0.,surfel_centers=[[8.,8.,8.]],
               surfel_radius=.1,surfel_velocities=[[0.,0.,0.]],minimum_clearance_m=.025)
    seed=IntentPath(original,original,.5,dict(route='timing',amplitude_m=0.,extension_s=0.,phase_rate=.45),runtime['subject'])
    search=KinematicSearch(state,np.zeros(8),0.,z,Limits(),scene=scene,search_config='executable')
    report=search._lift(MeasuredCameraPrediction(seed,state,z,0.,.5),8.4,'anticipatory',project_camera=True)
    assert report['feasible'],report['reason']
    target=FKCameraPath.from_lift(seed,report)
    assert target.duration==original.duration
    reconstructed=restore_path(original,target.to_spec())
    assert path_hash(reconstructed)==path_hash(target)
    for t in (0.,.49,.5):
        np.testing.assert_allclose(target.sample(t)[0],original.sample(t)[0],atol=1e-12)
    checked=search.lift(MeasuredCameraPrediction(reconstructed,state,z,0.,.5),8.4)
    assert checked['feasible'],checked['reason']
    assert not checked['camera_projected_for_generation']
    assert checked['max_position_error_m']<=checked['position_tolerance_m']
    assert checked['min_body_clearance_m']>=.025
