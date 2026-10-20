"""Observed body threats and finite, immutable within-phase programs."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jev_agent_wbc.body_anticipation import base_preference, tracking_progress, barrier_gains
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.phase_program import build, RATES
from jev_agent_wbc.reference import OriginalReference

ROOT=Path(__file__).resolve().parents[1]/'scenarios'
STATE=np.r_[0.,0.,0.,0.,-.4,1.7,0.,-1.3,0.]


def observed(height,speed=.3):
    return DynamicCollision(dict(format='jvs_lidar_surfels/v1',
        surfel_centers=[[-.05,-.6,height]],surfel_radii=[.12],surfel_radius=.12,
        surfel_velocities=[[0.,speed,0.]],surfel_track_ids=[1],
        observation_stamp=0.,minimum_clearance_m=.025),0.)


def test_low_incoming_body_threat_is_distinct_from_high_camera_threat():
    command,info=base_preference(observed(.16),STATE,np.zeros(8))
    assert command is not None and info['active']
    assert info['closest_approach_s']>0.
    command,info=base_preference(observed(1.2),STATE,np.zeros(8))
    assert command is None and not info['active']


def test_shoulder_axis_threat_requires_platform_anticipation():
    command,info=base_preference(observed(.45),STATE,np.zeros(8))
    assert command is not None and info['active']
    assert info['threatened_link'] in ('link1','link2')


def test_crossing_to_opposite_corridor_side_accounts_for_extra_distance():
    near,_=base_preference(observed(.16),STATE,np.zeros(8))
    far,info=base_preference(observed(.16),STATE,np.zeros(8),task_axis=np.array([-1.,0.,0.]),escape_memory={})
    assert info['corridor_crossing_extra_m']==pytest.approx(.1)
    assert abs(far[0])>abs(near[0])
    command,info=base_preference(observed(.16,0.),STATE,np.zeros(8))
    assert command is None and not info['active']


def single_reference(name='combo_01_easy'):
    task=json.loads((ROOT/name/'task.json').read_text())
    scene=json.loads((ROOT/name/'runtime_scene.json').read_text())
    first=task['shot_phase'][0];task['shot_phase']=[first]
    task['reference_camera_trajectory']['samples']=[r for r in task['reference_camera_trajectory']['samples'] if r['t']<=first['end_s']]
    return OriginalReference(task,scene['subject']['position'])


@pytest.mark.parametrize('rate',RATES)
def test_phase_program_has_no_near_duplicate_knots_or_acceleration_spikes(rate):
    original=single_reference();before=deepcopy(original.task);name=before['shot_phase'][0]['name']
    path=build(original,{name:rate})
    assert np.min(np.diff(path.geometry['times']))>1e-8
    times=np.linspace(0.,path.duration,5001)
    assert np.max(np.linalg.norm(path.translation(times,2),axis=1))<.4
    np.testing.assert_allclose(path.sample(0.)[0],original.sample(0.)[0],atol=1e-12)
    np.testing.assert_allclose(path.sample(path.duration)[0],original.sample(original.duration)[0],atol=1e-12)
    assert np.all(np.diff(path.geometry['progress_times'])>=0.)
    held=times>=original.duration/RATES[rate]+.001
    if np.any(held):
        basis=np.asarray(path.geometry['shape_basis'])
        ended=np.asarray(path.geometry['progress_times'])>=original.duration
        np.testing.assert_allclose(basis[ended],np.tile([0.,0.,1.],(sum(ended),1)),atol=1e-12)
    assert original.task==before and path.duration==original.duration


def test_feedback_progress_advances_continuously_and_never_rewinds():
    reference=single_reference();controller=SimpleNamespace()
    positions=[reference.sample(t)[0] for t in (2.,2.01,1.9)]
    progress=[tracking_progress(controller,reference,p,.5) for p in positions]
    assert progress[0]==pytest.approx(2.,abs=.002)
    assert progress[1]>progress[0]
    assert progress[2]==progress[1]


def test_sideways_avoidance_does_not_count_as_pedestal_progress():
    reference=single_reference();task=reference.task;task['shot_phase'][0]['motion']='pedestal_up'
    reference=OriginalReference(task,reference.subject)
    name=reference.task['shot_phase'][0]['name']
    path=build(reference,{name:'original'},shapes={name:'left_small'})
    point=path.sample(2.)[0]
    first=tracking_progress(SimpleNamespace(),path,point,.5)
    sideways=tracking_progress(SimpleNamespace(),path,point+np.array([.2,-.2,0.]),.5)
    assert sideways==pytest.approx(first,abs=1e-10)


def test_unconfirmed_objects_keep_height_evidence_without_claiming_motion():
    from jev_agent_wbc.dynamic_lidar_map import DynamicLidarMap
    mapping=DynamicLidarMap(voxel_size=.035);mapping.stamp=1.
    center=np.array([1.,0.,.16]);extent=np.full(3,.12)
    mapping.tracks[4]=dict(moving=False,stamp=.9,points=np.array([center]),
        center=center,velocity=np.array([.3,0.,0.]),extent=extent)
    snapshot=mapping.snapshot();obj=snapshot['object_observations'][0]
    assert snapshot['tracks']==[]
    assert obj['height_range_m']==pytest.approx([.04,.28])
    assert obj['velocity_status']=='motion_not_yet_confirmed'
    np.testing.assert_array_equal(obj['center'],center)
    np.testing.assert_array_equal(obj['velocity'],0.)
    assert 'keyframes' not in json.dumps(snapshot['object_observations'])


def test_barrier_policy_brakes_movers_early_without_changing_margin():
    collision=observed(.16);gains=barrier_gains(collision,np.array([0]))
    np.testing.assert_array_equal(gains,np.where(collision.geometry.shape_bodies[None,:]==0,12.,6.))
    assert np.all(barrier_gains(observed(.16,.01),np.array([0]))==2.)
    assert collision.margin==.025


def test_verified_prediction_is_not_replaced_by_static_posture_witness():
    from jev_agent_wbc.planning import ExecutionWBC
    from jev_agent_wbc.whole_body_qp import Limits,whole_body_kinematics
    from scipy.linalg import null_space
    runtime=json.loads((ROOT/'combo_01_easy'/'runtime_scene.json').read_text())
    state=np.asarray(runtime['initial_state']);reference=single_reference()
    scene=dict(format='jvs_lidar_surfels/v1',surfel_centers=[[3.,3.,.3]],surfel_radius=.07,
        surfel_velocities=[[0.,0.,0.]],surfel_track_ids=[1],observation_stamp=0.,minimum_clearance_m=.025)
    c=ExecutionWBC(Limits(),scene=scene);c.set_trajectory(reference);c.body_aware=True
    p,r,J=whole_body_kinematics(state,runtime['base_z'])
    predicted=null_space(J)[:,0]*.005
    c.reconfiguration.update=lambda *args:predicted.copy()
    c.reconfiguration.info=dict(tpr_prediction_valid=True,tpr_prediction_path_progress_free=True)
    c.kinematic_reference=dict(feasible=True,trace=[dict(t=t,state=state.tolist(),control=(-predicted).tolist()) for t in [0.,7.]])
    command,info=c.solve(state,p,r,np.zeros(6),.02,base_z=runtime['base_z'],trajectory_time=0.)
    assert command@predicted>0.
    assert info['body_anticipation']['verified_path_preview_preference']


def test_clearance_reserve_never_penalizes_extra_separation():
    from jev_agent_wbc.tpr_wbc import qp_with_reserve
    for desired in (.1,.7):
        command,_=qp_with_reserve(np.eye(2),np.array([-desired,0.]),np.eye(2),
            np.full(2,-1.),np.ones(2),np.zeros(2),np.array([[1.,0.]]),np.array([.3]),100.)
        if desired>.3:assert command[0]==pytest.approx(desired,abs=1e-6)
        else:assert .29<command[0]<.3


def test_horizontal_pedestal_bows_preserve_vertical_direction():
    from jev_agent_wbc.cinematic_geometry import curve_quality
    original=single_reference();task=original.task;task['shot_phase'][0]['motion']='pedestal_up'
    original=OriginalReference(task,original.subject)
    name=original.task['shot_phase'][0]['name']
    for shape in ('left_small','right_small'):
        path=build(original,{name:'original'},shapes={name:shape})
        assert curve_quality(path)['direction_pass']


def test_global_posture_lift_preserves_bow_and_keeps_joint_reserve():
    from jev_agent_wbc.kinematic_feasibility import KinematicSearch
    from jev_agent_wbc.whole_body_qp import Limits
    from jev_agent_wbc.joint_curve_optimizer import refine
    from jev_agent_wbc.reference import smoothstep
    ref=single_reference('combo_04_medium')
    runtime=json.loads((ROOT/'combo_04_medium'/'runtime_scene.json').read_text())
    name=ref.task['shot_phase'][0]['name']
    path=build(ref,{name:'original'},runtime['subject'],shapes={name:'positive_broad'})
    path.program.update(fixed_spatial_shape=True,joint_reserve_rad=.12)
    search=KinematicSearch(runtime['initial_state'],np.zeros(8),0.,runtime['base_z'],Limits())
    fitted,info=refine(path,search,seconds=10.,max_iterations=20)
    assert fitted is not None,info
    report=search.lift(fitted,fitted.duration)
    assert report['feasible'],report['reason']
    assert report['min_joint_margin_rad']>.10
    times=np.array(path.geometry['times'])
    shift=np.array(fitted.geometry['positions'])-path.geometry['positions']
    np.testing.assert_allclose(shift,smoothstep(times/path.duration)[:,None]*shift[-1],atol=1e-7)


def pedestal_fixture(name='combo_03_medium'):
    task=json.loads((ROOT/name/'task.json').read_text())
    runtime=json.loads((ROOT/name/'runtime_scene.json').read_text())
    nominal=json.loads((ROOT/'video_combo'/f'{name}_nominal.json').read_text())
    action=next(a for a in nominal['actions'] if 'pedestal' in a['phase']['motion'])
    phase=action['phase'];a,b=phase['start_s'],phase['end_s']
    task['shot_phase']=[dict(phase,start_s=0.,end_s=b-a)]
    task['reference_camera_trajectory']['samples']=[dict(r,t=round(r['t']-a,8))
        for r in task['reference_camera_trajectory']['samples'] if a<=r['t']<=b]
    return OriginalReference(task,runtime['subject']['position']),runtime,action['states'][0]


@pytest.mark.parametrize('sign',('positive_round','negative_round'))
def test_rounded_pedestal_preserves_endpoints_direction_and_smooth_turn(sign):
    from jev_agent_wbc.cinematic_geometry import curve_quality
    ref,runtime,_=pedestal_fixture();name=ref.task['shot_phase'][0]['name']
    path=build(ref,{name:'original'},runtime['subject'],shapes={name:sign})
    for t in (0.,ref.duration):
        np.testing.assert_allclose(path.sample(t)[0],ref.sample(t)[0],atol=1e-8)
    quality=curve_quality(path)
    assert quality['direction_pass'] and quality['strict_full_curve_pass']
    p=path.sample(1.)[0]+np.array([0.,0.,.001])
    progress=tracking_progress(SimpleNamespace(),path,p,.5)
    assert progress==pytest.approx(1.,abs=.01)


def test_fixed_pedestal_bow_gets_posture_reserve_without_extra_planar_direction_gate():
    from jev_agent_wbc.kinematic_feasibility import KinematicSearch
    from jev_agent_wbc.whole_body_qp import Limits
    from jev_agent_wbc.joint_curve_optimizer import refine
    ref,runtime,state=pedestal_fixture();name=ref.task['shot_phase'][0]['name']
    path=build(ref,{name:'original'},runtime['subject'],shapes={name:'left_broad'})
    path.program.update(fixed_spatial_shape=True,fixed_camera_path=True,joint_reserve_rad=.12)
    search=KinematicSearch(state,np.zeros(8),0.,runtime['base_z'],Limits())
    fitted,info=refine(path,search,seconds=10.,max_iterations=40,knots=49)
    assert fitted is not None,info
    report=search.lift(fitted,fitted.duration)
    assert report['feasible'] and report['min_joint_margin_rad']>.10
    np.testing.assert_allclose(fitted.geometry['positions'],path.geometry['positions'],atol=1e-8)


def test_corridor_probe_does_not_claim_unconfirmed_object_motion():
    from jev_agent_wbc.phase_program import camera_corridor_space
    ref,runtime,_=pedestal_fixture();name=ref.task['shot_phase'][0]['name']
    p=ref.sample(0.)[0]
    observed_scene=dict(object_observations=[dict(id=7,center=(p+[1.,0.,0.]).tolist(),
        half_extent=[.12,.12,.12],velocity=[0.,0.,0.],moving_confirmed=False)])
    path=build(ref,{name:'original'},runtime['subject'],shapes={name:'left_broad'})
    probe=camera_corridor_space(path,observed_scene)
    assert probe['probes'][0]['source']=='hypothetical_direct_approach_not_observed_motion'
    assert observed_scene['object_observations'][0]['velocity']==[0.,0.,0.]


def test_bundled_search_uses_one_request_and_refusal_still_prevents_search():
    from jev_agent_wbc.jev import CandidateDecider
    class Client:
        def __init__(self,selection):self.selection=selection;self.calls=[]
        def ask(self,evidence,questions):
            self.calls.append(questions)
            return dict(strategy=dict(choice=self.selection),configuration=dict(choice='base'))
    evidence=dict(stage='strategy',objective='visible_intent_completion',bundle_search_choices=True)
    client=Client('joint');decision=CandidateDecider(client).decide(evidence)
    assert len(client.calls)==1 and set(client.calls[0])=={'strategy','configuration'}
    assert decision==dict(stage='configuration',strategy='joint',search_config='base',bundled_strategy=True)
    client=Client('refuse');decision=CandidateDecider(client).decide(evidence)
    assert decision==dict(stage='strategy',strategy='refuse')


def test_observed_crossing_bow_binds_direction_and_keeps_vertical_endpoints():
    ref,runtime,_=pedestal_fixture();name=ref.task['shot_phase'][0]['name']
    axis=np.array([-.4,.9,0.]);axis/=np.linalg.norm(axis)
    path=build(ref,{name:'original'},runtime['subject'],shapes={name:'crossing_left'},directions={name:axis})
    np.testing.assert_allclose(path.program['observed_shape_directions'][name],axis)
    np.testing.assert_allclose(path.sample(path.duration/2)[0]-ref.sample(ref.duration/2)[0],.30*axis,atol=1e-7)
    for t in (0.,path.duration):np.testing.assert_allclose(path.sample(t)[0],ref.sample(t)[0],atol=1e-8)
    with pytest.raises(ValueError,match='horizontal'):
        build(ref,{name:'original'},shapes={name:'crossing_left'},directions={name:[0.,0.,1.]})


def test_delayed_curve_feedback_aims_forward_instead_of_across_a_turn():
    from jev_agent_wbc.body_anticipation import local_forward_target
    ref,runtime,_=pedestal_fixture('combo_04_medium');name=ref.task['shot_phase'][0]['name']
    path=build(ref,{name:'original'},runtime['subject'],shapes={name:'crossing_right'},directions={name:[.67,-.74,0.]})
    c=SimpleNamespace();at=2.5;point=path.sample(at)[0]
    tracking_progress(c,path,point,at)
    target_time=local_forward_target(c,path,7.)
    tangent=path.sample(at+.01)[0]-path.sample(at-.01)[0]
    assert (path.sample(7.)[0]-point)@tangent<0.
    assert at<target_time<7.
    assert (path.sample(target_time)[0]-point)@tangent>0.


def test_first_motion_vote_is_only_a_hint_and_does_not_move_collision_map():
    from jev_agent_wbc.dynamic_lidar_map import DynamicLidarMap
    mapping=DynamicLidarMap(voxel_size=.035);mapping.stamp=1.
    center=np.array([1.,0.,.8]);extent=np.full(3,.12)
    mapping.tracks[4]=dict(moving=False,stamp=1.,points=np.array([center]),
        center=center,velocity=np.array([-.08,0.,0.]),extent=extent,
        sphere=dict(radius=.12,residual_m=0.),motion_votes=1)
    snapshot=mapping.snapshot();obj=snapshot['object_observations'][0]
    assert len(snapshot['tracks'])==1 and not obj['moving_confirmed']
    assert not snapshot['tracks'][0]['moving_confirmed']
    np.testing.assert_allclose(snapshot['tracks'][0]['center'],center)
    np.testing.assert_array_equal(obj['velocity'],0.)
    assert obj['motion_hint']['confirmed'] is False
    np.testing.assert_allclose(obj['motion_hint']['velocity_estimate'],[-.08,0.,0.])
    assert not np.any(snapshot['surfel_velocities'])


def test_nearly_horizontal_capture_leg_does_not_skip_to_later_height_progress():
    from jev_agent_wbc.intent_paths import IntentPath
    ref,runtime,_=pedestal_fixture();name=ref.task['shot_phase'][0]['name']
    path=build(ref,{name:'original'},runtime['subject'],shapes={name:'positive_round'})
    geometry=deepcopy(path.geometry);positions=np.asarray(geometry['positions']);times=np.asarray(geometry['times'])
    # A measured arrival correction tilts a formerly flat approach by less
    # than a millimetre. Height alone must not skip that horizontal approach.
    positions[:,2]+=.0003*(1-times/path.duration)
    geometry['positions']=positions.tolist()
    path=IntentPath(path.original,path.previous,path.activation_s,path.program,path.model,path.intrinsics,geometry=geometry)
    point=path.sample(1.)[0]+np.array([0.,0.,.001])
    progress=tracking_progress(SimpleNamespace(),path,point,.5)
    assert progress==pytest.approx(1.,abs=.03)
