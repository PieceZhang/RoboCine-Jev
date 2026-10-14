"""Completion is directed visible content, independent of tracking or WBC."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.camera_intent import IntentSpec, visibility, default_intrinsics, selection_key
from jev_agent_wbc.candidates import SearchBudget, path_hash, path_spec, restore_path
from jev_agent_wbc.floating_camera import intent_prediction, camera_candidates, evaluate_path
from jev_agent_wbc.intent_paths import IntentPath, propose_intent_paths
from jev_agent_wbc.reference import OriginalReference, look_at
from jev_agent_wbc.jev import CandidateDecider, PhaseAwareDecider, compact_evidence

ROOT = Path(__file__).resolve().parents[1]/'scenarios'


def fixture(name='movein'):
    scene = json.loads((ROOT/name/'runtime_scene.json').read_text())
    original = OriginalReference(ROOT/name/'task.json', scene['subject']['position'])
    observed = dict(valid=True, observation_stamp=0., centers=[[9., 9., 9.]],
        radii=[.1], velocities=[[0., 0., 0.]], subject_model=scene['subject'])
    return original, observed, IntentSpec(original, scene['subject'])


def score(spec, position_fn=None, rotation_fn=None, visual_fn=None, times=None, progress=None):
    times = np.linspace(0, 8, 401) if times is None else times
    progress = times if progress is None else progress
    poses = [spec.original.sample(float(t)) for t in progress]
    positions = [p if position_fn is None else position_fn(p.copy(), t) for (p,r,v),t in zip(poses,times)]
    rotations = [r if rotation_fn is None else rotation_fn(r.copy(), t) for (p,r,v),t in zip(poses,times)]
    visual = [dict(visibility=1., in_frame=1.) if visual_fn is None else visual_fn(t) for t in times]
    return spec.score(times, positions, rotations, visual, progress)


@pytest.mark.parametrize('name', ['movein','truck_left','truck_right','pedestal_up','pedestal_down','orbit'])
def test_reference_has_full_directed_intent_and_retiming_preserves_it(name):
    original, scene, spec = fixture(name)
    assert score(spec)['intent_complete']
    assert score(spec, times=np.linspace(0,12,601), progress=np.linspace(0,8,601))['intent_complete']


def test_occlusion_is_not_moveout_progress_and_hidden_travel_is_not_covered():
    original, scene, spec = fixture()
    measured = score(spec, visual_fn=lambda t:dict(visibility=0. if 2<t<6 else 1., in_frame=1.))
    assert measured['geometric_complete']
    assert not measured['intent_complete'] and measured['effective_coverage'] < .5
    assert measured['longest_occlusion_s'] > 3.9
    p, r, _ = original.sample(4)
    clear = visibility(p,r,scene,4,spec.model,spec.intrinsics)
    scene['centers'] = [((p+original.subject)/2).tolist()]; scene['radii'] = [.25]
    hidden = visibility(p,r,scene,4,spec.model,spec.intrinsics)
    assert clear['projected_area'] == pytest.approx(hidden['projected_area'])
    assert hidden['visibility'] < clear['visibility']


def test_same_delta_at_wrong_scale_interval_does_not_complete():
    original, scene, spec = fixture()
    value = score(spec, position_fn=lambda p,t: original.subject+(p-original.subject)*1.5)
    assert not value['intent_complete']
    assert value['segments'][0]['start_error_fraction'] > .1


def test_shifted_or_reversed_orbit_sector_is_not_equivalent():
    original, scene, spec = fixture('orbit')
    rotate = Rotation.from_euler('z', 1.).as_matrix()
    shifted = score(spec, position_fn=lambda p,t:original.subject+rotate@(p-original.subject))
    assert not shifted['intent_complete']
    progress = np.r_[np.linspace(0,8,201), np.linspace(8,0,201)[1:], np.linspace(0,8,201)[1:]]
    repeated = score(spec, times=np.linspace(0,24,len(progress)), progress=progress)
    assert not repeated['intent_complete']
    assert repeated['segments'][0]['reverse_travel_fraction'] > .9


@pytest.mark.parametrize('name', ['truck_left','pedestal_up'])
def test_rotation_cannot_fake_required_translation(name):
    original, scene, spec = fixture(name)
    value = score(spec, position_fn=lambda p,t:original.sample(0)[0])
    assert not value['intent_complete'] and value['effective_coverage']==0.


def test_missing_geometry_is_unknown_not_verified_movein():
    original, scene, _ = fixture()
    spec = IntentSpec(original)
    assert not spec.segments[0]['supported']
    assert not score(spec)['intent_complete']
    assert visibility(*original.sample(0)[:2],scene,0,None,default_intrinsics())['visibility'] is None


def test_multiray_visibility_detects_offcenter_occlusion():
    original, scene, spec = fixture()
    p,r,_ = original.sample(0)
    # Obstacle hides the top of the subject while its center ray stays clear.
    center = (p+original.subject)/2+np.array([0.,0.,.07])
    scene.update(centers=[center.tolist()],radii=[.04])
    ray=original.subject-p
    nearest=p+np.clip((center-p)@ray/(ray@ray),0,1)*ray
    assert np.linalg.norm(nearest-center)>.04
    value=visibility(p,r,scene,0,spec.model,spec.intrinsics)
    assert value['surface_rays']>20 and value['visibility']<.98


def test_subject_is_a_collision_body_even_without_external_obstacles():
    original,scene,spec=fixture()
    scene.update(centers=[],radii=[],velocities=[])
    class InsideSubject:
        duration=1.
        def sample(self,t):return original.subject.copy(),np.eye(3),np.zeros(6)
    value,_=evaluate_path(InsideSubject(),scene,0.,.1,.06)
    assert not value.feasible and value.min_clearance_m<0.


def test_safe_detour_can_win_with_larger_reference_error_and_exact_replay():
    original, scene, spec = fixture()
    path = IntentPath(original,original,1.5,dict(route='above',amplitude_m=.12,extension_s=2.),spec.model)
    assert path.duration==10.
    assert intent_prediction(spec,path,scene,0)['intent_complete']
    assert np.linalg.norm(path.sample(4)[0]-original.sample(4)[0])>.05
    restored=restore_path(original,path_spec(path))
    assert path_hash(restored)==path_hash(path)
    for t in np.linspace(0,path.duration,37):
        for a,b in zip(path.sample(t),restored.sample(t)):
            np.testing.assert_array_equal(a,b)
    for t in (0.,1.,1.5):
        for a,b in zip(path.sample(t),original.sample(t)):
            np.testing.assert_array_equal(a,b)
    assert np.all(np.diff([path.progress_time(t) for t in np.linspace(0,10,401)])>=0)


def test_replanning_keeps_executed_occlusion_debt():
    original, scene, spec = fixture()
    history=[]
    for t in np.arange(0.,4.,.05):
        p,r,_=original.sample(float(t))
        history.append(dict(t=float(t),camera_position=p.tolist(),camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
            progress_s=float(t),visual=dict(visibility=0.,in_frame=1.)))
    result=intent_prediction(spec,original,scene,4.,history)
    assert not result['intent_complete'] and result['effective_coverage']<.75


def test_lexicographic_selection_never_trades_completion_for_cost():
    complete=dict(intent_complete=True,geometric_complete=True,effective_coverage=1.,
        longest_occlusion_s=0.,visibility_p05=1.,visibility_mean=1.,duration_s=12.)
    incomplete=dict(complete,intent_complete=False,effective_coverage=.5,duration_s=8.)
    assert selection_key(complete,1000)<selection_key(incomplete,0)


@pytest.mark.parametrize('name,route', [('movein','above'),('truck_left','outer'),
    ('truck_right','outer'),('pedestal_up','inner'),('pedestal_down','inner'),('orbit','above')])
def test_large_held_offset_covers_range_but_fails_frame_and_trend(name,route):
    original,scene,spec=fixture(name)
    path=IntentPath(original,original,1.5,dict(route=route,amplitude_m=.35,extension_s=0.,
        profile='hold',rise_s=1.2),spec.model)
    assert np.linalg.norm(path.sample(8.)[0]-original.sample(8.)[0])>.2
    result=intent_prediction(spec,path,scene,0.)
    assert not result['intent_complete'] and result['effective_coverage']==1.
    assert not result['terminal_frame']['pass_check'] or not result['motion_trend']['pass_check']
    assert path.return_s is None


def test_one_frame_jump_does_not_cover_motion_interval():
    original,scene,spec=fixture('orbit')
    progress=np.r_[np.zeros(200),np.full(201,8.)]
    result=score(spec,progress=progress)
    assert result['geometric_complete'] and not result['intent_complete']
    assert result['effective_coverage']==0.


def test_world_time_not_motion_progress_controls_dynamic_obstacle():
    original, scene, spec=fixture()
    slow=IntentPath(original,original,1.5,dict(route='timing',amplitude_m=0.,extension_s=4.),spec.model)
    at=6.
    p=slow.sample(at)[0]
    scene.update(centers=[(p-np.array([0.,at,0.])).tolist()],velocities=[[0.,1.,0.]],radii=[.1])
    value,_=evaluate_path(slow,scene,at,at+.02,.06)
    assert not value.feasible
    assert abs(slow.progress_time(at)-at)>1.


def test_joint_search_retains_opposite_routes_and_prefix_is_c2():
    original,scene,spec=fixture('pedestal_up')
    paths=propose_intent_paths(original,original,'joint',0.,SearchBudget(max_proposals=16,activation_lead_s=1.5),spec.model)
    assert {'inner','outer','left','right'} <= {p.program['route'] for p in paths}
    for path in paths:
        h=1e-4;a=path.activation_s
        np.testing.assert_allclose(path.sample(a+h)[0],original.sample(a+h)[0],atol=1e-7)
        np.testing.assert_allclose(path.translation(a,1),original.sample(a)[2][:3],atol=1e-8)
        expected=(original.sample(a+.001)[2][:3]-original.sample(a-.001)[2][:3])/.002
        np.testing.assert_allclose(path.translation(a,2),expected,atol=1e-7)


def test_intent_candidate_generation_never_invokes_wbc_or_robot_fk(monkeypatch):
    import jev_agent_wbc.whole_body_qp as wbc
    import jev_agent_wbc.planning as planning
    monkeypatch.setattr(wbc.WholeBodyQP,'solve',lambda *a,**k:pytest.fail('WBC invoked'))
    monkeypatch.setattr(wbc,'whole_body_kinematics',lambda *a,**k:pytest.fail('robot FK invoked'))
    monkeypatch.setattr(planning,'validate_rollout',lambda *a,**k:pytest.fail('WBC rollout invoked'))
    original,scene,spec=fixture('truck_left')
    result=camera_candidates(original,original,'view',scene,0.,SearchBudget(max_proposals=2,max_rollouts=2,activation_lead_s=1.5),.06)
    assert result['candidates'] and not result['error']
    cards=[c.card() for c in result['candidates']]
    evidence=dict(objective='visible_intent_completion',stage='selection',candidates=cards)
    decision=PhaseAwareDecider().decide(evidence)
    assert decision['candidate_id'] in [c.id for c in result['candidates']]
    class Client:
        def ask(self,state,questions):
            cleaned=compact_evidence(state)
            assert cleaned['candidates'][0]['intent']['intent_complete']
            assert 'tracking_cost' not in json.dumps(cleaned)
            assert 'NOT tracking' in self.policy
            return dict(candidate=dict(choice='c0',probabilities={'c0':1.}))
    assert CandidateDecider(Client()).decide(evidence)['candidate_id']==cards[0]['id']


def test_observed_occlusion_replans_even_without_legacy_mask_center_samples():
    from types import SimpleNamespace
    from jev_agent_wbc.engine import CinematicEngine
    engine=object.__new__(CinematicEngine)  # No controller is constructed or run.
    engine.perception=SimpleNamespace(snapshot=lambda t:dict(visual=dict(status='valid',stamp_s=t,
        occluded=True,truncated=False),diagnosis=''))
    engine.last_visual_stamp=None
    engine.active_candidate=SimpleNamespace(consequences=dict(intent=dict(intent_complete=True)))
    feedback=[]
    engine._write=lambda *a,**k:None
    engine._feedback=lambda cause,t,**k:feedback.append(cause)
    engine._monitor_visual(2.)
    assert feedback==['new_visibility_evidence_requires_review']
