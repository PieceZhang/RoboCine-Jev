"""Combo ingestion, phase preservation, joins and honest execution acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest

from jev_agent_wbc.combo import complete_task, combo_curves, gaps
from jev_agent_wbc.contracts import validate_task
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.cinematic_geometry import curve_quality
from jev_agent_wbc.shot_acceptance import evaluate_shot

ROOT=Path(__file__).resolve().parents[1]/'scenarios'
NAMES=('combo_01_easy','combo_02_easy','combo_03_medium','combo_04_medium')


def fixture(name=NAMES[0]):
    task=json.loads((ROOT/name/'task.json').read_text())
    scene=json.loads((ROOT/name/'runtime_scene.json').read_text())
    return task,scene


def selected(evidence,questions):
    assert 'keyframes' not in json.dumps(evidence)
    return {key:dict(choice='left',probabilities={k:float(k=='left') for k in q['criteria']}) for key,q in questions.items()}


def reference(name=NAMES[0]):
    task,scene=fixture(name)
    full,record=complete_task(task,scene['subject']['position'],selected)
    return OriginalReference(full,scene['subject']['position']),scene,record


@pytest.mark.parametrize('name',NAMES)
def test_gaps_are_explicitly_completed_without_rewriting_main_actions(name):
    task,scene=fixture(name);before=deepcopy(task)
    with pytest.raises(ValueError,match='contiguous'):validate_task(task)
    full,record=complete_task(task,scene['subject']['position'],selected)
    assert task==before and not gaps(full)
    assert len(record['transitions'])==len(gaps(task))
    old={r['t']:r for r in task['reference_camera_trajectory']['samples']}
    new={r['t']:r for r in full['reference_camera_trajectory']['samples']}
    assert all(new[t]==row for t,row in old.items())
    assert full['shot_phase'][-1]['end_s']==task['shot_phase'][-1]['end_s']
    ref=OriginalReference(full,scene['subject']['position'])
    for phase in full['shot_phase'][:-1]:
        t=phase['end_s']
        for order in (0,1,2):
            np.testing.assert_allclose(ref.translation(t-1e-7,order),ref.translation(t+1e-7,order),atol=3e-5)
        np.testing.assert_allclose(ref.rotation(t-1e-7,1),ref.rotation(t+1e-7,1),atol=3e-5)


def test_complete_task_does_not_call_model_and_invalid_choice_is_rejected():
    ref,scene,_=reference()
    def never(*args):raise AssertionError('unexpected model call')
    task,record=complete_task(ref.task,scene['subject']['position'],never)
    assert task==ref.task and not record['completed']
    source,_=fixture()
    with pytest.raises(ValueError,match='outside typed'):
        complete_task(source,scene['subject']['position'],lambda e,q:{k:dict(choice='teleport') for k in q})


@pytest.mark.parametrize('name',NAMES)
def test_reference_and_zero_deformation_preserve_every_motion(name):
    ref,scene,_=reference(name)
    path=combo_curves(ref,ref,0.,scene['subject'],None,seed_direction=np.zeros(3))[0]
    for phase in ref.task['shot_phase']:
        t=(phase['start_s']+phase['end_s'])/2
        np.testing.assert_allclose(path.sample(t)[0],ref.sample(t)[0],atol=1e-5)
        assert path.progress_time(t)==pytest.approx(t)
    quality=curve_quality(path)
    assert quality['direction_pass'] and quality['strict_full_curve_pass'],quality


def measured(ref,scene,positions=None,progress=None):
    ts=np.arange(0.,ref.duration+.401,.02);poses=[ref.sample(float(t)) for t in ts]
    return evaluate_shot(ref,scene['subject'],ts,
        [p for p,r,v in poses] if positions is None else positions(ts,poses),
        [r for p,r,v in poses],[dict(visibility=1.,in_frame=1.) for _ in ts],
        np.minimum(ts,ref.duration) if progress is None else progress(ts),safety_pass=True)


@pytest.mark.parametrize('name',NAMES)
def test_combo_has_verified_per_action_acceptance(name):
    ref,scene,_=reference(name);result=measured(ref,scene)
    assert result['accepted'],result
    assert result['protocol']['version']=='shot_acceptance/v2_combo'
    assert len(result['checks']['direction']['phases'])==len([p for p in ref.task['shot_phase'] if p['motion']!='transition'])


def test_reversing_later_action_cannot_pass_on_first_action_direction():
    ref,scene,_=reference()
    def reverse(ts,poses):
        points=np.array([p for p,r,v in poses]);mask=ts>=9.
        origin=ref.sample(9.)[0]
        points[mask]=origin-(points[mask]-origin)
        return points
    result=measured(ref,scene,positions=reverse)
    assert 'direction' in result['failed_checks']


def test_missing_later_phase_and_teleporting_connector_are_rejected():
    ref,scene,_=reference()
    result=measured(ref,scene,progress=lambda ts:np.minimum(ts,6.))
    assert 'direction' in result['failed_checks']
    def teleport(ts,poses):
        points=np.array([p for p,r,v in poses]);points[(ts>7.9)&(ts<8.1),2]+=.5
        return points
    result=measured(ref,scene,positions=teleport)
    assert 'transition_continuity' in result['failed_checks']


def test_long_combo_keeps_collision_sampling_resolution():
    from jev_agent_wbc.kinematic_feasibility import KinematicSearch
    from jev_agent_wbc.whole_body_qp import Limits
    from jev_agent_wbc.task_space_planning import configure_combo_samples
    ref,scene,_=reference(NAMES[2])
    search=KinematicSearch(scene['initial_state'],np.zeros(8),0.,scene['base_z'],Limits())
    step=search.policy.step_s
    configure_combo_samples(search,ref)
    assert search.policy.step_s==step
    assert search.policy.max_samples>=int(np.ceil((ref.duration+.4)/step))+1


def test_joint_combo_refinement_runs_phase_constraints():
    from jev_agent_wbc.kinematic_feasibility import KinematicSearch
    from jev_agent_wbc.whole_body_qp import Limits
    from jev_agent_wbc.joint_curve_optimizer import refine
    ref,scene,_=reference()
    search=KinematicSearch(scene['initial_state'],np.zeros(8),0.,scene['base_z'],Limits())
    path=combo_curves(ref,ref,0.,scene['subject'],None,seed_direction=np.zeros(3))[0]
    result,report=refine(path,search,seconds=.1,max_iterations=1)
    assert report['reason']!='insufficient_horizon'
    if result is not None:
        assert result.program['combo_phase_preserving']
        assert curve_quality(result)['direction_pass']


def test_observed_round_mover_uses_recent_braking_velocity_without_truth():
    from jev_agent_wbc.dynamic_lidar_map import DynamicLidarMap
    mapping=DynamicLidarMap()
    rng=np.random.default_rng(28);surface=rng.normal(size=(160,3))
    surface=.12*surface/np.linalg.norm(surface,axis=1)[:,None]
    for t in np.arange(9)*.1:
        center=np.array([1.+.5*t-.2*t*t,.8,.6])
        points=center+surface
        mapping.current_points=points
        mapping.track(points,1.+t)
    movers=[track for track in mapping.tracks.values() if track['moving']]
    assert len(movers)==1
    np.testing.assert_allclose(movers[0]['velocity'],[.5-.4*.8,0.,0.],atol=.02)


def test_nearby_low_and_middle_returns_do_not_merge_into_one_track():
    from jev_agent_wbc.sensors import RuntimeLidar
    from jev_agent_wbc.scenarios import RuntimeScene
    scene=RuntimeScene(ROOT/NAMES[0]/'runtime_scene.json');sensor=RuntimeLidar()
    state=np.array(scene.document['initial_state'])
    for t in np.round(np.arange(-.8,.51,.1),6):
        observed=sensor.synthetic_scan(scene,state,scene.document['base_z'],float(t))
    assert any(abs(track['center'][2]-.16)<.01 for track in observed['tracks'])
    assert all(track['shape_model']=='measured_sphere' for track in observed['tracks'])


def test_combo_choices_do_not_offer_unimplemented_global_timing():
    from jev_agent_wbc.jev import CandidateDecider
    from jev_agent_wbc.directed_search import options
    class Client:
        def ask(self,evidence,questions):
            choices=questions['generator']['criteria']
            assert set(choices)=={'joint_curve','joint_vertical'}
            return {'generator':{'choice':'joint_curve'}}
    decider=CandidateDecider(Client())
    decision=decider.decide(dict(stage='strategy',objective='visible_intent_completion',
        construction=dict(protocol='single_candidate/v1',combo_fixed_phase_clock=True)))
    assert decision['strategy']=='joint_curve'
    assert options('joint_curve',combo=True)['left']['tempo']==1.


def test_independent_audit_bounds_accelerating_obstacle_between_endpoints():
    import importlib.util
    from scipy.interpolate import CubicHermiteSpline
    file=ROOT.parent/'scripts/analyze_combo_cpu.py'
    spec=importlib.util.spec_from_file_location('combo_audit_test',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    curve=CubicHermiteSpline([0.,1.],[[0.,0.,0.],[1.,0.,0.]],[[0.,0.,0.],[0.,0.,0.]])
    assert module.speed_bound(curve,0.,1.)==pytest.approx(1.5)
    assert module.speed_bound(curve,1.1,1.2)==0.
