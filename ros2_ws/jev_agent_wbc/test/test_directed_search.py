"""Hard construction binding and transport failures independent of planning."""
import io
import json
from pathlib import Path

import numpy as np
import pytest

from jev_agent_wbc.candidates import SearchBudget
from jev_agent_wbc.directed_search import generate, options
from jev_agent_wbc.jev import CandidateDecider, SystemOneClient
from jev_agent_wbc.reference import OriginalReference

ROOT=Path(__file__).resolve().parents[1]/'scenarios'


def test_timeout_retries_same_typed_request_without_negative_decision(monkeypatch,tmp_path):
    calls=[]
    reply={'answers':{'x':{'type':'choice','choice':'a','probabilities':{'a':1.,'b':0.}}}}
    def open_request(request,timeout):
        calls.append(request.data)
        if len(calls)==1:raise TimeoutError('test connection timeout')
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr('urllib.request.urlopen',open_request)
    # A clock moving forward avoids an actual two-second test sleep.
    tick=iter(np.arange(0.,100.,3.))
    monkeypatch.setattr('jev_agent_wbc.jev.time.monotonic',lambda:float(next(tick)))
    client=SystemOneClient(journal=tmp_path/'api.jsonl')
    result=client.ask({'stage':'strategy'},{'x':{'type':'choice','criteria':{'a':'A','b':'B'}}})
    assert result==reply['answers'] and len(calls)==2 and calls[0]==calls[1]
    logs=[json.loads(l) for l in (tmp_path/'api.jsonl').read_text().splitlines()]
    assert logs[0]['status']=='transport_unverified_retry'
    assert logs[1]['response']==reply


def test_first_choice_limits_second_choice_and_binds_generation():
    class Client:
        def ask(self,state,questions):
            key=next(iter(questions))
            selected='regular_arch' if key=='generator' else 'left'
            assert selected in questions[key]['criteria']
            return {key:dict(type='choice',choice=selected)}
    decider=CandidateDecider(Client())
    evidence={'construction':{'protocol':'single_candidate/v1'}}
    first=decider.decide(dict(evidence,stage='strategy'))
    second=decider.decide(dict(evidence,stage='configuration',search_options={'selected_family':first['strategy']}))
    assert second['construction_plan']==options('regular_arch')['left']
    runtime=json.loads((ROOT/'movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'movein/task.json',runtime['subject']['position'])
    paths=generate(original,original,{},1.,runtime['subject'],None,SearchBudget(),second['construction_plan'])
    assert len(paths)==1 and paths[0].program['route']=='left'
    with pytest.raises(ValueError,match='outside typed choices'):
        generate(original,original,{},1.,runtime['subject'],None,SearchBudget(),dict(second['construction_plan'],amplitude_m=10.))


def test_timed_generator_checks_only_one_selected_spatial_seed(monkeypatch):
    from jev_agent_wbc import curve_timing
    seen=[]
    def fail_one(*args,**kwargs):
        seen.append(kwargs)
        return []
    monkeypatch.setattr(curve_timing,'timed_curves',fail_one)
    runtime=json.loads((ROOT/'movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'movein/task.json',runtime['subject']['position'])
    paths=generate(original,original,{},1.,runtime['subject'],None,SearchBudget(),options('timed_curve')['left'])
    assert paths==[] and len(seen)==1 and seen[0]['count']==1
    assert np.linalg.norm(seen[0]['displacement'])==pytest.approx(.2)


def test_first_choice_binds_displacement_plane_before_spatial_choice():
    seen=[]
    class Client:
        def ask(self,state,questions):
            key=next(iter(questions));seen.append(set(questions[key]['criteria']))
            value='joint_curve' if key=='generator' else 'outer'
            return {key:dict(type='choice',choice=value)}
    decider=CandidateDecider(Client())
    evidence=dict(construction=dict(protocol='single_candidate/v1',allowed_directions=['outer','above','below']))
    first=decider.decide(dict(evidence,stage='strategy'))
    second=decider.decide(dict(evidence,stage='configuration',search_options=dict(selected_family=first['strategy'])))
    assert {'joint_curve','joint_vertical'}<=seen[0]
    assert seen[1]=={'outer'}
    assert second['construction_plan']['direction']=='outer'
    assert set(options('joint_vertical'))=={'above','below'}
    seen.clear();evidence['construction']['allowed_directions']=['left','right']
    decider.decide(dict(evidence,stage='strategy'))
    assert 'joint_vertical' not in seen[0]


@pytest.mark.parametrize('velocity',[(.03,-.2,.11),(0.,0.,0.)])
def test_partial_round_returns_preserve_measured_vertical_motion(velocity):
    from jev_agent_wbc.dynamic_lidar_map import DynamicLidarMap
    mapping=DynamicLidarMap(voxel_size=.035)
    az,el=np.meshgrid(np.linspace(-1.,1.,13),np.linspace(-.3,.3,5))
    unit=np.c_[(np.cos(el)*np.cos(az)).ravel(),(np.cos(el)*np.sin(az)).ravel(),np.sin(el).ravel()]
    velocity=np.asarray(velocity)
    for k,t in enumerate(np.arange(0.,1.21,.1)):
        center=np.array([.7,.4,.8])+t*velocity
        # Change angular coverage of a <10 cm tall patch. A box centroid is
        # unreliable here; the measured sphere centre remains identifiable.
        points=center+.1*unit[unit[:,1]>(-.9 if k%2 else -.3)]
        mapping.current_points=points
        mapping.track(points,t+1.)
    assert len(mapping.tracks)==1
    track=next(iter(mapping.tracks.values()))
    np.testing.assert_allclose(track['velocity'],velocity,atol=1e-9)
    assert track['moving']==bool(np.linalg.norm(velocity)>.055)


def test_measured_start_is_exact_and_replayable_without_changing_terminal_task():
    from scipy.spatial.transform import Rotation
    from jev_agent_wbc.candidates import path_spec,restore_path,path_hash
    from jev_agent_wbc.intent_paths import positions_at
    runtime=json.loads((ROOT/'movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'movein/task.json',runtime['subject']['position'])
    p,r,_=original.sample(0.)
    p=p+np.array([.01,-.02,.015]);r=Rotation.from_rotvec([.01,.02,-.015]).as_matrix()@r
    boundary=dict(position=p.tolist(),quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),twist=[0.]*6)
    path=generate(original,original,{},0.,runtime['subject'],None,SearchBudget(),options('regular_arch')['left'],initial_boundary=boundary)[0]
    restored=restore_path(original,path_spec(path))
    assert path_hash(path)==path_hash(restored)
    for value in (path,restored):
        np.testing.assert_allclose(value.sample(0.)[0],p,atol=1e-12)
        np.testing.assert_allclose(value.sample(0.)[1],r,atol=1e-12)
        np.testing.assert_allclose(positions_at(value,[0.])[0],p,atol=1e-12)
        np.testing.assert_allclose(value.sample(8.)[0],original.sample(8.)[0]+np.asarray(value.program['endpoint_offset_m']),atol=1e-10)
    assert original.sha256==path.original.sha256


def test_initial_witness_fingerprint_changes_with_state_map_or_limits():
    from dataclasses import replace
    from jev_agent_wbc.task_space_planning import verification_snapshot
    from jev_agent_wbc.whole_body_qp import Limits
    state=np.zeros(9);velocity=np.zeros(8);scene={'observation_stamp':0.,'surfel_centers':[[1.,0.,0.]]}
    args=[scene,state,velocity,0.,.002,Limits(),velocity,state[3:],None]
    old=verification_snapshot(*args)
    for index,other in [(0,dict(scene,observation_stamp=.1)),(1,state+.001),(2,velocity+.001),
                        (3,.02),(4,.003),(5,replace(Limits(),wheel_speed=1.)),(6,velocity+.01),
                        (7,state[3:]+.01),(8,np.eye(3))]:
        changed=args.copy();changed[index]=other
        assert verification_snapshot(*changed)!=old


def test_transient_local_disagreement_never_suppresses_new_risk_or_persistent_failure():
    from copy import deepcopy
    from jev_agent_wbc.engine import needs_configuration_confirmation
    diagnosis=dict(snapshot_time_s=2.5,report=dict(issues=[],actual_tracking_error_m=.0002),
                   kinematics=dict(feasible=True))
    feedback=[dict(error='trajectory_preserving_configuration_degraded',time_s=2.52,
                   wbc_feasibility=dict(camera_actuator_feasible=True,
                       barrier_mode='acceleration_and_velocity',first_predicted_conflict_s=None,pairs=[]))]
    assert needs_configuration_confirmation(diagnosis,feedback,2.54)
    assert not needs_configuration_confirmation(diagnosis,feedback,3.4)
    fresh=dict(diagnosis,snapshot_time_s=3.)
    assert not needs_configuration_confirmation(fresh,feedback,3.02)
    local_task_slack=deepcopy(feedback)
    local_task_slack[0]['wbc_feasibility']['camera_actuator_feasible']=False
    assert needs_configuration_confirmation(diagnosis,local_task_slack,2.54)
    for field,value in [('first_predicted_conflict_s',2.7),('pairs',[{'clearance_m':-.01}])]:
        changed=deepcopy(feedback);changed[0]['wbc_feasibility'][field]=value
        assert not needs_configuration_confirmation(diagnosis,changed,2.54)
    for field,value in [('issues',['new_obstacle']),('actual_tracking_error_m',.016)]:
        changed=deepcopy(diagnosis);changed['report'][field]=value
        assert not needs_configuration_confirmation(changed,feedback,2.54)


def test_random_redundant_fixtures_preserve_original_six_dof_camera_pose(tmp_path):
    import importlib.util
    from scipy.spatial.transform import Rotation
    from jev_agent_wbc.whole_body_qp import whole_body_kinematics
    source=ROOT.parent/'scripts/benchmark_directed_jev.py'
    spec=importlib.util.spec_from_file_location('directed_benchmark_fixture',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.perturb(ROOT,tmp_path/'scenarios',20261019)
    for case in (tmp_path/'scenarios').iterdir():
        document=json.loads((case/'runtime_scene.json').read_text())
        original=OriginalReference(case/'task.json',document['subject']['position'])
        p0,r0,_=original.sample(0.)
        p,r,_=whole_body_kinematics(np.asarray(document['initial_state']),document['base_z'])
        assert np.linalg.norm(p-p0)<2e-11
        assert Rotation.from_matrix(r@r0.T).magnitude()<1e-10
        assert np.linalg.norm(document['perturbation']['initial_state_delta'])>.001
        assert document['perturbation']['initial_mode']=='redundant'
        assert (case/'task.json').read_bytes()==(ROOT/case.name/'task.json').read_bytes()


def test_joint_construction_starts_from_one_bound_analytic_seed(monkeypatch):
    from scipy.spatial.transform import Rotation
    from jev_agent_wbc import regular_curve_search
    from jev_agent_wbc.candidates import path_spec,restore_path,path_hash
    def unexpected_fit(*args,**kwargs):
        raise AssertionError('directed construction must not pre-optimize a different camera path')
    monkeypatch.setattr(regular_curve_search,'minimize',unexpected_fit)
    runtime=json.loads((ROOT/'pedestal_up/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'pedestal_up/task.json',runtime['subject']['position'])
    p,r,_=original.sample(0.)
    boundary=dict(position=p.tolist(),quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(),twist=[0.]*6)
    paths=generate(original,original,{},0.,runtime['subject'],None,SearchBudget(),
                   options('joint_curve')['left'],initial_boundary=boundary)
    assert len(paths)==1
    path=paths[0];points=np.asarray(path.geometry['positions'])
    normal=np.asarray(path.program['construction_plane_normal_world'])
    assert np.max(abs((points-p)@normal))<1e-10
    assert np.max((points-p)@np.asarray(path.program['construction_axis_world']))>.29
    restored=restore_path(original,path_spec(path))
    assert path_hash(restored)==path_hash(path)
    assert restored.program['construction_plane_tolerance_m']==.025
