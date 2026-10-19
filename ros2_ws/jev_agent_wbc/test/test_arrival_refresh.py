import json
from pathlib import Path
import numpy as np
import pytest

from jev_agent_wbc.arrival_refresh import without_detour,screen,path_length,propose
from jev_agent_wbc.continuous_combo import action_task
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.phase_program import build
from jev_agent_wbc.transfer_geometry import prepare_arrival
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from jev_agent_wbc.intent_paths import IntentPath
from jev_agent_wbc.reference import smoothstep


@pytest.fixture
def arrival():
    root=Path(__file__).resolve().parents[1]/'scenarios'
    task=json.loads((root/'combo_02_easy/task.json').read_text())
    scene=json.loads((root/'combo_02_easy/runtime_scene.json').read_text())
    nominal=json.loads((root/'video_combo/combo_02_easy_nominal.json').read_text())
    state=np.array(nominal['actions'][1]['states'][0]);z=scene['base_z']
    original=OriginalReference(action_task(task,task['shot_phase'][1]),scene['subject']['position'])
    name=original.task['shot_phase'][0]['name']
    path=build(original,{name:'original'},scene['subject'],[.72,.96,.5,.5],{name:'positive_broad'})
    simple,_=prepare_arrival(without_detour(path),state,z)
    observed=dict(format='jvs_lidar_surfels/v1',surfel_centers=[[3.5,1.,.2]],surfel_radius=.03,
                  surfel_velocities=[[0.,0.,0.]],surfel_track_ids=[0],observation_stamp=10.)
    return state,z,path,simple,observed


def test_removing_bow_preserves_selected_endpoint_timing_and_original(arrival):
    state,z,path,_,_=arrival
    geometry=dict(path.geometry);t=np.asarray(geometry['times']);offset=np.array([.004,-.002,.001])
    geometry['positions']=(np.asarray(geometry['positions'])+smoothstep(t/path.duration)[:,None]*offset).tolist()
    path=IntentPath(path.original,path.previous,0.,path.program,path.model,path.intrinsics,geometry=geometry)
    simple=without_detour(path)
    assert path_length(simple)<path_length(path)-.1
    assert simple.original is path.original and simple.duration==path.duration
    assert simple.program['phase_timing']==path.program['phase_timing']
    for t in (0.,path.duration):
        np.testing.assert_allclose(simple.sample(t)[0],path.sample(t)[0],atol=1e-9)
        np.testing.assert_allclose(simple.sample(t)[1],path.sample(t)[1],atol=1e-9)
    assert without_detour(simple) is None


def test_departed_obstacle_allows_shorter_path_but_static_occupancy_does_not(arrival):
    state,z,_,simple,scene=arrival
    p=whole_body_kinematics(state,z)[0]
    departed=dict(scene,surfel_centers=[(p+[1.3,0.,0.]).tolist()],surfel_track_ids=[1],
                  surfel_velocities=[[.3,0.,0.]])
    proof,report=screen(simple,state,np.zeros(8),departed,z,10.)
    assert proof is not None and report['eligible']
    occupied=dict(scene,surfel_centers=[p.tolist()])
    proof,report=screen(simple,state,np.zeros(8),occupied,z,10.)
    assert proof is None and report['reason']=='observed_corridor_occupied'


def test_shorter_proposal_is_rechecked_if_mover_returns(arrival):
    state,z,_,simple,scene=arrival
    proof,_=screen(simple,state,np.zeros(8),scene,z,10.)
    assert proof is not None
    p=whole_body_kinematics(state,z)[0]
    returning=dict(scene,surfel_centers=[(p+[.8,0.,0.]).tolist()],surfel_track_ids=[1],
                   surfel_velocities=[[-.6,0.,0.]])
    proof,report=screen(simple,state,np.zeros(8),returning,z,10.)
    assert proof is None and report['reason'] in ('observed_motion_conflict','observed_corridor_occupied')


@pytest.mark.parametrize('missing',[False,True])
def test_missing_or_stale_map_cannot_clear_a_detour(arrival,missing):
    state,z,_,simple,scene=arrival
    if missing:scene=dict(scene,surfel_centers=[])
    proof,report=screen(simple,state,np.zeros(8),scene,z,10.3)
    assert proof is None and report['reason']=='missing_or_stale_observation'


def test_verified_shorter_path_is_offered_to_typed_agent(arrival):
    state,z,path,_,scene=arrival
    calls=[]
    def choose(evidence,questions):
        calls.append(evidence)
        assert set(questions)=={'path'}
        return {'path':{'choice':'shorter'}}
    candidate,record=propose(path,state,np.zeros(8),scene,z,10.,choose)
    assert candidate is not None and record['selected'] and len(calls)==1
    assert calls[0]['stage']=='refresh_obsolete_arrival_detour'
    assert record['new_length_m']<record['old_length_m']
