import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from jev_agent_wbc.body_timing import clock_basis,construct
from jev_agent_wbc.body_threat import update
from jev_agent_wbc.directed_search import options
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.kinematic_feasibility import KinematicSearch
from jev_agent_wbc.whole_body_qp import Limits,whole_body_kinematics
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.candidates import path_hash,path_spec,restore_path
from test_tpr_wbc import STATE,Reference,scene


def test_phase_clock_keeps_c2_joins_and_does_not_change_other_action():
    phases=[dict(start_s=0.,end_s=5.),dict(start_s=5.,end_s=10.)]
    for at in (1.,5.,10.):
        times=np.array([at-1e-6,at,at+1e-6])
        b,d=clock_basis(times,phases,1.)
        np.testing.assert_allclose(b[1],0.,atol=1e-12)
        np.testing.assert_allclose(d[1],0.,atol=1e-12)
        assert np.max(abs((d[2]-d[0])/2e-6))<1e-4
    b,_=clock_basis(np.array([3.,7.]),phases,1.)
    assert np.all(b[0,2:]==0.) and np.all(b[1,:2]==0.)


def test_body_clock_zero_change_in_clear_scene_and_exact_replay():
    folder=Path(__file__).resolve().parents[1]/'scenarios/movein'
    runtime=json.loads((folder/'runtime_scene.json').read_text())
    original=OriginalReference(folder/'task.json',runtime['subject']['position'])
    observed=scene((9.,9.,9.));observed['surfel_track_ids']=[0]
    search=KinematicSearch(runtime['initial_state'],np.zeros(8),0.,runtime['base_z'],Limits(),scene=observed)
    paths=construct(original,original,observed,1.,runtime['subject'],None,search,options('body_timing')['adaptive'])
    assert len(paths)==1
    path=paths[0];replay=restore_path(original,path_spec(path))
    assert path_hash(path)==path_hash(replay)
    assert not np.any(path.program['clock_coefficients'])
    for at in np.linspace(0.,original.duration,100):
        np.testing.assert_allclose(path.sample(at)[0],original.sample(at)[0],atol=2e-4)
    assert path.duration==original.duration


def test_remote_passers_do_not_activate_body_adaptation_and_clearance_recovers():
    c=SimpleNamespace(collision=DynamicCollision(scene((3.,3.,.3),(.1,0.,0.)),0.))
    reference=Reference();p,r,_=whole_body_kinematics(STATE)
    assert not update(c,reference,STATE,p,r,0.)['body_adaptation_active']
    c.collision=DynamicCollision(scene((-.5,0.,.16),(.3,0.,0.)),0.)
    assert update(c,reference,STATE,p,r,.1)['body_adaptation_active']
    c.collision=DynamicCollision(scene((3.,3.,.3),(.1,0.,0.)),0.)
    assert update(c,reference,STATE,p,r,.2)['body_adaptation_active']
    assert not update(c,reference,STATE,p,r,.8)['body_adaptation_active']


def test_offline_vectorized_witness_matches_actual_cpu_integration():
    from CineGym.solvability import integrate_controls
    from jev_agent_wbc.kc_svsdf import advance
    commands=np.random.default_rng(7).normal(size=(100,8))*.2
    states=[STATE]
    for command in commands:states.append(advance(states[-1],command,.02))
    np.testing.assert_allclose(integrate_controls(STATE,commands,.02),states,atol=1e-12)
