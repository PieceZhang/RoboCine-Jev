"""Regression witnesses: legal bypasses and current sparse motion evidence."""
from pathlib import Path
import numpy as np
import pytest
from jev_agent_wbc.sensors import RuntimeLidar
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.candidates import SearchBudget
from jev_agent_wbc.directed_search import options
from jev_agent_wbc.task_space_planning import build
from jev_agent_wbc.whole_body_qp import Limits

SCENARIOS=Path(__file__).resolve().parents[1]/'scenarios'


def initial(case):
    runtime=RuntimeScene(SCENARIOS/case/'runtime_scene.json')
    state=np.array(runtime.document['initial_state'],float)
    lidar=RuntimeLidar()
    for t in np.linspace(-.8,0.,9):
        observed=lidar.synthetic_scan(runtime,state,runtime.document['base_z'],float(t))
    return runtime,state,observed


def test_sparse_current_sphere_returns_establish_motion_before_shot():
    _,_,observed=initial('orbit')
    moving=[r for r in observed['tracks'] if r.get('moving_confirmed')]
    assert len(moving)==1
    np.testing.assert_allclose(moving[0]['velocity'],[.035,-.24,0.],atol=1e-7)
    assert moving[0]['age_s']==0.


@pytest.mark.parametrize('case,direction',[
    ('movein','left'),('pedestal_up','left'),('pedestal_down','left'),
    ('truck_left','outer'),('truck_right','outer'),('orbit','outer')])
def test_authored_broad_bypass_keeps_independent_feasibility_gates(case,direction):
    runtime,state,scene=initial(case);doc=runtime.document
    original=OriginalReference(SCENARIOS/case/'task.json',doc['subject']['position'])
    fy=.5/np.tan(np.deg2rad(27.5));intrinsics=[fy*288/384,fy,.5,.5]
    scene=dict(scene,subject_model=doc['subject'],intrinsics=intrinsics)
    result=build(original,original,'joint_curve',scene,state,np.zeros(8),0.,doc['base_z'],Limits(),
        budget=SearchBudget(single_candidate=True,max_proposals=1,max_rollouts=1,candidate_wall_budget_s=30.),
        measured_control=np.zeros(8),commanded_q=state[3:],intrinsics=intrinsics,search_config='base',
        construction_plan=options('joint_curve')[direction],initial_boundary=True)
    assert len(result.get('candidates',[]))==1, result.get('evaluations')
    candidate=result['candidates'][0]
    assert candidate.kinematics['feasible']
    assert candidate.kinematics['whole_body_collision_checked']
    assert candidate.kinematics['min_body_clearance_m']>=.025
    assert candidate.kinematics['min_self_clearance_m']>=.004
