"""Execution-model separation: ideal camera geometry and observer-only WBC."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jev_agent_wbc.floating_camera import CameraObserver, evaluate_path, camera_candidates
from jev_agent_wbc.candidates import SearchBudget
from jev_agent_wbc.planning import TrackingOnlyWBC, validate_rollout
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.whole_body_qp import Limits, WholeBodyQP, whole_body_kinematics
from jev_agent_wbc.safety import solve_with_certificate

ROOT=Path(__file__).resolve().parents[1]


def scene_at(center):
    return dict(format='jvs_lidar_surfels/v1',surfel_centers=np.array([center]),surfel_radius=.1,
        surfel_velocities=np.zeros((1,3)),observation_stamp=0.,minimum_clearance_m=.025)


def test_tracking_only_command_is_independent_of_obstacles():
    runtime=json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    state=np.array(runtime['initial_state']);z=runtime['base_z']
    p,r,_=whole_body_kinematics(state,z)
    limits=Limits(obstacle_avoidance=False)
    near=TrackingOnlyWBC(limits,scene_at(p))
    far=TrackingOnlyWBC(limits,scene_at([5,5,5]))
    reference=OriginalReference(ROOT/'scenarios/movein/task.json',runtime['subject']['position'])
    near.set_trajectory(reference);far.set_trajectory(reference)
    a,info=solve_with_certificate(near,state,p+[.01,0,0],r,np.zeros(6),.02,base_z=z)
    b,_=solve_with_certificate(far,state,p+[.01,0,0],r,np.zeros(6),.02,base_z=z)
    np.testing.assert_allclose(a,b,atol=1e-12)
    assert info['command_sweep_lower_m']<0
    assert info['dynamic_constraint_count']==0 and info['collision_audit_only']
    assert 'safety_brake' not in info and 'certified_escape' not in info
    copy=deepcopy(near)
    assert copy.solver is None
    np.testing.assert_array_equal(copy.previous,near.previous)


def test_no_avoidance_solver_failure_does_not_invoke_escape():
    class Broken:
        obstacle_avoidance=False
        def solve(self,*args,**kwargs):raise RuntimeError('actuator infeasible')
    with pytest.raises(RuntimeError,match='actuator infeasible'):
        solve_with_certificate(Broken(),None,None,None,None,.02)


def test_collision_still_rejects_tracking_only_planning():
    runtime=json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    reference=OriginalReference(ROOT/'scenarios/movein/task.json',runtime['subject']['position'])
    state=np.array(runtime['initial_state']);z=runtime['base_z']
    p=whole_body_kinematics(state,z)[0]
    result=validate_rollout(reference,scene_at(p),state,np.zeros(8),0.,z,
        Limits(obstacle_avoidance=False),horizon=.1)
    assert not result.feasible and result.reason=='predicted_tracking_collision'


def test_camera_observer_uses_past_rays_and_estimates_motion():
    runtime=RuntimeScene(ROOT/'scenarios/movein/runtime_scene.json')
    reference=OriginalReference(ROOT/'scenarios/movein/task.json',runtime.document['subject']['position'])
    observer=CameraObserver()
    for t in np.linspace(-.8,0.,9):
        scene=observer.scan(runtime,reference.sample(0.)[0],float(t))
    sphere=runtime.obstacles(0.)[0]
    np.testing.assert_allclose(scene['centers'][0],sphere['center'],atol=1e-8)
    assert scene['valid'] and scene['observation_stamp']==0.
    np.testing.assert_allclose(scene['velocities'][0],[.035,-.24,0.],atol=1e-8)
    assert not any(k in scene for k in ('keyframes','initial_state','base_z'))


def test_ideal_camera_rollout_has_no_robot_state_and_tracks_exactly(monkeypatch):
    import jev_agent_wbc.whole_body_qp as robot
    monkeypatch.setattr(robot,'whole_body_kinematics',lambda *a,**k: pytest.fail('robot FK used'))
    runtime=RuntimeScene(ROOT/'scenarios/movein/runtime_scene.json')
    reference=OriginalReference(ROOT/'scenarios/movein/task.json',runtime.document['subject']['position'])
    scene=dict(valid=True,observation_stamp=0.,centers=[[5.,5.,5.]],radii=[.1],velocities=[[0.,0.,0.]])
    value,_=evaluate_path(reference,scene,0.,.2,.06,visual=True)
    assert value.feasible and value.joint_margin_rad is None
    for row in value.camera_trace:
        np.testing.assert_array_equal(row['position'],reference.sample(row['t'])[0])
        assert 'state' not in row and row['joint_margin_rad'] is None


def test_camera_segment_check_catches_between_sample_crossing():
    class Path:
        duration=.02
        def sample(self,t):return np.zeros(3),np.eye(3),np.zeros(6)
    scene=dict(valid=True,observation_stamp=0.,centers=[[-1.,0.,0.]],radii=[.01],velocities=[[100.,0.,0.]])
    value,_=evaluate_path(Path(),scene,0.,.02,.01,step=.02)
    assert not value.feasible and value.min_clearance_m==pytest.approx(-.02)


def test_logged_target_reconstruction_is_exact_and_bound_to_original():
    from jev_agent_wbc.candidates import restore_path, path_hash, propose_paths, path_spec
    runtime=RuntimeScene(ROOT/'scenarios/movein/runtime_scene.json')
    original=OriginalReference(ROOT/'scenarios/movein/task.json',runtime.document['subject']['position'])
    caps=dict(radial_m=1.5,lateral_m=1.5,height_m=.65,yaw_rad=np.pi,pitch_rad=np.pi/2,roll_rad=np.pi/2)
    path=propose_paths(original,original,'view',0.,SearchBudget(activation_lead_s=1.5),caps)[1]
    spec=path_spec(path)
    restored=restore_path(original,spec)
    assert path_hash(restored)==path_hash(path)
    for t in np.linspace(0,8,37):
        for a,b in zip(restored.sample(t),path.sample(t)):
            np.testing.assert_array_equal(a,b)
    spec['original_sha256']='different-task'
    with pytest.raises(ValueError,match='reference hash mismatch'):
        restore_path(original,spec)


@pytest.mark.parametrize('planning_override,expected', [(None,False),(True,True)])
def test_agent_planning_controller_is_independent_of_execution(tmp_path,monkeypatch,planning_override,expected):
    from jev_agent_wbc.engine import CinematicEngine
    from jev_agent_wbc.planning import make_wbc
    runtime=RuntimeScene(ROOT/'scenarios/movein/runtime_scene.json')
    original=OriginalReference(ROOT/'scenarios/movein/task.json',runtime.document['subject']['position'])
    scene=scene_at([5,5,5])
    engine=CinematicEngine(original,scene,tmp_path,mode='rule',
        limits=Limits(obstacle_avoidance=False),planning_obstacle_avoidance=planning_override)
    submitted=[]
    monkeypatch.setattr(engine,'_submit_math',lambda kind,fn,*args,**kwargs: submitted.append((kind,args)))
    try:
        state=np.array(runtime.document['initial_state']);z=runtime.document['base_z']
        engine.update(state,scene,0.,z)
        # Configuration selection now follows family selection before numeric
        # planning. Verify this stage still uses the separate planning limits.
        monkeypatch.setattr(engine.jev,'poll',lambda: dict(stage='configuration',strategy='view',search_config='local'))
        engine.update(state,scene,.02,z)
        assert [kind for kind,_ in submitted]==['diagnose','candidates']
        for _,args in submitted:
            limits=next(arg for arg in args if isinstance(arg,Limits))
            assert limits.obstacle_avoidance is expected
            predicted=make_wbc(limits,scene,original)
            assert isinstance(predicted,TrackingOnlyWBC) is (not expected)
        assert isinstance(engine.wbc,TrackingOnlyWBC)
        assert engine.wbc.limits.obstacle_avoidance is False
    finally:
        engine.close()
