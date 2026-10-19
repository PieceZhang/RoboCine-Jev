import numpy as np
from scipy.spatial.transform import Rotation
from jev_agent_wbc.transfer_geometry import capture_reference,fresh_reference
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.phase_program import build
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.motion_forecast import forecast


def test_arrival_capture_preserves_terminal_and_matches_measured_start():
    state=np.r_[0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.]
    p,r,_=whole_body_kinematics(state,0.);subject=p+[1.5,0.,0.]
    task=dict(intent='move closer',subject=dict(id='subject',description='test subject'),shot_phase=[dict(name='move',motion='movein',start_s=0.,end_s=7.)],
        reference_camera_trajectory=dict(frame='world_z_up_camera_link',samples=[dict(t=float(t),position=(p+[.1*t/7,0,0]).tolist(),quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist()) for t in np.linspace(0,7,41)]))
    original=OriginalReference(task,subject);path=build(original,{'move':'original'})
    path.geometry['joint_seed']=[dict(t=0.,state=(state+1).tolist(),control=np.zeros(8).tolist())]
    path.joint_seed=path.geometry['joint_seed']
    measured=state.copy();measured[0]+=.002
    captured=capture_reference(path,measured,np.zeros(8))
    assert not captured.joint_seed and path.joint_seed
    np.testing.assert_allclose(captured.sample(0.)[0],whole_body_kinematics(measured,0.)[0],atol=1e-10)
    np.testing.assert_allclose(captured.sample(7.)[0],path.sample(7.)[0],atol=1e-10)
    np.testing.assert_allclose(captured.sample(0.)[1],r,atol=1e-10)
    assert captured.original is original and captured.duration==7.


def test_acceleration_forecast_is_bounded_observation_only_and_does_not_mutate_safety_map():
    scene=dict(format='jvs_lidar_surfels/v1',surfel_centers=[[2.,0.,.2]],surfel_radius=.1,surfel_velocities=[[-.2,0.,0.]],surfel_track_ids=[1],
        observation_stamp=1.,tracks=[dict(id=1,age_s=0.,acceleration=[-.4,0.,0.])])
    collision=DynamicCollision(scene,1.);before=collision.centers.copy()
    value=forecast(collision,2.)
    np.testing.assert_allclose(value.centers[0,0],2.-.2*2.-.4*(.5*2.-.5*.5**2))
    np.testing.assert_array_equal(collision.centers,before)
    collision.set_time(1.3)
    np.testing.assert_allclose(forecast(collision,2.).centers,collision.at_time(2.).centers)


def test_original_reference_can_select_a_goal_without_intent_geometry():
    import json
    from pathlib import Path
    from jev_agent_wbc.continuous_combo import action_task
    from jev_agent_wbc.transfer_geometry import nearby_goal
    root=Path(__file__).resolve().parents[1]/'scenarios'
    task=json.loads((root/'combo_02_easy/task.json').read_text())
    scene=json.loads((root/'combo_02_easy/runtime_scene.json').read_text())
    state=np.asarray(scene['initial_state'])
    original=OriginalReference(action_task(task,task['shot_phase'][0]),scene['subject']['position'])
    observed=dict(format='jvs_lidar_surfels/v1',surfel_centers=[[4.,4.,.2]],surfel_radius=.05,
        surfel_velocities=[[0.,0.,0.]],surfel_track_ids=[0],observation_stamp=0.)
    goal,report=nearby_goal(state,state,original,observed,scene['base_z'])
    assert np.isfinite(goal).all() and report['next_shot_kinematics']['feasible']
    np.testing.assert_allclose(whole_body_kinematics(goal,scene['base_z'])[0],original.sample(0.)[0],atol=.003)
