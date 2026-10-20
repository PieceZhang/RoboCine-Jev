from types import SimpleNamespace as NS
import numpy as np
from jev_agent_wbc.observed_clock import reachable_guide_step
from jev_agent_wbc.whole_body_qp import Limits


def test_delayed_robot_forecast_cannot_jump_to_nominal_configuration():
    state=np.array([0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])
    expected=state.copy();expected[0]=2.;expected[3]+=.8
    guide=dict(times=np.array([0.,1.,2.]),states=np.tile(expected,(3,1)),controls=np.ones((2,8)))
    limits=Limits();previous=np.zeros(8)
    predicted,command=reachable_guide_step(guide,state,previous,1.,1.8,.1,limits)
    assert np.linalg.norm(predicted[:2]-state[:2])<=limits.acceleration[0]*.1**2+1e-9
    assert np.max(abs(command-previous)-limits.acceleration*.1)<=1e-10
    assert np.max(abs(limits.wheels@(command-previous)))<=limits.wheel_acceleration*.1+1e-10
    assert np.linalg.norm(predicted-expected)>1.
    np.testing.assert_array_equal(state,[0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])


def test_each_clock_forecast_step_preserves_wheel_and_joint_rate_bounds():
    state=np.array([0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])
    guide=dict(times=np.linspace(0.,3.,31),states=np.tile(state,(31,1)),controls=np.ones((30,8)))
    guide['states'][:,0]=np.linspace(0.,1.,31)
    limits=Limits();previous=np.zeros(8)
    for stamp in np.arange(.1,2.51,.1):
        state,command=reachable_guide_step(guide,state,previous,stamp,1.8,.1,limits)
        assert np.max(abs(command)-limits.velocity)<=1e-10
        assert np.max(abs(command-previous)-limits.acceleration*.1)<=1e-10
        assert np.max(abs(limits.wheels@command))<=limits.wheel_speed+1e-10
        assert np.max(abs(limits.wheels@(command-previous)))<=limits.wheel_acceleration*.1+1e-10
        previous=command


def test_remote_actor_does_not_retime_an_unthreatened_authored_shot():
    import json
    from pathlib import Path
    from jev_agent_wbc.observed_clock import update
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.dynamic_collision import DynamicCollision
    folder=Path(__file__).resolve().parents[1]/'scenarios/movein'
    scene=json.loads((folder/'runtime_scene.json').read_text())
    reference=OriginalReference(folder/'task.json',scene['subject']['position'])
    observed=dict(format='jvs_lidar_surfels/v1',surfel_centers=[[9.,9.,.4]],surfel_radius=.1,
        surfel_track_ids=[1],surfel_velocities=[[.1,0.,0.]],observation_stamp=0.)
    controller=NS(collision=DynamicCollision(observed),reconfiguration=NS(pool=None),limits=Limits())
    info=update(controller,reference,np.array(scene['initial_state']),np.zeros(8),0.,scene['base_z'])
    assert not info['active'] and info['rate']==1.
    assert 'body_control_preference' not in info
