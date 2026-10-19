import numpy as np
import pytest
from jev_agent_wbc.free_transfer import FreeTransferAgent,route_cards,nondominated_routes
from jev_agent_wbc.whole_body_qp import whole_body_kinematics

STATE=np.r_[0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.]
SCENE=dict(format='jvs_lidar_surfels/v1',surfel_centers=[[3.,3.,.3]],surfel_radius=.07,
    surfel_velocities=[[0.,0.,0.]],surfel_track_ids=[0],observation_stamp=0.,minimum_clearance_m=.025)


def choose(evidence,questions):
    assert 'goal_state' in evidence and set(questions)=={'route'}
    return dict(route=dict(choice='direct'))


def test_transfer_moves_configuration_without_camera_equality_and_keeps_actuator_bounds():
    goal=STATE.copy();goal[0]=.25;goal[4]+=.1
    agent=FreeTransferAgent(STATE,goal,SCENE,choose)
    try:
        command,info=agent.step(STATE,SCENE,0.)
        J=whole_body_kinematics(STATE,0.)[2]
        assert np.linalg.norm(J@command)>1e-5
        assert info['free_camera_transfer']
        assert np.all(abs(command)<=agent.wbc.limits.acceleration*.02+1e-6)
        assert info['command_sweep_lower_m']>=.025
        assert agent.wbc.limits.deadline_s is None and not agent.ready(STATE)
    finally:agent.close()


def test_camera_pose_match_alone_is_not_transfer_readiness():
    goal=STATE.copy();goal[2]+=.1
    agent=FreeTransferAgent(STATE,goal,SCENE,choose)
    try:assert not agent.ready(STATE)
    finally:agent.close()


def test_route_evidence_exposes_unnecessary_steering_without_claiming_safety():
    goal=STATE.copy();goal[:2]=[.08,.04]
    cards=route_cards(STATE,goal,SCENE)
    assert cards['direct']['platform_travel_m']<cards['right']['platform_travel_m']
    assert cards['direct']['platform_rotation_rad']<cards['right']['platform_rotation_rad']
    assert all('NOT a future-motion or safety certificate' in card['scope'] for card in cards.values())
    assert nondominated_routes(cards)==['direct']


def test_route_filter_preserves_clear_detour_when_direct_is_blocked():
    cards={'direct':dict(platform_travel_m=.2,platform_rotation_rad=0.,frozen_observed_map_sampled_clearance_m=-.01),
        'left':dict(platform_travel_m=.5,platform_rotation_rad=1.,frozen_observed_map_sampled_clearance_m=.2)}
    assert 'left' in nondominated_routes(cards)


def test_observed_mover_can_keep_a_detour_even_when_frozen_direct_map_is_clear():
    cards={'direct':dict(platform_travel_m=.2,platform_rotation_rad=0.,frozen_observed_map_sampled_clearance_m=.3,observed_motion_lease_clearance_m=-.02),
        'left':dict(platform_travel_m=.5,platform_rotation_rad=1.,frozen_observed_map_sampled_clearance_m=.2,observed_motion_lease_clearance_m=.15)}
    assert 'left' in nondominated_routes(cards)


def test_boundary_velocity_is_preserved_for_acceleration_limits():
    goal=STATE.copy();goal[0]=.25
    velocity=np.r_[.15,.12,np.zeros(6)]
    agent=FreeTransferAgent(STATE,goal,SCENE,choose,initial_control=velocity)
    try:
        command,_=agent.step(STATE,SCENE,0.)
        assert np.all(abs(command-velocity)<=agent.wbc.limits.acceleration*.02+1e-6)
    finally:agent.close()


def test_free_transfer_cannot_bypass_an_installed_shot_deadline():
    agent=FreeTransferAgent(STATE,STATE,SCENE,choose)
    try:
        agent.wbc.limits.deadline_s=7.
        with pytest.raises(ValueError,match='cannot override'):agent.step(STATE,SCENE,0.)
    finally:agent.close()


def test_complementary_drive_and_yaw_wrap_choose_short_turns():
    from jev_agent_wbc.transfer_geometry import approach_sign,angle
    goal=STATE.copy();goal[0]=-.2
    assert approach_sign(STATE,goal,goal[:2])==-1.
    state=STATE.copy();state[2]=np.pi-.02;goal[2]=-np.pi+.02
    assert approach_sign(state,goal,goal[:2])==1.
    assert abs(angle(goal[2]-state[2])-.04)<1e-12


def test_final_lateral_residual_does_not_restart_a_complementary_loop():
    from jev_agent_wbc.free_transfer import configuration_control
    goal=STATE.copy();goal[:2]=[.002,.009]
    navigation=dict(sign=1.,align=True)
    command=configuration_control(STATE,goal,[goal[:2].copy()],1.,navigation)
    assert navigation['align'] and abs(command[1])<1e-12
    assert abs(command[0])<.01


def test_same_route_reselection_keeps_original_waypoint():
    goal=STATE.copy();goal[:2]=[.3,.1]
    agent=FreeTransferAgent(STATE,goal,SCENE,choose)
    try:
        agent._install('left',STATE,0.);expected=agent.waypoints[0].copy()
        moved=STATE.copy();moved[:2]=[.1,.06]
        agent._install('left',moved,1.)
        np.testing.assert_array_equal(expected,agent.waypoints[0])
    finally:agent.close()


def test_late_or_old_goal_api_choice_cannot_retarget_connector():
    goal=STATE.copy();goal[0]=.2;agent=FreeTransferAgent(STATE,goal,SCENE,choose)
    try:
        proposal=dict(route='left',goal=goal.copy(),time_s=0.)
        assert not agent.accept_route(proposal,STATE,SCENE,2.)
        proposal['time_s']=2.;proposal['goal'][0]+=.1
        assert not agent.accept_route(proposal,STATE,SCENE,2.1)
        assert agent.route=='direct'
    finally:agent.close()


def test_close_camera_pose_is_not_ready_until_velocity_and_boundary_settle():
    agent=FreeTransferAgent(STATE,STATE,SCENE,choose)
    try:
        assert agent.ready(STATE)
        agent.wbc.previous[1]=.01
        assert not agent.ready(STATE)
        agent.wbc.previous[:]=0.;shifted=STATE.copy();shifted[0]+=.006
        assert not agent.ready(shifted)
    finally:agent.close()


def test_next_stop_command_cannot_replace_measured_arrival_velocity():
    agent=FreeTransferAgent(STATE,STATE,SCENE,choose)
    try:
        measured=np.zeros(8);measured[2]=.009
        assert not agent.ready(STATE,measured)  # New command is zero, body is still moving.
        assert agent.ready(STATE,np.zeros(8))
        agent.wbc.previous[0]=.01
        assert not agent.ready(STATE,np.zeros(8))  # Do not hand over a pending acceleration.
    finally:agent.close()


def test_escape_past_goal_reselects_reverse_instead_of_full_turn():
    goal=STATE.copy();goal[0]=.2;agent=FreeTransferAgent(STATE,goal,SCENE,choose)
    try:
        displaced=STATE.copy();displaced[0]=.35
        command,_=agent.step(displaced,SCENE,2.)
        assert agent.navigation['sign']==-1. and command[0]<0.
        assert abs(command[1])<.05  # Small redundant correction, never a large turn.
    finally:agent.close()


def test_unneeded_old_side_waypoint_is_removed_after_corridor_clears():
    goal=STATE.copy();goal[0]=.3;agent=FreeTransferAgent(STATE,goal,SCENE,choose)
    try:
        agent._install('left',STATE,0.)
        displaced=STATE.copy();displaced[:2]=[.15,.01]
        agent.step(displaced,SCENE,2.)
        assert agent.route=='direct' and len(agent.waypoints)==1
        np.testing.assert_allclose(agent.waypoints[0],goal[:2])
    finally:agent.close()
