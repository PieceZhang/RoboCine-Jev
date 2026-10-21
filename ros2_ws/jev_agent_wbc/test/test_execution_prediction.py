"""Physical transport/response semantics and epistemic limits of forecasts."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jev_agent_wbc.execution_prediction import (ExecutionModel, ExecutionObserver,
    failure_evidence, forecast, motor_step, response_step)
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics


STATE = np.array([0., 0., 0., 0., -.9, 2.1, 0., -1.2, 0.])


def test_forecast_motor_targets_match_physical_adapter_including_wheel_coupling():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from physx_compat import MotorTargets
    from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER
    rng = np.random.default_rng(42)
    limits = Limits()
    physical = MotorTargets(limits)
    for _ in range(50):
        state = STATE.copy()
        state[3:] = rng.uniform(JOINT_LOWER+.02, JOINT_UPPER-.02)
        velocity = rng.uniform(-limits.velocity, limits.velocity)
        command = rng.uniform(-limits.velocity, limits.velocity)
        expected = motor_step(command, velocity, state, limits, .01)
        physical.step(command, .01, velocity, state[3:])
        np.testing.assert_allclose(expected, physical.control, atol=1e-14)


def test_response_is_distinct_from_command_slew_and_keeps_initial_momentum():
    limits = Limits()
    v = np.array([.2, 0., 0., 0., 0., 0., 0., 0.])
    ideal = ExecutionModel(response_s=[0.]*8, bias_velocity=[0.]*8)
    slow = replace(ideal, response_s=[.1]*8)
    fast_state, fast_v, _ = response_step(STATE, v, np.zeros(8), limits, ideal)
    slow_state, slow_v, _ = response_step(STATE, v, np.zeros(8), limits, slow)
    assert slow_v[0] > fast_v[0] > 0
    assert slow_state[0] > fast_state[0] > STATE[0]
    assert np.max(abs(fast_v-v)/limits.acceleration) <= .01+1e-12


def test_relative_self_envelope_contains_perturbed_capsule_distances():
    from jev_agent_wbc.self_collision import SelfCollision
    from jev_agent_wbc.execution_prediction import relative_self_reach
    collision=SelfCollision();reach=relative_self_reach(collision)
    rng=np.random.default_rng(42)
    for _ in range(40):
        state=STATE.copy();state[3:]+=rng.uniform(-.2,.2,6)
        error=rng.uniform(0.,.04,6)
        perturbed=state.copy();perturbed[3:]+=rng.uniform(-1.,1.,6)*error
        lower=collision.distances(state)-reach@error
        assert np.all(collision.distances(perturbed)>=lower-1e-10)


def test_observer_is_causal_reports_breaches_and_never_tightens_bounds():
    observer = ExecutionObserver(ExecutionModel(residual_velocity=[.001]*8))
    limits = Limits()
    observer.observe(0., STATE, np.zeros(8), np.zeros(8), limits, command_delay_s=.04)
    velocity = np.zeros(8);velocity[0]=.02
    evidence = observer.observe(.02, STATE, velocity, velocity, limits, command_delay_s=.06)
    assert evidence['category']=='execution_model_mismatch'
    assert not evidence['root_cause_confirmed']
    widened = observer.model.residual_velocity.copy()
    assert widened[0] >= .02 and observer.model.breaches==1
    # Repeated/stale packets cannot overwrite a newer model timestamp.
    observer.observe(.01, STATE, np.ones(8), np.ones(8), limits)
    assert observer.model.snapshot_time_s==.02
    observer.observe(.04, STATE, velocity, velocity, limits)
    assert np.all(np.array(observer.model.residual_velocity)>=widened)


def test_public_capability_exposes_coupling_time_and_unknown_probability():
    model = ExecutionModel(snapshot_time_s=2.)
    public = model.public(STATE, np.zeros(8), Limits())
    assert public['valid_until_s']==pytest.approx(2.1)
    assert public['expected_command_effect_s']==pytest.approx(2.04)
    assert public['prediction']['probability'] is None
    assert public['prediction']['samples']==0
    assert 'cannot be combined' in public['directional_support_scope']
    assert not {'state','joint_names','response_s','residual_velocity'} & set(public)


@pytest.mark.parametrize('reason,category',[
    ('solver_qp_infeasible','numerical_unverified'),
    ('command_age_expired','time_alignment'),
    ('acceleration_limit','execution_capacity'),
    ('terminal_frame','intent_execution_mismatch'),
    ('measured_subject_frame_mismatch','intent_execution_mismatch'),
    ('network_timeout','network_unverified'),
    ('execution_tube_clearance_unverified','safety_coverage'),
    ('TPR hard constraint intersection is empty','constraint_conflict'),
    ('trajectory_preserving_configuration_degraded','execution_capacity'),
    ('single_candidate_missing_contract','interface_configuration'),
    ('unexplained_failure','unknown')])
def test_failure_categories_do_not_claim_unobserved_physical_root_cause(reason,category):
    evidence = failure_evidence(reason,1.)
    assert evidence['category']==category and not evidence['root_cause_confirmed']


def test_delayed_forecast_continues_committed_motion_before_new_command(monkeypatch):
    from jev_agent_wbc import planning, safety
    state = STATE.copy()
    p,r,_ = whole_body_kinematics(state)
    class Path:
        duration=.12
        def sample(self,t):return p,r,np.zeros(6)
    controller=SimpleNamespace(limits=Limits(),set_time=lambda t:None,close=lambda:None)
    monkeypatch.setattr(planning,'make_wbc',lambda *a,**k:controller)
    monkeypatch.setattr(safety,'solve_with_certificate',lambda *a,**k:(np.zeros(8),{}))
    scene=dict(format='jvs_lidar_surfels/v1',observation_stamp=0.,surfel_centers=[[8.,8.,8.]],surfel_radius=.1,
               surfel_velocities=[[0.,0.,0.]],minimum_clearance_m=.025)
    initial=np.array([.1,0.,0.,0.,0.,0.,0.,0.])
    model=ExecutionModel(command_mode='velocity',delay_s=.06,delay_jitter_s=0.,
        response_s=[0.]*8,residual_velocity=[0.]*8,position_floor_m=0.)
    delayed=forecast(Path(),None,scene,state,initial,initial,0.,0.,Limits(),model,end_s=.12)
    immediate=forecast(Path(),None,scene,state,initial,initial,0.,0.,Limits(),replace(model,delay_s=0.),end_s=.12)
    assert delayed['complete'] and immediate['complete']
    at_04=next(row for row in delayed['trace'] if abs(row['t']-.04)<1e-9)
    assert at_04['state'][0]==pytest.approx(.004,abs=1e-9)
    assert delayed['terminal']['state'][0]>immediate['terminal']['state'][0]


def test_model_validation_rejects_cross_robot_and_nonfinite_calibration():
    with pytest.raises(ValueError):ExecutionModel(robot='another_robot')
    with pytest.raises(ValueError):ExecutionModel(delay_s=float('nan'))
    with pytest.raises(ValueError):ExecutionModel(response_s=[-.01]*8)


def test_acceleration_packet_is_anchored_once_per_control_tick(monkeypatch):
    from jev_agent_wbc import planning, safety
    p,r,_=whole_body_kinematics(STATE)
    class Path:
        duration=.1
        def sample(self,t):return p,r,np.zeros(6)
    controller=SimpleNamespace(limits=Limits(),set_time=lambda t:None,close=lambda:None)
    monkeypatch.setattr(planning,'make_wbc',lambda *a,**k:controller)
    acceleration=np.array([.1,0.,0.,0.,0.,0.,0.,0.])
    monkeypatch.setattr(safety,'solve_with_certificate',
        lambda c,*a,**k:(c.measured_control+.02*acceleration,{}))
    scene=dict(format='jvs_lidar_surfels/v1',observation_stamp=0.,surfel_centers=[[8.,8.,8.]],
        surfel_radius=.1,surfel_velocities=[[0.,0.,0.]],minimum_clearance_m=.025)
    model=ExecutionModel(command_mode='acceleration',delay_s=0.,delay_jitter_s=0.,
        response_s=[0.]*8,residual_velocity=[0.]*8,position_floor_m=0.,
        calibrated_velocity_abs_max=[.003]*8)
    report=forecast(Path(),None,scene,STATE,np.zeros(8),np.zeros(8),0.,0.,Limits(),model,end_s=.04)
    assert report['complete']
    # Re-anchoring every 10 ms would yield .004 and .008, doubling acceleration.
    assert report['trace'][1]['control'][0]==pytest.approx(.002)
    assert report['trace'][2]['control'][0]==pytest.approx(.004)
    assert report['trace'][1]['state'][0]==pytest.approx(.00004)
    assert report['outside_calibrated_velocity_range']
    assert report['calibration_domain_exit_s']==pytest.approx(.04)


def test_applied_transition_does_not_learn_a_rejected_proposal_as_plant_error():
    observer=ExecutionObserver(ExecutionModel(response_s=[0.]*8,residual_velocity=[.001]*8))
    observer.observe(0.,STATE,np.zeros(8),np.ones(8),Limits())
    transition=dict(stamp_s=.02,dt_s=.01,previous_velocity=np.zeros(8),
                    drive_target=np.zeros(8),measured_velocity=np.zeros(8))
    evidence=observer.observe(.02,STATE,np.zeros(8),np.ones(8),Limits(),transition=transition)
    assert evidence is None and observer.model.breaches==0


def test_recent_fast_packets_do_not_shrink_calibrated_jitter():
    observer=ExecutionObserver(ExecutionModel(delay_jitter_s=.06))
    for stamp in (0.,.02,.04):
        observer.observe(stamp,STATE,np.zeros(8),np.zeros(8),Limits(),command_delay_s=.02)
    assert observer.model.delay_jitter_s>=.06


def test_held_packet_age_is_not_learned_as_new_transport_delay():
    observer=ExecutionObserver(ExecutionModel(delay_s=.02,delay_jitter_s=.01))
    for index,age in enumerate((.02,.04,.06,.08)):
        observer.observe(.02*index,STATE,np.zeros(8),np.zeros(8),Limits(),
            command_delay_s=age,command_sequence=7)
    assert observer.model.delay_s==pytest.approx(.02)
    assert observer.model.delay_jitter_s==pytest.approx(.01)
    assert len(observer.delays)==1
    observer.observe(.08,STATE,np.zeros(8),np.zeros(8),Limits(),command_delay_s=.06,command_sequence=8)
    assert len(observer.delays)==2


def test_bias_identification_uses_raw_residual_and_physics_step_duration():
    observer=ExecutionObserver(ExecutionModel(response_s=[0.]*8,residual_velocity=[.01]*8))
    for index in range(200):
        stamp=.02*index
        transition=dict(stamp_s=stamp,dt_s=.01,previous_velocity=np.zeros(8),
            drive_target=np.zeros(8),measured_velocity=np.full(8,.001))
        observer.observe(stamp,STATE,np.full(8,.001),np.zeros(8),Limits(),transition=transition)
    # The model bias is expressed per 20 ms; a 10 ms measurement contributes half.
    np.testing.assert_allclose(observer.model.bias_velocity,np.full(8,.002),atol=1e-7)


def test_forecast_preview_cannot_use_a_future_result_early(monkeypatch):
    from jev_agent_wbc.execution_prediction import delayed_reconfiguration
    from jev_agent_wbc import tpr_wbc
    calls=[]
    def calculate(*args):
        calls.append(True)
        return SimpleNamespace(updated=0.,times=np.array([0.,1.]),
            controls=np.full((2,8),.01),states=np.tile(STATE,(2,1)),anchor=STATE,
            info=dict(tpr_prediction_valid=True))
    monkeypatch.setattr(tpr_wbc,'compute_reconfiguration',calculate)
    preview=delayed_reconfiguration(.4);preview.reset(SimpleNamespace(duration=1.))
    c=SimpleNamespace(collision=SimpleNamespace(scene={'observation_stamp':0.}),
        physical_time=0.,limits=Limits(),previous=np.zeros(8))
    for t in (0.,.2):
        np.testing.assert_array_equal(preview.update(c,STATE,np.zeros(8),t,.02,0.,None,None),np.zeros(8))
    assert not calls
    np.testing.assert_allclose(preview.update(c,STATE,np.zeros(8),.4,.02,0.,None,None),np.full(8,.01))
    assert len(calls)==1 and preview.accepted==1
