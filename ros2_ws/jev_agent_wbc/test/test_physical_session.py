import pickle
from queue import Queue
from threading import Event

import numpy as np

from jev_agent_wbc.realtime import RuntimePolicy,CommandWatchdog
from jev_agent_wbc.physical_session import MeasuredExecution
from jev_agent_wbc.whole_body_qp import Limits


def packet(t,wall,sequence=1):
    return dict(t=t,observation_wall=wall,sequence=sequence,command=np.ones(8)*.01)


def test_command_gap_brakes_and_only_fresh_ordered_commands_recover():
    policy=RuntimePolicy(recover_decisions=True)
    watchdog=CommandWatchdog(policy,10.)
    assert watchdog.accept(packet(0.,10.),0.,10.)
    assert not watchdog.command(.12,10.12,.12).any()
    assert watchdog.waiting and watchdog.fault is None
    assert not watchdog.accept(packet(0.,10.,2),.14,10.14)
    assert not watchdog.accept(packet(.14,10.14,1),.14,10.14)
    assert watchdog.accept(packet(.14,10.14,3),.16,10.16)
    assert watchdog.command(.16,10.16,.16).any()
    assert watchdog.recoveries==1


def test_contact_and_invalid_sensor_clock_never_recover_automatically():
    for reason in ('sensor_stale','measured_external_contact'):
        w=CommandWatchdog(RuntimePolicy(recover_decisions=True),10.)
        if reason=='sensor_stale':w.command(.5,10.5,float('nan'))
        else:w.escalate(reason,.5,10.5)
        assert w.fault['reason']==reason
        assert not w.accept(packet(.5,10.5),.5,10.5)


def test_sensor_gap_requires_fresh_sensor_bound_command_before_recovery():
    w=CommandWatchdog(RuntimePolicy(recover_decisions=True),10.)
    w.command(.3,10.3,0.)
    assert w.sensor_waiting and w.fault is None
    assert not w.accept(dict(packet(.3,10.3),sensor_stamp=0.),.3,10.3)
    assert not w.command(.32,10.32,.3).any()
    assert w.sensor_waiting
    assert w.accept(dict(packet(.32,10.32,2),sensor_stamp=.3),.34,10.34)
    assert w.sensor_waiting is None
    assert w.command(.34,10.34,.3).any()


def test_wait_is_bounded_even_when_a_fresh_packet_arrives_after_deadline():
    w=CommandWatchdog(RuntimePolicy(recover_decisions=True,recovery_timeout_s=1.),10.)
    w.command(.3,10.3,.3)
    assert not w.accept(packet(1.5,11.5),1.5,11.5)
    assert w.fault['reason']=='command_recovery_exhausted'


def test_measured_exchange_preserves_source_stamp_and_uses_real_physical_time():
    inbox,outbox=Queue(),Queue()
    initial=dict(t=.1,observation_wall=20.,sequence=1,state=np.zeros(9),
        base_z=.002,measured_control=np.zeros(8),scene={'observation_stamp':.1})
    actual=dict(initial,t=.16,sequence=4,state=np.ones(9)*.012)
    inbox.put(pickle.dumps(actual))
    runtime=MeasuredExecution(inbox,outbox,Event(),initial,Limits(),RuntimePolicy())
    state,z,previous,scene,t=runtime.exchange(np.zeros(8),{},'main_action',0,.1,np.zeros(3),np.eye(3))
    sent=pickle.loads(outbox.get())
    assert sent['observation_wall']==20. and sent['t']==.1 and sent['sequence']==1
    assert t==.16
    np.testing.assert_array_equal(state,actual['state'])


def test_exact_sensor_deadline_does_not_trip_from_float_subtraction():
    w=CommandWatchdog(RuntimePolicy(),10.)
    assert w.accept(packet(1.5,10.),1.5,10.)
    w.command(1.5,10.,1.2999999999999998)
    assert w.fault is None
    w.command(1.50001,10.,1.2999999999999998)
    assert w.fault['reason']=='sensor_stale'


def test_wall_lease_boundary_only_allows_floating_point_roundoff():
    w=CommandWatchdog(RuntimePolicy(),1e6)
    assert w.accept(packet(0.,1e6),.1,1e6+.1)
    assert w.command(.1,1e6+.1,.1).any()
    assert not w.command(.10001,1e6+.10001,.10001).any()
    assert w.fault['reason']=='command_timeout'


def test_phase_clock_conversion_preserves_observation_age():
    from jev_agent_wbc.perception import SubjectEvidence
    from jev_agent_wbc.physical_session import phase_evidence
    evidence=SubjectEvidence('subject')
    evidence.update_geometry(dict(stamp_s=12.3,position=[1.,0.,0.],valid=True))
    local=phase_evidence(evidence,10.)
    assert abs((2.4-local.geometry['stamp_s'])-(12.4-evidence.geometry['stamp_s']))<1e-9
    assert evidence.geometry['stamp_s']==12.3
    assert local.geometry_history[-1]['stamp_s']==local.geometry['stamp_s']


def test_invalid_acceleration_packet_cannot_replace_valid_command():
    w=CommandWatchdog(RuntimePolicy(recover_decisions=True),10.)
    assert w.accept(packet(0.,10.),0.,10.)
    for acceleration in ([0.]*7,[float('nan')]*8):
        malformed=dict(packet(.02,10.02,2),control_acceleration=acceleration)
        assert not w.accept(malformed,.02,10.02)
    assert w.sequence==1
    assert not w.accept(dict(packet(.02,10.02,2),sensor_stamp=.03),.02,10.02)


def test_skipped_measurements_use_elapsed_control_interval():
    initial=dict(t=.1,observation_wall=20.,sequence=1,state=np.zeros(9),
        base_z=0.,measured_control=np.zeros(8),scene={})
    runtime=MeasuredExecution(Queue(),Queue(),Event(),initial,Limits(),RuntimePolicy())
    controller=object()
    assert abs(runtime.control_dt(controller)-.02)<1e-9
    runtime.snapshot=dict(initial,t=.16)
    assert abs(runtime.control_dt(controller)-.06)<1e-9


def test_skipped_measurement_does_not_expand_next_command_horizon():
    from types import SimpleNamespace as NS
    from jev_agent_wbc.perception import SubjectEvidence
    received=[]
    initial=dict(t=.1,observation_wall=20.,sequence=1,state=np.zeros(9),base_z=0.,
        base_rotation=np.eye(3),measured_control=np.zeros(8),scene={},
        perception=SubjectEvidence('subject'),camera=dict(position=np.zeros(3),rotation=np.eye(3)))
    runtime=MeasuredExecution(Queue(),Queue(),Event(),initial,Limits(),RuntimePolicy())
    engine=NS(wbc=NS(previous=np.zeros(8)),set_camera_output=lambda *a,**k:None,
        step=lambda state,scene,t,dt,**kw:(received.append(dt) or np.zeros(8),{}))
    runtime.step(engine,np.zeros(9),{},.1,.02,0.)
    runtime.snapshot=dict(initial,t=.18)
    _,info=runtime.step(engine,np.zeros(9),{},.18,.02,0.)
    assert received==[.02,.02]
    assert abs(info['measured_observation_interval_s']-.08)<1e-9
    assert info['measured_control_interval_s']==.02


def test_contiguous_double_motion_uses_all_physical_samples_and_global_phase_checks(tmp_path):
    import json
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.physical_session_audit import assess_session
    from CineGym.paths import camera_task
    from CineGym.config import GymConfig
    task,scene,_=camera_task(np.random.default_rng(41),'combo_01_easy',GymConfig())
    subject=scene['subject']
    duration=task['reference_camera_trajectory']['samples'][-1]['t']
    reference=OriginalReference(task,subject['position'])
    rows=[]
    for t in np.arange(0.,duration+.401,.02):
        p,r,_=reference.sample(float(t))
        # Markers deliberately omit alternating observations; parent samples
        # must still include the intervals where the controller was waiting.
        info=dict(session_stage='main_action',phase_index=0,phase_start_s=0.) if len(rows)%2==0 else {}
        rows.append(dict(t=float(t),camera_position=p,camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat(),
            visual=dict(visibility=1.,in_frame=1.),info=info))
    rows[-1]['info'].update(session_done=True,session_result=dict(completed=True,failure=''))
    assessed=assess_session(dict(safety_pass=True),rows,task,subject,None,tmp_path)
    assert len(assessed['physical_phase_acceptance'])==1
    phase=assessed['physical_phase_acceptance'][0]
    assert phase['samples']==len(rows)
    assert len(phase['acceptance']['checks']['direction']['phases'])==2
    assert 'transition_continuity' in phase['acceptance']['checks']


def test_preparation_retries_preserve_completed_work_and_are_bounded():
    from types import SimpleNamespace
    events=[];calls=[]
    engine=SimpleNamespace(_write=lambda *a,**kw:events.append((a,kw)))
    runtime=MeasuredExecution(Queue(),Queue(),Event(),{},Limits(),RuntimePolicy())
    def interrupted(engine):
        calls.append(1)
        if len(calls)<3:raise TimeoutError('selection temporarily unavailable')
        return 'selected'
    assert runtime.prepare(interrupted,engine)=='selected'
    assert engine.resumable_preparation and len(events)==2


def test_preparation_resumes_transport_but_reconsiders_rejected_geometry(monkeypatch):
    import pytest
    from types import SimpleNamespace as NS
    from jev_agent_wbc import initial_planning as module
    calls=[]
    def decide(evidence):
        calls.append(evidence)
        if len(calls)==2:raise TimeoutError('configuration transport interrupted')
        if evidence['stage']=='strategy':return dict(strategy='joint_curve')
        return dict(search_config='base',construction_plan={'family':'joint_curve'})
    original=NS(sha256='task',duration=8.,subject=np.zeros(3),phase=lambda t:'main',
        task=dict(subject={'id':'subject'},shot_phase='main',intent='movein'))
    observation=NS(summary=lambda relations:{},to_dict=lambda:dict(world_from_mount=dict(translation_m=[0.,0.,0.])))
    engine=NS(mode='jev',budget=NS(single_candidate=True),active=original,original=original,
        subject_model={},intrinsics={},planning_limits=Limits(),geometry_preferences={},
        geometry_backend=NS(observe=lambda *a,**k:observation,obstacle_relations=lambda *a:[],box_budget=8),
        jev=NS(decider=NS(client=NS(url='https://api.typesafe.ai/v1/systemone'),decide=decide)),
        resumable_preparation=True,_write=lambda *a,**k:None)
    monkeypatch.setattr(module.task_space_planning,'diagnose',lambda *a,**k:dict(report={}))
    monkeypatch.setattr(module,'construction_evidence',lambda *a,**k:{})
    monkeypatch.setattr(module,'offers',lambda *a:{})
    monkeypatch.setattr(module.task_space_planning,'build',lambda *a,**k:dict(
        error='no_verified_executable_candidate',candidates=[],wall_s=.1,evaluations=[{'rejected':'camera_collision'}]))
    with pytest.raises(TimeoutError):module.prepare(engine,np.zeros(9),{},0.)
    assert 'strategy' in engine.preparation_checkpoint
    with pytest.raises(RuntimeError,match='initial_single_candidate_rejected'):
        module.prepare(engine,np.zeros(9),{},0.)
    assert [e['stage'] for e in calls]==['strategy','configuration','configuration']
    assert not engine.preparation_checkpoint
    with pytest.raises(RuntimeError,match='initial_single_candidate_rejected'):
        module.prepare(engine,np.zeros(9),{},0.)
    assert calls[-2]['stage']=='strategy'
    assert calls[-2]['previous_preparation_failures'][0]['evaluations'][0]['rejected']=='camera_collision'


def test_decision_deadline_waits_for_pending_work_without_latching(monkeypatch):
    from types import SimpleNamespace
    from jev_agent_wbc.engine import CinematicEngine
    import jev_agent_wbc.engine as module
    events=[]
    engine=object.__new__(CinematicEngine)
    engine.runtime_policy=RuntimePolicy(recover_decisions=True)
    engine.runtime_fault=None;engine.pipeline_started_wall=10.
    engine.math_kind='configuration';engine.math_started_wall=None;engine.math_future=None
    engine.jev=SimpleNamespace(future=object(),submitted_wall=10.)
    engine.runtime_delays=set();engine.pending_candidates={}
    engine._write=lambda *a,**k:events.append((a,k))
    engine._feedback=lambda *a:None
    monkeypatch.setattr(module.time,'monotonic',lambda:12.)
    engine._runtime_deadline(2.)
    assert engine.runtime_fault is None and engine.jev.future is not None
    count=len(events);engine._runtime_deadline(2.02)
    assert len(events)==count
    monkeypatch.setattr(module.time,'monotonic',lambda:26.)
    engine._runtime_deadline(16.)
    assert engine.runtime_fault['reason']=='planning_pipeline_timeout'


def test_replacement_failure_keeps_certified_original_or_installed_objective_but_is_bounded(monkeypatch):
    from types import SimpleNamespace as NS
    from jev_agent_wbc.engine import CinematicEngine
    import jev_agent_wbc.engine as module
    monkeypatch.setattr(module.time,'monotonic',lambda:10.)
    for original in (False,True):
        engine=object.__new__(CinematicEngine)
        engine.runtime_policy=RuntimePolicy(recover_decisions=True)
        engine.runtime_fault=None;engine.runtime_retries=3
        engine.original=object();engine.active=engine.original if original else object()
        engine._write=lambda *a,**k:None
        engine.last_execution_certificate=dict(time_s=4.98,clearance_m=.1,self_clearance_m=.02)
        engine.geometry_scene=dict(observation_stamp=4.9)
        engine.wbc=NS(collision=NS(margin=.025),self_collision=NS(margin=.004))
        engine._runtime_failure('candidate rejected',5.)
        assert engine.runtime_fault is None and engine.runtime_retry_after==7.
        engine.pending_candidates={'late':object()}
        engine._runtime_deadline(20.01)
        assert engine.runtime_fault['reason']=='decision_recovery_exhausted'
        assert not engine.pending_candidates


def test_rejected_plan_cannot_continue_on_stale_or_failed_execution_certificate():
    from types import SimpleNamespace as NS
    from jev_agent_wbc.engine import CinematicEngine
    for stamp,gap in [(4.,.1),(4.98,.01)]:
        engine=object.__new__(CinematicEngine)
        engine.runtime_policy=RuntimePolicy(recover_decisions=True)
        engine.runtime_fault=None;engine.runtime_retries=3
        engine.last_execution_certificate=dict(time_s=stamp,clearance_m=gap,self_clearance_m=.02)
        engine.geometry_scene=dict(observation_stamp=4.9)
        engine.wbc=NS(collision=NS(margin=.025),self_collision=NS(margin=.004))
        engine.pending_candidates={};engine._write=lambda *a,**k:None
        engine._runtime_failure('rejected',5.)
        assert engine.runtime_fault['reason']=='rejected'


def test_authored_combo_actions_load_without_inventing_gap_trajectories():
    import json
    from pathlib import Path
    from jev_agent_wbc.continuous_combo import action_task
    from jev_agent_wbc.reference import OriginalReference
    root=Path(__file__).resolve().parents[1]/'scenarios'
    cases=list(root.glob('combo_*/task.json'))
    assert len(cases)==6
    for file in cases:
        task=json.loads(file.read_text())
        scene=json.loads(file.with_name('runtime_scene.json').read_text())
        for phase in task['shot_phase']:
            local=action_task(task,phase)
            reference=OriginalReference(local,scene['subject']['position'])
            assert abs(reference.duration-(phase['end_s']-phase['start_s']))<1e-8
            assert local['reference_camera_trajectory']['samples'][0]['t']==0.
            assert local['shot_phase'][0]['motion']==phase['motion']


def test_late_velocity_uses_physical_braking_without_extending_shot(monkeypatch):
    from types import SimpleNamespace
    from jev_agent_wbc.perception import SubjectEvidence
    from jev_agent_wbc import safety
    limits=Limits(deadline_s=1.)
    snapshot=dict(t=.9,state=np.zeros(9),measured_control=np.zeros(8),
        base_rotation=np.eye(3),camera=dict(position=np.zeros(3),rotation=np.eye(3)),
        perception=SubjectEvidence('subject'))
    engine=SimpleNamespace(wbc=SimpleNamespace(limits=limits),set_camera_output=lambda *a,**k:None,
        _write=lambda *a,**k:None,active=SimpleNamespace(sample=lambda t:(np.zeros(3),np.eye(3),np.zeros(6))))
    def missed(*args,**kwargs):
        error=RuntimeError('deadline conflicts with measured velocity')
        error.diagnostics=dict(actuator_bounds_feasible=False)
        raise error
    engine.step=missed
    def brake(controller,*args,**kwargs):
        assert kwargs['brake'] and controller.limits.deadline_s is None
        np.testing.assert_array_equal(controller.limits.acceleration,limits.acceleration)
        np.testing.assert_array_equal(controller.limits.velocity,limits.velocity)
        return np.zeros(8),{}
    monkeypatch.setattr(safety,'solve_with_certificate',brake)
    runtime=MeasuredExecution(Queue(),Queue(),Event(),snapshot,Limits(),RuntimePolicy())
    command,info=runtime.step(engine,np.zeros(9),{},.9,.02,0.)
    assert info['physical_deadline_braking'] and engine.wbc.limits.deadline_s==1.



def test_stale_verification_refresh_is_bounded_and_never_rebases_approval(monkeypatch):
    from types import SimpleNamespace
    from jev_agent_wbc.engine import CinematicEngine
    import jev_agent_wbc.engine as module
    engine=CinematicEngine.__new__(CinematicEngine)
    engine.runtime_policy=RuntimePolicy(recover_decisions=True)
    engine.math_future=None;engine.active='parent';engine.original='original'
    engine.wbc=SimpleNamespace(previous=np.zeros(8));engine.planning_limits=Limits()
    engine.perception=SimpleNamespace(snapshot=lambda t:{'stamp':t})
    engine.intrinsics=None;engine.budget=object()
    submissions=[];engine._submit_math=lambda *a,**k:submissions.append((a,k))
    engine._write=lambda *a,**k:None
    monkeypatch.setattr(module,'path_hash',lambda path:path)
    candidate=SimpleNamespace(id='approved',path=SimpleNamespace(activation_s=3.))
    approval=SimpleNamespace(parent_id='parent')
    result=dict(candidate=candidate,approval=approval)
    args=(result,'latest_verification_expired',{'observation_stamp':1.},np.zeros(9),1.,0.,None)
    assert engine._refresh_verification(*args)
    assert submissions[0][0][2] is candidate and submissions[0][0][3] is approval
    assert candidate.path.activation_s==3.
    assert not engine._refresh_verification(*args)
    engine.verification_refreshes={}
    candidate.path.activation_s=1.1
    assert not engine._refresh_verification(*args)
    candidate.path.activation_s=3.
    assert not engine._refresh_verification(result,'approval_binding_mismatch',*args[2:])
    assert not engine._refresh_verification(result,args[1],{'observation_stamp':0.},*args[3:])
    engine.active='different parent'
    assert not engine._refresh_verification(*args)


def test_certificate_wait_requires_fresh_sensor_command_and_explicit_actuator_proof():
    w=CommandWatchdog(RuntimePolicy(recover_decisions=True),10.)
    assert w.accept(dict(packet(0.,10.),sensor_stamp=0.),0.,10.)
    w.wait_for_certificate('uncertain_occupancy',.02,10.02,0.)
    assert not w.command(.02,10.02,0.).any()
    # A fresh packet alone cannot release the drive.
    assert w.accept(dict(packet(.04,10.04,2),sensor_stamp=0.),.04,10.04)
    assert w.certificate_candidate(.04,10.04,.04) is None
    assert w.accept(dict(packet(.06,10.06,3),sensor_stamp=.04),.06,10.06)
    assert not w.command(.06,10.06,.04).any()
    assert w.certificate_candidate(.06,10.06,.04).any()
    assert not w.certificate_passed(.06,10.06,float('nan'))
    assert w.certificate_passed(.06,10.06,.04)
    assert w.command(.06,10.06,.04).any() and w.recoveries==1
    assert w.packet['t']==.06


def test_certificate_wait_cannot_renew_deadline_or_clear_a_contact_fault():
    w=CommandWatchdog(RuntimePolicy(recover_decisions=True,recovery_timeout_s=1.),10.)
    assert w.accept(dict(packet(0.,10.),sensor_stamp=0.),0.,10.)
    w.wait_for_certificate('uncertain_occupancy',.02,10.02,0.)
    for t in [.2,.4,.6,.8]:
        w.wait_for_certificate('uncertain_occupancy',t,10+t,t)
        assert w.certificate_waiting['time_s']==.02
    assert not w.accept(dict(packet(1.04,11.04,2),sensor_stamp=1.04),1.04,11.04)
    assert w.fault['reason']=='certificate_recovery_exhausted'
    assert not w.certificate_passed(1.04,11.04,1.04)
    w=CommandWatchdog(RuntimePolicy(recover_decisions=True),10.)
    w.wait_for_certificate('uncertain_occupancy',.02,10.02,0.)
    w.escalate('measured_external_contact',.03,10.03)
    assert not w.accept(dict(packet(.04,10.04,2),sensor_stamp=.04),.04,10.04)
    assert not w.certificate_passed(.04,10.04,.04)


def test_physical_certificate_wait_emits_fresh_brake_without_resolving_uncertain_qp():
    from types import SimpleNamespace
    state=np.array([0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])
    s=dict(t=.2,measured_control=np.zeros(8),control_status=dict(waiting_for_certificate={'time_s':.1}))
    runtime=MeasuredExecution(Queue(),Queue(),Event(),s,Limits(),RuntimePolicy())
    engine=SimpleNamespace(wbc=SimpleNamespace(),active=SimpleNamespace(sample=lambda t:(np.ones(3),None,None)))
    command,info=runtime.step(engine,state,{},.2,.02,0.)
    assert not command.any() and info['waiting_for_physical_certificate']
    assert np.isfinite(info['position_error']) and info['position_error']>0.
    assert info['controller_timing_missing']


def test_forgotten_obstacle_cannot_clear_certificate_wait():
    w=CommandWatchdog(RuntimePolicy(recover_decisions=True),10.)
    w.accept(dict(packet(0.,10.),sensor_stamp=0.),0.,10.)
    w.wait_for_certificate('uncertain_occupancy',.02,10.02,0.,[7])
    w.accept(dict(packet(.04,10.04,2),sensor_stamp=.04),.04,10.04)
    assert not w.certificate_passed(.04,10.04,.04,[])
    assert not w.certificate_passed(.04,10.04,.04,[8])
    assert w.certificate_passed(.04,10.04,.04,[7])
