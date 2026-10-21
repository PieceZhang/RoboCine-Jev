from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from jev_agent_wbc.realtime import CommandWatchdog, RuntimePolicy, realtime_assessment
from jev_agent_wbc.realtime_engine import offer
from jev_agent_wbc.engine import CinematicEngine


def packet(**overrides):
    return dict(dict(t=0., observation_wall=10., sequence=1, command=np.ones(8)*.01), **overrides)


def test_watchdog_slow_physics_still_expires_in_wall_time_and_latches():
    guard = CommandWatchdog(RuntimePolicy(), 10.)
    assert guard.accept(packet(), 0., 10.)
    assert np.any(guard.command(.02, 10.02, 0.))
    assert not guard.command(.03, 10.11, 0.).any()
    assert guard.fault['reason'] == 'command_timeout'
    assert not guard.accept(packet(t=.04, observation_wall=10.12, sequence=2), .04, 10.12)
    assert not guard.command(.04, 10.12, .04).any()


@pytest.mark.parametrize('changes', [dict(t=1.), dict(observation_wall=11.),
    dict(command=[float('nan')]*8), dict(command=[1.]*7), dict(sequence=-1)])
def test_invalid_or_out_of_order_packets_never_authorize_motion(changes):
    guard = CommandWatchdog(RuntimePolicy(), 10.)
    assert not guard.accept(packet(**changes), 0., 10.)
    assert not guard.command(0., 10., 0.).any()


def test_stale_sensor_and_worker_death_latch_independently():
    for alive, stamp, reason in [(False, .3, 'controller_process_exited'),
                                 (True, 0., 'sensor_stale')]:
        guard = CommandWatchdog(RuntimePolicy(), 10.)
        assert guard.accept(packet(t=.3), .3, 10.)
        assert not guard.command(.3, 10., stamp, alive=alive).any()
        assert guard.fault['reason'] == reason


def test_no_first_response_holds_then_faults_while_clock_advances():
    guard = CommandWatchdog(RuntimePolicy(), 10.)
    assert not guard.command(.1, 10.1, .1).any()
    assert guard.fault is None
    assert not guard.command(.21, 10.21, .21).any()
    assert guard.fault['reason'] == 'first_command_timeout'


def test_stop_cannot_claim_safety_against_incoming_obstacle():
    guard = CommandWatchdog(RuntimePolicy(), 10.)
    guard.trip('command_timeout', .1, 10.1)
    guard.escalate('measured_external_contact', 1., 11.)
    assert guard.fault['reason'] == 'command_timeout'
    assert guard.emergency_stop_required['reason'] == 'measured_external_contact'
    assert not guard.command(1.,11.,1.).any()
    assert not guard.accept(packet(t=1.,observation_wall=11.,sequence=2),1.,11.)


def test_latest_snapshot_queue_has_bounded_backpressure():
    queue = Queue(1)
    for i in range(1000):
        assert offer(queue, i)
    assert queue.qsize() == 1
    assert queue.get_nowait() == 999


def test_completed_but_late_planner_cannot_install(monkeypatch):
    engine = CinematicEngine.__new__(CinematicEngine)
    engine.runtime_policy = RuntimePolicy(planning_timeout_s=1.)
    engine.runtime_fault = None
    engine.pipeline_started_wall = None
    engine.math_kind = 'verify'
    engine.math_started_wall = 10.
    engine.math_future = Future()
    engine.math_future.set_result({'candidate': 'late'})
    engine.jev = SimpleNamespace(future=None)
    engine.pending_candidates = {'late': object()}
    events = []
    engine._write = lambda *a, **k: events.append((a,k))
    engine._feedback = lambda *a, **k: None
    monkeypatch.setattr('jev_agent_wbc.engine.time.monotonic', lambda: 11.01)
    engine._runtime_deadline(.2)
    assert engine.runtime_fault['reason'] == 'planning_verify_timeout'
    assert not engine.pending_candidates
    assert events


def test_video_speed_or_shot_success_cannot_hide_realtime_failure():
    timing = dict(pace=1., max_tick_wall_s=.03, max_lag_s=1.)
    assert not realtime_assessment(timing, None, True)['success']
    timing.update(max_tick_wall_s=.01, max_lag_s=0.)
    assert realtime_assessment(timing, None, True)['success']
    assert not realtime_assessment(timing, {'reason':'timeout'}, True)['success']


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_deadlines_rejected(value):
    with pytest.raises(ValueError):
        RuntimePolicy(command_timeout_s=value)


def hung_controller(inbox, outbox, stop, original, scene, output, options, physical_limits):
    import os
    import pickle
    import time
    os.setsid()
    outbox.put(pickle.dumps(dict(ready=True, summary=dict(jev_calls=0, installed_plans=0))))
    while True:
        time.sleep(.1)


def test_receiver_delay_is_preserved_when_worker_skips_the_first_snapshot():
    import pickle
    import time
    from queue import Queue
    from types import SimpleNamespace as NS
    from jev_agent_wbc.realtime_engine import RealtimeEngine
    from jev_agent_wbc.whole_body_qp import Limits
    engine=object.__new__(RealtimeEngine)
    engine.watchdog=CommandWatchdog(RuntimePolicy(recover_decisions=True),time.monotonic())
    engine.inbox=Queue();engine.outbox=Queue();engine.process=NS(is_alive=lambda:True)
    engine.wbc=NS(limits=Limits());engine.jev=NS(decider=NS(client=NS(calls=0)))
    engine.original=engine.active=NS(sample=lambda t:(np.zeros(3),np.eye(3),np.zeros(6)))
    engine.camera=dict(source='execution',position=np.zeros(3),rotation=np.eye(3))
    engine.perception={};engine.sequence=0;engine.stream_on_fault=False
    engine.last_result=engine.session_result=None;engine.fault_logged=False
    engine.outbox.put(pickle.dumps(dict(t=0.,observation_wall=time.monotonic(),sequence=1,
        command=np.zeros(8),info=dict(loop_wall_s=.001),feedback={},active=None,
        summary=dict(installed_plans=0,jev_calls=0))))
    state=np.array([0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])
    engine.step(state,dict(observation_stamp=.02),.02,.02,measured_control=np.zeros(8))
    engine.inbox.get()  # This observation is overwritten before the worker sees it.
    _,info=engine.step(state,dict(observation_stamp=.06),.06,.02,measured_control=np.zeros(8))
    transport=pickle.loads(engine.inbox.get())['execution_transport']
    assert transport['command_age_s']==pytest.approx(.06)
    assert transport['command_delay_s']==pytest.approx(.02)
    assert info['command_received_time_s']==pytest.approx(.02)
    assert info['command_delivery_delay_s']==pytest.approx(.02)


def test_hung_real_process_does_not_block_physics_or_shutdown(tmp_path, monkeypatch):
    import json
    import time
    from pathlib import Path
    from jev_agent_wbc import realtime_engine
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.whole_body_qp import Limits
    root = Path(__file__).resolve().parents[1]/'scenarios/orbit'
    fixture = json.loads((root/'runtime_scene.json').read_text())
    original = OriginalReference(root/'task.json', fixture['subject']['position'])
    monkeypatch.setattr(realtime_engine, 'controller_process', hung_controller)
    engine = realtime_engine.RealtimeEngine(original, {'observation_stamp':0.}, tmp_path,
        physical_limits=Limits(), limits=Limits(), runtime_policy=RuntimePolicy(startup_timeout_s=.05))
    try:
        engine.arm(time.monotonic())
        with pytest.raises(RuntimeError, match='cannot be rearmed'):
            engine.arm(time.monotonic())
        state = np.asarray(fixture['initial_state'])
        for t in np.arange(0., .2, .02):
            p,r,_ = original.sample(t)
            engine.set_camera_output(t,p,r,source='stabilized')
            start = time.monotonic()
            command,info = engine.step(state, {'observation_stamp':t}, t, .02, observation_wall=start-.01,
                measured_control=np.zeros(8))
            assert time.monotonic()-start < .05
            assert not command.any()
        assert engine.watchdog.fault['reason'] == 'first_command_timeout'
    finally:
        start = time.monotonic()
        engine.close()
        assert time.monotonic()-start < 3.
    assert not engine.process.is_alive()
    events=[json.loads(line) for line in (tmp_path/'runtime_events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='actuator_fault_latched' for e in events)==1
