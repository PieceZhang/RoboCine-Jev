from types import SimpleNamespace
import time
from concurrent.futures import Future
import pytest
from jev_agent_wbc.engine import CinematicEngine
from jev_agent_wbc.realtime import RuntimePolicy


def engine():
    e=CinematicEngine.__new__(CinematicEngine)
    e.runtime_policy=RuntimePolicy(recover_decisions=True)
    e.runtime_fault=None;e.runtime_recovery_started=dict(time_s=1.,wall_s=time.monotonic())
    e.original=SimpleNamespace(task=dict(shot_phase=[dict(start_s=0.,end_s=8.),dict(start_s=8.,end_s=18.)]))
    e.last_execution_certificate=dict(time_s=8.4,clearance_m=.05,self_clearance_m=.05)
    e.geometry_scene=dict(observation_stamp=8.4)
    e.wbc=SimpleNamespace(collision=SimpleNamespace(margin=.025),self_collision=SimpleNamespace(margin=.025))
    e.camera_history=[];e.subject_model={};e.intrinsics=None;e.runtime_retries=3
    def submit(function,*args):
        f=Future();f.set_result(function(*args));return f
    e.pool=SimpleNamespace(submit=submit)
    e.pending_candidates={};e.pipeline_started_wall=None;e.math_kind='';e.math_future=None;e.jev=SimpleNamespace(future=None)
    e._write=lambda *a,**k:None
    return e


def test_only_verified_intermediate_phase_resets_accumulated_recovery(monkeypatch):
    import jev_agent_wbc.engine as module
    monkeypatch.setattr(module,'evaluate_phase_checkpoint',lambda *a:dict(accepted=True,failed_checks=[]))
    e=engine();assert not e._phase_recovery_checkpoint(8.39)
    assert not e._phase_recovery_checkpoint(8.4)
    assert e._phase_recovery_checkpoint(8.4)
    assert e.runtime_recovery_started is None and e.runtime_retries==0
    e.runtime_recovery_started=dict(time_s=8.4,wall_s=time.monotonic())
    assert not e._phase_recovery_checkpoint(8.4)
    assert e.runtime_recovery_started is not None


@pytest.mark.parametrize('condition',['failed_phase','stale_map','unsafe_certificate','old_checkpoint','expired_episode','physical_fault'])
def test_checkpoint_does_not_erase_unverified_failed_or_expired_recovery(monkeypatch,condition):
    import jev_agent_wbc.engine as module
    monkeypatch.setattr(module,'evaluate_phase_checkpoint',lambda *a:dict(accepted=condition!='failed_phase',failed_checks=['terminal_frame'] if condition=='failed_phase' else []))
    e=engine()
    if condition=='stale_map':e.geometry_scene['observation_stamp']=7.
    if condition=='unsafe_certificate':e.last_execution_certificate['clearance_m']=.02
    if condition=='old_checkpoint':e.runtime_recovery_started['time_s']=8.5
    if condition=='expired_episode':e.runtime_recovery_started['wall_s']=time.monotonic()-16.
    if condition=='physical_fault':e.runtime_fault={'reason':'hard_fault'}
    e._runtime_deadline(8.4)
    assert e.runtime_recovery_started is not None
    if condition=='expired_episode':assert e.runtime_fault['reason']=='decision_recovery_exhausted'
    if condition=='physical_fault':assert e.runtime_fault['reason']=='hard_fault'


def test_checkpoint_background_work_never_waits_in_measured_servo(monkeypatch):
    e=engine();future=Future();e.pool=SimpleNamespace(submit=lambda *a:future)
    assert not e._phase_recovery_checkpoint(8.4)
    assert not e._phase_recovery_checkpoint(8.4)
    assert e.runtime_recovery_started is not None
    future.set_result(dict(accepted=True,failed_checks=[]))
    e.geometry_scene['observation_stamp']=7.
    assert not e._phase_recovery_checkpoint(8.4)
    e.geometry_scene['observation_stamp']=8.4
    assert e._phase_recovery_checkpoint(8.4)


def test_checkpoint_worker_failure_retains_original_recovery_deadline():
    e=engine();future=Future();future.set_exception(OSError('worker checkpoint unavailable'))
    e.pool=SimpleNamespace(submit=lambda *a:future)
    original=e.runtime_recovery_started.copy()
    assert not e._phase_recovery_checkpoint(8.4)
    assert not e._phase_recovery_checkpoint(8.4)
    assert e.runtime_recovery_started==original and e.runtime_retries==3
