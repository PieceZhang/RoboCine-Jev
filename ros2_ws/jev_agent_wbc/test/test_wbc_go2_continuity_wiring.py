"""Actual runner factory and CLI preserve explicit output-vs-bias semantics."""
import importlib.util,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('continuity_factory_runner',ROOT/'Robot_Go2/adaptive_runner.py')
w=load('continuity_factory_wrapper',ROOT/'Robot_Go2/world_z_bumpless.py')

@pytest.mark.parametrize('profile',['none','fresh_steady_native_z_v1','fresh_actual_output_continuity_v2'])
def test_actual_factory_keeps_pinned_gain_caps_and_selects_only_declared_profile(profile):
 c=r.make_native_z_feedback(w._base,w,seed_profile=profile)
 assert c.enabled and c.ki_per_s==.2 and c.POSITION_CAP_M==.025 and c.INTEGRAL_CAP_M==.025
 if profile=='none':assert type(c)is w._base.WorldZFeedback
 else:assert type(c)is w.BumplessWorldZFeedback and c.initialization_profile==profile

@pytest.mark.parametrize('feedback,wrapper,profile',[(None,None,'fresh_actual_output_continuity_v2'),(w._base,None,'fresh_actual_output_continuity_v2'),(w._base,w,'unknown')])
def test_missing_or_unknown_initialization_cannot_silently_enable_profile(feedback,wrapper,profile):
 with pytest.raises(ValueError):r.make_native_z_feedback(feedback,wrapper,seed_profile=profile)
 assert r.make_native_z_feedback(None,None,seed_profile='none')is None

@pytest.mark.parametrize('robot,feedback',[('go2','none'),('g1','bounded_integral_v1')])
def test_real_cli_rejects_wrong_robot_or_missing_feedback_before_worker(monkeypatch,tmp_path,robot,feedback):
 monkeypatch.setattr(sys,'argv',['adaptive_runner.py','--robot',robot,'--output',str(tmp_path/'out'),
  '--go2-world-z-feedback',feedback,'--go2-world-z-seed','fresh_actual_output_continuity_v2'])
 def forbidden(*args,**kw):raise AssertionError('invalid profile reached physical worker')
 monkeypatch.setattr(r.subprocess,'run',forbidden)
 with pytest.raises(ValueError):r.main()
 assert not(tmp_path/'out').exists()


def test_real_cli_accepts_explicit_new_choice_without_bypassing_worker_or_admission(monkeypatch,tmp_path):
 monkeypatch.setattr(sys,'argv',['adaptive_runner.py','--robot','go2','--output',str(tmp_path/'out'),
  '--go2-world-z-feedback','bounded_integral_v1','--go2-world-z-seed','fresh_actual_output_continuity_v2'])
 class WorkerDispatch(Exception):pass
 def dispatched(argv):
  assert '--worker'in argv and argv[argv.index('--go2-world-z-seed')+1]=='fresh_actual_output_continuity_v2'
  raise WorkerDispatch
 monkeypatch.setattr(r.subprocess,'run',dispatched)
 with pytest.raises(WorkerDispatch):r.main()
