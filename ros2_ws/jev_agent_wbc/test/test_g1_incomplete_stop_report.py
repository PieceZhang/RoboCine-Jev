"""An interrupted native run must report missing evidence without hiding its cause."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def test_incomplete_stop_is_unverified_and_never_invokes_complete_window_audit():
    path=Path(__file__).resolve().parents[1]/'Robot_G1/adaptive_runner.py'
    spec=importlib.util.spec_from_file_location('g1_incomplete_stop_report',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[dict(stamp_s=t,phase='move',measured_body_velocity=[0.,0.,0.],
               requested_body_velocity=[0.,0.,0.],position_error_m=.01,
               orientation_error_rad=.02,joint_positions=[0.],
               T_world_base=np.eye(4).tolist(),T_world_camera_optical=np.eye(4).tolist())
          for t in (0.,.1,.2)]
    def forbidden(*args):
        raise AssertionError('a missing stop window cannot be audited as a complete stop')
    result=module.summarize('camera_shot',rows,SimpleNamespace(camera_excursion='diagnostic'),forbidden)
    assert result['strict_stop']==dict(passed=False,status='unverified_incomplete_measured_stop_window')
    assert result['passed'] is False


@pytest.mark.parametrize('already_frozen',[0,19])
def test_prepared_simulation_gc_preserves_automatic_collection_and_restores_ownership(monkeypatch,already_frozen):
    import sys
    path=Path(__file__).resolve().parents[1]/'Robot_G1/adaptive_runner.py'
    spec=importlib.util.spec_from_file_location('g1_prepared_gc',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[];state=[already_frozen];threshold=[700,10,10]
    def freeze():calls.append('freeze');state[0]=100
    def unfreeze():calls.append('unfreeze');state[0]=0
    def collect():calls.append('collect');return 3
    def set_threshold(*values):threshold[:]=values
    fake=SimpleNamespace(isenabled=lambda:True,get_freeze_count=lambda:state[0],
                         freeze=freeze,unfreeze=unfreeze,collect=collect,
                         get_threshold=lambda:tuple(threshold),set_threshold=set_threshold)
    monkeypatch.setitem(sys.modules,'gc',fake)
    with pytest.raises(RuntimeError,match='physical failure'):
        with module.PreparedSimulationGarbage(True) as prepared:
            assert prepared.report['automatic_gc_enabled'] is True
            assert threshold==[700,10,1000000]
            raise RuntimeError('physical failure')
    assert state[0]==already_frozen
    assert threshold==[700,10,10]
    assert calls==([] if already_frozen else ['collect','freeze','unfreeze'])
