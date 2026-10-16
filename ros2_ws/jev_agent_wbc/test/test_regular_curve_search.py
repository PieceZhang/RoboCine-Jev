"""Whole-curve timing, exact approval binding and bounded proposal regressions."""
import json
from pathlib import Path
import numpy as np
import pytest
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.intent_paths import IntentPath
from jev_agent_wbc.candidates import SearchBudget,path_hash,path_spec,restore_path
from jev_agent_wbc.regular_curve_search import bind_geometric_progress,bounded_paths
from jev_agent_wbc.camera_intent import default_intrinsics
from jev_agent_wbc.cinematic_geometry import curve_quality

ROOT=Path(__file__).resolve().parents[1]/'scenarios'

@pytest.mark.parametrize('name',['movein','pedestal_up','pedestal_down','truck_left','truck_right'])
def test_retimed_curve_progress_uses_geometry_and_survives_exact_replay(name):
    runtime=json.loads((ROOT/name/'runtime_scene.json').read_text())
    original=OriginalReference(ROOT/name/'task.json',runtime['subject']['position'])
    ts=np.linspace(.5,8.,81);u=(ts-.5)/7.5;clock=.5+7.5*u**2
    points=np.array([original.sample(t)[0] for t in clock])
    h=.001;v=original.sample(.5)[2][:3];a=(original.sample(.5+h)[2][:3]-original.sample(.5-h)[2][:3])/(2*h)
    path=IntentPath(original,original,.5,dict(route='timing',amplitude_m=0.,extension_s=0.,regular_curve=True,
        curve_family='progress_lattice_curve'),runtime['subject'],geometry=dict(times=ts.tolist(),positions=points.tolist(),start_velocity=v.tolist(),start_acceleration=a.tolist()))
    path=bind_geometric_progress(path)
    assert path.progress_time(4.)<3.
    for k,t in enumerate(ts[1:-1],1):
        np.testing.assert_allclose(original.sample(path.progress_time(t))[0],points[k],atol=3e-5)
    restored=restore_path(original,path_spec(path))
    assert path_hash(restored)==path_hash(path)
    np.testing.assert_allclose([restored.progress_time(t) for t in ts],[path.progress_time(t) for t in ts])
    assert path.progress_time(.2)==pytest.approx(.2)


def test_small_budget_is_obeyed_even_when_multiple_clock_families_are_available():
    runtime=json.loads((ROOT/'movein/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'movein/task.json',runtime['subject']['position'])
    scene=dict(valid=True,observation_stamp=0.,centers=[[8.,8.,8.]],radii=[.1],velocities=[[0.,0.,0.]],radius_rates=[0.])
    paths=bounded_paths(original,original,scene,.7,runtime['subject'],default_intrinsics(),
        SearchBudget(max_proposals=2,max_rollouts=2,activation_lead_s=.7,candidate_wall_budget_s=2.))
    assert 1<=len(paths)<=2
    for path in paths:
        assert np.isfinite(path.sample(8.)[0]).all()
        assert path.duration==8.


@pytest.mark.parametrize('corner_fraction',[.25,.5,.75])
def test_initial_join_allowance_cannot_hide_a_middle_corner(corner_fraction):
    class Corner:
        duration=1.
        activation_s=0.
        subject=np.array([2.,0.,0.])
        def phase(self,t):return {'motion':'truck_left'}
        def sample(self,t):
            return np.array([max(0.,t-corner_fraction),min(t,corner_fraction),0.]),np.eye(3),np.zeros(6)
    quality=curve_quality(Corner())
    assert not quality['regular_curve_pass']
    assert not quality['strict_full_curve_pass']
    assert quality['initial_join_turn_rad']==pytest.approx(0.)
    assert quality['core_max_step_turn_rad']>.45
