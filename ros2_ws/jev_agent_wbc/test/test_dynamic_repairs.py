"""Regression tests for observed-shape drift, fixed-deadline search and WBC stopping."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest

from jev_agent_wbc.dynamic_lidar_map import measured_sphere, DynamicLidarMap
from jev_agent_wbc.intent_paths import IntentPath, propose_intent_paths
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.candidates import SearchBudget
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from test_tpr_wbc import controller, scene, STATE

ROOT=Path(__file__).resolve().parents[1]


def test_round_shape_center_survives_partial_surface_visibility():
    center=np.array([.7,.4,.85]);radius=.1
    az,el=np.meshgrid(np.linspace(-.6,.6,13),np.linspace(-.5,.5,9))
    directions=np.c_[(np.cos(el)*np.cos(az)).ravel(),(np.cos(el)*np.sin(az)).ravel(),np.sin(el).ravel()]
    points=center+radius*directions
    # The packet centroid moves, but the fitted rigid center does not.
    for packet in (points,points[points[:,1]>.4]):
        fit=measured_sphere(packet,.005)
        assert fit is not None
        np.testing.assert_allclose(fit['center'],center,atol=1e-10)
        assert fit['radius']==pytest.approx(radius)
        assert np.linalg.norm(packet.mean(axis=0)-center)>.07


def test_round_shape_fit_rejects_plane_and_non_round_cloud():
    rng=np.random.default_rng(12)
    plane=np.c_[rng.uniform(-.2,.2,(50,2)),np.zeros(50)]
    assert measured_sphere(plane,.005) is None
    assert measured_sphere(rng.uniform(-.2,.2,(70,3)),.005) is None


def test_generic_proxy_still_contains_retained_box_and_surfel_padding():
    mapping=DynamicLidarMap(voxel_size=.035)
    extent=np.array([.1,.16,.09]);center=np.array([1.,.5,.8])
    corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    mapping.stamp=1.
    mapping.tracks[1]=dict(moving=True,stamp=1.,points=center+corners*extent,
        center=center,velocity=np.zeros(3),model_extent=extent)
    observed=mapping.snapshot();mask=observed['surfel_track_ids']>0
    rng=np.random.default_rng(7);box=center+rng.uniform(-1,1,(1000,3))*extent
    directions=rng.normal(size=(1000,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    expanded=box+directions*observed['surfel_radius']
    gap=np.linalg.norm(expanded[:,None]-observed['surfel_centers'][mask],axis=2)-observed['surfel_radii'][mask]
    assert gap.min(axis=1).max()<1e-10


@pytest.mark.parametrize('rate',[-.9,-.45,.45,.9])
def test_fixed_deadline_clock_preserves_prefix_monotonicity_and_terminal_frame(rate):
    runtime=json.loads((ROOT/'scenarios/orbit/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'scenarios/orbit/task.json',runtime['subject']['position'])
    path=IntentPath(original,original,1.5,dict(route='timing',amplitude_m=0.,extension_s=0.,phase_rate=rate))
    times=np.linspace(0,8,2001);clock=np.array([path.parent_time(t) for t in times])
    assert np.diff(clock).min()>0
    assert path.duration==original.duration
    np.testing.assert_allclose(path.sample(1.5)[0],original.sample(1.5)[0],atol=1e-12)
    np.testing.assert_allclose(path.sample(8)[0],original.sample(8)[0],atol=1e-10)
    np.testing.assert_allclose(path.translation(1.5,1),original.sample(1.5)[2][:3],atol=1e-10)
    np.testing.assert_allclose(path.translation(8,1),0.,atol=1e-10)
    assert abs(path.parent_time(4.75)-4.75)>.8


def test_terminal_stop_can_use_direct_barrier_without_relaxing_actuator_limits():
    p,_,_=whole_body_kinematics(STATE)
    c=controller(scene(p+[.20,0.,0.],(.001,0.,0.)))
    c.limits=replace(c.limits,command_ramp=False,deadline_s=.02)
    c.previous[0]=-.0099;c.measured_control=c.previous.copy()
    c.reconfiguration.update=lambda *args:np.zeros(8)
    u,info=c.solve(STATE,*c.trajectory.sample(0),.02,trajectory_time=0.,commanded_q=STATE[3:])
    np.testing.assert_allclose(u,0.,atol=1e-9)
    assert info['command_sweep_lower_m']>=.025
    assert info['qp_violation']<1e-6
    assert info['tpr_report']['barrier_mode']=='certified_velocity_barrier'


def test_kinematic_witness_is_only_a_secondary_preference():
    c=controller();p,r,J=whole_body_kinematics(STATE)
    c.reconfiguration.update=lambda *args:np.zeros(8)
    c.kinematic_reference=dict(feasible=True,trace=[
        dict(t=t,state=STATE.tolist(),control=(np.ones(8)*1e4).tolist()) for t in (0.,1.)])
    u,info=c.solve(STATE,p,r,np.zeros(6),.02,commanded_q=STATE[3:])
    assert info['tpr_report']['kinematic_preference_active']
    np.testing.assert_allclose(J@u,0.,atol=1e-8)
    assert np.all(abs(u)<=c.limits.acceleration*.02+1e-6)


def test_installed_intent_path_does_not_change_wbc_prediction_before_activation(monkeypatch):
    from jev_agent_wbc import tpr_wbc
    runtime=json.loads((ROOT/'scenarios/orbit/runtime_scene.json').read_text())
    original=OriginalReference(ROOT/'scenarios/orbit/task.json',runtime['subject']['position'])
    path=IntentPath(original,original,1.5,dict(route='above',amplitude_m=.1,extension_s=0.))
    c=controller();c.set_trajectory(path)
    monkeypatch.setattr(tpr_wbc,'solve_tpr',lambda *args,**kwargs:(np.zeros(8),{}))
    c.solve(STATE,*path.sample(1.),.02,trajectory_time=1.)
    assert c.reconfiguration.trajectory is original
    c.solve(STATE,*path.sample(1.5),.02,trajectory_time=1.5)
    assert c.reconfiguration.trajectory is path
