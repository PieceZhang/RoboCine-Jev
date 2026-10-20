"""Sparse current returns never manufacture a unique pose from ambiguity."""
import numpy as np
import pytest
from jev_agent_wbc.dynamic_lidar_map import DynamicLidarMap,exact_known_sphere


def cap():
    angles=np.array([0.,1.,2.])
    return np.c_[.08*np.cos(angles),.08*np.sin(angles),np.full(3,.06)]


def test_two_reachable_sphere_centers_are_rejected():
    assert exact_known_sphere(cap(),.1,np.zeros(3),.15) is None
    result=exact_known_sphere(cap(),.1,np.zeros(3),.03)
    assert result is not None
    np.testing.assert_allclose(result['center'],0.,atol=1e-12)


def test_noncollinear_four_returns_recover_previously_measured_radius():
    center=np.array([1.,-2.,.7]);points=center+.1*np.array([[1,0,0],[0,1,0],[0,0,1],[0,-1,0]])
    result=exact_known_sphere(points,.1,center+.01,.05)
    np.testing.assert_allclose(result['center'],center,atol=1e-12)
    points[0,0]+=.003
    assert exact_known_sphere(points,.1,center,.05) is None


def test_collinear_returns_cannot_refresh_a_track():
    points=np.array([[0,0,0],[.01,0,0],[.02,0,0]])
    assert exact_known_sphere(points,.1,np.zeros(3),.15) is None


def test_stationary_measured_shape_keeps_sparse_current_evidence():
    mapping=DynamicLidarMap(voxel_size=.035,cluster_distance=.08,velocity_window_s=.8)
    az,el=np.meshgrid(np.linspace(-.7,.7,15),np.linspace(-.6,.6,11))
    dense=.1*np.c_[(np.cos(el)*np.cos(az)).ravel(),(np.cos(el)*np.sin(az)).ravel(),np.sin(el).ravel()]
    center=np.array([1.,.3,.6])
    for t in np.arange(1.,1.61,.1):
        mapping.current_points=center+dense
        mapping.track(mapping.current_points,float(t));mapping.stamp=float(t)
    track=next(iter(mapping.tracks.values()))
    assert not track['moving']
    mapping.instantaneous_exact_geometry=True
    for t in np.arange(1.7,2.51,.1):
        mapping.current_points=center+cap()
        mapping.track(mapping.current_points,float(t));mapping.stamp=float(t)
        assert len(mapping.tracks)==1
        assert track['stamp']==pytest.approx(t)
        np.testing.assert_allclose(track['center'],center,atol=1e-10)
        assert not track['moving']
    old_stamp=track['stamp']
    mapping.current_points=center+np.array([[.1,0.,0.]])
    mapping.track(mapping.current_points,old_stamp+.1)
    assert track['stamp']==old_stamp


@pytest.mark.parametrize('exact',[True,False])
def test_sparse_tracking_is_limited_to_explicit_instantaneous_exact_scans(exact):
    mapping=DynamicLidarMap(voxel_size=.035,cluster_distance=.08,velocity_window_s=.8)
    az,el=np.meshgrid(np.linspace(-.7,.7,15),np.linspace(-.6,.6,11))
    dense=.1*np.c_[(np.cos(el)*np.cos(az)).ravel(),(np.cos(el)*np.sin(az)).ravel(),np.sin(el).ravel()]
    for t in np.arange(1.,1.61,.1):
        center=np.array([1.+.2*(t-1.),.3,.6]);mapping.current_points=center+dense
        mapping.track(mapping.current_points,float(t));mapping.stamp=float(t)
    assert len(mapping.tracks)==1
    track=next(iter(mapping.tracks.values()));assert track['moving']
    old_stamp=track['stamp'];t=old_stamp+.1;center=np.array([1.+.2*(t-1.),.3,.6])
    mapping.instantaneous_exact_geometry=exact;mapping.current_points=center+cap()
    mapping.track(mapping.current_points,t)
    track=next(iter(mapping.tracks.values()))
    assert track['stamp']==pytest.approx(t if exact else old_stamp)
    if exact:np.testing.assert_allclose(track['center'],center,atol=1e-10)


def test_finite_precision_sparse_fit_carries_center_uncertainty():
    from jev_agent_wbc.dynamic_lidar_map import bounded_known_sphere
    rng=np.random.default_rng(234);center=np.array([1.,-.2,.6]);radius=.12
    axes=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[-1.,0,0]])
    for _ in range(100):
        errors=rng.normal(size=(4,3));errors*=rng.uniform(0,.002,(4,1))/np.linalg.norm(errors,axis=1)[:,None]
        result=bounded_known_sphere(center+radius*axes+errors,radius,center,.03)
        assert result is not None
        assert np.linalg.norm(result['center']-center)<=result['center_uncertainty_m']
        assert result['center_uncertainty_m']>0.


def test_finite_precision_sparse_fit_rejects_ambiguous_and_degenerate_geometry():
    from jev_agent_wbc.dynamic_lidar_map import bounded_known_sphere
    assert bounded_known_sphere(cap(),.1,np.zeros(3),.15) is None
    assert bounded_known_sphere(np.array([[0,0,0],[.01,0,0],[.02,0,0]]),.1,np.zeros(3),.1) is None


def test_single_return_bounds_shape_without_fabricating_center_or_velocity():
    mapping=DynamicLidarMap(voxel_size=.035,cluster_distance=.08,velocity_window_s=.8)
    az,el=np.meshgrid(np.linspace(-.7,.7,15),np.linspace(-.6,.6,11))
    dense=.1*np.c_[(np.cos(el)*np.cos(az)).ravel(),(np.cos(el)*np.sin(az)).ravel(),np.sin(el).ravel()]
    for t in np.arange(1.,1.61,.1):
        center=np.array([1.+.2*(t-1.),.3,.6]);mapping.current_points=center+dense
        mapping.track(mapping.current_points,float(t));mapping.stamp=float(t)
    track=next(iter(mapping.tracks.values()));old_stamp=track['stamp'];old_velocity=track['velocity'].copy()
    t=old_stamp+.1;center=np.array([1.+.2*(t-1.),.3,.6]);point=center+np.array([0.,0.,.1])
    mapping.current_points=point[None];mapping.instantaneous_exact_geometry=False
    mapping.track(mapping.current_points,t);mapping.stamp=t
    assert track['stamp']==old_stamp
    np.testing.assert_array_equal(track['velocity'],old_velocity)
    sparse=track['sparse_surface'];assert sparse['stamp']==t
    # A fresh surface return bounds the entire rigid sphere even though its
    # center remains ambiguous; every allowed later displacement stays covered.
    rng=np.random.default_rng(111);directions=rng.normal(size=(200,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    for elapsed in [0.,.1,.3]:
        surface=center+.1*directions+elapsed*mapping.maximum_speed*directions[::-1]
        assert np.max(np.linalg.norm(surface-sparse['anchor'],axis=1))<=sparse['radius']+elapsed*mapping.maximum_speed
