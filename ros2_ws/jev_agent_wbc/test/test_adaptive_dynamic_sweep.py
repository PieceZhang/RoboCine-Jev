import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.kc_svsdf import advance_sequence


@pytest.mark.parametrize('close',[False,True])
def test_adaptive_sweep_lower_bounds_dense_actual_motion_including_moving_obstacle(close):
    x=np.array([0.,0.,.3,0.,-.9,2.1,0.,-1.2,0.])
    u=np.array([.2,.4,.1,-.2,.3,-.1,.2,-.2])
    center=[.2,.1,.3] if close else [1.,1.,.4]
    scene=dict(format='jvs_lidar_surfels/v1',surfel_centers=[center,[3.5,0.,.3]],
        surfel_radii=[.1,.04],surfel_velocities=[[-.4,.2,.1],[0.,0.,0.]],
        surfel_track_ids=[1,0],surfel_radius_rates=[.1,0.],minimum_clearance_m=.025,
        observation_stamp=0.,surfel_radius=.04)
    c=DynamicCollision(scene,0.)
    rotation=Rotation.from_euler('xyz',[.03,-.02,.3]).as_matrix()
    proof=c.sweep(x,u,.1,tolerance=.0005,max_samples=33,base_rotation=rotation)
    ts=np.linspace(0.,.1,401);states=advance_sequence(x,u,ts)
    rots=Rotation.from_euler('z',u[1]*ts).as_matrix()@rotation
    values=c.geometry.distances_sequence(states,c.centers[None]+ts[:,None,None]*c.velocities[None],
        base_rotations=rots)-(c.radii[None]+ts[:,None]*c.radius_rates[None])[:,:,None]
    assert proof['lower'].min()<=values.min()+1e-10
    c.adaptive_sweep=False
    full=c.sweep(x,u,.1,tolerance=.0005,max_samples=33,base_rotation=rotation)
    if not close:
        assert len(proof['times'])==3 and len(full['times'])==33
        assert proof['lower'].min()>=c.margin
    else:
        np.testing.assert_array_equal(proof['lower'],full['lower'])


def test_distant_center_with_growing_occupancy_is_retained_by_local_barriers():
    state=np.array([0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])
    scene=dict(format='jvs_lidar_surfels/v1',surfel_centers=[[.8,.8,.4],[2.,0.,.3]],
        surfel_radii=[.04,1.9],surfel_radius=.04,surfel_track_ids=[0,1],
        surfel_velocities=[[0.,0.,0.],[0.,0.,0.]],observation_stamp=0.)
    collision=DynamicCollision(scene)
    for count in [1,8,None]:
        centers,radii=collision.active_obstacles(state,per_body=count)
        assert any(np.array_equal(center,[2.,0.,.3]) and radius==1.9
                   for center,radius in zip(centers,radii))
    full=collision.clearance_to(state,collision.centers,collision.radii).min()
    assert full<0.
    assert collision.clearance(state).min()==pytest.approx(full)
