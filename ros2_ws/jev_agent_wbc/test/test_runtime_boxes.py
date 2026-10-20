import numpy as np
from jev_agent_wbc.runtime_obstacles import local_cover,box_rotation,ray_distances
from jev_agent_wbc.sensors import sphere_raycast
from jev_agent_wbc.scenarios import RuntimeScene


def box():
    return dict(id='ground_box',shape='box',size=[.4,.6,.8],yaw_rad=.7,
        radius=np.linalg.norm([.4,.6,.8])/2,center=np.array([1.,2.,.4]))


def test_cuboid_cover_contains_all_faces_corners_and_volume():
    obj=box();offsets,radii=local_cover(obj)
    mesh=np.stack(np.meshgrid(*[np.linspace(-s/2,s/2,13) for s in obj['size']],indexing='ij'),axis=-1).reshape(-1,3)
    points=mesh@box_rotation(obj).T
    assert (np.linalg.norm(points[:,None]-offsets,axis=2)-radii).min(axis=1).max()<1e-12
    assert len(offsets)>8


def test_raycast_hits_actual_rotated_box_face_and_respects_occlusion():
    obj=box();direction=box_rotation(obj)[:,0]
    origin=obj['center']-2*direction
    rays=np.array([direction,-direction])
    distances=ray_distances(origin,rays,obj)
    np.testing.assert_allclose(distances[0],1.8,atol=1e-12)
    assert np.isinf(distances[1])
    farther=dict(center=origin+3*direction,radius=.1)
    hits=sphere_raycast(origin,rays,[farther,obj])
    np.testing.assert_allclose(hits,[origin+1.8*direction],atol=1e-12)


def test_runtime_box_audit_expands_complete_volume_without_changing_spheres():
    obj=box();center=obj.pop('center')
    obj['keyframes']=[dict(t=t,position=center.tolist(),velocity=[0.,0.,0.]) for t in (0.,1.)]
    sphere=dict(id='sphere',radius=.1,keyframes=[dict(t=t,position=[3.,0.,.5],velocity=[0.,0.,0.]) for t in (0.,1.)])
    runtime=RuntimeScene(dict(obstacles=[obj,sphere]));truth=runtime.truth_collision(.3)
    assert len(runtime.obstacles(.3))==2
    assert len(truth['surfel_centers'])==len(local_cover(obj)[0])+1
    np.testing.assert_allclose(truth['surfel_centers'][-1],[3.,0.,.5])
    assert truth['surfel_radii'][-1]==.1
    assert np.count_nonzero(truth['surfel_track_ids']==1)>8
