"""Optional compiled backend parity, run in the Isaac numerical environment."""
import numpy as np
import pytest
pytest.importorskip('numba')
from scipy.spatial.transform import Rotation
from jev_agent_wbc.kc_svsdf import CollisionGeometry
from jev_agent_wbc.compiled_geometry import primitive_arrays


@pytest.mark.parametrize('batch',[False,True])
@pytest.mark.parametrize('gradient',[False,True])
def test_compiled_shapes_match_portable_geometry_with_measured_tilt(batch,gradient):
    rng=np.random.default_rng(721);g=CollisionGeometry()
    states=rng.normal(size=(17,9))*.3;states[:,4:6]+=[-.9,2.1]
    rotations=Rotation.from_euler('xyz',rng.normal(size=(17,3))*.3).as_matrix()
    positions,rotations=g.frame_arrays(states,.02,rotations)
    points=rng.normal(size=(17,431,3))*.7
    if not batch:positions,rotations,points=positions[0],rotations[0],points[0]
    original=g._primitive_arrays(positions,rotations,points,gradient)
    compiled=primitive_arrays(g,positions,rotations,points,gradient)
    if gradient:
        for a,b in zip(original,compiled):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
    else:np.testing.assert_allclose(original,compiled,rtol=1e-10,atol=1e-10)


def test_compiled_surfaces_inside_outside_and_empty_batch():
    g=CollisionGeometry();state=np.array([0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])
    pp,rr=g.frame_arrays(state[None]);points=[]
    for body,name,kind,size,offset,sr in g.shapes:
        origin=pp[0,body]+rr[0,body]@offset;rotation=rr[0,body]@sr
        samples=np.vstack([np.zeros(3),np.eye(3)*.01,-np.eye(3)*.01,np.eye(3)*.1,-np.eye(3)*.1])
        points.extend(origin+samples@rotation.T)
    points=np.array(points)
    np.testing.assert_allclose(g._primitive_arrays(pp[0],rr[0],points),primitive_arrays(g,pp[0],rr[0],points),atol=1e-12)
    empty=primitive_arrays(g,pp[0],rr[0],np.empty((0,3)),True)
    assert empty[0].shape==(0,len(g.shapes)) and empty[1].shape==(0,len(g.shapes),3)


def test_warmup_covers_broadcast_copy_and_readonly_inputs_without_live_compilation():
    from jev_agent_wbc.compiled_geometry import primitive_batch
    g=CollisionGeometry();state=np.array([0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])
    pp,rr=g.frame_arrays(state[None])
    primitive_arrays(g,pp,rr,np.array([[[3.,2.,1.]]]),True)
    warmed=set(primitive_batch.signatures)
    positions,rotations=g.frame_arrays(np.tile(state,(7,1)))
    for dtype in (np.float32,np.float64):
        for batch in (1,7):
            points=np.ones((batch,13,3),dtype=dtype)
            for readonly in (False,True):
                points.flags.writeable=not readonly
                for gradient in (False,True):
                    primitive_arrays(g,positions,rotations,points,gradient)
                assert points.flags.writeable is not readonly
                assert positions.flags.writeable and rotations.flags.writeable
                assert set(primitive_batch.signatures)==warmed
