import numpy as np
import pytest
pytest.importorskip('numba')
from scipy.spatial.transform import Rotation
from jev_agent_wbc.multi_robot_geometry import NativeKinematicModel
from jev_agent_wbc.native_lift_solver import log_rotation,residual,solve,geometry_arguments


def model():
    home=np.eye(4);home[0,3]=.8
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('xyz',[.08,-.1,.3]).as_matrix();base[2,3]=.5
    return NativeKinematicModel(dict(base=base.tolist(),
        joints=[dict(omega=[0.,0.,1.],velocity=[0.,-.2,0.],q0=0.,lower=-2.,upper=2.,speed=.5),
                dict(omega=[0.,0.,0.],velocity=[0.,0.,1.],q0=0.,lower=-.4,upper=.4,speed=.1)],
        homes={'camera':home.tolist()},chains={'camera':[0,1]},
        shapes=[dict(body='camera',center_body=[.1,0.,0.],half_extents_body=[.2,.1,.15],part='camera_manipulator')],
        camera_parent='camera',camera_local=np.eye(4).tolist(),omnidirectional=True))


def test_native_rotation_log_at_small_and_near_pi_angles():
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
    for angle in [0.,1e-9,1e-5,.4,2.,np.pi-1e-8]:
        matrix=Rotation.from_rotvec(angle*axis).as_matrix()
        np.testing.assert_allclose(Rotation.from_rotvec(log_rotation(matrix)).as_matrix(),matrix,atol=2e-7)


@pytest.mark.parametrize('omni',[True,False])
def test_complete_native_residual_gradient_matches_finite_differences(omni):
    m=model();g=geometry_arguments(m);x=np.array([.07,-.04,.02,.13,.02]);previous=np.zeros(5)
    camera=m.camera(x);target_p=camera[:3,3]+[.02,-.01,.03]
    target_r=Rotation.from_rotvec([.04,-.02,.01]).as_matrix()@camera[:3,:3]
    centers=np.array([camera[:3,3]+[.17,.12,.05]]);radii=np.array([.08]);indices=np.array([0],np.int64)
    args=(previous,g,target_p,target_r,np.array([.1,.2]),centers,radii,indices,indices,np.array([.2,.2,.35]),omni,-.1,.3,.1)
    e,J=residual(x,*args)
    for j in range(5):
        a=x.copy();b=x.copy();a[j]+=1e-6;b[j]-=1e-6
        numerical=(residual(a,*args)[0]-residual(b,*args)[0])/2e-6
        np.testing.assert_allclose(J[:,j],numerical,atol=2e-5,rtol=1e-6)


def test_native_step_keeps_strict_box_bounds_and_evaluation_cap_for_unreachable_target():
    m=model();g=geometry_arguments(m);x=np.zeros(5);lo=np.full(5,-.005);hi=-lo
    fitted,calls=solve(x,x,lo,hi,g,np.array([3.,3.,3.]),np.eye(3),np.zeros(2),
        np.array([[2.,2.,2.]]),np.array([.1]),np.array([0],np.int64),np.array([0],np.int64),
        m.platform_limits,True,0.,.3,.1,maxeval=14)
    assert np.isfinite(fitted).all() and np.all(fitted>=lo) and np.all(fitted<=hi)
    assert calls<=14
