import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from jev_agent_wbc.task_space_planning import MeasuredCameraPrediction
from jev_agent_wbc.whole_body_qp import whole_body_kinematics


class SmoothReference:
    duration=5.
    def sample(self,t):
        return (np.array([.2*t,.03*t*t,.7]),Rotation.from_rotvec([0.,.1*t,0.]).as_matrix(),
                np.array([.2,.06*t,0.,0.,.1,0.]))


@pytest.mark.parametrize('t',[.2,.5,1.1,1.9,2.4])
def test_measured_bridge_twist_is_the_derivative_of_its_pose(t):
    state=np.r_[0.,0.,.2,0.,-.7,1.8,0.,-1.1,0.]
    prediction=MeasuredCameraPrediction(SmoothReference(),state,.002,.1,2.)
    eps=1e-6;pa,ra=prediction.pose(t-eps);pb,rb=prediction.pose(t+eps)
    expected=np.r_[(pb-pa)/(2*eps),Rotation.from_matrix(rb@ra.T).as_rotvec()/(2*eps)]
    np.testing.assert_allclose(prediction.sample(t)[2],expected,atol=2e-9,rtol=2e-8)


def test_measured_bridge_keeps_initial_pose_and_joins_original_endpoint():
    state=np.r_[0.,0.,.2,0.,-.7,1.8,0.,-1.1,0.]
    path=SmoothReference();prediction=MeasuredCameraPrediction(path,state,.002,.1,2.)
    p,r,_=whole_body_kinematics(state,.002)
    np.testing.assert_allclose(prediction.sample(.1)[0],p,atol=1e-12)
    np.testing.assert_allclose(prediction.sample(.1)[1],r,atol=1e-12)
    for actual,expected in zip(prediction.sample(2.1),path.sample(2.1)):
        np.testing.assert_allclose(actual,expected,atol=1e-12)
