import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from jev_agent_wbc.native_allocation import witness_posture


def test_native_posture_hint_is_bounded_in_time_and_original_joints_and_preserves_base_tilt():
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix();base[:3,3]=[1.,2.,.4]
    report=dict(physical_valid=True,camera_path_id='exact_unchanged_camera',times_s=[1.,2.],
        states=[[0.,0.,0.,0.],[.2,.4,.2,.1]],native_model_document=dict(base=base.tolist(),
            joints=[dict(name='native_joint',q0=.5,lower=0.,upper=1.)]))
    hint=witness_posture(report,1.5)
    assert hint['camera_path_id']=='exact_unchanged_camera'
    assert hint['joint_positions']['native_joint']==pytest.approx(.55)
    T=np.array(hint['world_from_base']);np.testing.assert_allclose(T[:3,3],[1.1,2.2,.4])
    np.testing.assert_allclose(T[:3,:3],Rotation.from_euler('z',.1).as_matrix()@base[:3,:3])
    assert witness_posture(report,.9) is None and witness_posture(report,2.1) is None
    report['states'][1][3]=2.
    with pytest.raises(ValueError,match='joint bounds'):witness_posture(report,2.)
    report['physical_valid']=False
    assert witness_posture(report,1.5) is None
