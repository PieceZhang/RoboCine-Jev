"""Independent operating model deadline/caps/reference invariants, no Isaac."""
import importlib.util
from pathlib import Path
import itertools
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


model=module('terminal_model',ROOT/'validation/go2_terminal_allocation_model_husky_20261020.py')
backend=module('terminal_model_backend_test',ROOT/'Robot_Go2/wbc_backend.py')


def test_all_limit_signs_and_reversal_brake_by_original_two_second_deadline():
    for previous in itertools.product([-.2,0.,.2],repeat=3):
        for desired in itertools.product([-.2,0.,.2],repeat=2):
            times,commands=model.issued_schedule(desired,np.array(previous))
            assert np.max(abs(commands))<=.2+1e-12
            slew=np.diff(np.vstack([previous,commands]),axis=0)/.02
            assert np.all(abs(slew)<=np.array([.5,.5,.8])+1e-9)
            assert np.all(commands[times>=2.-1e-9]==0)
            assert len(commands[times>=2.-1e-9])==150


def test_model_preserves_original_full_pose_and_root_height_without_extrapolation():
    body=np.eye(4);body[:3,:3]=Rotation.from_euler('xyz',[.06,-.04,.2]).as_matrix();body[2,3]=.3
    link=body.copy();link[:3,3]+=body[:3,:3]@[0,0,.07]
    endpoint=np.eye(4);endpoint[:3,3]=link[:3,3]+body[:3,:3]@[.5,.1,.15]
    endpoint[:3,:3]=body[:3,:3]@Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix()
    original=endpoint.copy();domain=backend.PolicyDomain().check([.5,0,.5,1,0,0,0])
    data=model.model_rollout(np.array([-.08,.04]),np.zeros(3),np.zeros(3),[(-.6,[0,0,0])],body,endpoint,
        np.asarray(domain['xyz_outer_bounds']),np.asarray(domain['euler_outer_bounds_rad']))
    np.testing.assert_array_equal(endpoint,original)
    np.testing.assert_allclose(data['native_xyz'][0],backend.camera_to_mixed(endpoint,link,np.eye(4))[:3])
    np.testing.assert_array_equal(data['native_xyz'][:,2],np.full(1001,endpoint[2,3]))
    np.testing.assert_array_equal(data['base_positions'][:,2],np.full(1001,body[2,3]))
    assert np.all(data['issued'][100:]==0)
    assert data['margins'].shape[1]==12


def test_zero_input_stationary_model_cannot_fabricate_a_posture_change():
    body=np.eye(4);body[2,3]=.3;endpoint=np.eye(4);endpoint[:3,3]=[.5,.1,.5]
    domain=backend.PolicyDomain().check([.5,0,.5,1,0,0,0])
    data=model.model_rollout(np.zeros(2),np.zeros(3),np.zeros(3),[(-.6,[0,0,0])],body,endpoint,
        np.asarray(domain['xyz_outer_bounds']),np.asarray(domain['euler_outer_bounds_rad']))
    np.testing.assert_array_equal(data['body_velocity'],np.zeros((1001,3)))
    np.testing.assert_array_equal(data['native_xyz'],np.tile(data['native_xyz'][0],(1001,1)))
    assert data['margins'].min()>0
