"""Native Giskard physical bridge constraints, not a substitute for GPU evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROBOT=Path(__file__).resolve().parents[1]/'Robot_TIAGo'
sys.path[:0]=[str(ROBOT),str(ROBOT/'wbc')]
from collision_model import CollisionBarriers
from physical_control import CameraController


def box_fixture(distance=.04):
    metadata=dict(shapes=[dict(name='fixed',body='fixed',kind='box',size=[1.,1.,1.],T_body_shape=np.eye(4).tolist()),
        dict(name='moving',body='moving',kind='box',size=[1.,1.,1.],T_body_shape=np.eye(4).tolist())],allowed_pairs=[])
    poses={'fixed':np.eye(4),'moving':np.eye(4)}
    poses['moving'][0,3]=1.+distance
    js={'fixed':np.zeros((6,10)),'moving':np.zeros((6,10))};js['moving'][0,2]=1.
    return metadata,poses,js


def test_fcl_nearest_points_produce_correct_separation_gradient():
    pytest.importorskip('fcl')
    metadata,poses,js=box_fixture()
    A,b,report=CollisionBarriers(metadata).query(poses,js)
    np.testing.assert_allclose(A[0],np.eye(10)[2],atol=1e-10)
    np.testing.assert_allclose(b,[-3*(.04-.005)],atol=1e-10)
    assert report['active_pair_count']==1


def test_real_giskard_solver_respects_hard_self_separation():
    pytest.importorskip('piqp');pytest.importorskip('fcl')
    from native_solver import NativeGiskardSolver
    metadata,poses,js=box_fixture(distance=.025)
    spec=json.loads((ROBOT/'robot.json').read_text())
    solver=NativeGiskardSolver(spec,metadata)
    previous=np.zeros(10);previous[2]=-.06
    request=dict(q=np.array([j['home'] for j in spec['joints']]),previous=previous,
        poses=poses,body_jacobians=js,mode='actuator',desired_rates=np.r_[0.,0.,-.2,np.zeros(7)])
    result=solver.solve(request)
    assert result['solver'].startswith('giskardpy.') and not result['big_ball_mode']
    assert -.06-1e-7<=result['command'][2]<=-.05+1e-7
    assert result['hard_constraint_violation']<1e-6


def test_official_allowed_pair_is_explicitly_filtered():
    pytest.importorskip('fcl')
    metadata,poses,js=box_fixture(distance=-.01)
    metadata['allowed_pairs']=[['fixed','moving']]
    A,b,_=CollisionBarriers(metadata).query(poses,js)
    assert A.shape==(0,10) and b.shape==(0,)


def test_unallowed_penetration_is_rejected():
    pytest.importorskip('fcl')
    metadata,poses,js=box_fixture(distance=-.01)
    with pytest.raises(RuntimeError,match='Nonallowed physical collision'):
        CollisionBarriers(metadata).query(poses,js)


def test_independent_guard_preserves_braking_and_acceleration():
    spec=json.loads((ROBOT/'robot.json').read_text())
    guard=CameraController(spec['joints'],spec['wheel_radius_m'],spec['wheel_separation_m'])
    q=guard.home.copy();q[0]=guard.upper[0]-.002
    u=guard.guard(q,np.ones(10)*100.)
    assert np.all(np.abs(u)<=guard.accel*.01+1e-10)
    assert u[2]<=1e-10


def test_native_entry_does_not_call_historical_optimizer():
    source=(ROBOT/'giskard_runtime.py').read_text()
    assert "wbc/physical_control.py" in source
    assert 'wbc.velocity(' not in source
    assert 'native.command(packet)' in source
    assert "session.reference(absolute_t)" in source
