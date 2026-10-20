"""Private legged pose-only sampler equivalence; public sources stay untouched."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from jev_agent_wbc.representation import Pose, fingerprint
from jev_agent_wbc.wbc_protocol import CameraTrajectory, TimedTask, optical_camera

spec = importlib.util.spec_from_file_location('go2_private_optical_sampler', PACKAGE/'Robot_Go2/optical_sampler.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OpticalSampler = module.OpticalSampler


def task(start=13.7,seed=923):
    rng=np.random.default_rng(seed)
    times=np.r_[0., np.cumsum(rng.uniform(.01,1.,size=14))]
    rotations=Rotation.random(len(times),rng).as_quat()
    poses=tuple(Pose(tuple(p),tuple(q)) for p,q in zip(rng.normal(size=(len(times),3)),rotations))
    return TimedTask('same_camera',start,'measured_observation',CameraTrajectory(start,tuple(times),poses),.05,.15)


def reference(t, stamp):
    p=optical_camera(t.trajectory.sample(stamp));T=np.eye(4);T[:3,:3]=p.rotation;T[:3,3]=p.translation_m
    return T


@pytest.mark.parametrize('start',[0.,13.7,1_000_000.125])
def test_all_knots_random_and_adjacent_absolute_times(start):
    t=task(start);s=OpticalSampler(t);rng=np.random.default_rng(100)
    stamps=list(start+np.array(t.trajectory.times_s))
    stamps += list(rng.uniform(start,s.end_s,size=100))
    for k in stamps[1:-1][:12]:
        stamps += [np.nextafter(k,-np.inf),np.nextafter(k,np.inf)]
    for stamp in stamps:
        if start<=stamp<=s.end_s:np.testing.assert_allclose(s.sample(float(stamp)),reference(t,float(stamp)),rtol=0,atol=2e-14)
    assert s.task_id==t.task_id and s.camera_path_id==t.trajectory.id
    assert s.task_sha256==fingerprint(t.to_dict())


@pytest.mark.parametrize('axis',np.eye(3))
def test_pi_rotation_tie_preserves_original_arc(axis):
    t=TimedTask('pi',10.,'obs',CameraTrajectory(10.,(0.,1.),(
        Pose((1,2,3)),Pose((3,4,5),tuple(Rotation.from_rotvec(np.pi*axis).as_quat())))),.01,.02)
    s=OpticalSampler(t)
    for stamp in np.linspace(10,11,31):np.testing.assert_allclose(s(stamp),reference(t,stamp),rtol=0,atol=3e-15)


@pytest.mark.parametrize('stamp',[True,False,np.bool_(True),float('nan'),float('inf'),-float('inf'),'14',None])
def test_invalid_timestamp_rejected(stamp):
    with pytest.raises(ValueError):OpticalSampler(task()).sample(stamp)


def test_no_early_sample_or_implicit_expired_reference():
    s=OpticalSampler(task())
    for stamp in [np.nextafter(s.start_s,-np.inf),-1.]:
        with pytest.raises(ValueError):s(stamp,hold_terminal=True)
    for stamp in [np.nextafter(s.end_s,np.inf),s.end_s+1000.]:
        with pytest.raises(ValueError):s(stamp)
        np.testing.assert_array_equal(s(stamp,hold_terminal=True),s(s.end_s))
    for bad in [0,1,None,'True']:
        with pytest.raises(ValueError):s(s.end_s,hold_terminal=bad)


def test_returned_arrays_and_input_mutation_do_not_change_samples():
    original=task();document=original.to_dict();s=OpticalSampler(original);stamp=s.start_s+.05
    expected=s(stamp);changed=s(stamp);changed[:]=123
    for endpoint in [s.start_s,s.end_s]:s(endpoint)[:]=9
    document['camera_trajectory']['samples'][0]['pose']['translation_m'][0]=1e9
    np.testing.assert_array_equal(s(stamp),expected)
    np.testing.assert_allclose(s(s.start_s),reference(original,s.start_s),rtol=0,atol=3e-15)
    assert not s.manifest()['camera_reference_modified'] and not s.manifest()['supplies_twist']


def test_requires_validated_task_and_frame_contract():
    with pytest.raises(ValueError):OpticalSampler(task().to_dict())
    d=task().to_dict();d['camera_trajectory']['frame']='map'
    with pytest.raises(ValueError):OpticalSampler(TimedTask.from_dict(d))
