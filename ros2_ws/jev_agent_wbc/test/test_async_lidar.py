"""Equivalent map histories, acquisition timestamps and bounded transport faults."""
import copy
import time
import numpy as np
import pytest
from jev_agent_wbc.async_lidar import AsyncLidar
from jev_agent_wbc.sensors import RuntimeLidar
from jev_agent_wbc.contracts import native


def scan(lidar,t):
    state=np.array([0.,0.,0.,0.,-.6,1.,0.,.4,0.])
    origins=lidar.origins(state,0.)
    wall=np.array([[3.5,y,z] for y in np.linspace(-1.,1.,15) for z in np.linspace(.1,.8,8)])
    points=[wall.copy(),wall.copy()]
    return state,0.,t,points,origins


def wait_packet(worker,t):
    deadline=time.monotonic()+5.
    while time.monotonic()<deadline:
        packet=worker.poll(t)
        if packet is not None:return packet
        time.sleep(.005)
    pytest.fail('mapping worker did not return')


@pytest.mark.parametrize('options',[{},dict(acquisition_duration_s=0.)])
def test_async_mapping_preserves_preroll_and_every_output_field(options):
    reference=RuntimeLidar()
    for t in np.arange(-.8,.01,.1):reference.integrate(*scan(reference,float(t)),**options)
    worker=AsyncLidar(copy.deepcopy(reference),**options)
    try:
        for t in (.1,.2,.3,.4):
            inputs=scan(reference,t)
            expected=reference.integrate(*inputs,**options)
            worker.submit(*inputs,extra=dict(acquired=t))
            result=wait_packet(worker,t+.04)
            assert native(result['snapshot'])==native(expected)
            assert result['extra']['acquired']==t
            assert result['snapshot']['observation_stamp']==pytest.approx(t)
            assert worker.samples[-1]['consumed_s']==t+.04
    finally:worker.close()
    assert not worker.process.is_alive()


def test_busy_transport_rejects_backlog_and_dead_worker_fails_closed():
    lidar=RuntimeLidar();worker=AsyncLidar(lidar)
    try:
        worker.submit(*scan(lidar,.1),extra={})
        with pytest.raises(RuntimeError,match='deadline'):worker.submit(*scan(lidar,.2),extra={})
        wait_packet(worker,.15)
        with pytest.raises(ValueError,match='strictly'):worker.submit(*scan(lidar,.1),extra={})
        worker.process.terminate();worker.process.join(timeout=2.)
        with pytest.raises(RuntimeError,match='unavailable'):worker.poll(.2)
    finally:worker.close()


def test_future_result_is_rejected():
    lidar=RuntimeLidar();worker=AsyncLidar(lidar)
    try:
        worker.submit(*scan(lidar,.1),extra={})
        with pytest.raises(RuntimeError,match='future'):wait_packet(worker,0.)
    finally:worker.close()
