"""A single immutable acquisition, completed before the caller may advance physics."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from physx_lidar import NativeRaycaster


def adapter(scan):
    value=object.__new__(NativeRaycaster)
    value.executor=ThreadPoolExecutor(max_workers=1)
    value.pending=None;value.scan=scan
    value.batch=SimpleNamespace(close=lambda:None)
    return value


def test_pipeline_freezes_origins_and_rejects_overlapping_acquisitions():
    release=Event()
    def scan(origins):
        assert release.wait(2.)
        return origins
    value=adapter(scan)
    original=np.array([[.25,.5,.75]])
    try:
        value.begin(original)
        original[:]=9.
        with pytest.raises(RuntimeError,match='already in flight'):value.begin(original)
        release.set()
        np.testing.assert_array_equal(value.finish(),[[.25,.5,.75]])
        with pytest.raises(RuntimeError,match='no ray query'):value.finish()
    finally:release.set();value.close()


def test_pipeline_propagates_query_failure_before_release_of_barrier():
    def fail(origins):raise RuntimeError('query failed')
    value=adapter(fail)
    try:
        value.begin([[0.,0.,0.]])
        with pytest.raises(RuntimeError,match='query failed'):value.finish()
        assert value.pending is None
    finally:value.close()
