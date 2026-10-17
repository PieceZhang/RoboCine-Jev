"""CPU sensor simulation/map fusion off the servo thread, with measured stamps."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np


_sensor=None
_runtime=None


def initialize_sensor(sensor,runtime):
    from .shot_preview_worker import initialize_worker
    initialize_worker()
    global _sensor,_runtime
    _sensor,_runtime=sensor,runtime


def sensor_ready():
    return True


def scan_snapshot(state,base_z,stamp):
    return _sensor.synthetic_scan(_runtime,state,base_z,stamp)


class AsyncRuntimeLidar:
    """One acquisition in flight; old maps retain their age and braking rule."""
    def __init__(self,sensor,runtime,initial):
        self.pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_sensor,initargs=(sensor,runtime))
        self.pool.submit(sensor_ready).result(timeout=30.)
        self.observed=initial
        self.pending=None
        self.requested=float(initial['observation_stamp'])
        self.completed=0

    def poll(self,state,base_z,stamp):
        if self.pending is not None and self.pending.done():
            result=self.pending.result()
            self.pending=None
            at=float(result['observation_stamp'])
            if not self.observed['observation_stamp']<=at<=stamp+1e-9:
                raise ValueError('sensor result timestamp outside acquisition order')
            self.observed=result
            self.completed+=1
        if self.pending is None and stamp-self.requested>=.1-1e-9:
            self.pending=self.pool.submit(scan_snapshot,np.asarray(state).copy(),base_z,stamp)
            self.requested=stamp
        return self.observed

    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True)
