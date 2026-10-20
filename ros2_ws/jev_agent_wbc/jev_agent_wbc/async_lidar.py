"""One in-flight measured scan; mapping never blocks the physical drive clock.

The original RuntimeLidar instance (including pre-roll history) is transferred
once. Every accepted scan is integrated in order by its unmodified algorithm.
Result timestamps are acquisition times, never completion or consumption times.
"""
import multiprocessing as mp
import pickle
from queue import Empty, Full
import time


def _worker(inbox,outbox,lidar,integration_options):
    # This worker is bounded to one finite physical episode. Avoid a cyclic-GC
    # scan of the accumulated map in the middle of a sensor lease.
    import gc
    gc.collect()
    gc.disable()
    outbox.put(pickle.dumps(dict(ready=True)))
    while True:
        item=inbox.get()
        if item is None:return
        state,base_z,stamp,returns,origins,extra=pickle.loads(item)
        started=time.monotonic()
        try:
            snapshot=lidar.integrate(state,base_z,stamp,returns,origins,**integration_options)
            outbox.put(pickle.dumps(dict(snapshot=snapshot,extra=extra,
                integration_wall_s=time.monotonic()-started),protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as error:
            outbox.put(pickle.dumps(dict(error=f'{type(error).__name__}: {error}')))
            return


class AsyncLidar:
    def __init__(self,lidar,**integration_options):
        ctx=mp.get_context('spawn')
        self.inbox,self.outbox=ctx.Queue(1),ctx.Queue(1)
        self.process=ctx.Process(target=_worker,args=(self.inbox,self.outbox,lidar,dict(integration_options)),name='physx-lidar-mapping')
        self.pending=False;self.closed=False;self.last_stamp=float('-inf')
        self.process.start()
        try:
            if not pickle.loads(self.outbox.get(timeout=30.)).get('ready'):
                raise RuntimeError('LiDAR worker did not initialize')
        except BaseException:
            self.close();raise
        self.samples=[]

    def submit(self,state,base_z,stamp,returns,origins,extra):
        if self.closed or not self.process.is_alive():raise RuntimeError('lidar_worker_unavailable')
        if self.pending:raise RuntimeError('lidar_processing_deadline')
        if stamp<=self.last_stamp:raise ValueError('LiDAR acquisition stamps must strictly increase')
        payload=pickle.dumps((state,base_z,stamp,returns,origins,extra),protocol=pickle.HIGHEST_PROTOCOL)
        try:self.inbox.put_nowait(payload)
        except Full:raise RuntimeError('lidar_processing_deadline') from None
        self.pending=True;self.last_stamp=stamp
        self.submitted_wall=time.monotonic()

    def poll(self,t):
        if self.closed or not self.process.is_alive():raise RuntimeError('lidar_worker_unavailable')
        try:result=pickle.loads(self.outbox.get_nowait())
        except Empty:return None
        self.pending=False
        if 'error' in result:raise RuntimeError('lidar_worker_failed: '+result['error'])
        stamp=float(result['snapshot']['observation_stamp'])
        if stamp>t+1e-9:raise RuntimeError('future_lidar_observation')
        self.samples.append(dict(acquired_s=stamp,consumed_s=t,
            integration_wall_s=result['integration_wall_s'],transport_wall_s=time.monotonic()-self.submitted_wall))
        return result

    def close(self):
        if self.closed:return
        self.closed=True
        try:self.inbox.put_nowait(None)
        except Full:pass
        self.process.join(timeout=.5)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=.5)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=.5)
        for queue in (self.inbox,self.outbox):queue.cancel_join_thread();queue.close()
