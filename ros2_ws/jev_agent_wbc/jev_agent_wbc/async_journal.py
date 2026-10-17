"""Ordered evidence writes without filesystem latency in the control thread."""
from queue import SimpleQueue
from threading import Thread


class AsyncJournal:
    def __init__(self,path):
        self.stream=path.open('w')
        self.queue=SimpleQueue();self.error=None;self.closed=False
        self.worker=Thread(target=self._run,name='jev-evidence-writer',daemon=True)
        self.worker.start()

    def _run(self):
        try:
            while True:
                value=self.queue.get()
                if value is None:break
                self.stream.write(value)
                self.stream.flush()
        except Exception as error:self.error=error
        finally:self.stream.close()

    def write(self,value):
        if self.error is not None:raise RuntimeError('evidence journal failed') from self.error
        if self.closed:raise ValueError('evidence journal is closed')
        self.queue.put(value)

    def flush(self):
        # Every queued write is flushed by the worker. close() drains the queue.
        if self.error is not None:raise RuntimeError('evidence journal failed') from self.error

    def close(self):
        if not self.closed:
            self.closed=True;self.queue.put(None);self.worker.join()
        if self.error is not None:raise RuntimeError('evidence journal failed') from self.error
