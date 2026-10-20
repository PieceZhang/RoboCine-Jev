"""Explicit diagnostic faults. Never sleeps or rewinds the physical process."""
import json
import math
from pathlib import Path


class FaultSchedule:
    def __init__(self,events=()):
        self.events=[dict(event) for event in events]
        self.emitted=set()
        for event in self.events:
            if event.get('kind') not in ('controller_delay','controller_error','lidar_gap'):
                raise ValueError('unknown diagnostic fault kind')
            at=event.get('time_s');duration=event.get('duration_s',0.)
            if not isinstance(at,(int,float)) or not math.isfinite(at) or at<0:
                raise ValueError('fault time must be finite and nonnegative')
            if not isinstance(duration,(int,float)) or not math.isfinite(duration) or not 0<=duration<=2.:
                raise ValueError('diagnostic fault duration must be in [0,2] seconds')
            if event['kind']!='controller_error' and duration<=0:
                raise ValueError('delay/gap duration must be positive')

    @classmethod
    def load(cls,path):
        return cls(() if path is None else json.loads(Path(path).read_text())['events'])

    def due(self,kind,t):
        result=[]
        for index,event in enumerate(self.events):
            if event['kind']==kind and index not in self.emitted and t>=event['time_s']:
                self.emitted.add(index);result.append(event)
        return result

    def active(self,kind,t):
        return any(e['kind']==kind and e['time_s']<=t<e['time_s']+e.get('duration_s',0.) for e in self.events)
