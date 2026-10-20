"""Causal, bounded rate schedules for the local passage-clock preference."""
import numpy as np


def profiles(previous,desired,horizon,nodes,elapsed,acceleration=1.):
    if not (horizon>0 and nodes>0 and elapsed>0 and acceleration>0):
        raise ValueError('positive clock horizons and rate bounds required')
    definitions=[(str(rate),rate,float('inf'),rate) for rate in (0.,.5,1.,1.8,2.5)]
    definitions += [('yield_short',0.,.75,max(1.8,desired)),('yield_long',0.,1.5,2.5),
                    ('advance_then_yield',2.5,1.,.5)]
    step=horizon/nodes;result=[]
    for name,first,switch,second in definitions:
        old=float(previous);rates=[];increments=[]
        for index in range(nodes):
            target=first if index*step<switch else second
            new=old+np.clip(target-old,-acceleration*step,acceleration*step)
            increments.append(.5*(old+new)*step);rates.append(new);old=new
        result.append(dict(name=name,rates=np.array(rates),increments=np.array(increments),
            next_rate=float(previous+np.clip(first-previous,-acceleration*elapsed,acceleration*elapsed))))
    return result
