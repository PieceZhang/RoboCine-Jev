"""Exact strictly convex 1D/2D QP over the unchanged half-space intersection."""
import numpy as np


def solve(H,g,A,lo,hi):
    C=np.vstack([A[np.isfinite(hi)],-A[np.isfinite(lo)]])
    b=np.r_[hi[np.isfinite(hi)],-lo[np.isfinite(lo)]]
    inverse=np.linalg.inv(H);free=-inverse@g
    if not len(b) or np.all(C@free<=b+1e-9):return free
    if len(g)==1:
        low=np.max(b[C[:,0]<0]/C[C[:,0]<0,0],initial=-np.inf)
        high=np.min(b[C[:,0]>0]/C[C[:,0]>0,0],initial=np.inf)
        if low>high+1e-8:raise RuntimeError('TPR task-conditioned safety set empty')
        return np.array([np.clip(free[0],low,high)])
    if len(g)!=2:raise ValueError('only 1D/2D objectives are supported')
    # An optimum outside the interior lies on an edge. Intersect every edge
    # with every half-space to obtain its feasible interval, then project its
    # unconstrained quadratic minimum onto that interval. This is O(m^2),
    # avoiding O(m^3) enumeration and checking of all line intersections.
    normal=C@inverse
    projected=free-(C@free-b)[:,None]*normal/np.sum(normal*C,axis=1)[:,None]
    direction=np.c_[-C[:,1],C[:,0]]
    slopes=C@direction.T;remaining=b[:,None]-C@projected.T
    parallel=abs(slopes)<=1e-12
    with np.errstate(divide='ignore',invalid='ignore'):
        crossings=remaining/slopes
    lower=np.max(np.where(slopes < -1e-12,crossings,-np.inf),axis=0)
    upper=np.min(np.where(slopes > 1e-12,crossings,np.inf),axis=0)
    valid=(lower<=upper+1e-9)&~np.any(parallel&(remaining < -1e-8),axis=0)
    parameters=np.clip(np.zeros(len(b)),lower,upper)
    candidates=projected[valid]+parameters[valid,None]*direction[valid]
    if not len(candidates):raise RuntimeError('TPR task-conditioned safety set empty')
    valid=np.all(C@candidates.T<=b[:,None]+1e-8,axis=0)
    candidates=candidates[valid]
    if not len(candidates):raise RuntimeError('TPR task-conditioned safety set empty')
    objective=.5*np.einsum('ni,ij,nj->n',candidates,H,candidates)+candidates@g
    return candidates[np.argmin(objective)]
