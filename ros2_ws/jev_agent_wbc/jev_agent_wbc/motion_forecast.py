"""Bounded observed-acceleration preference; never a collision certificate."""
import numpy as np
from scipy.spatial import cKDTree


def acceleration(collision):
    values=np.zeros_like(collision.velocities)
    if collision.physical_time-collision.stamp>.25:return values
    for track in collision.scene.get('tracks',[]):
        a=track.get('acceleration')
        if a is None or track.get('age_s',1.)>.25:continue
        a=np.asarray(a,float)
        if a.shape==(3,) and np.isfinite(a).all() and np.linalg.norm(a)<=3.:
            values[collision.track_ids==track['id']]=a
    return values


def forecast(collision,offset):
    """Measured acceleration lasts at most 0.5 s, followed by coasting.

    This is a search hypothesis. Execution uses the original current-observation
    safety model and unchanged nonlinear swept certificates.
    """
    result=collision.at_time(offset);tau=min(.5,max(0.,offset))
    result.centers=result.centers+acceleration(collision)*(tau*max(0.,offset)-.5*tau*tau)
    result.tree=cKDTree(result.centers)
    return result
