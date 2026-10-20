"""Causal activation/recovery of geometric body adaptation.

This only selects a tracking objective. All current-time physical constraints
and independent swept certificates remain active in both modes.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def update(controller,reference,state,position,rotation,at,z=0.,base_rotation=None):
    collision=controller.collision
    movers=np.flatnonzero(collision.track_ids>0)
    threat=False
    if len(movers):
        horizons=np.array([0.,.5,1.,1.5])
        centers=collision.centers[movers,None]+horizons[None,:,None]*collision.velocities[movers,None]
        radii=collision.radii[movers,None]+horizons[None]*collision.radius_rates[movers,None]
        gaps=collision.geometry.distances(state,centers.reshape(-1,3),z,base_rotation)-radii.ravel()[:,None]
        threat=bool(gaps.min()<.16)
    active=bool(getattr(controller,'body_adaptation_seen',False))
    if threat:
        controller.body_adaptation_seen=True
        controller.body_clear_since=None
    elif active:
        if getattr(controller,'body_clear_since',None) is None:controller.body_clear_since=at
        target,orientation,_=reference.sample(at)
        # Return to ordinary tracking only after the original clock catches
        # up. Switching an advanced camera back too early would reverse it.
        recovered=(np.linalg.norm(position-target)<.015 and
            Rotation.from_matrix(rotation@orientation.T).magnitude()<.035)
        if recovered and at-controller.body_clear_since>=.5:
            controller.body_adaptation_seen=False
    return dict(predicted_body_conflict=threat,body_adaptation_active=bool(
        getattr(controller,'body_adaptation_seen',False)))
