"""Bounded integration of fresh Pink velocities for one explicit PD+gravity sum."""
import numpy as np


class UpperReference:
    def __init__(self,joint_names,initial,limits,max_lead_rad=.05):
        self.names=tuple(joint_names);self.reference=np.asarray(initial,float).copy();self.limits=np.asarray(limits,float)
        self.lead=float(max_lead_rad)
        if (self.reference.shape!=(len(self.names),) or self.limits.shape!=(len(self.names),2)
                or not np.isfinite(self.reference).all() or not np.isfinite(self.limits).all()
                or np.any(self.limits[:,0]>=self.limits[:,1]) or not np.isfinite(self.lead) or self.lead<=0):
            raise ValueError('finite original limits, joint state and positive reference lead required')
        self.reference=np.clip(self.reference,self.limits[:,0],self.limits[:,1])

    def update(self,measured,velocity,dt):
        measured=np.asarray(measured,float);velocity=np.asarray(velocity,float)
        if (measured.shape!=self.reference.shape or velocity.shape!=self.reference.shape
                or not np.isfinite([measured,velocity]).all() or not np.isfinite(dt) or dt<=0
                or np.any(np.abs(velocity)>.500001)):
            raise ValueError('fresh finite Pink velocity within original upper operating bound required')
        proposed=self.reference+velocity*dt
        low=np.maximum(self.limits[:,0],measured-self.lead);high=np.minimum(self.limits[:,1],measured+self.lead)
        if np.any(low>high):raise ValueError('measured upper state outside original physical position limits')
        self.reference=np.clip(proposed,low,high)
        return self.reference.copy(),dict(antiwindup_active=bool(np.any(np.abs(self.reference-proposed)>1e-12)),
            maximum_reference_lead_rad=float(np.max(np.abs(self.reference-measured))),
            maximum_position_tracking_error_rad=float(np.max(np.abs(self.reference-measured))),
            max_lead_rad=self.lead,fresh_velocity_integrated=True)
