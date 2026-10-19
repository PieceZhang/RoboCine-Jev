"""Bounded connector search: smooth whole-body forecast with a velocity fallback."""
from .free_motion_preview import FreeMotionPreview
from .free_motion_lattice import FreeMotionLattice


class FreeMotionSearch:
    def __init__(self):
        self.smooth=FreeMotionPreview();self.lattice=FreeMotionLattice();self.info={}

    def preference(self,controller,state,goal,target,t,z):
        preferred=self.smooth.preference(controller,state,goal,target,t,z)
        self.info=dict(self.smooth.info,method='whole_body_smooth_preview')
        if not self.info.get('active') and not self.info.get('failure'):return preferred
        if self.info.get('active') and self.info.get('sampled_forecast_clearance_m',-1.)>=.10:return preferred
        fallback=self.lattice.preference(controller,state,goal,target,t,z)
        self.info=dict(self.lattice.info,smooth_search=self.smooth.info)
        return fallback
