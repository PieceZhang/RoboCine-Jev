"""Original endpoint hold with subordinate XY/native-yaw operating allocation.

This operating model does not predict learned actions, contact or joint dynamics.
It inherits the unchanged original input/deadline/history guards. Physical use
requires its own frozen candidate and measured validation; CPU proof is limited
to the explicit operating model.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

_spec=importlib.util.spec_from_file_location('go2_terminal_xy_base',Path(__file__).with_name('terminal_allocator.py'))
_base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_base)
TerminalDeadlineInfeasible=_base.TerminalDeadlineInfeasible
CAPS=_base.CAPS
SLEW=_base.SLEW
DT=_base.DT


class TerminalContext(_base.TerminalContext):
    def __init__(self,*args,target_yaw_rad=0.,**kwargs):
        if not np.isfinite(target_yaw_rad)or not -.5<=target_yaw_rad<=.5:
            raise ValueError('bounded finite subordinate native yaw preference required')
        super().__init__(*args,**kwargs)
        self.target_yaw=float(target_yaw_rad)
        self.model_yaw_error=None

    def _trajectory(self,first,target):
        key=(tuple(first),tuple(target))
        if key in self.cache:return self.cache[key]
        inputs=_base.schedule(self.previous,first,target,self.allocation_remaining,self.command_times)
        delayed=np.vstack([self.history_inputs,inputs])[self.delayed_indices]
        dt=self.intervals;start=self.events[:-1];end=self.events[1:]
        steady=delayed[:,0]*np.where(delayed[:,0]<0,.55,1.)
        attenuation=np.exp(-np.cumsum(dt)/.35);one_minus=-np.expm1(-dt/.35)
        after=attenuation*(self.measured[0]+np.cumsum(steady*one_minus/attenuation))
        before=np.r_[self.measured[0],after[:-1]]
        dx=before*.35*one_minus+steady*(dt-.35*one_minus)
        old_index=max(0,np.searchsorted(self.history_times,-.15+1e-12,side='right')-1)
        pending=self.history_inputs[old_index]
        a=np.minimum(start,.15);b=np.minimum(end,.15)
        residual_integral=(b-a)-(b*b-a*a)/(.3)
        dy=delayed[:,1]*dt+(self.measured[1]-pending[1])*residual_integral
        da=delayed[:,2]*dt+(self.measured[2]-pending[2])*residual_integral
        yaw=np.r_[0.,np.cumsum(da)];middle=yaw[:-1]+da/2
        displacement=np.c_[dx,dy,np.zeros_like(dx)]@self.body[:3,:3].T
        c,s=np.cos(middle),np.sin(middle)
        planar=np.c_[c*displacement[:,0]-s*displacement[:,1],s*displacement[:,0]+c*displacement[:,1]]
        roots=np.tile(np.eye(4),(len(self.events),1,1));roots[:,:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@self.body[:3,:3]
        roots[:,:3,3]=self.body[:3,3];roots[1:,:2,3]+=np.cumsum(planar,axis=0)
        links=roots@self.root_to_link;links[0]=self.link
        xyz=np.einsum('nji,nj->ni',links[:,:3,:3],self.ee[:3,3]-links[:,:3,3]);xyz[:,2]=self.ee[2,3]
        result=dict(links=links,xyz=xyz,issued=inputs);self.cache[key]=result;return result

    def _preference_error(self,first,target):
        data=self._trajectory(first,target);link=data['links'][-1]
        angles=Rotation.from_matrix(link[:3,:3].T@self.ee[:3,:3]).as_euler('xyz')
        yaw_error=np.arctan2(np.sin(angles[2]-self.target_yaw),np.cos(angles[2]-self.target_yaw))
        return np.r_[data['xyz'][-1,:2]-self.target,yaw_error]

    def solve(self):
        if self.phase=='allocate':
            current=self.backend.camera_to_mixed(self.endpoint,self.link,self.extrinsic)
            angle=Rotation.from_quat(current[[4,5,6,3]]).as_euler('xyz')[2]
            yaw_error=np.arctan2(np.sin(angle-self.target_yaw),np.cos(angle-self.target_yaw))
            scale=max(.05,self.allocation_remaining+.15)
            desired=np.clip(np.r_[(current[:2]-self.target)/np.array([.55,1.]),yaw_error]/scale,-CAPS,CAPS)
            for _ in range(2):
                first=self._first(desired);value=self._preference_error(first,desired)
                if np.linalg.norm(value[:2])<.0005 and abs(value[2])<.001:break
                columns=[];eps=1e-4
                for axis in range(3):
                    candidate=desired.copy();candidate[axis]=min(CAPS[axis],desired[axis]+eps)
                    if abs(candidate[axis]-desired[axis])<1e-12:candidate[axis]=max(-CAPS[axis],desired[axis]-eps)
                    distance=candidate[axis]-desired[axis]
                    columns.append((self._preference_error(self._first(candidate),candidate)-value)/distance)
                jacobian=np.column_stack(columns)
                if not np.isfinite(jacobian).all()or np.linalg.cond(jacobian)>1e6:break
                desired=np.clip(desired-np.linalg.solve(jacobian,value),-CAPS,CAPS)
                self.solver_iterations+=1
            self.desired=desired
        self.proposal=self._first(self.desired)
        error=self._preference_error(self.proposal,self.desired)
        self.model_target_error=error[:2];self.model_yaw_error=float(error[2])
        return self.proposal.copy()

    def manifest(self):
        result=super().manifest()
        result.update(profile='endpoint_xy58_15_yaw0_v1',
            model='bounded50Hz slew then fixeddeadline brake,150ms actual held-input delay; continuous vx steady map with tau.35; vy/yaw measured residual decays in original150ms window; full measured tilt/fixed root height; uncalibrated operating approximation',
            subordinate_native_yaw_preference_rad=self.target_yaw,
            model_final_native_yaw_error_rad=self.model_yaw_error,
            pending_other_axes_model='real held input history plus measured rate residual linearly decaying in original150ms inertia window; no new tau',
            operating_preference_certificate='none: measured operating hypothesis only; not a training or stability domain')
        return result
