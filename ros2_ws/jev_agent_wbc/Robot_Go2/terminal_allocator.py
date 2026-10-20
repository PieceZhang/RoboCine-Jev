"""Private endpoint-hold operating model; native learned18 actions stay active.

This allocates only original bounded body inputs. It predicts neither joint
motion nor contact/stability. Original measured safety and parking gates remain
authoritative. The World camera endpoint is copied, never retimed or rewritten.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

CAPS=np.array([.2,.2,.2])
SLEW=np.array([.5,.5,.8])
DT=.02


class TerminalDeadlineInfeasible(ValueError):
    """Caller must log infeasible and issue this original-slew safe brake."""
    def __init__(self,previous):
        self.safe_brake_command=np.clip(np.zeros(3),previous-SLEW*DT,previous+SLEW*DT)
        super().__init__('actual previous input cannot meet original2s deadline under original slew')


def schedule(previous,first,future_target,allocation_remaining_s,times):
    """Exact50Hz issued input sequence: allocate<=1.6s, then original slew0."""
    previous,first,target=map(lambda value:np.asarray(value,float),(previous,first,future_target))
    times=np.asarray(times,float)
    count=len(times);commands=np.zeros((count,3))
    if not count:return commands
    commands[0]=first
    brake_index=min(count,max(0,int(np.floor((allocation_remaining_s+1e-9)/DT))))
    if brake_index>1:
        steps=np.arange(1,brake_index)[:,None]*DT
        commands[1:brake_index]=first+np.clip(target-first,-steps*SLEW,steps*SLEW)
    if brake_index==0:
        commands[:]=previous-np.clip(previous,-SLEW*(np.arange(count)[:,None]+1)*DT,
                                              SLEW*(np.arange(count)[:,None]+1)*DT)
    elif brake_index<count:
        before=commands[brake_index-1]
        steps=(np.arange(count-brake_index)[:,None]+1)*DT
        commands[brake_index:]=before-np.clip(before,-steps*SLEW,steps*SLEW)
    return np.clip(commands,-CAPS,CAPS)


class TerminalContext:
    """One fresh measured snapshot, local model and bounded input proposal."""
    def __init__(self,backend,*,stamp_s,task_end_s,stop_seconds,endpoint_optical,extrinsic,
                 body,link,measured_velocity,previous_applied,input_history,target_xy=(.58,.15),
                 prediction_intervals=(0.,.2,.4,.6,.8,1.,1.2)):
        if stop_seconds<5. or not np.isfinite([stamp_s,task_end_s,stop_seconds]).all() or stamp_s<task_end_s-1e-9:
            raise ValueError('fresh terminal time and original stop>=5s required')
        self.backend=backend;self.stamp=float(stamp_s);self.end=float(task_end_s)
        self.elapsed=max(0.,self.stamp-self.end);self.stop_seconds=float(stop_seconds)
        self.allocation_remaining=max(0.,1.6-self.elapsed);self.deadline_remaining=max(0.,2.-self.elapsed)
        # Commands are ZOH between actual50Hz updates. For a non-grid Task.end,
        # zero must be issued at the last update BEFORE the absolute deadline.
        self.zero_updates_remaining=max(0,int(np.floor((self.deadline_remaining+1e-9)/DT)))
        self.first_input_bound=np.minimum(CAPS,SLEW*DT*self.zero_updates_remaining)
        self.finish_remaining=max(0.,self.stop_seconds-self.elapsed)
        self.endpoint=backend.rigid(endpoint_optical).copy();self.endpoint.setflags(write=False)
        self.extrinsic=backend.rigid(extrinsic).copy();self.body=backend.rigid(body).copy();self.link=backend.rigid(link).copy()
        self.measured=np.asarray(measured_velocity,float).copy();self.previous=np.asarray(previous_applied,float).copy()
        self.target=np.asarray(target_xy,float).copy()
        if self.measured.shape!=(3,) or self.previous.shape!=(3,) or self.target.shape!=(2,) or not np.isfinite(np.r_[self.measured,self.previous,self.target]).all() or np.any(abs(self.previous)>CAPS+1e-9):
            raise ValueError('finite actual velocity/input and2D operating target required')
        if np.any(abs(self.previous)>SLEW*DT*(self.zero_updates_remaining+1)+1e-9):
            raise TerminalDeadlineInfeasible(self.previous)
        self.ee=self.endpoint@np.linalg.inv(self.extrinsic)
        self.root_to_link=np.linalg.inv(self.body)@self.link
        self.phase='allocate'if self.elapsed+DT<=1.6+1e-9 else'brake'if self.elapsed<2.-1e-9 else'zero'
        if self.phase!='allocate':self.allocation_remaining=0.
        self.command_times=np.arange(max(1,int(np.ceil(max(self.finish_remaining,1.2)/DT))+1))*DT
        # Zero-order held50Hz input events, delayed150ms. Include exact query
        # times so state interpolation never substitutes a different endpoint.
        history=[(float(t)-self.stamp,np.asarray(u,float))for t,u in input_history]
        if len(history)<9:raise ValueError('actual50Hz input history must cover150ms delay')
        self.history_times=np.asarray([x[0]for x in history]);self.history_inputs=np.asarray([x[1]for x in history])
        if (self.history_inputs.shape!=(len(history),3) or not np.isfinite(np.r_[self.history_times,self.history_inputs.reshape(-1)]).all()
                or np.any(np.diff(self.history_times)<=0) or np.any(abs(self.history_inputs)>CAPS+1e-9)):
            raise ValueError('finite actual input history required')
        if (not np.allclose(np.diff(self.history_times),DT,atol=1e-9,rtol=0)
                or not np.isclose(self.history_times[-1],-DT,atol=1e-9,rtol=0)
                or self.history_times[0]>-.15+1e-9 or not np.allclose(self.history_inputs[-1],self.previous,atol=1e-9,rtol=0)):
            raise ValueError('fresh complete actual input history must end one control tick ago at previous_applied')
        self.query_times=np.asarray(prediction_intervals,float)
        if not np.isfinite(self.query_times).all()or np.any(self.query_times<0):raise ValueError('nonnegative finite model queries required')
        horizon=max(float(np.max(self.query_times)),self.finish_remaining)
        events=np.r_[0.,self.command_times+.15,self.history_times+.15,self.query_times,horizon]
        self.events=np.unique(events[(events>=0.)&(events<=horizon+1e-12)])
        self.events[-1]=horizon;self.intervals=np.diff(self.events)
        delayed=self.events[:-1]-.15
        times=np.r_[self.history_times,self.command_times]
        self.delayed_indices=np.clip(np.searchsorted(times,delayed+1e-12,side='right')-1,0,len(times)-1)
        self.cache={};self.desired=np.zeros(3);self.proposal=self._first(self.desired)
        self.solver_iterations=0;self.model_target_error=None

    def _first(self,desired):
        target=np.asarray(desired)if self.phase=='allocate'else np.zeros(3)
        first=np.clip(target,self.previous-SLEW*DT,self.previous+SLEW*DT)
        return np.clip(first,-self.first_input_bound,self.first_input_bound)

    def _trajectory(self,first,target):
        key=(tuple(first),tuple(target))
        if key in self.cache:return self.cache[key]
        inputs=schedule(self.previous,first,target,self.allocation_remaining,self.command_times)
        all_inputs=np.vstack([self.history_inputs,inputs]);delayed=all_inputs[self.delayed_indices]
        delta=self.intervals;v0=self.measured
        # A single continuous state model across input0. The steady-state map
        # is continuous with a slope kink; candidate sign never swaps the
        # dynamics of the measured initial velocity or pending old commands.
        steady=delayed[:,0]*np.where(delayed[:,0]<0,.55,1.)
        cumulative=np.cumsum(delta);attenuation=np.exp(-cumulative/.35);decay=np.exp(-delta/.35)
        after=attenuation*(v0[0]+np.cumsum(steady*(1-decay)/attenuation))
        before=np.r_[v0[0],after[:-1]]
        dx=before*.35*(1-decay)+steady*(delta-.35*(1-decay))
        vy=np.where(self.events[:-1]<.15-1e-12,v0[1],delayed[:,1])
        wz=np.where(self.events[:-1]<.15-1e-12,v0[2],delayed[:,2])
        yaw=np.r_[0.,np.cumsum(wz*delta)];middle=yaw[:-1]+wz*delta/2
        rotation=Rotation.from_euler('z',yaw).as_matrix()@self.body[:3,:3]
        displacement=np.c_[dx,vy*delta,np.zeros_like(dx)]@self.body[:3,:3].T
        c,s=np.cos(middle),np.sin(middle)
        planar=np.c_[c*displacement[:,0]-s*displacement[:,1],s*displacement[:,0]+c*displacement[:,1]]
        positions=np.tile(self.body[:3,3],(len(self.events),1));positions[1:,:2]+=np.cumsum(planar,axis=0)
        transforms=np.tile(np.eye(4),(len(self.events),1,1));transforms[:,:3,:3]=rotation;transforms[:,:3,3]=positions
        links=transforms@self.root_to_link
        xyz=np.einsum('nji,nj->ni',links[:,:3,:3],self.ee[:3,3]-links[:,:3,3])
        xyz[:,2]=self.ee[2,3]
        result=dict(links=links,xyz=xyz,issued=inputs)
        self.cache[key]=result
        return result

    def solve(self):
        if self.phase=='allocate':
            current=self.backend.camera_to_mixed(self.endpoint,self.link,self.extrinsic)
            scale=max(.05,self.allocation_remaining+.15)
            desired=np.r_[np.clip((current[:2]-self.target)/np.array([.55,1.])/scale,-.2,.2),0.]
            # Two bounded local updates; no unbounded optimizer/wait in physics.
            for _ in range(2):
                first=self._first(desired);value=self._trajectory(first,desired)['xyz'][-1,:2]-self.target
                if np.linalg.norm(value)<.0005:break
                columns=[];eps=1e-4
                for axis in range(2):
                    v=desired.copy();v[axis]+=eps
                    columns.append((self._trajectory(self._first(v),v)['xyz'][-1,:2]-self.target-value)/eps)
                jacobian=np.column_stack(columns)
                if not np.isfinite(jacobian).all()or np.linalg.cond(jacobian)>1e6:break
                desired[:2]=np.clip(desired[:2]-np.linalg.solve(jacobian,value),-.2,.2)
                self.solver_iterations+=1
            self.desired=desired
        self.proposal=self._first(self.desired)
        final=self._trajectory(self.proposal,self.desired)['xyz'][-1,:2]
        self.model_target_error=final-self.target
        return self.proposal.copy()

    def predict_link(self,first,elapsed):
        first=np.asarray(first,float)
        if first.shape!=(3,)or not np.isfinite(first).all()or np.any(abs(first)>CAPS+1e-8):
            raise ValueError('finite bounded candidate native input required')
        if not np.allclose(first,self.guard_applied(first),atol=1e-9,rtol=0):
            raise ValueError('prediction candidate must equal the actual admissible first input')
        if self.phase!='allocate':target=np.zeros(3)
        else:target=np.clip(self.desired+first-self.proposal,-CAPS,CAPS)
        data=self._trajectory(first,target)
        indices=np.flatnonzero(np.isclose(self.events,elapsed,atol=1e-10,rtol=0))
        if not len(indices):raise ValueError('query must be an explicitly configured prediction timestamp')
        return data['links'][int(indices[-1])].copy()

    def guard_applied(self,desired):
        desired=np.asarray(desired,float)
        if desired.shape!=(3,)or not np.isfinite(desired).all():raise ValueError('finite body command required')
        candidate=np.clip(desired,self.previous-SLEW*DT,self.previous+SLEW*DT)
        if self.phase!='allocate':candidate=self._first(np.zeros(3))
        return np.clip(candidate,-self.first_input_bound,self.first_input_bound)

    def manifest(self):
        return dict(profile='endpoint_xy58_15_v1',phase=self.phase,stamp_s=self.stamp,
            reference_stamp_s=self.end,reference_mode='explicit_original_endpoint_hold',
            original_endpoint_optical_sha256=hashlib.sha256(self.endpoint.tobytes()).hexdigest(),
            allocation_remaining_s=self.allocation_remaining,zero_input_deadline_s=self.end+2.,
            discrete_zero_updates_remaining=self.zero_updates_remaining,
            body_input_target_xy_link0_m=self.target.tolist(),desired_body_input=self.desired.tolist(),
            first_body_input_proposal=self.proposal.tolist(),model_final_target_error_xy_m=self.model_target_error.tolist(),
            model_iterations=self.solver_iterations,model='bounded50Hz input ramp,150ms delay,tau.35 with continuous steady(u)=u*(.55 if negative else1); forward lag uncalibrated; other axes measured inertia then delayed input; measured full tilt and fixed height',
            prediction_scope='native workspace operating approximation only; future joint/contact/learned18 stability unknown',
            original_world_task_modified=False,all18_learned_actions_unchanged=True)
