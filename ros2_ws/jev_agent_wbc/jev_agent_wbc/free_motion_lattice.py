"""Short whole-body velocity lattice for an unconstrained-camera connector.

All observed movers and all eight rigid bodies enter each rollout. This is a
numerical preference only; fresh WBC and continuous certificates stay mandatory.
"""
import numpy as np
from .kc_svsdf import advance
from .motion_forecast import forecast
from .jvs_kinematics import JOINT_LOWER,JOINT_UPPER
from .transfer_geometry import angle


class FreeMotionLattice:
    def __init__(self):
        self.updated=-np.inf;self.selected=None;self.info={};self.goal=None

    def preference(self,controller,state,goal,target,t,z):
        collision=controller.collision
        moving=np.flatnonzero((collision.track_ids>0)&(np.linalg.norm(collision.velocities,axis=1)>.04))
        if not len(moving):self.info=dict(active=False);return target
        horizon=2.4;h=.12;n=20
        futures=[forecast(collision,(k+1)*h) for k in range(n)]
        probes=np.stack([f.centers[moving] for f in futures],axis=1)
        risk=collision.geometry.distances(state,probes.reshape(-1,3),z).reshape(len(moving),n,8)-collision.radii[moving,None,None]
        if risk.min()>.24:self.info=dict(active=False);return target
        if self.selected is not None and t-self.updated<.2 and np.linalg.norm(goal-self.goal)<.02:
            return np.r_[self.selected,target[2:]]
        pairs=[target[:2].copy(),np.zeros(2)]
        if self.selected is not None:pairs.append(self.selected.copy())
        pairs.extend(np.array([v,w]) for v in (-.45,-.25,.25,.45) for w in (-.8,-.4,0.,.4,.8))
        states=[];travel=[];turns=[];limits=controller.limits
        for pair in pairs:
            x=state.copy();old=controller.previous.copy();trace=[];distance=turn=0.
            for k in range(n):
                aim=np.r_[pair,np.clip(2.5*(goal[3:]-x[3:]),-.65,.65)]
                u=np.clip(aim,np.maximum(-limits.velocity,old-h*limits.acceleration),np.minimum(limits.velocity,old+h*limits.acceleration))
                u[2:]=np.clip(u[2:],(JOINT_LOWER+.015-x[3:])/h,(JOINT_UPPER-.015-x[3:])/h)
                # Joint and wheel authority are rollout bounds, not a claim
                # that the future motion will be accepted by execution.
                ratio=max(1.,float(np.max(abs(limits.wheels@u))/limits.wheel_speed))
                u[:2]/=ratio
                wheel_delta=limits.wheels@(u-old)
                ratio=max(1.,float(np.max(abs(wheel_delta))/(h*limits.wheel_acceleration)))
                u=old+(u-old)/ratio
                x=advance(x,u,h);trace.append(x.copy());old=u
                distance+=abs(u[0])*h;turn+=abs(u[1])*h
            states.append(trace);travel.append(distance);turns.append(turn)
        states=np.asarray(states);count=len(states)
        points=np.tile(np.stack([f.centers for f in futures]),(count,1,1))
        radii=np.tile(np.stack([f.radii for f in futures]),(count,1))
        clear=(collision.geometry.distances_sequence(states.reshape(-1,9),points,z)-radii[:,:,None]).reshape(count,n,-1).min((1,2))
        self_clear=controller.self_collision.distances_sequence(states.reshape(-1,9)).reshape(count,n,-1).min((1,2))
        margin=np.minimum(clear,self_clear*10.)
        best=float(margin.max());floor=.12 if best>=.12 else best-.015
        eligible=np.flatnonzero(margin>=floor)
        end=states[:,-1]
        cost=np.linalg.norm(end[:,:2]-goal[:2],axis=1)+.12*np.abs([angle(v) for v in end[:,2]-goal[2]])
        cost+=.08*np.asarray(travel)+.025*np.asarray(turns)
        if self.selected is not None:cost+=.025*np.linalg.norm(np.array(pairs)-self.selected,axis=1)
        index=int(eligible[np.argmin(cost[eligible])]);self.selected=np.asarray(pairs[index]);self.updated=t;self.goal=goal.copy()
        self.info=dict(active=True,method='whole_body_velocity_lattice',candidates=count,horizon_s=horizon,
            sampled_forecast_clearance_m=float(clear[index]),sampled_self_clearance_m=float(self_clear[index]),
            selected_base_velocity=self.selected.tolist(),scope='observed-motion numerical preference only; no safety certificate')
        return np.r_[self.selected,target[2:]]
