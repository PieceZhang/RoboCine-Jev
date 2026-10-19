"""Short configuration-space lookahead for a free-camera connector.

Forecast constraints are soft search guidance, never an execution certificate.
The original WBC retains all physical bounds and nonlinear swept checks.
"""
import numpy as np
from .coordination import CoordinationPreview
from .jvs_kinematics import JOINT_LOWER,JOINT_UPPER
from .kc_svsdf import advance
from .tpr_wbc import qp_with_reserve,witnesses
from .transfer_geometry import angle
from .motion_forecast import forecast


class FreeMotionPreview:
    def __init__(self):
        self.updated=-np.inf;self.controls=None;self.times=None;self.info={};self.goal=None

    def preference(self,controller,state,goal,target,t,z):
        collision=controller.collision;moving=np.flatnonzero((collision.track_ids>0)&(np.linalg.norm(collision.velocities,axis=1)>.04))
        if not len(moving):self.controls=None;self.info=dict(active=False);return target
        probes=np.linspace(0.,2.5,11)
        points=np.stack([forecast(collision,float(h)).centers[moving] for h in probes],axis=1)
        d=collision.geometry.distances(state,points.reshape(-1,3),z).reshape(len(moving),len(probes),8)-collision.radii[moving,None,None]
        threat=float(d.min())<.22
        if not threat:self.controls=None;self.info=dict(active=False);return target
        if self.controls is not None and t-self.updated<.3 and np.linalg.norm(goal-self.goal)<.02:
            return np.array([np.interp(t+.3,self.times,self.controls[:,j]) for j in range(8)])
        self.updated=t;self.goal=goal.copy();n=8;h=.3;nv=8*n;limits=controller.limits
        E=[np.pad(np.eye(8),((0,0),(8*k,8*(n-k-1)))) for k in range(n)]
        velocity=controller.previous.copy()
        if self.controls is None:nominal=np.tile(target,(n,1))
        else:nominal=np.array([[np.interp(t+(k+1)*h,self.times,self.controls[:,j]) for j in range(8)] for k in range(n)])
        old=velocity.copy()
        for k in range(n):nominal[k]=np.clip(nominal[k],old-h*limits.acceleration,old+h*limits.acceleration);old=nominal[k]
        goal=goal.copy();goal[2]=state[2]+angle(goal[2]-state[2]);failure=''
        for iteration in range(4):
            H=np.eye(nv)*.04;g=np.zeros(nv);matrices=[];lows=[];highs=[];soft=[];soft_target=[]
            x=state.copy();affine=x.copy();S=np.zeros((9,nv));oldE=np.zeros((8,nv));oldaff=velocity.copy()
            def bound(M,lo,hi):matrices.append(M);lows.append(np.broadcast_to(lo,len(M)));highs.append(np.broadcast_to(hi,len(M)))
            def cost(M,target,weight):
                nonlocal H,g
                H+=weight*M.T@M;g-=weight*M.T@target
            for k,select in enumerate(E):
                bound(select,-limits.velocity,limits.velocity)
                bound(select-oldE,oldaff-h*limits.acceleration,oldaff+h*limits.acceleration)
                bound(limits.wheels@select,-limits.wheel_speed,limits.wheel_speed)
                bound(limits.wheels@(select-oldE),limits.wheels@oldaff-h*limits.wheel_acceleration,limits.wheels@oldaff+h*limits.wheel_acceleration)
                cost(select,target,.12);cost(select-oldE,oldaff,.15)
                old_nominal=velocity if k==0 else nominal[k-1]
                # The endpoint velocity cannot act for the entire knot:
                # acceleration bounds require a ramp from the previous one.
                # Assuming an instantaneous change overestimated early escape
                # and underestimated braking displacement near a second mover.
                xn,A,B=CoordinationPreview.step(x,old_nominal,nominal[k],h)
                affine=A@affine+xn-A@x+B@(oldaff-old_nominal-nominal[k])
                S=A@S+B@(oldE+select);x=xn
                weights=np.diag(np.r_[2.,2.,.35,np.full(6,.18)])
                cost(weights@S,weights@(goal-affine),h*(3 if k==n-1 else 1))
                bound(S[3:],JOINT_LOWER+.02-affine[3:],JOINT_UPPER-.02-affine[3:])
                future=forecast(collision,(k+1)*h)
                _,_,_,dist,grad=witnesses(future,x,z)
                for body in range(8):
                    mask=future.geometry.shape_bodies==body
                    subset=np.where(mask[None],dist,np.inf)
                    i,j=np.unravel_index(subset.argmin(),subset.shape)
                    if subset[i,j]<.35:
                        normal=grad[i,j];soft.append(normal@S);soft_target.append(.14-dist[i,j]-normal@(affine-x))
                sd,sg=controller.self_collision.terms(x);mask=sd<.06
                bound(sg[mask]@S,controller.self_collision.margin+.003-sd[mask]-sg[mask]@(affine-x),np.inf)
                oldE=select;oldaff=np.zeros(8)
            try:
                answer,_=qp_with_reserve(H,g,np.vstack(matrices),np.concatenate(lows),np.concatenate(highs),nominal.ravel(),
                    np.asarray(soft).reshape(-1,nv),np.asarray(soft_target),300.)
            except RuntimeError as error:failure=str(error);break
            changed=np.max(abs(answer-nominal.ravel()));nominal=answer.reshape(n,8)
            if changed<.005:break
        if failure:
            self.controls=None;self.info=dict(active=False,failure=failure);return target
        self.times=np.r_[t,t+np.arange(1,n+1)*h];self.controls=np.vstack([velocity,nominal])
        x=state.copy();clearance=np.inf;old=velocity.copy()
        for k,u in enumerate(nominal):
            x=advance(x,.5*(old+u),h);old=u
            clearance=min(clearance,float(forecast(collision,(k+1)*h).clearance(x,z).min()))
        self.info=dict(active=True,horizon_s=n*h,sampled_forecast_clearance_m=clearance,
            scope='soft configuration-space numerical preference; not a safety certificate; original execution checks mandatory')
        return nominal[0].copy()
