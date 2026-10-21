"""Bounded native kinematic search for holonomic platform models.

This optimizer supplies a candidate witness, never a dynamics certificate.
All camera, original joint/rate, nonholonomic and observed-clearance acceptance
checks remain in multi_robot_planning.lift. At most 14 residual calls per step.
"""
import numpy as np
from numba import njit
from .native_geometry_kernel import evaluate

@njit(cache=True,nogil=True)
def log_rotation(R):
    angle=np.arccos(min(1.,max(-1.,(np.trace(R)-1.)/2.)))
    v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    if angle<1e-5:return .5*v*(1.+angle*angle/6.)
    if np.pi-angle<1e-5:
        a=np.argmax(np.diag(R));axis=np.zeros(3);axis[a]=np.sqrt(max(0.,(R[a,a]+1.)/2.))
        for i in range(3):
            if i!=a:axis[i]=(R[i,a]+R[a,i])/(4.*max(axis[a],1e-12))
        return axis*angle
    return v*angle/(2.*np.sin(angle))

@njit(cache=True,nogil=True)
def residual(x,previous,g,target_p,target_r,base_pref,centers,radii,pi,pj,limits,omni,lateral_gain,yaw0,dt):
    camera,J,distances,D,_=evaluate(g[0],x,g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8],g[9],g[10],centers,radii,.04)
    n=len(x);posture=8;regularization=posture+n-3;platform=regularization+n
    nonholonomic=platform+3;collision=nonholonomic+int(not omni)
    e=np.zeros(collision+len(pi));jac=np.zeros((len(e),n))
    angle=log_rotation(target_r@camera[:3,:3].T)
    ax,ay,az=angle;W=np.array([[0.,-az,ay],[az,0.,-ax],[-ay,ax,0.]])
    theta=np.sqrt(np.dot(angle,angle))
    coefficient=1./12. if theta<1e-5 else (1./theta**2-(1.+np.cos(theta))/(2.*theta*np.sin(theta)))
    log_jac=np.eye(3)+.5*W+coefficient*(W@W)
    e[:3]=60.*(camera[:3,3]-target_p);e[3:6]=16.*angle
    jac[:3]=60.*J[:3];jac[3:6]=-16.*(log_jac@J[3:])
    e[6:8]=.15*(x[:2]-base_pref)
    jac[6,0]=.15;jac[7,1]=.15
    for j in range(n-3):e[posture+j]=.025*x[3+j];jac[posture+j,3+j]=.025
    for j in range(n):e[regularization+j]=.01*(x[j]-previous[j]);jac[regularization+j,j]=.01
    heading=yaw0+(x[2]+previous[2])/2.
    dx=x[0]-previous[0];dy=x[1]-previous[1];dz=x[2]-previous[2]
    c=np.cos(heading);s=np.sin(heading);forward=dx*c+dy*s;lateral=dy*c-dx*s
    rates=np.array([forward,lateral,dz])/dt
    body_jac=np.array([[c,s,.5*lateral],[-s,c,-.5*forward],[0.,0.,1.]])/dt
    for k in range(3):
        excess=max(abs(rates[k])-.95*limits[k],0.);e[platform+k]=100.*excess
        if excess>0.:jac[platform+k,:3]=100.*np.sign(rates[k])*body_jac[k]
    if not omni:
        e[nonholonomic]=150.*(dy*c-dx*s-lateral_gain*dz)
        jac[nonholonomic,0]=-150.*s;jac[nonholonomic,1]=150.*c
        jac[nonholonomic,2]=150.*(-.5*(dy*s+dx*c)-lateral_gain)
    for k in range(len(pi)):
        gap=distances[pi[k],pj[k]]
        e[collision+k]=120.*max(.04-gap,0.)
        if gap<.04:jac[collision+k]=-120.*D[pi[k],pj[k]]
    return e,jac

@njit(cache=True,nogil=True)
def solve(initial,previous,lo,hi,g,target_p,target_r,base_pref,centers,radii,pi,pj,limits,omni,lateral_gain,yaw0,dt,maxeval=14):
    x=initial.copy();n=len(x)
    e,J=residual(x,previous,g,target_p,target_r,base_pref,centers,radii,pi,pj,limits,omni,lateral_gain,yaw0,dt)
    cost=np.dot(e,e);calls=1;damping=1e-5
    while calls<maxeval:
        gradient=J.T@e;H=J.T@J
        free=np.ones(n,np.bool_);largest=0.
        for j in range(n):
            if (x[j]<=lo[j]+1e-9 and gradient[j]>0.) or (x[j]>=hi[j]-1e-9 and gradient[j]<0.):free[j]=False
            if free[j]:largest=max(largest,abs(gradient[j]))
        if largest<1e-5:break
        d=np.zeros(n);A=H.copy()
        for j in range(n):A[j,j]+=damping*max(H[j,j],1e-3)
        for k in range(n+1):
            M=np.zeros((n,n));b=np.zeros(n)
            for i in range(n):
                if free[i]:
                    b[i]=-gradient[i]
                    for j in range(n):
                        if free[j]:M[i,j]=A[i,j]
                        else:b[i]-=A[i,j]*d[j]
                else:M[i,i]=1.;b[i]=d[i]
            d=np.linalg.solve(M,b);outside=False
            for j in range(n):
                if free[j] and (d[j]<lo[j]-x[j] or d[j]>hi[j]-x[j]):
                    d[j]=min(hi[j]-x[j],max(lo[j]-x[j],d[j]));free[j]=False;outside=True
            if not outside:break
        if np.linalg.norm(d)<1e-5*(1e-5+np.linalg.norm(x)):break
        accepted=False;change=0.
        for alpha in (1.,.5,.25,.125):
            trial=np.minimum(hi,np.maximum(lo,x+alpha*d))
            te,tJ=residual(trial,previous,g,target_p,target_r,base_pref,centers,radii,pi,pj,limits,omni,lateral_gain,yaw0,dt)
            calls+=1;score=np.dot(te,te)
            if score<cost:
                change=cost-score;x,e,J,cost=trial,te,tJ,score;accepted=True;damping=max(1e-8,damping*.3);break
            if calls>=maxeval:break
        if not accepted:
            damping*=20.
            if damping>1e4:break
        elif change<1e-5*max(cost,1e-10):break
    return x,calls


def geometry_arguments(model):
    return (model.base,model.omega,model.velocity,model._homes,model._chains,model._chain_lengths,
            model._shape_bodies,model.shape_centers,model.shape_half,model._camera_body,model.camera_local)


def prewarm():
    home=np.eye(4);home[0,3]=1.
    g=(np.eye(4),np.array([[0.,0.,1.]]),np.zeros((1,3)),np.array([home]),
       np.array([[0]],np.int64),np.array([1],np.int64),np.array([0],np.int64),
       np.zeros((1,3)),np.full((1,3),.1),0,np.eye(4))
    solve(np.zeros(4),np.zeros(4),np.full(4,-.1),np.full(4,.1),g,
          np.array([1.,0.,0.]),np.eye(3),np.zeros(2),np.array([[1.,.3,0.]]),
          np.array([.1]),np.array([0],np.int64),np.array([0],np.int64),np.full(3,.2),True,0.,0.,.1)
