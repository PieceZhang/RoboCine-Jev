"""Compiled arithmetic for the acquisition-calibrated private PoE model.

This is the same kinematic/OBB calculation as the readable reference implementation.
It adds no robot policy and makes no safety decision.
"""
import numpy as np
try:
    from numba import njit
except ImportError:
    AVAILABLE=False
else:
    AVAILABLE=True


def _evaluate(base0,state,omega,velocity,homes,chains,lengths,shape_bodies,shape_centers,
              shape_half,camera_body,camera_local,centers,radii,active_distance):
    n=len(state);nj=len(omega);nb=len(homes);ns=len(shape_bodies);nc=len(centers)
    base=base0.copy()
    c,s=np.cos(state[2]),np.sin(state[2])
    yaw=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    base[:3,:3]=yaw@base0[:3,:3]
    base[0,3]+=state[0];base[1,3]+=state[1]
    motions=np.empty((nj,4,4))
    for j in range(nj):
        x,y,z=omega[j];a=state[3+j]
        W=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        square=W@W
        motions[j]=np.eye(4)
        motions[j,:3,:3]=np.eye(3)+np.sin(a)*W+(1-np.cos(a))*square
        motions[j,:3,3]=(a*np.eye(3)+(1-np.cos(a))*W+(a-np.sin(a))*square)@velocity[j]
    spatial_w=np.zeros((nj,3));spatial_v=np.zeros((nj,3))
    poses=np.empty((nb,4,4))
    for bi in range(nb):
        T=base.copy()
        for k in range(lengths[bi]):
            j=chains[bi,k]
            w=T[:3,:3]@omega[j]
            spatial_w[j]=w
            spatial_v[j]=T[:3,:3]@velocity[j]+np.cross(T[:3,3],w)
            T=T@motions[j]
        poses[bi]=T@homes[bi]
    camera=poses[camera_body]@camera_local
    camera_j=np.zeros((6,n))
    camera_j[0,0]=1.;camera_j[1,1]=1.;camera_j[5,2]=1.
    delta=camera[:3,3]-base[:3,3]
    camera_j[0,2]=-delta[1];camera_j[1,2]=delta[0]
    for k in range(lengths[camera_body]):
        j=chains[camera_body,k]
        camera_j[:3,j+3]=np.cross(spatial_w[j],camera[:3,3])+spatial_v[j]
        camera_j[3:,j+3]=spatial_w[j]
    distances=np.empty((nc,ns));gradients=np.zeros((nc,ns,n))
    for si in range(ns):
        bi=shape_bodies[si];R=poses[bi,:3,:3]
        origin=R@shape_centers[si]+poses[bi,:3,3]
        for ci in range(nc):
            local=R.T@(centers[ci]-origin)
            difference=np.abs(local)-shape_half[si]
            outside=np.maximum(difference,0.)
            length=np.sqrt(np.dot(outside,outside))
            distance=length+min(np.max(difference),0.)-radii[ci]
            distances[ci,si]=distance
            if distance>=active_distance:
                continue
            normal_local=np.sign(local)*outside/max(length,1e-12)
            if length<1e-10:
                axis=np.argmax(difference)
                normal_local[axis]=1. if local[axis]>=0. else -1.
            normal=-(R@normal_local)
            closest=origin+R@np.minimum(np.maximum(local,-shape_half[si]),shape_half[si])
            gradients[ci,si,0]=normal[0];gradients[ci,si,1]=normal[1]
            delta=closest-base[:3,3]
            gradients[ci,si,2]=-normal[0]*delta[1]+normal[1]*delta[0]
            for k in range(lengths[bi]):
                j=chains[bi,k]
                gradients[ci,si,j+3]=np.dot(normal,np.cross(spatial_w[j],closest)+spatial_v[j])
    return camera,camera_j,distances,gradients,poses


evaluate=njit(cache=True,nogil=True)(_evaluate) if AVAILABLE else _evaluate


def prewarm():
    """Compile once before the first physical step; no agent-time pause."""
    if AVAILABLE:
        evaluate(np.eye(4),np.zeros(4),np.array([[0.,0.,1.]]),np.zeros((1,3)),
            np.array([np.eye(4)]),np.array([[0]],np.int64),np.array([1],np.int64),
            np.array([0],np.int64),np.zeros((1,3)),np.ones((1,3)),0,np.eye(4),
            np.zeros((1,3)),np.ones(1),.04)
