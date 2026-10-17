"""Conservative enclosing capsules for non-adjacent URDF collision primitives.

Cylinders use axial capsules; small boxes/frusta use enclosing spheres. Adjacent
links and the neck/link2 mounting overlap are explicitly excluded. Distances are
lower bounds for the original shapes, not a mesh-contact oracle.
"""
import numpy as np
from .kc_svsdf import CollisionGeometry


class SelfCollision:
    margin = .004

    def __init__(self):
        self.geometry = CollisionGeometry()
        shapes = self.geometry.shapes
        self.pairs = [(i, j) for i, a in enumerate(shapes) for j, b in enumerate(shapes)
                      if i < j and abs(a[0]-b[0]) > 1
                      and not (a[1] == 'neck_base_link' and b[0] == 2)]
        self.left, self.right = np.array(self.pairs).T
        self._bodies=np.array([s[0] for s in shapes])
        self._offsets=np.array([s[4] for s in shapes])
        self._axes=np.array([s[5][:,2]*s[3][1]/2 if s[2]=='cylinder' else np.zeros(3) for s in shapes])
        self._radii=np.array([s[3][0] if s[2] in ('sphere','cylinder') else
            np.linalg.norm(s[3])/2 if s[2]=='box' else np.hypot(max(s[3][:2]),s[3][2]/2) for s in shapes])

    def capsules(self, state, frames=None):
        frames = self.geometry.frames(state)[0] if frames is None else frames
        positions=np.array([p for p,_ in frames])[self._bodies]
        rotations=np.array([r for _,r in frames])[self._bodies]
        centers=positions+np.einsum('nij,nj->ni',rotations,self._offsets)
        axes=np.einsum('nij,nj->ni',rotations,self._axes)
        return centers-axes,centers+axes,self._radii

    def distances(self, state, with_witnesses=False, frames=None):
        a, b, radii = self.capsules(state, frames)
        return self._capsule_distances(a,b,radii,with_witnesses)

    def distances_sequence(self,states):
        positions,rotations=self.geometry.frame_arrays(states)
        positions=positions[:,self._bodies];rotations=rotations[:,self._bodies]
        centers=positions+np.einsum('npij,pj->npi',rotations,self._offsets)
        axes=np.einsum('npij,pj->npi',rotations,self._axes)
        return self._capsule_distances(centers-axes,centers+axes,self._radii,False)

    def _capsule_distances(self,a,b,radii,with_witnesses):
        p, q = a[...,self.left,:], a[...,self.right,:]
        u, v = b[...,self.left,:]-p, b[...,self.right,:]-q
        uu, vv, uv = np.sum(u*u, axis=-1), np.sum(v*v, axis=-1), np.sum(u*v, axis=-1)
        candidates, points_a, points_b = [], [], []
        for point in (p, p+u):
            alpha = np.clip(np.sum((point-q)*v, axis=-1)/np.maximum(vv, 1e-20), 0., 1.)
            candidates.append(np.linalg.norm(point-q-alpha[...,None]*v, axis=-1))
            points_a.append(point);points_b.append(q+alpha[...,None]*v)
        for point in (q, q+v):
            alpha = np.clip(np.sum((point-p)*u, axis=-1)/np.maximum(uu, 1e-20), 0., 1.)
            candidates.append(np.linalg.norm(point-p-alpha[...,None]*u, axis=-1))
            points_a.append(p+alpha[...,None]*u);points_b.append(point)
        w = p-q
        uw, vw = np.sum(u*w, axis=-1), np.sum(v*w, axis=-1)
        determinant = uu*vv-uv*uv
        denom = np.maximum(determinant, 1e-20)
        s, t = (uv*vw-vv*uw)/denom, (uu*vw-uv*uw)/denom
        valid = (determinant > 1e-15) & (s >= 0.) & (s <= 1.) & (t >= 0.) & (t <= 1.)
        interior = np.linalg.norm(w+s[...,None]*u-t[...,None]*v, axis=-1)
        candidates.append(np.where(valid, interior, np.inf))
        points_a.append(p+s[...,None]*u);points_b.append(q+t[...,None]*v)
        distance=np.min(candidates, axis=0)-radii[self.left]-radii[self.right]
        if not with_witnesses:
            return distance
        closest=np.argmin(candidates,axis=0);index=np.arange(len(closest))
        return distance,np.array(points_a)[closest,index],np.array(points_b)[closest,index]

    def terms(self, state):
        frames,origins,axes=self.geometry.frames(state)
        distance,pa,pb = self.distances(state,True,frames)
        delta=pa-pb
        normal=delta/np.maximum(np.linalg.norm(delta,axis=1)[:,None],1e-12)
        jacobians=[]
        for points,indices in [(pa,self.left),(pb,self.right)]:
            J=self.geometry.point_jacobians(state,points,6,origins,axes)
            J[:,:,3:]*=self.geometry.shape_joint_mask[indices,None,:]
            jacobians.append(J)
        gradient=np.einsum('ni,nij->nj',normal,jacobians[0]-jacobians[1])
        # The exact common rigid-body motion cancels, including base yaw.
        gradient[:,:3]=0.
        return distance, gradient

    def sweep(self, state, old, new, dt, max_samples=33):
        from .actuation import advance_command
        # Relative speed of two points, with the same conservative chain reach.
        speed = 2*np.maximum(np.abs(old[2:]), np.abs(new[2:]))@self.geometry.reach[1:]
        values={}
        count=3
        while True:
            missing=[s for s in np.linspace(0.,1.,count) if s not in values]
            states=[advance_command(state,old,old+s*(new-old),s*dt,True) for s in missing]
            distances=self.distances_sequence(states).min(axis=1)
            values.update(zip(missing,map(float,distances)))
            lower=min(values.values())-speed*dt/(2*(count-1))
            if lower>=self.margin or min(values.values())<self.margin or count>=max_samples:
                return float(lower)
            # Refine only this ambiguous interval, retaining sampled values.
            count=min(max_samples,2*count-1)
