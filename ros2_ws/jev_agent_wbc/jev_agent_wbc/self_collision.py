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

    def capsules(self, state, frames=None):
        frames = self.geometry.frames(state)[0] if frames is None else frames
        starts, ends, radii = [], [], []
        for body, _, kind, size, offset, orientation in self.geometry.shapes:
            p, r = frames[body]
            center = p+r@offset
            if kind == 'cylinder':
                axis = r@orientation[:, 2]*size[1]/2
                radius = size[0]
            else:
                axis = np.zeros(3)
                radius = (size[0] if kind == 'sphere' else np.linalg.norm(size)/2 if kind == 'box'
                          else np.hypot(max(size[:2]), size[2]/2))
            starts.append(center-axis); ends.append(center+axis); radii.append(radius)
        return np.array(starts), np.array(ends), np.array(radii)

    def distances(self, state, with_witnesses=False, frames=None):
        a, b, radii = self.capsules(state, frames)
        p, q = a[self.left], a[self.right]
        u, v = b[self.left]-p, b[self.right]-q
        uu, vv, uv = np.sum(u*u, axis=1), np.sum(v*v, axis=1), np.sum(u*v, axis=1)
        candidates, points_a, points_b = [], [], []
        for point in (p, p+u):
            alpha = np.clip(np.sum((point-q)*v, axis=1)/np.maximum(vv, 1e-20), 0., 1.)
            candidates.append(np.linalg.norm(point-q-alpha[:, None]*v, axis=1))
            points_a.append(point);points_b.append(q+alpha[:, None]*v)
        for point in (q, q+v):
            alpha = np.clip(np.sum((point-p)*u, axis=1)/np.maximum(uu, 1e-20), 0., 1.)
            candidates.append(np.linalg.norm(point-p-alpha[:, None]*u, axis=1))
            points_a.append(p+alpha[:, None]*u);points_b.append(point)
        w = p-q
        uw, vw = np.sum(u*w, axis=1), np.sum(v*w, axis=1)
        determinant = uu*vv-uv*uv
        denom = np.maximum(determinant, 1e-20)
        s, t = (uv*vw-vv*uw)/denom, (uu*vw-uv*uw)/denom
        valid = (determinant > 1e-15) & (s >= 0.) & (s <= 1.) & (t >= 0.) & (t <= 1.)
        interior = np.linalg.norm(w+s[:, None]*u-t[:, None]*v, axis=1)
        candidates.append(np.where(valid, interior, np.inf))
        points_a.append(p+s[:,None]*u);points_b.append(q+t[:,None]*v)
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
        bodies=np.array([s[0] for s in self.geometry.shapes])
        for points,indices in [(pa,self.left),(pb,self.right)]:
            J=np.zeros((len(distance),3,9))
            for body in np.unique(bodies[indices]):
                mask=bodies[indices]==body
                J[mask]=self.geometry.point_jacobians(state,points[mask],int(body),origins,axes)
            jacobians.append(J)
        gradient=np.einsum('ni,nij->nj',normal,jacobians[0]-jacobians[1])
        # The exact common rigid-body motion cancels, including base yaw.
        gradient[:,:3]=0.
        return distance, gradient

    def sweep(self, state, old, new, dt):
        from .actuation import advance_command
        values = [self.distances(advance_command(state, old, old+s*(new-old), s*dt, True)).min()
                  for s in np.linspace(0., 1., 5)]
        # Relative speed of two points, with the same conservative chain reach.
        speed = 2*np.maximum(np.abs(old[2:]), np.abs(new[2:]))@self.geometry.reach[1:]
        return float(min(values)-speed*dt/8)
