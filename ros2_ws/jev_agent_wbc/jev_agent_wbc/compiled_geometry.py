"""Optional CPU-compiled implementation of the identical primitive SDFs.

No fast-math, geometry approximation, witness pruning or safety margin changes.
Numba is optional; the portable NumPy implementation remains the default.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def primitive_batch(positions,rotations,points,bodies,offsets,shape_rotations,kinds,sizes,gradient):
    batch,count=points.shape[:2];shapes=len(bodies)
    distances=np.empty((batch,count,shapes));normals=np.empty((batch,count,shapes,3))
    for b in range(batch):
        for k in range(shapes):
            body=bodies[k];origin=positions[b,body]+rotations[b,body]@offsets[k]
            rotation=rotations[b,body]@shape_rotations[k]
            kind=kinds[k];s0,s1,s2=sizes[k]
            for n in range(count):
                dx=points[b,n,0]-origin[0];dy=points[b,n,1]-origin[1];dz=points[b,n,2]-origin[2]
                x=dx*rotation[0,0]+dy*rotation[1,0]+dz*rotation[2,0]
                y=dx*rotation[0,1]+dy*rotation[1,1]+dz*rotation[2,1]
                z=dx*rotation[0,2]+dy*rotation[1,2]+dz*rotation[2,2]
                nx=0.;ny=0.;nz=0.
                if kind==0:
                    length=np.sqrt(x*x+y*y+z*z);distance=length-s0
                    if gradient:
                        length=max(length,1e-12);nx=x/length;ny=y/length;nz=z/length
                elif kind==1:
                    a=abs(x)-s0/2;c=abs(y)-s1/2;d=abs(z)-s2/2
                    aa=max(a,0.);cc=max(c,0.);dd=max(d,0.)
                    length=np.sqrt(aa*aa+cc*cc+dd*dd);distance=length+min(max(a,c,d),0.)
                    if gradient:
                        denom=max(length,1e-12);nx=aa/denom*np.sign(x);ny=cc/denom*np.sign(y);nz=dd/denom*np.sign(z)
                        if length==0.:
                            if a>=c and a>=d:nx=np.sign(x)
                            elif c>=d:ny=np.sign(y)
                            else:nz=np.sign(z)
                elif kind==2:
                    radial=np.sqrt(x*x+y*y);dr=radial-s0;dz=abs(z)-s1/2
                    a=max(dr,0.);c=max(dz,0.);length=np.sqrt(a*a+c*c)
                    distance=length+min(max(dr,dz),0.)
                    if gradient:
                        a/=max(length,1e-12);c/=max(length,1e-12)
                        if length==0.:
                            if dr>=dz:a=1.
                            else:c=1.
                        nx=a*x/max(radial,1e-12);ny=a*y/max(radial,1e-12);nz=c*np.sign(z)
                else:
                    radial=np.sqrt(x*x+y*y);best=np.inf;best_r=0.;best_z=0.;edge_r=0.;edge_z=0.
                    for edge in range(3):
                        if edge==0:er=s0;ez=0.;rr=radial;zz=z+s2/2
                        elif edge==1:er=s1-s0;ez=s2;rr=radial-s0;zz=z+s2/2
                        else:er=-s1;ez=0.;rr=radial-s1;zz=z-s2/2
                        fraction=min(1.,max(0.,(rr*er+zz*ez)/(er*er+ez*ez)))
                        rr-=fraction*er;zz-=fraction*ez;length=np.sqrt(rr*rr+zz*zz)
                        if length<best:best=length;best_r=rr;best_z=zz;edge_r=er;edge_z=ez
                    inside=abs(z)<=s2/2 and radial<=s0+(s1-s0)*(z/s2+.5)
                    sign=-1. if inside else 1.;distance=best*sign
                    if gradient:
                        nr=best_r/max(best,1e-12)*sign;nz=best_z/max(best,1e-12)*sign
                        if best<1e-12:
                            length=np.sqrt(edge_r*edge_r+edge_z*edge_z);nr=edge_z/length;nz=-edge_r/length
                        nx=nr*x/max(radial,1e-12);ny=nr*y/max(radial,1e-12)
                distances[b,n,k]=distance
                if gradient:
                    normals[b,n,k,0]=rotation[0,0]*nx+rotation[0,1]*ny+rotation[0,2]*nz
                    normals[b,n,k,1]=rotation[1,0]*nx+rotation[1,1]*ny+rotation[1,2]*nz
                    normals[b,n,k,2]=rotation[2,0]*nx+rotation[2,1]*ny+rotation[2,2]*nz
    return distances,normals


def primitive_arrays(self,positions,rotations,centers,gradient=False):
    positions,rotations,centers=(np.asarray(value,dtype=np.float64)
                                for value in (positions,rotations,centers))
    lead=np.broadcast_shapes(positions.shape[:-2],rotations.shape[:-3],centers.shape[:-2])
    pp=np.ascontiguousarray(np.broadcast_to(positions,lead+positions.shape[-2:])).reshape(-1,*positions.shape[-2:])
    rr=np.ascontiguousarray(np.broadcast_to(rotations,lead+rotations.shape[-3:])).reshape(-1,*rotations.shape[-3:])
    cc=np.ascontiguousarray(np.broadcast_to(centers,lead+centers.shape[-2:])).reshape(int(np.prod(lead)) if lead else 1,*centers.shape[-2:])
    # Broadcasting may return either a read-only view or a writable copy.
    # Numba specializes on this flag: a different batch shape could otherwise
    # trigger seconds of compilation after the physical loop was armed. Use
    # read-only views consistently without changing the caller's array flags.
    pp,rr,cc=(value.view() for value in (pp,rr,cc))
    for value in (pp,rr,cc):value.flags.writeable=False
    if not hasattr(self,'_compiled_shape_data'):
        kinds=np.array([{'sphere':0,'box':1,'cylinder':2,'frustum':3}[s[2]] for s in self.shapes])
        sizes=np.zeros((len(self.shapes),3))
        for i,s in enumerate(self.shapes):sizes[i,:len(s[3])]=s[3]
        self._compiled_shape_data=kinds,sizes
    kinds,sizes=self._compiled_shape_data
    d,n=primitive_batch(pp,rr,cc,self.shape_bodies,self.shape_offsets,self.shape_rotations,kinds,sizes,gradient)
    d=d.reshape(lead+(centers.shape[-2],len(self.shapes)))
    return (d,n.reshape((*d.shape,3))) if gradient else d


def install():
    from .kc_svsdf import CollisionGeometry
    if CollisionGeometry._primitive_arrays is primitive_arrays:return
    geometry=CollisionGeometry();state=np.array([0.,0.,0.,0.,-.9,2.1,0.,-1.2,0.])
    p,r=geometry.frame_arrays(state[None])
    primitive_arrays(geometry,p,r,np.array([[[3.,2.,1.]]]),True)
    CollisionGeometry._primitive_arrays=primitive_arrays
