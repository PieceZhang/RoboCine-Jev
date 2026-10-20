"""Runtime sphere/cuboid geometry, separate from observation-only planning.

Cuboid ray hits are exact. CPU collision audits use a declared conservative
union of balls covering each complete cuboid, including its unobserved volume.
"""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def box_rotation(obstacle):
    return Rotation.from_euler('z',float(obstacle.get('yaw_rad',0.))).as_matrix()


def local_cover(obstacle,maximum_cell_m=.16):
    if obstacle.get('shape','sphere')=='sphere':
        return np.zeros((1,3)),np.array([float(obstacle['radius'])])
    if obstacle['shape']!='box':raise ValueError('unknown runtime obstacle shape')
    size=np.asarray(obstacle['size'],float)
    if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('box size must contain three positive finite dimensions')
    count=np.maximum(1,np.ceil(size/maximum_cell_m).astype(int))
    cell=size/count
    axes=[-s/2+(np.arange(n)+.5)*h for s,n,h in zip(size,count,cell)]
    centers=np.array(list(itertools.product(*axes)))@box_rotation(obstacle).T
    return centers,np.full(len(centers),np.linalg.norm(cell)/2)


def ray_distances(origin,rays,obstacle,maximum=4.5):
    origin=np.asarray(origin);rays=np.asarray(rays)
    if obstacle.get('shape','sphere')=='sphere':
        relative=origin-obstacle['center'];along=rays@relative
        disc=along**2-(relative@relative-obstacle['radius']**2)
        distance=-along-np.sqrt(np.maximum(disc,0.))
        return np.where((disc>=0.)&(distance>0.)&(distance<maximum),distance,np.inf)
    rotation=box_rotation(obstacle)
    start=(origin-obstacle['center'])@rotation;directions=rays@rotation
    half=np.asarray(obstacle['size'])/2
    parallel=np.abs(directions)<1e-12
    with np.errstate(divide='ignore',invalid='ignore'):
        a=(-half-start)/directions;b=(half-start)/directions
    low=np.where(parallel,-np.inf,np.minimum(a,b))
    high=np.where(parallel,np.inf,np.maximum(a,b))
    outside=np.any(parallel&(abs(start)>half),axis=1)
    near=low.max(axis=1);far=high.min(axis=1)
    distance=np.where(near>0.,near,far)
    return np.where(~outside&(far>=near)&(distance>0.)&(distance<maximum),distance,np.inf)
