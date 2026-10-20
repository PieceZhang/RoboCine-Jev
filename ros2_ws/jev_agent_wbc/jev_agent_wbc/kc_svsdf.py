"""Primitive-exact JVS geometry, swept SDF and chain-filtered interior SVSDF.

Implements the metric/ball-shrinking construction in arXiv:2608.07005, (6–9).
Environment input is either validated static primitives represented by
conservative sphere unions, or a cached ground-truth USD triangle mesh with
conservative surface voxels.
The chassis uses a circumscribed smooth frustum; passive caster wheels use
rotation-invariant enclosing spheres. All other URDF collision shapes are exact.
Ground contact is intentionally excluded from obstacle clearance.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .jvs_kinematics import JOINT_ORIGINS, JOINT_AXES, CAMERA_OFFSET, _rotation, arm_chain


def advance(x, u, dt):
    """Exact constant-control nonholonomic flow, including the zero-yaw limit."""
    y = np.array(x, dtype=float, copy=True)
    a = u[1]*dt
    y[:2] += u[0]*dt*np.sinc(a/(2*np.pi))*np.array([np.cos(x[2]+a/2), np.sin(x[2]+a/2)])
    y[2:] += dt*u[1:]
    return y


def advance_sequence(x,u,times):
    times=np.asarray(times)
    states=np.broadcast_to(x,(len(times),9)).copy()
    a=u[1]*times
    distance=u[0]*times*np.sinc(a/(2*np.pi))
    states[:,0]+=distance*np.cos(x[2]+a/2)
    states[:,1]+=distance*np.sin(x[2]+a/2)
    states[:,2:]+=times[:,None]*u[1:]
    return states


def obstacle_spheres(obstacle):
    """Return conservative sphere proxies for one rendered obstacle."""
    center = np.asarray(obstacle['center'], float)
    kind = obstacle.get('type')
    if kind == 'sphere':
        return [(center, float(obstacle['radius']))]
    if kind == 'cylinder':
        radius, height = float(obstacle['radius']), float(obstacle['height'])
        count = max(1, int(np.ceil(height/(2*radius))))
        segment = height/count
        proxy_radius = float(np.hypot(radius, segment/2))
        bottom = center[2]-height/2
        return [(np.array([center[0], center[1], bottom+(i+.5)*segment]), proxy_radius)
                for i in range(count)]
    if kind == 'cube':
        size = float(obstacle['size'])
        offset = size/4
        radius = np.sqrt(3)*offset
        return [(center+np.array([dx,dy,dz]), radius)
                for dx in (-offset,offset) for dy in (-offset,offset)
                for dz in (-offset,offset)]
    if kind == 'triangular_pyramid':
        size = np.asarray(obstacle['size'], float)
        # The center is the bounding-box center. This sphere contains the base
        # triangle and apex for every yaw angle.
        return [(center, float(np.linalg.norm(size)/2))]
    raise ValueError(f"unsupported obstacle type: {kind!r}")


def validate_scene(data):
    if data.get('format') != 'jvs_obstacles/v1' or not data.get('obstacles'):
        raise ValueError('expected nonempty jvs_obstacles/v1 scene')
    margin = float(data.get('minimum_clearance_m', .025))
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('clearance must be positive')
    names = set()
    for obstacle in data['obstacles']:
        name = obstacle.get('name')
        kind = obstacle.get('type')
        center = np.asarray(obstacle.get('center', []), float)
        if not name or name in names:
            raise ValueError('obstacles require unique nonempty names')
        names.add(name)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError(f'invalid center for obstacle {name!r}')
        if kind == 'sphere':
            dimensions = [obstacle.get('radius')]
        elif kind == 'cylinder':
            dimensions = [obstacle.get('radius'), obstacle.get('height')]
        elif kind == 'cube':
            dimensions = [obstacle.get('size')]
        elif kind == 'triangular_pyramid':
            dimensions = obstacle.get('size', [])
            if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 3:
                raise ValueError(f'triangular_pyramid {name!r} requires size [x, y, z]')
        else:
            raise ValueError(f'unsupported obstacle type {kind!r}')
        try:
            dimensions = np.asarray(dimensions, float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'invalid dimensions for obstacle {name!r}') from error
        if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
            raise ValueError(f'invalid dimensions for obstacle {name!r}')
        yaw = float(obstacle.get('yaw_deg', 0.))
        if not np.isfinite(yaw):
            raise ValueError(f'invalid yaw for obstacle {name!r}')
        obstacle_spheres(obstacle)
    return data


def load_scene(path):
    path = Path(path).expanduser().resolve()
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            try:
                metadata = json.loads(str(archive['metadata']))
                centers = np.asarray(archive['surfel_centers'], float)
                radius = float(archive['surfel_radius'])
                clearance = float(archive['minimum_clearance'])
                vertices = np.asarray(archive['vertices'])
                faces = np.asarray(archive['faces'])
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f'invalid ground-truth mesh cache: {error}') from error
        if (metadata.get('format') != 'jvs_groundtruth_mesh/v1'
                or centers.ndim != 2 or centers.shape[1] != 3 or not len(centers)
                or not np.all(np.isfinite(centers)) or not np.isfinite(radius) or radius <= 0
                or vertices.ndim != 2 or vertices.shape[1] != 3
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)):
            raise ValueError('invalid jvs_groundtruth_mesh/v1 geometry')
        if not np.isfinite(clearance) or clearance <= 0:
            raise ValueError('clearance must be positive')
        return dict(format='jvs_groundtruth_mesh/v1', mesh_file=str(path), metadata=metadata,
                    minimum_clearance_m=clearance)
    return validate_scene(json.loads(path.read_text()))


def primitive_sdf(points, kind, size, with_gradient=False):
    """Exact distance and optional analytic outward normal/subgradient.

    Finite differences require six extra distance queries for every witness.
    The analytic normal has the same geometry and a bounded subgradient at
    axes, corners and nearest-feature switches; no safety margin is changed.
    """
    p = np.atleast_2d(points)
    if kind == 'sphere':
        length = np.linalg.norm(p, axis=1)
        distance = length-size[0]
        return (distance, p/np.maximum(length[:, None], 1e-12)) if with_gradient else distance
    if kind == 'box':
        d = np.abs(p)-np.asarray(size)/2
        outside = np.maximum(d, 0)
        length = np.linalg.norm(outside, axis=1)
        distance = length+np.minimum(np.max(d, axis=1), 0)
        if not with_gradient:
            return distance
        normal = outside/np.maximum(length[:, None], 1e-12)*np.sign(p)
        inside = length == 0
        indices = np.flatnonzero(inside)
        face = np.argmax(d[inside], axis=1)
        normal[indices, face] = np.sign(p[indices, face])
        return distance, normal
    if kind == 'cylinder':
        radial = np.linalg.norm(p[:, :2], axis=1)
        d = np.c_[radial-size[0], np.abs(p[:, 2])-size[1]/2]
        outside = np.maximum(d, 0)
        length = np.linalg.norm(outside, axis=1)
        distance = length+np.minimum(np.max(d, axis=1), 0)
        if not with_gradient:
            return distance
        meridian = outside/np.maximum(length[:, None], 1e-12)
        indices = np.flatnonzero(length == 0)
        meridian[indices, np.argmax(d[indices], axis=1)] = 1.
        normal = np.c_[meridian[:, :1]*p[:, :2]/np.maximum(radial[:, None], 1e-12),
                       meridian[:, 1]*np.sign(p[:, 2])]
        return distance, normal
    if kind == 'frustum':
        # Distance to the three exposed edges of the meridian trapezoid.
        r0, r1, height = size
        q = np.c_[np.linalg.norm(p[:, :2], axis=1), p[:, 2]]
        vertices = np.array([[0, -height/2], [r0, -height/2], [r1, height/2], [0, height/2]])
        edges=np.diff(vertices,axis=0)
        relative=q[:,None]-vertices[:-1]
        s=np.clip(np.einsum('nei,ei->ne',relative,edges)/np.sum(edges*edges,axis=1),0,1)
        offsets=relative-s[...,None]*edges
        distances=np.linalg.norm(offsets,axis=-1)
        inside = (np.abs(q[:, 1]) <= height/2) & (q[:, 0] <= r0+(r1-r0)*(q[:, 1]/height+.5))
        distance = np.min(distances, axis=1)*np.where(inside, -1., 1.)
        if not with_gradient:
            return distance
        edge = np.argmin(distances, axis=1)
        delta = offsets[np.arange(len(p)),edge]
        meridian = delta/np.maximum(np.abs(distance[:, None]), 1e-12)*np.where(inside[:, None], -1., 1.)
        surface = np.abs(distance) < 1e-12
        edge_normals=np.column_stack([edges[:,1],-edges[:,0]])/np.linalg.norm(edges,axis=1)[:,None]
        meridian[surface] = edge_normals[edge[surface]]
        normal = np.c_[meridian[:, :1]*p[:, :2]/np.maximum(q[:, :1], 1e-12), meridian[:, 1]]
        return distance, normal
    raise ValueError(kind)


class CollisionGeometry:
    def __init__(self):
        self.shapes = []
        def add(body, name, kind, size, p, rpy=(0, 0, 0)):
            self.shapes.append((body, name, kind, np.asarray(size), np.asarray(p), Rotation.from_euler('xyz', rpy).as_matrix()))
        add(0, 'base_link', 'frustum', [.18, .13, .12], [0, 0, .13])
        for sign in (-1, 1):
            add(0, 'wheel', 'cylinder', [.07, .04], [0, sign*.105, .07], [np.pi/2, 0, 0])
        for x, y in ((.18, 0), (-.131, .089), (-.131, -.089)):
            add(0, 'caster_post', 'cylinder', [.022, .06], [x, y, .065])
            add(0, 'caster_wheel', 'sphere', [np.hypot(.02, .009)], [x, y, .02])
        add(0, 'arm_pedestal_link', 'box', [.08, .08, .076], [.098, 0, .228])
        add(0, 'rgbd_mount_link', 'box', [.072, .040, .026], [.156, 0, .125])
        add(0, 'rgbd_link', 'box', [.055, .09, .04], [.207, 0, .125], [0, -.118464, 0])
        add(0, 'neck_base_link', 'cylinder', [.04, .07], [.098, 0, .301])
        arm = [(.027, .10, [0,0,.05], [0,0,0]),
               (.025, .205218, [0,.00975,.102145], [0,.0952,np.pi/2]),
               (.024, .188, [-.041,-.010,.084], [0,.465,-2.902]),
               (.023, .094340, [0,.017,.044], [0,.369,np.pi/2]),
               (.021, .084528, [.034,-.0175,.018], [0,1.130847,-.475343]),
               (.020, .122540, [.019,0,.05825], [0,.315,0])]
        for i, (r,l,p,a) in enumerate(arm):
            add(i+1, f'link{i+1}', 'cylinder', [r,l], p,a)
        add(7, 'camera_link', 'box', [.06,.08,.05], [0,0,0])
        # Conservative distance from each rotation axis to any downstream point.
        self.reach = np.array([1.25, .98, .88, .66, .46, .36, .22])
        self.by_body = [[shape for shape in self.shapes if shape[0] == b] for b in range(8)]
        self.shape_bodies=np.array([s[0] for s in self.shapes])
        self.shape_offsets=np.array([s[4] for s in self.shapes])
        self.shape_rotations=np.array([s[5] for s in self.shapes])
        self.shape_groups={kind:np.array([i for i,s in enumerate(self.shapes) if s[2]==kind])
                           for kind in ('sphere','box','cylinder','frustum')}
        self.shape_sizes={kind:np.array([self.shapes[i][3] for i in ids]) for kind,ids in self.shape_groups.items()}
        self.axis_outer=np.array([np.outer(axis,axis) for axis in JOINT_AXES])
        self.axis_cross=np.array([[[0.,-z,y],[z,0.,-x],[-y,x,0.]] for x,y,z in JOINT_AXES])
        self.shape_joint_mask=np.arange(6)[None,:]<np.minimum(self.shape_bodies,6)[:,None]
        extents=[]
        for _,_,kind,size,offset,_ in self.shapes:
            radius=(size[0] if kind=='sphere' else np.linalg.norm(size)/2 if kind=='box'
                    else np.hypot(size[0],size[1]/2) if kind=='cylinder'
                    else np.hypot(max(size[:2]),size[2]/2))
            extents.append(np.linalg.norm(offset)+radius)
        self.max_body_extent=float(max(extents))

    def frames(self, x, z=0., base_rotation=None):
        p = np.array([x[0], x[1], z])
        r = Rotation.from_euler('z', x[2]).as_matrix() if base_rotation is None else base_rotation
        positions,rotations,origins,axes,_=arm_chain(x[3:])
        return list(zip(positions@r.T+p,r@rotations)),origins@r.T+p,axes@r.T

    def frame_arrays(self,states,z=0.,base_rotations=None):
        states=np.asarray(states);n=len(states)
        p=np.column_stack([states[:,:2],np.full(n,z)])
        r=Rotation.from_euler('z',states[:,2]).as_matrix() if base_rotations is None else np.asarray(base_rotations)
        positions=np.empty((n,8,3));orientations=np.empty((n,8,3,3))
        positions[:,0]=p;orientations[:,0]=r
        for j,(offset,axis) in enumerate(zip(JOINT_ORIGINS,JOINT_AXES)):
            p=p+np.einsum('nij,j->ni',r,offset)
            c=np.cos(states[:,3+j]);s=np.sin(states[:,3+j])
            rotation=c[:,None,None]*np.eye(3)+(1-c[:,None,None])*self.axis_outer[j]+s[:,None,None]*self.axis_cross[j]
            r=r@rotation
            positions[:,j+1]=p;orientations[:,j+1]=r
        positions[:,-1]=p+np.einsum('nij,j->ni',r,CAMERA_OFFSET)
        orientations[:,-1]=r
        return positions,orientations

    def body_sdf(self, points, frames, body):
        points = np.atleast_2d(points)
        bp, br = frames[body]
        ds = []
        for b, _, kind, size, p, r in self.by_body[body]:
            ds.append(primitive_sdf((points-(bp+br@p))@(br@r), kind, size))
        return np.min(ds, axis=0)

    def point_jacobian(self, x, point, body, origins, axes, z=0.):
        j = np.zeros((3, 9)); j[:2, :2] = np.eye(2)
        j[:, 2] = np.cross([0,0,1], point-np.r_[x[:2],z])
        for k in range(min(body, 6)):
            j[:, k+3] = np.cross(axes[k], point-origins[k])
        return j

    def point_jacobians(self, x, points, body, origins, axes, z=0.):
        j = np.zeros((len(points), 3, 9))
        j[:, :2, :2] = np.eye(2)
        j[:, 0, 2] = -(points[:, 1]-x[1])
        j[:, 1, 2] = points[:, 0]-x[0]
        count = min(body, 6)
        if count:
            j[:, :, 3:3+count] = np.cross(axes[None, :count],
                points[:, None, :]-origins[None, :count]).transpose(0, 2, 1)
        return j

    def distances(self, x, centers, z=0., base_rotation=None, gradient=False):
        frames, origins, axes = self.frames(x,z,base_rotation)
        result=self.primitive_arrays(frames,centers,gradient)
        if not gradient:
            return np.stack([result[:,self.shape_bodies==b].min(axis=1) for b in range(8)],axis=1)
        primitive_distances,primitive_normals=result
        distances, gradients = [], []
        for b in range(8):
            selected=np.flatnonzero(self.shape_bodies==b)
            nearest=selected[np.argmin(primitive_distances[:,selected],axis=1)]
            indices = np.arange(len(centers))
            distances.append(primitive_distances[indices,nearest])
            normal=primitive_normals[indices,nearest]
            gradients.append(-np.einsum('ni,nij->nj', normal,
                self.point_jacobians(x,centers,b,origins,axes,z)))
        return np.array(distances).T, np.array(gradients).transpose(1,0,2)

    def primitive_arrays(self, frames, centers, gradient=False):
        """Batch identical analytic primitives; preserve their exact SDFs."""
        positions=np.array([p for p,_ in frames]);rotations=np.array([r for _,r in frames])
        return self._primitive_arrays(positions,rotations,np.asarray(centers),gradient)

    def _primitive_arrays(self, positions, rotations, centers, gradient=False):
        br=rotations[...,self.shape_bodies,:,:]
        origins=positions[...,self.shape_bodies,:]+np.einsum('...pij,pj->...pi',br,self.shape_offsets)
        rotation=br@self.shape_rotations
        delta=centers[...,None,:]-origins[...,None,:,:]
        local=np.matmul(delta[...,None,:],rotation[...,None,:,:,:])[...,0,:]
        distances=np.empty(local.shape[:-1]);normals=np.empty_like(local) if gradient else None
        for kind,ids in self.shape_groups.items():
            points=local[...,ids,:];shape=points.shape
            sizes=self.shape_sizes[kind]
            if kind=='frustum':
                # The chassis has one frustum; its dimensions are common.
                size=sizes[0]
            else:
                expanded=np.broadcast_to(sizes,shape[:-1]+(sizes.shape[1],)).reshape(-1,sizes.shape[1])
                size=expanded if kind=='box' else expanded.T
            result=primitive_sdf(points.reshape(-1,3),kind,size,with_gradient=gradient)
            if gradient:
                distance,normal=result
                normals[...,ids,:]=normal.reshape(shape)
            else:distance=result
            distances[...,ids]=distance.reshape(shape[:-1])
        if gradient:
            return distances,np.matmul(rotation[...,None,:,:,:],normals[...,None])[...,0]
        return distances

    def primitive_distances(self,x,centers,z=0.,base_rotation=None,with_normals=False):
        """All primitive gradients for a nonsmooth union's safety constraints.

        Constraining only the currently nearest chassis primitive can miss an
        impending nearest-feature switch. Each nearby primitive gets a row.
        """
        frames,origins,axes=self.frames(x,z,base_rotation)
        ds,normals=self.primitive_arrays(frames,centers,True)
        # Point Jacobians depend on the body and witness, not its primitive.
        # Compute the full chain once, then mask joints downstream of each
        # primitive. Every primitive still supplies its own exact union row.
        full=self.point_jacobians(x,centers,6,origins,axes,z)
        gradients=-np.matmul(normals,full)
        gradients[...,3:]*=self.shape_joint_mask
        result=ds,gradients
        return (*result,normals) if with_normals else result

    def distances_sequence(self, states, centers, z=0., base_rotations=None):
        """Exact same primitive SDFs, vectorized across time samples and points."""
        points = np.asarray(centers)
        rotations=None if base_rotations is None or all(r is None for r in base_rotations) else base_rotations
        if rotations is not None and any(r is None for r in rotations):
            rotations=[Rotation.from_euler('z',state[2]).as_matrix() if r is None else r for state,r in zip(states,rotations)]
        positions,orientations=self.frame_arrays(states,z,rotations)
        primitive=self._primitive_arrays(positions,orientations,points)
        return np.stack([primitive[...,self.shape_bodies==b].min(axis=-1) for b in range(8)],axis=-1)

    def displacement_bound(self, dx, dz=0., tilt=0.):
        return np.linalg.norm(dx[:2])+abs(dz)+1.25*abs(tilt)+np.abs(dx[2:])@self.reach


class SweptCollision:
    def __init__(self, scene, geometry=None):
        self.scene = (load_scene(scene) if isinstance(scene, (str, Path))
                      else scene if scene.get('format') in ('jvs_groundtruth_mesh/v1', 'jvs_lidar_surfels/v1')
                      else validate_scene(scene))
        self.geometry = CollisionGeometry() if geometry is None else geometry
        self.is_mesh = self.scene.get('format') in ('jvs_groundtruth_mesh/v1', 'jvs_lidar_surfels/v1')
        if self.scene.get('format') == 'jvs_lidar_surfels/v1':
            self.centers = np.asarray(self.scene['surfel_centers'], float).reshape(-1, 3).copy()
            radius = float(self.scene['surfel_radius'])
            if not np.all(np.isfinite(self.centers)) or not np.isfinite(radius) or radius <= 0:
                raise ValueError('invalid LiDAR surfels')
            self.radii = np.asarray(self.scene.get('surfel_radii', np.full(len(self.centers), radius)), float).copy()
            if self.radii.shape != (len(self.centers),) or not np.isfinite(self.radii).all() or np.any(self.radii <= 0):
                raise ValueError('invalid per-surface radii')
            self.tree = cKDTree(self.centers)
        elif self.is_mesh:
            with np.load(self.scene['mesh_file'], allow_pickle=False) as archive:
                self.centers = np.asarray(archive['surfel_centers'], float)
                radius = float(archive['surfel_radius'])
            self.radii = np.full(len(self.centers), radius)
            self.tree = cKDTree(self.centers)
        else:
            proxies = [proxy for obstacle in self.scene['obstacles']
                       for proxy in obstacle_spheres(obstacle)]
            self.centers = np.array([proxy[0] for proxy in proxies],float)
            self.radii = np.array([proxy[1] for proxy in proxies],float)
            self.tree = None
        self.margin = float(self.scene.get('minimum_clearance_m',.025))
        k = np.arange(64)+.5
        zz = 1-2*k/len(k); angle = k*np.pi*(3-np.sqrt(5))
        self.directions = np.c_[np.sqrt(1-zz*zz)*np.cos(angle),np.sqrt(1-zz*zz)*np.sin(angle),zz]
        self.directions = np.vstack([self.directions,np.eye(3),-np.eye(3)])

    def active_obstacles(self, x, z=0., base_rotation=None, per_body=8):
        """Return the nearest mesh surfels for every coupled robot body."""
        if self.is_mesh and not len(self.centers):
            raise RuntimeError('collision map has no observed surfaces')
        if not self.is_mesh:
            return self.centers, self.radii
        frames = self.geometry.frames(x, z, base_rotation)[0]
        candidates = set()
        # Execution CBFs activate before geometric contact: at 0.5 m/s and
        # gain 0.6/s their stopping band already starts 0.833 m from a surface.
        # Include chassis extent and voxel radius before a witness enters it.
        radius = 1.5 if per_body is None else .8
        for position, _ in frames:
            candidates.update(self.tree.query_ball_point(position, radius))
        if not candidates:
            _, nearest = self.tree.query(np.array([frame[0] for frame in frames]), k=1)
            candidates.update(np.atleast_1d(nearest).tolist())
        indices = np.fromiter(candidates, dtype=int)
        if per_body is None:
            return self.centers[indices], self.radii[indices]
        distances = self.geometry.distances(x, self.centers[indices], z, base_rotation)
        keep = set()
        count = min(per_body, len(indices))
        for body in range(8):
            selected = np.argpartition(distances[:, body], count - 1)[:count]
            keep.update(indices[selected].tolist())
        keep = np.fromiter(keep, dtype=int)
        return self.centers[keep], self.radii[keep]

    def active_sequence(self, states, z=0., base_rotations=None):
        if not self.is_mesh:
            return self.centers, self.radii
        chosen = []
        rotations = [None] * len(states) if base_rotations is None else base_rotations
        for state, rotation in zip(states, rotations):
            centers, _ = self.active_obstacles(state, z, rotation)
            _, indices = self.tree.query(centers, k=1)
            chosen.extend(np.atleast_1d(indices).tolist())
        indices = np.unique(chosen)
        return self.centers[indices], self.radii[indices]

    def clearance_to(self, x, centers, radii, z=0., base_rotation=None, gradient=False):
        result = self.geometry.distances(x,centers,z,base_rotation,gradient)
        if gradient:
            d,g = result
            return d-radii[:,None], g
        return result-radii[:,None]

    def clearance(self, x, z=0., base_rotation=None, gradient=False):
        centers, radii = self.active_obstacles(x, z, base_rotation)
        return self.clearance_to(x, centers, radii, z, base_rotation, gradient)

    def sweep(self, x, u, duration, z=0., tolerance=.003, max_samples=257, base_rotation=None):
        """Certified lower bound for constant-control flow; finite upper witness.

        Every time lies within half a sample spacing of a sample. The global
        surface speed bounds SDF change there. No local-minimizer assumption.
        If the work cap is hit, return the larger bound error, never a false pass.
        """
        speed = abs(u[0])+np.abs(u[1:])@self.geometry.reach
        count = min(max_samples, max(3, int(np.ceil(speed*duration/(2*tolerance)))+1))
        times = np.linspace(0,duration,count)
        states = [advance(x,u,s) for s in times]
        rotations = ([None]*count if base_rotation is None else
                     [Rotation.from_euler('z',u[1]*t).as_matrix()@base_rotation for t in times])
        if self.is_mesh:
            centers, radii = self.active_sequence(states, z, rotations)
            raw = [self.clearance_to(s, centers, radii, z, r) for s,r in zip(states,rotations)]
        else:
            raw = [self.clearance(s,z,r) for s,r in zip(states,rotations)]
        values = np.array(raw)
        index = np.argmin(values,axis=0)
        upper = np.min(values,axis=0)
        error = speed*duration/(2*(count-1))
        return dict(lower=upper-error, upper=upper, error=error, index=index,
                    states=states,times=times,values=values)

    def interior(self, states, z, point, body, inflation, accumulator, coupled, frames=None):
        """Sampled ball-shrinking KC interior metric; bounded iterations.

        Inflation is obstacle radius + preventive clearance. This extends the
        paper's interior consistency rule to near-contact safety envelopes.
        Distances used for certification ALWAYS come from the uncoupled oracle.
        """
        if frames is None:
            frames = [self.geometry.frames(s,z) for s in states]
        def metric(points):
            return np.min([self.geometry.body_sdf(points,f[0],body) for f in frames],axis=0)-inflation
        vals = np.array([self.geometry.body_sdf(point,f[0],body)[0] for f in frames])-inflation
        idx = int(np.argmin(vals)); x=states[idx]
        j = self.geometry.point_jacobian(x,point,body,frames[idx][1],frames[idx][2],z)
        dirs = self.directions
        candidate = -dirs@j
        if coupled and body >= 2:
            # q0 is yaw (state index 2), q1..q6 are indices 3..8.
            upstream = [body,body+1]  # q_{j-2}, q_{j-1}
            cross=np.cross(j[:,upstream[0]],j[:,upstream[1]])
            if np.linalg.norm(cross)>1e-12:
                cross=cross/np.linalg.norm(cross)
                dirs=np.vstack([dirs,cross,-cross])
                candidate=-dirs@j
            mask = np.all(candidate[:,upstream]*accumulator[upstream] >= -1e-12,axis=1)
            dirs = dirs[mask]
        if not len(dirs):
            return None
        radius = max(.05, -float(vals[idx])+.08)
        # Expand until a surviving direction leaves the swept volume.
        for _ in range(8):
            if np.max(metric(point+radius*dirs)) > 0: break
            radius *= 2
        high,low=radius,0.
        converged = False
        for _ in range(24):
            values = metric(point+radius*dirs)
            k = int(np.argmax(values)); excess = float(values[k])
            if abs(excess) < .001:
                converged = True; break
            if excess>0:high=radius
            else:low=radius
            proposed=radius-excess
            # Safeguard the paper's radius update when a restricted direction
            # is almost tangent and convergence would otherwise be too slow.
            radius=proposed if low+.1*(high-low)<proposed<high-.1*(high-low) else (high+low)/2
        if not converged:
            return None
        direction=dirs[k]
        return -radius, -direction@j

    def terms(self, x, u, duration, z=0., coupled=True, influence=.065):
        # Temporal inflation conservatively covers the intervals between these
        # samples; the final execution audit uses a much tighter certificate.
        times=np.linspace(0,duration,3)
        states=[advance(x,u,s) for s in times]
        centers,radii=self.active_sequence(states,z)
        values=np.array([self.clearance_to(s,centers,radii,z) for s in states])
        idx=np.argmin(values,axis=0)
        speed=abs(u[0])+np.abs(u[1:])@self.geometry.reach
        inflation_error=speed*duration/4
        if np.min(values)-inflation_error >= influence:
            return [],0
        active = np.ones(values.shape[1:], dtype=bool)
        if self.is_mesh:
            # A dense flat surface supplies many almost identical soft costs.
            # Keep its two worst witnesses per body for the expensive interior
            # metric. Hard execution rows and the swept certificate still use
            # their independent collision witnesses, without this reduction.
            worst = np.min(values, axis=0)
            active[:] = False
            count = min(2, len(centers))
            for body in range(8):
                selected = np.argpartition(worst[:, body], count-1)[:count]
                active[selected, body] = True
            keep = np.any(active, axis=1)
            centers, radii = centers[keep], radii[keep]
            values, idx, active = values[:,keep], idx[keep], active[keep]
        frames=[self.geometry.frames(s,z) for s in states]
        gradients = np.array([self.geometry.distances(s,centers,z,gradient=True)[1] for s in states])
        output=[]; filtered=0
        for o,point in enumerate(centers):
            accum=np.zeros(9)
            for b in range(8):
                i=int(idx[o,b]); d=float(values[i,o,b])-inflation_error
                grad=gradients[i,o,b]
                if d < influence and active[o,b]:
                    result=self.interior(states,z,point,b,radii[o]+influence+inflation_error,accum,coupled,frames=frames)
                    if result is not None:
                        depth,grad=result
                        d=depth+influence
                        filtered += int(coupled and b>=2)
                    output.append((d,grad,float(times[i]/duration),o,b))
                accum += grad
        return output,filtered

    def planar_circles(self, reference_points):
        if not self.is_mesh:
            proxies = zip(self.centers, self.radii)
            return [(center[:2],radius+.205+self.margin+.015)
                    for center,radius in proxies if center[2]-radius<.35]
        points=np.asarray(reference_points)
        low=np.min(points,axis=0)-.8;high=np.max(points,axis=0)+.8
        mask=((self.centers[:,2]-self.radii<.35)
              & np.all(self.centers[:,:2]>=low,axis=1)
              & np.all(self.centers[:,:2]<=high,axis=1))
        centers=self.centers[mask,:2]
        if not len(centers): return []
        # Surfels already carry the conservative voxel radius. Grouping them
        # again in 12 cm cells added another 85 mm of artificial blockage and
        # forced avoidable arm reach. Keep their actual planar centers.
        centers=np.unique(centers,axis=0)
        radius=.205+self.margin+.015+float(np.max(self.radii[mask]))
        return [(center,radius) for center in centers]
