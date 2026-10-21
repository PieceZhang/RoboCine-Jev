"""Measured geometry adapters for the existing two-part and LiDAR contracts.

Robot names, joints and native actuator ordering never enter Agent evidence.
The numerical model is a kinematic screening tool, not a dynamics certificate.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .representation import Box, Pose, PARTS
from .wbc_protocol import semantic_camera


def _cross3(a,b):
    """Scalar algebra avoids generic axis dispatch for three-vector Jacobians."""
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])


class MeasuredGeometryFilter:
    """Replace only the robot self-filter in the unchanged dynamic mapper.

    Each scan carries its complete acquisition-time collider poses in ``state``.
    No latest-pose lookup can rewrite an earlier scan.
    """
    def __init__(self, geometry):
        self.geometry = geometry
        self.centers = geometry._centers
        self.half = geometry._half

    def state(self, body_poses, base):
        transforms = self.geometry._transforms(body_poses)[self.geometry._indices]
        return np.r_[np.asarray(base)[:2, 3], transforms.ravel()]

    def distances(self, state, points, base_z=0., base_rotation=None):
        transforms = np.asarray(state, float)[2:].reshape(-1, 4, 4)
        if len(transforms) != len(self.half):
            raise ValueError('scan is missing acquisition-time collider transforms')
        points = np.asarray(points).reshape(-1, 3)
        local = np.einsum('nsi,sij->nsj',
            points[:, None]-transforms[None, :, :3, 3], transforms[:, :3, :3])
        delta = np.abs(local-self.centers)-self.half
        return (np.linalg.norm(np.maximum(delta, 0.), axis=2)
                + np.minimum(delta.max(axis=2), 0.))


def shape_clearance(shapes, centers, radii):
    """Conservative actual-collider OBB to sphere lower bound, in metres."""
    centers = np.asarray(centers, float).reshape(-1, 3)
    if not len(centers) or not len(shapes):
        return None
    radii = np.broadcast_to(np.asarray(radii, float), (len(centers),))
    origins=np.asarray([s['center_world'] for s in shapes])
    rotations=np.asarray([s['rotation_world'] for s in shapes])
    half=np.asarray([s['half_extents_body'] for s in shapes])
    local=np.einsum('nsi,sij->nsj',centers[:,None]-origins[None],rotations)
    delta=np.abs(local)-half
    distance=np.linalg.norm(np.maximum(delta,0.),axis=2)+np.minimum(delta.max(axis=2),0.)-radii[:,None]
    return float(distance.min())


def observation_clearance(observation, centers, radii):
    points = observation.world_from_mount.inverse().apply(centers)
    if not len(points):
        return None
    return min(float(np.min(box.distance(points)-radii))
               for part in PARTS for box in observation.occupancy[part])


def obstacle_relations(observation, snapshot, max_obstacles=4):
    """Public module-relative evidence using the native map's exact forecast.

    A tracked surface is a measured proxy, not the complete object's volume.
    Missing or expired acquisitions cannot establish free space.
    """
    from .task_space_planning import camera_scene
    from .camera_intent import obstacle_state
    stamp=float(snapshot.get('observation_stamp',-1.))
    if not 0. <= observation.stamp_s-stamp <= .25:
        return dict(status='unknown',observation_stamp_s=stamp,relations=[])
    scene=camera_scene(snapshot,observation.stamp_s,observation.stamp_s)
    centers,radii=obstacle_state(scene,observation.stamp_s)
    if not len(centers):
        return dict(status='unknown',observation_stamp_s=stamp,relations=[])
    local=observation.world_from_mount.inverse().apply(centers)
    velocities=np.asarray(scene['velocities'])@observation.world_from_mount.rotation
    ids=np.asarray(snapshot.get('surfel_track_ids',np.zeros(len(centers))),int)
    relations=[]
    for part,boxes in observation.occupancy.items():
        if not boxes:continue
        distances=np.min([box.distance(local) for box in boxes],axis=0)-radii
        used=set()
        order=np.argsort(distances)
        # Preserve the nearest proxy of each tracked object even when the
        # nearer static subject has many separate surface cells.
        tracked=[];seen_tracks=set()
        for i in order:
            if ids[i]>0 and ids[i] not in seen_tracks:
                tracked.append(i);seen_tracks.add(ids[i])
        for i in [*tracked,*(i for i in order if ids[i]<=0)]:
            identity=f'track:{ids[i]}' if ids[i]>0 else f'surface:{i}'
            if identity in used:continue
            used.add(identity)
            center=local[i];axis=int(np.argmax(np.abs(center)))
            direction=(('front','back'),('left','right'),('above','below'))[axis][int(center[axis]<0)]
            relations.append(dict(part=part,obstacle_id=identity,
                center_m=center.tolist(),radius_m=float(radii[i]),velocity_m_s=velocities[i].tolist(),
                direction=direction,proxy_clearance_m=float(distances[i]),
                relation='overlap' if distances[i]<0 else 'near' if distances[i]<.15 else 'separated'))
            if len(used)>=max_obstacles:break
    return dict(status='observed_local_map',observation_stamp_s=stamp,
        projection_time_s=observation.stamp_s,frame='mount_snapshot',relations=relations,
        scope='sampled_observed_surfaces_only_not_free_space_certificate')


class NativeKinematicModel:
    """Product-of-exponentials model calibrated from measured native bodies.

    Active joints and their bounds belong to each WBC. A finite native body
    tree is used only inside this backend; no robot-specific Agent branch.
    """
    def __init__(self, document):
        self.document = deepcopy(document)
        self.base = np.asarray(document['base'])
        self.joints = document['joints']
        self.homes = {k: np.asarray(v) for k, v in document['homes'].items()}
        self.chains = document['chains']
        self.shapes = document['shapes']
        self.camera_parent = document['camera_parent']
        self.camera_local = np.asarray(document['camera_local'])
        self.q0 = np.asarray([j['q0'] for j in self.joints])
        self.lower = np.asarray([j['lower'] for j in self.joints])
        self.upper = np.asarray([j['upper'] for j in self.joints])
        self.speed = np.asarray([j['speed'] for j in self.joints])
        self.omni = document['omnidirectional']
        self.platform_limits=np.asarray(document.get('platform_limits',[.22,.22,.35]),float)
        if self.platform_limits.shape!=(3,) or not np.isfinite(self.platform_limits).all() or np.any(self.platform_limits<=0):
            raise ValueError('three positive native platform velocity limits required')
        self.lateral_yaw_gain=float(document.get('native_platform_state',{}).get('lateral_yaw_gain_m',0.))
        if not np.isfinite(self.lateral_yaw_gain) or abs(self.lateral_yaw_gain)>.35:
            raise ValueError('native lateral yaw calibration exceeds its original bounded domain')
        self.shape_bodies = [s['body'] for s in self.shapes]
        self.shape_centers = np.asarray([s['center_body'] for s in self.shapes])
        self.shape_half = np.asarray([s['half_extents_body'] for s in self.shapes])
        self.omega = np.asarray([j['omega'] for j in self.joints])
        self.velocity = np.asarray([j['velocity'] for j in self.joints])
        self.skews = np.asarray([[[0.,-w[2],w[1]],[w[2],0.,-w[0]],[-w[1],w[0],0.]]
                                for w in self.omega])
        self.skew_squares = self.skews@self.skews
        from .native_geometry_kernel import AVAILABLE
        self.compiled_geometry=AVAILABLE
        if AVAILABLE:
            names=list(self.homes)
            self._body_names=names
            self._homes=np.asarray([self.homes[name] for name in names],float)
            self._chain_lengths=np.asarray([len(self.chains[name]) for name in names],np.int64)
            self._chains=np.zeros((len(names),len(self.joints)),np.int64)
            for i,name in enumerate(names):
                self._chains[i,:len(self.chains[name])]=self.chains[name]
            self._shape_bodies=np.asarray([names.index(body) for body in self.shape_bodies],np.int64)
            self._camera_body=names.index(self.camera_parent)

    def _compiled(self,state,centers,radii,active_distance):
        from .native_geometry_kernel import evaluate
        return evaluate(np.asarray(self.base,float),np.asarray(state,float),np.asarray(self.omega,float),
            np.asarray(self.velocity,float),self._homes,self._chains,self._chain_lengths,
            self._shape_bodies,np.asarray(self.shape_centers,float).reshape(-1,3),
            np.asarray(self.shape_half,float).reshape(-1,3),self._camera_body,
            np.asarray(self.camera_local,float),np.asarray(centers,float).reshape(-1,3),
            np.asarray(radii,float),float(active_distance))

    def differential(self, state, centers, radii, active_distance=None):
        if self.compiled_geometry:
            return self._compiled(state,centers,radii,np.inf if active_distance is None else active_distance)[:4]
        return self.differential_reference(state,centers,radii,active_distance)

    def camera_and_clearance(self, state, centers, radii):
        """Reuse one full FK/OBB evaluation for two independent output checks."""
        if self.compiled_geometry:
            result=self._compiled(state,centers,radii,-np.inf)
            return result[0],float(result[2].min()) if result[2].size else None
        return self.camera(state),shape_clearance(self.world_shapes(state),centers,radii)

    def differential_reference(self, state, centers, radii, active_distance=None):
        """Analytic camera/OBB Jacobians for bounded native numerical search."""
        state=np.asarray(state);n=len(state)
        base=self.base.copy()
        base[:3,:3]=Rotation.from_euler('z',state[2]).as_matrix()@self.base[:3,:3]
        base[:2,3]+=state[:2]
        a=state[3:];motion=np.tile(np.eye(4),(len(a),1,1))
        motion[:,:3,:3]+=np.sin(a)[:,None,None]*self.skews+(1-np.cos(a))[:,None,None]*self.skew_squares
        V=a[:,None,None]*np.eye(3)+(1-np.cos(a))[:,None,None]*self.skews+(a-np.sin(a))[:,None,None]*self.skew_squares
        motion[:,:3,3]=np.einsum('nij,nj->ni',V,self.velocity)
        cache={():base}
        def prefix(chain):
            chain=tuple(chain)
            if chain not in cache:cache[chain]=prefix(chain[:-1])@motion[chain[-1]]
            return cache[chain]
        spatial={}
        for chain in self.chains.values():
            for j,index in enumerate(chain):
                if index not in spatial:
                    T=prefix(chain[:j]);w=T[:3,:3]@self.omega[index]
                    spatial[index]=(w,T[:3,:3]@self.velocity[index]+_cross3(T[:3,3],w))
        def jacobian(point,chain):
            J=np.zeros((6,n));J[0,0]=J[1,1]=J[5,2]=1.
            delta=point-base[:3,3];J[:3,2]=[-delta[1],delta[0],0.]
            for index in chain:
                w,v=spatial[index];J[:3,index+3]=_cross3(w,point)+v;J[3:,index+3]=w
            return J
        camera=prefix(self.chains[self.camera_parent])@self.homes[self.camera_parent]@self.camera_local
        camera_j=jacobian(camera[:3,3],self.chains[self.camera_parent])
        centers=np.asarray(centers).reshape(-1,3);radii=np.asarray(radii)
        if not self.shapes or not len(centers):
            shape=(len(centers),len(self.shapes))
            return camera,camera_j,np.empty(shape),np.empty((*shape,n))
        transforms=np.asarray([prefix(self.chains[body])@self.homes[body] for body in self.shape_bodies])
        rotations=transforms[:,:3,:3]
        origins=np.einsum('sij,sj->si',rotations,self.shape_centers)+transforms[:,:3,3]
        local=np.einsum('nsi,sij->nsj',centers[:,None]-origins[None],rotations)
        delta=np.abs(local)-self.shape_half
        outside=np.maximum(delta,0.);length=np.linalg.norm(outside,axis=2)
        distances=length+np.minimum(delta.max(axis=2),0.)-radii[:,None]
        gradients=np.zeros((*distances.shape,n))
        for si,shape in enumerate(self.shapes):
            active=np.ones(len(centers),bool) if active_distance is None else distances[:,si]<active_distance
            if not np.any(active):continue
            R=rotations[si];center=origins[si];point_local=local[active,si]
            normal_local=np.sign(point_local)*outside[active,si]/np.maximum(length[active,si,None],1e-12)
            inside=length[active,si]<1e-10
            for k in np.flatnonzero(inside):
                axis=np.argmax(delta[active,si][k]);normal_local[k,axis]=1. if point_local[k,axis]>=0 else -1.
            normal=-normal_local@R.T
            closest=center+np.clip(point_local,-self.shape_half[si],self.shape_half[si])@R.T
            J=jacobian(center,self.chains[shape['body']])
            point_J=J[None,:3,:]+np.cross(J[None,3:,:].transpose(0,2,1),
                (closest-center)[:,None,:]).transpose(0,2,1)
            gradients[active,si]=np.einsum('ni,nij->nj',normal,point_J)
        return camera,camera_j,distances,gradients

    @classmethod
    def from_stage(cls, stage, root, bodies, base_name, joint_names, positions,
                   lower, upper, speed, geometry, rig, *, omnidirectional):
        from pxr import Usd, UsdPhysics
        parents = {}
        native = {}
        for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not p.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(p)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not a or not b:
                continue
            parent, child = stage.GetPrimAtPath(a[0]).GetName(), stage.GetPrimAtPath(b[0]).GetName()
            parents[child] = (parent, p.GetName())
            if p.GetName() in joint_names:
                if parent not in bodies:
                    raise ValueError('native joint parent has no measured pose: '+parent)
                q = joint.GetLocalRot0Attr().Get()
                rotation = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
                axis_name = (UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get() if p.IsA(UsdPhysics.RevoluteJoint)
                             else UsdPhysics.PrismaticJoint(p).GetAxisAttr().Get())
                axis = np.eye(3)[:, 'XYZ'.index(axis_name)]
                axis_parent=rotation@axis
                point_parent=np.array(joint.GetLocalPos0Attr().Get())
                point = bodies[parent][:3, :3]@point_parent+bodies[parent][:3, 3]
                axis = bodies[parent][:3, :3]@axis_parent
                base_inv = np.linalg.inv(bodies[base_name])
                point = base_inv[:3, :3]@point+base_inv[:3, 3]
                axis = base_inv[:3, :3]@axis
                revolute = p.IsA(UsdPhysics.RevoluteJoint)
                native[p.GetName()] = dict(omega=axis.tolist() if revolute else [0., 0., 0.],
                    velocity=(-np.cross(axis, point) if revolute else axis).tolist(),
                    parent=parent,axis_parent=axis_parent.tolist(),point_parent=point_parent.tolist(),
                    revolute=bool(revolute))
        entries = []
        for i, name in enumerate(joint_names):
            entries.append(dict(native[name], name=name, q0=float(positions[i]),
                lower=float(lower[i]), upper=float(upper[i]), speed=float(speed[i])))
        chains = {}
        for body in bodies:
            sequence, seen, current = [], set(), body
            while current != base_name and current in parents and current not in seen:
                seen.add(current)
                parent, joint = parents[current]
                if joint in joint_names:
                    sequence.append(joint_names.index(joint))
                current = parent
            chains[body] = sequence[::-1]
        inv = np.linalg.inv(bodies[base_name])
        return cls(dict(base=bodies[base_name].tolist(),base_name=base_name, joints=entries,
            homes={k: (inv@v).tolist() for k, v in bodies.items()}, chains=chains,
            shapes=deepcopy(geometry.shapes), camera_parent=rig.camera_spec['parent'],
            camera_local=rig.camera_local.tolist(), omnidirectional=bool(omnidirectional)))

    def recalibrated(self,bodies,joint_positions,platform_state=None):
        """Rebase private screws/homes on one complete physical acquisition."""
        document=deepcopy(self.document)
        if platform_state is not None:
            document['native_platform_state']=deepcopy(platform_state)
        base=np.asarray(bodies[document['base_name']],float)
        inverse=np.linalg.inv(base)
        document['base']=base.tolist()
        document['homes']={name:(inverse@np.asarray(bodies[name])).tolist() for name in self.homes}
        for joint in document['joints']:
            parent=np.asarray(bodies[joint['parent']])
            point=inverse[:3,:3]@(parent[:3,:3]@joint['point_parent']+parent[:3,3])+inverse[:3,3]
            axis=inverse[:3,:3]@parent[:3,:3]@joint['axis_parent']
            joint['omega']=(axis if joint['revolute'] else np.zeros(3)).tolist()
            joint['velocity']=(-np.cross(axis,point) if joint['revolute'] else axis).tolist()
            value=float(joint_positions[joint['name']])
            if not np.isfinite(value):raise ValueError('nonfinite native joint acquisition')
            joint['q0']=value
        return NativeKinematicModel(document)

    def predicted_observation(self,state,stamp_s):
        """Publish only the existing two-part/full-SE(3) contract."""
        from dataclasses import replace
        from .wbc_physical import MeasuredColliderGeometry
        if not hasattr(self,'_representation_geometry'):
            self._representation_geometry=MeasuredColliderGeometry(self.shapes)
        base=self.base.copy()
        base[:3,:3]=Rotation.from_euler('z',state[2]).as_matrix()@self.base[:3,:3]
        base[:2,3]+=np.asarray(state)[:2]
        bodies=self.poses(state)
        camera=bodies[self.camera_parent]@self.camera_local
        result=self._representation_geometry.observe(stamp_s,base,camera,bodies,box_budget=3)
        return replace(result,source='predicted_fk')

    def poses(self, state):
        if self.compiled_geometry:
            values=self._compiled(state,[],[],-np.inf)[4]
            return dict(zip(self._body_names,values))
        return self.poses_reference(state)

    def poses_reference(self, state):
        state = np.asarray(state)
        base = self.base.copy()
        yaw = Rotation.from_euler('z', state[2]).as_matrix()
        base[:3, :3] = yaw@self.base[:3, :3]
        base[:2, 3] += state[:2]
        motion = []
        for value, j in zip(state[3:], self.joints):
            w, v = np.asarray(j['omega']), np.asarray(j['velocity'])
            T = np.eye(4)
            if np.linalg.norm(w):
                x, y, z = w
                W = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                T[:3, :3] = np.eye(3)+np.sin(value)*W+(1-np.cos(value))*W@W
                T[:3, 3] = (value*np.eye(3)+(1-np.cos(value))*W+(value-np.sin(value))*W@W)@v
            else:
                T[:3, 3] = value*v
            motion.append(T)
        result = {}
        cache = {(): np.eye(4)}
        for name, home in self.homes.items():
            chain = tuple(self.chains[name])
            if chain not in cache:
                transform = np.eye(4)
                for index in chain:
                    transform = transform@motion[index]
                cache[chain] = transform
            result[name] = base@cache[chain]@home
        return result

    def camera(self, state):
        if self.compiled_geometry:
            return self._compiled(state,[],[],-np.inf)[0]
        return self.camera_reference(state)

    def camera_reference(self, state):
        # Only the camera's ancestors are needed for numerical IK derivatives.
        state = np.asarray(state)
        base = self.base.copy()
        base[:3, :3] = Rotation.from_euler('z', state[2]).as_matrix()@self.base[:3, :3]
        base[:2, 3] += state[:2]
        result = base
        for index in self.chains[self.camera_parent]:
            value, j = state[3+index], self.joints[index]
            w, v = np.asarray(j['omega']), np.asarray(j['velocity'])
            T = np.eye(4)
            if np.linalg.norm(w):
                x, y, z = w
                W = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                T[:3, :3] = np.eye(3)+np.sin(value)*W+(1-np.cos(value))*W@W
                T[:3, 3] = (value*np.eye(3)+(1-np.cos(value))*W+(value-np.sin(value))*W@W)@v
            else:
                T[:3, 3] = value*v
            result = result@T
        return result@self.homes[self.camera_parent]@self.camera_local

    def world_shapes(self, state):
        bodies = self.poses(state)
        return [dict(center_world=bodies[s['body']][:3, :3]@s['center_body']+bodies[s['body']][:3, 3],
                     rotation_world=bodies[s['body']][:3, :3], half_extents_body=np.asarray(s['half_extents_body']),
                     part=s['part']) for s in self.shapes]

    def obstacle_distances(self, state, centers, radii):
        if not len(centers):
            return np.empty((0,len(self.shapes)))
        if self.compiled_geometry:
            return self._compiled(state,centers,radii,-np.inf)[2]
        poses = self.poses(state)
        transforms = np.asarray([poses[n] for n in self.shape_bodies])
        local = np.einsum('nsi,sij->nsj', np.asarray(centers)[:, None]-transforms[None, :, :3, 3],
                          transforms[:, :3, :3])-self.shape_centers
        delta = np.abs(local)-self.shape_half
        return (np.linalg.norm(np.maximum(delta, 0.), axis=2)
                + np.minimum(delta.max(axis=2), 0.)-np.asarray(radii)[:, None])
