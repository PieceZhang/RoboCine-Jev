"""JVS-only backend of the two-part representation.

Internal primitive ownership and state/control dimensions stop at this module.
The public interface uses the same functional part names for every embodiment.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .representation import (Pose, Box, Observation, Request, CompiledGuidance,
                             PARTS, compress_boxes, compress_bounds, vector)
from .kc_svsdf import CollisionGeometry, advance
from .jvs_kinematics import JOINT_ORIGINS


class JVSGeometryBackend:
    def __init__(self, box_budget=3):
        compress_boxes([], box_budget)  # validate before doing any work
        self.box_budget = box_budget
        self.geometry = CollisionGeometry()
        self._platform = None
        self._platform_arrays = None
        shapes=self.geometry.shapes
        self._primitive_radii=np.array([s[3][0] if s[2]!='frustum' else max(s[3][:2]) for s in shapes])
        self._primitive_height=np.array([s[3][1] if s[2]=='cylinder' else s[3][2] if s[2]=='frustum' else 0. for s in shapes])
        self._box_half=np.array([s[3]/2 for s in shapes if s[2]=='box'])

    def residual_geometry(self,state,base_z=0.,base_rotation=None,compress=True):
        """Internal FK/occupancy arrays with the same conservative primitives."""
        geometry=self.geometry
        frames=geometry.frames(state,base_z,base_rotation)[0]
        positions=np.array([p for p,_ in frames]);rotations=np.array([r for _,r in frames])
        mount_r=rotations[0];mount_p=positions[0]+mount_r@JOINT_ORIGINS[0]
        br=rotations[geometry.shape_bodies]
        centers=(positions[geometry.shape_bodies]+np.einsum('nij,nj->ni',br,geometry.shape_offsets)-mount_p)@mount_r
        relative=mount_r.T@br@geometry.shape_rotations
        half=np.empty_like(centers)
        for kind,ids in geometry.shape_groups.items():
            if kind=='sphere':half[ids]=self._primitive_radii[ids,None]
            elif kind=='box':half[ids]=np.einsum('nij,nj->ni',np.abs(relative[ids]),self._box_half)
            else:half[ids]=self._primitive_radii[ids,None]*np.linalg.norm(relative[ids,:,:2],axis=2)+self._primitive_height[ids,None]/2*np.abs(relative[ids,:,2])
        occupancy={}
        for part,mask in ((PARTS[0],geometry.shape_bodies==0),(PARTS[1],geometry.shape_bodies!=0)):
            if compress:
                if part==PARTS[0] and self._platform_arrays is not None:
                    occupancy[part]=self._platform_arrays
                    continue
                low,high=compress_bounds(centers[mask]-half[mask],centers[mask]+half[mask],self.box_budget)
                occupancy[part]=((low+high)/2,(high-low)/2)
                if part==PARTS[0]:self._platform_arrays=occupancy[part]
            else:
                occupancy[part]=(centers[mask],half[mask])
        return (mount_r,mount_p),(rotations[-1],positions[-1]),occupancy

    def primitive_boxes(self, state, base_z=0., base_rotation=None):
        state = vector(state, 9, 'backend state')
        frames, _, _ = self.geometry.frames(state, base_z, base_rotation)
        base_p, base_r = frames[0]
        mount = Pose.from_rt(base_r, base_p+base_r@JOINT_ORIGINS[0])
        camera = Pose.from_rt(frames[-1][1],frames[-1][0])
        inverse = mount.inverse()
        boxes = {k: [] for k in PARTS}
        for body, _, kind, size, offset, rotation in self.geometry.shapes:
            p, r = frames[body]
            center = inverse.apply(p+r@offset)
            relative_r = inverse.rotation@r@rotation
            # Analytic support extents, enclosing entire curved solids rather
            # than sampling their surfaces and accidentally underbounding them.
            if kind == 'sphere':
                half = np.full(3, size[0])
            elif kind == 'box':
                half = np.abs(relative_r)@(size/2)
            elif kind in ('cylinder','frustum'):
                radius = size[0] if kind == 'cylinder' else max(size[:2])
                height = size[1] if kind == 'cylinder' else size[2]
                half = radius*np.linalg.norm(relative_r[:,:2],axis=1) + height/2*np.abs(relative_r[:,2])
            else:
                raise ValueError('unsupported backend primitive')
            boxes[PARTS[0] if body == 0 else PARTS[1]].append(Box(tuple(center), tuple(half)))
        return mount, mount.inverse().compose(camera), boxes

    def relative_residual_batch(self,guidance,states):
        """Exact common-rigid-motion-invariant preferences, batched over FK.

        Mount-frame region containment depends only on the union bounds of
        the primitives. No greedy partition or world Pose objects are needed.
        """
        states=np.asarray(states).copy();states[:,:3]=0.
        positions,rotations=self.geometry.frame_arrays(states)
        n=len(states);identity=np.broadcast_to(np.eye(3),(n,3,3))
        mount=(identity,np.broadcast_to(JOINT_ORIGINS[0],(n,3)))
        camera=(rotations[:,-1],positions[:,-1]);frames=dict(mount=mount,camera=camera)
        values=[]
        for p in guidance.request.pose_preferences:
            fr,fp=frames[p['frame']];target=p['target']
            tr=fr@Rotation.from_quat(target['quaternion_xyzw']).as_matrix()
            tp=fp+np.einsum('nij,j->ni',fr,target['translation_m'])
            ar,ap=mount if p['part']==PARTS[0] else camera
            inverse=tr.transpose(0,2,1)
            delta=np.einsum('nij,nj->ni',inverse,ap-tp)
            excess=np.sign(delta)*np.maximum(np.abs(delta)-p['position_tolerance_m'],0.)
            angle=Rotation.from_matrix(inverse@ar).as_rotvec();length=np.linalg.norm(angle,axis=1)
            angle*=np.maximum(0.,length-p['orientation_tolerance_rad'])[:,None]/np.maximum(length[:,None],1e-12)
            values.append(np.sqrt(p['weight'])*np.concatenate([excess,.35*angle],axis=1))
        if guidance.request.occupancy_preferences:
            geometry=self.geometry;br=rotations[:,geometry.shape_bodies]
            centers=positions[:,geometry.shape_bodies]+np.einsum('npij,pj->npi',br,geometry.shape_offsets)-JOINT_ORIGINS[0]
            relative=br@geometry.shape_rotations;half=np.empty_like(centers)
            for kind,ids in geometry.shape_groups.items():
                selected=relative[:,ids]
                if kind=='sphere':half[:,ids]=self._primitive_radii[ids,None]
                elif kind=='box':half[:,ids]=np.einsum('npij,pj->npi',np.abs(selected),self._box_half)
                else:half[:,ids]=self._primitive_radii[ids,None]*np.linalg.norm(selected[...,:2],axis=-1)+self._primitive_height[ids,None]/2*np.abs(selected[...,2])
            for p in guidance.request.occupancy_preferences:
                mask=geometry.shape_bodies==0 if p['part']==PARTS[0] else geometry.shape_bodies!=0
                region=p['region']
                bound=np.max(np.abs(centers[:,mask]-region['center_m'])+half[:,mask],axis=1)
                values.append(np.sqrt(p['weight'])*np.maximum(bound-region['half_extents_m']+p['margin_m'],0.))
        return np.concatenate(values,axis=1) if values else np.empty((n,0))

    def observe(self, state, stamp_s, *, base_z=0., base_rotation=None, source='measured_fk'):
        state=vector(state,9,'backend state')
        (mr,mp),(cr,cp),raw=self.residual_geometry(state,base_z,base_rotation)
        mount=Pose.from_rt(mr,mp)
        camera=Pose.from_rt(mr.T@cr,mr.T@(cp-mp))
        if self._platform is None:
            self._platform=tuple(Box(tuple(c),tuple(h)) for c,h in zip(*raw[PARTS[0]]))
        occupancy = {PARTS[0]: self._platform,
                     PARTS[1]: tuple(Box(tuple(c),tuple(h)) for c,h in zip(*raw[PARTS[1]]))}
        return Observation(float(stamp_s), mount, camera, occupancy, source)

    def compile(self, request, observation, camera_path):
        from .candidates import path_hash
        if path_hash(camera_path) != request.camera_path_id:
            raise ValueError('camera path binding mismatch')
        if request.end_s > camera_path.duration+1e-8:
            raise ValueError('guidance cannot extend camera deadline')
        if request.start_s < getattr(camera_path, 'activation_s', 0.)-1e-8:
            raise ValueError('guidance cannot precede camera activation')
        return CompiledGuidance(request, observation, camera_path)

    def obstacle_relations(self, observation, scene, max_obstacles=4):
        """Same timestamp projection/uncertainty convention as the local backend.

        Nearest surfel proxies are reported, not invented semantic object IDs.
        Empty/stale observations are explicitly unknown, never free space.
        """
        centers = np.asarray(scene.get('surfel_centers', []), float).reshape(-1,3)
        stamp = float(scene.get('observation_stamp', -1.))
        age = observation.stamp_s-stamp
        if not len(centers) or not -1e-8 <= age <= .6:
            return dict(status='unknown', observation_stamp_s=stamp, relations=[])
        velocities = np.asarray(scene.get('surfel_velocities',np.zeros_like(centers)),float)
        radii = np.asarray(scene.get('surfel_radii',np.full(len(centers),scene.get('surfel_radius',.025))),float)
        ids = np.asarray(scene.get('surfel_track_ids',np.zeros(len(centers))),int)
        ages = np.asarray(scene.get('surfel_ages',np.zeros(len(centers))),float)
        rates = np.where(ids>0,float(scene.get('motion_uncertainty_rate_m_s',.035)),0.)
        radii = radii+rates*(ages+max(age,0.))
        centers = centers+velocities*max(age,0.)
        local = observation.world_from_mount.inverse().apply(centers)
        relations = []
        for part, boxes in observation.occupancy.items():
            if not boxes:
                continue
            d = np.min([b.distance(local) for b in boxes],axis=0)-radii
            for i in np.argsort(d)[:max_obstacles]:
                center = local[i]
                axis = int(np.argmax(np.abs(center)))
                direction = (('front','back'),('left','right'),('above','below'))[axis][int(center[axis]<0)]
                relations.append(dict(part=part, obstacle_id=f'track:{ids[i]}' if ids[i]>0 else f'surface:{i}',
                    center_m=center.tolist(), radius_m=float(radii[i]),
                    velocity_m_s=(observation.world_from_mount.rotation.T@velocities[i]).tolist(),
                    direction=direction, proxy_clearance_m=float(d[i]), relation=('overlap' if d[i]<0 else 'near' if d[i]<.15 else 'separated')))
        return dict(status='observed_local_map', observation_stamp_s=stamp,
                    projection_time_s=observation.stamp_s, frame='mount_snapshot', relations=relations,
                    scope='sampled_observed_surfaces_only_not_free_space_certificate')

    def quadratic_guidance(self, guidance, state, t, base_z=0., base_rotation=None):
        """Linearized soft geometry objective in this backend's control variables.

        Added only to WBC's subordinate objective; existing camera equality,
        actuator inequalities and collision certificates are unchanged.
        """
        zeros = np.zeros((8,8)), np.zeros(8)
        if not guidance.request.start_s <= t <= guidance.request.end_s:
            return zeros
        relative_only=all(p['frame'] in ('mount','camera') for p in
            (*guidance.request.pose_preferences,*guidance.request.occupancy_preferences))
        region_only=all(p['frame']=='mount' and p['relation']=='stay_in_region'
            for p in guidance.request.occupancy_preferences)
        if relative_only and region_only:
            epsilon=1e-5
            states=np.tile(state,(13,1))
            for j in range(6):
                states[1+2*j,j+3]-=epsilon;states[2+2*j,j+3]+=epsilon
            values=self.relative_residual_batch(guidance,states)
            residual=values[0]
            if not len(residual) or np.max(np.abs(residual))<=1e-12:return zeros
            jacobian=np.zeros((len(residual),8))
            jacobian[:,2:]=((values[2::2]-values[1::2])/(2*epsilon)).T
            return jacobian.T@jacobian,jacobian.T@(2*residual)
        # The union AABB in mount coordinates is unchanged by conservative
        # greedy merges. Region-containment residuals need only this union,
        # not the internal box partition used for display and other frames.
        compress=any(p['frame']!='mount' or p['relation']!='stay_in_region'
                     for p in guidance.request.occupancy_preferences)
        residual = guidance.residual_arrays(*self.residual_geometry(state,base_z,base_rotation,compress))
        if not len(residual) or np.max(np.abs(residual)) <= 1e-12:
            # Inside every preference deadband there is no corrective task.
            # Avoid 16 unnecessary FK evaluations in the fast control loop.
            return zeros
        jacobian = np.zeros((len(residual),8))
        epsilon = 1e-5
        for j in range(2 if relative_only else 0,8):
            u = np.eye(8)[j]
            values = []
            for sign in (-1,1):
                moved = advance(state,u,sign*epsilon)
                rotation = None if base_rotation is None else Rotation.from_euler('z',sign*epsilon*u[1]).as_matrix()@base_rotation
                values.append(guidance.residual_arrays(*self.residual_geometry(moved,base_z,rotation,compress)))
            jacobian[:,j] = (values[1]-values[0])/(2*epsilon)
        # Residual velocity tracks -2*residual; units converted by representation.
        return jacobian.T@jacobian, jacobian.T@(2*residual)


def request_for_profile(observation, path, profile):
    """Bounded semantic presets; selected through the existing configuration Choice.

    Pose targets refer to functional frames; occupancy includes whole parts.
    Spatial preferences move with the camera/mount where explicitly requested.
    """
    from .candidates import path_hash
    pose, occupancy = [], []
    if profile == 'base':
        # Maintain the current camera-to-mount relationship, giving the platform
        # an incentive to follow the camera while the primary camera task wins.
        pose.append(dict(part='mobile_platform',frame='camera',
            target=observation.mount_from_camera.inverse().to_dict(),
            position_tolerance_m=[.12,.12,.12],orientation_tolerance_rad=.3,weight=.3))
    if profile in ('local','base'):
        boxes = observation.occupancy['camera_manipulator']
        if boxes:
            bounds = Box.enclosing(np.vstack([b.vertices for b in boxes]))
            region = Box(bounds.center_m, tuple(np.array(bounds.half_extents_m)+.12))
            occupancy.append(dict(part='camera_manipulator',frame='mount',relation='stay_in_region',
                region=region.to_dict(),margin_m=0.,weight=.3))
    return Request(observation.id,path_hash(path),float(path.activation_s),float(path.duration),tuple(pose),tuple(occupancy))


def candidate_geometry(candidate, backend, observation, base_z, base_rotation=None, preferences=True):
    request = request_for_profile(observation,candidate.path,candidate.consequences.get('search_config') if preferences else None)
    guide = backend.compile(request,observation,candidate.path)
    trace = (candidate.kinematics or {}).get('trace',[])
    samples = []
    if trace:
        # Do not collapse the path into an atemporal swept union.
        indices = sorted(set(np.linspace(0,len(trace)-1,min(5,len(trace)),dtype=int)))
        yaw0 = trace[0]['state'][2]
        for index in indices:
            row = trace[index]
            rotation = None if base_rotation is None else Rotation.from_euler('z',row['state'][2]-yaw0).as_matrix()@base_rotation
            snapshot = backend.observe(row['state'],row['t'],base_z=base_z,base_rotation=rotation,source='predicted_fk')
            samples.append(guide.feedback(snapshot,kind='predicted',backend_status='unknown'))
    return dict(request=request.to_dict(),observation=observation.to_dict(),samples=samples,
        scope='sampled_geometric_prediction_not_continuous_collision_certificate')
