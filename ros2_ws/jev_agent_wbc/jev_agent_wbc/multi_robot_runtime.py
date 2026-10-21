"""Shared sensor/Representation ingress for five *native* physical WBCs.

Scene authoring and independent truth audit live here. The Jev and numerical
workers receive only measured geometry, timestamped LiDAR, and cinematic tasks.
"""
from concurrent.futures import Future, InvalidStateError, ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .contracts import native
from .representation import Pose
from .multi_robot_geometry import MeasuredGeometryFilter, shape_clearance, obstacle_relations
from .multi_robot_scenarios import bind_scenario, timed_path
from .scenarios import RuntimeScene


def _mapping_initialize(geometry_filter):
    from .dynamic_lidar_map import DynamicLidarMap
    global _process_mapper, _subject_cells, _other_cells
    _process_mapper=DynamicLidarMap(geometry=geometry_filter,maximum_height=3.,
                                  voxel_size=.04,minimum_clearance=.025)
    _subject_cells=set();_other_cells=set()


def _mapping_integrate(packets):
    if packets is not None:
        _process_mapper.integrate_scans([
            {key:value for key,value in packet.items() if key!='subject_return_mask'}
            for packet in packets])
        # Side-channel labels come from acquired closest-hit identities.
        # They do not remove or alter any point in the physical mapper.
        for packet in packets:
            world=packet['points']@packet['rotation'].T+packet['translation']
            keys=np.floor(world/_process_mapper.voxel_size).astype(np.int64)
            subject=np.asarray(packet['subject_return_mask'],bool)
            _subject_cells.update(map(tuple,keys[subject]))
            _other_cells.update(map(tuple,keys[~subject]))
        snapshot=_process_mapper.snapshot()
        keys=np.floor(np.asarray(snapshot['surfel_centers'])/_process_mapper.voxel_size).astype(np.int64)
        snapshot['surfel_subject_mask']=np.array([
            track==0 and tuple(key) in _subject_cells and tuple(key) not in _other_cells
            for key,track in zip(keys,snapshot['surfel_track_ids'])])
        snapshot['subject_label_source']='acquired_closest_hit_identity_visibility_only'
        return snapshot
    return True


def add_arguments(parser):
    parser.add_argument('--avoidance-case', choices=('orbit', 'movein'))
    parser.add_argument('--avoidance-mode', choices=('jev', 'nominal', 'probe'), default='jev')
    parser.add_argument('--avoidance-lead-s', type=float, default=6.)
    parser.add_argument('--avoidance-obstacle-height-offset-m',type=float,default=0.)
    parser.add_argument('--avoidance-probe-axis',choices=('radial','lateral','height'),default='radial')
    parser.add_argument('--avoidance-probe-m',type=float,default=.30)
    parser.add_argument('--avoidance-probe-candidate-index',type=int,choices=range(6))
    parser.add_argument('--avoidance-probe-family',choices=('framing','view','joint'),default='joint')
    parser.add_argument('--avoidance-probe-activation-s',type=float,default=2.)


def freeze_package(package, destination):
    """Freeze all numerical/semantic dependencies without touching old runs."""
    import shutil
    shutil.copytree(Path(package)/'jev_agent_wbc', destination,
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))


def _numeric_initialize():
    # Compile before physics begins, in a separate interpreter/GIL.
    from .native_geometry_kernel import prewarm
    prewarm()
    from .native_visibility import prewarm as visibility_prewarm
    visibility_prewarm()
    from .native_geometry_kernel import AVAILABLE
    if AVAILABLE:
        from .native_lift_solver import prewarm as solver_prewarm
        solver_prewarm()
    from . import multi_robot_planning, intent_paths, camera_intent, floating_camera


def _numeric_ready():
    return True


def _native_candidates(original, subject_model, intrinsics, snapshot, observation,
                       model_document, start_s, family, configuration, relaxed):
    """One bounded native search; no simulator, network, or authored obstacle input."""
    from dataclasses import asdict
    from .task_space_planning import camera_scene
    from .multi_robot_planning import candidate_paths, lift, card, native_search_budget, CameraAttachmentScreen
    from .candidates import path_hash
    from .floating_camera import evaluate_path
    from .native_visibility import intent_prediction
    from .camera_intent import IntentSpec
    budget=native_search_budget()
    relative_stamp=observation.stamp_s-start_s
    activation=max(0.,relative_stamp+budget.activation_lead_s)
    events=[]
    begun=time.monotonic()
    def emit(event, **details):
        events.append(native(dict(event=event, **details)))
    observed = camera_scene(snapshot, observation.stamp_s, start_s+8.)
    from .multi_robot_audit import visibility_scene
    view=visibility_scene(snapshot,observation.stamp_s,start_s+8.)
    relative_observed=dict(view,observation_stamp=view['observation_stamp']-start_s)
    relative_collision=dict(observed,observation_stamp=observed['observation_stamp']-start_s)
    specification=IntentSpec(original,subject_model,intrinsics)
    ordered=[]
    deadline=time.monotonic()+budget.candidate_wall_budget_s
    attachment_screen=CameraAttachmentScreen(model_document,observed,
        start_s,max(0.,relative_stamp),budget.prediction_step_s)
    for path in candidate_paths(original,family,configuration,activation):
        score,_=evaluate_path(path,relative_collision,0.,8.,.06)
        intent=intent_prediction(specification,path,relative_observed,0.,step=.1)
        attached_clearance=attachment_screen.clearance(path)
        emit('camera_proposal_screen',path_id=path_hash(path),camera=score.to_dict(),intent=intent,
                  rigid_camera_attachment_clearance_m=attached_clearance)
        if intent['geometric_complete'] and intent['motion_trend']['pass_check'] and intent['camera_limits_pass']:
            ordered.append((attachment_screen.rank(attached_clearance,score),path,intent))
    ordered.sort(key=lambda entry:entry[0])
    candidates = []
    for _,path,intent in ordered[:budget.max_rollouts]:
        try:
            report = lift(path, model_document, observed, start_s=start_s,
                          progress_start_s=max(0.,relative_stamp),
                          relaxed=relaxed, step=budget.prediction_step_s,deadline=deadline)
        except TimeoutError as exc:
            emit('candidate_budget_exhausted',reason=str(exc),budget=asdict(budget))
            break
        report['cinematic_prediction']=intent
        c = card(path, report, observation, start_s)
        emit('candidate', card=c)
        if report['physical_valid']:
            candidates.append((path, report, c))
            # max_rollouts is an upper bound. Submit the first fully
            # screened candidate while its original activation is fresh.
            break
    emit('native_search_completed',wall_s=time.monotonic()-begun,
         source_observation_stamp_s=observation.stamp_s)
    return candidates,events


class AvoidanceRuntime:
    def __init__(self, args, package, stage, rig, geometry, base_name, robot, *, native_platform_limits=None):
        from .native_geometry_kernel import prewarm
        prewarm()
        # Load numerical adapters before any physical command lease exists.
        # A first probe/selection must not pay SciPy and intent-module import
        # costs in the 100 Hz supervision loop.
        from . import multi_robot_planning, intent_paths, search_configuration, camera_intent, task_space_planning
        from pxr import UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf
        self.args, self.package, self.stage = args, Path(package), stage
        self.rig, self.geometry, self.base_name, self.robot = rig, geometry, base_name, robot
        self.platform_limits=None if native_platform_limits is None else np.asarray(native_platform_limits,float)
        if self.platform_limits is not None and (self.platform_limits.shape!=(3,)
                or not np.isfinite(self.platform_limits).all() or np.any(self.platform_limits<=0)):
            raise ValueError('three positive native platform command limits required')
        self.output = args.output/'avoidance'
        self.output.mkdir()
        self.events = (self.output/'events.jsonl').open('w', buffering=1)
        self.measurements = (self.output/'measurements.jsonl').open('w', buffering=1)
        self.observations = (self.output/'observations.jsonl').open('w', buffering=1)
        self.native_states = (self.output/'native_states.jsonl').open('w', buffering=1)
        self.scans_log = (self.output/'lidar.jsonl').open('w', buffering=1)
        self.maps_log = (self.output/'observed_maps.jsonl').open('w', buffering=1)
        self.runtime_timing = (self.output/'supervision_timing.jsonl').open('w', buffering=1)
        (self.output/'scans').mkdir()
        self.mapping_pool = None
        self.numeric_pool = None
        self.api_transport = None
        if args.avoidance_mode == 'jev':
            import multiprocessing
            from .multi_robot_transport import OfficialPersistentTransport
            self.api_transport=OfficialPersistentTransport()
            self.numeric_pool=ProcessPoolExecutor(max_workers=1,
                mp_context=multiprocessing.get_context('spawn'),initializer=_numeric_initialize)
            self.numeric_pool.submit(_numeric_ready).result(timeout=30.)
        self.planning_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='shared-jev-planning')
        self.selection_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='official-jev-selection')
        self.mapping_future = self.planning_future = None
        self.seed_request=None
        self._last_recorded_observation_id=None
        self.snapshot = self.binding = self.original = self.model = None
        self.chosen = self.lift = None
        self.last_scan = -1e6
        self.rows = []
        self.map_errors = []
        self.closed = False
        self.execution_complete = False
        self.execution_failure = None
        self.generation = 0
        self.planning_attempts = 0
        self.search_feedback=[]
        self.next_plan_s = -1e6
        self.map_filter = self.mapping = None
        if geometry is not None:
            self.attach_geometry(geometry)
        self.scene_start = None
        self.client = None
        self.raycaster = None
        self.validated_scanners = set()
        self.scanner_validation_progress = {}
        from physx_lidar import load_native
        self.ray_module = load_native(self.output)
        self.actors = []
        material=UsdShade.Material.Define(stage,'/World/AvoidanceMaterial')
        physical_material=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physical_material.CreateStaticFrictionAttr(1.)
        physical_material.CreateDynamicFrictionAttr(1.)
        physical_material.CreateRestitutionAttr(0.)
        combine='multiply' if getattr(args,'ground_material','')=='source_multiply' else 'average'
        physical_settings=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physical_settings.CreateFrictionCombineModeAttr(combine)
        physical_settings.CreateRestitutionCombineModeAttr(combine)
        authored = json.loads((self.package/'scenarios'/args.avoidance_case/'runtime_scene.json').read_text())
        for i, obj in enumerate(authored['obstacles']):
            sphere = UsdGeom.Sphere.Define(stage, f'/World/Moving/MultiRobotObstacle{i}')
            sphere.CreateRadiusAttr(float(obj['radius']))
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*obj['color'])])
            sphere.AddTranslateOp().Set(Gf.Vec3d(100.+i, 100., 10.))
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material,materialPurpose='physics')
            rigid = UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
            rigid.CreateKinematicEnabledAttr(True)
            self.actors.append(sphere)
        subject = UsdGeom.Cylinder.Define(stage, '/World/BlueSubject')
        subject.CreateRadiusAttr(authored['subject']['radius'])
        subject.CreateHeightAttr(authored['subject']['height'])
        subject.CreateDisplayColorAttr([Gf.Vec3f(.05, .25, .9)])
        subject.AddTranslateOp().Set(Gf.Vec3d(102., 100., 10.))
        UsdPhysics.CollisionAPI.Apply(subject.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(subject.GetPrim()).Bind(material,materialPurpose='physics')
        UsdPhysics.RigidBodyAPI.Apply(subject.GetPrim()).CreateKinematicEnabledAttr(True)
        self.subject_prim = subject

    def attach_geometry(self, geometry):
        import multiprocessing
        self.geometry=geometry
        self.map_filter=MeasuredGeometryFilter(geometry)
        self.mapping_pool=ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_mapping_initialize,initargs=(self.map_filter,))
        # Initialization occurs before the physical control loop. No physics,
        # stage object, ray query or authored scene crosses this process boundary.
        self.mapping_pool.submit(_mapping_integrate,None).result(timeout=20.)

    def emit(self, event, **details):
        self.events.write(json.dumps(native(dict(event=event, **details)), allow_nan=False)+'\n')

    def bind(self, stamp, bodies, camera, model):
        from pxr import Gf
        observation = self.geometry.observe(stamp, bodies[self.base_name], camera, bodies, box_budget=8)
        self.original, document, self.binding = bind_scenario(self.args.avoidance_case, observation, self.package/'scenarios',
            obstacle_height_offset_m=getattr(self.args,'avoidance_obstacle_height_offset_m',0.))
        self.scene = RuntimeScene(document)
        self.subject_model=deepcopy(document['subject'])
        from .multi_robot_audit import camera_intrinsics
        self.intrinsics=camera_intrinsics(self.rig.manifest()['camera'])
        if self.platform_limits is not None:
            model.document['platform_limits']=self.platform_limits.tolist()
            model.platform_limits=self.platform_limits.copy()
        self.model = model
        self.scene_start = stamp+self.args.avoidance_lead_s
        self.initial_observation = observation
        # The authored upstream shot is already authorized. A late semantic
        # decision may replace only its still-unexecuted, identical prefix.
        self.chosen = self.original
        if self.args.avoidance_mode=='probe':
            from .contracts import Envelope,PARAMETERS
            from .reference import DeformedReference
            value=float(self.args.avoidance_probe_m)
            probe_activation=float(self.args.avoidance_probe_activation_s)
            if not np.isfinite(probe_activation) or not 0.<=probe_activation<=6.:
                raise ValueError('diagnostic activation must leave at least two authored seconds')
            if not np.isfinite(value) or abs(value)>.4:
                raise ValueError('diagnostic deformation exceeds existing bounded search range')
            axis=('radial','lateral','height').index(self.args.avoidance_probe_axis)
            values=np.zeros(6);values[axis]=value
            bounds={name:[-abs(value),abs(value)] if k==axis else [0.,0.] for k,name in enumerate(PARAMETERS)}
            envelope=Envelope(self.original.sha256,'local_WBC_diagnostic_NOT_Jev','obstacle',
                              0.,0.,8.,3.,bounds,('subject_aim',),'recover','Physical native-WBC diagnostic only.')
            self.chosen=DeformedReference(self.original,envelope,values,activation_s=probe_activation,return_s=5.)
            if self.args.avoidance_probe_candidate_index is None:
                self.emit('numerical_probe_without_agent_approval',axis=self.args.avoidance_probe_axis,metres=value,
                          excluded_from_agent_acceptance=True)
            else:
                from .multi_robot_planning import candidate_paths
                self.chosen=list(candidate_paths(self.original,self.args.avoidance_probe_family,'explore',probe_activation))[
                    self.args.avoidance_probe_candidate_index]
                from .candidates import path_spec,path_hash
                self.emit('generated_candidate_probe_without_agent_approval',
                    family=self.args.avoidance_probe_family,index=self.args.avoidance_probe_candidate_index,
                    path_id=path_hash(self.chosen),trajectory=path_spec(self.chosen),
                    excluded_from_agent_acceptance=True)
        for filename, value in [('scenario_binding.json', self.binding), ('runtime_scene.json', document),
                                ('original_task.json', self.original.task), ('native_model.json', model.document),
                                ('initial_observation.json', observation.to_dict())]:
            (self.output/filename).write_text(json.dumps(native(value), indent=2)+'\n')
        from .candidates import path_spec,path_hash
        (self.output/'initial_execution_path.json').write_text(json.dumps(native(dict(
            path_id=path_hash(self.chosen),trajectory=path_spec(self.chosen),
            source='local_probe' if self.args.avoidance_mode=='probe' else 'authorized_original')),
            indent=2)+'\n')
        # Reset only the scene actors once. Writing a USD xform while physics
        # runs can sweep a kinematic body from its staging pose through Robot.
        import omni.physics.tensors as tensors
        self.actor_view=tensors.create_simulation_view('numpy')
        self.actor_view.set_subspace_roots('/')
        self.obstacle_views=[self.actor_view.create_rigid_body_view(str(p.GetPath())) for p in self.actors]
        self.subject_view=self.actor_view.create_rigid_body_view(str(self.subject_prim.GetPath()))
        subject_pose=np.r_[document['subject']['position'],document['subject']['orientation_xyzw']][None].astype(np.float32)
        self.subject_view.set_transforms(subject_pose,np.array([0],np.uint32))
        self.subject_view.set_kinematic_targets(subject_pose,np.array([0],np.uint32))
        for view,obj in zip(self.obstacle_views,self.scene.obstacles(stamp-self.scene_start)):
            pose=np.r_[obj['center'],[0.,0.,0.,1.]][None].astype(np.float32)
            view.set_transforms(pose,np.array([0],np.uint32))
            view.set_kinematic_targets(pose,np.array([0],np.uint32))
        from omni.physx import get_physx_scene_query_interface
        self.query = get_physx_scene_query_interface()
        self.raycaster = self.ray_module.RayBatch(4)
        self.update_scene(stamp)
        self.emit('scenario_bound', stamp_s=stamp, shot_start_s=self.scene_start, binding=self.binding,
                  agent_core='unchanged CandidateDecider + SystemOneClient',
                  native_wbc=robot_backend(self.robot), raw_camera_feedback=True, requested_pace=1.)

    def update_scene(self, stamp):
        if self.scene_start is None:
            return
        for view, obj in zip(self.obstacle_views, self.scene.obstacles(stamp-self.scene_start)):
            view.set_kinematic_targets(np.r_[obj['center'],[0.,0.,0.,1.]][None].astype(np.float32),
                                       np.array([0],np.uint32))

    def _scan(self, stamp, bodies):
        from pxr import PhysicsSchemaTools
        packets, audit = [], []
        state = self.map_filter.state(bodies, bodies[self.base_name])
        for item in self.rig.lidars:
            spec = item['spec']
            T = bodies[spec['parent']]@item['local']
            rays = np.asarray(item['rays']@T[:3, :3].T, dtype=np.float32)
            origin = np.asarray(T[:3, 3], dtype=np.float32)
            hit, points, identities, labels, pairs = self.raycaster.scan(
                self.query, origin[None], rays, float(spec['max_range_m']))
            if spec['name'] not in self.validated_scanners:
                from carb import Float3
                # Check the full uniform direction schedule over four live
                # acquisitions. Each comparison still uses native and public
                # queries at the SAME physical state. This bounds startup
                # supervision work without removing a beam from any scan.
                progress=self.scanner_validation_progress.setdefault(
                    spec['name'],dict(round=0,rays_checked=0,first_stamp_s=float(stamp)))
                uniform=np.linspace(0,len(rays)-1,min(128,len(rays)),dtype=int)
                contacts=np.flatnonzero(hit)[::max(1,int(hit.sum())//64)]
                indices=np.unique(np.r_[uniform[progress['round']::4],contacts[progress['round']::4]])
                for k in indices:
                    public=self.query.raycast_closest(Float3(*map(float,origin)),Float3(*map(float,rays[k])),
                                                       float(spec['max_range_m']))
                    if bool(hit[k])!=public['hit']:
                        raise RuntimeError('native/public LiDAR closest-hit mismatch')
                    if hit[k]:
                        np.testing.assert_allclose(points[k],public['position'],atol=1e-6,rtol=0.)
                        paths=tuple(str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in identities[k])
                        if paths!=(public['collision'],public['rigidBody']):
                            raise RuntimeError('native/public LiDAR occluder identity mismatch')
                progress['round']+=1;progress['rays_checked']+=len(indices)
                if progress['round']==4:
                    self.validated_scanners.add(spec['name'])
                    self.emit('native_ray_public_api_check',sensor=spec['name'],
                              rays_checked=progress['rays_checked'],acquisitions=4,
                              first_stamp_s=progress['first_stamp_s'],last_stamp_s=float(stamp),
                              full_scan_beams=len(rays),range_m=spec['max_range_m'])
            own = np.zeros(len(pairs), bool)
            subject_hits=np.zeros(len(pairs),bool)
            for i, pair in enumerate(pairs):
                paths=[str(PhysicsSchemaTools.intToSdfPath(int(v))) for v in pair]
                own[i] = any(path.startswith(self.rig.root+'/') for path in paths)
                subject_hits[i]=str(self.subject_prim.GetPath()) in paths
            distance = np.linalg.norm(points-origin, axis=1)
            valid = hit & ~own[labels] & (distance >= spec['min_range_m'])
            world = points[valid].astype(float)
            # World-coordinate points with an explicit acquisition transform.
            packets.append(dict(points=(world-T[:3, 3])@T[:3, :3],
                rotation=T[:3, :3].copy(), translation=T[:3, 3].copy(),
                state=state.copy(), base_z=float(bodies[self.base_name][2, 3]),
                base_rotation=bodies[self.base_name][:3, :3].copy(), stamp=float(stamp),
                sensor=spec['name'], minimum_range=spec['min_range_m'], maximum_range=spec['max_range_m'],
                subject_return_mask=subject_hits[labels][valid],
                exact_ray_geometry=True, acquisition_duration_s=0.))
            file = f'scans/{round(stamp*1000):09d}_{spec["name"]}.npz'
            np.savez(self.output/file, points_world=world, hit_mask=hit,
                     self_blocked_mask=own[labels], valid_mask=valid,
                     subject_return_mask=subject_hits[labels][valid],T_world_lidar=T)
            audit.append(dict(sensor=spec['name'], rays=len(rays), valid_returns=int(valid.sum()),
                              self_blocked=int(np.sum(hit & own[labels])), maximum_range_m=spec['max_range_m'],
                              T_world_lidar=T.tolist(), measured_points_file=file))
        self.scans_log.write(json.dumps(dict(stamp_s=stamp, sensors=audit))+'\n')
        return packets

    def update(self, stamp, bodies, camera, observation=None, *, native_joint_positions=None,
               native_platform_state=None):
        if self.scene_start is None:
            return
        wall_started=time.monotonic()
        scan_wall_s=0.
        if self.mapping_future is not None and self.mapping_future.done():
            try:
                self.snapshot = self.mapping_future.result()
                self.maps_log.write(json.dumps(native(dict(consumed_s=stamp, snapshot=self.snapshot)))+'\n')
            except Exception as exc:
                self.map_errors.append(str(exc))
                self.emit('mapping_failure', stamp_s=stamp, error=str(exc))
            self.mapping_future = None
        scanned=False
        if stamp-self.last_scan >= .1-1e-8 and self.mapping_future is None:
            scan_started=time.monotonic()
            packets = self._scan(stamp, bodies)
            self.mapping_future = self.mapping_pool.submit(_mapping_integrate, packets)
            self.last_scan = stamp
            scanned=True
            scan_wall_s=time.monotonic()-scan_started
        if observation is None:
            observation = self.geometry.observe(stamp, bodies[self.base_name], camera, bodies)
        if abs(observation.stamp_s-stamp)>1e-8:
            raise ValueError('runtime observation must identify the current physical acquisition')
        if native_platform_state is not None and native_platform_state.get('stamp_s')!=stamp:
            raise ValueError('private native platform calibration must bind the current acquisition')
        def record_acquisition():
            if self._last_recorded_observation_id==observation.id:return
            self.observations.write(json.dumps(observation.to_dict(),allow_nan=False)+'\n')
            self.native_states.write(json.dumps(native(dict(stamp_s=stamp,observation_id=observation.id,
                joint_positions=native_joint_positions,body_poses=bodies,
                native_platform_state=native_platform_state)),allow_nan=False)+'\n')
            self._last_recorded_observation_id=observation.id
        if scanned:
            record_acquisition()
        # Strategy/configuration may use the pre-roll observations. Numerical
        # construction receives a fresh acquisition after those decisions.
        # Pre-roll is useful computation time too; the authored shot clock,
        # activation lead and installation age checks remain unchanged.
        requested=self.seed_request
        if (requested is not None and stamp>=requested[1] and not requested[0].done()
                and self.snapshot is not None and 0<=stamp-self.snapshot['observation_stamp']<=.25
                and any(t['moving_confirmed'] for t in self.snapshot.get('tracks',[]))
                and native_joint_positions is not None):
            calibrated=self.model.recalibrated(bodies,native_joint_positions,native_platform_state)
            record_acquisition()
            try:
                requested[0].set_result((deepcopy(self.snapshot),observation,calibrated.document))
            except InvalidStateError:
                pass  # A concurrently expired request cannot resume execution.
        # The first bounded semantic search can prepare on observed occupancy.
        # Numerical construction still waits for measured motion confirmation.
        moving = self.snapshot is not None and any(t['moving_confirmed'] for t in self.snapshot.get('tracks', []))
        initial_map=(self.planning_attempts==0 and self.snapshot is not None
                     and 0<=stamp-self.snapshot['observation_stamp']<=.25)
        if (self.args.avoidance_mode == 'jev' and self.planning_future is None and (moving or initial_map)
                and len(self.validated_scanners)==len(self.rig.lidars)
                and self.generation==0 and self.planning_attempts<3 and stamp>=self.next_plan_s
                and stamp<self.scene_start+2.):
            self.planning_attempts += 1
            if native_joint_positions is None:
                raise ValueError('planning requires acquisition-bound native joint state')
            model=self.model.recalibrated(bodies,native_joint_positions,native_platform_state)
            record_acquisition()
            self.planning_future = self.planning_pool.submit(self._plan, deepcopy(self.snapshot), observation,model.document)
            self.emit('planning_submitted', stamp_s=stamp, observation_id=observation.id)
        if self.planning_future is not None and self.planning_future.done():
            try:
                chosen, report = self.planning_future.result()
                activation = self.scene_start+getattr(chosen, 'activation_s', 0.)
                fresh=self.installation_gate(chosen,report,stamp,bodies,camera) if chosen is not None else dict(pass_check=False,reason=report.get('reason'))
                self.emit('fresh_installation_check',stamp_s=stamp,check=fresh)
                if chosen is not None and stamp < activation and fresh['pass_check']:
                    self.chosen, self.lift = chosen, report
                    self.generation += 1
                    self.emit('exact_plan_installed', stamp_s=stamp, path_id=report['path_id'])
                elif chosen is not None:
                    self.emit('late_plan_rejected', stamp_s=stamp, activation_s=activation)
            except Exception as exc:
                self.emit('planning_failed', stamp_s=stamp, error=f'{type(exc).__name__}: {exc}')
            self.planning_future = None
            self.next_plan_s=stamp+.5
        if stamp-self.scene_start >= -.001:
            truth = self.scene.obstacles(stamp-self.scene_start)
            actual_obstacles=[view.get_transforms()[0,:3].astype(float).copy() for view in self.obstacle_views]
            clearance = shape_clearance(self.geometry.shape_world_bounds(bodies),
                actual_obstacles, [o['radius'] for o in truth])
            t = float(np.clip(stamp-self.scene_start, 0., 8.))
            p, r, _ = self.original.sample(t)
            chosen_p,chosen_r,_=self.chosen.sample(t)
            actual = observation.world_from_camera
            from scipy.spatial.transform import Rotation
            row = dict(stamp_s=stamp, shot_s=t, observation=observation.to_dict(),
                truth_clearance_lower_bound_m=clearance,
                actual_obstacle_centers=actual_obstacles,
                scene_tracking_error_m=max((float(np.linalg.norm(p-o['center'])) for p,o in zip(actual_obstacles,truth)),default=0.),
                original_position_error_m=float(np.linalg.norm(np.asarray(actual.translation_m)-p)),
                original_orientation_error_rad=float(Rotation.from_matrix(r@actual.rotation.T).magnitude()),
                selected_position_error_m=float(np.linalg.norm(np.asarray(actual.translation_m)-chosen_p)),
                selected_orientation_error_rad=float(Rotation.from_matrix(chosen_r@actual.rotation.T).magnitude()),
                environment=self.environment(stamp, bodies), plan_installed=self.generation > 0,
                actual_camera=actual.to_dict())
            self.rows.append(row)
            self.measurements.write(json.dumps(native(row), allow_nan=False)+'\n')
        wall_s=time.monotonic()-wall_started
        if scanned or wall_s>.02:
            self.runtime_timing.write(json.dumps(dict(stamp_s=stamp,wall_s=wall_s,
                scan_wall_s=scan_wall_s,scanned=scanned))+'\n')

    def _plan(self, snapshot, observation, model_document):
        from dataclasses import asdict
        from .jev import CandidateDecider, SystemOneClient
        from .search_configuration import offers
        from .task_space_planning import camera_scene
        from .multi_robot_planning import candidate_paths, lift, card, native_search_budget, CameraAttachmentScreen
        from .candidates import path_hash
        from .floating_camera import evaluate_path, intent_prediction
        from .camera_intent import IntentSpec
        budget=native_search_budget()
        relative_stamp=observation.stamp_s-self.scene_start
        activation=max(0.,relative_stamp+budget.activation_lead_s)
        if activation+2.>=self.original.duration:
            return None,dict(reason='insufficient_remaining_authored_time')
        os.environ['TYPESAFE_API_KEY_FILE'] = str(self.package.parents[1]/'typesafe_Jev_api_key')
        os.environ.pop('TYPESAFE_API_KEY', None)
        self.client = SystemOneClient(url='https://api.typesafe.ai/v1/systemone', model='jev-1.14.0',
                                     journal=self.output/'jev_api.jsonl')
        if getattr(self,'api_transport',None) is not None:
            self.client.open_request=self.api_transport.open
        decider = CandidateDecider(self.client)
        from .multi_robot_evidence import reference_context,native_search_feedback
        from .representation_agent import search_outcome
        context,risk=reference_context(self.original,observation,snapshot,self.scene_start)
        history=getattr(self,'search_feedback',[])
        moving=any(t['moving_confirmed'] for t in snapshot.get('tracks',[]))
        evidence = dict(stage='strategy', objective='legacy_reference_tracking',
            intent=self.original.task['intent'], shot_phase=self.original.task['shot_phase'],
            robot_geometry=observation.summary(obstacle_relations(observation,snapshot,max_obstacles=2)), original_duration_s=8.,
            obstacles=dict(source='measured_lidar_only', tracks=snapshot.get('tracks', [])),
            **context,recent_backend_outcomes=[search_outcome(row) for row in history[-3:]],
            difficulty=dict(issues=['observed_moving_obstacle'] if moving else ['observed_occupancy_motion_unknown'],
                            first_risk_s=risk),
            active_modification=False, prediction_scope='native_kinematic_screen_then_independent_physical_WBC')
        evidence.update(search_budget=asdict(budget),snapshot_time_s=relative_stamp,
                        activation_s=activation,original_deadline_s=8.,
                        backend_capabilities=dict(common_features=['timed_camera_se3'],
                            geometric_preferences='not_supported_by_all_native_executors'))
        strategy = decider.decide(evidence)
        self.emit('strategy', decision=strategy)
        if strategy['strategy'] == 'refuse':
            return None, dict(reason='semantic_refusal')
        options=offers(strategy['strategy'], {},history)
        # Preserve the typed configuration stage, but advertise only presets
        # requiring features actually implemented by all five native WBCs.
        # Unsupported platform/occupancy requests must never be ignored.
        options['configurations']={k:v for k,v in options['configurations'].items()
                                   if k in ('timing','explore')}
        configuration = decider.decide(dict(evidence, stage='configuration',search_options=options))
        self.emit('configuration', decision=configuration)
        if configuration['search_config'] not in options['configurations']:
            return None,dict(reason='unsupported_native_geometric_preference_configuration')
        if hasattr(self,'planning_pool'):
            request=Future()
            # Keep construction near the shot: a much earlier map adds avoidable
            # uncertainty to every future occupancy proxy. This is asynchronous
            # pre-roll scheduling, not a change to activation or freshness gates.
            self.seed_request=(request,max(observation.stamp_s,
                                          self.scene_start-(2*budget.activation_lead_s+.3)))
            try:
                snapshot,observation,model_document=request.result(timeout=budget.candidate_wall_budget_s)
            finally:
                request.cancel()
                if self.seed_request is not None and self.seed_request[0] is request:
                    self.seed_request=None
            relative_stamp=observation.stamp_s-self.scene_start
            activation=max(0.,relative_stamp+budget.activation_lead_s)
            if activation+2.>=self.original.duration:
                return None,dict(reason='fresh_construction_has_insufficient_authored_time')
            evidence.update(robot_geometry=observation.summary(obstacle_relations(observation,snapshot,max_obstacles=2)),snapshot_time_s=relative_stamp,
                activation_s=activation,obstacles=dict(source='measured_lidar_only',tracks=snapshot.get('tracks',[])))
            context,risk=reference_context(self.original,observation,snapshot,self.scene_start)
            evidence.update(context)
            evidence['difficulty']['first_risk_s']=risk
            self.emit('fresh_numerical_acquisition',stamp_s=observation.stamp_s,
                      observation_id=observation.id,map_stamp_s=snapshot['observation_stamp'],
                      activation_s=activation)
        arguments=(self.original,self.subject_model,self.intrinsics,snapshot,observation,
                   model_document,self.scene_start,strategy['strategy'],configuration['search_config'],
                   self.robot in ('g1','go2'))
        if getattr(self,'numeric_pool',None) is not None:
            candidates,events=self.numeric_pool.submit(_native_candidates,*arguments).result(
                timeout=budget.candidate_wall_budget_s)
        else:
            candidates,events=_native_candidates(*arguments)
        feedback=native_search_feedback(events,relative_stamp,activation)
        self.search_feedback=[*history,feedback][-3:]
        pending=None
        if candidates and getattr(self,'selection_pool',None) is not None:
            # Full numerical checks have finished. Overlap their immutable
            # journal serialization with the single final network request.
            pending=self.selection_pool.submit(decider.decide,
                dict(evidence,stage='selection',candidates=[c for _,_,c in candidates]))
        for event in events:
            self.emit(event.pop('event'),**event)
        if not candidates:
            return None, dict(reason='no_native_model_screened_candidate')
        decision = (pending.result() if pending is not None else
            decider.decide(dict(evidence, stage='selection', candidates=[c for _, _, c in candidates])))
        self.emit('selection', decision=decision)
        for path, report, c in candidates:
            if c['id'] == decision['candidate_id']:
                report['path_id'] = path_hash(path)
                report['camera_path_id']=c['visual']['geometry']['request']['camera_path_id']
                report['source_observation_stamp_s']=observation.stamp_s
                report['source_map_stamp_s']=snapshot['observation_stamp']
                report['native_model_document']=model_document
                executable=getattr(path,'_native_execution_trajectory',None)
                if executable is None:executable=timed_path(path,observation,self.scene_start).trajectory
                if executable.id!=report['camera_path_id']:
                    raise ValueError('selected candidate executable knots changed before recording')
                (self.output/'selected_plan.json').write_text(json.dumps(native(dict(
                    card=c,lift=report,camera_trajectory=executable.to_dict())),indent=2)+'\n')
                return path, report
        raise ValueError('exact candidate selection has no matching immutable path')

    def task(self, stamp, bodies, camera):
        observation = self.geometry.observe(stamp, bodies[self.base_name], camera, bodies)
        task=timed_path(self.chosen, observation, self.scene_start,
                          relaxed=self.robot in ('g1', 'go2'),
                          task_id=f'{self.robot}_{self.args.avoidance_case}_{self.generation}')
        if self.generation and task.trajectory.id!=self.lift['camera_path_id']:
            raise ValueError('native camera knots differ from the exact selected candidate')
        return task

    def installation_gate(self,path,report,stamp,bodies,camera):
        """Recheck the exact selected numerical witness against a fresh map."""
        from .candidates import path_hash
        from .multi_robot_planning import native_search_budget
        from .task_space_planning import camera_scene
        from .camera_intent import obstacle_state
        from .multi_robot_geometry import NativeKinematicModel
        from scipy.spatial.transform import Rotation
        budget=native_search_budget()
        age=stamp-report['source_observation_stamp_s']
        if (path_hash(path)!=report['path_id'] or not 0<=age<=budget.max_prediction_age_s
                or self.snapshot is None or not 0<=stamp-self.snapshot['observation_stamp']<=.25):
            return dict(pass_check=False,reason='path_binding_or_observation_age')
        times=np.asarray(report['times_s']);states=np.asarray(report['states'])
        at=max(0.,stamp-self.scene_start)
        current=np.array([np.interp(at,times,states[:,i]) for i in range(states.shape[1])])
        model=NativeKinematicModel(report['native_model_document'])
        predicted=model.camera(current)
        pe=float(np.linalg.norm(predicted[:3,3]-camera[:3,3]))
        ae=float(Rotation.from_matrix(predicted[:3,:3]@camera[:3,:3].T).magnitude())
        observed=camera_scene(self.snapshot,stamp,self.scene_start+8.)
        gaps=[]
        for t,state in zip(times,states):
            if t<at:continue
            centers,radii=obstacle_state(observed,self.scene_start+t)
            gaps.append(model.camera_and_clearance(state,centers,radii)[1])
        clearance=min((g for g in gaps if g is not None),default=None)
        relaxed=self.robot in ('g1','go2')
        passed=bool(pe<(.08 if relaxed else .03) and ae<(.20 if relaxed else .08)
                    and clearance is not None and clearance>=.025 and self.environment(stamp,bodies)['status']=='pass')
        return dict(pass_check=passed,position_model_residual_m=pe,orientation_model_residual_rad=ae,
                    minimum_fresh_forecast_clearance_m=clearance,verification_stamp_s=stamp,
                    scope='fresh_native_witness_screen_plus_independent_measured_environment_gate')

    def revision(self, session, stamp, bodies, camera):
        identity=f'{self.robot}_{self.args.avoidance_case}_{self.generation}'
        if session.task_id == identity:
            return None
        task=self.task(stamp,bodies,camera)
        observation=self.geometry.observe(stamp,bodies[self.base_name],camera,bodies)
        admission=session.accept(task,observation,stamp)
        if not admission['supported']:
            raise RuntimeError('fresh native task revision rejected: '+str(admission))
        self.emit('task_revision_admitted',stamp_s=stamp,task=task.to_dict(),admission=admission)
        return task

    def environment(self, stamp, bodies):
        from .task_space_planning import camera_scene
        from .camera_intent import obstacle_state
        if self.snapshot is None or not 0 <= stamp-self.snapshot['observation_stamp'] <= .25:
            return dict(status='unknown', stamp_s=None, scope='not_available')
        scene = camera_scene(self.snapshot, stamp, stamp)
        centers, radii = obstacle_state(scene, stamp)
        distance = shape_clearance(self.geometry.shape_world_bounds(bodies), centers, radii)
        return dict(status='unknown' if distance is None else 'pass' if distance >= .025 else 'fail',
            stamp_s=float(stamp),acquisition_stamp_s=float(self.snapshot['observation_stamp']),
            valid_until_s=float(self.snapshot['observation_stamp'])+.25,scope='fresh_environment_map',
            clearance_m=distance)

    def should_brake(self, stamp, bodies):
        if self.scene_start is None or stamp < self.scene_start:
            return False
        return self.chosen is None or self.environment(stamp, bodies)['status'] != 'pass'

    def environment_barriers(self, stamp, bodies, body_jacobians):
        """Native velocity inequalities from measured collider/surfel distances."""
        from .task_space_planning import camera_scene
        from .camera_intent import obstacle_state
        if self.snapshot is None or not 0 <= stamp-self.snapshot['observation_stamp'] <= .25:
            return None
        scene = camera_scene(self.snapshot, stamp, stamp)
        centers, radii = obstacle_state(scene, stamp)
        if not len(centers):
            return None
        velocities = np.asarray(scene['velocities'])
        rates = np.asarray(scene['radius_rates'])
        rows, lower = [], []
        for shape in self.geometry.shape_world_bounds(bodies):
            rotation, origin = shape['rotation_world'], shape['center_world']
            local = (centers-origin)@rotation
            clipped = np.clip(local, -shape['half_extents_body'], shape['half_extents_body'])
            delta = local-clipped
            length = np.linalg.norm(delta, axis=1)
            distances = length-radii
            for k in np.flatnonzero(distances < .4):
                if length[k] < 1e-8:
                    # Measured overlap cannot be certified by a zero gradient.
                    return dict(status='infeasible', stamp_s=stamp, reason='observed_volume_intersection')
                normal = -rotation@delta[k]/length[k]
                point = rotation@clipped[k]+origin
                J = np.asarray(body_jacobians[shape['body']])
                point_J = J[:3]+np.cross(J[3:].T, point-np.asarray(bodies[shape['body']])[:3, 3]).T
                rows.append((normal@point_J).tolist())
                lower.append(float(normal@velocities[k]+rates[k]-2.*(distances[k]-.025)))
        return dict(status='measured', stamp_s=stamp,
            acquisition_stamp_s=float(self.snapshot['observation_stamp']),
            valid_until_s=float(self.snapshot['observation_stamp'])+.25,
            A=rows, lower=lower, margin_m=.025,
            scope='actual_collider_OBB_and_observed_surfels_velocity_barriers')

    def native_environment(self, stamp):
        from .task_space_planning import camera_scene
        if self.snapshot is None or not 0 <= stamp-self.snapshot['observation_stamp'] <= .25:
            return None
        scene=camera_scene(self.snapshot,stamp,stamp)
        return dict(stamp_s=float(scene['observation_stamp']),
            valid_until_s=float(scene['observation_stamp'])+.25,
            spheres=[dict(center=c.tolist(),velocity=v.tolist(),radius_m=float(r),radius_rate_m_s=float(g))
                for c,v,r,g in zip(scene['centers'],scene['velocities'],scene['radii'],scene['radius_rates'])])

    def tangent_heading(self, stamp, current_yaw):
        """Private nonholonomic allocation from the exact current camera path."""
        t=float(np.clip(stamp-self.scene_start,0.,self.chosen.duration))
        a=max(0.,min(t,self.chosen.duration-1.))
        b=min(self.chosen.duration,a+1.)
        delta=self.chosen.sample(b)[0]-self.chosen.sample(a)[0]
        if np.linalg.norm(delta[:2])<1e-6:
            return float(current_yaw)
        heading=float(np.arctan2(delta[1],delta[0]))
        difference=np.arctan2(np.sin(heading-current_yaw),np.cos(heading-current_yaw))
        if abs(difference)>np.pi/2:
            difference=np.arctan2(np.sin(difference+np.pi),np.cos(difference+np.pi))
        return float(current_yaw+difference)

    def platform_velocity(self, stamp, bodies, camera, *, yaw_share=1., yaw_feedback_gain=0.):
        """Native locomotion input allocation; the Agent never sees this vector."""
        if not np.isfinite(yaw_share) or not 0.<=yaw_share<=1.:
            raise ValueError('native platform yaw allocation must be in [0,1]')
        if not np.isfinite(yaw_feedback_gain) or yaw_feedback_gain<0.:
            raise ValueError('finite nonnegative native yaw feedback required')
        if self.chosen is None or self.should_brake(stamp,bodies):
            return np.zeros(3)
        at=float(np.clip(stamp-self.scene_start,0.,self.chosen.duration))
        target, rotation, twist=self.chosen.sample(at)
        from .wbc_protocol import semantic_camera
        from .wbc_physical import pose
        actual=semantic_camera(pose(camera))
        error=np.asarray(target)-actual.translation_m
        base=bodies[self.base_name]
        # A rotating platform already translates an offset mounted camera.
        # Remove that measured lever-arm velocity before allocating translation.
        base_yaw_rate=yaw_share*twist[5]
        if yaw_feedback_gain:
            from scipy.spatial.transform import Rotation
            yaw_error=Rotation.from_matrix(rotation@actual.rotation.T).as_rotvec()[2]
            lever=np.linalg.norm(np.asarray(actual.translation_m)[:2]-base[:2,3])
            spare=max(0.,min(self.platform_limits[:2])-np.linalg.norm(twist[:2]))
            support=min(self.platform_limits[2],spare/max(lever,.1))
            base_yaw_rate+=np.clip(yaw_feedback_gain*yaw_error,-support,support)
        base_yaw_rate=float(np.clip(base_yaw_rate,-self.platform_limits[2],self.platform_limits[2]))
        world=(np.asarray(twist[:3])
               -np.cross([0.,0.,base_yaw_rate],np.asarray(actual.translation_m)-base[:3,3])+.8*error)
        yaw=np.arctan2(base[1,0],base[0,0])
        planar=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@world[:2]
        # The native arm retains full camera orientation control; base yaw is a
        # bounded supporting preference, not a second camera target.
        if self.platform_limits is None:
            raise ValueError('native locomotion capabilities were not provided')
        return np.clip(np.r_[planar,base_yaw_rate],-self.platform_limits,self.platform_limits)

    def constrain_platform(self,stamp,bodies,velocity,previous,dt):
        """Last native input gate, after servo/coordinator allocations."""
        if self.scene_start is None or stamp<self.scene_start:
            return np.asarray(velocity)
        base=bodies[self.base_name];yaw=np.arctan2(base[1,0],base[0,0])
        lo=np.maximum(-self.platform_limits,np.asarray(previous)-np.array([.5,.5,.8])*dt)
        hi=np.minimum(self.platform_limits,np.asarray(previous)+np.array([.5,.5,.8])*dt)
        brake=np.clip(np.zeros(3),lo,hi)
        velocity=np.clip(velocity,lo,hi)
        jacobians={}
        for name,T in bodies.items():
            J=np.zeros((6,3));J[:3,0]=[np.cos(yaw),np.sin(yaw),0.]
            J[:3,1]=[-np.sin(yaw),np.cos(yaw),0.]
            J[:3,2]=np.cross([0.,0.,1.],T[:3,3]-base[:3,3]);J[5,2]=1.
            jacobians[name]=J
        barriers=self.environment_barriers(stamp,bodies,jacobians)
        if barriers is None or barriers['status']!='measured':
            return brake
        if barriers['A']:
            from scipy.optimize import minimize
            A=np.asarray(barriers['A']);b=np.asarray(barriers['lower'])
            fit=minimize(lambda u:.5*np.sum((u-velocity)**2),velocity,jac=lambda u:u-velocity,
                bounds=list(zip(lo,hi)),constraints=[dict(type='ineq',fun=lambda u:A@u-b,jac=lambda u:A)],
                method='SLSQP',options=dict(maxiter=20,ftol=1e-8))
            if not fit.success or np.min(A@fit.x-b)<-1e-6:
                return brake
            velocity=fit.x
        return velocity

    def finish(self):
        if self.closed:
            return
        self.closed = True
        if self.client is not None:
            self.client.closed = True
        if self.seed_request is not None:
            self.seed_request[0].cancel()
        self.planning_pool.shutdown(wait=True, cancel_futures=True)
        self.selection_pool.shutdown(wait=True,cancel_futures=True)
        if self.api_transport is not None:
            self.api_transport.close()
        if self.numeric_pool is not None:
            self.numeric_pool.shutdown(wait=True,cancel_futures=True)
        if self.mapping_pool is not None:
            self.mapping_pool.shutdown(wait=True, cancel_futures=True)
        if self.raycaster is not None:
            self.raycaster.close()
        clearances = [r['truth_clearance_lower_bound_m'] for r in self.rows
                      if r['truth_clearance_lower_bound_m'] is not None]
        import sys
        exception=sys.exc_info()[1]
        if exception is not None:self.execution_failure=f'{type(exception).__name__}: {exception}'
        shots=[r for r in self.rows if 0<=r['stamp_s']-self.scene_start<=8.+1e-8]
        relaxed=self.robot in ('g1','go2')
        position=np.array([r['selected_position_error_m'] for r in shots])
        angles=np.array([r['selected_orientation_error_rad'] for r in shots])
        tracking=bool(len(position) and np.sqrt(np.mean(position**2))<(.08 if relaxed else .03)
                      and np.sqrt(np.mean(angles**2))<(.20 if relaxed else .08)
                      and position[-1]<(.08 if relaxed else .03) and angles[-1]<(.20 if relaxed else .08))
        result = dict(schema='five_robot_avoidance/v1', robot=self.robot, case=self.args.avoidance_case,
            mode=self.args.avoidance_mode, complete=self.execution_complete,
            execution_failure=self.execution_failure,
            exact_plan_installed=self.generation > 0,
            native_tracking_pass=tracking,
            position_rmse_m=float(np.sqrt(np.mean(position**2))) if len(position) else None,
            orientation_rmse_rad=float(np.sqrt(np.mean(angles**2))) if len(angles) else None,
            terminal_position_error_m=float(position[-1]) if len(position) else None,
            terminal_orientation_error_rad=float(angles[-1]) if len(angles) else None,
            measured_minimum_clearance_lower_bound_m=min(clearances) if clearances else None,
            safety_pass=bool(clearances and min(clearances) >= .025),
            mapping_errors=self.map_errors, physical_execution=True, agent_decision_workflow_unchanged=True,
            agent_core_source_byte_identical=False,
            agent_source_change_scope='optional_HTTP_transport_hook_only; decision_workflow_unchanged',
            requested_pace=1., sensor_scan_period_s=.1,
            sensor_coverage='all configured scanner beams; observed occupancy only',
            all_environment_constraints_pass=bool(self.rows and all(r['environment']['status']=='pass' for r in self.rows)),
            acceptance_tier='relaxed_tracking' if self.robot in ('g1', 'go2') else 'standard',
            successful=False)
        result['physical_shot_pass']=bool(result['complete'] and not self.execution_failure and tracking
            and result['safety_pass'] and result['all_environment_constraints_pass'])
        from .multi_robot_audit import classify_agent_trial
        self.events.flush()
        recorded_events=[json.loads(line) for line in (self.output/'events.jsonl').read_text().splitlines()]
        api_path=self.output/'jev_api.jsonl'
        api_records=[json.loads(line) for line in api_path.read_text().splitlines()] if api_path.exists() else []
        result['agent_trial_status']=classify_agent_trial(
            self.args.avoidance_mode,self.generation>0,recorded_events,api_records)
        result['awaiting_independent_intent_and_native_constraint_audit']=True
        (self.output/'summary.json').write_text(json.dumps(native(result), indent=2)+'\n')
        for stream in (self.events, self.measurements, self.observations, self.native_states,
                       self.scans_log, self.maps_log, self.runtime_timing):
            stream.close()
        return result


def robot_backend(robot):
    return dict(husky='OCS2_SLQ', ridgeback='OCS2_SLQ', tiago='Giskard_Executor',
                g1='AGILE_Pink', go2='LeggedManip_WBC')[robot]
