#!/usr/bin/env python3
"""Actual RTX/PhysX posture audit; distinct from a locomotion/controller test.

The isolated fixture anchors the base and disables gravity. Joint states are
placed at finite sampled configurations, then read back from PhysX. It does not
claim continuous joint-space certification or physical payload validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=['g1', 'go2', 'ridgeback'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--poses', type=int, default=64, help='Sobol joint configurations, plus home and individual joint endpoints.')
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--skip-endpoints', action='store_true', help='Short diagnostic prefix only; omits individual joint endpoint poses.')
    p.add_argument('--ray-backend', choices=['scalar', 'native'], default='native')
    p.add_argument('--camera-offset-grid', action='store_true', help='Compare finite fixed mounting offsets using housing overlaps and frustum queries.')
    p.add_argument('--only-camera', action='store_true', help='Skip LiDAR ring tests; never counts as LiDAR acceptance.')
    return p.parse_args()


def main():
    a = arguments()
    if a.poses < 0:
        raise ValueError('--poses must be nonnegative')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'validator_source.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'arguments.json').write_text(json.dumps(vars(a), indent=2, default=str)+'\n')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    os.environ.setdefault('ACCEPT_EULA', 'Y')
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, renderer='RaytracedLighting', anti_aliasing=0,
                             multi_gpu=False, limit_cpu_threads=8))
    try:
        status = run(a, app)
    except BaseException:
        import traceback
        (a.output/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        app.close(wait_for_replicator=False)
    return status


def run(a, app):
    import carb
    import numpy as np
    from scipy.spatial.transform import Rotation
    from scipy.stats import qmc
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema, PhysicsSchemaTools
    import omni.replicator.core as rep
    from omni.physx import get_physx_scene_query_interface
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.semantics import add_update_semantics
    from isaacsim.core.utils.types import ArticulationAction
    from jev_agent_wbc.robot_sensor_rig import RobotSensorRig, load_config, pose_transform, set_usd_transform, transform

    world = World(physics_dt=.01, rendering_dt=.01, stage_units_in_meters=1.)
    stage = world.stage
    root = '/World/Robot'
    spec, config_hash = load_config(a.robot, a.config)
    asset = (ROOT/spec['asset']).resolve()
    add_reference_to_stage(str(asset), root)
    body_prims = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    body_paths = [p.GetPath() for p in body_prims]
    body_path_set = set(map(str, body_paths))
    neighbours = {str(path): {str(path)} for path in body_paths}
    original_joint_limits = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            ends = [str(t) for rel in (j.GetBody0Rel(), j.GetBody1Rel()) for t in rel.GetTargets()]
            if len(ends) == 2 and all(e in neighbours for e in ends):
                neighbours[ends[0]].add(ends[1]); neighbours[ends[1]].add(ends[0])
        if p.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(p)
            original_joint_limits[str(p.GetPath())] = [j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()]
    # Adjacent links and short fixed joint chains can overlap by construction.
    allowed_pairs = {key: set.union(*(neighbours[v] for v in values)) for key, values in neighbours.items()}
    colliders = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():
            parent = p
            while parent and str(parent.GetPath()) not in body_path_set:
                parent = parent.GetParent()
            if parent:
                colliders.append((p.GetPath(), str(parent.GetPath())))
    for p in body_prims:
        PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
    anchor = None
    base_name = {'g1': 'pelvis', 'go2': 'base', 'ridgeback': 'base_link'}[a.robot]
    height = {'g1': .80, 'go2': .45, 'ridgeback': 0.}[a.robot]
    if a.robot != 'ridgeback':
        set_usd_transform(stage.GetPrimAtPath(root), transform([0., 0., height], np.eye(3)))
        anchor = UsdPhysics.FixedJoint.Define(stage, '/World/FixtureAnchor')
        anchor.CreateBody1Rel().SetTargets([root+'/'+base_name])
        anchor.CreateLocalPos0Attr(Gf.Vec3f(0., 0., height))
        anchor.CreateLocalRot0Attr(Gf.Quatf(1.))
        anchor.CreateLocalPos1Attr(Gf.Vec3f(0.))
        anchor.CreateLocalRot1Attr(Gf.Quatf(1.))
    else:
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(root+'/world/FixedJoint'))
        joint.CreateLocalPos0Attr(Gf.Vec3f(0., 0., .025930123))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.))
    # The fixture deliberately has no contact floor; low targets are at known
    # world heights and can be observed even for extreme sampled arm postures.
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(1000.)
    UsdGeom.Cube.Define(stage, '/World/Backdrop').CreateSizeAttr(20.)
    set_usd_transform(stage.GetPrimAtPath('/World/Backdrop'), transform([0., 0., -10.1], np.eye(3)))
    add_update_semantics(stage.GetPrimAtPath(root), 'robot')
    rig = RobotSensorRig(stage, root, a.robot, a.config)
    joint_limits_unchanged = all([UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path)).GetLowerLimitAttr().Get(),
                                  UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path)).GetUpperLimitAttr().Get()] == limits
                                 for path, limits in original_joint_limits.items())
    (a.output/'sensor_mounts.json').write_text(json.dumps(rig.manifest(), indent=2)+'\n')
    robot = world.scene.add(SingleArticulation(prim_path=root, name='robot'))
    bodies = {}
    for name in rig.parent_names | {base_name}:
        p = next(p for p in body_prims if p.GetName() == name)
        bodies[name] = world.scene.add(SingleRigidPrim(prim_path=str(p.GetPath()), name=name))
    probes = []
    for i in range(36):
        cube = UsdGeom.Cube.Define(stage, f'/World/Probes/P{i:02d}')
        cube.CreateSizeAttr(.12)
        cube.CreateDisplayColorAttr([Gf.Vec3f(.85, .3, .1)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        # Test objects participate in scene queries but do not push the robot.
        UsdPhysics.FilteredPairsAPI.Apply(cube.GetPrim()).CreateFilteredPairsRel().SetTargets(body_paths)
        set_usd_transform(cube.GetPrim(), transform([100.+i, 0., 0.], np.eye(3)))
        probes.append(cube)
    world.reset()
    names = robot.dof_names
    limits = np.asarray(robot._articulation_view.get_dof_limits())[0]
    home = np.asarray(robot.get_joint_positions()).copy()
    for i, name in enumerate(names):
        if a.robot == 'g1':
            for token, value in [('hip_pitch', -.312), ('knee', .669), ('ankle_pitch', -.363), ('elbow', .6)]:
                if token in name:
                    home[i] = value
            if 'shoulder_roll' in name:
                home[i] = -.2 if name.startswith('right') else .2
        elif a.robot == 'go2':
            if name.endswith('thigh_joint'):
                home[i] = .8
            if name.endswith('calf_joint'):
                home[i] = -1.5
        elif name.startswith('ur_arm_'):
            home[i] = dict(zip(['ur_arm_shoulder_pan_joint', 'ur_arm_shoulder_lift_joint', 'ur_arm_elbow_joint',
                               'ur_arm_wrist_1_joint', 'ur_arm_wrist_2_joint', 'ur_arm_wrist_3_joint'],
                              [0., -1.1, 1.6, -.5, np.pi/2, 0.]))[name]
    if a.robot == 'g1':
        indices = [i for i, n in enumerate(names) if any(s in n for s in ('shoulder', 'elbow', 'wrist', 'waist'))]
    elif a.robot == 'go2':
        indices = [names.index('joint'+str(i)) for i in range(1, 7)]
    else:
        indices = [i for i, n in enumerate(names) if n.startswith('ur_arm_')]
    lower = np.maximum(limits[indices, 0], -2*np.pi)+.02
    upper = np.minimum(limits[indices, 1], 2*np.pi)-.02
    home = np.clip(home, limits[:, 0]+.001, limits[:, 1]-.001)
    targets = [('home', home.copy())]
    for i, lo, hi in ([] if a.skip_endpoints else zip(indices, lower, upper)):
        for label, value in [('low', lo), ('high', hi)]:
            q = home.copy(); q[i] = value
            targets.append((names[i]+'_'+label, q))
    if a.poses:
        samples = qmc.Sobol(len(indices), scramble=True, seed=a.seed).random_base2(int(np.ceil(np.log2(a.poses))))[:a.poses]
        for k, values in enumerate(lower+(upper-lower)*samples):
            q = home.copy(); q[indices] = values
            targets.append((f'sobol_{k:03d}', q))
    robot._articulation_view.set_gains(kps=np.full((1, len(names)), 1000.), kds=np.full((1, len(names)), 60.))
    controller = robot.get_articulation_controller()
    query = get_physx_scene_query_interface()
    native_batch = None
    validated_native_rays = 0
    if a.ray_backend == 'native':
        from physx_lidar import load_native
        native_batch = load_native(a.output).RayBatch(8)
    product = rep.create.render_product(rig.camera_path, tuple(spec['camera']['resolution']))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    seg = rep.AnnotatorRegistry.get_annotator('semantic_segmentation', init_params={'colorize': False})
    rgb.attach(product); seg.attach(product)
    frames = a.output/'rgb'; frames.mkdir()
    masks = a.output/'self_masks'; masks.mkdir()
    scans = a.output/'scans'; scans.mkdir()
    rows, detection_rows = [], []
    offset_scores = []
    if a.camera_offset_grid:
        if native_batch is None:
            raise ValueError('camera offset grid requires the validated native backend')
        forward = np.asarray(rig.camera_spec['forward_parent'])
        up = np.asarray(rig.camera_spec['up_parent'])
        right = np.cross(forward, up)
        distances, elevations = ([.12, .16, .20], [.08, .11, .14]) if a.robot == 'g1' else ([0., .025, .05, .075], [-.06, 0., .06, .12])
        if a.robot == 'ridgeback':
            distances, elevations = [0., .04, .08], [-.06, 0., .06]
        offset_scores = [dict(translation_m=(forward*d+up*u+right*r).tolist(), housing_overlap_poses=0,
                              self_view_poses=0, self_ray_fraction_sum=0., any_failure_poses=0)
                         for d in distances for u in elevations for r in [-.04, 0., .04]]
        half_width = np.tan(np.deg2rad(rig.camera_spec['horizontal_fov_deg'])/2)
        xx, yy = np.meshgrid(np.linspace(-half_width, half_width, 13), np.linspace(-half_width*.75, half_width*.75, 9))
        optical_rays = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
        optical_rays /= np.linalg.norm(optical_rays, axis=1)[:, None]
    started = time.monotonic()
    steps = 0

    def step():
        nonlocal steps
        world.step(render=False)
        steps += 1
        remaining = started+steps*.01-time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def poses():
        return {name: pose_transform(*body.get_world_pose()) for name, body in bodies.items()}

    def overlap_box(matrix, extent, ignore_body):
        collisions = []
        def hit(value):
            body = str(value.rigid_body)
            if body.startswith(root+'/') and body != ignore_body:
                collisions.append(str(value.collision))
            return True
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        query.overlap_box(carb.Float3(*map(float, extent)), carb.Float3(*map(float, matrix[:3, 3])),
                          carb.Float4(*map(float, q)), hit, False)
        return sorted(set(collisions))

    def self_collisions():
        pairs = set()
        for path, owner in colliders:
            def report(hit):
                other = str(hit.rigid_body)
                if other in body_path_set and other not in allowed_pairs[owner]:
                    pairs.add(tuple(sorted([owner, other])))
                return True
            encoded = PhysicsSchemaTools.encodeSdfPath(path)
            query.overlap_shape(encoded[0], encoded[1], report, False)
        return sorted(pairs)

    try:
        for number, (label, target) in enumerate(targets):
            tilt = [.20*np.sin(number*.7), .20*np.sin(number*.9), (number%12)*np.pi/6] if number else [0., 0., 0.]
            if anchor is not None:
                quat = Rotation.from_euler('xyz', tilt).as_quat()
                anchor.GetLocalRot0Attr().Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*map(float, quat[:3]))))
                robot.set_world_pose(np.array([0., 0., height]), quat[[3, 0, 1, 2]])
            else:
                target = target.copy()
                target[names.index('dummy_base_revolute_z_joint')] = tilt[2]
            robot.set_joint_positions(target)
            robot.set_joint_velocities(np.zeros(len(names)))
            controller.apply_action(ArticulationAction(joint_positions=target, joint_velocities=np.zeros(len(names))))
            step()
            body_poses = poses()
            camera_world = body_poses[rig.camera_spec['parent']] @ rig.camera_local
            for _ in range(4 if number else 12):
                world.render()
            pixels = np.asarray(rgb.get_data()).copy()
            segmentation = seg.get_data()
            ids = [int(k) for k, v in segmentation['info']['idToLabels'].items() if v.get('class') == 'robot']
            labels = np.asarray(segmentation['data'])
            if not pixels.size or not labels.size:
                raise RuntimeError('empty RGB or segmentation: rendering is not verified')
            mask = np.isin(labels, ids)
            image = frames/f'{number:03d}_{label}.png'
            Image.fromarray(pixels).save(image)
            mask_path = masks/(image.stem+'.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
            h, w = mask.shape[:2]
            housing = camera_world @ transform([0., 0., -.025], np.eye(3))
            overlaps = overlap_box(housing, [.032, .024, .022], str(bodies[rig.camera_spec['parent']].prim_path))
            lidar_overlaps = {}
            for lidar in rig.lidars:
                matrix = body_poses[lidar['spec']['parent']] @ lidar['local']
                lidar_overlaps[lidar['spec']['name']] = overlap_box(matrix, [.005, .005, .005], '')
            record = dict(index=number, label=label, stamp_s=steps*.01, joint_target=target.tolist(),
                          measured_joints=np.asarray(robot.get_joint_positions()).tolist(),
                          max_joint_state_error_rad=float(np.max(np.abs(robot.get_joint_positions()-target))),
                          base_rpy_requested=tilt, T_world_camera_optical=camera_world.tolist(),
                          camera_self_occlusion_fraction=float(mask.mean()),
                          camera_centre_self_occluded=bool(mask[h//2-8:h//2+8, w//2-8:w//2+8].any()),
                          camera_housing_other_body_overlaps=overlaps, rgb=str(image.relative_to(a.output)))
            record['lidar_origin_robot_overlaps'] = lidar_overlaps
            record['robot_nonadjacent_body_overlaps'] = self_collisions()
            record['self_mask'] = str(mask_path.relative_to(a.output))
            record['semantic_labels'] = segmentation['info']['idToLabels']
            record['rgb_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
            if offset_scores:
                parent_world = body_poses[rig.camera_spec['parent']]
                origins = np.asarray([parent_world[:3, 3]+parent_world[:3, :3]@s['translation_m'] for s in offset_scores], dtype=np.float32)
                directions = np.asarray(optical_rays @ camera_world[:3, :3].T, dtype=np.float32)
                flags, points, identities, labels, pairs = native_batch.scan(query, origins, directions)
                if number == 0:
                    for k in range(len(flags)):
                        old = query.raycast_closest(tuple(map(float, origins[k//len(directions)])),
                                                    tuple(map(float, directions[k%len(directions)])), 4.5)
                        if bool(flags[k]) != old['hit']:
                            raise RuntimeError('camera native/scalar hit mismatch')
                        if old['hit']:
                            np.testing.assert_array_equal(points[k], old['position'])
                    validated_native_rays += len(flags)
                is_robot = np.asarray([False if k == 0 else str(PhysicsSchemaTools.intToSdfPath(int(pair[0]))).startswith(root+'/')
                                       for k, pair in enumerate(pairs)])
                fractions = is_robot[labels].reshape(len(offset_scores), -1).mean(axis=1)
                for index, score in enumerate(offset_scores):
                    housing = transform(origins[index], camera_world[:3, :3]) @ transform([0., 0., -.025], np.eye(3))
                    collides = bool(overlap_box(housing, [.032, .024, .022], str(bodies[rig.camera_spec['parent']].prim_path)))
                    score['housing_overlap_poses'] += int(collides)
                    score['self_view_poses'] += int(fractions[index] > 0.)
                    score['self_ray_fraction_sum'] += float(fractions[index])
                    score['any_failure_poses'] += int(collides or fractions[index] > 0.)
            rows.append(record)
            for radius in ([] if a.only_camera else [.7, 1.5]):
                for z in [.06, .20, .50, .90, 1.35]:
                    centre = body_poses[base_name][:3, 3]
                    centres = []
                    for k, cube in enumerate(probes):
                        angle = k*2*np.pi/len(probes)
                        point = np.array([centre[0]+radius*np.cos(angle), centre[1]+radius*np.sin(angle), z])
                        set_usd_transform(cube.GetPrim(), transform(point, np.eye(3)))
                        centres.append(point.tolist())
                    step()
                    detected, self_hits = set(), 0
                    raw_points, raw_sensors, raw_rays, raw_path_ids = [], [], [], []
                    path_ids, matrices = {}, []
                    measured_poses = poses()
                    for sensor_index, lidar in enumerate(rig.lidars):
                        matrix = measured_poses[lidar['spec']['parent']] @ lidar['local']
                        matrices.append(matrix.tolist())
                        rays = lidar['rays'] @ matrix[:3, :3].T
                        if native_batch is not None:
                            origins = np.asarray([matrix[:3, 3]], dtype=np.float32)
                            directions = np.asarray(rays, dtype=np.float32)
                            hit_mask, points, identities, labels, pairs = native_batch.scan(query, origins, directions)
                            if number == 0 and radius == .7 and z == .06:
                                for k, direction in enumerate(directions):
                                    old = query.raycast_closest(tuple(map(float, origins[0])), tuple(map(float, direction)), 4.5)
                                    if bool(hit_mask[k]) != old['hit']:
                                        raise RuntimeError('native/scalar hit mismatch')
                                    if old['hit']:
                                        np.testing.assert_array_equal(points[k], old['position'])
                                        path = str(PhysicsSchemaTools.intToSdfPath(int(identities[k, 0])))
                                        if path != str(old['collision']):
                                            raise RuntimeError('native/scalar collision identity mismatch')
                                validated_native_rays += len(directions)
                            hits = [(int(k), str(PhysicsSchemaTools.intToSdfPath(int(pairs[int(labels[k]), 0]))), points[k])
                                    for k in np.flatnonzero(hit_mask)]
                        else:
                            hits = []
                            for ray_index, direction in enumerate(rays):
                                hit = query.raycast_closest(tuple(map(float, matrix[:3, 3])), tuple(map(float, direction)),
                                                            float(lidar['spec']['max_range_m']))
                                if hit['hit']:
                                    hits.append((ray_index, str(hit['collision']), hit['position']))
                        for ray_index, path, point in hits:
                                if path not in path_ids:
                                    path_ids[path] = len(path_ids)
                                raw_points.append(point); raw_sensors.append(sensor_index)
                                raw_rays.append(ray_index); raw_path_ids.append(path_ids[path])
                                if path.startswith('/World/Probes/P'):
                                    if np.linalg.norm(np.asarray(point)-matrix[:3, 3]) >= lidar['spec']['min_range_m']:
                                        detected.add(int(path.rsplit('P', 1)[1]))
                                elif path.startswith(root+'/'):
                                    self_hits += 1
                    scan_file = scans/f'{number:03d}_r{radius}_z{z}.npz'
                    np.savez_compressed(scan_file, points_world=np.asarray(raw_points, dtype=np.float32).reshape(-1, 3),
                                        sensor=np.asarray(raw_sensors, dtype=np.uint8), ray_index=np.asarray(raw_rays, dtype=np.int32),
                                        path_id=np.asarray(raw_path_ids, dtype=np.uint16), paths=np.asarray(list(path_ids)),
                                        T_world_lidars=np.asarray(matrices), stamp_s=steps*.01)
                    detection_rows.append(dict(pose_index=number, radius_m=radius, target_centre_z_m=z,
                                               stamp_s=steps*.01, targets_world=centres, detected=sorted(detected),
                                               missing=sorted(set(range(36))-detected), robot_shadow_rays=self_hits,
                                               raw_scan=str(scan_file.relative_to(a.output))))
            with (a.output/'postures.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            with (a.output/'detection.jsonl').open('a') as f:
                for r in ([] if a.only_camera else detection_rows[-10:]):
                    f.write(json.dumps(r)+'\n')
            print('POSTURE', a.robot, number+1, '/', len(targets), 'self_pixels', round(record['camera_self_occlusion_fraction'], 4),
                  'targets', sum(len(r['detected']) for r in detection_rows[-10:]), '/ 360', flush=True)
        elapsed = time.monotonic()-started
        feasible_rows = [r for r in rows if not r['robot_nonadjacent_body_overlaps']]
        summary = dict(robot=a.robot, protocol='actual_rtx_physx_sampled_sensor_coverage/v1',
                       asset=str(asset), asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(), config_sha256=config_hash,
                       posture_count=len(rows), sampled_joints=[names[i] for i in indices], joint_names=names,
                       original_revolute_joint_limits_unchanged=joint_limits_unchanged,
                       original_robot_collision_shapes_checked=len(colliders),
                       original_robot_self_overlap_pose_count=len(rows)-len(feasible_rows),
                       no_nonadjacent_overlap_pose_count=len(feasible_rows),
                       camera_clear_in_no_overlap_subset=bool(feasible_rows) and all(r['camera_self_occlusion_fraction']==0 for r in feasible_rows),
                       sampled_joint_lower_rad=lower.tolist(), sampled_joint_upper_rad=upper.tolist(), seed=a.seed,
                       camera_gpu_render_verified=True, max_camera_self_occlusion_fraction=max(r['camera_self_occlusion_fraction'] for r in rows),
                       camera_self_occluded_pose_count=sum(r['camera_self_occlusion_fraction'] > 0 for r in rows),
                       camera_centre_occluded_pose_count=sum(r['camera_centre_self_occluded'] for r in rows),
                       camera_housing_overlap_pose_count=sum(bool(r['camera_housing_other_body_overlaps']) for r in rows),
                       lidar_origin_overlap_pose_count=sum(any(r['lidar_origin_robot_overlaps'].values()) for r in rows),
                       target_observations=len(detection_rows)*36,
                       target_detections=sum(len(r['detected']) for r in detection_rows),
                       lidar_all_test_targets_detected=bool(detection_rows) and all(not r['missing'] for r in detection_rows),
                       camera_all_sampled_views_clear=all(r['camera_self_occlusion_fraction']==0 for r in rows),
                       camera_all_sampled_housings_clear=all(not r['camera_housing_other_body_overlaps'] for r in rows),
                       diagnostic_prefix=a.skip_endpoints,
                       lidar_ring_test_executed=not a.only_camera,
                       ray_backend=a.ray_backend, native_rays_validated_against_scalar=validated_native_rays,
                       query_max_range_m=4.5 if native_batch is not None else max(l['spec']['max_range_m'] for l in rig.lidars),
                       full_continuous_joint_space_verified=False, agent_avoidance_verified=False,
                       payload_mass_and_bracket_verified=False, simulation_seconds=steps*.01, wall_seconds=elapsed,
                       pace_requested=1, realtime_ratio=steps*.01/elapsed,
                       scope='Anchored gravity-free finite posture queries with actual RTX images and actual PhysX rays. Pose placement is not a locomotion trajectory. Camera housing is a 64x48x44 mm query proxy; no payload mass or cable model.')
        (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        if offset_scores:
            ordered = sorted(offset_scores, key=lambda s: (s['any_failure_poses'], s['housing_overlap_poses'], s['self_ray_fraction_sum']))
            (a.output/'camera_mount_search.json').write_text(json.dumps(dict(postures=len(rows), rays_per_view=len(optical_rays),
                scope='Finite collision-proxy/frustum search; chosen offset still requires actual RGB and continuous clearance validation.',
                candidates=ordered), indent=2)+'\n')
        stage.GetRootLayer().Export(str(a.output/'tested_scene.usda'))
        print('SUMMARY', json.dumps(summary), flush=True)
        return 0
    finally:
        if native_batch is not None:
            native_batch.close()
        rgb.detach(product); seg.detach(product); product.destroy()


if __name__ == '__main__':
    raise SystemExit(main())
