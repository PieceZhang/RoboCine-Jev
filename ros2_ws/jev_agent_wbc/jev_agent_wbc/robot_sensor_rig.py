"""End-effector cameras and body-fixed, closest-hit PhysX scanners.

All transforms use column vectors, metres and acquisition-time physical poses.
USD's row-vector matrices are transposed only at the authoring boundary. No
Isaac imports occur until a stage, scanner or render product is actually used.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config/robot_sensor_mounts.json'
OPTICAL_FROM_USD = np.diag([1., -1., -1., 1.])


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    if not np.isfinite(result).all():
        raise ValueError('nonfinite sensor transform')
    return result


def pose_transform(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    return transform(position, Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix())


def optical_transform(spec):
    forward = np.asarray(spec['forward_parent'], dtype=float)
    up = np.asarray(spec['up_parent'], dtype=float)
    if not (np.isclose(np.linalg.norm(forward), 1.) and np.isclose(np.linalg.norm(up), 1.)
            and np.isclose(forward @ up, 0.)):
        raise ValueError('camera forward/up must be orthogonal unit vectors')
    # Optical: +X right, +Y down, +Z forward.
    return transform(spec['translation_m'], np.column_stack([np.cross(forward, up), -up, forward]))


def lidar_directions(spec):
    count = spec['horizontal_samples']
    elevations = np.deg2rad(spec['elevation_deg'])
    if isinstance(count, bool) or not isinstance(count, int) or count < 4:
        raise ValueError('LiDAR horizontal_samples must be an integer >= 4')
    if elevations.ndim != 1 or not len(elevations) or not np.isfinite(elevations).all() or np.any(np.abs(elevations) >= np.pi/2):
        raise ValueError('invalid LiDAR elevations')
    if not 0 < spec['min_range_m'] < spec['max_range_m'] < float('inf'):
        raise ValueError('invalid LiDAR range')
    azimuth, elevation = np.meshgrid(np.linspace(-np.pi, np.pi, count, endpoint=False), elevations)
    return np.column_stack([(np.cos(elevation)*np.cos(azimuth)).ravel(),
                            (np.cos(elevation)*np.sin(azimuth)).ravel(), np.sin(elevation).ravel()])


def load_config(robot, path=None):
    path = Path(path or DEFAULT_CONFIG)
    source = path.read_bytes()
    config = json.loads(source)
    if config['schema'] != 'robot_sensor_mounts/v1':
        raise ValueError('unsupported sensor mount schema')
    spec = deepcopy(config['robots'][robot])
    spec['camera'] = dict(config['camera'], **spec['camera'])
    spec['lidars'] = [dict(config['lidar'], **item) for item in spec['lidars']]
    names = [item['name'] for item in spec['lidars']]
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError('LiDAR names must be nonempty and unique')
    optical_transform(spec['camera'])
    for item in spec['lidars']:
        lidar_directions(item)
    return spec, hashlib.sha256(source).hexdigest()


def set_usd_transform(prim, matrix):
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(np.asarray(matrix).T.tolist()))


def _body(stage, root, name):
    from pxr import Usd, UsdPhysics
    matches = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root))
               if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(matches) != 1:
        raise ValueError(f'expected one rigid body {name!r} under {root}, found {len(matches)}')
    return matches[0]


class RobotSensorRig:
    def __init__(self, stage, robot_root, robot, config_path=None):
        from pxr import Gf, Sdf, UsdGeom
        self.stage, self.root, self.robot = stage, robot_root.rstrip('/'), robot
        self.spec, self.config_sha256 = load_config(robot, config_path)
        self.camera_spec = self.spec['camera']
        self.camera_local = optical_transform(self.camera_spec)
        parent = _body(stage, self.root, self.camera_spec['parent'])
        optical = UsdGeom.Xform.Define(stage, str(parent.GetPath())+'/CineCameraOptical')
        set_usd_transform(optical.GetPrim(), self.camera_local)
        self.camera_path = str(optical.GetPath())+'/Camera'
        camera = UsdGeom.Camera.Define(stage, self.camera_path)
        set_usd_transform(camera.GetPrim(), OPTICAL_FROM_USD)
        width, height = self.camera_spec['resolution']
        fov = self.camera_spec['horizontal_fov_deg']
        if not (0 < fov < 180 and width > 0 and height > 0):
            raise ValueError('invalid camera FOV/resolution')
        aperture = 36.
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(aperture*height/width)
        camera.CreateFocalLengthAttr(aperture/(2*np.tan(np.deg2rad(fov)/2)))
        camera.CreateClippingRangeAttr(Gf.Vec2f(*self.camera_spec['clipping_m']))
        self.lidars = []
        for item in self.spec['lidars']:
            parent = _body(stage, self.root, item['parent'])
            path = str(parent.GetPath())+'/CineLidar_'+item['name']
            if not Sdf.Path.IsValidIdentifier('CineLidar_'+item['name']):
                raise ValueError('invalid LiDAR name')
            sensor = UsdGeom.Xform.Define(stage, path)
            local = transform(item['translation_m'], Rotation.from_euler('xyz', item['rpy_deg'], degrees=True).as_matrix())
            set_usd_transform(sensor.GetPrim(), local)
            sensor.GetPrim().CreateAttribute('cine:sensorType', Sdf.ValueTypeNames.String).Set('physx_closest_hit_lidar')
            self.lidars.append(dict(spec=item, path=path, local=local, rays=lidar_directions(item)))

    @property
    def parent_names(self):
        return {self.camera_spec['parent'], *(item['spec']['parent'] for item in self.lidars)}

    def manifest(self):
        return dict(schema='robot_sensor_rig/v1', robot=self.robot, robot_root=self.root,
                    config_sha256=self.config_sha256,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    installation_status='prototype_mounts_require_pose_and_render_audit',
                    camera=dict(self.camera_spec, path=self.camera_path,
                                T_parent_optical=self.camera_local.tolist(),
                                T_optical_usd_camera=OPTICAL_FROM_USD.tolist(),
                                optical_axes='x_right_y_down_z_forward'),
                    lidars=[dict(item['spec'], path=item['path'], T_parent_lidar=item['local'].tolist(),
                                 horizontal_fov_deg=360, beam_count=len(item['rays']),
                                 implementation='instantaneous_physx_closest_hit') for item in self.lidars])

    def sample(self, stamp, body_poses, query):
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError('invalid acquisition timestamp')
        camera_world = body_poses[self.camera_spec['parent']] @ self.camera_local
        packets = []
        for item in self.lidars:
            spec = item['spec']
            world = body_poses[spec['parent']] @ item['local']
            origin = world[:3, 3]
            rays = item['rays'] @ world[:3, :3].T
            ranges = np.full(len(rays), spec['max_range_m'])
            hit_mask = np.zeros(len(rays), bool)
            self_mask = np.zeros(len(rays), bool)
            for i, ray in enumerate(rays):
                # Start at the optical origin, so even self-hits inside min_range
                # block the ray. Never skip a robot hit to see through the body.
                hit = query.raycast_closest(tuple(map(float, origin)), tuple(map(float, ray)), float(spec['max_range_m']))
                if hit['hit']:
                    hit_mask[i] = True
                    ranges[i] = float(hit['distance'])
                    if not np.isfinite(ranges[i]) or not 0 <= ranges[i] <= spec['max_range_m']:
                        raise ValueError('invalid closest-hit LiDAR distance')
                    self_mask[i] = any(str(hit.get(key, '')).startswith(self.root+'/') for key in ('collision', 'rigidBody'))
            valid = hit_mask & ~self_mask & (ranges >= spec['min_range_m'])
            local_points = item['rays'][valid] * ranges[valid, None]
            world_points = local_points @ world[:3, :3].T + origin
            blocked = self_mask.reshape(-1, spec['horizontal_samples'])
            horizontal = np.isclose(spec['elevation_deg'], 0.)
            packets.append(dict(sensor=spec['name'], stamp_s=float(stamp), acquisition_duration_s=0.,
                                frame='world', T_world_lidar=world.tolist(),
                                ranges_m=ranges.tolist(), hit_mask=hit_mask.tolist(),
                                self_blocked_mask=self_mask.tolist(), valid_environment_mask=valid.tolist(),
                                points_sensor=local_points.tolist(), points_world=world_points.tolist(),
                                self_blocked_fraction=float(self_mask.mean()),
                                horizontal_self_blocked_mask=blocked[horizontal].any(axis=0).tolist() if horizontal.any() else None,
                                all_elevations_blocked_by_azimuth=blocked.all(axis=0).tolist()))
        return dict(stamp_s=float(stamp), pose_source='measured_physical_rigid_bodies',
                    T_world_camera_optical=camera_world.tolist(), lidars=packets)

    def capture_camera(self, output, render, stamp):
        """Save actual RGB and a front/behind colour probe without physics steps."""
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        output = Path(output)
        product = rep.create.render_product(self.camera_path, tuple(self.camera_spec['resolution']))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach(product)
        probe_path = str(Path(self.camera_path).parent)+'/DirectionProbe'
        try:
            for _ in range(12):
                render()
            rgb = np.asarray(annotator.get_data()).copy()
            if rgb.size == 0:
                raise RuntimeError('camera returned no RGB pixels')
            Image.fromarray(rgb).save(output/'camera_rgb.png')
            for name, z, colour in [('Front', .25, (1., 0., 0.)), ('Behind', -.25, (0., 1., 0.))]:
                path = probe_path+'/'+name
                cube = UsdGeom.Cube.Define(self.stage, path)
                cube.CreateSizeAttr(.08)
                set_usd_transform(cube.GetPrim(), transform([0., 0., z], np.eye(3)))
                material = UsdShade.Material.Define(self.stage, path+'/Material')
                shader = UsdShade.Shader.Define(self.stage, path+'/Material/Shader')
                shader.CreateIdAttr('UsdPreviewSurface')
                shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
                shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
            for _ in range(12):
                render()
            probe = np.asarray(annotator.get_data()).copy()
            Image.fromarray(probe).save(output/'camera_direction_probe.png')
            h, w = probe.shape[:2]
            centre = probe[h//2-4:h//2+4, w//2-4:w//2+4, :3].astype(float).mean(axis=(0, 1))
            result = dict(stamp_s=float(stamp), physics_advanced=False, centre_rgb=centre.tolist(),
                          forward_render_check_pass=bool(centre[0] > centre[1]+40 and centre[0] > centre[2]+40),
                          method='red cube at optical +Z; green cube at optical -Z; centre must be red')
            (output/'camera_render_check.json').write_text(json.dumps(result, indent=2)+'\n')
            return result
        finally:
            self.stage.RemovePrim(probe_path)
            annotator.detach(product)
            product.destroy()


def lab_body_poses(robot, names):
    """Read full measured body rotations; Fabric USD transforms may be stale."""
    states = robot.data.body_state_w[0].detach().cpu().numpy()
    return {name: pose_transform(states[robot.body_names.index(name), :3], states[robot.body_names.index(name), 3:7])
            for name in names}


def add_sensor_arguments(parser):
    parser.add_argument('--sensor-config', type=Path, help='Override the shared sensor mount JSON.')
    parser.add_argument('--sensor-audit', action='store_true', help='Record closest-hit LiDAR and measured TF at 2 Hz; included in run timing.')
    parser.add_argument('--sensor-rgb', action='store_true', help='After the rollout, render raw Camera RGB and a forward-direction colour probe.')


class SensorAudit:
    def __init__(self, rig, output, enabled=False):
        self.rig, self.output, self.enabled = rig, Path(output), enabled
        self.last = -float('inf')
        self.count = 0
        self.persistent = {}
        self.persistent_horizontal = {}
        (self.output/'sensor_mounts.json').write_text(json.dumps(rig.manifest(), indent=2)+'\n')
        (self.output/'sensor_rig_source.py').write_bytes(Path(__file__).read_bytes())

    def record(self, stamp, poses, label=None):
        if not self.enabled or stamp-self.last < .5-1e-9:
            return
        from omni.physx import get_physx_scene_query_interface
        packet = self.rig.sample(stamp, poses, get_physx_scene_query_interface())
        packet['pose_label'] = label
        with (self.output/'sensor_samples.jsonl').open('a') as stream:
            stream.write(json.dumps(packet)+'\n')
        for item in packet['lidars']:
            mask = np.asarray(item['all_elevations_blocked_by_azimuth'])
            self.persistent[item['sensor']] = self.persistent.get(item['sensor'], np.ones_like(mask)) & mask
            if item['horizontal_self_blocked_mask'] is not None:
                horizontal = np.asarray(item['horizontal_self_blocked_mask'])
                self.persistent_horizontal[item['sensor']] = self.persistent_horizontal.get(item['sensor'], np.ones_like(horizontal)) & horizontal
        self.count += 1
        self.last = stamp

    def finish(self, stamp, render=None):
        result = dict(schema='robot_sensor_audit/v1', samples=self.count,
                      full_motion_range_verified=False, agent_connected=False,
                      scope='Only sampled rollout poses; unsampled arm poses and low-range blind zones remain unverified.',
                      azimuth_reference='sensor frame: 0 deg +X, +90 deg +Y; full per-beam shadows in sensor_samples.jsonl',
                      persistent_all_elevations_blocked_deg={key: (-180+360*np.flatnonzero(mask)/len(mask)).tolist() for key, mask in self.persistent.items()},
                      persistent_horizontal_blocked_deg={key: (-180+360*np.flatnonzero(mask)/len(mask)).tolist() for key, mask in self.persistent_horizontal.items()},
                      camera_render_verified=False)
        if render is not None:
            try:
                check = self.rig.capture_camera(self.output, render, stamp)
                result['camera_render_verified'] = check['forward_render_check_pass']
            except Exception as exc:
                # A post-run RGB diagnostic must not discard physical measurements.
                import traceback
                result['camera_render_error'] = repr(exc)
                (self.output/'camera_render_error.txt').write_text(traceback.format_exc())
                print('SENSOR_RGB_UNVERIFIED', repr(exc), flush=True)
        (self.output/'sensor_audit.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
