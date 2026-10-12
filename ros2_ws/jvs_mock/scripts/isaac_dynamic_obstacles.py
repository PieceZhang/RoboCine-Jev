"""Isaac-only kinematic obstacle authoring and physical-time playback."""
import csv
import importlib.util
import json
import os
from pathlib import Path


def load_scene_type():
    # The Isaac Python ABI must not import system ROS. Load only this pure
    # numpy/scipy module from the source tree or the installed ROS package.
    candidates = [Path(__file__).resolve().parents[1] / 'jvs_mock/dynamic_scene.py']
    candidates += [Path(path) / 'jvs_mock/dynamic_scene.py'
                   for path in os.environ.get('JVS_XACRO_PYTHONPATH', '').split(os.pathsep) if path]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location('jvs_dynamic_scene_data', candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.DynamicScene
    raise RuntimeError('cannot locate the installed dynamic_scene module')


class IsaacDynamicObstacles:
    def __init__(self, stage, path, spawn, audit_file=''):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_interface, get_physx_simulation_interface

        self.scene = load_scene_type()(path)
        self.spawn = spawn
        self.time = 0.
        self.physics_steps = 0
        self.last_contact_audit = -1.
        self.epoch = None
        self.last_audit = -1.
        self.origin = np.array([spawn['x'], -spawn['z'], spawn['height']])
        self.rotation = Rotation.from_euler('z', spawn['yaw'] - np.pi / 2).as_matrix()
        self.ops = {}
        self.rigid_view = None
        self.np = np
        self.Gf = Gf
        self.contact_events = []
        self.static_contact_events = []
        self.obstacle_static_contact_events = []
        self.audit_file = audit_file
        self.PhysicsSchemaTools = PhysicsSchemaTools
        for obstacle in self.scene.obstacles:
            name, model = obstacle['name'], obstacle['model']
            root = UsdGeom.Xform.Define(stage, '/World/DynamicObstacles/' + name)
            self.ops[name] = root.AddTranslateOp()
            root.AddOrientOp().Set(Gf.Quatf(Rotation.from_matrix(self.rotation).as_quat()[3],
                Gf.Vec3f(*Rotation.from_matrix(self.rotation).as_quat()[:3])))
            body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            body.CreateKinematicEnabledAttr(True)
            shape_path = root.GetPath().AppendChild('Model')
            if model['type'] == 'box':
                shape = UsdGeom.Cube.Define(stage, shape_path)
                shape.CreateSizeAttr(1.)
                shape.AddScaleOp().Set(Gf.Vec3f(*model['size']))
            elif model['type'] == 'cylinder':
                shape = UsdGeom.Cylinder.Define(stage, shape_path)
                shape.CreateRadiusAttr(model['radius'])
                shape.CreateHeightAttr(model['height'])
                shape.CreateAxisAttr(model.get('axis', 'z').upper())
            else:
                shape = UsdGeom.Sphere.Define(stage, shape_path)
                shape.CreateRadiusAttr(model['radius'])
            shape.CreateDisplayColorAttr([Gf.Vec3f(*model.get('color', [.85, .3, .12]))])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        from isaac_dynamic_contacts import configure_contact_audit
        self.contact_configuration = configure_contact_audit(stage)
        self.file = None
        self.writer = None
        if audit_file:
            target = Path(audit_file).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            self.file = target.open('w')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['simulation_time_s', 'scene_time_s', 'obstacle', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        self.update(0.)
        self.subscription = get_physx_interface().subscribe_physics_step_events(self.on_step)
        self.contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.on_contacts)
        self.save_contacts()

    def attach(self, node):
        from isaacsim.core.prims import RigidPrim
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        self.rigid_view = RigidPrim('/World/DynamicObstacles/*', name='dynamic_obstacle_bodies',
                                    reset_xform_properties=False)
        self.rigid_view.initialize()
        self.body_names = [str(path).rsplit('/', 1)[-1] for path in self.rigid_view.prim_paths]
        self.targets = self.rigid_view._physics_view.get_transforms().copy()
        self.indices = self.np.arange(len(self.body_names), dtype=self.np.uint32)
        self.start_subscription = node.create_subscription(String, '/jvs/dynamic_scene/start', self.start, 1)
        self.reset_service = node.create_service(Trigger, '/jvs/reset_dynamic_scene', self.reset)

    def start(self, message):
        epoch = float(json.loads(message.data)['epoch_s'])
        if not __import__('math').isfinite(epoch):
            raise ValueError('scene epoch must be finite')
        self.epoch = epoch
        print(f'JVS dynamic scene started at simulation t={self.epoch:.3f}s', flush=True)

    def reset(self, request, response):
        del request
        self.epoch = None
        self.update(0.)
        response.success = True
        response.message = 'Dynamic obstacles at t=0; awaiting controller start'
        return response

    def update(self, t):
        commanded = {}
        for obstacle in self.scene.obstacles:
            name = obstacle['name']
            position, velocity = self.scene.sample(name, t)
            local = self.scene.to_initial_base(position, self.spawn)
            world = self.origin + self.rotation @ local
            commanded[name] = (world, velocity)
            # Kinematic poses are application inputs. Publish the same target
            # to USD/Hydra for RTX, and to PhysX for contact/velocity integration.
            # Tensor teleportation alone leaves the RTX geometry at its spawn.
            self.ops[name].Set(self.Gf.Vec3d(*world))
        if self.rigid_view is not None:
            self.targets[:, :3] = self.np.array([commanded[name][0] for name in self.body_names])
            self.rigid_view._physics_view.set_kinematic_targets(self.targets, self.indices)
            # Reading the tensor pose is needed only for recorded truth. Avoid
            # a GPU/CPU synchronization at all 1000 physics steps per second.
            if self.writer is None or self.time - self.last_audit < .019:
                return
            actual, _ = self.rigid_view.get_world_poses()
            measured = {name: self.rotation.T @ (position - self.origin)
                        for name, position in zip(self.body_names, actual)}
        else:
            measured = {name: self.rotation.T @ (world - self.origin) for name, (world, _) in commanded.items()}
        for obstacle in self.scene.obstacles:
            name = obstacle['name']
            local, velocity = measured[name], commanded[name][1]
            if self.writer is not None and self.time - self.last_audit >= .019:
                if self.scene.frame == 'scene_y_up':
                    velocity = self.rotation.T @ (velocity[[0, 2, 1]] * [1, -1, 1])
                self.writer.writerow([self.time, t, name, *local, *velocity])
        if self.time - self.last_audit >= .019:
            self.last_audit = self.time
            if self.file:
                self.file.flush()

    def on_step(self, dt):
        self.time += float(dt)
        self.physics_steps += 1
        self.update(0. if self.epoch is None else max(0., self.time - self.epoch))
        if self.time - self.last_contact_audit >= 1.:
            self.save_contacts()

    def on_contacts(self, headers, data):
        for header in headers:
            paths = [str(self.PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
            robot = any(path.startswith('/jvs_mock/') for path in paths)
            moving = any(path.startswith('/World/DynamicObstacles/') for path in paths)
            static = any(path.startswith('/World/Simview/') and '/floor/' not in path.lower()
                         and 'groundplane' not in path.lower() for path in paths)
            event = dict(time_s=self.time, actors=paths, event=str(header.type),
                         contact_points=int(header.num_contact_data))
            if robot and moving:
                self.contact_events.append(event)
            elif robot and static:
                self.static_contact_events.append(event)
            elif moving and static:
                self.obstacle_static_contact_events.append(event)
            else:
                continue
            if robot or moving:
                self.save_contacts()

    def save_contacts(self):
        if self.audit_file:
            target = Path(self.audit_file).with_name('dynamic_contacts.json')
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(enabled=True, configuration=self.contact_configuration,
                simulation_time_s=self.time,
                physics_steps=self.physics_steps, robot_obstacle_events=self.contact_events,
                robot_static_events=self.static_contact_events,
                obstacle_static_events=self.obstacle_static_contact_events), indent=2) + '\n')
            temporary.replace(target)
            self.last_contact_audit = self.time

    def close(self):
        self.subscription = None
        self.contact_subscription = None
        if self.file:
            self.file.close()
        self.save_contacts()
