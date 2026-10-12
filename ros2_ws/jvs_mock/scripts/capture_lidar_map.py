#!/usr/bin/env python3
"""Capture actual ROS RTX scans in a private Kit session at static viewpoints.

Every scan records simulator ground-truth world extrinsics as a diagnostic
reference. Multi-view relocation is a visibility test, not autonomous mapping,
collision-checked navigation or a SLAM implementation. Two LiDARs by default.
Run via jvs_isaac_bootstrap.py to isolate the Isaac/ROS Python ABI.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from functools import partial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--xacro', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--poses', default='[[5.0,1.0,0.0]]',
                        help='JSON list of [simview x, simview z, yaw]')
    parser.add_argument('--scans', type=int, default=10)
    parser.add_argument('--lidar-mode', choices=('front', 'both'), default='both')
    parser.add_argument('--rear-only', action='store_true', help='single-sensor regression mode')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--kit-root', default='', help='private portable root; reuse only sequentially')
    parser.add_argument('--keep-environment-physics', action='store_true',
                        help='leave the original scene physics enabled (single-pose integration test)')
    args = parser.parse_args()
    if args.rear_only and args.lidar_mode != 'both':
        parser.error('--rear-only conflicts with --lidar-mode front')
    output = Path(args.output).resolve()
    if (output / 'capture.json').exists() or any(output.glob('scan_*.npz')):
        raise ValueError('output already contains scans; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    if not 1 <= domain <= 232:
        raise ValueError('set a dedicated ROS_DOMAIN_ID in [1, 232]')
    poses = json.loads(args.poses)
    if not poses or args.scans < 1 or args.timeout <= 0:
        raise ValueError('require at least one pose/scan and a positive timeout')
    if args.keep_environment_physics and len(poses) != 1:
        raise ValueError('physical integration test does not support unplanned teleports')
    import jvs_simview_isaac as simview
    import jvs_isaac_sim as jvs
    sim_args = simview.build_parser().parse_args([
        '--simview-stage', args.stage, '--xacro', args.xacro,
        '--xacro-executable', '/opt/ros/humble/bin/xacro', '--headless', 'true',
        '--activate-camera', 'false', '--physics-hz', '200',
        '--lidar-mode', args.lidar_mode,
        '--enable-front-lidar', str(not args.rear_only)])
    simview.validate_args(sim_args)
    # Kit's portable root, not only XDG variables, isolates caches and locks.
    kit_root = Path(args.kit_root).resolve() if args.kit_root else output / 'kit'
    sys.argv[:] = [sys.argv[0], '--portable-root', str(kit_root)]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, width=64, height=64, multi_gpu=False,
                             disable_viewport_updates=True, enable_motion_bvh=False,
                             limit_cpu_threads=8, extra_args=[
                                 '--/app/settings/persistent=false',
                                 '--/app/asyncRendering=false',
                                 '--/app/renderer/skipWhileMinimized=false']))
    node = None
    try:
        import numpy as np
        import omni.graph.core as og
        import omni.timeline
        jvs.enable_extensions(app)
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        from tf2_msgs.msg import TFMessage
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaacsim.core.prims import SingleArticulation
        # rclpy becomes available after the bridge extension is enabled.
        rclpy.init()
        node = rclpy.create_node('isolated_lidar_map_capture')
        with tempfile.TemporaryDirectory(prefix='urdf_', dir=output) as folder:
            urdf = Path(folder) / 'jvs_mock.urdf'
            jvs.expand_xacro(sim_args, urdf)
            stage, _source_rotation = simview.prepare_composed_stage(sim_args, app)
            disabled_physics = {'rigid_bodies': 0, 'colliders': 0}
            environment_prims = ([] if args.keep_environment_physics else
                                 Usd.PrimRange(stage.GetPrimAtPath('/World/Simview')))
            for prim in environment_prims:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                    disabled_physics['rigid_bodies'] += 1
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                    disabled_physics['colliders'] += 1
            print('ENVIRONMENT_PHYSICS ' + json.dumps(dict(keep=args.keep_environment_physics,
                                                         disabled=disabled_physics)), flush=True)
            robot_path = jvs.import_robot(urdf)
            print('PROGRESS robot imported; updating renderer', flush=True)
            app.update()
            print('PROGRESS renderer updated; configuring LiDARs', flush=True)
            sim_args.spawn_x, sim_args.spawn_y, sim_args.spawn_z, sim_args.spawn_yaw = (
                simview.simview_to_isaac_spawn(*poses[0]))
            jvs.configure_articulation(stage, robot_path, sim_args, urdf)
            jvs.configure_physics_rate(stage, sim_args.physics_hz)
            sensor_path = (jvs.create_rtx_lidar(stage, robot_path, sim_args)
                           if 'rear' in jvs.enabled_lidars(sim_args) else '')
            front_path = jvs.create_front_rtx_lidar(stage, robot_path, sim_args)
            sensor_specs = {}
            if sensor_path:
                sensor_specs['rear'] = dict(path=sensor_path, frame='lidar_link',
                    topic=sim_args.lidar_topic, config=sim_args.lidar_config,
                    min_range=.1, max_range=30.)
            if front_path:
                sensor_specs['front'] = dict(path=front_path, frame='front_lidar_link',
                    topic=sim_args.front_lidar_topic, config=sim_args.front_lidar_config,
                    min_range=1., max_range=200.)
            app.update()
            print('PROGRESS LiDAR created; configuring ROS graph', flush=True)
            jvs.create_ros_graph(robot_path, lidar_path=sensor_path, front_lidar_path=front_path)
            og.Controller.set(og.Controller.attribute('/World/JvsRosGraph/Context.inputs:useDomainIDEnvVar'), False)
            og.Controller.set(og.Controller.attribute('/World/JvsRosGraph/Context.inputs:domain_id'), domain)
            sensors = [str(p.GetPath()) for p in stage.Traverse()
                       if p.GetTypeName() == 'OmniLidar' or
                       (p.HasAttribute('cameraSensorType') and p.GetAttribute('cameraSensorType').Get() == 'lidar')]
            if set(sensors) != {s['path'] for s in sensor_specs.values()}:
                raise RuntimeError(f'unexpected LiDARs: {sensors}')
            stage.SetEndTimeCode(1000000)
            robot = stage.GetPrimAtPath(robot_path)
            tf_records = {}
            scan_records = []
            active = {'pose': -1}
            counts = {name: 0 for name in sensor_specs}
            warmup = counts.copy()

            def on_tf(message):
                for t in message.transforms:
                    tf_records[t.child_frame_id] = dict(parent=t.header.frame_id,
                        xyz=[t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
                        xyzw=[t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])

            def on_scan(message, name):
                if active['pose'] < 0:
                    return
                warmup[name] += 1
                # A shared barrier prevents the faster native front publisher
                # finishing its capture before the rear publisher is warmed up.
                if min(warmup.values()) <= 6 or counts[name] >= args.scans:
                    return
                spec = sensor_specs[name]
                if message.header.frame_id != spec['frame']:
                    raise RuntimeError(f'wrong frame for {name}: {message.header.frame_id}')
                lidar = stage.GetPrimAtPath(spec['path'])
                lidar_link = stage.GetPrimAtPath('/jvs_mock/' + spec['frame'])
                fields = {f.name: f for f in message.fields}
                dtype = np.dtype(dict(names=['x', 'y', 'z'],
                    formats=[('>' if message.is_bigendian else '<') + 'f4'] * 3,
                    offsets=[fields[k].offset for k in ['x', 'y', 'z']], itemsize=message.point_step))
                cloud = np.ndarray((message.height, message.width), dtype=dtype,
                    buffer=message.data, strides=(message.row_step, message.point_step))
                points = np.stack([cloud[k].ravel() for k in ['x', 'y', 'z']], axis=1)
                points = points[np.isfinite(points).all(axis=1)]
                if not len(points):
                    return
                transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
                world_sensor = np.array(transforms.GetLocalToWorldTransform(lidar))
                world_base = np.array(transforms.GetLocalToWorldTransform(robot))
                base_link_sensor = (np.array(transforms.GetLocalToWorldTransform(lidar_link))
                                    @ np.linalg.inv(world_base))
                boxes, box_frames = [], []
                bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
                for link in stage.GetPrimAtPath('/jvs_mock').GetChildren():
                    visuals = link.GetChild('visuals')
                    if not visuals:
                        continue
                    box = bounds.ComputeLocalBound(visuals)
                    extent = box.ComputeAlignedRange()
                    if extent.IsEmpty():
                        continue
                    boxes.append([list(extent.GetMin()), list(extent.GetMax())])
                    box_frames.append(np.array(transforms.GetLocalToWorldTransform(link)))
                index = len(scan_records)
                filename = f'scan_{index:03d}.npz'
                np.savez_compressed(output / filename, points=points, world_sensor=world_sensor,
                                    world_base=world_base, base_link_sensor=base_link_sensor,
                                    self_boxes=np.array(boxes), self_box_frames=np.array(box_frames))
                row = dict(file=filename, pose_index=active['pose'], points=len(points), sensor=name,
                    min_range=spec['min_range'], max_range=spec['max_range'],
                    frame_id=message.header.frame_id,
                    stamp=message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                    world_sensor_xyz=world_sensor[3, :3].tolist())
                scan_records.append(row)
                counts[name] += 1
                print('CAPTURE ' + json.dumps(row), flush=True)

            for name, spec in sensor_specs.items():
                node.create_subscription(PointCloud2, spec['topic'],
                                         partial(on_scan, name=name), qos_profile_sensor_data)
            node.create_subscription(TFMessage, '/tf', on_tf, qos_profile_sensor_data)
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            print('PROGRESS timeline started; warming up', flush=True)
            for _ in range(60):
                app.update()
                rclpy.spin_once(node, timeout_sec=0)
            articulation = SingleArticulation(prim_path=robot_path)
            articulation.initialize()
            for i, pose in enumerate(poses):
                if i:
                    x, y, z, yaw = simview.simview_to_isaac_spawn(*pose)
                    articulation.set_world_pose(np.array([x, y, z]),
                        np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
                    articulation.set_linear_velocity(np.zeros(3))
                    articulation.set_angular_velocity(np.zeros(3))
                active.update(pose=i)
                counts.update({name: 0 for name in sensor_specs})
                warmup.update(counts)
                deadline = time.monotonic() + args.timeout
                while min(counts.values()) < args.scans and time.monotonic() < deadline:
                    app.update()
                    rclpy.spin_once(node, timeout_sec=0)
                    time.sleep(0.025)  # avoid monopolizing the shared GPU/CPU
                if min(counts.values()) < args.scans:
                    raise RuntimeError(f'pose {i}: timed out after {counts} scans')
            active['pose'] = -1
            timeline.stop()
            metadata = dict(stage=str(Path(args.stage).resolve()), domain_id=domain,
                sensor_paths=sensors, config=(sim_args.lidar_config if sensor_path else
                    sim_args.front_lidar_config), poses=poses, scans=scan_records,
                sensor_specs=sensor_specs,
                transforms=tf_records, physics_hz=sim_args.physics_hz,
                disabled_environment_physics=disabled_physics,
                keep_environment_physics=args.keep_environment_physics,
                note=('fixed viewpoints; original environment physics retained' if args.keep_environment_physics else
                      'static viewpoint audit; environment collision/dynamics disabled, visual geometry and authored animations unchanged')
                      + '; registration uses simulator world extrinsics, not SLAM')
            (output / 'capture.json').write_text(json.dumps(metadata, indent=2))
            print('CAPTURE_COMPLETE ' + str(output), flush=True)
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()
        app.close()


if __name__ == '__main__':
    main()
