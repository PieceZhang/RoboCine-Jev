#!/usr/bin/env python3
"""Record original camera and obstacle trajectories in Isaac Sim observer views.

This replays task.json and runtime_scene.json directly. The robot is displayed
at its initial URDF pose for context; this is not a controller execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', type=Path, default=PACKAGE / 'scenarios')
    parser.add_argument('--case', default='all', help='all or comma-separated case names')
    parser.add_argument('--output', type=Path, help='defaults to scenarios/video')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--preview', action='store_true', help='only save t=0,4,8 preview PNGs')
    args = parser.parse_args()
    args.scenarios = args.scenarios.resolve()
    args.output = (args.output or args.scenarios / 'video').resolve()
    catalog = json.loads((args.scenarios / 'catalog.json').read_text())
    args.names = catalog['cases'] if args.case == 'all' else args.case.split(',')
    if not args.names or set(args.names) - set(catalog['cases']):
        parser.error('unknown scenario')
    if args.fps <= 0 or args.width < 960 or args.height < 600:
        parser.error('fps must be positive; minimum resolution is 960 x 600')
    if args.width % 2 or args.height % 2:
        parser.error('H.264 output requires even dimensions')
    args.output.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an earlier recording silently.
    if not args.preview:
        for name in args.names:
            if (args.output / f'{name}_third_person.mp4').exists():
                parser.error(f'video already exists: {name}_third_person.mp4')
    return args


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_robot(stage, robot_path, urdf, fixture):
    """Author each URDF link's initial FK; leave the simulation clock paused."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import Gf, UsdGeom, UsdPhysics

    document = ET.parse(urdf).getroot()
    scope = stage.GetPrimAtPath(robot_path).GetParent().GetPath()
    children, child_links = {}, set()
    for joint in document.findall('joint'):
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        child_links.add(child)
        origin, axis = joint.find('origin'), joint.find('axis')
        xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        axis = np.fromstring(axis.get('xyz') if axis is not None else '1 0 0', sep=' ')
        children.setdefault(parent, []).append((child, joint.get('name'), xyz,
            Rotation.from_euler('xyz', rpy), axis, joint.get('type') != 'fixed'))
    links = {link.get('name') for link in document.findall('link')}
    root, = links - child_links
    state = fixture['initial_state']
    angles = {f'joint{i}': state[i + 2] for i in range(1, 7)}
    poses = {}

    def visit(name, p, r):
        prim = stage.GetPrimAtPath(scope.AppendChild(name))
        if not prim:
            raise RuntimeError(f'missing robot link {name}')
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        for key in tuple(prim.GetPropertyNames()):
            if key.startswith('xformOp:'):
                prim.RemoveProperty(key)
        xform.SetResetXformStack(True)
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
        q = r.as_quat()
        xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        poses[name] = p
        for child, joint, xyz, rotation, axis, moving in children.get(name, []):
            cr = r * rotation
            if moving:
                cr *= Rotation.from_rotvec(axis * angles.get(joint, 0.))
            visit(child, p + r.apply(xyz), cr)

    visit(root, np.array([state[0], state[1], fixture['base_z']]), Rotation.from_euler('z', state[2]))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    return poses['camera_link'].tolist()


def make_curve(stage, path, points, color, width):
    from pxr import Gf, UsdGeom, Vt
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr('linear')
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    curve.CreateWidthsAttr([width])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return curve


def make_camera(stage, path, eye, target, width, height, focal):
    from pxr import Gf, UsdGeom
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateVerticalApertureAttr(36. * height / width)
    camera.CreateFocalLengthAttr(focal)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)),
        Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1))
    camera.AddTransformOp().Set(matrix.GetInverse())
    return str(camera.GetPath()), dict(eye=list(map(float, eye)), target=list(map(float, target)), focal_mm=focal)


def caption(rgb, detail, name, reference, runtime, t, args):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 18)
    labels = dict(orbit='环绕', movein='推进', truck_left='左移', truck_right='右移',
                  pedestal_up='升高', pedestal_down='降低')
    frame = Image.new('RGB', (args.width, args.height), (17, 25, 36))
    frame.paste(Image.fromarray(rgb), (0, 56))
    draw = ImageDraw.Draw(frame)
    draw.text((20, 9), f'{labels[name]}  {name}  |  Isaac Sim 第三人称轨迹回放', font=font, fill=(238, 244, 252))
    draw.text((args.width - 210, 16), f't = {t:05.2f} / {reference.duration:.1f} s', font=small, fill=(238, 244, 252))
    iw, ih = detail.shape[1], detail.shape[0]
    ix, iy = args.width - iw - 18, args.height - ih - 108
    draw.rounded_rectangle((ix - 3, iy - 3, ix + iw + 3, iy + ih + 30), radius=5, fill=(17, 25, 36), outline=(40, 219, 238), width=2)
    frame.paste(Image.fromarray(detail), (ix, iy))
    draw.text((ix + 8, iy + ih + 2), '参考相机局部 · 固定第三人称', font=small, fill=(55, 225, 244))
    y = args.height - 66
    draw.line((20, y + 11, 55, y + 11), fill=(30, 225, 246), width=4)
    draw.text((64, y - 3), '参考相机路径', font=small, fill=(55, 225, 244))
    draw.line((260, y + 11, 295, y + 11), fill=(255, 110, 58), width=4)
    draw.text((304, y - 3), '障碍物路径', font=small, fill=(255, 130, 80))
    draw.text((510, y - 3), '蓝色：主体  |  机器人：初始姿态示意', font=small, fill=(200, 211, 228))
    dp = reference.sample(t)[0] - reference.sample(0.)[0]
    velocity = runtime.obstacles(t)[0]['velocity']
    draw.text((20, y + 29), f'相机位移 ΔXYZ = ({dp[0]:+.3f}, {dp[1]:+.3f}, {dp[2]:+.3f}) m', font=small, fill=(210, 226, 238))
    draw.text((690, y + 29), f'障碍物速度 = {np.linalg.norm(velocity):.2f} m/s   |   原始轨迹 · 1×', font=small, fill=(210, 226, 238))
    return np.asarray(frame).copy()


def record_case(args, app, name, urdf):
    import numpy as np
    from scipy.spatial.transform import Rotation
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom, UsdLux
    from types import SimpleNamespace
    import jvs_isaac_sim as backend
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.scenarios import RuntimeScene

    task_path = args.scenarios / name / 'task.json'
    scene_path = args.scenarios / name / 'runtime_scene.json'
    hashes = {str(p): digest(p) for p in (task_path, scene_path)}
    runtime = RuntimeScene(scene_path)
    fixture = runtime.document
    reference = OriginalReference(task_path, fixture['subject']['position'])
    stage = backend.prepare_stage(SimpleNamespace(stage='', add_ground_plane=True, obstacles=''), app)
    robot_path = backend.import_robot(urdf)
    app.update()
    backend.isolate_urdf_templates(stage, robot_path)
    robot_camera = static_robot(stage, robot_path, urdf, fixture)
    if not np.allclose(robot_camera, reference.sample(0.)[0], atol=1e-5):
        raise RuntimeError('initial URDF camera does not match the reference start')
    dome = UsdLux.DomeLight.Define(stage, '/World/SoftFill')
    dome.CreateIntensityAttr(800.)
    wall = UsdGeom.Cube.Define(stage, '/World/BackWall')
    wall.CreateSizeAttr(1.)
    wall.AddTranslateOp().Set(Gf.Vec3d(3.5, 0, .9))
    wall.AddScaleOp().Set(Gf.Vec3f(.08, 5, 1.8))
    wall.CreateDisplayColorAttr([Gf.Vec3f(.40, .45, .48)])
    subject = fixture['subject']
    cylinder = UsdGeom.Cylinder.Define(stage, '/World/BlueSubject')
    cylinder.CreateAxisAttr('Z')
    cylinder.CreateRadiusAttr(subject['radius'])
    cylinder.CreateHeightAttr(subject['height'])
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*subject['position']))
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(.025, .24, .85)])
    top = UsdGeom.Sphere.Define(stage, '/World/SubjectTop')
    top.CreateRadiusAttr(.07)
    top.AddTranslateOp().Set(Gf.Vec3d(*map(float, np.array(subject['position']) + [0, 0, subject['height'] / 2])))
    top.CreateDisplayColorAttr([Gf.Vec3f(.95, .68, .03)])
    times = np.linspace(0., reference.duration, 241)
    positions = np.array([reference.sample(t)[0] for t in times])
    cyan, orange = (.015, .9, 1.), (1., .25, .04)
    make_curve(stage, '/World/ReferencePath', positions, cyan, .009)
    obstacle_ops = {}
    for obj in runtime.obstacles(0.):
        ball = UsdGeom.Sphere.Define(stage, '/World/Obstacle_' + obj['id'])
        ball.CreateRadiusAttr(obj['radius'])
        ball.CreateDisplayColorAttr([Gf.Vec3f(*orange)])
        obstacle_ops[obj['id']] = ball.AddTranslateOp()
        path = np.array([next(o['center'] for o in runtime.obstacles(t) if o['id'] == obj['id']) for t in times])
        make_curve(stage, '/World/ObstaclePath_' + obj['id'], path, orange, .007)
    # The cyan camera is an independent reference marker, not the URDF camera.
    marker = UsdGeom.Xform.Define(stage, '/World/ReferenceCameraMarker')
    translation, orientation = marker.AddTranslateOp(), marker.AddOrientOp()
    body = UsdGeom.Cube.Define(stage, '/World/ReferenceCameraMarker/Body')
    body.CreateSizeAttr(1.)
    body.AddScaleOp().Set(Gf.Vec3f(.07, .065, .045))
    body.CreateDisplayColorAttr([Gf.Vec3f(*cyan)])
    lens = UsdGeom.Cylinder.Define(stage, '/World/ReferenceCameraMarker/Lens')
    lens.CreateAxisAttr('X')
    lens.CreateRadiusAttr(.024)
    lens.CreateHeightAttr(.035)
    lens.AddTranslateOp().Set(Gf.Vec3d(.048, 0, 0))
    lens.CreateDisplayColorAttr([Gf.Vec3f(.015, .10, .13)])
    corners = [[.27, y, z] for y, z in ((-.11, -.07), (.11, -.07), (.11, .07), (-.11, .07))]
    for i, corner in enumerate(corners):
        make_curve(stage, f'/World/ReferenceCameraMarker/Frustum{i}', [[.065, 0, 0], corner, corners[(i + 1) % 4]], cyan, .003)
    aim = make_curve(stage, '/World/AimAtSubject', [positions[0], subject['position']], (.26, .68, .70), .002)
    # A metric ground grid makes lateral motion and scene scale readable.
    for i, value in enumerate(np.arange(-1., 2.51, .25)):
        make_curve(stage, f'/World/GridX{i}', [[value, -1, .003], [value, 1, .003]], (.46, .49, .52), .002)
    for i, value in enumerate(np.arange(-1., 1.01, .25)):
        make_curve(stage, f'/World/GridY{i}', [[-.5, value, .003], [2.5, value, .003]], (.46, .49, .52), .002)
    render_height = args.height - 128
    main_camera, main_pose = make_camera(stage, '/World/Observer', [2.7, -3.5, 2.25], [.70, 0., .53], args.width, render_height, 44.)
    iw, ih = 320, 200
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    detail_camera, detail_pose = make_camera(stage, '/World/ReferenceDetail', center + [.6, -1.35, .52], center, iw, ih, 65.)
    products, annotators = [], []
    encoder = None
    frame_records = []
    output = args.output / f'{name}_third_person.mp4'
    temporary = output.with_suffix('.partial.mp4')
    try:
        for camera, size in ((main_camera, (args.width, render_height)), (detail_camera, (iw, ih))):
            product = rep.create.render_product(camera, size)
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach(product)
            products.append(product)
            annotators.append(annotator)
        frame_count = round(reference.duration * args.fps)
        render_times = [0., reference.duration / 2, reference.duration] if args.preview else np.linspace(0., reference.duration, frame_count)
        if not args.preview:
            encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{args.width}x{args.height}', '-r', str(args.fps),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
        for index, t in enumerate(render_times):
            p, r, _ = reference.sample(t)
            q = Rotation.from_matrix(r).as_quat()
            translation.Set(Gf.Vec3d(*map(float, p)))
            orientation.Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
            obstacles = runtime.obstacles(t)
            for obj in obstacles:
                obstacle_ops[obj['id']].Set(Gf.Vec3d(*map(float, obj['center'])))
            aim.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)), Gf.Vec3f(*subject['position'])])
            if index == 0:
                for _ in range(20):
                    app.update()
            rep.orchestrator.step(rt_subframes=8 if index == 0 else 2, delta_time=0., pause_timeline=True)
            images = [np.asarray(a.get_data())[:, :, :3].copy() for a in annotators]
            if images[0].shape != (render_height, args.width, 3) or images[1].shape != (ih, iw, 3):
                raise RuntimeError('Isaac RGB annotators returned incomplete frames')
            frame = caption(*images, name, reference, runtime, float(t), args)
            if args.preview or index in (0, frame_count // 2, frame_count - 1):
                from PIL import Image
                Image.fromarray(frame).save(args.output / f'{name}_third_person_{float(t):05.2f}.png')
            if encoder is not None:
                encoder.stdin.write(frame.tobytes())
            frame_records.append(dict(frame=index, trajectory_time_s=float(t), camera_position=p.tolist(),
                camera_quaternion_xyzw=q.tolist(), obstacles=[dict(id=o['id'], center=o['center'].tolist(),
                velocity=o['velocity'].tolist(), radius=o['radius']) for o in obstacles]))
            if index % args.fps == 0:
                print(f'RECORD {name}: {index + 1}/{len(render_times)} frames, trajectory t={t:.2f}s', flush=True)
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('ffmpeg encoding failed')
            encoder = None
            temporary.rename(output)
        if any(digest(p) != h for p, h in hashes.items()):
            raise RuntimeError('source scenario changed during recording')
        metadata = dict(case=name, renderer='Isaac Sim RTX RaytracedLighting', view='fixed third person with fixed third-person detail inset',
            source='original camera reference and runtime obstacle trajectories', robot='static initial URDF pose; no controller execution',
            camera_interpolation='OriginalReference CubicSpline + RotationSpline',
            obstacle_interpolation='RuntimeScene CubicHermiteSpline',
            fps=args.fps, width=args.width, height=args.height, duration_s=reference.duration,
            trajectory_endpoints_included=True, frame_count=len(frame_records), source_sha256=hashes,
            script_sha256=digest(__file__), observer_camera=main_pose, detail_camera=detail_pose,
            initial_robot_camera_position=robot_camera, frames=frame_records)
        suffix = '_preview.json' if args.preview else '_recording.json'
        (args.output / f'{name}_third_person{suffix}').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        print(f'COMPLETE {name}: {output if not args.preview else args.output}', flush=True)
    finally:
        if encoder is not None:
            if not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait()
        for annotator in annotators:
            annotator.detach()
        for product in products:
            product.destroy()


def main():
    args = arguments()
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    os.environ.setdefault('ACCEPT_EULA', 'Y')
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, renderer='RaytracedLighting', width=args.width,
        height=args.height, anti_aliasing=3, multi_gpu=False, limit_cpu_threads=8))
    try:
        import omni.kit.app
        import omni.timeline
        import omni.replicator.core as rep
        import jvs_isaac_sim as backend
        from types import SimpleNamespace
        omni.timeline.get_timeline_interface().stop()
        rep.orchestrator.set_capture_on_play(False)
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
        for _ in range(3):
            app.update()
        with tempfile.TemporaryDirectory(prefix='jev_trajectory_recording_') as folder:
            urdf = Path(folder) / 'robot.urdf'
            backend.expand_xacro(SimpleNamespace(xacro_executable='/opt/ros/humble/bin/xacro',
                xacro=str(PACKAGE / 'urdf/jev_agent_wbc.urdf.xacro')), urdf)
            for name in args.names:
                record_case(args, app, name, urdf)
    finally:
        app.close()


if __name__ == '__main__':
    main()
