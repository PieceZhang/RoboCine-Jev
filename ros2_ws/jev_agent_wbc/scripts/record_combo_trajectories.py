#!/usr/bin/env python3
"""Render incomplete combo designs without inventing their missing transitions.

This is a scene-design preview, not an Agent run or PhysX execution. Each main
action uses an independent offline nominal WBC rollout. The robot and reference
marker are hidden during gaps; obstacle time continues. No transition path,
robot motion, or global interpolator is constructed across those gaps.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from record_scenario_trajectories import animated_robot, make_curve, make_camera, digest


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', type=Path, default=PACKAGE/'scenarios')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', default='all')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--prepare', action='store_true', help='offline per-action nominal motion only')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    args.scenarios = args.scenarios.resolve()
    args.output = (args.output or args.scenarios/'video_combo').resolve()
    args.catalog = json.loads((args.scenarios/'combined_catalog.json').read_text())
    args.names = args.catalog['cases'] if args.case == 'all' else args.case.split(',')
    if set(args.names)-set(args.catalog['cases']): parser.error('unknown combo')
    if args.fps <= 0 or args.width != 1280 or args.height != 720:
        parser.error('this matching layout requires 1280x720 and positive fps')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.prepare and not args.preview:
        for name in args.names:
            if (args.output/f'{name}_third_person.mp4').exists():
                parser.error(f'refusing to overwrite existing film: {name}')
    return args


class ComboReference:
    """Independent valid action references; sample() returns None inside gaps."""
    def __init__(self, path, subject):
        from jev_agent_wbc.reference import OriginalReference
        self.task = json.loads(Path(path).read_text())
        self.duration = self.task['reference_camera_trajectory']['samples'][-1]['t']
        self.actions = []
        for phase in self.task['shot_phase']:
            local = deepcopy(self.task)
            a, b = phase['start_s'], phase['end_s']
            local['shot_phase'] = [dict(phase, start_s=0., end_s=b-a)]
            local['reference_camera_trajectory']['samples'] = [dict(s, t=round(s['t']-a, 8))
                for s in self.task['reference_camera_trajectory']['samples'] if a <= s['t'] <= b]
            self.actions.append((phase, OriginalReference(local, subject)))

    def action(self, t):
        return next(((i, p, r) for i, (p, r) in enumerate(self.actions)
                     if p['start_s'] <= t <= p['end_s']), None)

    def sample(self, t):
        active = self.action(t)
        return None if active is None else active[2].sample(t-active[1]['start_s'])


def prepare(args):
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from jev_agent_wbc.nominal import nominal_rollout
    from jev_agent_wbc.whole_body_qp import whole_body_kinematics
    from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER
    audits = []
    for name in args.names:
        folder = args.scenarios/name
        fixture = json.loads((folder/'runtime_scene.json').read_text())
        reference = ComboReference(folder/'task.json', fixture['subject']['position'])
        seed = np.array(fixture['initial_state'], float)
        actions = []
        for i, (phase, ref) in enumerate(reference.actions):
            target_p, target_r, _ = ref.sample(0.)
            if i:
                # Static IK placement for a separately authored shot, never a
                # path across the gap. Its relocation is hidden and disclosed.
                camera_yaw = float(np.arctan2(target_r[1, 0], target_r[0, 0]))
                offset = {'truck_left': 1.1, 'truck_right': -1.1, 'orbit': .9}.get(phase['motion'], 0.)
                seed = np.array([0., 0., camera_yaw+offset, -offset, -.9, 2.1, 0., -1.2, 0.])
                seed_p, _, _ = whole_body_kinematics(seed, fixture['base_z'])
                seed[:2] += target_p[:2]-seed_p[:2]
                anchor = seed.copy()
                def residual(state):
                    p, r, _ = whole_body_kinematics(state, fixture['base_z'])
                    return np.r_[20*(p-target_p), 5*Rotation.from_matrix(r@target_r.T).as_rotvec(), .001*(state-anchor)]
                fit = least_squares(residual, seed, bounds=(np.r_[[-10.,-10.,-20.], JOINT_LOWER+.04],
                    np.r_[[10.,10.,20.], JOINT_UPPER-.04]), max_nfev=500,
                    ftol=1e-11, xtol=1e-11, gtol=1e-11)
                seed = fit.x
            p, r, _ = whole_body_kinematics(seed, fixture['base_z'])
            assert np.linalg.norm(p-target_p) < 1e-5
            times, states, failure = nominal_rollout(ref, seed, fixture['base_z'])
            if failure: raise RuntimeError(f'{name}/{phase["name"]}: {failure}')
            errors, angular = [], []
            for t, state in zip(times, states):
                p, r, _ = whole_body_kinematics(state, fixture['base_z'])
                rp, rr, _ = ref.sample(t)
                errors.append(float(np.linalg.norm(p-rp)))
                angular.append(float(Rotation.from_matrix(r@rr.T).magnitude()))
            # A visual nominal robot illustration is distinct from the exact
            # cyan reference marker. Record the errors; this is no safety gate.
            if max(errors) > .05 or max(angular) > .15:
                raise RuntimeError(f'{name}/{phase["name"]}: nominal tracking inaccurate: {max(errors):.4f} m, {max(angular):.4f} rad')
            actions.append(dict(phase=phase, times_s=(times+phase['start_s']).tolist(),
                states=states.tolist(), max_camera_error_m=max(errors), max_orientation_error_rad=max(angular),
                start_state_source='fixture' if i == 0 else 'independent static IK; no gap motion'))
            seed = states[-1]
            print(f'PREPARED {name}/{phase["name"]}: max camera error {max(errors):.5f} m', flush=True)
        cache = dict(case=name, actions=actions, source_sha256={str(p):digest(p)
            for p in (folder/'task.json', folder/'runtime_scene.json')},
            robot='independent offline nominal WBC per main action; no avoidance; not a closed-loop Agent or PhysX run',
            gap_policy='no robot states or camera samples in gaps; hidden relocation between independent actions',
            presentation_playback_rate=1.)
        (args.output/f'{name}_nominal.json').write_text(json.dumps(cache, indent=2)+'\n')
        audits.append(dict(case=name, actions=[{k:v for k,v in a.items() if k not in ('times_s','states')} for a in actions]))
    (args.output/'nominal_audit.json').write_text(json.dumps(audits, indent=2)+'\n')


def caption(rgb, detail, name, reference, runtime, t, args, clearances=None):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 18)
    mini = ImageFont.truetype(font_path, 16)
    labels = dict(orbit='环绕', movein='推进', truck_left='左移', truck_right='右移', pedestal_up='升高', pedestal_down='下降')
    design = next(c for c in args.catalog['scenarios'] if c['name']==name)
    title = design['title']
    active = reference.action(t)
    frame = Image.new('RGB', (args.width, args.height), (17,25,36))
    frame.paste(Image.fromarray(rgb), (0,56))
    draw = ImageDraw.Draw(frame)
    draw.text((20,9), f'{title}  |  Isaac Sim 第三人称轨迹回放', font=font, fill=(238,244,252))
    draw.text((1060,16), f't = {t:05.2f} / {reference.duration:.1f} s', font=small, fill=(238,244,252))
    iw,ih=detail.shape[1],detail.shape[0];ix,iy=args.width-iw-18,args.height-ih-108
    draw.rounded_rectangle((ix-3,iy-3,ix+iw+3,iy+ih+30),radius=5,fill=(17,25,36),outline=(40,219,238),width=2)
    frame.paste(Image.fromarray(detail),(ix,iy))
    draw.text((ix+8,iy+ih+2),'参考相机局部 · 固定第三人称',font=small,fill=(55,225,244))
    if active:
        status=f'动作 {active[0]+1}/{len(reference.actions)} · {labels[active[1]["motion"]]}'
    else:
        status='衔接留空 · 待 Agent 设计 arc'
    draw.rounded_rectangle((20,70,450 if not active else 300,110),radius=6,fill=(17,25,36))
    draw.text((32,77),status,font=small,fill=(255,202,92) if not active else (230,244,250))
    if 'interference_audit' in design:
        audit = design['interference_audit']
        heights = ' / '.join(f"{o['height_label']} {o['center_height_m']:.2f}m" for o in audit['obstacles'])
        draw.rounded_rectangle((20,119,650,157), radius=6, fill=(17,25,36))
        draw.text((32,125), f"{design['difficulty_zh']} | {heights} | 干涉已验证 3/3", font=small, fill=(210,226,238))
        hits=[] if clearances is None else [o['height_label'] for o in audit['obstacles'] if clearances[o['obstacle_id']] < -.003]
        status = ('当前干涉：'+'、'.join(hits)+'（无避障回放）' if hits else
                  '当前无干涉' if clearances is not None else '衔接留空：无机器人位姿，不判定干涉')
        draw.rounded_rectangle((20,163,600,201), radius=6, fill=(17,25,36))
        draw.text((32,169),status,font=small,fill=(255,140,100) if hits else (210,226,238))
    elif 'clearance_audit' in design:
        clearance = design['clearance_audit']['nominal_main_min_surface_clearance_m']
        draw.rounded_rectangle((20,119,560,157), radius=6, fill=(17,25,36))
        draw.text((32,125), f"{design['difficulty_zh']} | 障碍物 {design['obstacle_count']} 个 | 名义整机最小净距 {clearance:.2f} m",
                  font=small, fill=(210,226,238))
    # Timeline explicitly leaves the transition sections empty/dashed.
    y=617; scale=(ix-40)/reference.duration
    for phase, _ in reference.actions:
        x1,x2=20+phase['start_s']*scale,20+phase['end_s']*scale
        draw.rectangle((x1,y,x2,y+10),fill=(30,170,190))
    x=20+t*scale;draw.line((x,y-5,x,y+15),fill=(255,255,255),width=2)
    y=args.height-66
    draw.line((20,y+11,55,y+11),fill=(30,225,246),width=4)
    draw.text((64,y-3),'主动作路径（衔接留空）',font=small,fill=(55,225,244))
    draw.line((310,y+11,345,y+11),fill=(255,110,58),width=4)
    draw.text((354,y-3),'障碍物：低黄 / 中橙 / 高紫',font=small,fill=(255,170,100))
    draw.text((760,y-3),'逐段无避障运动示意',font=small,fill=(200,211,228))
    if active:
        dp=reference.sample(t)[0]-reference.sample(0.)[0]
        text=f'相机位移 ΔXYZ = ({dp[0]:+.3f}, {dp[1]:+.3f}, {dp[2]:+.3f}) m'
    else:
        text='无相机/机器人轨迹；障碍物按场景时间继续运动'
    draw.text((20,y+29),text,font=mini,fill=(210,226,238))
    speed=max(np.linalg.norm(o['velocity']) for o in runtime.obstacles(t))
    draw.text((670,y+29),f'障碍物最大速度 {speed:.2f} m/s | 场景设计预览 · 1×',font=mini,fill=(210,226,238))
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
    from jev_agent_wbc.nominal import nominal_rollout
    from jev_agent_wbc.kc_svsdf import CollisionGeometry

    task_path = args.scenarios / name / 'task.json'
    scene_path = args.scenarios / name / 'runtime_scene.json'
    hashes = {str(p): digest(p) for p in (task_path, scene_path)}
    runtime = RuntimeScene(scene_path)
    fixture = runtime.document
    collision_geometry = CollisionGeometry()
    reference = ComboReference(task_path, fixture['subject']['position'])
    cache_path = args.output/f'{name}_nominal.json'
    cache = json.loads(cache_path.read_text())
    if cache['source_sha256'] != hashes:
        raise RuntimeError('stale nominal cache; rerun --prepare')
    hashes[str(cache_path)] = digest(cache_path)
    hashes[str(args.scenarios/'combined_catalog.json')] = digest(args.scenarios/'combined_catalog.json')
    stage = backend.prepare_stage(SimpleNamespace(stage='', add_ground_plane=True, obstacles=''), app)
    robot_path = backend.import_robot(urdf)
    app.update()
    backend.isolate_urdf_templates(stage, robot_path)
    robot_camera, set_robot_state = animated_robot(stage, robot_path, urdf, fixture)
    robot_scope = stage.GetPrimAtPath(robot_path).GetParent().GetPath()
    robot_images = [UsdGeom.Imageable(stage.GetPrimAtPath(robot_scope.AppendChild(link.get('name'))))
                    for link in ET.parse(urdf).getroot().findall('link')]
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
    positions = np.concatenate([np.array([ref.sample(t)[0] for t in np.linspace(0., ref.duration, 121)]) for _,ref in reference.actions])
    cyan, orange = (.015, .9, 1.), (1., .25, .04)
    for i, (phase, ref) in enumerate(reference.actions):
        points = np.array([ref.sample(t)[0] for t in np.linspace(0., ref.duration, 121)])
        make_curve(stage, f'/World/ReferencePath_{i}', points, cyan, .009)
    obstacle_ops = {}
    for obj in runtime.obstacles(0.):
        ball = UsdGeom.Sphere.Define(stage, '/World/Obstacle_' + obj['id'])
        ball.CreateRadiusAttr(obj['radius'])
        color = next(o.get('color', orange) for o in fixture['obstacles'] if o['id']==obj['id'])
        ball.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        obstacle_ops[obj['id']] = ball.AddTranslateOp()
        path = np.array([next(o['center'] for o in runtime.obstacles(t) if o['id'] == obj['id']) for t in times])
        make_curve(stage, '/World/ObstaclePath_' + obj['id'], path, color, .007)
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
    marker_images = [UsdGeom.Imageable(prim) for prim in stage.Traverse()
                     if str(prim.GetPath()).startswith('/World/ReferenceCameraMarker')]
    aim = make_curve(stage, '/World/AimAtSubject', [positions[0], subject['position']], (.26, .68, .70), .002)
    # A metric ground grid makes lateral motion and scene scale readable.
    for i, value in enumerate(np.arange(-1., 2.51, .25)):
        make_curve(stage, f'/World/GridX{i}', [[value, -1.5, .003], [value, 2., .003]], (.46, .49, .52), .002)
    for i, value in enumerate(np.arange(-1.5, 2.01, .25)):
        make_curve(stage, f'/World/GridY{i}', [[-.5, value, .003], [2.5, value, .003]], (.46, .49, .52), .002)
    render_height = args.height - 128
    # Include the spacious ground lanes as well as the camera-height action.
    main_camera, main_pose = make_camera(stage, '/World/Observer', [3.1, -4.5, 2.8], [.75, .10, .40], args.width, render_height, 38.)
    iw, ih = 320, 200
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    detail_camera, detail_pose = make_camera(stage, '/World/ReferenceDetail', center + [1., -2., .9], center, iw, ih, 40.)
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
        snapshots = {0, frame_count//2, frame_count-1}
        snapshots.update(round((p['start_s']+.6*(p['end_s']-p['start_s']))*args.fps) for p,_ in reference.actions)
        snapshots.update(round((a[0]['end_s']+b[0]['start_s'])/2*args.fps) for a,b in zip(reference.actions,reference.actions[1:]))
        design = next(c for c in args.catalog['scenarios'] if c['name']==name)
        if 'interference_audit' in design:
            snapshots.update(round(o['witness_time_s']*args.fps) for o in design['interference_audit']['obstacles'])
        render_times = sorted({i/args.fps for i in snapshots}|{reference.duration}) if args.preview else np.arange(frame_count)/args.fps
        if not args.preview:
            encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{args.width}x{args.height}', '-r', str(args.fps),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE)
        for index, t in enumerate(render_times):
            active = reference.action(t)
            p = q = robot_state = None
            for img in [*robot_images, *marker_images, UsdGeom.Imageable(aim.GetPrim())]:
                img.MakeVisible() if active else img.MakeInvisible()
            if active:
                motion = cache['actions'][active[0]]
                states = np.array(motion['states'])
                robot_state = np.array([np.interp(t, motion['times_s'], states[:,j]) for j in range(9)])
                set_robot_state(robot_state)
                p,r,_ = reference.sample(t)
                q = Rotation.from_matrix(r).as_quat()
                translation.Set(Gf.Vec3d(*map(float,p)))
                orientation.Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))
                aim.GetPointsAttr().Set([Gf.Vec3f(*map(float,p)),Gf.Vec3f(*subject['position'])])
            obstacles = runtime.obstacles(t)
            for obj in obstacles:
                obstacle_ops[obj['id']].Set(Gf.Vec3d(*map(float,obj['center'])))
            clearances = None
            if active:
                distances = collision_geometry.distances(robot_state, [o['center'] for o in obstacles], fixture['base_z']).min(axis=1)
                clearances = {o['id']:float(d-o['radius']) for o,d in zip(obstacles,distances)}
            if index == 0:
                for _ in range(20):
                    app.update()
            rep.orchestrator.step(rt_subframes=8 if index == 0 else 2, delta_time=0., pause_timeline=True)
            images = [np.asarray(a.get_data())[:, :, :3].copy() for a in annotators]
            if images[0].shape != (render_height, args.width, 3) or images[1].shape != (ih, iw, 3):
                raise RuntimeError('Isaac RGB annotators returned incomplete frames')
            frame = caption(*images, name, reference, runtime, float(t), args, clearances)
            if args.preview or index in snapshots:
                from PIL import Image
                Image.fromarray(frame).save(args.output / f'{name}_third_person_{float(t):05.2f}.png')
            if encoder is not None:
                encoder.stdin.write(frame.tobytes())
            frame_records.append(dict(frame=index, trajectory_time_s=float(t), phase=active[1]['name'] if active else None, transition_empty=active is None,
                obstacle_surface_clearances_m=clearances,
                robot_state=None if robot_state is None else robot_state.tolist(), camera_position=None if p is None else p.tolist(),
                camera_quaternion_xyzw=None if q is None else q.tolist(), obstacles=[dict(id=o['id'], center=o['center'].tolist(),
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
            source='original camera reference and runtime obstacle trajectories',
            robot=cache['robot'], gap_policy=cache['gap_policy'],
            camera_interpolation='independent per-action CubicSpline + RotationSpline; no samples or interpolation inside gaps',
            obstacle_interpolation='RuntimeScene CubicHermiteSpline',
            fps=args.fps, width=args.width, height=args.height, duration_s=reference.duration,
            sample_clock="frame_index/fps; final source endpoint at the video end boundary", frame_count=len(frame_records), source_sha256=hashes,
            script_sha256=digest(__file__), shared_renderer_sha256=digest(Path(__file__).with_name('record_scenario_trajectories.py')), observer_camera=main_pose, detail_camera=detail_pose,
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
    if args.prepare:
        prepare(args)
        return
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
