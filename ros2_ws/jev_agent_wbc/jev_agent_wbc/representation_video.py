"""Shared, log-backed two-part geometry display for CPU and PhysX films."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .representation import PARTS, SIGNS, Pose
from .representation_jvs import JVSGeometryBackend


PART_COLORS = ('#69b9ff', '#ffbf69')
AXIS_COLORS = ('#ff9184', '#5ce0b2', '#a1adff')
BOX_FACES = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
BOX_EDGES = tuple((i, j) for i in range(8) for j in range(i+1, 8)
                  if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1)


class RepresentationPanel:
    def __init__(self, run):
        if run.floating or not run.rows or any('state' not in r for r in run.rows):
            raise ValueError('representation panel requires saved whole-body states')
        self.run = run
        self.backend = JVSGeometryBackend(int(run.summary.get('representation_boxes', 3)))
        self.observations = [self.backend.observe(r['state'], r['t'],
                             base_z=r.get('base_z', run.scene.get('base_z', 0.)),
                             base_rotation=np.asarray(r['base_rotation']) if 'base_rotation' in r else None)
                             for r in run.rows]
        position_errors, rotation_errors = [], []
        for row, obs in zip(run.rows, self.observations):
            camera = obs.world_from_camera
            # Geometry follows physical state, even when the published image
            # comes from a virtual stabilized camera with a different pose.
            position = row.get('actual_camera_position', row['camera_position'])
            quaternion = row.get('actual_camera_quaternion_xyzw', row.get('camera_quaternion_xyzw'))
            rotation = (Rotation.from_quat(quaternion).as_matrix() if quaternion is not None else
                        np.asarray(row['actual_camera_rotation'] if 'actual_camera_rotation' in row else row['camera_rotation']))
            position_errors.append(np.linalg.norm(np.asarray(camera.translation_m)-position))
            rotation_errors.append(Rotation.from_matrix(camera.rotation.T @ rotation).magnitude())
        if run.cpu and (max(position_errors) > 1e-8 or max(rotation_errors) > 1e-8):
            raise ValueError('current geometry backend disagrees with saved camera FK')
        verified = 0
        for event in run.events:
            geometry = event.get('evidence', {}).get('robot_geometry') or event.get('geometry')
            if not geometry or geometry.get('source') != 'measured_fk':
                continue
            stamp = geometry['stamp_s']
            row_index = int(np.argmin(np.abs(run.times-stamp)))
            if abs(run.times[row_index]-stamp) > 1e-8:
                raise ValueError('logged geometry has no matching telemetry state')
            obs = self.observations[row_index]
            for key in ('world_from_mount', 'mount_from_camera'):
                logged = Pose.from_dict(geometry[key])
                current = getattr(obs, key)
                if (not np.allclose(logged.translation_m, current.translation_m, atol=1e-9, rtol=0.)
                        or not np.allclose(logged.rotation, current.rotation, atol=1e-9, rtol=0.)):
                    raise ValueError('current transforms differ from recorded representation')
            for part in PARTS:
                regions = geometry['parts'][part]['regions']
                boxes = obs.occupancy[part]
                if len(regions) != len(boxes) or any(
                    not np.allclose(region[key], getattr(box, key), atol=1e-9, rtol=0.)
                    for region, box in zip(regions, boxes)
                    for key in ('center_m', 'half_extents_m')):
                    raise ValueError('current occupancy differs from recorded representation')
            verified += 1
        # One metric scale and a fixed camera for each chapter; only geometry at
        # the latest past telemetry tick is drawn, never future poses or choices.
        horizontal = np.array([.78, -.62, 0.])
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross([.554, .697, .448], horizontal)
        vertical /= np.linalg.norm(vertical)
        self.view = np.vstack([horizontal, vertical, np.cross(horizontal, vertical)])
        all_points = np.vstack([b.vertices for obs in self.observations
                                for boxes in obs.occupancy.values() for b in boxes] +
                              [np.zeros((1, 3))] +
                              [np.asarray(o.mount_from_camera.translation_m)[None, :]
                               + .12*o.mount_from_camera.rotation.T for o in self.observations])
        projected = all_points @ self.view.T
        self.center = (projected.min(axis=0)+projected.max(axis=0))/2
        self.scale = min(370/np.ptp(projected[:, 0]), 260/np.ptp(projected[:, 1]))
        self.audit = dict(schema='jev_geometry/v1', telemetry_states_verified=len(run.rows),
            logged_geometry_snapshots_verified=verified,
            max_camera_position_error_m=float(max(position_errors)),
            max_camera_rotation_error_rad=float(max(rotation_errors)),
            occupancy_frame='mount', representation_boxes=self.backend.box_budget,
            source='current JVSGeometryBackend applied to saved whole-body states, including full base rotation; '
                'physical camera is distinct from stabilized output; no integration or new model calls',
            transform_convention='p_parent = R_parent_child @ p_child + translation_m',
            displayed_orientation='fixed-axis RPY degrees (Rz Ry Rx) and original XYZW quaternion',
            geometry_scope='conservative proxy of backend collision geometry, not a safety certificate',
            obstacle_relations='saved runtime_tracks projected at constant velocity into current mount; '
                'signed center-to-box distance without object radius; unknown after 0.6 s; no scene truth')

    @classmethod
    def from_recording(cls, folder):
        folder = Path(folder)
        rows = json.loads((folder/'telemetry.json').read_text())
        summary_path = folder/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        scene_path = folder/'runtime_scene.json'
        scene = json.loads(scene_path.read_text()) if scene_path.is_file() else {}
        event_path = folder/'events.jsonl'
        events = [json.loads(line) for line in event_path.read_text().splitlines() if line] if event_path.is_file() else []
        backend = summary.get('backend', 'unknown')
        return cls(SimpleNamespace(folder=folder, rows=rows, summary=summary, scene=scene,
            events=events, times=np.array([r['t'] for r in rows]),
            cpu='physx' not in backend.lower(), floating=backend == 'ideal_floating_camera'))

    @staticmethod
    def track_relations(row, observation):
        """Use saved observed centers; missing object radii are never invented."""
        stamp = row.get('observation_stamp')
        if stamp is None or not 0. <= observation.stamp_s-stamp <= .6+1e-8:
            return None
        dt = max(0., observation.stamp_s-stamp)
        inverse = observation.world_from_mount.inverse()
        candidates = []
        for track in row.get('runtime_tracks', []):
            if not 0. <= track.get('age_s', np.inf)+dt <= .6+1e-8:
                continue
            point = inverse.apply(np.asarray(track['center'])+dt*np.asarray(track['velocity']))
            distances = [min(float(b.distance(point)) for b in observation.occupancy[part])
                         if observation.occupancy[part] else None for part in PARTS]
            axis = int(np.argmax(np.abs(point)))
            direction = (('前方', '后方'), ('左方', '右方'), ('上方', '下方'))[axis][int(point[axis] < 0)]
            candidates.append(dict(track_id=track['id'], center_mount=point.tolist(),
                distances_m=distances, direction=direction,
                observation_stamp_s=stamp, projection_s=dt, age_s=track['age_s']+dt))
        return min(candidates, key=lambda c:min(d for d in c['distances_m'] if d is not None)) if candidates else None

    def graphic(self, observation, fonts):
        image = Image.new('RGB', (430, 338), '#111d2d')
        draw = ImageDraw.Draw(image)

        def uv(points):
            p = np.atleast_2d(points) @ self.view.T
            return np.column_stack([215+(p[:, 0]-self.center[0])*self.scale,
                                    150+(p[:, 1]-self.center[1])*self.scale])

        floor = -observation.world_from_mount.translation_m[2]
        for offset in np.arange(-.4, .61, .1):
            for line in ([[offset, -.4, floor], [offset, .6, floor]],
                         [[-.4, offset, floor], [.6, offset, floor]]):
                draw.line([tuple(p) for p in uv(line)], fill='#243346')
        faces, edges = [], []
        for color, part in zip(PART_COLORS, PARTS):
            for box in observation.occupancy[part]:
                vertices = box.vertices
                points = uv(vertices)
                for face in BOX_FACES:
                    depth = float((vertices[list(face)] @ self.view[2]).mean())
                    faces.append((depth, [tuple(p) for p in points[list(face)]], color))
                edges.extend(([tuple(points[a]), tuple(points[b])], color) for a, b in BOX_EDGES)
        layer = Image.new('RGBA', image.size)
        translucent = ImageDraw.Draw(layer)
        for _, polygon, color in sorted(faces, key=lambda f: -f[0]):
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            translucent.polygon(polygon, fill=(*rgb, 38))
        image = Image.alpha_composite(image.convert('RGBA'), layer).convert('RGB')
        draw = ImageDraw.Draw(image)
        for line, color in edges:
            draw.line(line, fill=color, width=2)

        def triad(pose, label):
            origin = np.asarray(pose.translation_m)
            points = uv(np.vstack([origin, origin+.13*pose.rotation.T]))
            for j, color in enumerate(AXIS_COLORS):
                draw.line([tuple(points[0]), tuple(points[j+1])], fill=color, width=3)
                draw.text(tuple(points[j+1]), 'XYZ'[j], font=fonts[14], fill=color)
            x, y = points[0]
            draw.ellipse((x-4, y-4, x+4, y+4), fill='#edf3fa')
            draw.text((x+8, y-22), label, font=fonts[17], fill='#edf3fa')

        mount = Pose((0., 0., 0.))
        camera = observation.mount_from_camera
        draw.line([tuple(p) for p in uv([mount.translation_m, camera.translation_m])],
                  fill='#ffd47a', width=2)
        triad(mount, 'mount')
        triad(camera, 'camera')
        draw.line((24, 302, 24+.2*self.scale, 302), fill='#9baec4', width=2)
        draw.text((24, 309), '0.20 m · 等比例', font=fonts[15], fill='#9baec4')
        draw.text((224, 309), 'X 前 / Y 左 / Z 上', font=fonts[15], fill='#9baec4')
        return image

    def draw(self, image, run, t, state=None, txt=None, wrap=None, fonts=None, cutoff=None):
        if fonts is None:
            from PIL import ImageFont
            fonts = {s:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', s)
                     for s in range(14, 35)}
        if txt is None:
            def txt(d, xy, value, size=22, color='#edf3fa', width=None):
                value = str(value)
                if width:
                    while len(value) > 1 and d.textlength(value, font=fonts[size]) > width:
                        value = value[:-2]+'…'
                d.text(xy, value, font=fonts[size], fill=color)
        if wrap is None:
            def wrap(d, xy, value, width, size=20, limit=2, color='#edf3fa'):
                rows, line = [], ''
                for char in value:
                    if d.textlength(line+char, font=fonts[size]) > width:
                        rows.append(line); line = ''
                    line += char
                rows.append(line)
                for i, line in enumerate(rows[:limit]):
                    txt(d, (xy[0], xy[1]+i*(size+7)), line, size, color, width)
        row_index = max(0, np.searchsorted(run.times, t+1e-9, side='right')-1)
        obs = self.observations[row_index]
        draw = ImageDraw.Draw(image)
        draw.rectangle((648, 513, 1906, 987), fill='#101827')
        draw.rounded_rectangle((652, 516, 1904, 984), radius=9, fill='#172337')
        txt(draw, (668, 522), '统一 Representation · 两个模块 / 两个 6D 位姿变换', 23)
        txt(draw, (668, 557), f'移动平台 · {len(obs.occupancy[PARTS[0]])} 盒', 18, PART_COLORS[0])
        txt(draw, (868, 557), f'相机机构 · {len(obs.occupancy[PARTS[1]])} 盒', 18, PART_COLORS[1])
        image.paste(self.graphic(obs, fonts), (668, 589))
        txt(draw, (668, 936), 'mobile_platform + camera_manipulator', 17, '#9baec4')
        txt(draw, (668, 961), 'mount 系保守占用包络；不代替碰撞检查', 16, '#9baec4')
        txt(draw, (1114, 557), 'world ── mount ── camera  位姿链', 20, '#ffd47a')
        for y, key, title in ((591, 'world_from_mount', 'T_world_mount · 世界 ← 安装接口'),
                              (724, 'mount_from_camera', 'T_mount_camera · 安装接口 ← 相机')):
            pose = getattr(obs, key)
            draw.rounded_rectangle((1114, y, 1540, y+120), radius=6, fill='#111d2d')
            txt(draw, (1126, y+7), title, 18, '#ffd47a')
            txt(draw, (1126, y+36), 'p [m]    '+'  '.join(f'{v:+.3f}' for v in pose.translation_m), 17)
            rpy = Rotation.from_matrix(pose.rotation).as_euler('xyz', degrees=True)
            txt(draw, (1126, y+63), 'RPY [°]  '+'  '.join(f'{v:+.2f}' for v in rpy), 17)
            txt(draw, (1126, y+90), 'q [xyzw] '+' '.join(f'{v:+.3f}' for v in pose.quaternion_xyzw), 16, '#9baec4')
        txt(draw, (1114, 860), f'时间：回放 {t:.2f}s / 状态 {obs.stamp_s:.2f}s', 18)
        txt(draw, (1114, 889), '来源：保存的机器人状态 + 当前 FK', 18)
        txt(draw, (1114, 918), '空间：两部分占用统一在 mount 系', 18)
        distance = np.linalg.norm(obs.mount_from_camera.translation_m)
        txt(draw, (1114, 950), f'接口→相机 {distance:.3f} m  |  旋转按 Rz Ry Rx', 17, '#9baec4')
        draw.line((1555, 554, 1555, 967), fill='#34435a')
        if state is not None:
            txt(draw, (1570, 557), '决策摘要 · 原始日志', 20)
            txt(draw, (1570, 591), f'第 {state["round"]} 轮 / 合格候选 {len(state["cards"])}', 18)
            answer = state['answer'] or {}
            response = answer.get('response', {}).get('answers', {})
            choice = next(iter(response.values()), {}).get('choice', '等待')
            txt(draw, (1570, 622), f'Choice：{choice}', 19, '#ffd47a', width=315)
            installs = sum(e['event'] == 'plan_installed' for _, e in state['past'])
            txt(draw, (1570, 653), f'截至此刻安装 {installs} 次', 18, '#5ce0b2')
            last = state['past'][-1][1] if state['past'] else {}
            txt(draw, (1570, 684), f'最近事件 @{last.get("time_s", 0.):.2f}s', 17, '#9baec4')
        else:
            row = run.rows[row_index]
            txt(draw, (1570, 557), '闭环摘要 · 原始日志', 20)
            txt(draw, (1570, 591), 'CPU 数值闭环' if run.cpu else 'PhysX 实测闭环', 18)
            txt(draw, (1570, 622), f'控制状态：{row.get("info", {}).get("tpr_status", "unknown")}', 18, '#ffd47a')
            txt(draw, (1570, 653), '几何来自物理机器人状态', 18, '#5ce0b2')
            txt(draw, (1570, 684), '增稳画面位姿单独保留', 17, '#9baec4')
        txt(draw, (1570, 725), '观测空间关系 · 保存的跟踪', 18)
        relation = self.track_relations(run.rows[row_index], obs)
        if relation:
            txt(draw, (1570, 757), f'采集 @{relation["observation_stamp_s"]:.2f}s / 观测龄 {relation["age_s"]:.2f}s', 16, '#9baec4')
            txt(draw, (1570, 788), f'track:{relation["track_id"]} · 接口{relation["direction"]}', 18)
            txt(draw, (1570, 817), '中心 [m] '+' '.join(f'{v:+.2f}' for v in relation['center_mount']), 16)
            for j, name in enumerate(('平台', '机构')):
                distance = relation['distances_m'][j]
                value = f'{distance:+.3f} m' if distance is not None else 'unknown'
                txt(draw, (1570, 849+j*30), f'{name}包络→中心 {value}', 17, PART_COLORS[j])
        else:
            wrap(draw, (1570, 768), '保存的观测缺失或已过期；当前障碍关系 unknown', 306, 18, 3, '#9baec4')
        txt(draw, (1570, 917), '中心到包络距离；未扣物体尺寸', 16, '#9baec4')
        txt(draw, (1570, 944), ('CPU' if run.cpu else 'PhysX')+' 遥测回放 · 未重新运行实验', 16, '#9baec4')
        return dict(representation_stamp_s=obs.stamp_s, representation_observation_id=obs.id,
                    obstacle_relation_stamp_s=relation['observation_stamp_s'] if relation else None)


def append_representation_video(source, destination, folder, frame_times=None):
    """Preserve a film's full picture and append the shared geometry panel.

    Used by the fixed-camera TPR renderer, including the three-camera PhysX
    branch. Raw RGB streams remain measurement artifacts, not final films.
    """
    import hashlib
    import subprocess
    import cv2
    from PIL import ImageFont

    source, destination = Path(source), Path(destination)
    panel = RepresentationPanel.from_recording(folder)
    fonts = {s:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', s)
             for s in range(14, 35)}
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f'cannot read source video: {source}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    source_width, source_height = (int(capture.get(key)) for key in
                                  (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or source_width <= 0 or source_height <= 0:
        capture.release()
        raise ValueError('invalid source video dimensions or timebase')
    width = max(1920, source_width)
    height = (source_height+500+1)//2*2
    temporary = destination.with_name(destination.stem+'.representation.new.mp4')
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '19', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(temporary)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = 0
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        while True:
            valid, frame = capture.read()
            if not valid:
                break
            if frame_times is not None and count >= len(frame_times):
                raise ValueError('video has more frames than its timestamp journal')
            t = float(frame_times[count]) if frame_times is not None else float(panel.run.times[0])+count/fps
            if t < panel.run.times[0]-1e-8 or t > panel.run.times[-1]+1/fps:
                raise ValueError('video timestamp falls outside recorded telemetry')
            image = Image.new('RGB', (width, height), '#101827')
            image.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                        ((width-source_width)//2, 0))
            scratch = Image.new('RGB', (1920, 1080), '#101827')
            audit = panel.draw(scratch, panel.run, t, fonts=fonts)
            assert audit['representation_stamp_s'] <= t+1e-8
            image.paste(scratch.crop((652, 516, 1904, 984)), ((width-1252)//2, source_height+16))
            d = ImageDraw.Draw(image)
            d.text((24, source_height+36), '统一 Representation', font=fonts[24], fill='#edf3fa')
            d.text((24, source_height+81), f'仿真时间 {t:.2f}s', font=fonts[20], fill='#9baec4')
            d.text((24, source_height+121), '两模块占用 / 两个 6D 变换', font=fonts[20], fill='#9baec4')
            encoder.stdin.write(image.tobytes())
            count += 1
        if count == 0 or (frame_times is not None and count != len(frame_times)):
            raise ValueError('video and timestamp journal frame counts differ')
    finally:
        capture.release()
        encoder.stdin.close()
        status = encoder.wait()
    if status:
        raise RuntimeError('representation video encoding failed')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(temporary), '-f', 'null', '-'], check=True)
    temporary.replace(destination)
    return dict(representation_panel=True, representation_audit=panel.audit,
        source_video_sha256=source_hash, fps=fps, frames=count, resolution=[width, height],
        source_picture='unaltered pixels, centered above a separate geometry footer',
        video_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
