#!/usr/bin/env python3
"""Render measured five-robot diagnostics as explicitly schematic replay films.

The saved runs have public geometry and physical camera poses, but no RGB or
per-link replay stream. This renderer never advances physics or calls Jev.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import render_agent_demos as shared_renderer
from jev_agent_wbc.reference import OriginalReference, look_at
from jev_agent_wbc.representation import Observation
from jev_agent_wbc.representation_video import RepresentationPanel, BOX_EDGES, PART_COLORS

FPS = 16
WIDTH, HEIGHT = 1920, 1080
NAMES = tuple(f'{robot}_{case}' for robot in ('husky', 'ridgeback', 'tiago', 'g1', 'go2')
              for case in ('orbit', 'movein'))


def read_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(draw, at, value, size=20, color='#edf3fa'):
    draw.text(at, value, font=shared_renderer.FONTS[size], fill=color)


def make_panel(rows, observations, maps):
    first = observations[0]
    horizontal = np.array([.78, -.62, 0.]); horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross([.554, .697, .448], horizontal); vertical /= np.linalg.norm(vertical)
    view = np.vstack([horizontal, vertical, np.cross(horizontal, vertical)])
    points = np.vstack([box.vertices for boxes in first.occupancy.values() for box in boxes]
                       + [np.zeros((1, 3)), np.asarray(first.mount_from_camera.translation_m)[None]])
    projected = points @ view.T
    panel = RepresentationPanel.__new__(RepresentationPanel)
    panel.view = view
    panel.center = (projected.min(axis=0) + projected.max(axis=0)) / 2
    panel.scale = min(370 / max(np.ptp(projected[:, 0]), .1),
                      260 / max(np.ptp(projected[:, 1]), .1))
    panel.observations = observations
    mapping = []
    m = -1
    for row in rows:
        while m + 1 < len(maps) and maps[m + 1]['consumed_s'] <= row['stamp_s'] + 1e-9:
            m += 1
        snapshot = maps[m]['snapshot'] if m >= 0 else None
        fresh = snapshot is not None and 0 <= row['stamp_s'] - snapshot['observation_stamp'] <= .25
        mapping.append(dict(info=dict(tpr_status='诊断回放 · 未经 Jev 选择'),
                            observation_stamp=snapshot['observation_stamp'] if fresh else None,
                            runtime_tracks=snapshot.get('tracks', []) if fresh else []))
    panel.run = SimpleNamespace(rows=mapping,
                                times=np.asarray([r['stamp_s'] for r in rows]), cpu=False)
    return panel


class RecordedObstacles:
    def __init__(self, rows, radius):
        self.rows, self.radius = rows, radius

    def obstacles(self, t):
        index = max(0, min(len(self.rows)-1,
            np.searchsorted([r['shot_s'] for r in self.rows], t + 1e-9, side='right') - 1))
        return [dict(center=center, radius=self.radius, shape='sphere')
                for center in self.rows[index]['actual_obstacle_centers']]


def overview_bounds(observations, rows, scene, reference_positions):
    points = [reference_positions[:, :2],
              np.asarray([o.world_from_camera.translation_m[:2] for o in observations]),
              np.asarray(scene['subject']['position'][:2])[None]]
    for row, obs in zip(rows, observations):
        points.extend(np.asarray(c[:2])[None] for c in row['actual_obstacle_centers'])
        for boxes in obs.occupancy.values():
            points.extend(obs.world_from_mount.apply(box.vertices)[:, :2] for box in boxes)
    xy = np.vstack(points)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    center = (lo + hi) / 2
    span = np.maximum(hi - lo, .75)
    return center, span


def overview_boxes(observation, row, scene, reference_positions, measured_positions, bounds):
    """Autoscaled world-XY diagram with the actual saved two-part occupancy."""
    picture = Image.new('RGB', (624, 300), '#1c2b3f')
    draw = ImageDraw.Draw(picture)
    center, span = bounds
    scale = min(520 / span[0], 235 / span[1])
    def point(xy):
        p = np.asarray(xy)[:2] - center
        return (312 + p[0] * scale, 150 - p[1] * scale)
    draw.line([point(p) for p in reference_positions], fill='#8393aa', width=2)
    draw.line([point(p) for p in measured_positions], fill='#5ce0b2', width=3)
    subject = scene['subject']
    sx, sy = point(subject['position'])
    sr = max(4, subject.get('radius', .13) * scale)
    draw.ellipse((sx-sr, sy-sr, sx+sr, sy+sr), fill='#397bd2', outline='#efd66d', width=2)
    for center_m in row['actual_obstacle_centers']:
        x, y = point(center_m)
        radius = scene['obstacles'][0]['radius'] * scale
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill='#a46655', outline='#ffd47a', width=2)
    for part, color in zip(('mobile_platform', 'camera_manipulator'), PART_COLORS):
        for box in observation.occupancy[part]:
            corners = observation.world_from_mount.apply(box.vertices)
            pixels = [point(c) for c in corners]
            for a, b in BOX_EDGES:
                draw.line((pixels[a], pixels[b]), fill=color, width=2)
    cx, cy = point(observation.world_from_camera.translation_m)
    draw.ellipse((cx-4, cy-4, cx+4, cy+4), fill='#ffffff')
    label(draw, (12, 268), '灰：原参考  绿：实测相机  蓝/橙：机器人两模块  红：障碍', 15)
    return picture


def render_case(folder, destination):
    evidence = folder / 'avoidance'
    paths = {name: evidence / name for name in
             ('measurements.jsonl', 'observed_maps.jsonl', 'runtime_scene.json',
              'original_task.json', 'summary.json', 'independent_audit_final.json')}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f'incomplete saved experiment: {folder}')
    if destination.exists():
        raise FileExistsError(destination)
    rows = read_lines(paths['measurements.jsonl'])
    maps = read_lines(paths['observed_maps.jsonl'])
    scene = json.loads(paths['runtime_scene.json'].read_text())
    summary = json.loads(paths['summary.json'].read_text())
    audit = json.loads(paths['independent_audit_final.json'].read_text())
    if not rows or summary['mode'] != 'probe' or not summary['complete']:
        raise ValueError('requires a complete, saved diagnostic execution')
    observations = [Observation.from_dict(row['observation']) for row in rows]
    errors = []
    for row, obs in zip(rows, observations):
        if abs(obs.stamp_s - row['stamp_s']) > 1e-8:
            raise ValueError('measured observation timestamp mismatch')
        camera = obs.world_from_camera
        actual = row['actual_camera']
        distance = np.linalg.norm(np.asarray(camera.translation_m) - actual['translation_m'])
        angle = (Rotation.from_matrix(camera.rotation) *
                 Rotation.from_quat(actual['quaternion_xyzw']).inv()).magnitude()
        errors.append((distance, angle))
    if max(max(error) for error in errors) > 1e-7:
        raise ValueError('public Representation differs from measured physical camera')
    if abs(rows[0]['shot_s']) > 1e-8 or abs(rows[-1]['shot_s'] - 8.) > 1e-8:
        raise ValueError('the saved eight-second shot is incomplete')
    shot_times = np.asarray([r['shot_s'] for r in rows])
    if not np.all(np.diff(shot_times) >= 0):
        raise ValueError('nonmonotonic saved physical time')
    reference = OriginalReference(json.loads(paths['original_task.json'].read_text()),
                                  scene['subject']['position'])
    reference_positions = np.asarray([reference.sample(float(t))[0]
                                      for t in np.linspace(0, 8, 161)])
    scene_run = SimpleNamespace(rows=[dict(camera_position=o.world_from_camera.translation_m,
                                           camera_quaternion_xyzw=o.world_from_camera.quaternion_xyzw,
                                           progress_s=r['shot_s']) for r, o in zip(rows, observations)],
                                times=shot_times, positions=np.asarray(
                                    [o.world_from_camera.translation_m for o in observations]),
                                reference=reference_positions, scene=scene, original=reference,
                                runtime=RecordedObstacles(rows, scene['obstacles'][0]['radius']),
                                floating=False)
    panel = make_panel(rows, observations, maps)
    bounds = overview_bounds(observations, rows, scene, reference_positions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.partial.mp4')
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '21', '-threads', '2',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    records = []
    try:
        for frame in range(FPS * 8 + 1):
            requested = frame / FPS
            index = max(0, np.searchsorted(shot_times, requested + 1e-9, side='right') - 1)
            row, obs = rows[index], observations[index]
            if requested - row['shot_s'] > .021:
                raise ValueError('frame has no sufficiently fresh recorded state')
            body = Image.new('RGB', (WIDTH, HEIGHT), '#101827')
            draw = ImageDraw.Draw(body)
            robot, case = summary['robot'], summary['case']
            label(draw, (20, 14), f'{robot.upper()} · {"环绕 orbit" if case == "orbit" else "推进 movein"}', 31)
            label(draw, (1210, 21), '实测状态几何回放 · 非 RGB 录像', 23, '#ffd47a')
            label(draw, (20, 63), f'运镜 {row["shot_s"]:.2f} / 8.00 s  |  原生 WBC 诊断  |  Jev 未选择候选', 20)
            titles = ('实测占用包络 · 世界 XY 俯视', '原参考相机 · 几何示意',
                      '原始相机（Agent 输入位姿）· 几何示意')
            pictures = [overview_boxes(obs, row, scene, reference_positions,
                                       scene_run.positions[:index+1], bounds),
                        shared_renderer.synthetic_view(scene_run, row['shot_s'], reference=True),
                        shared_renderer.synthetic_view(scene_run, row['shot_s'])]
            for column, (title, picture) in enumerate(zip(titles, pictures)):
                x = 16 + 640 * column
                label(draw, (x, 118), title, 20)
                body.paste(picture, (x + (624-picture.width)//2, 151))
            label(draw, (335, 482), '统一 Representation · 实测两模块 / 完整 6D 位姿', 23)
            panel_image = Image.new('RGB', (1920, 1080), '#101827')
            panel_audit = panel.draw(panel_image, panel.run, row['stamp_s'],
                                     fonts=shared_renderer.FONTS)
            if abs(panel_audit['representation_stamp_s'] - row['stamp_s']) > 1e-8:
                raise ValueError('geometry panel and physical state time differ')
            panel_draw = ImageDraw.Draw(panel_image)
            panel_draw.rectangle((1114, 856, 1541, 915), fill='#172337')
            label(panel_draw, (1114, 860),
                  f'仿真 {row["stamp_s"]:.2f}s / 镜头 {row["shot_s"]:.2f}s', 18)
            label(panel_draw, (1114, 889), '来源：保存的实测 Representation', 18)
            panel_draw.rectangle((1568, 681, 1895, 710), fill='#172337')
            label(panel_draw, (1570, 684), '相机位姿：原始实测', 17, '#9baec4')
            body.paste(panel_image.crop((652, 516, 1904, 984)), (334, 511))
            label(draw, (22, 1011),
                  f'候选跟踪 {"通过" if summary["native_tracking_pass"] else "未通过"}  |  '
                  f'完整物理验收 {"通过" if audit["physical_acceptance"] else "未通过"}  |  '
                  f'位置 RMS {100 * summary["position_rmse_m"]:.2f} cm', 20,
                  '#5ce0b2' if summary['native_tracking_pass'] else '#ff9184')
            label(draw, (22, 1042), '画面由同一时刻已保存的观测与实测障碍位置重绘；未重跑物理或 Agent',
                  17, '#9baec4')
            encoder.stdin.write(body.tobytes())
            records.append(dict(frame=frame, playback_s=requested,
                                measured_shot_s=row['shot_s'], measured_stamp_s=row['stamp_s'],
                                observation_id=obs.id, map_observation_stamp_s=
                                panel.run.rows[index]['observation_stamp']))
    finally:
        encoder.stdin.close()
        status = encoder.wait()
    if status:
        raise RuntimeError(f'ffmpeg exited with {status}: {destination}')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(temporary), '-f', 'null', '-'], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate',
        '-of', 'json', str(temporary)]))['streams'][0]
    if int(probe['nb_read_frames']) != len(records) or int(probe['width']) != WIDTH or int(probe['height']) != HEIGHT:
        raise ValueError('decoded film frame count or dimensions differ from source journal')
    temporary.replace(destination)
    report = dict(protocol='measured_public_geometry_schematic_replay/v1',
        source=str(folder), video=str(destination), robot=summary['robot'], case=summary['case'],
        mode=summary['mode'], agent_trial_status=summary['agent_trial_status'],
        rgb_sensor_replay=False, native_robot_mesh_replay=False, physics_reexecuted=False,
        agent_reexecuted=False, source_cameras='recorded measured camera pose and immutable original reference',
        source_obstacles='recorded physical centers at or before displayed time',
        source_occupancy='recorded unified two-part Representation',
        geometry_scope='conservative measured occupancy proxy, not full robot mesh',
        full_physical_acceptance=audit['physical_acceptance'], tracking_pass=summary['native_tracking_pass'],
        frame_count=len(records), fps=FPS, width=WIDTH, height=HEIGHT,
        decoded_frame_count=int(probe['nb_read_frames']),
        maximum_public_camera_position_m=max(x[0] for x in errors),
        maximum_public_camera_orientation_rad=max(x[1] for x in errors),
        maximum_record_to_playback_lag_s=max(r['playback_s']-r['measured_shot_s'] for r in records),
        input_sha256={str(path): sha(path) for path in paths.values()},
        renderer_sha256={str(Path(__file__)): sha(__file__),
                         str(Path(shared_renderer.__file__)): sha(shared_renderer.__file__),
                         str(Path(sys.modules['jev_agent_wbc.representation_video'].__file__)):
                         sha(sys.modules['jev_agent_wbc.representation_video'].__file__)},
        video_sha256=sha(destination), frame_journal=records)
    destination.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return {key: report[key] for key in ('robot', 'case', 'frame_count', 'tracking_pass',
                                        'full_physical_acceptance', 'video')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', choices=NAMES, default=NAMES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in args.runs:
        result = render_case(args.source / name, args.output / f'{name}.mp4')
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
