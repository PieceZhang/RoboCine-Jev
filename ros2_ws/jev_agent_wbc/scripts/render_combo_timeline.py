#!/usr/bin/env python3
"""Presentation-only rerecording of six continuous combos in the prior UI.

Geometry/camera views and the shared Representation come from the verified
encoded source frames. Recorded state supplies the plots and causal HUD.
"""
import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import multiprocessing

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = None
SOURCE = None
REFERENCE = ROOT / 'results/combo_layers_20261018/demo_video_unified_timeline_20261019/combo_01_04_unified_timeline.mp4'
W, H = 1920, 1080
BG, PANEL, FG = '#101827', '#172337', '#edf3fa'
MUTED, GREEN, GOLD, RED, BLUE = '#9baec4', '#5ce0b2', '#ffd47a', '#ff9184', '#69b9ff'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS = {size: ImageFont.truetype(FONT, size) for size in range(14, 35)}
MOTIONS = dict(movein='推进', moveout='拉远', truck_left='左移', truck_right='右移',
               pedestal_up='抬升', pedestal_down='下降', orbit='环绕')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=2048)
def glyph(value, size, color):
    font = FONTS[size]
    box = font.getbbox(value)
    image = Image.new('RGBA', (max(1, int(font.getlength(value)) + 3), size + 15))
    ImageDraw.Draw(image).text((0, -box[1] + 2), value, font=font, fill=color)
    return image


def text(image, xy, value, size=20, color=FG, width=None):
    value = str(value)
    if width:
        while len(value) > 1 and FONTS[size].getlength(value) > width:
            value = value[:-2] + '…'
    layer = glyph(value, size, color)
    image.paste(layer, tuple(map(int, xy)), layer)


class Timeline:
    def __init__(self, record):
        self.record = record
        self.folder = Path(record['source'])
        self.source = Path(record['video'])
        self.source_hash = sha(self.source)
        self.rows = read(self.folder / 'telemetry.json')
        self.times = np.asarray([row['t'] for row in self.rows])
        self.positions = np.asarray([row['camera_position'] for row in self.rows])
        self.references = np.asarray([row['reference_position'] for row in self.rows])
        self.task = read(self.folder / 'source_task.json')
        self.events = [json.loads(line) for line in (self.folder / 'events.jsonl').read_text().splitlines() if line]
        self.duration = float(record['source_duration_s'])
        self.fps = record['fps']
        self.frame_times = record['frame_source_times_s']
        self.frames = record['frames']
        assert self.frames == len(self.frame_times)
        assert abs(self.duration - self.times[-1]) < 1e-8
        self.input_hashes = dict(record['input_sha256'])
        for filename, digest in self.input_hashes.items():
            assert sha(self.folder / filename) == digest, filename
        self.number = int(record['case'].split('_')[1])
        self.stages = []
        previous = None
        for i, row in enumerate(self.rows):
            key = row['stage'], row['phase_index']
            if key == previous:
                continue
            if self.stages:
                self.stages[-1]['end_s'] = row['t']
                self.stages[-1]['last_row'] = i
            main = row['stage'] == 'main_action'
            index = row['phase_index']
            motion = self.task['shot_phase'][index]['motion']
            label = MOTIONS.get(motion, motion) if main else ('初始衔接' if index == 0 else f'衔接 {index}')
            planned = self.folder / f'phase_{index+1:02d}' / 'arrival_boundary.json'
            if not planned.exists():
                planned = self.folder / f'phase_{index+1:02d}' / 'phase_program.json'
            plan = read(planned) if main and planned.exists() else None
            if plan is not None:
                self.input_hashes[str(planned.relative_to(self.folder))] = sha(planned)
            self.stages.append(dict(kind='main' if main else 'transfer', phase_index=index,
                label=label, start_s=float(row['t']), first_row=i, plan=plan))
            previous = key
        self.stages[-1].update(end_s=self.duration, last_row=len(self.rows))
        self.starts = [s['start_s'] for s in self.stages]
        for stage in self.stages:
            a, b = stage['first_row'], stage['last_row']
            reference = self.references[a:b]
            stage['reference'] = reference if stage['kind'] == 'main' else reference[:1]
            plan = stage['plan']
            stage['target'] = (np.asarray(plan['geometry']['positions']) if plan else reference[:1])
            points = np.vstack([reference, stage['target'], self.positions[a:b]])
            stage['lo'], stage['hi'] = points.min(0) - .12, points.max(0) + .12
        self.header = Image.new('RGB', (W, 142), BG)
        difficulty = '简单' if self.number <= 2 else '中等' if self.number <= 4 else '困难'
        text(self.header, (22, 13), f'COMBO {self.number:02d} · {difficulty}', 30)
        text(self.header, (1090, 19), 'Typesafe Jev 1.14.0 · CPU 闭环回放', 23, GOLD)
        width = (1880 - 20 * (len(self.stages) - 1)) / len(self.stages)
        self.boxes = [(20 + i*(width+20), 96, 20 + i*(width+20) + width, 136)
                      for i in range(len(self.stages))]
        self.footer = Image.new('RGB', (W, 95), BG)
        d = ImageDraw.Draw(self.footer)
        for stage in self.stages:
            left = 20 + 1880 * stage['start_s'] / self.duration
            right = 20 + 1880 * stage['end_s'] / self.duration
            text(self.footer, ((left+right)/2 - FONTS[17].getlength(stage['label'])/2, 2),
                 stage['label'], 17, BLUE if stage['kind'] == 'main' else GOLD)
            d.rectangle((left, 29, right, 43), fill='#21364a' if stage['kind'] == 'main' else '#423b2c')
        text(self.footer, (22, 64), '来源：整组连续闭环记录 · 动作切换无状态 / 障碍时钟复位', 17, MUTED)
        text(self.footer, (1180, 64), '统一播放进度 · 1× 仿真时间 · 无讲解停顿', 17, MUTED)
        self.audit = []

    def plot(self, image, box, stage, row_index, axes):
        x, y, w, h = box
        a, b = axes
        left, right, top, bottom = x+55, x+w-15, y+25, y+h-20
        lo, hi = stage['lo'], stage['hi']
        scale = min((right-left)/(hi[a]-lo[a]), (bottom-top)/(hi[b]-lo[b]))
        center = (lo+hi)/2
        d = ImageDraw.Draw(image)
        text(image, (x, y), f'{"XYZ"[a]}–{"XYZ"[b]} / m', 17, MUTED)
        def uv(points):
            p = np.asarray(points)
            return np.c_[(left+right)/2+(p[:, a]-center[a])*scale,
                         (top+bottom)/2-(p[:, b]-center[b])*scale]
        def path(points, color, width):
            if len(points) > 1:
                d.line([tuple(p) for p in uv(points)], fill=color, width=width)
        for fraction in (0, .5, 1):
            yy = top + fraction*(bottom-top)
            d.line((left, yy, right, yy), fill='#2b3c54')
            text(image, (x+2, yy-8), f'{center[b]+((top+bottom)/2-yy)/scale:.2f}', 14, MUTED)
        path(stage['reference'], BLUE, 2)
        path(stage['target'], GOLD, 2)
        if stage['kind'] == 'transfer':
            u, v = uv(stage['target'][:1])[0]
            d.ellipse((u-6, v-6, u+6, v+6), outline=GOLD, width=2)
        visible = self.positions[stage['first_row']:row_index+1]
        path(visible, GREEN, 3)
        u, v = uv(visible[-1:])[0]
        d.ellipse((u-4, v-4, u+4, v+4), fill=GREEN)

    def draw(self, source, index):
        t = float(self.frame_times[index])
        position = bisect_right(self.starts, t+1e-9)-1
        stage = self.stages[position]
        row_index = max(0, int(np.searchsorted(self.times, t+1e-9, side='right')-1))
        row = self.rows[row_index]
        assert row['stage'] == ('main_action' if stage['kind'] == 'main' else row['stage'])
        assert row['t'] <= t+1e-8 and row['phase_index'] == stage['phase_index']
        image = source.copy()
        image.paste(self.header, (0, 0))
        d = ImageDraw.Draw(image)
        d.rectangle((0, 142, W, 515), fill=BG)
        titles = ['机器人与障碍物 · 离线示意',
                  '原始镜头参考 · 离线示意' if stage['kind'] == 'main' else '下一段起始相机目标 · 离线示意',
                  '实际执行相机 · FK 示意']
        for i, title in enumerate(titles):
            x = 16+640*i
            text(image, (x, 151), title, 20)
            # Same geometry pixels as the encoded recording; match the old
            # UI's 271-pixel high view windows by cropping only their bottom.
            region = source.crop((x, 148, x+624, 419))
            image.paste(region, (x, 183))
            assert image.crop((x, 183, x+624, 454)).tobytes() == region.tobytes()
        text(image, (24, 465), '机器人 / 相机均来自保存的 FK · 场景为离线示意 · 不代表墙钟实时性能', 17, MUTED)
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((16, 516, 640, 984), radius=9, fill=PANEL)
        text(image, (30, 527), '相机运动 · 世界坐标', 23)
        text(image, (30, 560), '原参考', 17, BLUE)
        text(image, (140, 560), '实际执行', 17, GREEN)
        text(image, (265, 560), '入段目标' if stage['kind'] == 'main' else '下一段起点', 17, GOLD)
        self.plot(image, (25, 595, 601, 160), stage, row_index, (0, 1))
        self.plot(image, (25, 770, 601, 160), stage, row_index, (0, 2))
        text(image, (30, 948), '曲线来自保存的目标与 FK；未重跑规划器', 17, MUTED)
        # Preserve the source's shared Representation, both full 6D transform
        # cards, and causal observed-object relation. Only its HUD is restyled.
        for area in ((668, 589, 1098, 927), (1114, 591, 1540, 857), (1568, 748, 1895, 970)):
            assert image.crop(area).tobytes() == source.crop(area).tobytes()
        d = ImageDraw.Draw(image)
        d.rectangle((1568, 553, 1895, 716), fill=PANEL)
        info = row['info']
        text(image, (1570, 557), '闭环摘要 · 当时日志', 20)
        text(image, (1570, 591), '控制状态：'+str(info.get('tpr_status', 'unknown')), 18, GOLD, 318)
        gap = info.get('command_sweep_lower_m')
        text(image, (1570, 622), '命令扫掠下界 '+(f'{gap:.3f} m' if gap is not None else 'unknown'), 18)
        route = info.get('transfer_route')
        profile = (stage['plan'] or {}).get('program', {}).get('phase_shapes', {})
        summary = '通路：'+route if route else '主运镜形状：'+str(next(iter(profile.values()), '保存的入段曲线'))
        text(image, (1570, 653), summary, 17, GREEN, 318)
        text(image, (1570, 684), '换路计算中' if info.get('transfer_replan_pending') else '控制连续推进', 17, MUTED)
        text(image, (22, 57), f'Combo 总进度  {t:05.1f} / {self.duration:.1f} s    ·    当前：{stage["label"]}', 21)
        local = t-stage['start_s']
        text(image, (1100, 57), f'动作内 {local:.1f}/{stage["end_s"]-stage["start_s"]:.1f} s · 动作 {position+1}/{len(self.stages)}', 18, MUTED)
        d = ImageDraw.Draw(image)
        for i, (box, item) in enumerate(zip(self.boxes, self.stages)):
            x, y, right, bottom = box
            d.rounded_rectangle(box, radius=7, fill='#29463d' if i == position else '#19293a',
                outline=GREEN if i == position else '#2a3c50', width=2 if i == position else 1)
            text(image, (x+14, y+10), f'{i+1}  '+item['label'], 19, GREEN if i == position else MUTED)
            if i < position:
                text(image, (right-55, y+10), '已过', 16, MUTED)
        image.paste(self.footer, (0, 985))
        progress = t/self.duration
        cursor = 20+1880*progress
        d = ImageDraw.Draw(image)
        d.rectangle((20, 1014, cursor, 1028), fill=GREEN)
        for boundary in self.starts[1:]:
            mark = 20+1880*boundary/self.duration
            d.line((mark, 1010, mark, 1032), fill='#bdcad7')
        d.ellipse((cursor-5, 1016, cursor+5, 1026), fill=FG)
        if index == self.frames-1:
            d.rectangle((1160, 1045, 1908, 1078), fill=BG)
            ok = self.record['completed'] and self.record['accepted']
            text(image, (1180, 1049), '本组连续闭环：严格检查通过' if ok else '本组连续闭环：存在未通过检查', 17, GREEN if ok else RED)
        self.audit.append(dict(frame=index, combo_time_s=t, progress=progress, action_index=position,
            action=stage['label'], source_telemetry_time_s=float(row['t']), source_frame_index=index))
        return image


def encode(record, preview_only=False):
    cv2.setNumThreads(1)
    timeline = Timeline(record)
    cap = cv2.VideoCapture(str(timeline.source))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == timeline.frames
    assert abs(cap.get(cv2.CAP_PROP_FPS)-timeline.fps) < 1e-6
    name = record['case']
    previews = {0: 'start', timeline.frames//2: 'middle', timeline.frames-1: 'end'}
    for stamp in timeline.starts[1:]:
        previews[min(timeline.frames-1, int(np.searchsorted(timeline.frame_times, stamp)))] = 'switch_'+str(round(stamp*100))
    if preview_only:
        for index, suffix in previews.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            assert ok
            timeline.draw(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), index).save(OUT/f'{name}_{suffix}.jpg')
        cap.release()
        return
    dest = OUT / f'{name}.mp4'
    assert not dest.exists()
    encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}', '-r', str(timeline.fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-threads', '2', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)], stdin=subprocess.PIPE)
    try:
        for index in range(timeline.frames):
            ok, bgr = cap.read()
            assert ok, (name, index)
            image = timeline.draw(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), index)
            if index in previews:
                image.save(OUT/f'{name}_{previews[index]}.jpg')
            encoder.stdin.write(image.tobytes())
        assert not cap.read()[0]
    finally:
        cap.release()
        encoder.stdin.close()
        assert encoder.wait() == 0
    assert sha(timeline.source) == timeline.source_hash
    assert timeline.audit[0]['progress'] == 0. and timeline.audit[-1]['progress'] == 1.
    assert all(a['progress'] <= b['progress'] and a['combo_time_s'] <= b['combo_time_s']
               for a, b in zip(timeline.audit, timeline.audit[1:]))
    for boundary in timeline.starts[1:]:
        index = int(np.searchsorted(timeline.frame_times, boundary))
        assert timeline.audit[index]['progress'] > timeline.audit[index-1]['progress'] > 0.
    for path, digest in timeline.input_hashes.items():
        assert sha(timeline.folder/path) == digest
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(dest), '-f', 'null', '-'], check=True)
    audit = name+'_timeline_audit.json'
    save(OUT/audit, dict(one_progress_bar_per_combo=True, monotone_progress_verified=True,
        no_reset_at_action_boundaries=True, no_explanatory_hold_frames=True, frames=timeline.audit))
    stages = [{key: stage[key] for key in ('kind', 'phase_index', 'label', 'start_s', 'end_s')} for stage in timeline.stages]
    result = dict(case=name, video=dest.name, frames=timeline.frames, fps=timeline.fps,
        duration_s=timeline.frames/timeline.fps, source_duration_s=timeline.duration,
        source_video=str(timeline.source), source_video_sha256=timeline.source_hash,
        source_recording=str(timeline.folder), source_inputs_sha256=timeline.input_hashes,
        source_representation_audit=record['representation_audit'], timeline_audit=audit,
        stages=stages, full_decode_pass=True, sha256=sha(dest))
    print(json.dumps(dict(done=name, frames=timeline.frames)), flush=True)
    return result


def main():
    global OUT, SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    OUT, SOURCE = args.output.resolve(), args.source.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    source_manifest = read(SOURCE/'recording_manifest.json')
    records = source_manifest['runs']
    assert len(records) == 6 and all(r['completed'] and r['accepted'] for r in records)
    if args.preview_only:
        for record in records:
            encode(record, True)
        return
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        results = list(pool.map(encode, records))
    fps = source_manifest['fps']
    assert all(r['fps'] == fps for r in results)
    playlist = OUT/'concat.txt'
    playlist.write_text(''.join("file '"+r['video']+"'\n" for r in results))
    plain = OUT/'collection_without_chapters.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(playlist), '-c', 'copy', str(plain)], check=True)
    chapters = []
    metadata = [';FFMETADATA1', 'title=Combo 01–06 · 统一动作时间轴']
    cursor = 0.
    for i, result in enumerate(results, 1):
        end = cursor+result['duration_s']
        chapters.append(dict(case=result['case'], start_s=cursor, end_s=end))
        metadata += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={round(cursor*1000)}',
                     f'END={round(end*1000)}', f'title=COMBO {i:02d}']
        cursor = end
    (OUT/'chapters.ffmetadata').write_text('\n'.join(metadata)+'\n')
    combined = OUT/'combo_01_06_unified_timeline.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(plain), '-i', str(OUT/'chapters.ffmetadata'),
        '-map', '0:v:0', '-map_metadata', '1', '-map_chapters', '1', '-c', 'copy', '-movflags', '+faststart', str(combined)], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(combined), '-f', 'null', '-'], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames:format=duration:chapter=start_time,end_time,tags',
        '-of', 'json', str(combined)]))
    assert int(probe['streams'][0]['nb_read_frames']) == sum(r['frames'] for r in results)
    assert (probe['streams'][0]['width'], probe['streams'][0]['height']) == (W, H)
    assert len(probe['chapters']) == 6
    save(OUT/'manifest.json', dict(format='continuous_combo_unified_timeline/v1',
        video=combined.name, fps=fps, resolution=[W, H], frames=sum(r['frames'] for r in results),
        duration_s=cursor, chapters=chapters, runs=results, full_decode_pass=True,
        one_progress_bar_per_combo=True, progress_resets_only_between_combos=True,
        continuous_physical_state_within_each_combo=True, renderer_modifies_trajectory=False,
        renderer_model_calls=0, renderer_control_executions=0, representation_preserved=True,
        layout_reference=str(REFERENCE), layout_reference_sha256=sha(REFERENCE),
        source_manifest=str(SOURCE/'recording_manifest.json'), source_manifest_sha256=sha(SOURCE/'recording_manifest.json'),
        renderer_sha256=sha(__file__), preserved_regions='Source camera view pixels and shared two-module Representation / 6D transforms; before re-encoding',
        camera_window_source_crop=[148, 419], camera_window_dest_y=183,
        video_sha256=sha(combined), probe=probe))
    print(json.dumps(dict(video=str(combined), frames=sum(r['frames'] for r in results), duration_s=cursor, full_decode_pass=True)), flush=True)


if __name__ == '__main__':
    main()
