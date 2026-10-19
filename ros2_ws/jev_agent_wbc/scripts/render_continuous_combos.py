#!/usr/bin/env python3
"""Continuous combo adapter for the standard CPU / Representation renderer.

No planner is rerun. Every chapter follows one unbroken physical clock. The
reference panel shows the authored action, or the next start pose in transfers.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
import render_agent_demos as shared
from jev_agent_wbc.representation_video import RepresentationPanel
from jev_agent_wbc.scenarios import RuntimeScene


def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SavedReference:
    def __init__(self,run):self.run=run;self.duration=run.duration
    def sample(self,t):
        row=self.run.rows[max(0,np.searchsorted(self.run.times,t,side='right')-1)]
        return np.array(row['reference_position']),Rotation.from_quat(row['reference_quaternion_xyzw']).as_matrix(),np.zeros(6)


class ContinuousRun:
    def __init__(self,folder):
        self.folder=Path(folder).resolve();self.summary=read(self.folder/'summary.json')
        self.rows=read(self.folder/'telemetry.json');self.scene=read(self.folder/'runtime_scene.json')
        self.task=read(self.folder/'source_task.json');self.runtime=RuntimeScene(self.folder/'runtime_scene.json')
        self.events=shared.lines(self.folder/'events.jsonl')
        self.times=np.array([r['t'] for r in self.rows]);self.duration=float(self.times[-1])
        self.positions=np.array([r['camera_position'] for r in self.rows]);self.references=np.array([r['reference_position'] for r in self.rows])
        self.reference=self.references;self.original=SavedReference(self)
        self.floating=False;self.cpu=True;self.paths={}
        points=np.vstack([self.references,self.positions]);self.lo=points.min(0)-.08;self.hi=points.max(0)+.08
        self.representation_panel=RepresentationPanel(self)


MOTIONS=dict(movein='推近',moveout='拉远',truck_left='左横移',truck_right='右横移',orbit='环绕',pedestal_up='升镜',pedestal_down='降镜')


def frame(run,t):
    row=run.rows[max(0,np.searchsorted(run.times,t,side='right')-1)];index=row['phase_index'];stage=row['stage']
    phase=run.task['shot_phase'][index];motion=MOTIONS.get(phase['motion'],phase['motion'])
    main=stage=='main_action';title=f'运镜 {index+1}：{motion}' if main else f'衔接 → 运镜 {index+1}：{motion}'
    ids=[i for i,r in enumerate(run.rows) if r['phase_index']==index and r['stage']=='main_action']
    run.reference=run.references[ids] if main and ids else np.asarray([row['reference_position']])
    im=Image.new('RGB',(shared.W,shared.H),shared.BG);d=ImageDraw.Draw(im);txt=shared.txt
    name=run.scene.get('source_combo',run.folder.parent.name)
    txt(d,(20,12),f'{name}  |  CPU 整机连续闭环',29)
    txt(d,(1040,16),'Typesafe 官方 Jev · jev-1.14.0',24,shared.GOLD)
    txt(d,(20,61),f'场景时间 {t:.2f} / {run.duration:.2f} s   ·   {title}',24,shared.GREEN)
    txt(d,(1010,65),'机器人、速度、感知地图与障碍时钟连续',21,shared.MUTED)
    labels=['整机 / 障碍场景（CPU 示意）','原始运镜参考（CPU 示意）' if main else '下一段初始位姿（衔接目标）','执行相机画面（FK 示意）']
    for i,label in enumerate(labels):
        x=16+640*i;txt(d,(x+8,112),label,22)
        im.paste(shared.synthetic_view(run,t,overview=i==0,reference=i==1),(x,148))
    txt(d,(24,460),'蓝：当前参考 / 交接目标    绿：截至当前真实执行轨迹',19,shared.MUTED)
    txt(d,(24,488),'CPU 数值积分闭环；离线示意渲染，无 PhysX / RGB 输入',18,shared.MUTED)
    txt(d,(1010,488),'主运镜严格指标保持；衔接仅要求安全和下一段可执行',18,shared.MUTED)
    past=[(i,e) for i,e in enumerate(run.events) if e.get('time_s',0.)<=t+1e-9]
    simple=dict(cards=[],past=past)
    shared.plot(d,(16,520,620,220),run,simple,t,(0,1));shared.plot(d,(16,746,620,220),run,simple,t,(0,2))
    audit=run.representation_panel.draw(im,run,t,None,txt,shared.wrap,shared.FONTS)
    d=ImageDraw.Draw(im)
    latest=next((e for _,e in reversed(past) if e['event'] in ('main_phase_finished','run_failed','transfer_failed')),None)
    status='主运镜执行中' if main else '自由相机衔接 · 无镜头 / 时间指标'
    if latest and latest['event']=='run_failed':status='闭环停止：'+latest['failure']
    elif latest and latest['event']=='main_phase_finished':
        status=f"最近完成运镜 {latest['phase_index']+1}："+('严格检查通过' if latest['accepted'] else '严格检查未通过')
    txt(d,(20,972),status,18,shared.RED if latest and latest.get('failure') else shared.GOLD,width=1870)
    d.rectangle((20,1002,1900,1009),fill='#35465a');d.rectangle((20,1002,20+1880*t/max(run.duration,.001),1009),fill=shared.GREEN)
    txt(d,(20,1024),f'物理时间连续 1× 回放 · 无停帧 · 采样时间 {row["t"]:.2f}s',19,shared.MUTED)
    txt(d,(1040,1024),f'执行墙钟 {run.summary["wall_s"]:.1f}s · 请求 pace=1',19,shared.MUTED)
    assert row['t']<=t+1e-8
    return im,audit


def render_one(args):
    folder,output,fps=args;output=Path(output);shared.FPS=fps;run=ContinuousRun(folder)
    name=run.folder.parent.name;movie=output/(name+'.mp4');encoder=shared.Encoder(movie)
    times=np.arange(int(np.ceil(run.duration*fps))+1)/fps;times=np.minimum(times,run.duration)
    stamps=[]
    for i,t in enumerate(times):
        im,audit=frame(run,float(t));encoder(im);stamps.append(audit)
        if i in (0,len(times)//2,len(times)-1):im.save(output/f'{name}_{i:05d}.jpg')
    encoder.close()
    subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    files=['source_task.json','runtime_scene.json','telemetry.json','events.jsonl','summary.json']
    record=dict(case=name,source=str(run.folder),video=str(movie.resolve()),fps=fps,frames=len(times),duration_s=len(times)/fps,
        source_duration_s=run.duration,completed=run.summary['completed'],accepted=run.summary['shot_acceptance']['accepted'],
        failed_checks=run.summary['shot_acceptance']['failed_checks'],explanatory_hold_frames=0,decode_check='pass',
        frame_source_times_s=times.tolist(),input_sha256={f:sha(run.folder/f) for f in files},
        representation_audit=run.representation_panel.audit)
    save(output/(name+'_recording.json'),record);print(json.dumps(dict(case=name,frames=len(times),video=str(movie))),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=int,default=16,choices=[8,16,24]);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cases',nargs='+',default=['combo_01_easy','combo_02_easy','combo_03_medium','combo_04_medium','combo_05_hard','combo_06_hard'])
    p.add_argument('--preview-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    folders=[a.runs/c/(c+'_jev') for c in a.cases]
    if a.preview_only:
        for folder in folders:
            run=ContinuousRun(folder);frame(run,run.duration/2)[0].save(a.output/(folder.parent.name+'.png'))
        return 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:records=list(pool.map(render_one,[(f,a.output,a.fps) for f in folders]))
    concat=a.output/'concat.txt';concat.write_text(''.join("file '"+str(Path(r['video']).resolve()).replace("'","'\\''")+"'\n" for r in records))
    combined=a.output/'combo_01_to_06_collection.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(combined)],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(combined),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    offset=0.;chapters=[]
    for record in records:
        chapters.append(dict(case=record['case'],start_s=offset,end_s=offset+record['duration_s']));offset+=record['duration_s']
    save(a.output/'recording_manifest.json',dict(collection=str(combined.resolve()),chapters=chapters,runs=records,
        renderer_sha256={str(f):sha(f) for f in [Path(__file__),Path(shared.__file__),Path(shared.__file__).parents[1]/'jev_agent_wbc/representation_video.py']},
        backend='CPU schematic replay from continuous closed-loop telemetry',decode_check='pass',fps=a.fps))
    return 0


if __name__=='__main__':raise SystemExit(main())
