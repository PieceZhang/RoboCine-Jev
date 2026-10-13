#!/usr/bin/env python3
"""Causal, log-backed CPU/PhysX decision films; never rerun a planner to invent history.

Each source run is a separate chapter with continuous simulation-time playback.
Never insert explanatory pauses; event and result cards are separate still previews.
Candidate TARGET paths are reconstructed with the production reference class.
Predicted executed poses are deliberately not invented from aggregate metrics.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.contracts import Envelope, digest
from jev_agent_wbc.reference import OriginalReference, DeformedReference, look_at
from jev_agent_wbc.scenarios import RuntimeScene

W, H, FPS = 1920, 1080, 8
BG = '#101827'; PANEL = '#172337'; FG = '#edf3fa'; MUTED = '#9baec4'
BLUE = '#69b9ff'; GREEN = '#5ce0b2'; RED = '#ff9184'; GOLD = '#ffd47a'
COLORS = ['#da9dff', '#ffbf69', '#f47fba', '#8dc8fa']
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTS = {s: ImageFont.truetype(FONT, s) for s in range(14, 35)}
KEY_EVENTS = {'target_replayed', 'search_strategy', 'candidate_set',
              'proposal_rejected', 'continuation_selected', 'candidate_approved',
              'installation_rejected', 'plan_installed', 'math_failure', 'run_failed'}
EVENT_NAMES = {'target_replayed': '复用悬空相机 Agent 目标', 'whole_body_difficulty': '整机困难诊断', 'strategy_requested': '请求搜索策略',
    'search_strategy': '模型返回搜索策略', 'candidate_set': '候选预测完成',
    'selection_requested': '请求候选意图排序', 'proposal_rejected': '本轮结束：未安装',
    'continuation_selected': '选择继续原目标', 'candidate_approved': '模型已选择候选',
    'installation_rejected': '最新状态复核拒绝', 'plan_installed': '修复安装',
    'math_failure': '数学模块失败', 'run_failed': '闭环停止',
    'certified_safety_fallback': '底层安全回退', 'visual_feedback': '视觉反馈'}
REASONS = {'search_refused': '模型拒绝搜索；未生成候选',
    'semantic: no_complete_candidate_accepted': '模型拒绝所有完整候选',
    'common_prefix_not_physically_valid': '激活前共同轨迹未通过物理预测',
    'all_candidates_expired_before_selection': '审批前候选已过期'}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s] if Path(path).exists() else []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def txt(draw, xy, value, size=22, color=FG, width=None):
    value = str(value)
    if width:
        while len(value) > 1 and draw.textlength(value, font=FONTS[size]) > width:
            value = value[:-2] + '…'
    draw.text(xy, value, font=FONTS[size], fill=color)


def wrap(draw, xy, value, width, size=20, limit=2, color=FG):
    x, y = xy; row = ''; rows = []
    for ch in str(value):
        if ch == '\n' or draw.textlength(row+ch, font=FONTS[size]) > width:
            rows.append(row); row = '' if ch == '\n' else ch
        else:
            row += ch
    if row:
        rows.append(row)
    for i, row in enumerate(rows[:limit]):
        txt(draw, (x, y+i*(size+7)), row+('…' if i == limit-1 and len(rows) > limit else ''), size, color, width)


def fmt(value, digits=3):
    return 'unknown' if value is None else f'{value:.{digits}f}'


def verification_status(state):
    names={e['event'] for _,e in state['cycle']}
    if state['decision'].get('event') == 'target_replayed':
        return '源轨迹回放（独立碰撞评分）'
    if state['installed']:
        return '已安装，见原始复核日志'
    if state['decision'].get('event')=='continuation_selected':
        return '不适用（继续原目标）'
    if 'installation_rejected' in names or ('candidate_approved' in names and state['rejected']):
        return '失败；未安装'
    return '处理中' if 'candidate_approved' in names else '未进入'


def restore(original, spec):
    if 'values' not in spec:
        return original
    return DeformedReference(original, Envelope(**spec['envelope']), spec['values'],
        previous=restore(original, spec['previous']) if spec.get('previous') else None,
        activation_s=spec['activation_s'], return_s=spec.get('return_s'))


class Run:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.summary = read(self.folder/'summary.json')
        self.task = read(self.folder/'original_task.json')
        self.scene = read(self.folder/'runtime_scene.json')
        self.runtime = RuntimeScene(self.folder/'runtime_scene.json')
        self.original = OriginalReference(self.task, self.scene['subject']['position'])
        self.events = lines(self.folder/'events.jsonl')
        self.api = lines(self.folder/'jev_api.jsonl')
        self.rows = read(self.folder/'telemetry.json')
        self.times = np.array([r['t'] for r in self.rows])
        self.positions = np.array([r['camera_position'] for r in self.rows])
        self.targets = np.array([r['info']['target_position'] for r in self.rows])
        self.floating = self.summary['backend'] == 'ideal_floating_camera'
        self.replay = bool(self.summary.get('target_replay_source'))
        self.no_avoidance = self.summary.get('obstacle_avoidance_enabled') is False
        self.split_avoidance = self.no_avoidance and self.summary.get('planning_obstacle_avoidance_enabled') is True
        self.cpu = self.summary['backend'] in ('kinematic_integration', 'ideal_floating_camera')
        self.model = 'none (strict baseline)' if self.summary['mode'] == 'strict' else self.summary.get('semantic_model')
        models = {a.get('response', {}).get('model') for a in self.api} - {None}
        if models:
            assert len(models) == 1, models
            self.model = models.pop()
        self.model = '源 Agent 轨迹回放；无新调用' if self.replay else self.model or 'unknown backend'
        self.frames = [] if self.cpu else read(self.folder/'video/frames.json')
        self.frame_times = np.array([r['t'] for r in self.frames])
        self.name = self.scene['name']
        self.paths = {}; self.card_by_id = {}; self.api_at = {}; self.request_at = {}
        requests = [(i, e) for i, e in enumerate(self.events) if e['event'] in ('strategy_requested','selection_requested')]
        # One serial model worker: pair requests by stage and original evidence timestamp.
        for a in self.api:
            stage = 'strategy' if 'strategy' in a.get('request', {}).get('questions', {}) else 'selection'
            state = a.get('request', {}).get('state', '')
            decoder = json.JSONDecoder(); evidence = None
            for pos in (i for i, ch in enumerate(state) if ch == '{'):
                try:
                    obj, _ = decoder.raw_decode(state[pos:])
                    if isinstance(obj, dict) and obj.get('stage') == stage and 'time_s' in obj:
                        evidence = obj; break
                except ValueError:
                    pass
            matches = [(i,e) for i,e in requests if i not in self.request_at
                and e['evidence'].get('stage') == stage
                and (evidence is None or abs(e['evidence']['time_s']-evidence['time_s']) < .001)]
            assert matches, ('unmatched API request', self.folder)
            i, e = matches[0]; self.request_at[i] = a
            terminal = ('search_strategy','proposal_rejected') if stage == 'strategy' else ('candidate_approved','continuation_selected','proposal_rejected')
            for j in range(i+1, len(self.events)):
                if self.events[j]['event'] in terminal:
                    self.api_at[j] = a; break
                if self.events[j]['event'] in ('strategy_requested','selection_requested'):
                    break
        for event in self.events:
            for card in event.get('cards', []):
                assert digest(card['trajectory']) == card['id'], 'candidate hash mismatch'
                assert card['trajectory']['original_sha256'] == self.original.sha256
                path = restore(self.original, card['trajectory'])
                ts = np.linspace(0., self.original.duration, 121)
                self.paths[card['id']] = np.array([path.sample(float(t))[0] for t in ts])
                self.card_by_id[card['id']] = card
        self.reference = np.array([self.original.sample(float(t))[0] for t in np.linspace(0,self.original.duration,121)])
        allpts = np.concatenate([self.reference, self.positions, *self.paths.values()])
        self.lo = allpts.min(axis=0)-.08; self.hi = allpts.max(axis=0)+.08
        self.image_cache = {}; self.render_audit = []

    def state(self, t, cutoff=None):
        past = [(i,e) for i,e in enumerate(self.events) if e['time_s'] <= t+1e-9 and (cutoff is None or i <= cutoff)]
        starts = [i for i,e in past if e['event']=='strategy_requested']
        start = starts[-1] if starts else -1
        cycle = [(i,e) for i,e in past if i>=start]
        def last(name, default=None, subset=cycle):
            return next((e for _,e in reversed(subset) if e['event']==name), default)
        request = last('strategy_requested', {})
        evidence = last('selection_requested', request).get('evidence', {})
        bank = last('target_replayed', {}) if self.replay else last('candidate_set', {})
        answers = [(self.api_at[i],e) for i,e in cycle if i in self.api_at]
        selected = last('target_replayed', {}) if self.replay else last('candidate_approved', last('continuation_selected', {}))
        return dict(past=past, cycle=cycle, round=len(starts), evidence=evidence, bank=bank,
            cards=bank.get('cards', []), strategy=last('search_strategy', {}),
            answer=answers[-1][0] if answers else None,
            decision=selected, rejected=last('proposal_rejected', last('installation_rejected', {})),
            installed=last('plan_installed', {}), requests=sum(e['event'] in ('strategy_requested','selection_requested') for _,e in past),
            diagnosis=last('whole_body_difficulty', {}, past))


def project(points, eye, rotation, width=624, height=300, fov=55):
    q = (np.asarray(points)-eye) @ rotation
    z = q[:,0]; f = height*.5/np.tan(np.deg2rad(fov*.5))
    uv = np.column_stack([width*.5-f*q[:,1]/np.maximum(z,.01), height*.5-f*q[:,2]/np.maximum(z,.01)])
    return uv, z


def synthetic_view(run, t, overview=False, reference=False):
    """Offline schematic renderer, not a sensor or future-obstacle input to Agent."""
    row = run.rows[max(0, np.searchsorted(run.times, t, side='right')-1)]
    p = np.array(row['camera_position']); rot = Rotation.from_quat(row['camera_quaternion_xyzw']).as_matrix()
    if reference:
        p, rot, _ = run.original.sample(t)
    eye, rotation = p, rot
    if overview:
        center = .5*(run.reference.mean(axis=0)+np.array(run.scene['subject']['position']))
        eye = center+[-1.7,-2.0,1.35]; rotation = look_at(eye, center)
    img = Image.new('RGB', (624,300), '#1c2b3f'); d = ImageDraw.Draw(img)
    def line(points, color, width=1):
        uv,z = project(points,eye,rotation)
        if np.all(z>.05) and np.max(np.abs(uv))<10000:
            d.line([tuple(x) for x in uv],fill=color,width=width)
    for x in np.arange(-1,4,.5): line([[x,-2,0],[x,2,0]], '#35465a')
    for y in np.arange(-2,2.1,.5): line([[-1,y,0],[4,y,0]], '#35465a')
    # Subject cylinder with a yellow cap; scene truth is used only for this replay illustration.
    subject=run.scene['subject']; center=np.array(subject['position']); rr=subject.get('radius',.13); h=subject.get('height',.36)
    angle=np.linspace(0,2*np.pi,25)
    lower=np.column_stack([rr*np.cos(angle),rr*np.sin(angle),np.full(25,-h/2)])+center
    upper=lower+[0,0,h]
    faces=[(np.array([lower[i],lower[i+1],upper[i+1],upper[i]]),'#397bd2') for i in range(24)]
    faces.append((upper,'#efd66d'))
    for face,color in sorted(faces,key=lambda f:-np.linalg.norm(f[0].mean(axis=0)-eye)):
        uv,z=project(face,eye,rotation)
        if np.all(z>.02): d.polygon([tuple(v) for v in uv],fill=color,outline='#284d78')
    for obj in run.scene.get('obstacles',[]):
        c=np.asarray(run.runtime.curves[obj['id']](t)); uv,z=project([c],eye,rotation)
        if z[0]>.1:
            rad=300*.5/np.tan(np.deg2rad(27.5))*obj['radius']/z[0]; x,y=uv[0]
            d.ellipse((x-rad,y-rad,x+rad,y+rad),fill='#df654f',outline='#ffad76',width=2)
    if overview and not run.floating and 'state' in row:
        from jev_agent_wbc.jvs_kinematics import JOINT_ORIGINS, JOINT_AXES, CAMERA_OFFSET, _rotation
        q=np.asarray(row['state']);base=np.r_[q[:2],run.scene['base_z']]
        rot_base=Rotation.from_euler('z',q[2]).as_matrix();point=base.copy();orientation=rot_base.copy();joints=[point.copy()]
        for offset,axis,angle in zip(JOINT_ORIGINS,JOINT_AXES,q[3:]):
            point=point+orientation@offset;joints.append(point.copy());orientation=orientation@_rotation(axis,float(angle))
        joints.append(point+orientation@CAMERA_OFFSET);line(joints,'#c6d0dc',5)
        box=np.array([[-.18,-.15,0],[.18,-.15,0],[.18,.15,0],[-.18,.15,0],[-.18,-.15,0]])@rot_base.T+base+[0,0,.1]
        line(box,'#c6d0dc',4)
    if overview:
        line(run.reference, BLUE, 2)
        actual=run.positions[run.times<=t+1e-9]
        if len(actual)>1: line(actual,GREEN,3)
        corners=np.array([[.24,.15,.09],[.24,-.15,.09],[.24,-.15,-.09],[.24,.15,-.09]])@rot.T+p
        for c in corners:line([p,c],GOLD,3)
        line(np.vstack([corners,corners[0]]),GOLD,3)
        line([p,p+rot[:,0]*.5], '#ffffff', 2)
    return img


def plot(draw, box, run, state, t, axes):
    x,y,w,h=box; a,b=axes
    txt(draw,(x,y),f'世界坐标 {"XYZ"[a]}–{"XYZ"[b]} / m',18,MUTED)
    left,top,right,bottom=x+60,y+30,x+w-20,y+h-25
    lo,hi=run.lo.copy(),run.hi.copy()
    # Equal metric scale: avoid exaggerating small vertical errors.
    scale=min((right-left)/(hi[a]-lo[a]),(bottom-top)/(hi[b]-lo[b]))
    center=(lo+hi)/2
    def uv(points):
        p=np.asarray(points)
        return np.column_stack([(left+right)/2+(p[:,a]-center[a])*scale,
                                (top+bottom)/2-(p[:,b]-center[b])*scale])
    for frac in np.linspace(0,1,4):
        yy=top+frac*(bottom-top);draw.line((left,yy,right,yy),fill='#28374b')
        val=center[b]+((top+bottom)/2-yy)/scale
        txt(draw,(x+3,yy-10),f'{val:.2f}',14,MUTED)
    for xx in [left,(left+right)/2,right]:
        txt(draw,(xx-18,bottom+2),f'{center[a]+(xx-(left+right)/2)/scale:.2f}',14,MUTED)
    def path(points,color,width=2,dash=False):
        pp=uv(points)
        if len(pp)<2:return
        if dash:
            for i in range(0,len(pp)-1,3):draw.line([tuple(v) for v in pp[i:i+2]],fill=color,width=width)
        else:draw.line([tuple(v) for v in pp],fill=color,width=width)
    path(run.reference,BLUE,2)
    for j,c in enumerate(state['cards']):path(run.paths[c['id']],COLORS[j%len(COLORS)],3,True)
    n=max(1,np.searchsorted(run.times,t,side='right'))
    installed=next((e for _,e in reversed(state['past']) if e['event'] in ('plan_installed','target_replayed')),None)
    target=run.paths.get(installed['candidate_id'],run.reference) if installed else run.reference
    path(target,GOLD,3,True)
    path(run.positions[:n],GREEN,4)
    u,v=uv(run.positions[n-1:n])[0];draw.ellipse((u-5,v-5,u+5,v+5),fill=GREEN)


def draw_frame(run,t,cutoff=None,label='',end=False):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);s=run.state(t,cutoff)
    backend=('CPU 独立悬空相机 · 1:1 任务空间跟踪' if run.floating else
             'CPU 整机 · 规划避障开 / 执行避障关' if run.split_avoidance else
             'CPU 整机 · WBC 避障关闭' if run.cpu and run.no_avoidance else
             'CPU 数值闭环 · 悬空相机可视化' if run.cpu else 'PhysX 物理闭环 · 整机运行')
    txt(d,(20,10),f'{run.name}  |  {backend}',29)
    txt(d,(1030,16),f'决策模型：{run.model}',24,GOLD,width=870)
    phase=run.original.phase(t)
    txt(d,(20,53),f'原始时刻 {t:.2f} / {run.original.duration:.2f} s   阶段 {phase["name"]} / {phase["motion"]}',21)
    txt(d,(900,53),label or '仿真时间回放 8 FPS；不代表在线墙钟实时速度',21,GOLD,width=990)
    names=['感知 / 诊断','搜索策略','候选预测','模型选择','复核 / 安装',
           '直接跟踪 / 反馈' if run.floating else 'WBC / 反馈']
    baseline=run.summary['mode']=='strict'
    cycle_names={e['event'] for _,e in s['cycle']}
    flags=[bool(s['diagnosis']),bool(s['strategy']),bool(s['bank']),
           'selection_requested' in cycle_names,
           bool(cycle_names & {'candidate_approved','installation_rejected','plan_installed'}),True]
    for i,name in enumerate(names):
        x=20+i*315; fill='#24384a' if flags[i] else '#1b2737'
        d.rounded_rectangle((x,94,x+299,140),radius=8,fill=fill)
        suffix='关闭' if baseline and i in (1,2,3,4) else ('已到达' if flags[i] else '未到达' if s['rejected'] else '等待')
        if i==4 and s['decision'].get('event')=='continuation_selected':suffix='不适用'
        txt(d,(x+10,103),f'{i+1} {name} · {suffix}',20,GREEN if flags[i] else MUTED,width=283)
    captures=[]
    for col,title in enumerate(['独立相机（无底盘 / 机械臂）' if run.floating else 'CPU 相机位姿（整机 FK 示意）' if run.cpu else 'PhysX 整机实录',
                               '原始参考相机画面'+('（示意）' if run.cpu else ''),
                               'CPU 执行相机画面（示意）' if run.cpu else 'PhysX 实际执行 RGB']):
        x=16+col*640;txt(d,(x,151),title,21)
        if run.cpu:
            pic=synthetic_view(run,t,overview=col==0,reference=col==1)
        else:
            idx=max(0,np.searchsorted(run.frame_times,t+1e-9,side='right')-1)
            capture=run.frames[idx];captures.append(capture['t'])
            stream=['overview','reference','execution'][col];key=(idx,stream)
            if key not in run.image_cache:
                src=Image.open(run.folder/'video'/stream/f'{idx:05d}.jpg').convert('RGB');src.thumbnail((624,300))
                pic=Image.new('RGB',(624,300),'#0b111c');pic.paste(src,((624-src.width)//2,(300-src.height)//2));run.image_cache[key]=pic
            pic=run.image_cache[key]
        im.paste(pic,(x,183))
    d.rectangle((16,454,1904,483),fill=BG)
    txt(d,(24,456),'意图原文：'+run.task['intent'],17,FG,width=1868)
    txt(d,(20,484),'CPU：仅离线示意渲染，无 RGB 输入' if run.cpu else f'RGB 采集时刻 {captures[0]:.2f} s；事件页沿用最近已采集帧',17,MUTED)
    txt(d,(760,484),('碰撞仅检测不干预；相机半径 '+fmt(run.summary.get('camera_radius_m'))+' m' if run.floating else
        '规划 WBC 避障开启；执行关闭，碰撞仅检测' if run.split_avoidance else
        'WBC 避障关闭；碰撞仅检测不干预' if run.no_avoidance else
        'LiDAR 在线；SAM 未进入本次决策。未来遮挡 / 完整尺度：unknown'),17,MUTED)
    for box in [(16,516,624,468),(652,516,1252,244),(652,773,612,211),(1276,773,628,211)]:
        x,y,w,h=box;d.rounded_rectangle((x,y,x+w,y+h),radius=9,fill=PANEL)
    txt(d,(30,522),'相机轨迹对照',23)
    txt(d,(216,527),'原参考',17,BLUE);txt(d,(298,527),'实际执行',17,GREEN);txt(d,(406,527),'当前目标虚线',17,GOLD)
    plot(d,(25,552,601,186),run,s,t,(0,1));plot(d,(25,743,601,187),run,s,t,(0,2))
    txt(d,(30,936),'虚线：本轮候选目标（按日志精确参数重建）',18,MUTED)
    txt(d,(30,960),'候选目标 ≠ 预测执行；重合曲线可能被覆盖',17,MUTED)
    bank=s['bank'];cards=s['cards'];ev=bank.get('evaluations',[])
    counts=Counter('full' if 'full' in e else 'coarse' if 'coarse' in e else 'other' for e in ev)
    txt(d,(668,521),'源 Agent 已安装的精确目标（当前仅回放，无重新选择）' if run.replay else f'第 {s["round"]} 轮候选  |  粗筛 {counts["coarse"]}  完整预测 {counts["full"]}  可展示 {len(cards)}',23)
    txt(d,(1370,524),'来源：已录制的悬空相机 Agent' if run.replay else f'规划墙钟 {fmt(bank.get("planner_wall_s"))} s',20,MUTED,width=515)
    if not cards:
        reason='strict 基线：Agent 关闭，没有候选和模型决策' if baseline else ('搜索被拒绝，未生成候选' if s['strategy'].get('strategy')=='refuse' else '本轮尚无完整候选；不能从粗筛指标虚构轨迹')
        txt(d,(680,572),reason,24,GOLD,width=1200)
        wrap(d,(680,617),run.task['intent'],1195,20,3,MUTED)
    else:
        txt(d,(670,558),'ID / 参数 r,l,h (m) / 激活→结束',18,MUTED)
        txt(d,(1190,558),'源相机净距 / 无关节约束' if run.floating or run.replay else '预测净距 / 关节余量',18,MUTED)
        txt(d,(1480,558),'预测构图 / 尺度代理 / 观察角',18,MUTED)
        for j,c in enumerate(cards):
            assert j<3,'layout supports the configured maximum of 3 full candidates'
            yy=585+j*51;pred=c['prediction'];visual=c['visual']['summary'];tr=c['trajectory'];vals=tr['values']
            submitted=s['evidence'].get('candidates',cards)
            alias=next((f'c{k}' for k,v in enumerate(submitted) if v['id']==c['id']), '未送审')
            txt(d,(670,yy),f'{alias} {c["id"][:8]}  [{vals[0]:+.2f}, {vals[1]:+.2f}, {vals[2]:+.2f}]',20,COLORS[j])
            txt(d,(685,yy+25),f'{tr["activation_s"]:.2f}→{tr["envelope"]["end_s"]:.2f}s  {tr["envelope"]["terminal"]}  物理预测：{c["physical_valid"]}',16,MUTED)
            txt(d,(1190,yy),f'{fmt(pred.get("min_clearance_m"))} m'+(' / 不适用' if run.floating or run.replay else f' / {fmt(pred.get("joint_margin_rad"))} rad'),19)
            txt(d,(1480,yy),f'{fmt(visual.get("composition_error"))} / {fmt(visual.get("range_scale_error"))} / {fmt(visual.get("view_angle_error_rad"))}',19)
            txt(d,(1190,yy+25),f'下一阶段已检查：{pred.get("next_phase_checked",False)}',16,MUTED)
            txt(d,(1480,yy+25),f'出画中心时长 {fmt(visual.get("center_out_of_frame_duration_s"))} s',16,MUTED)
    evidence=s['evidence'];diag=evidence.get('difficulty',s['diagnosis'].get('report',{}))
    txt(d,(668,778),'输入证据与搜索决策',22)
    family=s['strategy'].get('strategy','未调用' if baseline else '等待')
    txt(d,(668,811),f'修复族：{family}   API 已请求 {s["requests"]} 次',20,GOLD)
    issues=', '.join(diag.get('issues',[])) or ('基线控制，无语义修复' if baseline else '等待诊断')
    wrap(d,(668,845),issues,575,18,2,MUTED)
    txt(d,(668,902),f'预测风险时刻 {fmt(diag.get("first_risk_s"),2)} s  净距 {fmt(diag.get("min_clearance_m"))} m',18)
    nxt=evidence.get('next_phase') or {}
    row=run.rows[max(0,np.searchsorted(run.times,t,side='right')-1)];info=row['info']
    txt(d,(668,931),f'当前执行误差 {fmt(info.get("position_error"))} m / {fmt(info.get("orientation_error"))} rad',18,MUTED)
    txt(d,(668,958),f'独立真值净距 {fmt(row.get("truth_clearance_m"))} m | 阈值 0.025 m',18,RED if row.get('truth_clearance_m',1)<.025 else GREEN)
    txt(d,(1292,778),'模型输出 → 验证 → 执行',22)
    answer=s['answer']; response=answer.get('response',{}) if answer else {}
    ans=next(iter(response.get('answers',{}).values()),{})
    choice=ans.get('choice','未调用' if baseline else '等待')
    question=next(iter(answer.get('request',{}).get('questions',{})), '') if answer else ''
    txt(d,(1292,811),f'{question or "Choice"}：{choice}   API {fmt(answer.get("wall_s") if answer else None)} s',20,GOLD,width=598)
    probs=ans.get('probabilities',{})
    txt(d,(1292,844),'  '.join(f'{k} {v:.2f}' for k,v in probs.items()) or '概率无记录；不补写模型解释',17,MUTED,width=594)
    reason=s['rejected'].get('error','')
    selected=s['decision'].get('candidate_id',s['decision'].get('approval',{}).get('candidate_id',''))
    outcome=REASONS.get(reason,reason) if reason else ('修复已安装' if s['installed'] else '继续原目标；不算修复安装' if s['decision'].get('event')=='continuation_selected' else '语义通过，等待最新状态复核' if selected else '未批准新修复；WBC 独立执行')
    if run.replay:
        outcome='复用源 Agent 目标；WBC '+('避障关闭' if run.no_avoidance else '避障开启')
    wrap(d,(1292,877),outcome,590,19,2,RED if reason else GREEN)
    installs=sum(e['event']=='plan_installed' for _,e in s['past'])
    txt(d,(1292,931),f'已复用 {sum(e["event"]=="target_replayed" for _,e in s["past"])} 条源目标；无新修复安装' if run.replay else f'截至当前安装 {installs} 次 | 概率不代表物理安全',18,MUTED)
    gates=verification_status(s)
    txt(d,(1292,958),f'最新状态 / 时效 / 视觉复核：{gates}',16,MUTED,width=590)
    # Original-time progress bar and observed event markers, with no future choices revealed.
    d.rectangle((20,1002,1900,1009),fill='#2b3c54')
    d.rectangle((20,1002,20+1880*t/run.original.duration,1009),fill=GREEN)
    for _,e in s['past']:
        if e['event'] in KEY_EVENTS:
            xx=20+1880*e['time_s']/run.original.duration;d.line((xx,996,xx,1016),fill=GOLD,width=2)
    last=s['past'][-1][1] if s['past'] else {}
    txt(d,(20,1024),f'最近事件：{EVENT_NAMES.get(last.get("event"),last.get("event","初始化"))}  @ {fmt(last.get("time_s"),2)} s',20,width=990)
    txt(d,(1050,1024),f'整段运行墙钟 {run.summary["wall_s"]:.1f} s | 连续回放 · 无讲解停留',19,MUTED,width=840)
    if end:
        d.rounded_rectangle((350,198,1570,456),radius=14,fill='#101827',outline=GOLD,width=2)
        result='镜头达标' if run.summary['within_budget_success'] else '镜头未达标'
        txt(d,(390,211),f'本段结果：{result}  |  '+(f'复用 {len(run.summary["replayed_targets"])} 条源目标' if run.replay else f'修复安装 {run.summary["installed_plans"]} 次'),30,GREEN if run.summary['within_budget_success'] else RED)
        txt(d,(390,262),f'记录 {run.summary["elapsed_shot_s"]:.2f} / {run.original.duration:.2f} s；位置 RMS {run.summary["original_position_rms_m"]:.4f} m',24)
        physical=(f'相机球体净距 {fmt(run.summary.get("min_truth_clearance_m"))} m；直接跟踪误差 0；无机器人约束' if run.floating else
                  f'WBC 避障关闭；独立检测最小净距 {fmt(run.summary.get("min_truth_clearance_m"))} m' if run.cpu and run.no_avoidance else
                  'CPU 数值约束检查；不代表 PhysX 物理验证') if run.cpu else f'已记录物理子步约束：{run.summary.get("physical_audit",{}).get("pass_constraints","unknown")}；中止时仅覆盖已执行前缀'
        txt(d,(390,305),physical,22,MUTED,width=1140)
        wrap(d,(390,348),run.summary.get('failure') or ('固定复用源 Agent 的相同目标与时刻，比较 WBC 跟踪及碰撞结果。' if run.replay else 'Agent 关闭：本段用于底层基线对照。' if baseline else '保留真实拒绝和复核失败；未将继续原目标计为修复。'),1140,21,2,GOLD)
        txt(d,(390,417),'结果摘要（独立预览图）；不插入视频，不重新判定实验结果',20,MUTED)
    run.render_audit.append(dict(t=t,cutoff=cutoff,max_event_time=max((e['time_s'] for _,e in s['past']),default=0.),
        cards=[c['id'] for c in cards],rgb_t=captures[0] if captures else None))
    return im


def render_run(run,write,timeline,preview_dir):
    start=write.count
    # Uniform source-time ticks only: events never add frames or freeze playback.
    frame_times=np.arange(0,float(run.times[-1])-1e-8,1/FPS).tolist() or [0.]
    for t in frame_times:
        write(draw_frame(run,t))
    timeline.append(dict(output_start_s=start/FPS,output_end_s=write.count/FPS,
        source=str(run.folder),kind='continuous_playback',frame_times_s=frame_times))
    # Explanations remain available as stills without interrupting any chapter.
    for index,event in enumerate(run.events):
        if event['event']=='candidate_set':
            draw_frame(run,event['time_s'],index,'候选轨迹与决策（独立预览图）').save(
                preview_dir/f'{run.folder.parent.name}_{run.name}_candidates.jpg')
            break
    end=draw_frame(run,float(run.times[-1]),label='结果摘要（独立预览图）',end=True)
    end.save(preview_dir/f'{run.folder.parent.name}_{run.name}_end.jpg')
    for row in run.render_audit:
        assert row['max_event_time']<=row['t']+1e-8
        assert row['rgb_t'] is None or row['rgb_t']<=row['t']+1e-8
    return dict(folder=str(run.folder),backend=run.summary['backend'],model=run.model,
        frames=write.count-start,output_start_s=start/FPS,output_end_s=write.count/FPS,
        candidate_hashes_verified=len(run.paths),api_requests_matched=len(run.request_at),
        source_events=len(run.events),rendered_event_holds=0,rendered_summary_hold_frames=0,
        summary=run.summary,input_sha256={n:sha(run.folder/n) for n in ['original_task.json','runtime_scene.json','events.jsonl','telemetry.json','summary.json']})


class Encoder:
    def __init__(self,path):
        self.count=0
        self.process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2',
            '-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def __call__(self,image,count=1):
        data=image.tobytes()
        for _ in range(count):self.process.stdin.write(data)
        self.count+=count
    def close(self):
        self.process.stdin.close()
        if self.process.wait():raise RuntimeError('ffmpeg failed')


def pairs(root,name,manifest):
    result=[]
    for entry in manifest['runs']:
        phys=Path(entry['folder']).resolve();case=read(phys/'runtime_scene.json')['name']
        if name in ('dynamic_diagnostics','stage3_dynamic_diagnostics'):
            cpu=root/'demo_rebuild_20261013/dynamic_cpu'/f'{case}_jev'
        elif name=='nominal_six_shots':cpu=root/'demo_rebuild_20261013/nominal_cpu'/f'{case}_strict'
        elif name=='orbit_pan':cpu=root/'composite_wide_cpu/orbit_pan_wide_strict'
        elif name=='four_models':
            model=read(phys/'summary.json')['semantic_model'];cpu=root/f'model_{model}_cpu/movein_jev'
        else:raise ValueError(name)
        assert read(cpu/'original_task.json')==read(phys/'original_task.json'), 'task mismatch'
        assert read(cpu/'runtime_scene.json')==read(phys/'runtime_scene.json'), 'scene mismatch'
        cpu_budget=lines(cpu/'events.jsonl')[0].get('budget')
        phys_budget=lines(phys/'events.jsonl')[0].get('budget')
        assert cpu_budget==phys_budget, ('budget mismatch',name,case)
        result.extend([cpu,phys])
    return result


def render_explicit_runs(folders, dest):
    """Render fresh CPU or PhysX logs in the existing demo layout."""
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = dest.with_suffix('.json')
    if dest.exists() or manifest.exists():
        raise FileExistsError(f'refusing to replace an existing recording: {dest}')
    preview = dest.parent/(dest.stem+'_previews')
    preview.mkdir(exist_ok=False)
    temp = dest.with_suffix('.new.mp4')
    writer = Encoder(temp)
    runs, timeline = [], []
    try:
        for folder in folders:
            print('render', folder, flush=True)
            runs.append(render_run(Run(folder), writer, timeline, preview))
    finally:
        writer.close()
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(temp), '-f', 'null', '-'], check=True)
    temp.replace(dest)
    report = dict(format='agent_demo/v2', resolution=[W,H], fps=FPS,
        frames=writer.count, duration_s=writer.count/FPS,
        playback='continuous simulation-time playback; no editorial or summary holds; not wall-clock realtime',
        candidate_paths='exact logged target paths; not predicted executed poses',
        cpu_views='offline CPU schematic; run backend distinguishes ideal floating camera from whole-body FK; no RGB input',
        gold_path='full currently installed target reconstructed only after its logged installation/replay event',
        causal_audit='displayed decisions and RGB timestamps never exceed replay time',
        runs=runs, timeline=timeline, video_sha256=sha(dest), renderer_sha256=sha(__file__))
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('DONE', dest, writer.count/FPS, 'seconds', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path)
    p.add_argument('--runs', nargs='+', type=Path, help='explicit new run folders in chapter order')
    p.add_argument('--output', type=Path, help='new MP4 path for --runs')
    p.add_argument('--only',nargs='*');p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    if a.runs:
        if not a.output or a.root or a.only or a.preview_only:
            p.error('--runs requires --output and cannot be combined with --root/--only/--preview-only')
        render_explicit_runs(a.runs, a.output)
        return
    if not a.root or a.output:
        p.error('use --root for historical demos, or --runs with --output for new recordings')
    root=a.root.resolve();work=root/'demo_rebuild_20261013';backup=work/'original_demo';preview=work/'previews'
    backup.mkdir(parents=True,exist_ok=True);preview.mkdir(exist_ok=True)
    # Preserve old videos/manifests before replacement, including original archive checksums.
    for source in (root/'demo').iterdir():
        if source.suffix in ('.json','.mp4') and not (backup/source.name).exists():shutil.copy2(source,backup/source.name)
    for name in a.only or ['four_models','nominal_six_shots','orbit_pan','dynamic_diagnostics','stage3_dynamic_diagnostics']:
        inputs=pairs(root,name,read(backup/f'{name}.json'));print(name, 'chapters',len(inputs),flush=True)
        if a.preview_only:
            for folder in inputs[:2]:
                run=Run(folder);event=next(((i,e) for i,e in enumerate(run.events) if e['event']=='candidate_set'),None)
                i,e=event if event else (None,{'time_s':2.})
                draw_frame(run,e['time_s'],i,'独立预览图：候选轨迹与决策').save(preview/f'{name}_{"cpu" if run.cpu else "physx"}.png')
            continue
        # The historical stage3 name is an alias of precisely the same source runs.
        # Reuse the validated encoding when inputs and renderer fingerprints match.
        if name=='stage3_dynamic_diagnostics' and (root/'demo/dynamic_diagnostics.json').exists():
            previous=read(root/'demo/dynamic_diagnostics.json')
            if (previous.get('renderer_sha256')==sha(__file__) and
                    [str(p.resolve()) for p in inputs]==[r['folder'] for r in previous.get('runs',[])]):
                shutil.copy2(root/'demo/dynamic_diagnostics.mp4',root/'demo/stage3_dynamic_diagnostics.mp4')
                previous['alias_of']='dynamic_diagnostics.mp4'
                (root/'demo/stage3_dynamic_diagnostics.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n')
                print('DONE',name,'identical-source alias',flush=True)
                continue
        dest=root/'demo'/f'{name}.mp4';temp=work/f'{name}.new.mp4';writer=Encoder(temp);runs=[];timeline=[]
        try:
            for folder in inputs:
                run=Run(folder);print('render',folder,flush=True)
                runs.append(render_run(run,writer,timeline,preview))
        finally:writer.close()
        # Decode every frame before replacing the visible video.
        subprocess.run(['ffmpeg','-v','error','-i',str(temp),'-f','null','-'],check=True)
        temp.replace(dest)
        report=dict(format='agent_demo/v2',resolution=[W,H],fps=FPS,frames=writer.count,duration_s=writer.count/FPS,
            playback='CPU and PhysX separate chapters; continuous simulation-time playback; no editorial or summary holds; not wall-clock realtime',
            candidate_paths='exact target paths reconstructed from logged parameters; NOT predicted executed poses',
            cpu_views='offline schematic render of CPU camera poses; scene truth for display only; no RGB input',
            baseline='strict chapters explicitly disable semantic stages; no fabricated model decisions',
            causal_audit='all displayed decisions <= replay time; RGB uses last capture <= replay time',
            runs=runs,timeline=timeline,video_sha256=sha(dest),renderer_sha256=sha(__file__))
        (root/'demo'/f'{name}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('DONE',name,writer.count/FPS,'seconds',flush=True)


if __name__=='__main__':main()
