#!/usr/bin/env python3
"""PhysX presentation adapter around the current Representation film renderer.

Display measured overview, live original-reference floating camera and the declared
Agent camera. Legacy recordings without a trusted live reference retain their
explicit old view layout. All source streams and shared render code are preserved.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np
import render_agent_demos as renderer
from physx_acceptance import assess_physx, POLICY_VERSION


def film_views(source,reference):
    replay=(reference.get('protocol')=='measured_state_replay_original_reference_same_timestamp/v1'
            and reference.get('measured_state_replay') is True)
    if replay:
        if not (reference.get('valid') is True and reference.get('same_render_pass') is True
                and reference.get('live_sensor_capture') is False):
            raise ValueError('replay requires verified measured link transforms')
        return [dict(stream='overview',title='PhysX 整机（实测状态重放）',agent_input=False),
                dict(stream='reference',title='参考相机（原参考轨迹重绘）',agent_input=False),
                dict(stream=source,title='原始相机（Agent 输入位姿重放）',agent_input=True)]
    trusted=(reference.get('protocol')=='original_reference_floating_camera_same_scene_time/v1'
             and reference.get('valid') is True and reference.get('same_render_pass') is True)
    if source=='stabilized' and not trusted:
        raise ValueError('stabilized Agent films require a verified live original-reference camera')
    if trusted:
        return [
            dict(stream='overview',title='PhysX 整机实录（同步实测）',agent_input=False),
            dict(stream='reference',title='参考相机（原参考轨迹）',agent_input=False),
            dict(stream=source,title='增稳相机（Agent输入）' if source=='stabilized' else '原始相机（Agent输入）',agent_input=True)]
    return [
        dict(stream='overview',title='PhysX 整机实录（同步实测）',agent_input=False),
        dict(stream='execution',title='实际相机（Agent 默认输入）',agent_input=True),
        dict(stream='stabilized',title='虚拟增稳相机（仅诊断）',agent_input=False)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=int,choices=(8,16,24),default=16)
    parser.add_argument('--acceptance-policy',choices=('recorded','user_20261017'),default='user_20261017')
    args=parser.parse_args()
    for folder in args.runs:
        if renderer.read(folder/'summary.json').get('backend')!='isaac_physx':
            parser.error('this adapter accepts measured PhysX recordings only')
    original_render=renderer.render_run
    original_frame=renderer.draw_frame
    original_text=renderer.txt
    labels={}

    def physx_text(draw,xy,value,*pos,**kw):
        value=str(value)
        if args.acceptance_policy=='user_20261017':
            if value.startswith('已记录物理子步约束：'):
                value='原始物理审计保留；加速度仅诊断，不计失败'
            elif value.startswith('仿真时间回放'):
                value+='；加速度仅诊断'
        return original_text(draw,xy,labels.get(str(value),value),*pos,**kw)

    def render_measured(run,write,timeline,preview):
        if args.acceptance_policy=='user_20261017':
            run.summary=assess_physx(run.summary,run.rows,run.api)
        if run.summary.get('realtime') and not any(a.get('response') for a in run.api):
            run.model=str(run.model)+'（配置值；无模型响应）'
        source=renderer.read(run.folder/'video/camera_output.json')['source']
        reference_file=run.folder/'video/reference_rendering.json'
        reference=renderer.read(reference_file) if reference_file.exists() else {}
        views=film_views(source,reference)
        labels.clear()
        labels.update({
            '原始参考相机画面':views[0]['title'],
            '原始实际执行相机（中间检查）':views[1]['title'],
            '增稳相机（默认输出）':views[2]['title'],
            '结果摘要（独立预览图）；不插入视频，不重新判定实验结果':
                '结果摘要（独立预览图）；按用户验收口径；不插入视频'})
        if source=='stabilized':
            labels['实际执行']='增稳输出'
        if run.summary.get('physx_reporting_policy',{}).get('network_invalid_trial'):
            labels['本段结果：镜头未达标  |  修复安装 0 次']='本段结果：网络超时，本次不计失败'
        for slot,view in zip(('reference','execution','stabilized'),views):
            if slot==view['stream']:continue
            for i in range(len(run.frames)):
                path=run.folder/'video'/view['stream']/f'{i:05d}.jpg'
                with Image.open(path) as image:
                    image=image.convert('RGB');image.thumbnail((624,300))
                    picture=Image.new('RGB',(624,300),'#0b111c')
                    picture.paste(image,((624-image.width)//2,(300-image.height)//2))
                run.image_cache[(i,slot)]=picture
        result=original_render(run,write,timeline,preview)
        result['physx_rgb_views']=views
        result['reference_camera_audit']=reference
        for filename in ('runtime_events.jsonl','realtime_timing.json'):
            if (run.folder/filename).exists():
                result['input_sha256'][filename]=renderer.sha(run.folder/filename)
        return result

    def runtime_frame(run,t,*pos,**kw):
        frame=original_frame(run,t,*pos,**kw)
        if not run.summary.get('realtime'):
            return frame
        draw=ImageDraw.Draw(frame)
        row=run.rows[max(0,np.searchsorted(run.times,t,side='right')-1)]
        fault=row['info'].get('runtime_fault')
        emergency=row['info'].get('emergency_stop_required')
        draw.rectangle((899,51,1919,88),fill=renderer.BG)
        camera_metadata=renderer.read(run.folder/'video/camera_output.json')
        replay=camera_metadata.get('capture_mode')=='offline_measured_state_replay'
        caption=('无录制实测状态 · 离线重绘 · 实时结论见执行日志' if replay else
                 'pace=1 · 按仿真时间连续回放 · 不代表墙钟实时')
        original_text(draw,(900,53),caption,21,renderer.GOLD,width=990)
        if replay:
            index=max(0,np.searchsorted(run.frame_times,t+1e-9,side='right')-1)
            draw.rectangle((16,483,750,510),fill=renderer.BG)
            original_text(draw,(20,484),f"实测状态时刻 {run.frames[index]['t']:.2f} s；RGB 离线重绘",17,renderer.MUTED)
        draw.rectangle((16,1020,1040,1077),fill=renderer.BG)
        status=('停车保护不足，需要外部安全介入' if emergency else '故障锁存，驱动减速' if fault else '在线运行，无轨迹等待')
        reason=(emergency or fault or {}).get('reason','')
        original_text(draw,(20,1024),status+(' | '+reason if reason else ''),20,
                      renderer.RED if fault else renderer.GREEN,width=1000)
        timing=run.summary['realtime']
        draw.rectangle((1040,1020,1919,1079),fill=renderer.BG)
        original_text(draw,(1050,1024),
            f"执行墙钟 {timing['execution_wall_s']:.2f}s · 仿真/墙钟 {timing['sim_wall_ratio']:.2f} · 无讲解停留",
            18,renderer.MUTED,width=850)
        return frame

    renderer.FPS=args.fps
    renderer.txt=physx_text
    renderer.render_run=render_measured
    renderer.draw_frame=runtime_frame
    try:
        renderer.render_explicit_runs(args.runs,args.output,representation_panel=True)
    finally:
        renderer.txt=original_text
        renderer.render_run=original_render
        renderer.draw_frame=original_frame
    manifest=args.output.with_suffix('.json')
    report=json.loads(manifest.read_text())
    report.update(physx_presentation_adapter_sha256=renderer.sha(__file__),
        physx_acceptance_policy=(POLICY_VERSION if args.acceptance_policy=='user_20261017' else 'recorded'),
        physx_acceptance_source_sha256=renderer.sha(Path(__file__).with_name('physx_acceptance.py')),
        reference_rgb_displayed=all(any(v['stream']=='reference' for v in r['physx_rgb_views']) for r in report['runs']),
        physical_camera_rgb_displayed=any(any(v['stream']=='execution' for v in r['physx_rgb_views']) for r in report['runs']),
        physical_state_modified=False,source_streams_modified=False)
    manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
