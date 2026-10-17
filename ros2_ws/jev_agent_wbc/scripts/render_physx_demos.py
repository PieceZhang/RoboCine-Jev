#!/usr/bin/env python3
"""PhysX presentation adapter around the current Representation film renderer.

Display measured overview, live original-reference floating camera and stabilized
Agent camera. Legacy recordings without a trusted live reference retain their
explicit old view layout. All source streams and shared render code are preserved.
"""
import argparse
import json
from pathlib import Path

from PIL import Image
import render_agent_demos as renderer
from physx_acceptance import assess_physx, POLICY_VERSION


def film_views(source,reference):
    trusted=(reference.get('protocol')=='original_reference_floating_camera_same_scene_time/v1'
             and reference.get('valid') is True and reference.get('same_render_pass') is True)
    if source=='stabilized' and not trusted:
        raise ValueError('stabilized Agent films require a verified live original-reference camera')
    if trusted:
        return [
            dict(stream='overview',title='PhysX 整机实录（同步实测）',agent_input=False),
            dict(stream='reference',title='参考相机（原参考轨迹）',agent_input=False),
            dict(stream='stabilized',title='增稳相机（Agent输入）' if source=='stabilized' else '增稳相机（诊断）',agent_input=source=='stabilized')]
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
        return result

    renderer.FPS=args.fps
    renderer.txt=physx_text
    renderer.render_run=render_measured
    try:
        renderer.render_explicit_runs(args.runs,args.output,representation_panel=True)
    finally:
        renderer.txt=original_text
        renderer.render_run=original_render
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
