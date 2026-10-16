#!/usr/bin/env python3
"""Compare official closed-loop results with a specified historical adapter run."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def latency(records, field='wall_s'):
    values = [r[field] for r in records if 'error' not in r]
    return dict(successes=len(values), errors=sum('error' in r for r in records),
                median_s=float(np.median(values)) if values else None,
                p95_s=float(np.percentile(values,95)) if values else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--official', type=Path, required=True)
    p.add_argument('--paired', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    rows=[]; apis={'perplexity':[], 'official':[]}; failures={}
    names=['orbit','movein','truck_left','truck_right','pedestal_up','pedestal_down']
    for name in names:
        before=a.baseline/(name+'_jev'); after=a.official/name/(name+'_jev')
        for file in ('original_task.json','runtime_scene.json'):
            assert read(before/file)==read(after/file), (name,file)
        for label,folder in [('perplexity',before),('official',after)]:
            s=read(folder/'summary.json'); intent=s['executed_camera_intent']
            api=lines(folder/'jev_api.jsonl');apis[label].extend(api)
            events=lines(folder/'events.jsonl')
            failures[label+':'+name]=dict(Counter(e.get('error','') for e in events if e['event']=='proposal_rejected'))
            rows.append(dict(case=name,backend=label,completed=s['completed'],safety=s['safety_pass'],
                intent=s['intent_success'],installed=s['installed_plans'],
                visible_coverage=intent['effective_coverage'],terminal=intent['terminal_frame']['pass_check'],
                trend=intent['motion_trend']['pass_check'],range_complete=intent['range_complete'],
                occlusion_s=intent['longest_occlusion_s'],min_clearance_m=s['min_truth_clearance_m'],
                original_position_rms_m=s['original_position_rms_m'],api_calls=len(api),
                api_errors=sum('error' in x for x in api),wall_s=s['wall_s']))
    totals={}
    for label in apis:
        selected=[r for r in rows if r['backend']==label]
        totals[label]={k:sum(r[k] for r in selected) for k in ('completed','safety','intent','terminal','trend','installed')}
        totals[label].update(installed_scenes=sum(r['installed']>0 for r in selected),
                             api=latency(apis[label]),models=sorted({r['response']['model'] for r in apis[label] if 'response' in r}))
    paired=lines(a.paired);agreements=Counter();counts=Counter()
    for r in paired:
        if r.get('baseline_response') and r.get('response'):
            assert r['response']['model']=='jev-1.14.0'
            for key,answer in r['baseline_response']['answers'].items():
                if answer['type']=='choice':
                    counts[key]+=1
                    agreements[key]+=answer['choice']==r['response']['answers'][key]['choice']
    report=dict(baseline=str(a.baseline.resolve()),official=str(a.official.resolve()),rows=rows,
                totals=totals,rejections=failures,paired=dict(calls=len(paired),official_latency=latency(paired),
                baseline_latency=latency([dict(wall_s=r['baseline_wall_s'],**({'error':r['baseline_error']} if r['baseline_error'] else {})) for r in paired]),
                agreement={k:dict(matches=agreements[k],total=counts[k]) for k in counts}),
                limitations=['one run per scene/backend; historical rather than concurrent baseline',
                    'slow CPU integration, pace=.03, six concurrent scenes; not wall-clock realtime or PhysX',
                    'different choices change later states; exact-input replay is diagnostic only',
                    'choice agreement is not correctness; latency includes network/TLS and host load'])
    (a.output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    with (a.output/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    texts=['TypeSafe 官方 Jev 1.14 与 Perplexity 兼容 Agent 对比（2026-10-15）','',
        '固定模型 jev-1.14.0；官方 https://api.typesafe.ai/v1/systemone。',
        '最新版本依据：https://docs.typesafe.ai/models；实际返回模型已在每次响应中核验。',
        '鉴权从给定密钥文件读取；结果不保存 Authorization 或密钥内容。','',
        '六场景沿用历史最终版：CPU 整机、WBC 避障开启、3 盒几何表示、几何偏好开启，',
        'pace=.03、激活提前量1.2秒、候选协作预算20秒、六进程并行。',
        '每场景8秒镜头及0.4秒终帧确认；评分与安全门槛未修改。','',
        '指标 | 历史 Perplexity | 官方 Jev 1.14']
    b=totals['perplexity'];c=totals['official']
    for label,key in [('运行完成','completed'),('CPU 安全通过','safety'),('安装新轨迹的场景','installed_scenes'),
                      ('完整镜头意图通过','intent'),('终帧通过','terminal'),('过程趋势通过','trend')]:
        texts.append(f'{label} | {b[key]}/6 | {c[key]}/6')
    for label,key in [('轨迹安装总数','installed')]:texts.append(f'{label} | {b[key]} | {c[key]}')
    for label,x in totals.items():texts.append(f'{label} API：{x["api"]["successes"]}成功/{x["api"]["errors"]}错误；中位{x["api"]["median_s"]:.3f}s，P95 {x["api"]["p95_s"]:.3f}s。')
    texts+=['','逐场景：后端 | 场景 | 安装数 | 可见覆盖 | 终帧 | 趋势 | 最长遮挡s | 完整意图']
    for r in rows:texts.append(f'{r["backend"]} | {r["case"]} | {r["installed"]} | {r["visible_coverage"]:.1%} | {r["terminal"]} | {r["trend"]} | {r["occlusion_s"]:.2f} | {r["intent"]}')
    texts+=['','相同输入回放：仅替换 model，历史 state/questions 完全一致；不执行机器人。',json.dumps(report['paired'],ensure_ascii=False,indent=2),'',
        '限制：每个后端每场景一次，历史基线并非同时重跑。并发负载、网络、模型采样及后续',
        '状态差异影响结果；不能据此做统计显著排名。选择一致率不是正确率。',
        '视频为真实执行日志的离线 CPU 示意回放，无讲解停留；不是屏幕录制、RGB、PhysX或实机验收。',
        '安装新轨迹不等于镜头完成；终帧通过也不等于完整意图通过。','',
        '历史根目录：'+str(a.baseline.resolve()),'官方根目录：'+str(a.official.resolve()),
        '原始响应、事件、传感器观测、遥测、冻结源码、各场景 provenance 均保留。']
    (a.output/'REPORT.txt').write_text('\n'.join(texts)+'\n')
    print(json.dumps(totals,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
