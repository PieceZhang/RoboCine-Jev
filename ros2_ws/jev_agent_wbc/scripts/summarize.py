#!/usr/bin/env python3
"""Produce auditable CSV/Markdown comparisons from saved, individually scored runs."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(root):
    records = []
    for file in sorted(root.glob('*/summary.json')):
        data = json.loads(file.read_text())
        visual = data.get('visual_metrics', {})
        calls = []
        api = file.parent/'jev_api.jsonl'
        if api.exists():
            calls = [json.loads(line) for line in api.read_text().splitlines()]
        permissions, installs = [], []
        events = file.parent/'events.jsonl'
        if events.exists():
            for line in events.read_text().splitlines():
                value = json.loads(line)
                if value['event'] == 'jev_permission': permissions.append(value)
                if value['event'] == 'plan_installed': installs.append(value)
        row = dict(case=data.get('case', file.parent.name.rsplit('_', 1)[0]), mode=data['mode'],
            completed=data['completed'], safety_pass=data.get('safety_pass', False),
            budget_success=data.get('within_budget_success', False), elapsed_s=data.get('elapsed_shot_s'),
            composition_mean=visual.get('composition_error', {}).get('mean'),
            scale_relative_mean=visual.get('subject_scale_error', {}).get('mean'),
            angle_rad_mean=visual.get('observation_angle_error_rad', {}).get('mean'),
            jerk_rms=data.get('visual_smoothness', {}).get('translation_jerk_rms_m_s3'),
            angular_acceleration_rms=data.get('visual_smoothness', {}).get('angular_acceleration_rms_rad_s2'),
            min_clearance_m=data.get('min_truth_clearance_m'), contacts=data.get('contacts'),
            joint_margin_rad=data.get('min_joint_margin_rad'), position_rms_m=data.get('original_position_rms_m'),
            phases_completed=sum(p['completed'] for p in data.get('phases', [])),
            phases_total=len(data.get('phases', [])),
            planning_batches=data.get('validation_attempts', 0), feasible_batches=data.get('feasible_plans', 0),
            installed=data.get('installed_plans', 0), nonzero_installed=sum(any(abs(v)>1e-7 for v in e['actual_parameters'].values()) for e in installs),
            permissions=len(permissions), api_calls=len(calls),
            api_latency_mean_s=statistics.mean(c['wall_s'] for c in calls) if calls else None,
            input_tokens=sum(c.get('response', {}).get('usage', {}).get('input_tokens', 0) for c in calls),
            invalid_masks=visual.get('segmentation_failure_frames'), occluded=visual.get('occluded_frames'),
            out_of_frame=visual.get('out_of_frame_frames'),
            physical_pass=(data.get('physical_audit') or {}).get('pass_constraints'),
            failure=data.get('failure', ''), folder=file.parent.name,
            evaluation_sha256=data.get('evaluation_sha256'))
        records.append(row)
    if not records:
        raise ValueError('no case summaries')
    with (root/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    def fmt(value):
        if value is None: return '—'
        if isinstance(value, bool): return '通过' if value else '未通过'
        return f'{value:.4f}' if isinstance(value, float) else str(value)
    lines = ['# 单镜头评估记录', '', f'目录：`{root.resolve()}`。所有结果按各目录保存的原始时刻与评价协议计算。', '',
        '“执行完成”表示运行到原始预算终点；“预算成功”还要求物理安全与固定阶段终点判据。中断运行的均值只覆盖已有帧，不能与完整镜头均值直接认定优劣。', '',
        '| 场景 | 模式 | 执行完成 | 安全 | 预算成功 | 构图均值 | 尺度相对误差 | 观察角 rad | jerk RMS | 角加速度 RMS | 非零安装 |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        fields = ['case', 'mode', 'completed', 'safety_pass', 'budget_success', 'composition_mean',
                  'scale_relative_mean', 'angle_rad_mean', 'jerk_rms', 'angular_acceleration_rms', 'nonzero_installed']
        lines.append('| '+' | '.join(fmt(r[k]) for k in fields)+' |')
    lines += ['', '## 计数与限制', '']
    for mode in ('strict', 'jev'):
        rs = [r for r in records if r['mode'] == mode]
        if not rs: continue
        attempts = sum(r['planning_batches'] for r in rs)
        feasible = sum(r['feasible_batches'] for r in rs)
        lines.append(f'- {mode}: {len(rs)} 组，完成 {sum(r["completed"] for r in rs)}，安全通过 {sum(r["safety_pass"] for r in rs)}，'
            f'预算成功 {sum(r["budget_success"] for r in rs)}；数学规划可行 {feasible}/{attempts} 批次，'
            f'安装 {sum(r["installed"] for r in rs)} 次，其中非零 {sum(r["nonzero_installed"] for r in rs)} 次。')
    lines += ['', '## 失败与审核入口', '']
    for r in records:
        lines.append(f'- [{r["folder"]}]({r["folder"]}/summary.json)：'
            f'{r["failure"] or "执行无异常"}；阶段 {r["phases_completed"]}/{r["phases_total"]}；'
            f'实测物理限制 {fmt(r["physical_pass"])}；最小净距 {fmt(r["min_clearance_m"])} m。')
    lines += ['', '几何指标不等同于完整语义意图；Jev 权限不改变评分标准。API、逐候选预测、权限与实际用量见各组 events.jsonl / jev_api.jsonl。', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    (root/'comparison.json').write_text(json.dumps(records, indent=2, ensure_ascii=False)+'\n')
    print(root/'REPORT.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path)
    summarize(parser.parse_args().runs)
