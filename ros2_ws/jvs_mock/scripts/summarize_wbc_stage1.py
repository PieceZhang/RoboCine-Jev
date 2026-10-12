#!/usr/bin/env python3
"""Create the Stage 1 review document from saved Isaac results (no simulation)."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1] / 'results'
    cases = [('正向 24 s', 'final/coordinated'),
             ('反向 28 s', 'final/reverse'),
             ('快速 8 s', 'final/fast'),
             ('压力 4 s', 'final/limit_stress'),
             ('ROS 快速 8 s', 'final/ros_fast')]
    reports = {name: json.loads((root/path/'summary.json').read_text()) for name, path in cases}
    lines = ['# Stage 1 Isaac Sim 验收报告', '',
        '日期：2026-09-17。平台：NVIDIA RTX 5090、Isaac Sim 5.1、ROS 2 Humble、OSQP 0.6.7.post3。', '',
        '**结论：Stage 1 未通过全部验收。** constrained QP、6D 跟踪、base/arm 协同运动和日志链路已实现；'
        '命令约束满足，但原始物理状态的加速度仍有超限，压力测试还可出现关节速度超限，因此不能宣称所有约束始终满足。', '',
        '最终 WBC launch 默认 PGS。独立测试 QP 50 Hz、物理 1000 Hz，控制与物理时钟差小于 2 μs；'
        '完整 ROS 测试使用真实 ROS Bridge。未使用运动学仿真代替 Isaac。'
        '位置和姿态误差分别为欧氏距离和 SO(3) 最短旋转角。', '',
        '## 跟踪与协同运动', '',
        '| 场景 | 位置 RMS / 最大 (mm) | 姿态 RMS / 最大 (deg) | base 净位移 (m) | 最大关节运动范围 (rad) | 完成 |',
        '|---|---:|---:|---:|---:|---|']
    for name, path in cases:
        r = reports[name]
        lines.append(f'| [{name}]({path}/summary.json) | {1e3*r["position_error_rms"]:.3f} / {1e3*r["position_error_max"]:.3f} | '
                     f'{np.rad2deg(r["orientation_error_rms"]):.3f} / {np.rad2deg(r["orientation_error_max"]):.3f} | '
                     f'{r["base_travel_m"]:.3f} | {r["arm_motion_rad"]:.3f} | {r["completed"]} |')
    lines += ['', '压力轨迹故意超过可及时执行的速度/加速度能力，期望保持执行器约束并允许相机跟踪误差增大；'
              '它不计入普通轨迹的精度通过判定。', '',
              '## 每个物理步的约束检查', '',
              '下表为原始速度差分得到的最大绝对加速度，未低通滤波。基线限值：'
              'wheel=20 rad/s²，joint=2 rad/s²，base linear=0.5 m/s²，base angular=1.5 rad/s²。', '',
              '| 场景 | 物理样本数 | wheel 加速度 | joint 加速度 | base 线加速度 | base 角加速度 | joint 速度超限 (rad/s) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, path in cases:
        if 'physics_audit' not in reports[name]:
            continue
        r = reports[name]
        with gzip.open(root/path/'physics.csv.gz', 'rt') as stream:
            rows = list(csv.DictReader(stream))
        keys = [f'measured_{i}' for i in range(8)] + ['wheel_0', 'wheel_1']
        values = np.array([[float(row[key]) for key in keys] for row in rows])
        acceleration = np.max(np.abs(np.diff(values, axis=0))/r['physics_audit']['physics_dt'], axis=0)
        speed_excess = np.max(np.maximum(np.abs(values[:, 2:8])-np.array([1., 1., 1., 1.2, 1.2, 1.5]), 0.))
        lines.append(f'| {name} | {len(rows)} | {max(acceleration[8:]):.3f} | {max(acceleration[2:8]):.3f} | {acceleration[0]:.3f} | {acceleration[1]:.3f} | {speed_excess:.5f} |')
    peak_qp = max(r['max_qp_violation'] for r in reports.values())
    peak_command = max(v for r in reports.values() for k, v in r.items() if k.startswith('max_command_') and 'violation' in k)
    peak_solve = max(r['max_solve_s'] for r in reports.values())
    normal = [reports[name] for name in ['正向 24 s', '反向 28 s', '快速 8 s', 'ROS 快速 8 s']]
    tracking_pass = all(r['completed'] and r['position_error_rms'] <= .02 and r['position_error_max'] <= .05
                        and r['orientation_error_rms'] <= .035 and r['orientation_error_max'] <= .0873 for r in normal)
    lines += ['', f'全部列出场景的最大 QP 残差 {peak_qp:.3g}，最大命令约束超限 {peak_command:.3g}，'
              f'均在 1e-6 数值容差以内。记录到的最大 QP 求解时间 {peak_solve*1000:.3f} ms。', '',
              'ROS 报告仅覆盖 ROS 状态采样时刻，不能代替每个物理步的检查。'
              'sampled_checks_pass 只说明采样检查通过；stage1_pass 还要求独立 physics_audit 通过。', '',
              '## 验收逐项判断', '',
              f'1. 普通无障碍 6D 轨迹跟踪：{"通过" if tracking_pass else "未通过"}预设误差门槛。',
              '2. base 与 arm 同时运动，QP 命令曲线连续；反向运动时可触及关节制动约束，控制器不独立裁剪分配结果。',
              '3. 命令 wheel/base/joint 约束满足；物理加速度有超限，**不通过**。',
              '4. PGS 降低了原 TGS 模型的速度抖动，未观察到控制发散；1 ms 原始速度仍有尖峰，**不能宣布完整通过**。',
              '5. 相机误差、base v/omega、joint velocity、wheel velocity、命令和实测 violation 均已输出保存。', '',
              '## 调试证据与限制', '',
              '- `acceptance/` 和 `ros_fast/` 是原 TGS 参数对照，物理速度抖动更明显。',
              '- `fast_pgs_verified/` 为正确显式设置 PGS 后的实验；`fast_pgs_servo/` 进一步降低伺服增益，'
              '仍未消除逐物理步加速度超限，因此最终默认保留原增益。',
              '- `debug_excluded/` 为排除的早期调试记录，包含时间基准偏差或参数被初始化覆盖等情况，不能作为验收证据。',
              '- 部分旧实验在完成并写出报告后，Kit 完整清理阶段发生 SIGSEGV；最终 runner 恢复原后端 fast shutdown，'
              '由外部测试套件读取报告判定结果。旧报告的 stage1_pass 可能仅检查 50 Hz；最终报告明确区分 sampled_checks_pass 与完整验收。',
              '- 压力测试曾暴露实测/命令使用相同内缩限位会耗尽伺服制动余量；最终实现对命令预留 0.01 rad，实测约束使用 URDF 硬限位。',
              '- 未放宽约束阈值、未删除运动中的尖峰、未用额外惯量实验作为最终默认配置。',
              '- 2 s 初始接触 settling 不属于跟踪验收；其后的完整轨迹和停止阶段都计入物理检查。',
              '- 速度级 QP 不包含刚体动力学、接触冲量或执行器加速度跟踪模型，'
              '因此可行的速度命令不构成实际关节加速度始终受限的证明。', '',
              '后续需要解决物理执行层：验证逐物理步的统一命令插值和位置/速度伺服一致性，'
              '必要时采用带执行器动力学的加速度级控制，再用同一严格物理验收复测。'
              '不能以更换为 KC-SVSDF 或增加 shot preference 来解决这里的执行器加速度问题。', '',
              '## 文件', '',
              '- [实现和运行说明](../WBC_STAGE1.md)',
              '- 每个实验目录下 `summary.json`、`telemetry.csv`、`physics.csv.gz`（独立物理测试）、启动日志和 `tracking.png`。',
              '- [正向曲线](final/coordinated/tracking.png)、[反向曲线](final/reverse/tracking.png)、'
              '[快速曲线](final/fast/tracking.png)、[ROS 曲线](final/ros_fast/tracking.png)。', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
