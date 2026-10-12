#!/usr/bin/env python3
"""Summarize camera-shot precision and export reference/PhysX trajectory plots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def summarize_case(folder, trajectory_path):
    document = json.loads(trajectory_path.read_text())
    meta = document['shot_metadata']
    summary = json.loads((folder/'summary.json').read_text())
    with (folder/'telemetry.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    def column(key):
        return np.array([float(row[key]) for row in rows])
    t = column('time_s')
    motion = (t >= meta['motion_window_s'][0]) & (t <= meta['motion_window_s'][1])
    full = t <= document['duration_s']
    output = dict(metadata=meta, completed=summary['completed'], failure=summary['failure'],
                  source_summary='summary.json', control_samples=int(full.sum()),
                  motion_samples=int(motion.sum()))
    for label, mask in [('motion', motion), ('full', full)]:
        if not mask.any():
            raise ValueError(f'{folder}: no {label} samples; simulation failed before motion')
        for key, factor in [('position_error', 1000.), ('orientation_error', 180./np.pi)]:
            values = column(key)[mask]*factor
            output[f'{label}_{key}'] = dict(rms=float(np.sqrt(np.mean(values**2))),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()))
    pe, re = output['motion_position_error'], output['motion_orientation_error']
    output['tracking_accuracy_pass'] = bool(summary['completed'] and not summary['failure']
        and pe['rms'] <= 20. and pe['maximum'] <= 50.
        and re['rms'] <= np.rad2deg(.035) and re['maximum'] <= np.rad2deg(.0873))
    output['units'] = dict(position_error='mm', orientation_error='deg')
    output['thresholds'] = dict(position_rms_mm=20., position_max_mm=50.,
        orientation_rms_deg=float(np.rad2deg(.035)), orientation_max_deg=float(np.rad2deg(.0873)))
    output['max_qp_violation'] = summary['max_qp_violation']
    output['physics_audit_pass'] = summary['physics_audit']['pass_constraints']
    target = np.column_stack([column('target_camera_'+axis) for axis in 'xyz'])
    actual = np.column_stack([column('actual_camera_'+axis) for axis in 'xyz'])
    rt = Rotation.from_quat(np.column_stack([column('target_camera_q'+axis) for axis in 'xyzw']))
    ra = Rotation.from_quat(np.column_stack([column('actual_camera_q'+axis) for axis in 'xyzw']))
    # Express paths and orientations in the captured initial camera frame.
    target_local, actual_local = (target-target[0]) @ rt[0].as_matrix(), (actual-target[0]) @ rt[0].as_matrix()
    target_angles = (rt[0].inv()*rt).as_euler('xyz', degrees=True)
    actual_angles = (rt[0].inv()*ra).as_euler('xyz', degrees=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(meta['name']+' — Isaac Sim camera trajectory tracking')
    ax = axes[0, 0]
    ax.plot(target_local[full, 0], target_local[full, 1], '--', label='reference')
    ax.plot(actual_local[full, 0], actual_local[full, 1], linewidth=1, label='PhysX camera')
    ax.scatter([target_local[0, 0]], [target_local[0, 1]], marker='o', label='start')
    ax.set(xlabel='Initial camera X (m)', ylabel='Initial camera Y (m)', title='Camera path (XY)')
    ax.set_aspect('equal', adjustable='datalim')
    for i, (name, color) in enumerate(zip('xyz', ['tab:blue', 'tab:orange', 'tab:green'])):
        axes[0, 1].plot(t, target_local[:, i], '--', color=color, label=f'{name} ref')
        axes[0, 1].plot(t, actual_local[:, i], color=color, linewidth=1, label=f'{name} actual')
        axes[1, 0].plot(t, target_angles[:, i], '--', color=color, label=f'{["roll", "pitch", "yaw"][i]} ref')
        axes[1, 0].plot(t, actual_angles[:, i], color=color, linewidth=1, label=f'{["roll", "pitch", "yaw"][i]} actual')
    axes[0, 1].set(ylabel='Camera displacement (m)', title='Position reference / actual')
    axes[1, 0].set(ylabel='Angle (deg)', title='Orientation reference / actual')
    axes[0, 2].plot(t, column('position_error')*1000, label='position error')
    axes[0, 2].set(ylabel='Position error (mm)', title=f'Motion RMS {pe["rms"]:.3f} mm')
    axes[1, 1].plot(t, np.rad2deg(column('orientation_error')), label='SO(3) error')
    axes[1, 1].set(ylabel='Orientation error (deg)', title=f'Motion RMS {re["rms"]:.3f} deg')
    for name in ['v', 'omega']:
        axes[1, 2].plot(t, column('command_'+name), '--', label=f'{name} command')
        axes[1, 2].plot(t, column('measured_'+name), linewidth=1, label=f'{name} actual')
    axes[1, 2].set(ylabel='m/s, rad/s', title='Base motion')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
        if ax is not axes[0, 0]:
            ax.set_xlabel('Simulation time (s)')
            ax.axvspan(*meta['motion_window_s'], alpha=.06, color='green')
    fig.savefig(folder/'camera_tracking.png', dpi=150)
    plt.close(fig)
    (folder/'tracking_accuracy.json').write_text(json.dumps(output, indent=2)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    manifest = json.loads((package/'trajectories/camera_shots_manifest.json').read_text())
    results = {}
    for case in manifest['cases']:
        results[case['name']] = summarize_case(args.results/case['name'], package/'trajectories'/case['trajectory'])
    (args.results/'tracking_accuracy.json').write_text(json.dumps(results, indent=2)+'\n')
    passed = all(r['tracking_accuracy_pass'] for r in results.values())
    lines = ['# 常规运镜轨迹跟踪测试', '',
        '本轮只测试常规 orbit、pan、moveout 运镜，不运行极限压力测试。', '',
        f'**结论：六条轨迹的运动区间跟踪精度检查{"全部通过" if passed else "未全部通过"}。**', '',
        'Isaac Sim 5.1 headless，空场景与地面接触，原 JVS URDF，PGS，物理 1000 Hz、QP 50 Hz。'
        '相机实际位姿直接读取 PhysX camera_link 刚体；没有修改 WBC 分配权重、伺服增益或约束限值。', '',
        '每条轨迹首尾各静止 2 s。下表只统计运动区间，误差单位为 mm / deg；姿态误差为 SO(3) 最短旋转角。', '',
        '| 轨迹 | 参数 | 时长 / 运动 (s) | 位置 RMS / P95 / 最大 (mm) | 姿态 RMS / P95 / 最大 (deg) | 精度检查 |',
        '|---|---|---:|---:|---:|---|']
    for case in manifest['cases']:
        name = case['name']
        r = results[name]
        meta = r['metadata']
        parameter = (f'半径 {meta["radius_m"]} m，{meta["angle_deg"]:+.0f}°' if meta['shot']=='orbit' else
                     f'{meta["angle_deg"]:+.0f}°' if meta['shot']=='pan' else f'后退 {meta["distance_m"]} m')
        p, o = r['motion_position_error'], r['motion_orientation_error']
        lines.append(f'| [{name}]({name}/camera_tracking.png) | {parameter} | {meta["duration"]:.0f} / {meta["duration"]-4:.0f} | '
            f'{p["rms"]:.3f} / {p["p95"]:.3f} / {p["maximum"]:.3f} | '
            f'{o["rms"]:.3f} / {o["p95"]:.3f} / {o["maximum"]:.3f} | {"通过" if r["tracking_accuracy_pass"] else "未通过"} |')
    lines += ['', '精度门槛沿用前次定义：位置 RMS ≤20 mm、最大 ≤50 mm；姿态 RMS ≤0.035 rad（2.005°）、'
              '最大 ≤0.0873 rad（5.002°）。全部行也必须完整执行且无控制器 fault。', '',
              '运镜定义：初始 camera_link 的 +X 为视线、+Y 为左、+Z 为上。orbit 的主体固定在初始相机前方，'
              '位置沿定半径圆弧运动且朝向始终指向主体；pan 固定位置只改变 yaw；moveout 沿 -X 后退，姿态不变。'
              '轨迹直接定义于相机空间，采用五次平滑时间律，端点速度和加速度为零。', '',
              '每个目录中的 `camera_tracking.png` 对比目标和 PhysX 实际位置/姿态；`tracking_accuracy.json` '
              '同时包含运动区间和全时段统计，`telemetry.csv` 保存两套相机位姿、误差和控制量，'
              '`physics.csv.gz` 保留逐物理步约束诊断，`summary.json` 保留原完整验收字段。', '',
              '本报告判断的是常规运镜跟踪精度。它不将精度通过等同于全部物理加速度约束通过，'
              '也不要求 pan 等运镜一定同时使用 base 和 arm。既有逐物理步诊断保留为补充信息。', '',
              '复现命令见包内 `CAMERA_SHOTS.md`。', '']
    ros_path = args.results/'ros_orbit_ccw10'/'summary.json'
    if ros_path.exists():
        ros = json.loads(ros_path.read_text())
        if ros.get('completed') and 'position_error_rms' in ros:
            lines += ['## ROS launch 补充验证', '',
                f'`jvs_wbc_isaac.launch.py` 加载同一 orbit_ccw10 文件完整运行。'
                f'包含首尾静止段的全时段位置 RMS {ros["position_error_rms"]*1000:.3f} mm，'
                f'姿态 RMS {np.rad2deg(ros["orientation_error_rms"]):.3f}°。'
                '该补充试验通过实际 ROS Bridge 收发状态与命令，ROS 节点相机误差用实测 odometry + q 的 FK 计算；'
                '上表六条独立测试使用直接 PhysX 相机刚体数据。', '',
                '[ROS 原始报告](ros_orbit_ccw10/summary.json)。', '']
        else:
            lines += ['## ROS launch 补充验证', '',
                      f'未完成：{ros.get("failure", "unknown")}；详情见 ros_orbit_ccw10/launch.log。', '']
    (args.results/'REPORT.md').write_text('\n'.join(lines))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
