#!/usr/bin/env python3
"""Summarize the final Isaac suite, with shot geometry and actual base/camera paths."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jvs_mock.wbc_trajectory import CameraTrajectory
from trajectory_record import resolve_recorded_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results')
    parser.add_argument('--write-trajectory-index', action='store_true',
                        help='Explicitly replace the current index; default only writes the historical report')
    args = parser.parse_args()
    root = Path(args.results).resolve()
    package = Path(__file__).resolve().parents[1]
    stage = package.parents[1]/'cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda'
    locations = [('shot_combo', root/'final_original_v3/shot_combo')]
    locations += [(f'shot_combo{i}', root/f'final/combos/shot_combo{i}') for i in (1, 2, 3)]
    locations += [(name, root/'final/single_actions'/name) for name in
                  ('shot_moveout', 'shot_orbit', 'shot_pan', 'shot_moveout_forward', 'shot_pan_limit')]
    reports, datasets = [], {}
    for name, folder in locations:
        summary = json.loads((folder/'tracking_summary.json').read_text())
        metadata = json.loads((folder/'run_metadata.json').read_text())
        reference = json.loads((folder/'reference.json').read_text())
        trajectory_file = resolve_recorded_trajectory(package, metadata, name, folder)
        digest = hashlib.sha256(trajectory_file.read_bytes()).hexdigest()
        if digest != metadata['trajectory_sha256']:
            raise ValueError(f'{name}: trajectory changed since validation')
        for relative, expected in metadata['source_sha256'].items():
            if hashlib.sha256((package/relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f'{name}: controller source changed since validation: {relative}')
        document = json.loads(trajectory_file.read_text())
        spawn = document.get('initial_base_pose', document['scene_context']['assumed_spawn'])
        record = dict(name=name, trajectory_sha256=digest, initial_base_pose=spawn,
            result_directory=str(folder.relative_to(package)), **summary)
        reports.append(record)
        with (folder/'telemetry.csv').open() as stream:
            rows = [row for row in csv.DictReader(stream) if row['phase'] == 'tracking']
        wall = np.array([float(row['wall_elapsed_s']) for row in rows])
        dt = np.maximum(np.diff(wall, prepend=wall[0]), 0.)
        for key in ('position_error', 'orientation_error'):
            values = np.array([float(row[key]) for row in rows])
            record['time_weighted_'+key+'_rms'] = float(np.sqrt(np.average(values**2, weights=dt)))
        trajectory = CameraTrajectory(trajectory_file, np.array(reference['initial_camera_position']),
            np.array(reference['initial_camera_rotation']), np.array(reference['initial_base_rotation']))
        datasets[name] = rows, trajectory, spawn

    manifest = dict(format='jvs_shot_validation/v1', scene=str(stage),
        scene_sha256=hashlib.sha256(stage.read_bytes()).hexdigest(),
        controller_source_sha256=metadata['source_sha256'],
        plant='Isaac Sim / PhysX', verification_scope='control_rate_samples_only',
        accuracy_thresholds=dict(position_rms_m=.1, orientation_rms_rad=.1),
        approximate_criteria=dict(completion_ratio=[.7, 1.3], focus_rms_rad=.15,
            reverse_motion='Active motion window: 0.02 rad / 0.05 m + 25% requested motion; 0.5 s sampling, 0.002 noise allowance',
            command_constraint_tolerance=1e-6, minimum_command_sweep_m=.025,
            camera_sample_step_m=.1, camera_sample_step_rad=.2),
        results=reports)
    (root/'validation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Keep a small, versioned verification index alongside the trajectory files.
    keys = ('name', 'trajectory_sha256', 'initial_base_pose', 'result_directory', 'completed',
            'tracking_accuracy_pass', 'approximate_success', 'position_error_rms',
            'orientation_error_rms', 'arm_reach_p95_m', 'minimum_command_clearance_m', 'shots')
    index = dict(manifest, results=[{key: row[key] for key in keys} for row in reports])
    if args.write_trajectory_index:
        (package/'trajectories/shot_validation.json').write_text(json.dumps(index, indent=2)+'\n')

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/jvs_shot_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    mesh = np.load(root/'original_exit/shot_combo/scene.kc_mesh.npz')
    centers = mesh['surfel_centers']
    centers = centers[(centers[:, 2] > .03)&(centers[:, 2] < 1.2)]
    for ax in axes[:2]:
        ax.scatter(5.+centers[:, 1], 1.+centers[:, 0], s=2, color='.80')
        ax.scatter([4.725], [3.72], marker='*', s=110, color='crimson', label='Kid')
        ax.set(xlabel='Scene X (m)', ylabel='Scene Z (m)', xlim=(2.6, 7.5), ylim=(-.4, 4.1))
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
    for index, name in enumerate(['shot_combo', 'shot_combo1', 'shot_combo2', 'shot_combo3']):
        rows, trajectory, spawn = datasets[name]
        t = np.array([float(row['time_s']) for row in rows])
        camera = np.array([[float(row[f'camera_{a}']) for a in 'xyz'] for row in rows])
        base = np.array([[float(row[f'state_{a}']) for a in 'xy'] for row in rows])
        reference = np.array([trajectory.sample(v)[0] for v in t])
        a = spawn['yaw']
        transform = np.array([[np.sin(a), np.cos(a)], [np.cos(a), -np.sin(a)]])
        to_scene = lambda values: np.array([spawn['x'], spawn['z']])+values[:, :2]@transform.T
        ax = axes[0 if index == 0 else 1]
        color = f'C{index}'
        for values, style, label in ((reference, '--', name+' planned'), (camera, '-', name+' camera'),
                                      (base, ':', name+' base')):
            path = to_scene(values)
            ax.plot(path[:, 0], path[:, 1], style, color=color, lw=1.3, label=label)
        start = to_scene(base[:1])[0]
        ax.scatter(*start, color=color, marker='o', s=25)
        reach = np.array([float(row['arm_horizontal_reach_m']) for row in rows])
        axes[2].plot(t, reach, color=color, lw=1., label=name)
    axes[0].set_title('Original furniture-edge combination')
    axes[1].set_title('Three different initial poses')
    axes[2].set(title='Arm horizontal reach', xlabel='Shot path time (s)', ylabel='Reach (m)')
    axes[2].axhline(.25, color='gray', ls='--', label='Comfort reference 0.25 m')
    for ax in axes:
        ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(root/'validation_paths.png', dpi=170)

    lines = ['# Kid 场景运镜验证', '',
        '所有结果来自独立 ROS domain 209 中的真实 Isaac/PhysX 执行；不是理想运动学回放。',
        '原始组合与最终八项回归均核对轨迹及控制器源码 SHA-256。', '',
        '| 轨迹 | 位置 RMS (m) | 姿态 RMS (rad) | 臂伸出 P95 (m) | 净空下界 (m) | 精度通过 | 近似成功 |',
        '|---|---:|---:|---:|---:|---|---|']
    for row in reports:
        lines.append(f"| {row['name']} | {row['position_error_rms']:.4f} | {row['orientation_error_rms']:.4f} | "
            f"{row['arm_reach_p95_m']:.4f} | {row['minimum_command_clearance_m']:.4f} | "
            f"{'是' if row['tracking_accuracy_pass'] else '否'} | {'是' if row['approximate_success'] else '否'} |")
    lines += ['', '## 验收口径', '',
        '精度门槛为位置 RMS ≤ 0.1 m、姿态 RMS ≤ 0.1 rad。误差仍相对于原始几何轨迹计算；',
        '自适应时序允许执行比标称时长更久，报告另列实际仿真执行时间。',
        '近似成功还要求全部动作的转角/径向位移达到计划量的 70%～130%，反向移动量受限，',
        'kid 构图误差 RMS ≤ 0.15 rad，命令连续且符合硬约束，采样相机位姿连续，扫掠净空 ≥ 25 mm。',
        '这些额外比例和连续性阈值是本次采用的工程判据；不是完整物理步认证。', '',
        '所有运行仍保留实测速度、加速度、轮加速度和侧滑违例数据。`stage1_pass=false`；',
        '不能据此宣称实际执行器在每个 PhysX 子步均满足限制。', '',
        '## 文件', '',
        '- `validation_manifest.json`：逐动作完成量、反向量、关节余量、耗时及验收结果。',
        '- `validation_paths.png`：场景中的相机/底盘轨迹及机械臂伸出距离。',
        '- 各运行目录：原始 `telemetry.csv`、`tracking_summary.json`、`reference.json`、日志和源码哈希。',
        '- `original_combo/`、早期 `single_actions/`、`combos/` 保留调试版结果，最终验收见上表。', '',
        '重新汇总：`python3 scripts/analyze_shot_validation.py results/kc_shots_20260922`。', '']
    (root/'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:len(reports)+7]))


if __name__ == '__main__':
    main()
