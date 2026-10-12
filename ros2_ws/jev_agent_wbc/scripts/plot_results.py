#!/usr/bin/env python3
"""Export fixed-reference comparison figures from recorded runs, without rescoring."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/jev_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = ('orbit', 'movein', 'truck_left', 'truck_right', 'pedestal_up', 'pedestal_down')
COLORS = {'strict': '#df7b32', 'jev': '#2878b5'}


def read(path):
    return json.loads(path.read_text())


def plot(root):
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    summaries = {}
    for name, ax in zip(NAMES, axes.flat):
        plane = (0, 2) if name.startswith('pedestal') else (0, 1)
        task = None
        for mode in ('strict', 'jev'):
            folder = root/f'{name}_{mode}'
            if not (folder/'summary.json').exists():
                continue
            summaries[name, mode] = summary = read(folder/'summary.json')
            rows = read(folder/'telemetry.json')
            if not rows:
                continue
            task = read(folder/'original_task.json')
            p = np.array([r['camera_position'] for r in rows])
            ax.plot(p[:, plane[0]], p[:, plane[1]], color=COLORS[mode],
                    label=f'{mode} ({rows[-1]["t"]:.2f} s)')
            ax.scatter(*p[-1, list(plane)], color=COLORS[mode], marker='x', s=35)
        if task:
            p = np.array([r['position'] for r in task['reference_camera_trajectory']['samples']])
            ax.plot(p[:, plane[0]], p[:, plane[1]], 'k--', linewidth=1.5, label='original (8 s)')
            ax.scatter(*p[0, list(plane)], color='black', s=20, marker='o')
        ax.set(title=name, xlabel='World X (m)', ylabel=('World Z (m)' if plane[1] == 2 else 'World Y (m)'))
        ax.axis('equal'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    figure.suptitle('Measured camera paths; original reference remains fixed')
    figure.savefig(root/'camera_paths.png', dpi=170)
    figure.savefig(root/'camera_paths.pdf')
    plt.close(figure)

    metrics = [
        ('Composition error', 'Normalized image distance', lambda d: d.get('visual_metrics', {}).get('composition_error', {}).get('mean')),
        ('Subject scale error', 'Absolute relative area error', lambda d: d.get('visual_metrics', {}).get('subject_scale_error', {}).get('mean')),
        ('Observation angle error', 'Radians', lambda d: d.get('visual_metrics', {}).get('observation_angle_error_rad', {}).get('mean')),
        ('Translation jerk RMS', 'm/s³', lambda d: d.get('visual_smoothness', {}).get('translation_jerk_rms_m_s3')),
        ('Angular acceleration RMS', 'rad/s²', lambda d: d.get('visual_smoothness', {}).get('angular_acceleration_rms_rad_s2')),
        ('Minimum joint margin', 'Radians (larger is better)', lambda d: d.get('min_joint_margin_rad')),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (title, unit, getter) in zip(axes.flat, metrics):
        for mode, offset in [('strict', -.18), ('jev', .18)]:
            values, unfinished = [], []
            for name in NAMES:
                summary = summaries.get((name, mode), {})
                value = getter(summary)
                values.append(np.nan if value is None else value)
                unfinished.append(not summary.get('completed', False))
            bars = ax.bar(np.arange(len(NAMES))+offset, values, .35, color=COLORS[mode], label=mode)
            for bar, short in zip(bars, unfinished):
                if short: bar.set_hatch('//')
        ax.set(title=title, ylabel=unit, xticks=np.arange(len(NAMES)))
        ax.set_xticklabels([n.replace('pedestal_', 'ped_') for n in NAMES], rotation=35, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.2); ax.legend(fontsize=8)
    figure.suptitle('Fixed-time metrics; hatched = truncated execution (means are not directly comparable)')
    figure.savefig(root/'metric_comparison.png', dpi=170)
    figure.savefig(root/'metric_comparison.pdf')
    plt.close(figure)

    # Show permission vs actual optimizer use for every parameter used in a run.
    for folder in sorted(root.glob('*_jev')):
        if not (folder/'events.jsonl').exists() or not (folder/'telemetry.json').exists(): continue
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        installs = [e for e in events if e['event'] == 'plan_installed']
        parameters = ['radial_m', 'lateral_m', 'height_m', 'yaw_rad', 'pitch_rad', 'roll_rad']
        used = [key for key in parameters if any(abs(e['actual_parameters'].get(key, 0)) > 1e-7 for e in installs)]
        if not used: continue
        rows = read(folder/'telemetry.json')
        times = np.array([r['t'] for r in rows])
        figure, axes = plt.subplots(len(used), 1, figsize=(10, 2.7*len(used)), squeeze=False, constrained_layout=True)
        for key, ax in zip(used, axes.flat):
            bounds = np.zeros((len(times), 2))
            for i, t in enumerate(times):
                active = next((e for e in reversed(installs) if e['time_s'] <= t), None)
                if active is None: continue
                env = active['envelope']
                if t > env['end_s']: continue
                settled = t >= active['activation_s']+env['blend_s']
                permitted = (env.get('settled_bounds') or env['bounds']) if settled else env['bounds']
                bounds[i] = permitted.get(key, [0., 0.])
            actual = [r['info']['parameters'][parameters.index(key)] for r in rows]
            ax.fill_between(times, bounds[:, 0], bounds[:, 1], alpha=.22, color=COLORS['jev'], label='Jev permission')
            ax.plot(times, actual, color=COLORS['jev'], label='optimizer target use')
            ax.axhline(0, color='black', linestyle='--', linewidth=.8)
            ax.set(xlabel='Original elapsed time (s)', ylabel=key, xlim=(0, 8))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        figure.suptitle(folder.name+': absolute changes relative to the original; target use is not tracking error')
        figure.savefig(folder/'permission_use.png', dpi=160)
        plt.close(figure)
    print(root/'metric_comparison.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    plot(parser.parse_args().runs)
