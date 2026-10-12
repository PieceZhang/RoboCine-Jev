#!/usr/bin/env python3
"""Export tracking/actuation/constraint plots from Isaac telemetry CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.folder / 'telemetry.csv').open()))
    values = lambda key: np.array([float(row[key]) for row in rows])
    t = values('time_s')
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True, constrained_layout=True)
    axes[0].plot(t, values('position_error')*1000, label='position (mm)')
    axes[0].plot(t, np.rad2deg(values('orientation_error')), label='orientation (deg)')
    axes[0].set_title(f'Isaac Sim WBC — {args.folder.name}')
    for key in ('v', 'omega'):
        axes[1].plot(t, values('command_'+key), label='command '+key)
        axes[1].plot(t, values('measured_'+key), alpha=.6, label='measured '+key)
    axes[1].set_ylabel('m/s, rad/s')
    for i in range(1, 7):
        axes[2].plot(t, values(f'command_dq{i}'), label=f'command dq{i}')
        axes[3].plot(t, values(f'measured_dq{i}'), label=f'measured dq{i}', linewidth=.6)
    axes[2].set_ylabel('rad/s')
    axes[3].set_ylabel('rad/s')
    for key in ('qp_violation', 'measured_acceleration_violation', 'measured_wheel_acceleration_violation'):
        axes[4].plot(t, values(key), label=key)
    axes[4].set_ylabel('constraint excess')
    axes[4].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
    fig.savefig(args.folder / 'tracking.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
