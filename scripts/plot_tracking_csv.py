#!/usr/bin/env python3
"""plot_tracking_csv — plot the tracking node's per-tick error CSV.

Input: the err_csv file the tracking node writes when started with
``-p err_csv:=<path>`` (columns t_s, pos_err_m, rot_err_rad, cam_xyz,
des_xyz). Output: a 3-panel PNG (position error, rotation error,
measured-vs-desired camera xyz). Complements record_cam_error.py: this
one needs no live ROS subscription — the data is logged in-process by
the tracker itself, immune to the DDS-link dropouts an external
recorder node suffers under Isaac bridge load.

Usage:
    python3 scripts/plot_tracking_csv.py <err.csv> <out.png> [--title T]
"""

import argparse
import csv

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("png_path")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    with open(args.csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{args.csv_path}: no rows")
    col = lambda name: np.array([float(r[name]) for r in rows])  # noqa: E731
    t, e_p, e_o = col("t_s"), col("pos_err_m"), col("rot_err_rad")
    cam = np.stack([col(f"cam_{a}") for a in "xyz"], axis=1)
    des = np.stack([col(f"des_{a}") for a in "xyz"], axis=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(args.title or "jarvis WBC tracking — camera pose error "
                 "vs desired path (in-process tracker log)")

    ax = axes[0]
    ax.plot(t, e_p * 100, ".-", ms=2, lw=0.8)
    ax.set_ylabel("position error [cm]")
    ax.grid(alpha=0.3)
    m = e_p
    ax.set_title(f"pass: mean {m.mean()*100:.1f} cm, "
                 f"max {m.max()*100:.1f} cm, final {m[-5:].mean()*100:.1f} cm",
                 fontsize=9)

    ax = axes[1]
    ax.plot(t, np.degrees(e_o), ".-", ms=2, lw=0.8, color="C1")
    ax.set_ylabel("rotation error [deg]")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for k, c, lbl in zip(range(3), "rgb", "xyz"):
        ax.plot(t, cam[:, k] * 100, color=c, label=f"meas {lbl}")
        ax.plot(t, des[:, k] * 100, color=c, ls="--", lw=0.8)
    ax.set_ylabel("camera position [cm]")
    ax.set_xlabel("t since tracking_start [s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(args.png_path, dpi=150)
    print(f"{args.png_path}: {len(t)} ticks, "
          f"mean err {e_p.mean()*100:.1f} cm, final {e_p[-5:].mean()*100:.1f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
