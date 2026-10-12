#!/usr/bin/env python3
"""Generate project-page figures for RoboCine (docs/figures/).

fig1_shot_vocabulary.png — the atomic movement set drawn as camera-frustum
    trails colored by time (paper-style: blue → red), incl. our three
    subject-tracking extensions and the 1-3-1 composition pattern.
fig3_trajectory_views.png — the REAL three-input chain result: bird's-eye
    and side views of the executed orbit-follow trajectory over the
    Replica point cloud (subject track + time-colored camera path).

    python scripts/make_paper_figures.py            # all figures
    python scripts/make_paper_figures.py --fig 1    # just the vocabulary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "figures"
_CMAP = plt.get_cmap("coolwarm")            # time encoding: blue → red

TARGET_COLOR = "#2b7de0"
SUBJECT_COLOR = "#e8973f"


def _time_colors(n: int):
    return [_CMAP(i / max(n - 1, 1)) for i in range(n)]


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {p.relative_to(REPO)}")
    return p


# --------------------------------------------------------------------------- #
# Frustum drawing (top-down): triangle apex at pos, opening along look dir
# --------------------------------------------------------------------------- #
def _frustum_poly(pos, yaw, fov_deg, depth, width_scale=1.0):
    half = math.radians(fov_deg) / 2.0
    a, b = yaw - half, yaw + half
    return Polygon([[pos[0], pos[1]],
                    [pos[0] + depth * math.cos(a) * width_scale,
                     pos[1] + depth * math.sin(a) * width_scale],
                    [pos[0] + depth * math.cos(b) * width_scale,
                     pos[1] + depth * math.sin(b) * width_scale]],
                   closed=True)


def _draw_frustum_trail(ax, positions, looks, fov=55.0, depth=0.32,
                        linewidth=2.2, n_frustums=7, width_scale=1.0):
    """Time-colored camera trail + evenly spaced translucent frustums."""
    pos = np.asarray(positions, float)[:, :2]
    look = np.asarray(looks, float)[:, :2]
    n = len(pos)
    colors = _time_colors(n)
    segs = np.stack([pos[:-1], pos[1:]], axis=1)
    ax.add_collection(LineCollection(segs, colors=colors[:-1],
                                     linewidths=linewidth, capstyle="round"))
    step = max(1, n // n_frustums)
    for i in range(0, n, step):
        d = look[i] - pos[i]
        yaw = math.atan2(d[1], d[0])
        poly = _frustum_poly(pos[i], yaw, fov, depth, width_scale)
        poly.set_facecolor((*colors[i][:3], 0.16))
        poly.set_edgecolor((*colors[i][:3], 0.85))
        poly.set_linewidth(0.9)
        ax.add_patch(poly)
    ax.scatter(*pos[0], color=colors[0], s=28, zorder=5,
               edgecolor="white", linewidth=0.6)
    ax.scatter(*pos[-1], color=colors[-1], s=28, zorder=5,
               edgecolor="white", linewidth=0.6)


def _panel(ax, title, subtitle, draw, xlim=(-1.6, 1.6), ylim=(-1.3, 1.5)):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=14)
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
            fontsize=8.2, color="#555555")
    draw(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def _target_dot(ax, at=(0.0, 0.0)):
    ax.add_patch(plt.Circle(at, 0.09, color=TARGET_COLOR, zorder=6))


# --------------------------------------------------------------------------- #
# Figure 1 — shot vocabulary
# --------------------------------------------------------------------------- #
def _arc_positions(center, r, a0_deg, a1_deg, n=40):
    t = np.radians(np.linspace(a0_deg, a1_deg, n))
    return np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)], 1)


def _straight(a, b, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def _look_at(positions, target):
    return np.tile(np.asarray(target, float), (len(positions), 1))


def fig_shot_vocabulary():
    fig, axes = plt.subplots(2, 8, figsize=(19, 5.8))
    fig.suptitle("RoboCine atomic shot vocabulary — 13 paper primitives "
                 "+ 3 subject-tracking extensions", fontsize=13.5,
                 fontweight="bold", y=0.985)
    T = (0.0, 0.0)
    idx = [0]

    def next_ax():
        ax = axes.flat[idx[0]]
        idx[0] += 1
        return ax

    # --- orbits ----------------------------------------------------------- #
    for name, sweep, sub in (("orbit_full", 360, "360° around target"),
                             ("orbit_half", 180, "180° around target"),
                             ("orbit_quarter", 90, "90° around target")):
        def draw(ax, sweep=sweep):
            p = _arc_positions(T, 0.85, 20, 20 + sweep)
            _draw_frustum_trail(ax, p, _look_at(p, T))
            _target_dot(ax)
        _panel(next_ax(), name, sub, lambda ax: draw(ax))

    # --- pan: parked, yaw sweep ------------------------------------------- #
    def pan_draw(ax):
        base = (-0.85, -0.5)
        p = _straight(base, base)
        yaw0 = math.atan2(T[1] - base[1], T[0] - base[0])
        looks = np.array([[base[0] + math.cos(yaw0 + a),
                           base[1] + math.sin(yaw0 + a)]
                          for a in np.radians(np.linspace(-40, 40, 40))])
        _draw_frustum_trail(ax, p, looks, n_frustums=6)
        _target_dot(ax)
    _panel(next_ax(), "pan_left / pan_right", "fixed position, yaw sweep", lambda ax: pan_draw(ax))

    # --- tilt: parked, pitch sweep (drawn as side profile) ----------------- #
    def tilt_draw(ax):
        base = (-0.85, -0.5)
        p = _straight(base, base)
        looks = np.array([[base[0] + 1.4,
                           base[1] + 1.4 * math.tan(math.radians(a))]
                          for a in np.linspace(-25, 30, 40)])
        _draw_frustum_trail(ax, p, looks, n_frustums=6)
        _target_dot(ax)
    _panel(next_ax(), "tilt_up / tilt_down", "fixed position, pitch sweep", lambda ax: tilt_draw(ax))

    # --- dolly in / out ---------------------------------------------------- #
    for name, r0, r1, sub in (("move_in", 1.2, 0.5, "dolly toward target"),
                              ("move_out", 0.5, 1.2, "dolly away from target")):
        def draw(ax, r0=r0, r1=r1):
            d = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
            p = _straight(np.asarray(T) + d * r0, np.asarray(T) + d * r1)
            _draw_frustum_trail(ax, p, _look_at(p, T))
            _target_dot(ax)
        _panel(next_ax(), name, sub, lambda ax: draw(ax))

    # --- crane (radial + height profile) ----------------------------------- #
    def crane_draw(ax):
        t = np.linspace(0, 1, 40)
        p = np.stack([-0.9 + 1.05 * t, -0.3 + 0.8 * np.sin(math.pi * t)], 1)
        _draw_frustum_trail(ax, p, _look_at(p, T))
        _target_dot(ax)
        ax.annotate("", xy=(0.45, -1.0), xytext=(-0.6, -1.0),
                    arrowprops=dict(arrowstyle="->", color="#888"))
        ax.text(-0.08, -1.2, "height ↑", ha="center", fontsize=7.5, color="#666")
    _panel(next_ax(), "crane", "vertical arc, fixed azimuth", lambda ax: crane_draw(ax))

    # --- zoom (fixed camera, FOV envelope) ---------------------------------- #
    def zoom_draw(ax):
        base = (-0.75, -0.42)
        p = _straight(base, base)
        yaw = math.atan2(T[1] - base[1], T[0] - base[0])
        for i, u in zip(range(0, 40, 5), np.linspace(0, 1, 8)):
            fov = 55 - 28 * math.sin(math.pi * u)
            poly = _frustum_poly(p[i], yaw, fov, 0.55)
            c = _time_colors(40)[i]
            poly.set_facecolor((*c[:3], 0.15))
            poly.set_edgecolor((*c[:3], 0.85))
            poly.set_linewidth(0.9)
            ax.add_patch(poly)
        ax.plot(p[:, 0], p[:, 1], color="#999", lw=1.6, ls="--")
        _target_dot(ax)
    _panel(next_ax(), "zoom_in_out / zoom_out_in", "fixed camera, FOV envelope", lambda ax: zoom_draw(ax))

    # --- static ------------------------------------------------------------- #
    def static_draw(ax):
        base = (-0.7, -0.4)
        p = _straight(base, base)
        _draw_frustum_trail(ax, p, _look_at(p, T), n_frustums=4)
        _target_dot(ax)
    _panel(next_ax(), "static", "locked-off hold", lambda ax: static_draw(ax))

    # --- arc transition ------------------------------------------------------ #
    def arc_draw(ax):
        chord = _straight((-1.15, -0.65), (1.05, 0.5))
        normal = np.array([-(0.5 - (-0.65)), 1.05 - (-1.15)])
        normal = normal / np.linalg.norm(normal)
        p = chord + 0.42 * np.sin(np.linspace(0, math.pi, 40))[:, None] * normal
        _draw_frustum_trail(ax, p, _look_at(p, T), n_frustums=6)
        ax.scatter([0], [0], marker="s", s=95, color=TARGET_COLOR, zorder=6)
    _panel(next_ax(), "arc (transition)", "curved A→B connector, α∈[−89°, 89°]", lambda ax: arc_draw(ax))

    # --- row 2: subject-tracking extensions --------------------------------- #
    def _track_panel(name, sub, sweep_deg=None):
        def draw(ax):
            t = np.linspace(0, 1, 40)
            subj = np.stack([-1.15 + 2.3 * t, -0.55 + 0.5 * np.sin(math.pi * t)], 1)
            heading = np.arctan2(np.gradient(subj[:, 1]), np.gradient(subj[:, 0]))
            if sweep_deg is None:
                az = np.full(40, math.radians(35))
            else:
                az = np.radians(np.linspace(-sweep_deg / 2, sweep_deg / 2, 40))
            dist = 0.62
            cam = subj + np.stack([dist * np.cos(heading + az),
                                   dist * np.sin(heading + az)], 1)
            _draw_frustum_trail(ax, cam, subj, depth=0.22, n_frustums=6,
                                width_scale=0.7)
            ax.plot(subj[:, 0], subj[:, 1], color=SUBJECT_COLOR, lw=1.8,
                    ls=(0, (4, 2)))
            ax.scatter(subj[::8, 0], subj[::8, 1], s=14, color=SUBJECT_COLOR,
                       zorder=6)
            ax.annotate("subject", xy=(subj[-1, 0], subj[-1, 1] - 0.18),
                        ha="center", fontsize=7.5, color="#b06f1f")
        _panel(next_ax(), name, sub, lambda ax: draw(ax))

    _track_panel("track_follow 🆕", "fixed offset while she moves")
    _track_panel("track_orbit 🆕", "sweep azimuth while following", 90)
    _track_panel("track_lead 🆕", "over-shoulder / lead angle")

    # --- 1-3-1 composition pattern ------------------------------------------- #
    def _pattern_panel():
        ax = next_ax()
        ax.set_title("1-3-1 composition", fontsize=11, fontweight="bold", pad=14)
        ax.text(0.5, 1.015, "A₁ → T₁ → arc → A₂ → T₂ …", transform=ax.transAxes,
                ha="center", fontsize=8.2, color="#555555")
        seq = [("A₁", (-1.2, 0.6)), ("T₁", (-0.6, -0.2)), ("arc", (0.0, -0.7)),
               ("A₂", (0.7, -0.15)), ("T₂", (1.25, 0.6))]
        for i, (lab, xy) in enumerate(seq):
            c = _time_colors(len(seq))[i]
            shape = "o" if lab.startswith("A") else \
                "s" if lab == "arc" else "D"
            ax.scatter(*xy, marker=shape, s=170, color=c, edgecolor="white",
                       linewidth=1.2, zorder=5)
            ax.text(xy[0], xy[1] - 0.3 if lab == "arc" else xy[1] + 0.18,
                    lab, ha="center", fontsize=9, fontweight="bold")
        for (_, p1), (_, p2) in zip(seq, seq[1:]):
            ax.annotate("", xy=p2, xytext=p1,
                        arrowprops=dict(arrowstyle="->", color="#aab", lw=1.4))
        ax.text(0.5, 0.02, "anchor → shot → arc → …  (validated, placement-aware)",
                transform=ax.transAxes, ha="center", fontsize=7.6, color="#666")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.3, 1.5)
        ax.set_aspect("equal"); ax.axis("off")

    _pattern_panel()

    # --- time-color legend (last panel) --------------------------------------- #
    ax = axes.flat[-1]
    ax.set_title("time encoding", fontsize=11, fontweight="bold", pad=14)
    ax.text(0.5, 1.015, "shared by every figure", transform=ax.transAxes,
            ha="center", fontsize=8.2, color="#555555")
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), extent=[-1.1, 1.1, -0.18, 0.18],
              aspect="auto", cmap=_CMAP)
    ax.text(-1.1, 0.28, "t = 0", fontsize=8.5, ha="left")
    ax.text(1.1, 0.28, "t = end", fontsize=8.5, ha="right")
    ax.text(0, -0.42, "triangle = frustum apex · dots = trail start / end",
            ha="center", fontsize=7.6, color="#666")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.9, 0.7)
    ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "fig1_shot_vocabulary.png")


# --------------------------------------------------------------------------- #
# Figure 3 — real three-input chain over the real point cloud
# --------------------------------------------------------------------------- #
def _load_cloud(voxel: float = 0.12):
    sys.path.insert(0, str(REPO))
    from cinema_traj.stages.extract import _to_yup, load_points, voxel_downsample

    scene = json.loads((REPO / "cinema_traj/scenes/frl_apartment_0.json")
                       .read_text(encoding="utf-8"))
    prov = scene["provenance"]
    pts = load_points(str(REPO / prov["source_map"]), up="y")
    pts = _to_yup(pts, up=prov.get("map_up", "y"))
    pts = pts + np.asarray(prov.get("map_shift", [0, 0, 0]))
    return voxel_downsample(pts, voxel), scene


def fig_trajectory_views():
    cloud, scene = _load_cloud()
    exports = REPO / "cinema_traj_out/frl_apartment_0_autosem/exports"
    sample_path = exports / "traj_samples_retimed.json"
    if not sample_path.exists():
        sample_path = exports / "traj_samples.json"
    samples = json.loads(sample_path.read_text(encoding="utf-8"))["samples"]
    cam = np.array([[s["x"], s["y"], s["z"]] for s in samples])
    subj = np.array([f["pos"] for f in scene["subject_track"]["frames"]])
    lo = np.array(scene["rooms"][0]["bounds_min"])
    hi = np.array(scene["rooms"][0]["bounds_max"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4),
                             gridspec_kw={"width_ratios": [1.12, 1]})
    fig.suptitle("RoboCine three-input chain on Replica frl_apartment_0 — "
                 "movement recommendation + 3DGS cloud + mannequin 6D → executed trajectory",
                 fontsize=12.5, fontweight="bold", y=0.98)

    height = cloud[:, 1]
    for ax, (ix, iy, xl, yl, title, equal) in zip(
            axes,
            [(0, 2, "x [m]", "z [m]", "bird's-eye (ground plane XZ)", True),
             (0, 1, "x [m]", "y [m]", "side view (XY)", False)]):
        sc = ax.scatter(cloud[:, ix], cloud[:, iy], s=0.7, c=height,
                        cmap="cividis", alpha=0.35, linewidths=0)
        ax.plot([lo[ix], hi[ix], hi[ix], lo[ix], lo[ix]],
                [lo[iy], lo[iy], hi[iy], hi[iy], lo[iy]],
                color="#8899aa", lw=1.0, ls="--", alpha=0.8)
        ax.plot(subj[:, ix], subj[:, iy], color=SUBJECT_COLOR, lw=2.4,
                ls=(0, (5, 2)), label="mannequin (subject 6D)")
        ax.scatter(subj[0, ix], subj[0, iy], s=70, marker="^",
                   color=SUBJECT_COLOR, zorder=6, edgecolor="white", linewidth=0.8)
        ax.scatter(subj[-1, ix], subj[-1, iy], s=70, marker="s",
                   color=SUBJECT_COLOR, zorder=6, edgecolor="white", linewidth=0.8)
        pts2 = cam[:, [ix, iy]]
        segs = np.stack([pts2[:-1], pts2[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=_time_colors(len(pts2)),
                                         linewidths=3.0, capstyle="round",
                                         label="camera trajectory"))
        ax.scatter(cam[0, ix], cam[0, iy], s=80, color=_CMAP(0.0), zorder=7,
                   edgecolor="white", linewidth=1.0)
        ax.scatter(cam[-1, ix], cam[-1, iy], s=80, color=_CMAP(1.0), zorder=7,
                   edgecolor="white", linewidth=1.0)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title, fontsize=11)
        if equal:
            ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.12)

    cbar = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(height.min(), height.max()), cmap="cividis"),
        ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("point-cloud height [m]", fontsize=9)
    fig.text(0.5, 0.005,
             "camera trail color = time (blue→red) · ▲/■ = mannequin start/end · "
             "dashed orange = her 6D track (0.6 m/s walk + 12 s pause)",
             ha="center", fontsize=9, color="#444")
    return _save(fig, "fig3_trajectory_views.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig", choices=["1", "3"], help="generate a single figure")
    a = ap.parse_args()
    if a.fig in (None, "1"):
        fig_shot_vocabulary()
    if a.fig in (None, "3"):
        fig_trajectory_views()


if __name__ == "__main__":
    main()
