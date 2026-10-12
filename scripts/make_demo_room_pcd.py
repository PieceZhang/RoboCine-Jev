#!/usr/bin/env python3
"""Synthetic demo-room point cloud → binary PCD.

Generates the same room geometry as ``tests/test_traj_stages.py::_room_cloud``
(6×6×2.7 m: floor / ceiling / 4 walls) plus 2 furniture boxes and 1 "target"
box, written as a ``binary_little_endian``-style PCD (DATA binary, float32 xyz)
— the exact dual of ``cinema_traj/stages/extract.py::_read_pcd``.

Use it as a zero-dependency starting point to exercise the full
PCD → extract → (hand-refine labels) → plan → build → optimize → export →
retime chain without a real scan:

    python scripts/make_demo_room_pcd.py
    python run_traj.py --stage extract --map cinema_traj/demo/demo_room.pcd --force

After extract, rename the generic object_NN labels to something semantic
(the extract stage warns about this) before planning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ROOM = {"half": 3.0, "height": 2.7}

# furniture / target boxes: (center_x, center_z, size_x, height, size_z)
BOXES = [
    (-1.2, -1.2, 1.8, 0.8, 0.9),     # sofa-ish, against the corner
    (1.5, 1.5, 1.2, 0.75, 1.2),      # table-ish
    (0.0, 0.0, 0.6, 1.0, 0.6),       # "target" box near room center
]


def _room_points(rng: np.random.Generator, n_per_surface: int = 3000) -> np.ndarray:
    pts = []
    for _ in range(n_per_surface):                          # floor + ceiling
        pts.append([rng.uniform(-3, 3), rng.normal(0, 0.01), rng.uniform(-3, 3)])
        pts.append([rng.uniform(-3, 3), ROOM["height"] + rng.normal(0, 0.01),
                    rng.uniform(-3, 3)])
    for _ in range(n_per_surface):                          # 4 walls
        u, v = rng.uniform(-3, 3), rng.uniform(0, ROOM["height"])
        pts.extend([[3 + rng.normal(0, 0.01), v, u],
                    [-3 + rng.normal(0, 0.01), v, u],
                    [u, v, 3 + rng.normal(0, 0.01)],
                    [u, v, -3 + rng.normal(0, 0.01)]])
    return np.asarray(pts)


def _box_surface_points(rng: np.random.Generator, cx: float, cz: float,
                        sx: float, sy: float, sz: float, n: int = 900) -> np.ndarray:
    """Points on the 5 visible faces of an axis-aligned box sitting on y=0."""
    top = np.stack([rng.uniform(cx - sx / 2, cx + sx / 2, n),
                    np.full(n, sy) + rng.normal(0, 0.01, n),
                    rng.uniform(cz - sz / 2, cz + sz / 2, n)], axis=1)
    sides = []
    m = n // 2
    for face in range(4):                                   # ±x / ±z faces
        xy = rng.uniform(0, sy, m)
        uv = rng.uniform(-0.5, 0.5, m)
        if face == 0:
            sides.append(np.stack([np.full(m, cx + sx / 2), xy,
                                   cz + uv * sz], axis=1))
        elif face == 1:
            sides.append(np.stack([np.full(m, cx - sx / 2), xy,
                                   cz + uv * sz], axis=1))
        elif face == 2:
            sides.append(np.stack([cx + uv * sx, xy,
                                   np.full(m, cz + sz / 2)], axis=1))
        else:
            sides.append(np.stack([cx + uv * sx, xy,
                                   np.full(m, cz - sz / 2)], axis=1))
    return np.concatenate([top] + sides)


def build_demo_room(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = [_room_points(rng)]
    for cx, cz, sx, sy, sz in BOXES:
        pts.append(_box_surface_points(rng, cx, cz, sx, sy, sz))
    return np.concatenate(pts)


def write_pcd_binary(path: Path, pts: np.ndarray) -> None:
    """binary PCD (DATA binary = little-endian float32 xyz) — dual of _read_pcd."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
              f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              f"POINTS {len(pts)}\nDATA binary\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(pts.astype("<f4").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="合成演示房间 binary PCD")
    ap.add_argument("--out", default="cinema_traj/demo/demo_room.pcd")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    pts = build_demo_room(a.seed)
    write_pcd_binary(out, pts)
    print(f"🗺️  {out}（{len(pts)} 点，6×6×{ROOM['height']}m 房间 + {len(BOXES)} 个盒子）")
    print("    下一步: python run_traj.py --stage extract --map "
          f"{out} --force   然后人工把 object_NN 改成语义 label")


if __name__ == "__main__":
    main()
