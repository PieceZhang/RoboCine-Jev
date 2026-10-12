"""Extract stage: unlabeled 3D map (PCD / PLY, incl. 3DGS point dumps) →
scene-graph JSON draft.

Honest scope (documented): with NO semantics available, this stage performs
geometry-only structure detection — floor/ceiling planes, Manhattan-axis wall
planes, obstacle clustering → PCA OBBs — and emits generic labels
(object_01 …). Expect to HAND-REFINE the draft: rename labels, fix placement,
split rooms, add doors. The hand-refined JSON is the canonical scene input.

Pipeline: load → voxel downsample (0.05 m) → floor/ceiling (y-histogram
modes) → walls (axis-aligned histogram peaks, thin OBBs) → remaining points
clustered (voxel-hash Euclidean, radius 0.25 m) → PCA OBB + placement
classification → scene JSON draft + warnings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common.obb import OBB, fit_obb_pca
from vlm_intention.common.atomic_io import atomic_write_json
from vlm_intention.common.config import load_config

# --------------------------------------------------------------------------- #
# Readers (pure numpy — no open3d/trimesh dependency)
# --------------------------------------------------------------------------- #
def _to_yup(pts: np.ndarray, up: str = "y") -> np.ndarray:
    """Declared-convention axis swap at ingestion — NO auto-detection.

    The pipeline's world frame is +Y up (se3.py); maps exported Z-up (ground
    on XY, e.g. the SceneSplat-7K Replica 3DGS PLYs) declare ``up="z"`` and
    get one right-handed rotation (x,y,z) → (x,z,−y) here, so every stage
    downstream sees the usual Y-up frame."""
    if up in ("y", "+y", "Y", "+Y"):
        return pts
    if up in ("z", "+z", "Z", "+Z"):
        out = pts.copy()
        out[:, 0] = pts[:, 0]
        out[:, 1] = pts[:, 2]
        out[:, 2] = -pts[:, 1]
        return out
    raise ValueError(f"unsupported up axis {up!r} (y | z)")


def load_points(path: str, up: str = "y") -> np.ndarray:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".pcd":
        pts = _read_pcd(p)
    elif suf == ".ply":
        pts = _read_ply(p)
    elif suf == ".npz":
        pts = np.load(p)["points"].astype(np.float64)
    else:
        raise ValueError(f"unsupported map format '{suf}' (pcd / ply / npz)")
    return _to_yup(pts, up)


_TYPEMAP = {"I": "i", "U": "u", "F": "f"}   # PCD: I = signed int, U = unsigned int


def _read_pcd(p: Path) -> np.ndarray:
    with open(p, "rb") as fh:
        raw = fh.read()
    header, idx, data_start, mode = {}, 0, None, "ascii"
    while data_start is None:
        nl = raw.find(b"\n", idx)
        if nl < 0:
            break
        line = raw[idx:nl].decode("ascii", "ignore").strip()
        idx = nl + 1
        parts = line.split()
        if parts and parts[0] == "DATA":
            mode = parts[1] if len(parts) > 1 else "ascii"
            data_start = idx
        elif len(parts) >= 2:
            header[parts[0]] = parts[1:]
    if data_start is None:
        raise ValueError(f"{p}: PCD header has no DATA line")
    fields = header.get("FIELDS", ["x", "y", "z"])
    sizes = [int(s) for s in header.get("SIZE", ["4"] * len(fields))]
    types = header.get("TYPE", ["F"] * len(fields))
    n = int(header.get("POINTS", ["0"])[0])
    dtype = np.dtype([(f, "<" + _TYPEMAP[t] + str(s)) for f, t, s in zip(fields, types, sizes)])
    if mode == "ascii":
        rows = []
        for line in raw[data_start:].decode("ascii", "ignore").splitlines():
            vals = line.split()
            if len(vals) >= len(fields):
                rows.append(tuple(float(v) for v in vals[:len(fields)]))
        arr = np.asarray(rows, dtype=np.float64)
        xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
        return arr[:, [xi, yi, zi]]
    raw_pts = np.frombuffer(raw[data_start:data_start + n * dtype.itemsize], dtype=dtype, count=n)
    return np.stack([raw_pts["x"], raw_pts["y"], raw_pts["z"]], axis=1).astype(np.float64)


def _read_ply(p: Path) -> np.ndarray:
    with open(p, "rb") as fh:
        raw = fh.read()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "ignore")
    if "binary_little_endian" in header:
        fmt_name = "bin_le"
    elif "binary_big_endian" in header:
        fmt_name = "bin_be"
    else:
        fmt_name = "ascii"
    props, n_vertex, in_vertex = [], 0, False
    for line in header.splitlines():
        parts = line.split()
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n_vertex = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            t, name = parts[1], parts[2]
            props.append((t, name))
    if not any(n == "x" for _, n in props):
        raise ValueError("PLY has no x property")
    if fmt_name == "ascii":
        rows = []
        for line in raw[end:].decode("ascii", "ignore").splitlines()[:n_vertex]:
            vals = line.split()
            if len(vals) >= 3:
                rows.append((float(vals[0]), float(vals[1]), float(vals[2])))
        return np.asarray(rows, dtype=np.float64)
    type_fmt = {"char": "b", "uchar": "B", "short": "h", "ushort": "H",
                "int": "i", "uint": "I", "float": "f", "float32": "f",
                "double": "d", "int8": "b", "uint8": "B", "int16": "h",
                "uint16": "H", "int32": "i", "uint32": "I", "float64": "d"}
    endian = "<" if fmt_name == "bin_le" else ">"
    dtype = np.dtype([(f"{n}_{i}", endian + type_fmt[t]) for i, (t, n) in enumerate(props)])
    rec = np.frombuffer(raw[end:end + n_vertex * dtype.itemsize], dtype=dtype, count=n_vertex)
    xk, yk, zk = [next(k for k in dtype.names if k.startswith(ax + "_")) for ax in "xyz"]
    return np.stack([rec[xk], rec[yk], rec[zk]], axis=1).astype(np.float64)


# --------------------------------------------------------------------------- #
# Geometry pipeline (pure numpy)
# --------------------------------------------------------------------------- #
def voxel_downsample(pts: np.ndarray, voxel: float = 0.05) -> np.ndarray:
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def detect_floor_ceiling(pts: np.ndarray, bin_w: float = 0.05):
    """Floor = y-histogram mode in the lower mass; ceiling = mode in the
    upper mass (None if the scan is single-sided)."""
    ys = pts[:, 1]
    lo, hi = ys.min(), ys.max()
    if hi - lo < 1.0:
        return float(np.median(ys)), None
    bins = np.arange(lo, hi + bin_w, bin_w)
    hist, edges = np.histogram(ys, bins=bins)
    lower = hist[: len(hist) // 3]
    upper = hist[-len(hist) // 3:]
    floor = float(edges[int(np.argmax(lower))])
    ceil_i = len(hist) - len(upper) + int(np.argmax(upper))
    ceiling = float(edges[ceil_i]) if upper.max() > 0.15 * hist.max() else None
    return floor, ceiling


def detect_walls(pts: np.ndarray, floor_y: float, ceiling_y: float | None,
                 bin_w: float = 0.08, peak_factor: float = 6.0):
    """Manhattan-axis wall planes via x/z histogram peaks of mid-height
    points. Returns [(axis, coord, points_xyz)] — points near that plane.

    Adjacent passing bins are grouped (a thick wall / 3DGS fuzz raises a
    broad hump, not a 1-bin spike) and each group collapses to its strongest
    bin, so one wall is reported once."""
    top = ceiling_y if ceiling_y is not None else floor_y + 2.8
    band = pts[(pts[:, 1] > floor_y + 0.3) & (pts[:, 1] < top - 0.2)]
    walls = []
    for axis in (0, 2):
        vals = band[:, axis]
        if vals.size == 0:
            continue
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1.0:
            continue
        hist, edges = np.histogram(vals, bins=np.arange(lo, hi + bin_w, bin_w))
        med = max(np.median(hist[hist > 0]), 1.0)
        peaks = np.where(hist > peak_factor * med)[0]
        groups: list[list] = []                       # consecutive-bin grouping
        for i in peaks:
            if groups and i - groups[-1][-1] <= 2:
                groups[-1].append(i)
            else:
                groups.append([i])
        for g in groups:
            best = g[int(np.argmax(hist[g]))]
            coord = float(edges[best] + bin_w / 2)
            near = np.abs(band[:, axis] - coord) < 1.5 * bin_w
            wall_pts = band[near]
            # must span ≥1.5 m along the perpendicular horizontal direction
            perp_axis = 2 - axis                     # x-wall ↔ z-span, z-wall ↔ x-span
            span = wall_pts[:, perp_axis].max() - wall_pts[:, perp_axis].min()
            if span < 1.5 or wall_pts.shape[0] < 200:
                continue
            walls.append((axis, coord, wall_pts))
    return walls


def wall_obb(axis: int, coord: float, wall_pts: np.ndarray, floor_y: float,
             ceiling_y: float, thickness: float = 0.12) -> OBB:
    perp = 2 - axis                               # other horizontal axis
    pmin, pmax = wall_pts[:, perp].min(), wall_pts[:, perp].max()
    top = ceiling_y if ceiling_y is not None else max(wall_pts[:, 1].max(), floor_y + 2.5)
    center = np.zeros(3)
    center[axis] = coord
    center[perp] = (pmin + pmax) / 2
    center[1] = (floor_y + top) / 2
    extents = np.zeros(3)
    extents[axis] = thickness
    extents[perp] = (pmax - pmin) + 0.1
    extents[1] = top - floor_y
    return OBB(center, np.eye(3), extents)


def cluster_objects(pts: np.ndarray, voxel: float = 0.15, radius: float = 0.25,
                    min_pts: int = 20, max_pts: int = 200000) -> list:
    """Voxel-hash Euclidean clustering (BFS over neighbor voxels)."""
    keys = np.floor(pts / voxel).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    starts = np.searchsorted(inverse[order], np.arange(len(uniq)))
    voxel_index = {tuple(k): i for i, k in enumerate(uniq)}
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)]
    visited = np.zeros(len(uniq), dtype=bool)
    clusters = []
    for seed in range(len(uniq)):
        if visited[seed]:
            continue
        comp, q = [], [seed]
        visited[seed] = True
        while q:
            v = q.pop()
            comp.append(v)
            k = uniq[v]
            for dx, dy, dz in offs:
                nb = voxel_index.get((k[0] + dx, k[1] + dy, k[2] + dz))
                if nb is not None and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        idxs = []
        for v in comp:
            s = starts[v]
            e = starts[v + 1] if v + 1 < len(starts) else len(order)
            idxs.extend(order[s:e].tolist())
        if min_pts <= len(idxs) <= max_pts:
            clusters.append(pts[idxs])
    return clusters


def classify_placement(obb: OBB, walls: list, ceiling_y: float | None) -> str:
    top = obb.center[1] + obb.half[1]
    if ceiling_y is not None and abs(top - ceiling_y) < 0.3 and obb.center[1] > ceiling_y - 1.0:
        return "ceiling"
    for axis, coord, _ in walls:
        d = abs(obb.center[axis] - coord) - obb.half[axis]
        if d < 0.25 and obb.extents[axis] < 0.8:
            return "wall"
    return "freestanding"


# --------------------------------------------------------------------------- #
# Scene draft assembly
# --------------------------------------------------------------------------- #
def build_scene_draft(points: np.ndarray, scene_id: str = "extracted",
                      voxel: float = 0.05, wall_peak: float = 6.0) -> tuple:
    warnings = []
    pts = voxel_downsample(points, voxel)
    floor_y, ceiling_y = detect_floor_ceiling(pts)
    walls = detect_walls(pts, floor_y, ceiling_y, peak_factor=wall_peak)
    if not walls:
        warnings.append("no walls detected — check the map or hand-author the scene")

    room_lo = pts.min(axis=0) - np.array([0.1, 0.0, 0.1])
    room_lo[1] = floor_y - 0.05
    top = ceiling_y if ceiling_y is not None else floor_y + 2.8
    room_hi = np.array([pts[:, 0].max() + 0.1, top + 0.05, pts[:, 2].max() + 0.1])
    warnings.append("single-room assumption — split rooms + add doors by hand if needed")

    structures = [
        {"id": "floor", "kind": "floor", "room": "room_01",
         "obb": OBB((room_lo + room_hi) / 2 * np.array([1, 0, 1]) + np.array([0, floor_y - 0.05, 0]),
                    np.eye(3),
                    np.array([room_hi[0] - room_lo[0], 0.1, room_hi[2] - room_lo[2]])).to_dict()},
    ]
    if ceiling_y is not None:
        structures.append({
            "id": "ceiling", "kind": "ceiling", "room": "room_01",
            "obb": OBB(np.array([(room_lo[0] + room_hi[0]) / 2, ceiling_y + 0.05,
                                 (room_lo[2] + room_hi[2]) / 2]), np.eye(3),
                       np.array([room_hi[0] - room_lo[0], 0.1,
                                 room_hi[2] - room_lo[2]])).to_dict()})
    for i, (axis, coord, wp) in enumerate(walls):
        obb = wall_obb(axis, coord, wp, floor_y, ceiling_y if ceiling_y is not None else top)
        structures.append({"id": f"wall_{i}", "kind": "wall", "room": "room_01",
                           "obb": obb.to_dict()})

    # objects = points not near any wall plane / floor / ceiling
    mask = np.ones(len(pts), dtype=bool)
    mask &= (pts[:, 1] > floor_y + 0.15)
    if ceiling_y is not None:
        mask &= (pts[:, 1] < ceiling_y - 0.2)
    for axis, coord, _ in walls:
        mask &= np.abs(pts[:, axis] - coord) > 0.3
    clusters = cluster_objects(pts[mask])
    objects = []
    for i, cpts in enumerate(sorted(clusters, key=len, reverse=True)[:40]):
        obb = fit_obb_pca(cpts)
        objects.append({
            "id": f"object_{i + 1:02d}", "label": f"object_{i + 1:02d}",
            "room": "room_01", "obb": obb.to_dict(),
            "placement": classify_placement(obb, walls, ceiling_y)})

    scene = {
        "scene_id": scene_id,
        "rooms": [{"id": "room_01", "label": "room_01",
                   "bounds_min": room_lo.tolist(), "bounds_max": room_hi.tolist()}],
        "structures": structures,
        "objects": objects,
        "doors": [],
        "provenance": {"source_points": int(len(points)),
                       "downsampled": int(len(pts)), "voxel_m": voxel,
                       "floor_y": round(floor_y, 3),
                       "ceiling_y": round(ceiling_y, 3) if ceiling_y is not None else None,
                       "walls": len(walls), "objects": len(objects)},
    }
    warnings.append(f"{len(objects)} objects got GENERIC labels — rename them "
                    f"to semantic labels (e.g. 沙发 sofa) before LLM planning")
    return scene, warnings


# --------------------------------------------------------------------------- #
# Stage entry
# --------------------------------------------------------------------------- #
def run(cfg=None, map_path=None, out_path=None, voxel: float = 0.05,
        wall_peak: float = 6.0, up: str = "y",
        review: bool = False, force: bool = False) -> dict:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    map_path = map_path or ct.get("map", "")
    pts = load_points(map_path, up=up)
    scene_id = Path(map_path).stem
    print(f"🗺️  读取地图: {map_path}（{len(pts)} 点，up={up}）")

    scene, warnings = build_scene_draft(pts, scene_id=scene_id, voxel=voxel,
                                        wall_peak=wall_peak)
    scene.setdefault("provenance", {})["map_up"] = up
    scene["provenance"]["source_map"] = str(map_path)   # viewer cloud layer reads this
    out_path = Path(out_path) if out_path else Path(
        ct.get("scenes_dir", "cinema_traj/scenes")) / f"{scene_id}.json"
    if not out_path.is_absolute():
        base = Path(cfg.get("paths", {}).get("base", "."))
        out_path = base / out_path
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} exists (use --force to overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, scene)
    print(f"✅ scene draft → {out_path}")
    for w in warnings:
        print(f"   ⚠️  {w}")
    if review:
        print(json.dumps(scene, ensure_ascii=False, indent=2))
    return scene


if __name__ == "__main__":                                # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="无语义 3D 地图 → 场景图 JSON 草稿")
    ap.add_argument("--map", help="PCD / PLY / NPZ 点云路径")
    ap.add_argument("--out", help="输出 scene JSON 路径")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--wall-peak", type=float, default=6.0,
                    help="墙面直方图峰阈值（×中位数）；3DGS 点云背景密建议 3.0")
    ap.add_argument("--up", default="y", choices=["y", "z"],
                    help="地图高度轴约定（y=地面在XZ；z=地面在XY，如 SceneSplat "
                         "Replica 3DGS，入口处旋转成内部 Y-up）")
    ap.add_argument("--review", action="store_true", help="打印 JSON 供人工修订")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, map_path=a.map, out_path=a.out, voxel=a.voxel, wall_peak=a.wall_peak,
        up=a.up, review=a.review, force=a.force)
