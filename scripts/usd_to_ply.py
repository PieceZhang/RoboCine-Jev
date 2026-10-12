#!/usr/bin/env python3
"""USD interior scene (InteriorAgent kujiale_*) → point-cloud PLY + semantic sidecar.

InteriorAgent scenes are USDA stages referencing per-asset USD meshes
(Meshes/<asset>.usd, plain references, no instancing). cinema_traj's extract
stage consumes point clouds, so this converter samples mesh surfaces uniformly
by area and writes:

  <out>.ply        binary-little-endian PLY, x/y/z float32, NATIVE stage frame
                   (InteriorAgent is Z-up — pass --up z to extract at ingestion,
                   exactly like the SceneSplat Replica 3DGS flow)
  <sidecar>.json   per-asset world OBB + semantic label (asset name with the
                   _NNNN suffix stripped), room from the enclosing scope +
                   rooms.json point-in-polygon, all in the INTERNAL Y-up frame
                   so the refine step can upgrade extract's generic object_NN
                   labels in one pass

Usage:
    python scripts/usd_to_ply.py \
        --usd datasets/interioragent/kujiale_0021/kujiale_0021.usda \
        --rooms datasets/interioragent/kujiale_0021/rooms.json

Requires: usd-core (pip install usd-core). Sampling is pure numpy, fixed seed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    from pxr import Usd, UsdGeom
except ImportError:                                   # deferred error, see main()
    Usd = UsdGeom = None

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common.obb import fit_obb_pca
from cinema_traj.stages.extract import _to_yup  # same axis swap as --up z ingest

ASSET_RE = re.compile(r"_\d+$")           # bed_0002 → bed
STRUCTURE_SCOPES = {"floor", "wall", "ceiling", "other"}


def _triangulate(counts: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fan-triangulate a polygon soup → (a, b, c) vertex-index arrays."""
    ntris = np.maximum(counts.astype(np.int64) - 2, 0)
    total = int(ntris.sum())
    if total == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e
    tri_face = np.repeat(np.arange(len(counts), dtype=np.int64), ntris)
    tri_base = np.repeat(np.concatenate(([0], np.cumsum(ntris)[:-1])), ntris)
    within = np.arange(total, dtype=np.int64) - tri_base
    s = starts = np.concatenate(([0], np.cumsum(counts)[:-1]))[tri_face]
    a = idx[s]
    b = idx[s + 1 + within]
    c = idx[s + 2 + within]
    return a, b, c


def _stride_faces(counts: np.ndarray, indices: np.ndarray, max_faces: int):
    """Downsample a face list to ≤ max_faces by uniform stride, slicing the
    index buffer to only the kept faces' ranges (collision clouds don't need
    sculpt-level tessellation — the 119 MB bed_0000 has proved the point)."""
    n = len(counts)
    if n <= max_faces:
        return counts, indices
    stride = int(np.ceil(n / max_faces))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    keep = np.arange(0, n, stride)
    seg_starts = starts[keep]
    seg_lens = counts[keep]
    total = int(seg_lens.sum())
    offs = np.repeat(seg_starts, seg_lens)
    within = np.arange(total, dtype=np.int64) - \
        np.repeat(np.cumsum(seg_lens) - seg_lens, seg_lens)
    print(f"      ⚠️  {n} faces → stride {stride} → {len(keep)} faces")
    return seg_lens.astype(np.int64), indices[offs + within]


def sample_mesh(mesh, world_m3: np.ndarray, world_t: np.ndarray,
                density: float, rng: np.random.Generator,
                chunk: int = 1_000_000, max_faces: int = 2_000_000,
                max_samples: int = 1_500_000) -> np.ndarray:
    """Area-proportional barycentric sampling of one UsdGeom.Mesh → world pts.

    Sampling happens in WORLD space on purpose: InteriorAgent asset layers are
    authored in centimetres while the composed instance transforms carry the
    cm→m scale, so LOCAL areas are 10 000× inflated and world areas are the
    truth. Triangles are processed in chunks to stay memory-bounded."""
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.empty((0, 3))
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not counts or not indices:
        return np.empty((0, 3))
    counts, indices = _stride_faces(np.asarray(counts, dtype=np.int64),
                                    np.asarray(indices, dtype=np.int64), max_faces)
    a, b, c = _triangulate(counts, indices)
    if a.size == 0:
        return np.empty((0, 3))
    out = []
    for lo in range(0, a.size, chunk):
        hi = min(lo + chunk, a.size)
        wa = pts[a[lo:hi]] @ world_m3 + world_t
        wb = pts[b[lo:hi]] @ world_m3 + world_t
        wc = pts[c[lo:hi]] @ world_m3 + world_t
        area = 0.5 * np.linalg.norm(np.cross(wb - wa, wc - wa), axis=1)
        keep = area > 1e-12
        if not keep.any():
            continue
        wa, wb, wc, area = wa[keep], wb[keep], wc[keep], area[keep]
        n_target = min(max(1, int(round(float(area.sum()) * density))), max_samples)
        per_tri = rng.multinomial(n_target, area / area.sum()) if area.size > 1 \
            else np.array([n_target], dtype=np.int64)
        nz = per_tri > 0
        if not nz.any():
            continue
        fa = np.repeat(np.arange(wa.shape[0])[nz], per_tri[nz])
        n = fa.size
        s = np.sqrt(rng.random(n))[:, None]
        v = rng.random(n)[:, None]
        P = wa[fa] * (1.0 - s) + wb[fa] * (s * (1.0 - v)) + wc[fa] * (s * v)
        out.append(P)
    if not out:
        return np.empty((0, 3))
    return np.vstack(out)


def iter_assets(stage):
    """Yield (prim, scope_name) for every top-level asset Xform under
    /Root/Meshes/<scope>/<asset> — the InteriorAgent layout."""
    meshes_scope = None
    for child in stage.GetPseudoRoot().GetAllChildren():
        if child.GetName() == "Meshes":
            meshes_scope = child
            break
    if meshes_scope is None:                       # fallback: any Scope "Meshes"
        for prim in stage.Traverse():
            if prim.GetName() == "Meshes" and prim.GetTypeName() == "Scope":
                meshes_scope = prim
                break
    if meshes_scope is None:
        raise SystemExit("no /Root/Meshes scope found — unexpected scene layout")
    for scope in meshes_scope.GetAllChildren():
        for asset in scope.GetAllChildren():
            yield asset, scope.GetName()


def collect_asset_samples(stage, asset, density, rng) -> tuple[np.ndarray, int]:
    """Sample every mesh under one asset Xform into world-space points."""
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    chunks, n_mesh = [], 0
    for prim in Usd.PrimRange(asset):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == "invisible" or img.ComputePurpose() != "default":
            continue
        m = xf.GetLocalToWorldTransform(prim)
        # Gf row-vector convention: translation lives in the last ROW,
        # p_world = p_local @ M3 + row3 (ExtractTranslation() reads the same)
        m3 = np.array([[m[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
        t = np.array([m[3][c] for c in range(3)], dtype=np.float64)
        chunks.append(sample_mesh(UsdGeom.Mesh(prim), m3, t, density, rng))
        n_mesh += 1
    if not chunks:
        return np.empty((0, 3)), 0
    return np.vstack(chunks), n_mesh


def point_in_poly(x: float, y: float, poly: list) -> bool:
    """Ray-casting test; poly is an [(x, y), …] ring."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def main() -> None:
    ap = argparse.ArgumentParser(description="InteriorAgent USD 场景 → 点云 PLY + 语义边车")
    ap.add_argument("--usd", required=True, help="场景 usda 路径")
    ap.add_argument("--ply", help="输出 PLY（默认 <usd_stem>.ply 同目录）")
    ap.add_argument("--sidecar", help="输出边车 JSON（默认 <usd_stem>_sidecar.json）")
    ap.add_argument("--rooms", help="rooms.json（房间平面多边形，用于房间归属）")
    ap.add_argument("--density", type=float, default=1000.0,
                    help="采样密度 点/m²（默认 1000，extract 0.05m 体素足够稠密）")
    ap.add_argument("--up", default="z", choices=["y", "z"],
                    help="舞台高度轴（InteriorAgent=Z-up；边车 OBB/房间旋转成内部 Y-up，"
                         "PLY 保持原生坐标，extract 用 --up z 摄入）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    try:
        if Usd is None:
            raise ImportError
    except ImportError:
        raise SystemExit("usd-core 未安装：pip install usd-core")

    usd_path = Path(a.usd)
    out_ply = Path(a.ply) if a.ply else usd_path.with_suffix(".ply")
    out_side = Path(a.sidecar) if a.sidecar else usd_path.with_name(f"{usd_path.stem}_sidecar.json")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise SystemExit(f"cannot open stage: {usd_path}")
    rng = np.random.default_rng(a.seed)

    rooms_poly = []
    if a.rooms:
        for i, r in enumerate(json.loads(Path(a.rooms).read_text()), 1):
            rooms_poly.append({"id": f"room_{i:02d}",
                               "label": r["room_type"].replace(" ", "_"),
                               "polygon": r["polygon"]})

    assets, all_pts = [], []
    for asset, scope in iter_assets(stage):
        samples, n_mesh = collect_asset_samples(stage, asset, a.density, rng)
        print(f"   · {asset.GetName():<28} scope={scope}", flush=True)
        if samples.shape[0] == 0:
            print(f"   ⚠️  无可采样网格: {asset.GetPath()}")
            continue
        all_pts.append(samples)
        # native (x, y, 0) ground center → internal frame for room assignment
        center_native = samples.mean(axis=0)
        center_rot = _to_yup(center_native[None, :].copy(), a.up)[0]
        room = ""
        for r in rooms_poly:
            if point_in_poly(float(center_native[0]), float(center_native[1]), r["polygon"]):
                room = r["label"]
                break
        label = ASSET_RE.sub("", asset.GetName())
        kind = "structure" if scope in ("floor", "wall", "ceiling") else "object"
        assets.append({
            "label": label, "prim": str(asset.GetPath()), "scope": scope,
            "room_hint": room or ("" if scope in ("other",) else ASSET_RE.sub("", scope)),
            "kind": kind, "n_meshes": n_mesh, "n_samples": int(samples.shape[0]),
            "obb_internal": fit_obb_pca(_to_yup(samples.copy(), a.up)).to_dict(),
        })
        print(f"   {label:<28} scope={scope:<16} pts={samples.shape[0]:>7}  room={room or '—'}")

    if not all_pts:
        raise SystemExit("no mesh samples collected — check the stage")
    pts_native = np.vstack(all_pts)
    lo, hi = pts_native.min(axis=0), pts_native.max(axis=0)
    print(f"\n🗺️  舞台原生包围盒(Z-up): x[{lo[0]:.2f},{hi[0]:.2f}] "
          f"y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]  共 {len(pts_native)} 点")

    _write_ply(out_ply, pts_native)
    print(f"✅ 点云 → {out_ply}")

    # room bounds in the INTERNAL frame: ground (x, y_poly) → (x, 0, -y_poly)
    y_up = a.up in ("z", "+z", "Z", "+Z")
    rooms_out = []
    for r in rooms_poly:
        xs = [p[0] for p in r["polygon"]]
        ys = [p[1] for p in r["polygon"]]
        if y_up:
            bmin, bmax = [min(xs), -0.05, -max(ys)], [max(xs), 2.9, -min(ys)]
        else:
            bmin, bmax = [min(xs), -0.05, min(ys)], [max(xs), 2.9, max(ys)]
        rooms_out.append({"id": r["id"], "label": r["label"],
                          "bounds_min": [round(v, 3) for v in bmin],
                          "bounds_max": [round(v, 3) for v in bmax]})

    sidecar = {
        "source_usd": str(usd_path), "up": a.up, "density_pts_per_m2": a.density,
        "seed": a.seed, "n_assets": len(assets),
        "n_points": int(pts_native.shape[0]),
        "native_bounds_min": lo.tolist(), "native_bounds_max": hi.tolist(),
        "rooms": rooms_out, "assets": assets,
    }
    out_side.write_text(json.dumps(sidecar, ensure_ascii=False, indent=1))
    print(f"✅ 语义边车（{len(assets)} 资产，内部 Y-up OBB）→ {out_side}")


def _write_ply(path: Path, pts: np.ndarray) -> None:
    """Binary-little-endian float32 xyz PLY (extract._read_ply-compatible)."""
    data = np.asarray(pts, dtype=np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {data.shape[0]}\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


if __name__ == "__main__":
    main()
