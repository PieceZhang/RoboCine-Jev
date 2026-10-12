#!/usr/bin/env python3
"""simview_isaac.py — RoboCine trajectory viewer on NVIDIA Isaac Sim.

Phase-1 upgrade path for the offline HTML viewer (cinema_traj/stages/view.py):
same inputs (scene JSON + traj_final.json), rendered by Isaac Sim's RTX
renderer with a real perspective camera (per-frame FOV), an animated subject
capsule, and a POV "shot preview" that the canvas diagram cannot give.

Subcommands:

  build    pure-python USD authoring (numpy only, NO Isaac Sim needed) —
           writes <traj_dir>/simview/simview.usda (+ meta.json sidecar)
  gui      interactive playback: launch Isaac Sim GUI, open the stage, scrub
           the native Kit timeline (POV = viewport camera /World/ShotCam)
  render   headless POV + overview PNGs (optional mp4 via ffmpeg)

Typical use (inside the isaac env, e.g. conda env_isaaclab):

    # 1) build — any python with numpy works
    python tools/simview_isaac.py build \
        --traj cinema_traj_out/frl_apartment_0/traj_final.json

    # 2) interactive GUI (needs a display)
    OMNI_KIT_ACCEPT_EULA=yes python tools/simview_isaac.py gui \
        cinema_traj_out/frl_apartment_0/traj_final.json

    # 3) headless render (no display needed)
    OMNI_KIT_ACCEPT_EULA=yes python tools/simview_isaac.py render \
        cinema_traj_out/frl_apartment_0/traj_final.json --every 1.0 --mp4

gui/render accept either the .usda built by `build` or a traj_final.json
(the stage is auto-built on first use and cached next to it).

Standalone by design: the USD authoring half mirrors cinema_traj's schemas
(scene JSON / traj_final frames / 3DGS PLY f_dc colors) WITHOUT importing the
repo, so it also runs under Isaac Sim's python where the repo is not installed.

Coordinate conventions match cinema_traj: +Y up, meters, yaw 0 → +Z. The
virtual camera uses the physical/gluLookAt convention (USD camera looks down
its local -Z; image right = local +X), so the POV render matches what a real
drone camera at that pose would film.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except Exception:                                   # tqdm is optional
    def _tqdm(it, **kw):
        seq = list(it); n = len(seq)
        for k, item in enumerate(seq):
            print(f"  {k + 1}/{n}", flush=True)
            yield item

_PROG_STREAM = None
def _prog_stream():
    """Progress stream for tqdm. Kit's [py stderr] logger is line-buffered, so
    a normal \\r tqdm bar on stderr only flushes when the bar closes. Writing
    straight to /dev/tty (the controlling terminal) bypasses Kit entirely and
    the bar animates live; falls back to stderr for detached/headless runs."""
    global _PROG_STREAM
    if _PROG_STREAM is None:
        try:
            _PROG_STREAM = open("/dev/tty", "w", buffering=1)
        except OSError:
            _PROG_STREAM = sys.stderr
    return _PROG_STREAM

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SH_C0 = 0.2820547913868781        # degree-0 SH constant (3DGS f_dc → rgb)
_DEFAULT_FOV = 55.0                # deg, vertical — cinema_traj's fallback

_SEGMENT_PALETTE = [
    (0.55, 0.75, 1.00), (1.00, 0.65, 0.30), (0.55, 0.90, 0.55), (0.95, 0.40, 0.35),
    (0.75, 0.55, 0.90), (0.90, 0.75, 0.40), (0.60, 0.85, 0.80), (0.90, 0.55, 0.70),
    (0.70, 0.70, 0.75), (0.85, 0.85, 0.50),
]
_KIND_COLORS = {"wall": (0.78, 0.76, 0.72), "floor": (0.42, 0.40, 0.38),
                "ceiling": (0.85, 0.85, 0.83)}
_PLACEMENT_COLORS = {"freestanding": (0.55, 0.42, 0.30), "wall": (0.45, 0.52, 0.60),
                     "ceiling": (0.75, 0.72, 0.55), "floor_wall": (0.50, 0.55, 0.50)}
_DOOR_COLOR = (0.35, 0.75, 0.45)
_SUBJECT_COLOR = (0.90, 0.55, 0.15)
_GLB_YAW_OFFSET = 0.0          # deg; spin the baked model if its forward isn't +Z
# default humanoid baked into the viewer when present (fetch once via
# scripts/download_assets.py); pass --subject-glb to override/None.
_DEFAULT_SUBJECT_GLB = _REPO_ROOT / "datasets" / "simview_assets" / "Xbot.glb"


def _resolve_subject_glb(arg: str | None) -> str | None:
    """--subject-glb resolution: 'none'/'off' disables the stand-in entirely
    (the cast/env carries the subject); unset → default Xbot.glb if present;
    a path is used as-is."""
    if arg and arg.strip().lower() in ("none", "off"):
        return None
    return arg if arg else (str(_DEFAULT_SUBJECT_GLB)
                            if _DEFAULT_SUBJECT_GLB.exists() else None)
_CAST_COLORS = [  # per-character costume tints (cloth-ish), cycled by index
    (0.55, 0.35, 0.25), (0.35, 0.45, 0.65), (0.50, 0.30, 0.55),
    (0.30, 0.55, 0.40), (0.65, 0.55, 0.35), (0.60, 0.35, 0.35),
]


# --------------------------------------------------------------------------- #
# Math helpers (mirror cinema_traj.common.se3 / obb, standalone)
# --------------------------------------------------------------------------- #
def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _camera_basis(eye: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Rotation with columns (xc, yc, zc): USD camera at `eye` looking at
    `target` with +Y up. USD cameras look down local -Z, image right = +X.
    Returns None when eye ≈ target."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    if np.linalg.norm(f) < 1e-9:
        return None
    zc = _normalize(-f)                       # backward
    up = np.array([0.0, 1.0, 0.0])
    if np.linalg.norm(np.cross(up, zc)) < 1e-6:   # looking straight up/down
        up = _normalize(up + np.array([1e-3, 0.0, 0.0]))
    xc = _normalize(np.cross(up, zc))         # image right
    yc = np.cross(zc, xc)                     # image up
    return np.stack([xc, yc, zc], axis=1)


def _quat_from_R(R: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) from a right-handed rotation matrix (columns = axes)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return s / 4.0, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    if m00 >= m11 and m00 >= m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (m21 - m12) / s, s / 4.0, (m01 + m10) / s, (m02 + m20) / s
    if m11 >= m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return (m02 - m20) / s, (m01 + m10) / s, s / 4.0, (m12 + m21) / s
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4.0


def _obb_matrix(obb: dict) -> np.ndarray:
    """4x4 row-major matrix placing a unit Cube (size=1, centered) over the OBB.

    Matches cinema_traj OBB.to_world: q_world = (q_local ⊙ extents) @ R.T + c
    with p_row-vector convention — pxr GfMatrix4d is row-major, translation in
    the last row.
    """
    R = np.asarray(obb["rotation"], float)          # columns = local axes
    e = np.asarray(obb["extents"], float)
    A = np.diag(e) @ R.T
    M = np.eye(4)
    M[:3, :3] = A
    M[3, :3] = np.asarray(obb["center"], float)
    return M


def _angle_lerp(a: float, b: float, u: float) -> float:
    d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + d * u


def _subject_state(track: dict, t: float) -> tuple[np.ndarray, float]:
    """Clamped interpolation of the subject track — same math as
    cinema_traj.common.subject.SubjectTrack.state_at."""
    fr = track["frames"]
    if t <= fr[0]["t"]:
        f = fr[0]
        return np.asarray(f["pos"], float), float(f["heading"])
    if t >= fr[-1]["t"]:
        f = fr[-1]
        return np.asarray(f["pos"], float), float(f["heading"])
    lo, hi = 0, len(fr) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fr[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = fr[lo], fr[hi]
    u = (t - a["t"]) / max(b["t"] - a["t"], 1e-9)
    pos = (1.0 - u) * np.asarray(a["pos"], float) + u * np.asarray(b["pos"], float)
    return pos, _angle_lerp(float(a["heading"]), float(b["heading"]), u)


def _focal_from_vfov(vfov_deg: float) -> float:
    """Vertical FOV → focalLength (mm) for 16:9 apertures 36 × 20.25 mm."""
    return (20.25 / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)


def _heading_vec(heading_rad: float) -> np.ndarray:
    """yaw 0 → +Z, positive toward +X (cinema_traj convention)."""
    return np.array([math.sin(heading_rad), 0.0, math.cos(heading_rad)])


def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not s or s[0].isdigit():
        s = "_" + s
    return s


# --------------------------------------------------------------------------- #
# Point-cloud loading (PLY incl. 3DGS f_dc colors / PCD / NPZ)
# --------------------------------------------------------------------------- #
_PLY_TYPES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B",
              "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
              "int": "i", "int32": "i", "uint": "I", "uint32": "I",
              "float": "f", "float32": "f", "double": "d", "float64": "d"}


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith((b"end_header\n", b"end_header\r\n")):
            byte = fh.read(1)
            if not byte:
                raise ValueError(f"{path}: PLY header has no end_header")
            header += byte
        h = header.decode("latin1")
        if "binary_little_endian" in h:
            endian, fmt = "<", "bin"
        elif "binary_big_endian" in h:
            endian, fmt = ">", "bin"
        else:
            fmt = "ascii"
        props, n, in_v = [], 0, False
        for line in h.splitlines():
            p = line.split()
            if not p:
                continue
            if p[0] == "element":
                in_v = p[1] == "vertex"
                if in_v:
                    n = int(p[2])
            elif p[0] == "property" and in_v:
                props.append((p[1], p[2]))
        names = [nm for _, nm in props]
        if "x" not in names:
            raise ValueError(f"{path}: PLY has no x property")
        if fmt == "ascii":
            rows = []
            for line in fh.read().decode("ascii", "ignore").splitlines()[:n]:
                vals = line.split()
                if len(vals) >= 3:
                    rows.append((float(vals[0]), float(vals[1]), float(vals[2])))
            pts = np.asarray(rows, dtype=np.float64)
            rgb = None
        else:
            dt = np.dtype([(f"{nm}_{i}", endian + _PLY_TYPES[t])
                           for i, (t, nm) in enumerate(props)])
            rec = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
            idx = {nm: i for i, (_, nm) in enumerate(props)}
            pts = np.stack([rec[f"x_{idx['x']}"], rec[f"y_{idx['y']}"],
                            rec[f"z_{idx['z']}"]], axis=1).astype(np.float64)
            rgb = None
            if all(k in idx for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
                rgb = np.stack([np.clip(0.5 + _SH_C0 * rec[f"f_dc_{i}_{idx[f'f_dc_{i}']}"]
                                         .astype(np.float64), 0.0, 1.0)
                                for i in range(3)], axis=1)
    return pts, rgb


def _read_pcd(path: Path) -> np.ndarray:
    raw = path.read_bytes()
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
        raise ValueError(f"{path}: PCD header has no DATA line")
    fields = header.get("FIELDS", ["x", "y", "z"])
    sizes = [int(s) for s in header.get("SIZE", ["4"] * len(fields))]
    types = header.get("TYPE", ["F"] * len(fields))
    n = int(header.get("POINTS", ["0"])[0])
    tm = {"I": "i", "U": "u", "F": "f"}   # PCD: I=signed, U=unsigned int
    if mode == "ascii":
        rows = []
        for line in raw[data_start:].decode("ascii", "ignore").splitlines():
            vals = line.split()
            if len(vals) >= len(fields):
                rows.append(tuple(float(v) for v in vals[:len(fields)]))
        arr = np.asarray(rows, dtype=np.float64)
        return arr[:, [fields.index(a) for a in "xyz"]]
    dt = np.dtype([(f, "<" + tm[t] + str(s))
                   for f, t, s in zip(fields, types, sizes)])
    rec = np.frombuffer(raw[data_start:data_start + n * dt.itemsize], dtype=dt, count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)


def _load_source_cloud(scene: dict, explicit: str | None, voxel: float,
                       max_points: int, margin: float):
    """(points, rgb|None, provenance-note) or (None, None, reason)."""
    prov = scene.get("provenance", {}) or {}
    if explicit in (None, "auto"):
        rel = prov.get("source_map")
        if not rel:
            return None, None, "scene has no provenance.source_map"
        cand = [Path(rel), _REPO_ROOT / rel, scene and (Path(scene["_path"]).parent / rel)]
        path = next((p for p in cand if p and p.exists()), None) if rel else None
        if path is None:
            return None, None, f"source_map not found: {rel}"
    elif explicit == "none":
        return None, None, "disabled"
    else:
        path = Path(explicit)
        if not path.exists():
            return None, None, f"--points not found: {explicit}"

    if path.suffix.lower() == ".ply":
        pts, rgb = _read_ply(path)
    elif path.suffix.lower() == ".pcd":
        pts, rgb = _read_pcd(path), None
    elif path.suffix.lower() == ".npz":
        pts, rgb = np.load(path)["points"].astype(np.float64), None
    else:
        return None, None, f"unsupported cloud format: {path.suffix}"

    note = f"{len(pts)} pts from {path.name}"
    if str(prov.get("map_up", "y")).lower() == "z":        # Z-up → Y-up (RH)
        out = pts.copy()
        out[:, 1], out[:, 2] = pts[:, 2], -pts[:, 1]
        pts = out
        note += " (z-up rotated)"
    shift = prov.get("map_shift")
    if shift:
        pts = pts + np.asarray(shift, float)
        note += f" shifted {shift}"
    rooms = scene.get("rooms") or []
    if margin >= 0 and rooms:
        lo = np.min([r["bounds_min"] for r in rooms], axis=0) - margin
        hi = np.max([r["bounds_max"] for r in rooms], axis=0) + margin
        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts = pts[keep]
        if rgb is not None:
            rgb = rgb[keep]
        note += f" → {len(pts)} after crop(margin {margin}m)"
    if voxel > 0 and len(pts) > max_points:
        keys = np.floor(pts / voxel).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        keep = np.sort(keep)
        pts, rgb = pts[keep], (rgb[keep] if rgb is not None else None)
        note += f" → {len(pts)} after voxel {voxel}m"
    if max_points and len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        sel.sort()
        pts, rgb = pts[sel], (rgb[sel] if rgb is not None else None)
        note += f" → capped {max_points}"
    if rgb is None:                                          # height ramp
        if len(pts):
            lo, hi = float(pts[:, 1].min()), float(pts[:, 1].max())
            u = (pts[:, 1] - lo) / max(hi - lo, 1e-6)
            rgb = np.stack([0.25 + 0.55 * u, 0.45 + 0.25 * u, 0.85 - 0.55 * u], axis=1)
    return pts, rgb, note


# --------------------------------------------------------------------------- #
# USDA emission
# --------------------------------------------------------------------------- #
def _f(v: float) -> str:
    return f"{float(v):.5f}"


def _t3(v) -> str:
    return f"({_f(v[0])}, {_f(v[1])}, {_f(v[2])})"


_c3 = _t3  # alias: color triples and vec3 triples share formatting


def _vec3_array(arr: np.ndarray, per_line: int = 4) -> str:
    items = [_t3(p) for p in arr]
    lines = ["    " + ", ".join(items[i:i + per_line]) + ","
             for i in range(0, len(items), per_line)]
    return "\n".join(lines).rstrip(",")


def _time_samples(attr_decl: str, samples: dict[int, str], indent: str = "    ") -> str:
    lines = [f"{indent}{attr_decl}.timeSamples = {{"]
    for code in sorted(samples):
        lines.append(f"{indent}    {code}: {samples[code]},")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


class _Usda:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def raw(self, s: str) -> None:
        self.lines.append(s)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _emit_cube(w: _Usda, name: str, obb: dict, color, doc: str,
               opacity: float = 1.0, mat_path: str | None = None) -> None:
    """Environment box as a BAKED-vertex Mesh (not a scaled unit Cube).

    RTX/Hydra lights Cube gprims wrongly under the non-uniform scale that a
    matrix4d bakes in (interior faces come out black regardless of lights);
    a plain Mesh with world-space verts + explicit normals renders correctly.
    opacity < 1 makes boxes translucent (structure context without hiding the
    point cloud) — the collision-field visualization becomes a glass overlay.
    """
    R = np.asarray(obb["rotation"], float)
    e = np.asarray(obb["extents"], float) / 2.0
    c = np.asarray(obb["center"], float)
    # 8 local corners (unit box), world = corner*extents @ R.T + c
    corners_local = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1)
                              for sz in (-1, 1)], float)
    cw = corners_local * e @ R.T + c
    # 6 quad faces, outward-CCW windings (corner ids: sx outer, sz inner)
    faces_local = [((4, 6, 7, 5), (+1, 0)), ((0, 1, 3, 2), (-1, 0)),
                   ((2, 3, 7, 6), (+1, 1)), ((0, 4, 5, 1), (-1, 1)),
                   ((1, 5, 7, 3), (+1, 2)), ((0, 2, 6, 4), (-1, 2))]
    pts, nrms, fvc, fvi = [], [], [], []
    for idxs, (sgn, axis) in faces_local:
        n_world = R[:, axis] * sgn
        base = len(pts)
        for li in idxs:
            pts.append(cw[li])
            nrms.append(n_world)
        fvc.append(4)
        fvi.extend([base, base + 1, base + 2, base + 3])
    if opacity < 1.0 and mat_path:
        bind = f"\n        rel material:binding = <{mat_path}>"
    else:
        bind = ""
    w.raw(f'''    def Mesh "{_sanitize(name)}" (
        doc = "{doc}"
    )
    {{
        uniform bool doubleSided = 1
        int[] faceVertexCounts = [{", ".join(map(str, fvc))}]
        int[] faceVertexIndices = [{", ".join(map(str, fvi))}]
        point3f[] points = [{", ".join(_t3(p) for p in pts)}]
        normal3f[] normals = [{", ".join(_t3(n) for n in nrms)}] (
            interpolation = "faceVarying"
        )
        color3f[] primvars:displayColor = [{_c3(color)}]{bind}
        custom string robocine:label = "{doc}"
    }}''')


def _frustum_pts_local(depth: float = 1.2, vfov_deg: float = _DEFAULT_FOV,
                       aspect: float = 16 / 9) -> np.ndarray:
    """Camera-local FOV line frustum: pupil at origin, view along -Z, +Y up —
    8 line segments (4 pupil→corner + 4 corner rectangle) as (16,3) points.
    ONE such curve is parented to CamBody (same time-sampled pose as ShotCam)
    → it rides the trajectory live."""
    hh = math.tan(math.radians(vfov_deg) / 2.0) * depth
    hw = hh * aspect
    corners = [np.array([sx * hw, sy * hh, -depth])
               for sx in (-1, 1) for sy in (-1, 1)]    # LL, LR, UL? (x outer)
    # corner order from the double loop: (-1,-1),(-1,1),(1,-1),(1,1)
    # rectangle loop: c0-c2, c2-c3, c3-c1, c1-c0
    o = np.zeros(3)
    segs = [o, corners[0], o, corners[1], o, corners[2], o, corners[3],
            corners[0], corners[2], corners[2], corners[3],
            corners[3], corners[1], corners[1], corners[0]]
    return np.asarray(segs)


def _local_box_arrays(center=(0, 0, 0), extents=(1, 1, 1)) -> tuple:
    """(pts, nrms, faceVertexCounts, faceVertexIndices) for an axis-aligned box
    in LOCAL space (24 verts, 6 quads, outward normals) — used for the camera
    body proxy. Same winding scheme as _emit_cube."""
    c = np.asarray(center, float)
    e = np.asarray(extents, float) / 2.0
    lc = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1)
                   for sz in (-1, 1)], float)
    verts = lc * e + c
    faces = [((4, 6, 7, 5), (1, 0, 0)), ((0, 1, 3, 2), (-1, 0, 0)),
             ((2, 3, 7, 6), (0, 1, 0)), ((0, 4, 5, 1), (0, -1, 0)),
             ((1, 5, 7, 3), (0, 0, 1)), ((0, 2, 6, 4), (0, 0, -1))]
    pts, nrms, fvc, fvi = [], [], [], []
    for idxs, nrm in faces:
        base = len(pts)
        for li in idxs:
            pts.append(verts[li])
            nrms.append(np.asarray(nrm, float))
        fvc.append(4)
        fvi.extend([base, base + 1, base + 2, base + 3])
    return pts, nrms, fvc, fvi


def _emit_wireframe_box(w: _Usda, name: str, obb: dict, color, doc: str) -> None:
    """OBB as 12 edge lines (BasisCurves) — the collision field as an overlay
    instead of solid blocks; renders reliably in every RTX mode."""
    R = np.asarray(obb["rotation"], float)
    e = np.asarray(obb["extents"], float) / 2.0
    c = np.asarray(obb["center"], float)
    lc = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1)
                   for sz in (-1, 1)], float)
    cw = lc * e @ R.T + c
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 6), (4, 5), (5, 7), (6, 7)]
    pts = np.asarray([cw[a] for a, b in edges] + [cw[b] for a, b in edges])
    counts = ", ".join(["2"] * len(edges))
    w.raw(f'''    def BasisCurves "{_sanitize(name)}" (
        doc = "{doc}"
    )
    {{
        uniform token type = "linear"
        int[] curveVertexCounts = [{counts}]
        point3f[] points = [
{_vec3_array(pts)}
        ]
        float[] widths = [0.007]
        color3f[] primvars:displayColor = [{_c3(color)}]
    }}''')


def _emit_static_camera(w: _Usda, name: str, eye: np.ndarray, target: np.ndarray,
                        fov_deg: float, doc: str) -> None:
    R = _camera_basis(eye, target)
    if R is None:
        R = np.eye(3)
    q = _quat_from_R(R)
    w.raw(f'''    def Camera "{name}" (
        doc = "{doc}"
    )
    {{
        float2 clippingRange = (0.05, 500)
        float horizontalAperture = 36
        float verticalAperture = 20.25
        float focalLength = {_f(_focal_from_vfov(fov_deg))}
        token projection = "perspective"
        float3 xformOp:translate = {_t3(eye)}
        quatf xformOp:orient = ({_f(q[0])}, {_f(q[1])}, {_f(q[2])}, {_f(q[3])})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}''')


def _load_cast_usda_cached(f: Path, loader) -> dict:
    """Disk-cached cast geometry parse: deterministic in (path, mtime, size).

    The flattened people usda files are static repo assets re-parsed on
    every simview build (~0.4 s per character); the .npz fingerprint makes
    repeat builds skip the parse entirely.
    """
    import hashlib
    st = f.stat()
    fp = f"{f.name}:{st.st_mtime_ns}:{st.st_size}".encode()
    key = hashlib.md5(fp).hexdigest()[:16]
    cache_dir = _REPO_ROOT / ".cache" / "cast_geom"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"{f.stem}.{key}.npz"
    if cpath.exists():
        z = np.load(cpath, allow_pickle=False)
        return {"verts": z["verts"], "faces": z["faces"],
                "face_colors": z["face_colors"]}
    out = loader(str(f))
    np.savez_compressed(cpath, verts=out["verts"], faces=out["faces"],
                        face_colors=out["face_colors"])
    return out


def _bake_cast_character(spec: dict, index: int) -> dict:
    """Bake one --cast entry to seat-local triangles.

    spec: {id, glb, anim?, t?, height_m?, seat:[x,y,z], heading_deg?, color?,
           align?}  — OR — {id, usda, ...} for a pre-flattened people_dh file
    (tools/usd_dh_flatten.py output: real human geometry, per-face palette
    colours baked in; align defaults to "floor" and heights map to native
    stature).

    The GLB is skinned at `anim`@`t` via tools/glb_pose_bake
    (KayKit rigs ship ``Sit_Chair_Pose`` / ``T-Pose``), scaled so its STANDING
    height equals height_m, then pre-transformed so the requested anchor sits
    at the local origin: align="seat" (default) anchors the hips-contact point
    (lowest vertex near the hips joint — bottom of the seated buttock); "floor"
    anchors bbox feet-centre.  The emitted Xform then only carries
    translate=`seat` + yaw=`heading_deg` (0 → model faces +Z, matching the
    subject's heading convention — KayKit forward is +Z).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if spec.get("usda"):                                  # flattened human
        from usd_dh_flatten import load_cast_usda
        p = Path(spec["usda"]).expanduser()
        f = p if p.is_absolute() else _REPO_ROOT / p
        out = _load_cast_usda_cached(f, load_cast_usda)
        V = out["verts"].astype(np.float64)
        F = np.asarray(out["faces"], dtype=np.int64)
        fc = out["face_colors"]
        h_t = spec.get("height_m")
        native_h = float(V[:, 1].max() - V[:, 1].min())
        if h_t:                                          # rescale if asked
            V *= float(h_t) / max(native_h, 1e-6)
        ref = np.array([(V[:, 0].min() + V[:, 0].max()) / 2.0,
                        V[:, 1].min(),
                        (V[:, 2].min() + V[:, 2].max()) / 2.0])  # feet centre
        if "color" in spec:                       # vivid probe override
            fc = None
        return {
            "id": _sanitize(spec.get("id", f"char_{index}")),
            "V": V - ref, "F": F,
            "seat": np.array(spec["seat"], float),
            "yaw": math.radians(float(spec.get("heading_deg", 0.0))),
            "color": tuple(spec.get("color",
                                    _CAST_COLORS[index % len(_CAST_COLORS)])),
            "face_colors": fc,
            "note": f"{Path(spec['usda']).name} flat "
                    f"(h={native_h:.2f} m, align=floor"
                    f"{', probe-colour' if fc is None else ', palette'})",
        }
    from glb_pose_bake import load_glb_posed
    p = Path(spec["glb"]).expanduser()
    glb = p if p.is_absolute() else _REPO_ROOT / p
    anim = spec.get("anim")
    t = float(spec.get("t", 0.0))
    use_tex = bool(spec.get("texture_colors", False)) and "color" not in spec
    out = load_glb_posed(str(glb), anim, t, sample_textures=use_tex)
    V = out["verts"].astype(np.float64)
    F = np.asarray(out["faces"], dtype=np.int64)
    h_t = float(spec.get("height_m", 1.65))
    stand_h = None
    for ref_anim in ("T-Pose", "static"):              # standing reference
        try:
            stand = load_glb_posed(str(glb), ref_anim, 0.0)["verts"]
            stand_h = float(stand[:, 1].max() - stand[:, 1].min())
            break
        except ValueError:
            continue
    native_h = stand_h or float(V[:, 1].max() - V[:, 1].min())
    s = h_t / max(native_h, 1e-6)
    V *= s
    joints = {k.lower(): v * s for k, v in out["joints"].items()}
    hips = next((v for k, v in joints.items() if "hips" in k or "pelvis" in k),
                None)
    torso = next((v for k, v in joints.items() if k in ("torso", "root")), None)
    align = spec.get("align", "seat")
    if align == "seat" and hips is None:
        align = "contact" if torso is not None else "floor"
        print(f"⚠️  cast {spec.get('id')}: no hips/pelvis joint — "
              f"{align}-aligned instead")
    if align == "floor":
        ref = np.array([(V[:, 0].min() + V[:, 0].max()) / 2.0,
                        V[:, 1].min(),
                        (V[:, 2].min() + V[:, 2].max()) / 2.0])
    elif align == "contact" or (align == "seat" and hips is None):
        # bottom of the torso column: lowest vertex under the torso/root joint
        # (rigid-part figures' "chair contact" when no hip bone exists)
        a0 = torso if torso is not None else np.array(
            [(V[:, 0].min() + V[:, 0].max()) / 2.0, 0.0,
             (V[:, 2].min() + V[:, 2].max()) / 2.0])
        m = ((np.abs(V[:, 0] - a0[0]) < 0.20)
             & (np.abs(V[:, 2] - a0[2]) < 0.30))
        contact_y = float(V[m][:, 1].min()) if m.any() else float(V[:, 1].min())
        ref = np.array([a0[0], contact_y, a0[2]])
    else:
        # buttock contact: lowest vertex near the hips joint AND within a band
        # just under it (excludes dangling feet in front of the hips and any
        # torso folds above the seat line)
        m = ((np.abs(V[:, 0] - hips[0]) < 0.20)
             & (np.abs(V[:, 2] - hips[2]) < 0.20)
             & (V[:, 1] > hips[1] - 0.25) & (V[:, 1] < hips[1] + 0.03))
        butt_y = float(V[m][:, 1].min()) if m.any() else float(hips[1] - 0.08)
        ref = np.array([hips[0], butt_y, hips[2]])
    fc = out.get("face_colors") if use_tex else None
    return {
        "id": _sanitize(spec.get("id", f"char_{index}")),
        "V": V - ref, "F": F,
        "seat": np.array(spec["seat"], float),
        "yaw": math.radians(float(spec.get("heading_deg", 0.0))),
        "color": tuple(spec.get("color", _CAST_COLORS[index % len(_CAST_COLORS)])),
        "face_colors": fc,
        "note": f"{Path(spec['glb']).name} @{anim or 'bind'} "
                f"(h={h_t:.2f} m, align={align}"
                f"{', textured' if fc is not None else ''})",
    }


def build_usda(traj_path: Path, scene_path: Path, out_path: Path,
               fps: int = 30, points: str | None = "auto",
               cloud_voxel: float = 0.03, cloud_max: int = 200000,
               cloud_margin: float = 1.5,
               env_opacity: float = 1.0, env_style: str = "solid",
               subject_glb: str | None = None,
               cast: str | None = None, markers: bool = False,
               overview_offset=(0.62, 0.55, 0.66),
               _subject_glb_arg_none: bool = False) -> dict:
    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["_path"] = str(scene_path)
    subject_glb_none = subject_glb is None and _subject_glb_arg_none
    frames = traj["frames"]
    if not frames:
        raise ValueError("traj_final.json has no frames")
    cast_chars = []
    if cast:
        specs = json.loads(Path(cast).read_text(encoding="utf-8"))
        for i, spec in enumerate(specs.get("characters", [])):
            cast_chars.append(_bake_cast_character(spec, i))
            print(f"  cast  {cast_chars[-1]['id']:12s} {cast_chars[-1]['note']}")
    t0, t1 = frames[0]["t"], frames[-1]["t"]
    ts = np.array([f["t"] for f in frames], float)
    pos = np.array([f["pos"] for f in frames], float)
    look = np.array([f["look"] for f in frames], float)
    fov = np.array([f["fov_deg"] if f.get("fov_deg") is not None else _DEFAULT_FOV
                    for f in frames], float)

    n_codes = max(1, int(round((t1 - t0) * fps)))
    codes = np.arange(n_codes + 1)
    tt = t0 + codes / fps
    P = np.stack([np.interp(tt, ts, pos[:, i]) for i in range(3)], axis=1)
    L = np.stack([np.interp(tt, ts, look[:, i]) for i in range(3)], axis=1)
    FV = np.interp(tt, ts, fov)

    # camera keys (translate / orient / focalLength)
    trans_samples: dict[int, str] = {}
    orient_samples: dict[int, str] = {}
    focal_samples: dict[int, str] = {}
    last_q = (1.0, 0.0, 0.0, 0.0)
    for k, code in enumerate(codes):
        trans_samples[int(code)] = _t3(P[k])
        R = _camera_basis(P[k], L[k])
        if R is not None:
            last_q = _quat_from_R(R)
        orient_samples[int(code)] = (f"({_f(last_q[0])}, {_f(last_q[1])}, "
                                     f"{_f(last_q[2])}, {_f(last_q[3])})")
        focal_samples[int(code)] = _f(_focal_from_vfov(FV[k]))

    # segment palette index per frame
    seg_names = []
    seg_idx = np.zeros(len(frames), int)
    for i, f in enumerate(frames):
        nm = f.get("segment", "?")
        if nm not in seg_names:
            seg_names.append(nm)
        seg_idx[i] = seg_names.index(nm)

    # subject keys
    track = scene.get("subject_track")
    subj_samples: dict[int, str] | None = None
    head_samples: dict[int, str] | None = None
    nose_samples: dict[int, str] | None = None
    if track and not subject_glb_none:
        # --subject-glb none → no Subject prim at all (the cast figure plays
        # the subject); render/GUI drivers skip absent prims safely
        subj_samples, head_samples, nose_samples = {}, {}, {}
        h = float(track.get("height_m", 1.75))
        subj_samples, head_samples, nose_samples = {}, {}, {}
        for code in codes:
            t = t0 + code / fps
            p, hd = _subject_state(track, t)
            subj_samples[int(code)] = _t3(p + np.array([0.0, 0.5 * h, 0.0]))
            head_samples[int(code)] = _t3(p + np.array([0.0, 0.92 * h, 0.0]))
            nose = p + _heading_vec(hd) * 0.33 + np.array([0.0, 0.80 * h, 0.0])
            nose_samples[int(code)] = _t3(nose)

    # environment bbox for overview framing
    corners = []
    for grp, kind_key, color_fn in (
            (scene.get("structures", []), "kind", None),
            (scene.get("objects", []), "placement", None),
            (scene.get("doors", []), None, None)):
        for item in grp:
            c = _obb_matrix(item["obb"])
            for sx in (-0.5, 0.5):
                for sy in (-0.5, 0.5):
                    for sz in (-0.5, 0.5):
                        corners.append((np.array([sx, sy, sz, 1.0]) @ c)[:3])
    corners.extend(list(P))
    corners = np.asarray(corners)
    # overview: frame the ACTION (trajectory + subject track), not the whole
    # apartment bbox — the env bbox can span rooms away from where the camera
    # plays (and clamp a whole-apartment corner puts the eye inside walls).
    lo, hi = corners.min(0), corners.max(0)        # furniture extent (context)
    act_sets = [P.copy()]
    if track:
        fr_sorted = sorted(track["frames"], key=lambda f: f["t"])
        sps = np.array([f["pos"] for f in fr_sorted], float)
        sps[:, 1] += 0.5 * float(track.get("height_m", 1.75))  # eye-band ≈ torso
        act_sets.append(sps)
    if cast_chars:
        # seated figures count as action (chest band) so the overview frames them
        act_sets.append(np.array([c["seat"] + np.array([0.0, 0.55, 0.0])
                                  for c in cast_chars]))
    act = np.vstack(act_sets)
    alo, ahi = act.min(0), act.max(0)
    acenter = (alo + ahi) / 2.0
    aspan = float(np.linalg.norm((ahi - alo))) or 1.0
    # eye: diagonal, slightly above the action; distance scales with action size
    dist = max(2.8, 0.9 * aspan)
    ov_eye = acenter + np.asarray(overview_offset, float) * dist
    # never leave the env shell: clamp to the apartment bbox margin, and into
    # the ACTION ROOM's bounds (else the overview lands in walls/above-ceiling
    # for open-plan scenes with mismatched room heights — kujiale_0021 hit this)
    ov_eye = np.clip(ov_eye, lo + 0.15, hi - 0.15)
    ceil_ys = [s["obb"]["center"][1] - s["obb"]["extents"][1] / 2
               for s in scene.get("structures", []) if s.get("kind") == "ceiling"]
    if ceil_ys:
        ov_eye[1] = min(ov_eye[1], max(ceil_ys) - 0.25)
    act_room = None
    for r in scene.get("rooms", []):
        rlo, rhi = np.asarray(r["bounds_min"]), np.asarray(r["bounds_max"])
        if np.all(acenter >= rlo) and np.all(acenter <= rhi):
            act_room = (rlo, rhi)
            break
    if act_room is not None:
        rlo, rhi = act_room
        ov_eye = np.clip(ov_eye, rlo + 0.30, rhi - 0.30)

    # cloud
    pts, rgb, cloud_note = _load_source_cloud(scene, points, cloud_voxel,
                                              cloud_max, cloud_margin)

    w = _Usda()
    w.raw(f'''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
    startTimeCode = 0
    endTimeCode = {n_codes}
    timeCodesPerSecond = {fps}
    framesPerSecond = {fps}
    doc = "RoboCine simview — scene {traj.get('scene_id', '?')} | traj {traj_path.name} | tools/simview_isaac.py"
)

def Xform "World" (
    customData = {{
        string "robocine:sceneId" = "{traj.get('scene_id', '')}"
        string "robocine:traj" = "{traj_path.name}"
        float "robocine:t0" = {t0}
        float "robocine:fps" = {fps}
    }}
)
{{''')
    # -- environment ------------------------------------------------------- #
    # scene provenance may point at a full USD environment (e.g. InteriorAgent
    # furnished apartment); when present and usable, REFERENCE it instead of
    # the collision boxes: the viewer renders the real furnished scene.
    env_usd = (scene.get("provenance") or {}).get("env_usd")
    if env_usd:
        p = Path(env_usd)
        if not p.is_absolute():
            p = _REPO_ROOT / env_usd
        if not p.exists():
            print(f"⚠️  provenance.env_usd missing: {p} — falling back to boxes")
            env_usd = None

    if env_usd and env_style != "boxes":
        # RELATIVE layer path: the stage must survive repo moves (an
        # absolute /home/... reference silently vanishes the room on any
        # other checkout/machine — props + walls disappear from Isaac)
        rel = os.path.relpath(p.resolve(), out_path.parent.resolve())
        w.raw(f'''    def Xform "Environment" (
        prepend references = @{rel}@
    )
    {{}}
''')
        n_env = 0
    else:
        def _mat_key(s_kind=None, o=None, door=False):
            if door:
                return "door"
            if o is not None:
                return o.get("placement", "freestanding")
            return s_kind

        use_mats = env_opacity < 1.0
        if use_mats:
            w.raw('    def Scope "_Materials"\n    {')
            seen = set()
            for grp, is_door in ((scene.get("structures", []), False),
                                 (scene.get("objects", []), False),
                                 (scene.get("doors", []), True)):
                for item in grp:
                    key = _mat_key(item.get("kind"), None if grp is not scene.get("objects") else item,
                                   is_door)
                    if key in seen:
                        continue
                    seen.add(key)
                    col = (_DOOR_COLOR if key == "door" else
                           _KIND_COLORS.get(key) or _PLACEMENT_COLORS.get(key, (0.7, 0.7, 0.7)))
                    w.raw(f'''        def Material "RoboMat_{_sanitize(key)}"
            {{
                token outputs:surface.connect = </World/_Materials/RoboMat_{_sanitize(key)}/Preview.outputs:surface>
                def Shader "Preview"
                {{
                    uniform token info:id = "UsdPreviewSurface"
                    token outputs:surface
                    color3f inputs:diffuseColor = {_c3(col)}
                    float inputs:opacity = {_f(env_opacity)}
                    float inputs:roughness = 0.85
                    float inputs:metallic = 0.00000
                }}
            }}''')
            w.raw("    }\n")

    def _mat_path(key):
        return f"/World/_Materials/RoboMat_{_sanitize(key)}" if use_mats else None

    n_env = 0
    if env_style != "off" and not (env_usd and env_style != "boxes"):
        w.raw('    def Scope "Environment"\n    {')
        for s in scene.get("structures", []):
            doc = f"{s.get('kind', '?')} {s.get('room', '')}".strip()
            col = _KIND_COLORS.get(s.get("kind"), (0.7, 0.7, 0.7))
            if env_style == "wireframe":
                _emit_wireframe_box(w, s["id"], s["obb"], col, doc)
            else:
                _emit_cube(w, s["id"], s["obb"], col, doc, opacity=env_opacity,
                           mat_path=_mat_path(_mat_key(s.get("kind"))))
            n_env += 1
        for o in scene.get("objects", []):
            doc = f"{o.get('label', o['id'])} ({o.get('placement', '')})"
            col = _PLACEMENT_COLORS.get(o.get("placement", "freestanding"),
                                        _PLACEMENT_COLORS["freestanding"])
            if env_style == "wireframe":
                _emit_wireframe_box(w, o["id"], o["obb"], col, doc)
            else:
                _emit_cube(w, o["id"], o["obb"], col, doc, opacity=env_opacity,
                           mat_path=_mat_path(_mat_key(o=o)))
            n_env += 1
        for d in scene.get("doors", []):
            doc = f"door {d.get('kind', '')}"
            if env_style == "wireframe":
                _emit_wireframe_box(w, d["id"], d["obb"], _DOOR_COLOR, doc)
            else:
                _emit_cube(w, d["id"], d["obb"], _DOOR_COLOR, doc,
                           opacity=env_opacity, mat_path=_mat_path("door"))
            n_env += 1
        w.raw("    }\n")
    # env_style == "off": no Environment scope at all

    # interior light rig — the sealed room shell occludes sun/dome in RTX, so
    # without interior lights POV renders are pitch black. Grid of sphere
    # lights just under the ceiling, ~2.5 m spacing over the room union.
    ys = [s["obb"]["center"][1] + s["obb"]["extents"][1] / 2
          for s in scene.get("structures", []) if s.get("kind") == "ceiling"]
    y_light = (min(ys) - 0.25) if ys else float(corners.max(0)[1]) - 0.25
    x0, z0 = float(corners.min(0)[0]), float(corners.min(0)[2])
    x1, z1 = float(corners.max(0)[0]), float(corners.max(0)[2])
    xs = np.arange(x0 + 0.6, x1 + 0.01, 2.5) if x1 > x0 else [x0]
    zs = np.arange(z0 + 0.6, z1 + 0.01, 2.5) if z1 > z0 else [z0]
    w.raw('    def Xform "Lights"\n    {')
    li = 0
    for lx in xs:
        for lz in zs:
            w.raw(f'''        def SphereLight "Key_{li:02d}"
        {{
            float inputs:intensity = 40000
            float inputs:radius = 0.25
            color3f inputs:color = (1.00000, 0.96000, 0.90000)
            float3 xformOp:translate = {_t3((lx, y_light, lz))}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}''')
            li += 1
    w.raw("    }\n")

    # -- cloud -------------------------------------------------------------- #
    if pts is not None and len(pts):
        w.raw(f'''    def Points "Cloud"
    {{
        point3f[] points = [
{_vec3_array(pts)}
        ]
        float[] widths = [{_f(max(cloud_voxel * 0.6, 0.01))}] (
            interpolation = "constant"
        )
        color3f[] primvars:displayColor = [
{_vec3_array(rgb)}
        ] (
            interpolation = "vertex"
        )
    }}''')

    # -- subject ------------------------------------------------------------ #
    # marker radii (fraction of height); 0.0 = marker prim omitted entirely
    _HEAD_R = 0.0
    _NOSE_R = 0.0
    if subj_samples:
        h = float(track.get("height_m", 1.75))
        # feet position + yaw quats for an articulated/statue subject
        feet_samples = {}
        yaw_samples = {}
        for code in codes:
            t = t0 + code / fps
            p, hd = _subject_state(track, t)
            feet_samples[int(code)] = _t3(p)
            yaw_samples[int(code)] = (f"({math.cos(hd / 2):.5f}, 0.00000, "
                                       f"{math.sin(hd / 2):.5f}, 0.00000)")

        if subject_glb:
            # -- human statue (bind pose baked from GLB; parent's xform carries
            #    the walk): proven-good path — same pattern as CamBody hull.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from glb_static_mesh import load_glb_static
            V, F, _ = load_glb_static(subject_glb)
            native_h = float(V[:, 1].max() - V[:, 1].min()) or 1.0
            s = h / native_h
            V[:, 1] -= float(V[:, 1].min())                  # feet at y=0
            V[:, [0, 2]] -= (V[:, [0, 2]].min(0) + V[:, [0, 2]].max(0)) / 2.0  # center xz
            V = V * s
            if _GLB_YAW_OFFSET:                              # model fwd fix
                ca, sa = math.cos(math.radians(_GLB_YAW_OFFSET)), math.sin(math.radians(_GLB_YAW_OFFSET))
                xr = V[:, 0] * ca + V[:, 2] * sa
                zr = -V[:, 0] * sa + V[:, 2] * ca
                V = np.stack([xr, V[:, 1], zr], axis=1)
            fc = np.asarray(F, dtype=np.int64)
            w.raw(f'''    def Xform "Subject"
    {{
{_time_samples("float3 xformOp:translate", feet_samples)}
{_time_samples("quatf xformOp:orient", yaw_samples)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        def Mesh "Human"
        {{
            uniform bool doubleSided = 1
            int[] faceVertexCounts = [{", ".join(["3"] * len(F))}]
            int[] faceVertexIndices = [{", ".join(map(str, fc.ravel()))}]
            point3f[] points = [
{_vec3_array(np.asarray(V))}
            ]
            color3f[] primvars:displayColor = [(0.52000, 0.56000, 0.64000)]
            custom string robocine:label = "subject (static human mesh, source {Path(subject_glb).name})"
        }}
    }}''')
        else:
            head_blk = (f'''
        def Sphere "Head"
        {{
            float radius = {_f(_HEAD_R * h)}
{_time_samples("float3 xformOp:translate", head_samples, "            ")}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            color3f[] primvars:displayColor = [{_c3(_SUBJECT_COLOR)}]
        }}''' if _HEAD_R > 0 else "")
            nose_blk = (f'''
        def Sphere "Nose"
        {{
            float radius = {_f(_NOSE_R * h)}
{_time_samples("float3 xformOp:translate", nose_samples, "            ")}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            color3f[] primvars:displayColor = [{_c3((0.95, 0.25, 0.15))}]
        }}''' if _NOSE_R > 0 else "")
            w.raw(f'''    def Xform "Subject"
    {{
        def Capsule "Body"
        {{
            float height = {_f(0.8 * h)}
            float radius = {_f(0.30 * h / 1.75)}
            uniform token axis = "Y"
{_time_samples("float3 xformOp:translate", subj_samples, "            ")}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            color3f[] primvars:displayColor = [{_c3(_SUBJECT_COLOR)}]
        }}{head_blk}{nose_blk}
    }}''')
    if cast_chars:
        blocks = []
        for c in cast_chars:
            h = c["yaw"]
            q = (math.cos(h / 2), 0.0, math.sin(h / 2), 0.0)
        for c in cast_chars:
            h = c["yaw"]
            q = (math.cos(h / 2), 0.0, math.sin(h / 2), 0.0)
            if c.get("face_colors") is not None:
                color_blk = (f"color3f[] primvars:displayColor = [\n"
                             f"{_vec3_array(np.asarray(c['face_colors']), per_line=4)}\n"
                             f"                ] (\n"
                             f"                    interpolation = \"uniform\"\n"
                             f"                )")
            else:
                color_blk = f"color3f[] primvars:displayColor = [{_c3(c['color'])}]"
            blocks.append(f'''        def Xform "{c['id']}"
        {{
            float3 xformOp:translate = {_t3(c['seat'])}
            quatf xformOp:orient = ({_f(q[0])}, {_f(q[1])}, {_f(q[2])}, {_f(q[3])})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
            def Mesh "Figure"
            {{
                uniform bool doubleSided = 1
                int[] faceVertexCounts = [{", ".join(["3"] * len(c['F']))}]
                int[] faceVertexIndices = [{", ".join(map(str, c['F'].ravel()))}]
                point3f[] points = [
{_vec3_array(np.asarray(c['V']), per_line=3)}
                ]
                {color_blk}
                custom string robocine:label = "cast ({c['note']})"
            }}
        }}''')
        w.raw('''    def Scope "Cast"
    {
''' + "\n".join(blocks) + "\n    }")
    # --- static props (pure references, no baking): mugs & friends ---------
    # cast json may carry "props": [{id, usd, at:[x,y,z], heading_deg?, up?,
    # scale?}] — `up="Z"` wraps Z-up pack assets with rotateX -90.
    props = []
    if cast:
        specs0 = json.loads(Path(cast).read_text(encoding="utf-8"))
        for pspec in specs0.get("props", []):
            ppath = Path(pspec["usd"]).expanduser()
            if not ppath.is_absolute():
                ppath = _REPO_ROOT / pspec["usd"]
            yaw = math.degrees(math.radians(float(pspec.get("heading_deg", 0.0))))
            rx = -90.0 if pspec.get("up") == "Z" else 0.0
            rot = (rx, yaw, 0.0)
            scl = float(pspec.get("scale", 1.0))
            ops = ['"xformOp:translate"', '"xformOp:rotateXYZ"']
            rows = [f"            float3 xformOp:translate = {_t3(pspec['at'])}",
                    f"            float3 xformOp:rotateXYZ = ({_f(rot[0])}, "
                    f"{_f(rot[1])}, {_f(rot[2])})"]
            if scl != 1.0:
                rows.append(f"            float3 xformOp:scale = ({_f(scl)}, "
                            f"{_f(scl)}, {_f(scl)})")
                ops.append('"xformOp:scale"')
            props.append("\n".join([
                f'        def Xform "{_sanitize(pspec.get("id", "prop"))}"',
                '        {',
                *rows,
                '            uniform token[] xformOpOrder = ['
                + ", ".join(ops) + ']',
                '            def Xform "Asset" (',
                # relative to the stage — see the Environment reference note
                f'                prepend references = @{os.path.relpath(ppath.resolve(), out_path.parent.resolve())}@',
                '            )',
                '            {',
                '            }',
                f'            custom string robocine:label = '
                f'"prop ({Path(pspec["usd"]).name})"',
                '        }']))
    if props:
        w.raw('''    def Scope "Props"
    {
''' + "\n".join(props) + "\n    }")
    # FOV: emit a STATIC focalLength when the trajectory's FOV never changes —
    # a fixed-lens camera should not carry timeSamples (recurring review
    # complaint); zoom (Hitchcock) shots keep per-code samples.
    fov_static = bool(np.allclose(FV, FV[0], atol=1e-4))
    focal_blk = (f"        float focalLength = {_f(_focal_from_vfov(FV[0]))}"
                 if fov_static else
                 _time_samples("float focalLength", focal_samples))
    w.raw(f'''    def Camera "ShotCam"
    {{
        float2 clippingRange = (0.05, 500)
        float horizontalAperture = 36
        float verticalAperture = 20.25
        token projection = "perspective"
{focal_blk}
{_time_samples("float3 xformOp:translate", trans_samples)}
{_time_samples("quatf xformOp:orient", orient_samples)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}''')
    # CamBody: small visible proxy riding the same pose as ShotCam — the camera
    # prim itself is invisible, so without this nothing marks the camera in
    # third-person views. Hull sits ABOVE and BEHIND the lens (pupil at local
    # origin looking down -Z) so it can never occlude the POV render.
    hull_pts, hull_nrms, hull_fvc, hull_fvi = _local_box_arrays(
        center=(0.0, 0.06, 0.10), extents=(0.20, 0.09, 0.24))
    w.raw(f'''    def Xform "CamBody"
    {{
{_time_samples("float3 xformOp:translate", trans_samples)}
{_time_samples("quatf xformOp:orient", orient_samples)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        def Mesh "Hull"
        {{
            uniform bool doubleSided = 1
            int[] faceVertexCounts = [{", ".join(map(str, hull_fvc))}]
            int[] faceVertexIndices = [{", ".join(map(str, hull_fvi))}]
            point3f[] points = [{", ".join(_t3(p) for p in hull_pts)}]
            normal3f[] normals = [{", ".join(_t3(n) for n in hull_nrms)}] (
                interpolation = "faceVarying"
            )
            color3f[] primvars:displayColor = [(0.16000, 0.17000, 0.21000)]
            custom string robocine:label = "camera body proxy"
        }}
        def BasisCurves "Frustum"
        {{
            uniform token type = "linear"
            int[] curveVertexCounts = [{", ".join(["2"] * 8)}]
            point3f[] points = [{", ".join(_t3(p) for p in _frustum_pts_local())}]
            float[] widths = [0.006]
            color3f[] primvars:displayColor = [(0.35000, 0.65000, 1.00000)]
            custom string robocine:label = "live camera FOV (moves with trajectory)"
        }}
    }}''')
    # CamLight: small headlight riding the camera so POV subjects are lit
    # even in floor-lit-only interiors (ceiling rig lights vertical cam-facing
    # surfaces poorly; without this the subject silhouette reads black).
    caml_samples = {}
    for k, code in enumerate(codes):
        p = P[k] + np.array([0.0, 0.15, 0.0])
        caml_samples[int(code)] = _t3(p)
    w.raw(f'''    def SphereLight "CamLight"
    {{
        float inputs:intensity = 3500
        float inputs:radius = 0.18
        color3f inputs:color = (1.00000, 0.98000, 0.95000)
{_time_samples("float3 xformOp:translate", caml_samples)}
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}''')
    _emit_static_camera(w, "OverviewCam", ov_eye, acenter, 55.0, "third-person overview (action-framed)")

    # -- lights ------------------------------------------------------------- #
    w.raw('''    def DistantLight "Sun"
    {
        float inputs:intensity = 1500
        color3f inputs:color = (1.00000, 0.96000, 0.90000)
        float inputs:angle = 3
        float3 xformOp:rotateXYZ = (-42.00000, 32.00000, 0.00000)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
    def DomeLight "Sky"
    {
        color3f inputs:color = (0.35000, 0.42000, 0.55000)
        float inputs:intensity = 1
    }''')

    # -- annotations -------------------------------------------------------- #
    seg_colors = [_SEGMENT_PALETTE[i % len(_SEGMENT_PALETTE)]
                  for i in seg_idx]
    w.raw(f'''    def Scope "Annotations"
    {{
        def BasisCurves "Trajectory"
        {{
            uniform token type = "linear"
            int[] curveVertexCounts = [{len(frames)}]
            point3f[] points = [
{_vec3_array(pos)}
            ]
            float[] widths = [0.015]
            color3f[] primvars:displayColor = [
{_vec3_array(np.asarray(seg_colors))}
            ] (
                interpolation = "vertex"
            )
        }}''')
    # path beads (camera-position spheres every ~0.5 s) — DISABLED for now:
    # per user request the dots clutter the view; the trajectory curve carries
    # the visual information. Set False → skip emission entirely.
    PATH_BEADS = False
    bead_dt = 0.5
    bead_times = np.arange(0.0, max(t1 - t0, 1e-6) + 1e-9, bead_dt)
    if PATH_BEADS and len(bead_times):
        w.raw('        def Xform "PathBeads"\n        {')
        for i, tb in enumerate(bead_times):
            t = min(t0 + float(tb), t1)
            j = int(np.searchsorted(ts, t, side="right") - 1)
            j = max(0, min(j, len(frames) - 1))
            col = _SEGMENT_PALETTE[seg_idx[j] % len(_SEGMENT_PALETTE)]
            w.raw(f'''            def Sphere "bead_{i:04d}"
            {{
                float radius = 0.00022   # path bead (was 0.022; /100 per request)
                float3 xformOp:translate = {_t3(pos[j])}
                uniform token[] xformOpOrder = ["xformOp:translate"]
                color3f[] primvars:displayColor = [{_c3(col)}]
            }}''')
        w.raw("        }\n")

    # collision / occlusion markers (opt-in via --markers)
    for flag, mname, col in ((("colliding", "CollisionMarkers", (1.0, 0.12, 0.08)),
                              ("occluded", "OcclusionMarkers", (1.0, 0.60, 0.05)))
                             if markers else ()):
        hits = [f for f in frames if f.get(flag)]
        if hits:
            w.raw(f'        def Xform "{mname}"\n        {{')
            for i, f in enumerate(hits):
                w.raw(f'''            def Sphere "m_{i:04d}"
            {{
                float radius = 0.01
                float3 xformOp:translate = {_t3(f["pos"])}
                uniform token[] xformOpOrder = ["xformOp:translate"]
                color3f[] primvars:displayColor = [{_c3(col)}]
            }}''')
            w.raw("        }\n")
    w.raw("    }\n}")   # Annotations scope + World

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(w.text(), encoding="utf-8")
    meta = {"fps": fps, "duration_s": round(t1 - t0, 3), "t0": t0,
            "n_codes": n_codes, "n_frames": len(frames),
            "scene_id": traj.get("scene_id", ""), "traj": str(traj_path),
            "segments": seg_names, "cloud": cloud_note,
            "fov_static": fov_static,
            "cast": (str(Path(cast).resolve()) if cast else None),
            "cast_ids": [c["id"] for c in cast_chars],
            "overview_offset": list(overview_offset),
            "n_env_prims": n_env, "n_cloud_pts": 0 if pts is None else len(pts)}
    out_path.parent.joinpath("meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- #
# Kit-backed modes (gui / render)
# --------------------------------------------------------------------------- #
_CAM_ALIASES = {"shot": "ShotCam", "overview": "OverviewCam",
                "ShotCam": "ShotCam", "OverviewCam": "OverviewCam"}


def _resolve_input(input_path: Path, rebuild: bool, fps: int):
    """→ (usda_path, meta_dict). traj_final.json inputs are auto-built."""
    if input_path.suffix.lower() == ".usda":
        meta_p = input_path.parent / "meta.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
        else:
            # no sidecar: parse the usda header itself (authoritative anyway)
            head = input_path.read_text(encoding="utf-8", errors="ignore")[:2000]
            mc = re.search(r"endTimeCode = ([0-9.]+)", head)
            mp = re.search(r"timeCodesPerSecond = ([0-9.]+)", head)
            meta = {}
            if mc and mp:
                meta = {"fps": float(mp.group(1)),
                        "duration_s": round(float(mc.group(1)) / float(mp.group(1)), 3),
                        "n_codes": int(mc.group(1))}
                print(f"⚠️  no meta.json next to {input_path} — used usda header "
                      f"(duration {meta['duration_s']}s @ {meta['fps']}fps)")
            else:
                print(f"⚠️  no meta.json sidecar for {input_path}; "
                      f"fallback duration=5s — rebuild for exact timing")
        return input_path, meta
    traj = json.loads(input_path.read_text(encoding="utf-8"))
    sid = traj.get("scene_id", "")
    scene = _REPO_ROOT / "cinema_traj" / "scenes" / f"{sid}.json"
    out_dir = input_path.parent / "simview"
    usda = out_dir / "simview.usda"
    meta_p = out_dir / "meta.json"
    if rebuild or not usda.exists():
        if not scene.exists():
            raise FileNotFoundError(f"scene for '{sid}' not found: {scene}")
        print(f"🔨 building {usda} ...")
        old_meta = (json.loads(meta_p.read_text()) if meta_p.exists() else {})
        cast = old_meta.get("cast")
        if cast is None:                          # convention: cast.json beside traj
            cand = input_path.parent / "cast.json"
            cast = str(cand) if cand.exists() else None
        meta = build_usda(input_path, scene, usda, fps=fps, cast=cast,
                          overview_offset=tuple(old_meta.get("overview_offset",
                                                             (0.62, 0.55, 0.66))))
        print(f"✅ {usda}  ({meta['n_frames']} frames → {meta['n_codes']} keys @ {fps}fps, "
              f"cloud: {meta['cloud']})")
    else:
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return usda, meta


def cmd_gui(args) -> None:
    usda, _ = _resolve_input(Path(args.input), args.rebuild, args.fps)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": False})
    try:
        import omni.timeline
        import omni.usd
        ctx = omni.usd.get_context()
        ctx.open_stage(str(usda))
        for _ in range(10):
            app.update()
        tl = omni.timeline.get_timeline_interface()
        tl.set_auto_update(True)
        print("\n" + "=" * 68)
        print("RoboCine simview — interactive playback")
        print(f"  stage   : {usda}")
        print("  POV     : viewport camera dropdown → /World/ShotCam")
        print("  overview: /World/OverviewCam   |  timeline: bottom bar")
        print("  layers : Environment / Cloud / Subject / Cast / Annotations scopes")
        print("=" * 68 + "\n")
        while app.is_running():
            app.update()
    finally:
        app.close()


def _shooting_windows(traj):
    """[t0, t1) windows of shooting segments — arc_* transitions are
    'camera not shooting' repositioning and never enter the film."""
    wins = []
    for s in traj.get("segments", []):
        if str(s.get("name", "")).startswith("arc_") or s.get("type") == "arc":
            continue
        t0 = float(s.get("t_start", 0.0))
        wins.append((t0, t0 + float(s.get("duration_s", 0.0))))
    return wins


def cmd_render(args) -> None:
    usda, meta = _resolve_input(Path(args.input), args.rebuild, args.fps)
    duration = float(meta.get("duration_s", 0.0)) or 5.0
    fps = float(meta.get("fps", args.fps))
    if args.times:
        times = sorted({max(0.0, min(float(x), duration)) for x in args.times.split(",")})
    else:
        every = max(args.every, 1e-3)
        times = sorted({0.0} | {float(t) for t in np.arange(0.0, duration + 1e-9, every)}
                       | {duration})
    if getattr(args, "shooting_only", False):
        traj = json.loads(Path(meta["traj"]).read_text(encoding="utf-8"))
        wins = _shooting_windows(traj)
        if not wins:
            raise SystemExit("--shooting-only: no non-arc segments found in traj")
        # one sample per output frame -> the cut film plays 1:1 wall-clock
        step = 1.0 / max(int(args.video_fps), 1)
        sel = []
        for t0, t1 in wins:
            n = max(int(round((t1 - t0) / step)), 1)
            sel.extend(round(t0 + i * step, 4) for i in range(n))
        sel.append(round(wins[-1][1], 4))  # true end pose of the last shot
        times = sorted(set(sel))
        print(f"\u2702\ufe0f  shooting-only: {len(wins)} shot(s) "
              f"{[f'{t0:g}-{t1:g}s' for t0, t1 in wins]} -> {len(times)} frames "
              f"@ {args.video_fps}fps (real-time)")
    cam_names = [_CAM_ALIASES[s.strip()] for s in args.cameras.split(",")]

    _render_fast(args, usda, meta, times, cam_names)


def _assemble_videos(out_dir, cam_names, video_fps, stamp: str = "") -> None:
    video_dir = out_dir.parent / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    suf = f"_{stamp}" if stamp else ""
    for nm in cam_names:
        files = sorted(out_dir.glob(f"{nm}_*.png"))
        if not files:
            continue
        mp4 = video_dir / f"{nm}{suf}.mp4"
        ff = shutil.which("ffmpeg")
        if ff:
            r = subprocess.run(
                [ff, "-y", "-loglevel", "error", "-framerate", str(video_fps),
                 "-i", str(out_dir / f"{nm}_%04d.png"), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(mp4)], check=False)
            print(f"  {'✅' if r.returncode == 0 else '⚠️'} {mp4}")
        else:
            try:
                import cv2
                from PIL import Image as _Img
                w, h = _Img.open(files[0]).size
                vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                     float(video_fps), (w, h))
                for f in _tqdm(files, desc=f"mp4 {nm}", unit="f",
                               file=_prog_stream()):
                    vw.write(cv2.imread(str(f)))
                vw.release()
                print(f"  ✅ {mp4} (cv2 mp4v)")
            except Exception as e:
                print(f"  ⚠️ video assembly failed for {nm}: {e}")


def _write_frame(d, path) -> None:
    arr = np.asarray(d)[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(arr).save(path)


def _render_fast(args, usda, meta, times, cam_names) -> None:
    """Open the stage ONCE, strip time samples from the prims we pose per
    frame, drive their transforms by pxr.Set + update, capture.
    ~2.7× the old snapshot pipeline (no per-frame stage parse/upload).
    RTX/replicator do follow pxr default-value writes (verified
    with a moving-camera probe on this codebase)."""
    from isaacsim import SimulationApp
    t0 = float(meta.get("t0", 0.0))
    # trajectory source for driving
    traj = json.loads(Path(meta["traj"]).read_text(encoding="utf-8"))
    frames = traj["frames"]
    ts = np.array([f["t"] for f in frames], float)
    pos = np.array([f["pos"] for f in frames], float)
    look = np.array([f["look"] for f in frames], float)
    fov = np.array([f["fov_deg"] if f.get("fov_deg") is not None else _DEFAULT_FOV
                    for f in frames], float)
    scene_id = traj.get("scene_id", "")
    scene_path = _REPO_ROOT / "cinema_traj" / "scenes" / f"{scene_id}.json"
    track = None
    if scene_path.exists():
        track = json.loads(scene_path.read_text(encoding="utf-8")).get("subject_track")

    launch_cfg = {"headless": True}
    if args.renderer:
        launch_cfg["renderer"] = args.renderer
    import time as _time
    timers = {"open": 0.0, "update": 0.0, "capture": 0.0, "write": 0.0}
    app = SimulationApp(launch_cfg)
    try:  # mute Kit's runtime warning spam (startup ext spam precedes python)
        import carb
        carb.settings.get_settings().set("/log/level", "error")
    except Exception:
        pass
    out_dir = usda.parent / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*_????.png"):     # frames from previous runs —
        stale.unlink()                          # never mix old and new shots
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        ctx = omni.usd.get_context()
        s0 = _time.monotonic()
        ctx.open_stage(str(usda))
        for _ in range(30):
            app.update()
        stage = ctx.get_stage()

        driven: dict[str, dict] = {}
        WANTED = {
            "/World/ShotCam": ("xformOp:translate", "xformOp:orient", "focalLength"),
            "/World/CamBody": ("xformOp:translate", "xformOp:orient"),
            "/World/CamLight": ("xformOp:translate",),
            "/World/Subject": ("xformOp:translate", "xformOp:orient"),
            "/World/Subject/Body": ("xformOp:translate",),
        }
        for prim_path, attrs in WANTED.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not (prim and prim.IsValid()):
                continue
            got = {}
            for nm in attrs:
                a = prim.GetAttribute(nm)
                if a and a.IsValid():
                    if a.GetTimeSamples():
                        a.Clear()
                    got[nm] = a
            if got:
                driven[prim_path] = got
        timers["open"] += _time.monotonic() - s0
        print(f"🎥 fast render {len(times)}×{len(cam_names)} cam(s) → {out_dir}"
              f" (driven: {', '.join(sorted(driven))})")

        rps, anns = {}, {}
        for nm in cam_names:
            rps[nm] = rep.create.render_product(f"/World/{nm}",
                                                (args.width, args.height))
            anns[nm] = rep.AnnotatorRegistry.get_annotator("rgb")
            anns[nm].attach(rps[nm])
        for _ in range(max(args.settle, 20)):
            app.update()

        def pose_time(t_rel: float) -> None:
            t = t0 + t_rel
            eye = np.array([np.interp(t, ts, pos[:, i]) for i in range(3)])
            tgt = np.array([np.interp(t, ts, look[:, i]) for i in range(3)])
            R = _camera_basis(eye, tgt)
            if R is None:
                R = np.eye(3)
            qw, qx, qy, qz = _quat_from_R(R)
            eyeV = Gf.Vec3f(float(eye[0]), float(eye[1]), float(eye[2]))
            qV = Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz)))
            for pth in ("/World/ShotCam", "/World/CamBody"):
                if pth in driven:
                    driven[pth]["xformOp:translate"].Set(eyeV)
                    driven[pth]["xformOp:orient"].Set(qV)
            if "/World/ShotCam" in driven:
                driven["/World/ShotCam"]["focalLength"].Set(
                    _focal_from_vfov(float(np.interp(t, ts, fov))))
            if "/World/CamLight" in driven:
                driven["/World/CamLight"]["xformOp:translate"].Set(
                    Gf.Vec3f(float(eye[0]), float(eye[1] + 0.15), float(eye[2])))
            if track and "/World/Subject" in driven:
                h = float(track.get("height_m", 1.75))
                sp, hd = _subject_state(track, t)
                driven["/World/Subject"]["xformOp:translate"].Set(
                    Gf.Vec3f(float(sp[0]), float(sp[1]), float(sp[2])))
                ca, sa = math.cos(hd / 2), math.sin(hd / 2)
                driven["/World/Subject"]["xformOp:orient"].Set(
                    Gf.Quatf(float(ca), Gf.Vec3f(0.0, float(sa), 0.0)))
            elif track and "/World/Subject/Body" in driven:
                h = float(track.get("height_m", 1.75))
                sp, hd = _subject_state(track, t)
                driven["/World/Subject/Body"]["xformOp:translate"].Set(
                    Gf.Vec3f(float(sp[0]), float(sp[1] + 0.5 * h), float(sp[2])))

        def set_hidden(paths, hidden: bool) -> None:
            for pth in paths:
                prim = stage.GetPrimAtPath(pth.strip())
                if prim and prim.IsValid():
                    img = UsdGeom.Imageable(prim)
                    (img.MakeInvisible if hidden else img.MakeVisible)()
            for _ in range(6):                    # let Hydra re-evaluate
                app.update()

        # POV hygiene: annotation layers (trajectory curve etc.) belong in the
        # overview, not in the first-person footage. Hide `hide_in_shot` paths
        # for the ShotCam pass, restore them for the overview pass (two-phase
        # only when both cameras are requested in one session).
        hide_paths = [p for p in (args.hide_in_shot or "").split(",") if p.strip()]
        shot_like = [c for c in cam_names if c == "ShotCam"]
        others = [c for c in cam_names if c != "ShotCam"]
        if hide_paths and shot_like and others:
            phases = [(shot_like, True), (others, False)]
        else:
            phases = [(cam_names, bool(hide_paths and shot_like))]

        for phase_cams, do_hide in phases:
            if do_hide:
                set_hidden(hide_paths, True)
            first = True
            bar = _tqdm(list(enumerate(times)),
                        desc="frames" + (" (shot, clean)" if do_hide else ""),
                        unit="f", dynamic_ncols=True, file=_prog_stream())
            for i, t in bar:
                pose_time(t)
                s1 = _time.monotonic()
                for _ in range(max(args.settle, 40) if first else args.settle):
                    app.update()
                timers["update"] += _time.monotonic() - s1
                s2 = _time.monotonic()
                rep.orchestrator.step(rt_subframes=args.subframes)
                timers["capture"] += _time.monotonic() - s2
                first = False
                s2 = _time.monotonic()
                for nm in phase_cams:
                    d = anns[nm].get_data()
                    if d is None or getattr(d, "size", 0) == 0:
                        print(f"  ⚠️ t={t}: empty frame from {nm}")
                        continue
                    _write_frame(d, out_dir / f"{nm}_{i:04d}.png")
                timers["write"] += _time.monotonic() - s2
            if do_hide:
                set_hidden(hide_paths, False)
        print(f"⏱️  open {timers['open']:.1f}s | updates {timers['update']:.1f}s | "
              f"capture {timers['capture']:.1f}s | write {timers['write']:.1f}s")
        # user-reported lingering hang AFTER all frames were captured: the
        # close can wedge. We don't debug it — outputs are already on disk;
        # verify counts, assemble, hard-exit if close stalls past grace.
        n_frames = len(list(out_dir.glob('*_????.png')))
        expect = len(times) * len(cam_names)
        print(f"[render] frames on disk: {n_frames} (expected {expect})")
        if args.mp4:
            _assemble_videos(out_dir, cam_names, args.video_fps,
                             stamp=_time.strftime("%Y%m%d_%H%M%S"))
    finally:
        _close_app_guarded(app)


def _close_app_guarded(app, grace_s: float = 30.0) -> None:
    """app.close() occasionally hangs after a fully-finished render (shared
    GPU teardown contention).  Bound it: force-exit once grace expires — all
    artifacts are already written by then."""
    import threading as _th

    def _force_exit():
        print("⚠️ app.close() hung past grace — outputs are on disk; "
              "forcing exit.", flush=True)
        os._exit(0)

    t = _th.Timer(grace_s, _force_exit)
    t.daemon = True
    t.start()
    try:
        app.close()
    except Exception as e:
        print(f"⚠️ app.close() raised: {e!r}", flush=True)
    finally:
        t.cancel()


def cmd_build(args) -> None:
    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    sid = traj.get("scene_id", "")
    if args.scene:
        scene_path = Path(args.scene)
    else:
        scene_path = _REPO_ROOT / "cinema_traj" / "scenes" / f"{sid}.json"
    if not scene_path.exists():
        raise FileNotFoundError(f"scene for '{sid}' not found: {scene_path}")
    out = Path(args.out) if args.out else traj_path.parent / "simview" / "simview.usda"
    cast = args.cast
    if cast is None:                              # convention: cast.json beside traj
        cand = traj_path.parent / "cast.json"
        cast = str(cand) if cand.exists() else None
    ov_off = tuple(float(x) for x in args.overview_offset.split(","))
    meta = build_usda(traj_path, scene_path, out, fps=args.fps, points=args.points,
                      cloud_voxel=args.cloud_voxel, cloud_max=args.cloud_max_points,
                      cloud_margin=args.cloud_margin,
                      env_opacity=args.env_opacity, env_style=args.env_style,
                      markers=args.markers,
                      subject_glb=_resolve_subject_glb(args.subject_glb),
                      _subject_glb_arg_none=bool(args.subject_glb and
                                                 args.subject_glb.strip().lower()
                                                 in ("none", "off")),
                      cast=cast, overview_offset=ov_off)
    print(f"✅ {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"   frames {meta['n_frames']} → keys {meta['n_codes']} @ {args.fps}fps "
          f"| duration {meta['duration_s']}s | env prims {meta['n_env_prims']}"
          + (f" | cloud {meta['n_cloud_pts']} pts" if meta["n_cloud_pts"] else ""))
    print(f"   cloud: {meta['cloud']}")
    print("next: OMNI_KIT_ACCEPT_EULA=yes python tools/simview_isaac.py render "
          f"{traj_path} --every 1.0 --mp4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="author the USDA stage (pure python)")
    b.add_argument("--traj", required=True, help="traj_final.json")
    b.add_argument("--scene", help="scene JSON (default: cinema_traj/scenes/<scene_id>.json)")
    b.add_argument("--points", default="auto",
                   help="point cloud: auto (provenance) | none | path.ply/.pcd/.npz")
    b.add_argument("--fps", type=int, default=30, help="timeline fps (default 30)")
    b.add_argument("--cloud-voxel", type=float, default=0.03)
    b.add_argument("--cloud-max-points", type=int, default=200000)
    b.add_argument("--cloud-margin", type=float, default=1.5,
                   help="crop cloud to rooms+margin m (-1 = keep all)")
    b.add_argument("--env-style", default="solid",
                   choices=["solid", "wireframe", "off", "boxes"],
                   help="collision-field boxes: solid meshes | wireframe edges over the "
                        "cloud | not at all. For scenes wiring provenance.env_usd "
                        "(furnished USD env): reference WINS by default — use 'boxes' "
                        "to force the debug boxes there instead.")
    b.add_argument("--markers", action="store_true",
                   help="bake collision/occlusion marker beads along the path "
                        "(default off)")
    b.add_argument("--env-opacity", type=float, default=1.0,
                   help="(solid style only) env mesh opacity (1=solid; <1 = translucent "
                        "via PreviewSurface material)")
    b.add_argument("--subject-glb", default=None,
                   help="bake a GLB's bind pose as the animated subject mesh "
                        "(replaces the capsule). Default: "
                        "datasets/simview_assets/Xbot.glb if present; "
                        "'none' = no subject stand-in at all (the cast or "
                        "env carries the subject)")
    b.add_argument("--cast", default=None,
                   help="cast JSON: posed rigged-GLB characters baked as static "
                        "seated figures (see tools/glb_pose_bake.py). Default: "
                        "cast.json next to the traj file, if present")
    b.add_argument("--overview-offset", default="0.62,0.55,0.66",
                   help="OverviewCam eye = action centre + this vec × dist "
                        "(default '0.62,0.55,0.66'; e.g. '0.6,0.5,-0.8' shoots "
                        "from the −Z side)")
    b.add_argument("--out", help="output .usda (default: <traj_dir>/simview/simview.usda)")
    b.set_defaults(func=cmd_build)

    g = sub.add_parser("gui", help="interactive Isaac Sim playback (needs display)")
    g.add_argument("input", help=".usda or traj_final.json")
    g.add_argument("--rebuild", action="store_true", help="force USD rebuild")
    g.add_argument("--fps", type=int, default=30)
    g.set_defaults(func=cmd_gui)

    r = sub.add_parser("render", help="headless POV/overview PNGs (+ mp4)")
    r.add_argument("input", help=".usda or traj_final.json")
    r.add_argument("--rebuild", action="store_true")
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--cameras", default="shot,overview",
                   help="comma list: shot,overview")
    r.add_argument("--width", type=int, default=1280)
    r.add_argument("--height", type=int, default=720)
    r.add_argument("--every", type=float, default=2.0, help="sample every N s")
    r.add_argument("--shooting-only", action="store_true",
                   help="render ONLY the shooting segments — cut out the "
                        "non-shooting arc transitions (the video has one "
                        "movement per yml move line, hard cuts between "
                        "shots)")
    r.add_argument("--times", help="explicit times, e.g. '0,1.5,3'")
    r.add_argument("--settle", type=int, default=12,
                   help="render warm-up updates per frame")
    r.add_argument("--subframes", type=int, default=24,
                   help="orchestrator RT subframes per capture (convergence)")
    r.add_argument("--hide-in-shot",
                   default="/World/Annotations,/World/CamBody",
                   help="comma list of prim paths hidden while rendering the "
                        "ShotCam (POV) pass — overview keeps them. Default: the "
                        "Annotations scope (trajectory curve, marker beads) and "
                        "the camera body proxy ('' disables)")
    r.add_argument("--mp4", action="store_true")
    r.add_argument("--video-fps", type=int, default=10)
    r.add_argument("--renderer", default=None,
                   help="SimulationApp renderer override (e.g. RayTracedLighting)")
    r.set_defaults(func=cmd_render)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
