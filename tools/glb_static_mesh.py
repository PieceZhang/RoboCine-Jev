#!/usr/bin/env python3
"""glb_static_mesh.py — extract a skinned GLTF's BIND-POSE mesh as static
geometry for the simview stage (rigged USD hits a pathology in headless RTX;
a baked statue that rides the subject track renders exactly like our env
meshes, which is verified-reliable).

numpy-only, no external deps. Used by tools/simview_isaac.py via
load_glb_static(".../Xbot.glb").
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
         5125: np.uint32, 5126: np.float32}
_TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_glb_static(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (verts (N,3) world-of-node-tree, faces (M,3) int, groups (M,) mesh id)."""
    data = Path(path).read_bytes()
    _, _, _ = struct.unpack_from("<III", data, 0)
    jlen, jtype = struct.unpack_from("<II", data, 12)
    assert data[:4] == b"glTF" and jtype == 0x4E4F534A, "not a GLB/JSON-first file"
    g = json.loads(data[20:20 + jlen])
    off = 20 + jlen
    blen, btype = struct.unpack_from("<II", data, off)
    assert btype == 0x004E4942, "no BIN chunk"
    binbuf = data[off + 8: off + 8 + blen]

    def accessor(idx: int) -> np.ndarray:
        acc = g["accessors"][idx]
        bv = g["bufferViews"][acc["bufferView"]]
        n, comp = acc["count"], _COMP[acc["componentType"]]
        num = _TYPE_N[acc["type"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or num * np.dtype(comp).itemsize
        if stride == num * np.dtype(comp).itemsize:
            arr = np.frombuffer(binbuf, dtype=comp, count=n * num, offset=start)
        else:                                        # strided
            out = np.empty((n, num), dtype=comp)
            for i in range(n):
                out[i] = np.frombuffer(binbuf, dtype=comp, count=num,
                                       offset=start + i * stride)
            arr = out.ravel()
        return arr.reshape(n, num) if num > 1 else arr

    def local_trs(node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], float).reshape(4, 4).T
        T = np.eye(4)
        T[:3, 3] = node.get("translation", [0, 0, 0])
        R = np.eye(4)
        if "rotation" in node:                       # (x,y,z,w)
            x, y, z, wq = node["rotation"]
            R[:3, :3] = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * wq), 2 * (x * z + y * wq)],
                [2 * (x * y + z * wq), 1 - 2 * (x * x + z * z), 2 * (y * z - x * wq)],
                [2 * (x * z - y * wq), 2 * (y * z + x * wq), 1 - 2 * (x * x + y * y)]])
        S = np.diag(list(node.get("scale", [1, 1, 1])) + [1.0])
        return T @ R @ S

    verts_all, faces_all, groups = [], [], []
    scenes = g.get("scenes", [{"nodes": []}])
    roots = scenes[g.get("scene", 0)]["nodes"]

    def walk(ni: int, parent: np.ndarray):
        node = g["nodes"][ni]
        M = parent @ local_trs(node)
        if "mesh" in node:
            mesh = g["meshes"][node["mesh"]]
            for prim in mesh["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue                        # triangles only
                P = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
                P4 = np.c_[P, np.ones(len(P))] @ M.T
                P3 = P4[:, :3]
                if "indices" in prim:
                    F = accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
                else:
                    F = np.arange(len(P3), dtype=np.int64).reshape(-1, 3)
                base = sum(len(v) for v in verts_all)
                verts_all.append(P3)
                faces_all.append(F + base)
                groups.append(np.full(len(F), len(verts_all) - 1))
        for c in node.get("children", []):
            walk(c, M)

    for r in roots:
        walk(r, np.eye(4))
    V = np.vstack(verts_all)
    F = np.vstack(faces_all)
    G = np.concatenate(groups)
    return V, F, G


if __name__ == "__main__":
    for name in ("Xbot",):
        V, F, Gc = load_glb_static(f"/tmp/opencode/human/{name}.glb")
        print(f"{name}: verts={len(V)} faces={len(F)} prims={len(set(Gc))}")
        print(f"  bbox min={V.min(0).round(3)} max={V.max(0).round(3)}")
        print(f"  height(Y)={V[:,1].max()-V[:,1].min():.3f}  "
              f"foot_Y={V[:,1].min():.3f}  XZ-center={(V[:,[0,2]].min(0)+V[:,[0,2]].max(0))/2}")
