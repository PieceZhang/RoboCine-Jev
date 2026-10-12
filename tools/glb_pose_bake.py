#!/usr/bin/env python3
"""glb_pose_bake.py — bake a RIGGED GLTF (skinned, animated) into a static
triangle mesh at a chosen animation pose, for the simview stage.

Why: rigged USD hits a pathology in headless RTX (see glb_static_mesh.py),
so simview bakes statues instead of playing skeletons.  glb_static_mesh only
bakes the BIND pose (usually a T-pose); this module additionally samples a
glTF animation clip (e.g. KayKit's ``Sit_Chair_Pose`` or ``T-Pose``) at a fixed
time and bakes the SKINNED result.

numpy-only, no external deps.  Used by tools/simview_isaac.py via
load_glb_posed(); importable API mirrors glb_static_mesh.load_glb_static.

    load_glb_posed(path, anim="Sit_Chair_Pose", t=0.0) -> dict(
        verts (N,3) f8, faces (M,3) i8, groups (M,) i8,
        group_colors (K,3) f8 (material baseColorFactor, else white),
        joints {name: world_pos (3,)}  — posed skeleton landmarks,
        info {anim, t, names:[all anims], bind_bbox}
    )

All output is raw glTF model space (typically +Y up, meters-ish); scaling /
placement to world is the caller's job (simview normalizes via align rules).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
         5125: np.uint32, 5126: np.float32}
_TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_NORM_DENOM = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0,
               5125: 4294967295.0}


class _Glb:
    """Parsed GLB: json doc, binary chunk, accessor reader (handles strides,
    normalized integer encodings, matrix/vector types)."""

    def __init__(self, path: str | Path):
        data = Path(path).read_bytes()
        _, _, _ = struct.unpack_from("<III", data, 0)
        jlen, jtype = struct.unpack_from("<II", data, 12)
        assert data[:4] == b"glTF" and jtype == 0x4E4F534A, "not a GLB/JSON-first file"
        self.g = json.loads(data[20:20 + jlen])
        off = 20 + jlen
        blen, btype = struct.unpack_from("<II", data, off)
        assert btype == 0x004E4942, "no BIN chunk"
        self.bin = data[off + 8: off + 8 + blen]

    def accessor(self, idx: int, normalized: bool = False) -> np.ndarray:
        acc = self.g["accessors"][idx]
        bv = self.g["bufferViews"][acc["bufferView"]]
        n, comp = acc["count"], _COMP[acc["componentType"]]
        num = _TYPE_N[acc["type"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or num * np.dtype(comp).itemsize
        if stride == num * np.dtype(comp).itemsize:
            arr = np.frombuffer(self.bin, dtype=comp, count=n * num, offset=start)
        else:                                             # strided
            out = np.empty((n, num), dtype=comp)
            for i in range(n):
                out[i] = np.frombuffer(self.bin, dtype=comp, count=num,
                                       offset=start + i * stride)
            arr = out.ravel()
        arr = arr.reshape(n, num) if num > 1 else arr
        if normalized and acc.get("normalized"):          # int weights → [0,1]
            arr = np.maximum(arr.astype(np.float32) / _NORM_DENOM[acc["componentType"]],
                             -1.0 if comp in (np.int8, np.int16) else 0.0)
        return arr


def _quat_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _trs(translation, rotation, scale) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = translation
    R = np.eye(4)
    R[:3, :3] = _quat_mat(rotation) if rotation is not None else np.eye(3)
    S = np.diag(list(scale) + [1.0])
    return M @ R @ S


def _node_trs(node: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (t(3), q(4)|None, s(3)); expands a `matrix` node to TRS if needed."""
    if "matrix" in node:
        M = np.array(node["matrix"], float).reshape(4, 4).T
        t = M[:3, 3].copy()
        R = M[:3, :3]
        s = np.linalg.norm(R, axis=0)          # per-column scale
        R = R / np.maximum(s, 1e-12)
        # orthonormal polar part → quat (x,y,z,w)
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q = [(R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S,
                 (R[1, 0] - R[0, 1]) / S, 0.25 * S]
        else:
            i = int(np.argmax(np.diag(R)))
            j, k = (i + 1) % 3, (i + 2) % 3
            S = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12)) * 2
            q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / S]
            q[i] = 0.25 * S
            q[j] = (R[j, i] + R[i, j]) / S
            q[k] = (R[k, i] + R[i, k]) / S
        q = np.array(q, float)
        q /= np.linalg.norm(q)
        return t, np.array([q[0], q[1], q[2], q[3]]), s
    return (np.array(node.get("translation", [0, 0, 0]), float),
            (np.array(node["rotation"], float) if "rotation" in node else None),
            np.array(node.get("scale", [1, 1, 1]), float))


def _slerp(q0, q1, u):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        out = q0 + u * (q1 - q0)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def _sample_sampler(inp: np.ndarray, out: np.ndarray, t: float,
                    interp: str) -> np.ndarray:
    """Linear/step sampling of an animation sampler at time t (clamped)."""
    t = float(np.clip(t, inp[0], inp[-1]))
    if len(inp) == 1 or interp == "STEP":
        i = int(np.searchsorted(inp, t, "right")) - 1
        return out[max(i, 0)]
    i = int(np.searchsorted(inp, t, "right")) - 1
    i = min(max(i, 0), len(inp) - 2)
    u = (t - inp[i]) / max(inp[i + 1] - inp[i], 1e-9)
    a, b = out[i], out[i + 1]
    if a.shape == (4,):                                  # quaternion
        return _slerp(a.astype(np.float64), b.astype(np.float64), u)
    return a.astype(np.float64) + u * (b.astype(np.float64) - a.astype(np.float64))


def _pose_nodes(glb: _Glb, anim: str | None, t: float) -> list[np.ndarray]:
    """→ world (4×4) matrix per node, with animation channel overrides at t."""
    g = glb.g
    overrides: dict[int, dict[str, np.ndarray]] = {}
    names: list[str] = []
    clips = g.get("animations", [])
    names = [c.get("name", f"anim_{i}") for i, c in enumerate(clips)]
    if clips and anim is not None:
        low = {n.lower(): c for n, c in zip(names, clips)}
        clip = low.get(anim.lower())
        if clip is None:
            raise ValueError(f"animation '{anim}' not found; have {names}")
        for ch in clip["channels"]:
            tgt = ch["target"]
            node, path = tgt.get("node"), tgt.get("path")
            if node is None or path not in ("translation", "rotation", "scale"):
                continue
            smp = clip["samplers"][ch["sampler"]]
            inp = glb.accessor(smp["input"]).reshape(-1).astype(np.float64)
            out = glb.accessor(smp["output"])
            if smp.get("interpolation", "LINEAR") == "CUBICSPLINE":
                out = out.reshape(len(inp), 3, -1)[:, 1, :]     # middle = value
            val = _sample_sampler(inp, out, t,
                                  smp.get("interpolation", "LINEAR"))
            overrides.setdefault(node, {})[path] = val

    world: list[np.ndarray | None] = [None] * len(g["nodes"])

    def compose(ni: int, parent: np.ndarray) -> np.ndarray:
        node = g["nodes"][ni]
        t0, q0, s0 = _node_trs(node)
        ov = overrides.get(ni, {})
        if "translation" in ov:
            t0 = ov["translation"]
        if "rotation" in ov:
            q0 = ov["rotation"] / np.linalg.norm(ov["rotation"])
        if "scale" in ov:
            s0 = ov["scale"]
        M = parent @ _trs(t0, q0, s0)
        world[ni] = M
        for c in node.get("children", []):
            compose(c, M)
        return M

    scenes = g.get("scenes", [{"nodes": []}])
    for r in scenes[g.get("scene", 0)]["nodes"]:
        compose(r, np.eye(4))
    return world, names                               # type: ignore[return-value]


def _load_image(path: Path) -> np.ndarray:
    """RGB uint8 image via cv2 (no new deps; cv2 is already a repo dep)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_glb_posed(path: str | Path, anim: str | None = None,
                   t: float = 0.0, sample_textures: bool = False) -> dict:
    """Bake `path`'s skinned mesh at animation `anim` time `t` to static arrays.

    anim=None → bind pose (rest TRS).  Raises ValueError on unknown clip name.
    sample_textures → also bake per-face colours (M,3)∈[0,1], sampled from the
    material's baseColorTexture at each triangle's UV centroid (Kenney/KayKit
    palette textures render recognisably this way even without USD materials).
    See module docstring for the returned dict.
    """
    glb = _Glb(path)
    g = glb.g
    world, names = _pose_nodes(glb, anim, t)
    img_cache: dict[str, np.ndarray] = {}
    faces_per_prim: list[np.ndarray] = []        # for UV sampling pass

    def tex_image(prim: dict) -> np.ndarray | None:
        if not sample_textures or "material" not in prim:
            return None
        tex = (g["materials"][prim["material"]].get("pbrMetallicRoughness", {})
               .get("baseColorTexture"))
        if tex is None:
            return None
        img = g["images"][g["textures"][tex["index"]]["source"]]
        uri = img.get("uri")
        if not uri or uri.startswith("data:"):
            return None                          # embedded buffers unsupported
        real = (Path(path).parent / uri).resolve()
        if str(real) not in img_cache:
            img_cache[str(real)] = _load_image(real)
        return img_cache[str(real)]

    verts_all, faces_all, groups, gcolors = [], [], [], []
    joints_named: dict[str, np.ndarray] = {}
    for ni, node in enumerate(g["nodes"]):
        if "name" in node and world[ni] is not None:
            joints_named[node["name"]] = world[ni][:3, 3].copy()

    for ni, node in enumerate(g["nodes"]):
        if "mesh" not in node:
            continue
        M_node = world[ni]
        skin = node.get("skin")
        jmats = None
        if skin is not None:
            sk = g["skins"][skin]
            joints = sk["joints"]
            ibm = glb.accessor(sk["inverseBindMatrices"]).astype(np.float64)
            ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)     # column-major → row
            # skinning matrix per joint:  worldJoint @ invBind
            jmats = np.stack([world[j] @ ibm[k] for k, j in enumerate(joints)])

        for prim in g["meshes"][node["mesh"]]["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            A = prim["attributes"]
            P = glb.accessor(A["POSITION"]).astype(np.float64)
            if jmats is not None and "JOINTS_0" in A and "WEIGHTS_0" in A:
                J = glb.accessor(A["JOINTS_0"]).astype(np.int64)
                W = glb.accessor(A["WEIGHTS_0"], normalized=True).astype(np.float64)
                W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
                P4 = np.c_[P, np.ones(len(P))]
                P3 = np.zeros((len(P), 3))
                for k in range(J.shape[1]):
                    Mk = jmats[J[:, k]]                          # (N,4,4) gather
                    P3 += W[:, k, None] * np.einsum("nij,nj->ni", Mk[:, :3, :], P4)
            else:                                                # rigid (props)
                P3 = (np.c_[P, np.ones(len(P))] @ M_node.T)[:, :3]
            if "indices" in prim:
                F = glb.accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                F = np.arange(len(P3), dtype=np.int64).reshape(-1, 3)
            base = sum(len(v) for v in verts_all)
            verts_all.append(P3)
            faces_all.append(F + base)
            groups.append(np.full(len(F), len(verts_all) - 1))
            col = [1.0, 1.0, 1.0]
            if "material" in prim:
                mat = g["materials"][prim["material"]]
                col = (mat.get("pbrMetallicRoughness", {})
                       .get("baseColorFactor", [1, 1, 1, 1]))[:3]
            gcolors.append(col)
            # per-face texture colour (face centroid UV)
            img = tex_image(prim)
            if img is not None and "TEXCOORD_0" in A:
                uv = glb.accessor(A["TEXCOORD_0"]).astype(np.float64)
                cuv = uv[F].mean(1)                            # (M,2)
                h, w = img.shape[:2]
                pu = np.clip((cuv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
                pv = np.clip((1.0 - cuv[:, 1] % 1.0) * h, 0, h - 1).astype(int)
                faces_per_prim.append(img[pv, pu].astype(float) / 255.0)
            else:
                faces_per_prim.append(None)

    V = np.vstack(verts_all)
    F = np.vstack(faces_all)
    G = np.concatenate(groups)
    face_colors = None
    if any(x is not None for x in faces_per_prim):
        face_colors = np.vstack([pc if pc is not None
                                 else np.full((cnt, 3), col)
                                 for pc, cnt, col in
                                 zip(faces_per_prim,
                                     [len(f) for f in faces_all],
                                     gcolors)])
    return {
        "verts": V, "faces": F, "groups": G,
        "group_colors": np.array(gcolors, float),
        "face_colors": face_colors,
        "joints": joints_named,
        "info": {"anim": anim, "t": t, "names": names,
                 "bind_bbox": (V.min(0), V.max(0))},
    }


def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("glb")
    p.add_argument("--anim", default=None)
    p.add_argument("--t", type=float, default=0.0)
    p.add_argument("--list", action="store_true", help="list animations only")
    p.add_argument("--save", help="save posed arrays to .npz")
    a = p.parse_args()
    out = load_glb_posed(a.glb, a.anim, a.t)
    V = out["verts"]
    if a.list or a.anim is None:
        print("animations:", out["info"]["names"])
    low = {k.lower(): v for k, v in out["joints"].items()}
    hips = next((v for k, v in low.items() if "hips" in k or "pelvis" in k), None)
    print(f"verts={len(V)} faces={len(out['faces'])} prims={len(set(out['groups']))}")
    print(f"bbox min={V.min(0).round(3)} max={V.max(0).round(3)}  "
          f"height={V[:,1].max()-V[:,1].min():.3f}")
    print(f"foot_y={V[:,1].min():.3f}  xz_center="
          f"{((V[:,[0,2]].min(0)+V[:,[0,2]].max(0))/2).round(3)}")
    if hips is not None:
        print(f"hips_world={hips.round(4)}")
    if a.save:
        np.savez_compressed(a.save, verts=V, faces=out["faces"],
                            groups=out["groups"],
                            group_colors=out["group_colors"])
        print(f"saved → {a.save}")


if __name__ == "__main__":
    _cli()
