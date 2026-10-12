#!/usr/bin/env python3
"""Sample visible environment USD surfaces for an offline LiDAR coverage audit.

This is a reference for comparison, never an observed/free-space map. Hidden
back surfaces are retained, so all-surface recall is not visible-surface recall.
Uses USD Python and NumPy only; it does not start Kit, ROS or a GPU context.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom


def triangulate(counts, indices):
    counts = np.asarray(counts)
    indices = np.asarray(indices, dtype=np.int32)
    if np.all(counts == 3):
        return indices.reshape(-1, 3)
    offsets = np.r_[0, np.cumsum(counts)]
    return np.array([(indices[start], indices[start + k], indices[start + k + 1])
                     for start, count in zip(offsets, counts)
                     for k in range(1, count - 1)], dtype=np.int32).reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=1500000)
    args = parser.parse_args()
    source = Path(args.stage).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(source))
    root = stage.GetPrimAtPath('/World/Environment')
    if not root:
        raise ValueError('expected /World/Environment in simview stage')
    cache = UsdGeom.XformCache(Usd.TimeCode(0))
    chunks, identifiers, names = [], [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility(Usd.TimeCode(0)) == UsdGeom.Tokens.invisible:
            continue
        if 'groundplane' in str(prim.GetPath()).lower():
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(0), dtype=np.float64)
        faces = triangulate(mesh.GetFaceVertexCountsAttr().Get(0),
                            mesh.GetFaceVertexIndicesAttr().Get(0))
        transform = np.array(cache.GetLocalToWorldTransform(prim))
        points = points @ transform[:3, :3] + transform[3, :3]
        if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
            points = points[:, [0, 2, 1]] * [1, -1, 1]
        triangles = points[faces].astype(np.float32)
        valid = (triangles[:, :, 2].max(axis=1) >= -0.1) & (triangles[:, :, 2].min(axis=1) <= 3.2)
        triangles = triangles[valid]
        if len(triangles):
            identifiers.append(np.full(len(triangles), len(names), dtype=np.int32))
            names.append(str(prim.GetPath()))
            chunks.append(triangles)
    triangles = np.concatenate(chunks)
    ids = np.concatenate(identifiers)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    rng = np.random.default_rng(20260922)
    chosen = rng.choice(len(triangles), args.samples, p=area / area.sum(dtype=np.float64))
    selected = triangles[chosen]
    uv = rng.random((len(chosen), 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    samples = (selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0])
               + uv[:, 1:] * (selected[:, 2] - selected[:, 0])).astype(np.float32)
    np.savez_compressed(output / 'reference.npz', points=samples, mesh_ids=ids[chosen],
                        names=np.array(names), triangles=triangles)
    metadata = dict(stage=str(source), stage_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    reference_frame='Isaac world Z-up (x, -simview_z, simview_y)',
                    meshes=len(names), triangles=len(triangles), sampled_surfaces=len(samples),
                    area_m2=float(area.sum()), bounds=[triangles.min(axis=(0, 1)).tolist(),
                                                      triangles.max(axis=(0, 1)).tolist()],
                    excluded='invisible geometry, synthetic groundPlane; all roots outside /World/Environment, including Cast, Props, CamBody and Annotations',
                    layers=[l.realPath for l in stage.GetUsedLayers() if l.realPath])
    (output / 'reference.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != 'layers'}, indent=2))


if __name__ == '__main__':
    main()
