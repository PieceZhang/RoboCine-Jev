#!/usr/bin/env python3
"""Extract and cache a local KC-SVSDF collision mesh from a simview USD stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


FORMAT = "jvs_groundtruth_mesh/v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--spawn-x", type=float, default=5.0)
    result.add_argument("--spawn-z", type=float, default=1.0)
    result.add_argument("--spawn-yaw", type=float, default=0.0)
    result.add_argument("--spawn-height", type=float, default=0.002)
    result.add_argument("--map-radius", type=float, default=3.0)
    result.add_argument("--voxel-size", type=float, default=0.08)
    result.add_argument("--minimum-height", type=float, default=0.03)
    result.add_argument("--maximum-height", type=float, default=1.60)
    result.add_argument("--minimum-clearance", type=float, default=0.025)
    result.add_argument("--force", action="store_true")
    return result


def fingerprint(stage: Path, args: argparse.Namespace) -> dict:
    stat = stage.stat()
    return {
        "format": FORMAT,
        "exclude_shot_helpers": True,
        "stage": str(stage),
        "stage_size": stat.st_size,
        "stage_mtime_ns": stat.st_mtime_ns,
        "spawn": [args.spawn_x, args.spawn_z, args.spawn_yaw, args.spawn_height],
        "map_radius_m": args.map_radius,
        "voxel_size_m": args.voxel_size,
        "height_range_m": [args.minimum_height, args.maximum_height],
        "minimum_clearance_m": args.minimum_clearance,
    }


def metadata_from_cache(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data["metadata"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def source_to_odom(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    # simview source is Y-up. The Isaac runner applies Rx(+90 deg):
    # (x, y, z) -> (x, -z, y). Odom starts at the spawned base pose.
    isaac = points[:, [0, 2, 1]].copy()
    isaac[:, 1] *= -1.0
    isaac -= np.array([args.spawn_x, -args.spawn_z, args.spawn_height])
    yaw = args.spawn_yaw - math.pi * 0.5
    c, s = math.cos(yaw), math.sin(yaw)
    world_to_odom = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return isaac @ world_to_odom.T


def triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        if count >= 3:
            triangles.extend((face[0], face[k], face[k + 1]) for k in range(1, count - 1))
    return np.asarray(triangles, dtype=np.int32).reshape(-1, 3)


def triangle_intersects_roi(triangles: np.ndarray, radius: float, zlo: float, zhi: float) -> np.ndarray:
    low = triangles.min(axis=1)
    high = triangles.max(axis=1)
    nearest_xy = np.maximum(low[:, :2], np.minimum(0.0, high[:, :2]))
    planar = np.linalg.norm(nearest_xy, axis=1) <= radius
    vertical = (high[:, 2] >= zlo) & (low[:, 2] <= zhi)
    return planar & vertical


def surface_voxels(triangles: np.ndarray, size: float, zlo: float, zhi: float) -> np.ndarray:
    chunks = []
    for triangle in triangles:
        longest = max(np.linalg.norm(triangle[(k + 1) % 3] - triangle[k]) for k in range(3))
        subdivisions = max(1, min(64, int(math.ceil(longest / size))))
        if subdivisions == 1:
            chunks.append(np.vstack((triangle, triangle.mean(axis=0))))
            continue
        barycentric = []
        for i in range(subdivisions + 1):
            for j in range(subdivisions + 1 - i):
                a = i / subdivisions
                b = j / subdivisions
                barycentric.append((1.0 - a - b) * triangle[0] + a * triangle[1] + b * triangle[2])
        chunks.append(np.asarray(barycentric))
    points = np.vstack(chunks)
    points = points[(points[:, 2] >= zlo) & (points[:, 2] <= zhi)]
    return np.unique(np.floor(points / size).astype(np.int32), axis=0)


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment ground-truth mesh extracted from USD\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        output.write(header)
        output.write(np.asarray(vertices, dtype="<f4").tobytes())
        records = np.empty(len(faces), dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]))
        records["count"] = 3
        records["indices"] = faces
        output.write(records.tobytes())
    os.replace(temporary, path)


def extract(args: argparse.Namespace) -> int:
    stage_path = Path(args.stage).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    ply = output.with_suffix(".ply")
    if not stage_path.is_file():
        raise FileNotFoundError(f"USD stage does not exist: {stage_path}")
    if min(args.map_radius, args.voxel_size, args.minimum_clearance) <= 0:
        raise ValueError("map radius, voxel size and clearance must be positive")
    if not args.minimum_height < args.maximum_height:
        raise ValueError("invalid height range")
    expected = fingerprint(stage_path, args)
    cached = metadata_from_cache(output) if output.is_file() else None
    if not args.force and cached is not None and all(cached.get(key) == value for key, value in expected.items()) and ply.is_file():
        print(
            f"GROUNDTRUTH_MESH_CACHE_HIT file={output} triangles={cached['triangle_count']} "
            f"surfels={cached['surfel_count']}",
            flush=True,
        )
        return 0

    from isaacsim import SimulationApp

    application = SimulationApp({"headless": True})
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {stage_path}")
    up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    if up_axis != "Y":
        raise ValueError(f"simview ground-truth extraction expects Y-up USD, got {up_axis}")
    cache = UsdGeom.XformCache(Usd.TimeCode.EarliestTime())
    all_triangles = []
    mesh_count = 0
    skipped_floor = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.IsActive():
            continue
        path_lower = str(prim.GetPath()).lower()
        if '/cambody/' in path_lower or '/annotations/' in path_lower:
            continue
        if "/floor/" in path_lower or "/ceiling/" in path_lower:
            skipped_floor += 1
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.EarliestTime())
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.EarliestTime())
        indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.EarliestTime())
        if points is None or counts is None or indices is None:
            continue
        local = np.asarray(points, dtype=float)
        if not len(local):
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float)
        source_world = np.c_[local, np.ones(len(local))] @ matrix
        vertices = source_to_odom(source_world[:, :3], args)
        faces = triangulate(np.asarray(counts, dtype=int), np.asarray(indices, dtype=int))
        if not len(faces):
            continue
        triangles = vertices[faces]
        keep = triangle_intersects_roi(
            triangles, args.map_radius, args.minimum_height, args.maximum_height
        )
        if np.any(keep):
            all_triangles.append(triangles[keep])
            mesh_count += 1
    if not all_triangles:
        raise RuntimeError("USD produced no obstacle triangles inside the requested local workspace")
    triangles = np.concatenate(all_triangles)
    voxel_indices = surface_voxels(
        triangles, args.voxel_size, args.minimum_height, args.maximum_height
    )
    centers = (voxel_indices.astype(float) + 0.5) * args.voxel_size
    if len(centers) < 100:
        raise RuntimeError(f"USD mesh produced only {len(centers)} collision surfels")
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
    metadata = dict(
        expected,
        source_sha256=hashlib.sha256(stage_path.read_bytes()).hexdigest(),
        generated_unix_s=time.time(),
        source_mesh_count=mesh_count,
        skipped_floor_ceiling_meshes=skipped_floor,
        triangle_count=len(triangles),
        surfel_count=len(centers),
        surfel_radius_m=math.sqrt(3.0) * args.voxel_size * 0.5,
        ply_file=str(ply),
        coordinate_frame="spawn-relative odom, Z-up",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=faces,
            surfel_centers=np.asarray(centers, dtype=np.float32),
            surfel_radius=np.float64(metadata["surfel_radius_m"]),
            minimum_clearance=np.float64(args.minimum_clearance),
            metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))),
        )
    os.replace(temporary, output)
    write_binary_ply(ply, vertices, faces)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"GROUNDTRUTH_MESH_GENERATED file={output} meshes={mesh_count} "
        f"triangles={len(triangles)} surfels={len(centers)}",
        flush=True,
    )
    application.close()
    return 0


def main() -> int:
    args = parser().parse_args()
    try:
        return extract(args)
    except Exception as error:
        print(f"ground-truth mesh extraction failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
