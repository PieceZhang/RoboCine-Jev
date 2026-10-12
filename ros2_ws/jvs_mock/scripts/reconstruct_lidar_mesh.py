#!/usr/bin/env python3
"""Reconstruct a partial triangle mesh from measured LiDAR returns only.

Triangulate each scan in azimuth/elevation, reject long edges and angular gaps,
then weld vertices across calibrated sensors/viewpoints. No source USD, reference
mesh, Poisson hole filling, or unobserved surface completion is used. This is an
offline static-viewpoint reconstruction with simulator pose registration, not
SLAM, a watertight solid, or a collision-certified model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from analyze_lidar_map import measured_mask, transform


def scan_mesh(points, angular_step=.4, max_edge=.25, max_angle=3.):
    """Return measured vertices and locally supported triangles in sensor frame."""
    if not 0 < angular_step <= max_angle or max_edge <= 0:
        raise ValueError('require 0 < angular step <= max angle and positive max edge')
    ranges = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & (ranges > 0)
    points, ranges = points[valid], ranges[valid]
    angles = np.degrees(np.column_stack((np.arctan2(points[:, 1], points[:, 0]),
        np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)))))
    # Keep nearest measured return per angular cell (never invent a mean point).
    order = np.argsort(ranges, kind='stable')
    cells = np.floor(angles[order] / angular_step).astype(np.int32)
    _, indices = np.unique(cells, axis=0, return_index=True)
    selected = order[indices]
    points, angles = points[selected], angles[selected]
    if len(points) < 3 or np.linalg.matrix_rank(angles - angles.mean(axis=0)) < 2:
        return points, np.empty((0, 3), dtype=np.int32)
    faces = Delaunay(angles).simplices
    triangles = points[faces]
    edges = triangles - np.roll(triangles, 1, axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    angle_edges = angles[faces] - np.roll(angles[faces], 1, axis=1)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(normal, axis=1)
    keep = ((lengths.max(axis=1) <= max_edge) & (area2 > 1e-8) &
            (np.linalg.norm(angle_edges, axis=2).max(axis=1) <= max_angle) &
            (area2 / np.maximum(lengths.max(axis=1)**2, 1e-12) > .04))
    faces, normal, triangles = faces[keep], normal[keep], triangles[keep]
    # Winding faces the scan origin; do not connect across the +/-180 seam.
    flip = np.einsum('ij,ij->i', normal, triangles.mean(axis=1)) > 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return points, faces


def weld_mesh(vertices, faces, size=.03, max_edge=.25):
    if size <= 0 or max_edge <= 0:
        raise ValueError('weld size and max edge must be positive')
    if not len(faces):
        raise ValueError('no supported triangles; cannot produce a scene mesh')
    _, first, inverse = np.unique(np.floor(vertices / size).astype(np.int64),
                                  axis=0, return_index=True, return_inverse=True)
    vertices, faces = vertices[first], inverse[faces]
    canonical = np.sort(faces, axis=1)
    keep = (canonical[:, 0] != canonical[:, 1]) & (canonical[:, 1] != canonical[:, 2])
    faces = faces[keep]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[unique]
    tri = vertices[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    lengths = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    faces = faces[(area2 > 1e-8) & (lengths.max(axis=1) <= max_edge)]
    if not len(faces):
        raise ValueError('no triangles remain after welding')
    used, remap = np.unique(faces, return_inverse=True)
    return vertices[used], remap.reshape(-1, 3)


def write_mesh(output, vertices, faces):
    with (output / 'scene_mesh.ply').open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\n'
            'comment partial measured dual LiDAR surface; holes are unknown\n'
            f'element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n'
            f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n').encode())
        stream.write(np.asarray(vertices, dtype='<f4').tobytes())
        packed = np.empty(len(faces), dtype=[('count', 'u1'), ('indices', '<i4', (3,))])
        packed['count'], packed['indices'] = 3, faces
        stream.write(packed.tobytes())
    with (output / 'scene_mesh.obj').open('w') as stream:
        stream.write('# Partial measured LiDAR surface; Isaac world metres, Z up\n')
        np.savetxt(stream, vertices, fmt='v %.7f %.7f %.7f')
        np.savetxt(stream, faces + 1, fmt='f %d %d %d')


def reconstruct(capture, output, angular_step=.4, max_edge=.25, max_angle=3., weld=.03):
    metadata = json.loads((capture / 'capture.json').read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh output directory; existing mesh will not be overwritten')
    vertices, faces, rows, used = [], [], [], set()
    count = 0
    for record in metadata['scans']:
        key = (record['pose_index'], record.get('sensor', 'rear'))
        if key in used:
            continue  # One full scan per sensor/viewpoint avoids repeated animated geometry.
        used.add(key)
        with np.load(capture / record['file']) as data:
            _, _, valid, _, _, _ = measured_mask(data, record)
            points, triangles = scan_mesh(data['points'][valid], angular_step, max_edge, max_angle)
            world = transform(points, data['world_sensor'])
        vertices.append(world)
        faces.append(triangles + count)
        count += len(world)
        rows.append(dict(file=record['file'], sensor=key[1], pose_index=key[0],
                         stamp=record.get('stamp'), triangles=len(triangles)))
    if not vertices:
        raise ValueError('capture has no scans')
    vertices, faces = weld_mesh(np.concatenate(vertices), np.concatenate(faces), weld, max_edge)
    output.mkdir(parents=True, exist_ok=True)
    write_mesh(output, vertices, faces)
    np.savez_compressed(output / 'mesh.npz', vertices=vertices, faces=faces)
    tri = vertices[faces]
    area = .5 * np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1).sum()
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    report = dict(capture=str(capture.resolve()), vertices=len(vertices), triangles=len(faces),
                  area_m2=float(area), boundary_edges=int((counts == 1).sum()),
                  nonmanifold_edges=int((counts > 2).sum()), used_scans=rows,
                  angular_step_deg=angular_step, max_edge_m=max_edge, max_angle_deg=max_angle,
                  weld_m=weld, reference_geometry_used=False,
                  registration=metadata['note'], limitations=__doc__)
    (output / 'mesh_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return vertices, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--angular-step', type=float, default=.4)
    parser.add_argument('--max-edge', type=float, default=.25)
    parser.add_argument('--max-angle', type=float, default=3.)
    parser.add_argument('--weld', type=float, default=.03)
    args = parser.parse_args()
    reconstruct(args.capture, args.output, args.angular_step, args.max_edge, args.max_angle, args.weld)


if __name__ == '__main__':
    main()
