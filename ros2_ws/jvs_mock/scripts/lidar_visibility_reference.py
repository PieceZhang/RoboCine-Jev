#!/usr/bin/env python3
"""CPU ray-casting audit: distinguish occluded surfaces from visible misses.

Uses opaque environment triangles (no material transmission, robot, Cast,
Props or animated CamBody). This is an
ideal visibility diagnostic, not a replacement for measured RTX returns.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import warp as wp

from analyze_lidar_map import voxelize


@wp.kernel
def visible_query(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), origin: wp.vec3,
                  visible: wp.array(dtype=wp.int32)):
    i = wp.tid()
    delta = points[i] - origin
    distance = wp.length(delta)
    if distance > 0.1 and distance < 30.0:
        # The level profile has elevations in [-45, 45] degrees.
        planar = wp.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
        if wp.abs(delta[2]) <= planar:
            query = wp.mesh_query_ray(mesh, origin, delta / distance, distance + 0.02)
            if query.result and wp.abs(query.t - distance) <= 0.08:
                visible[i] = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(output.parent / 'warp_cache')
    wp.init()
    data = np.load(args.reference)
    mapping = np.load(args.map)
    gt = voxelize(data['points'])
    triangles = data['triangles']
    mesh = wp.Mesh(points=wp.array(triangles.reshape(-1, 3), dtype=wp.vec3, device='cpu'),
                   indices=wp.array(np.arange(triangles.size // 3, dtype=np.int32), device='cpu'))
    samples = wp.array(gt, dtype=wp.vec3, device='cpu')
    visible = wp.zeros(len(gt), dtype=wp.int32, device='cpu')
    # Suppress sub-millimetre chassis jitter when grouping the tested viewpoints.
    origins = voxelize(mapping['origins'], .01)
    for origin in origins:
        wp.launch(visible_query, dim=len(gt), inputs=[mesh.id, samples, wp.vec3(*origin), visible], device='cpu')
    seen = visible.numpy().astype(bool)
    match = cKDTree(mapping['observed']).query(gt, workers=2)[0] <= .12
    nav = (gt[:, 2] >= .035) & (gt[:, 2] <= 1.60)
    local = nav & (np.linalg.norm(gt[:, :2] - mapping['reference_origin_xy'], axis=1) <= 3)
    metrics = dict(definition=__doc__, origins=origins.tolist(), reference_voxels=len(gt),
                   ideal_visible_reference_voxels=int(seen.sum()))
    for name, roi in [('scene_navigation', nav), ('local_3m_navigation', local)]:
        select = roi & seen
        metrics[name] = dict(reference_voxels=int(roi.sum()), visible_voxels=int(select.sum()),
                             theoretical_visible_fraction=float(select.sum() / roi.sum()),
                             measured_recall_of_visible=float(match[select].mean()))
    output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
