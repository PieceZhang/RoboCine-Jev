#!/usr/bin/env python3
"""Compare measured ROS scans against sampled USD surfaces; export partial maps.

Coverage is recall of 8 cm reference surface voxels within 12 cm of a measured
return. It includes occluded/back surfaces, not just surfaces visible from the
tested viewpoints. PLY files contain observations only; missing space is unknown.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def voxelize(points, size=0.08):
    if not len(points):
        return points
    _, indices = np.unique(np.floor(points / size).astype(np.int32), axis=0, return_index=True)
    return points[indices]


def keys(points, size=0.08):
    return {tuple(row) for row in np.floor(points / size).astype(np.int32)}


def transform(points, matrix):
    return points @ matrix[:3, :3] + matrix[3, :3]


def measured_mask(data, record):
    """Range and per-link self filtering shared by coverage and reconstruction."""
    sensor = data['points']
    world = transform(sensor, data['world_sensor'])
    self_mask = np.zeros(len(world), dtype=bool)
    if 'self_boxes' in data:
        for box, frame in zip(data['self_boxes'], data['self_box_frames']):
            p = transform(world, np.linalg.inv(frame))
            self_mask |= ((p >= box[0] - .015) & (p <= box[1] + .015)).all(axis=1)
        method = 'per-link visual bounding boxes inflated by 15 mm'
    else:
        local = transform(world, np.linalg.inv(data['world_base']))
        self_mask = (((np.linalg.norm(local[:, :2], axis=1) < .23) & (local[:, 2] < .24)) |
                     ((np.linalg.norm(local[:, :2] - [.098, 0], axis=1) < .14) & (local[:, 2] < 1.15)))
        method = 'conservative default-pose base/arm envelope (diagnostic only)'
    ranges = np.linalg.norm(sensor, axis=1)
    in_range = (ranges >= record.get('min_range', .1)) & (ranges <= record.get('max_range', 30.))
    valid = np.isfinite(sensor).all(axis=1) & in_range & ~self_mask
    return world, ranges, valid, self_mask, method, in_range


def ply(path, points, reference=False):
    description = ('GROUND TRUTH unobserved reference; NOT measured returns' if reference else
                   'measured partial map; unobserved space is unknown')
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'comment {description}\n'
              f'element vertex {len(points)}\nproperty float x\nproperty float y\n'
              'property float z\nend_header\n')
    with path.open('wb') as output:
        output.write(header.encode())
        output.write(np.asarray(points, dtype='<f4').tobytes())


def analyze(capture_dir, reference, output, sensor_name=None):
    metadata = json.loads((capture_dir / 'capture.json').read_text())
    if sensor_name:
        metadata['scans'] = [r for r in metadata['scans'] if r.get('sensor', 'rear') == sensor_name]
        spec = metadata.get('sensor_specs', {}).get(sensor_name)
        if spec:
            metadata['sensor_paths'] = [spec['path']]
            metadata['config'] = spec['config']
    if not metadata['scans']:
        raise ValueError('capture has no scans; coverage cannot be evaluated')
    clean_world, rows, origins, by_pose, sets = [], [], [], {}, {}
    matrices = {}
    for record in metadata['scans']:
        data = np.load(capture_dir / record['file'])
        sensor = data['points']
        world, ranges, valid, self_mask, self_method, in_range = measured_mask(data, record)
        if not valid.any():
            raise ValueError(f'{record["file"]}: no valid environment returns')
        clean = world[valid]
        angles = np.arctan2(sensor[valid, 1], sensor[valid, 0])
        az = np.minimum(((angles + np.pi) / (2 * np.pi) * 72).astype(int), 71)
        elevation = np.degrees(np.arctan2(sensor[:, 2], np.linalg.norm(sensor[:, :2], axis=1)))
        nav = clean[(clean[:, 2] >= .035) & (clean[:, 2] <= 1.60)]
        near = nav[np.linalg.norm(nav[:, :2] - data['world_base'][3, :2], axis=1) < 1]
        current = keys(nav)
        sensor_name = record.get('sensor', 'rear')
        scan_key = (record['pose_index'], sensor_name)
        previous = sets.get(scan_key)
        jac = len(current & previous) / max(len(current | previous), 1) if previous else None
        sets[scan_key] = current
        rows.append(dict(**record, range_p01_p50_p99_m=np.quantile(ranges[valid], [.01, .5, .99]).tolist(),
                         retained_environment_candidates=int(valid.sum()),
                         outside_range=int((~in_range).sum()),
                         elevation_min_max_deg=[float(elevation.min()), float(elevation.max())],
                         azimuth_hit_bins_of_72=len(np.unique(az)), self_removed=int(self_mask.sum()),
                         self_filter=self_method, obstacle_voxels_inside_1m=len(keys(near)),
                         interscan_jaccard=jac))
        clean_world.append(voxelize(clean))
        origins.append(data['world_sensor'][3, :3])
        matrices.setdefault(sensor_name, []).append(data['base_link_sensor'])
        by_pose.setdefault(record['pose_index'], []).append(voxelize(clean))
    observed = voxelize(np.concatenate(clean_world))
    gt = reference['points']
    gt_voxels = voxelize(gt)
    tree = cKDTree(observed)
    distances = tree.query(gt_voxels, workers=2)[0]
    gt_tree = cKDTree(gt)
    residuals = gt_tree.query(observed, workers=2)[0]
    base = np.array([metadata['poses'][0][0], -metadata['poses'][0][1]])
    xy_distance = np.linalg.norm(gt_voxels[:, :2] - base, axis=1)
    height = gt_voxels[:, 2]
    masks = {
        'entire_scene_0_to_3m': (height >= 0) & (height <= 3),
        'entire_scene_navigation_0.035_to_1.60m': (height >= .035) & (height <= 1.60),
        'local_3m_navigation': (height >= .035) & (height <= 1.60) & (xy_distance <= 3),
        'local_1m_navigation': (height >= .035) & (height <= 1.60) & (xy_distance <= 1),
        'low_obstacles_0.035_to_0.30m_local_3m': (height >= .035) & (height <= .30) & (xy_distance <= 3),
        'arm_height_0.30_to_1.60m_local_3m': (height >= .30) & (height <= 1.60) & (xy_distance <= 3),
    }
    coverage = {}
    for name, mask in masks.items():
        values = distances[mask]
        coverage[name] = dict(reference_voxels=int(mask.sum()),
            observed_within_12cm=int((values <= .12).sum()),
            recall_at_8_12_20cm=[float((values <= t).mean()) for t in [.08, .12, .20]],
            median_distance_m=float(np.median(values)), p95_distance_m=float(np.quantile(values, .95)))
    per_pose = {}
    for i, clouds in by_pose.items():
        cloud = voxelize(np.concatenate(clouds))
        d = cKDTree(cloud).query(gt_voxels, workers=2)[0]
        per_pose[i] = {name: float((d[mask] <= .12).mean()) for name, mask in masks.items()}
    obstacle_height = (observed[:, 2] >= .035) & (observed[:, 2] <= 1.60)
    ref_xy = voxelize(gt_voxels[masks['entire_scene_navigation_0.035_to_1.60m'], :2])
    observed_xy = voxelize(observed[obstacle_height, :2])
    xy_distances = cKDTree(observed_xy).query(ref_xy, workers=2)[0]
    report = dict(capture=str(capture_dir), scans=rows, sensor_paths=metadata['sensor_paths'],
        domain_id=metadata['domain_id'], config=metadata['config'],
        sensor_specs=metadata.get('sensor_specs', {}),
        observed_8cm_voxels=len(observed), world_bounds=[observed.min(axis=0).tolist(), observed.max(axis=0).tolist()],
        lidar_base_extrinsic=next(iter(matrices.values()))[0].tolist(),
        extrinsic_max_abs_variation=max(float(np.max(np.abs(np.array(m) - m[0]))) for m in matrices.values()),
        per_sensor_extrinsics={name: dict(base_link_sensor=m[0].tolist(),
            max_abs_variation=float(np.max(np.abs(np.array(m) - m[0])))) for name, m in matrices.items()},
        surface_match_residual_p50_p95_p99_m=np.quantile(residuals, [.5, .95, .99]).tolist(),
        observed_within_12cm_of_reference_fraction=float((residuals <= .12).mean()),
        projected_obstacle_xy=dict(reference_cells=len(ref_xy), observed_cells=len(observed_xy),
            recall_within_12cm=float((xy_distances <= .12).mean()),
            note='occupied surface projection only; not a free-space/occupancy grid or SLAM map'),
        coverage=coverage, per_pose_recall=per_pose,
        coverage_definition=__doc__, registration=metadata['note'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    ply(output / 'observed_world.ply', observed)
    ply(output / 'unobserved_reference.ply',
        gt_voxels[(distances > .12) & masks['entire_scene_navigation_0.035_to_1.60m']], reference=True)
    np.savez_compressed(output / 'map.npz', observed=observed, reference=gt_voxels,
                        reference_distance=distances, origins=np.array(origins),
                        reference_origin_xy=base)
    plot(output, gt_voxels, distances, observed, np.array(origins), metadata, coverage)
    print(json.dumps({k: v for k, v in report.items() if k not in ['scans', 'coverage_definition']}, indent=2))


def plot(output, gt, distance, observed, origins, metadata, coverage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    nav = (gt[:, 2] >= .035) & (gt[:, 2] <= 1.60)
    for ax in axes:
        ax.scatter(gt[nav & (distance > .12), 0], gt[nav & (distance > .12), 1],
                   s=.5, c='#c1c5ca', label='Reference surfaces not observed')
        ax.scatter(gt[nav & (distance <= .12), 0], gt[nav & (distance <= .12), 1],
                   s=1, c='#168b88', label='Within 12 cm of a LiDAR return')
        ax.scatter(origins[:, 0], origins[:, 1], c='#cf452d', s=35, marker='x', label='LiDAR viewpoints')
        ax.set_aspect('equal')
        ax.set_xlabel('Isaac world X (m)')
        ax.set_ylabel('Isaac world Y = -simview Z (m)')
        ax.grid(alpha=.15)
    axes[0].set_title('Scene coverage, obstacle height 0.035–1.60 m')
    axes[0].legend(loc='upper right', fontsize=8)
    x, y = metadata['poses'][0][0], -metadata['poses'][0][1]
    axes[1].set_xlim(x - 3.2, x + 3.2)
    axes[1].set_ylim(y - 3.2, y + 3.2)
    axes[1].add_patch(plt.Circle((x, y), 1, fill=False, color='#cf452d', ls='--'))
    axes[1].add_patch(plt.Circle((x, y), 3, fill=False, color='#333333', ls=':'))
    axes[1].set_title('Spawn neighborhood (1 m / 3 m radii)')
    recall = coverage['entire_scene_navigation_0.035_to_1.60m']['recall_at_8_12_20cm'][1]
    fig.suptitle(f'Actual RTX ROS LiDAR — {len(metadata["poses"])} viewpoint(s); '
                 f'all-surface obstacle recall {recall:.1%}\n'
                 '8 cm surface voxels, 12 cm matching tolerance; unobserved space remains unknown')
    fig.savefig(output / 'coverage.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    mask = (observed[:, 2] >= -.1) & (observed[:, 2] <= 3.2)
    pts = observed[mask]
    axes[0].scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='viridis', s=.8, vmin=0, vmax=3)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    axes[1].scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], cmap='viridis', s=.8)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Height (m)')
    fig.suptitle('Observed point cloud only, simulator world registration')
    fig.savefig(output / 'observed.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sensor', choices=['rear', 'front'], help='single-sensor ablation of this same capture')
    args = parser.parse_args()
    analyze(Path(args.capture), np.load(args.reference), Path(args.output), args.sensor)


if __name__ == '__main__':
    main()
