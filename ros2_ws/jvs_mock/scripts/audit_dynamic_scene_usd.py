#!/usr/bin/env python3
"""Independent dense USD-surface reference and obstacle-route preflight.

Run with the Isaac Python USD libraries. This data never enters the controller.
The reference excludes the supporting floor and preview drawing helpers.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom

from jvs_mock.dynamic_scene import DynamicScene, model_axis_rotation
from jvs_mock.kc_svsdf import primitive_sdf


def clipped_triangle(triangle, low, high):
    polygon = list(triangle)
    for axis in range(3):
        for boundary, sign in ((low[axis], 1), (high[axis], -1)):
            result = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                da, db = sign * (a[axis] - boundary), sign * (b[axis] - boundary)
                if da >= 0:
                    result.append(a)
                if (da >= 0) != (db >= 0):
                    result.append(a + da / (da - db) * (b - a))
            polygon = result
            if not polygon:
                return []
    return [[polygon[0], polygon[k], polygon[k + 1]] for k in range(1, len(polygon) - 1)]


def reference(stage, low, high, spacing=.02):
    cache = UsdGeom.XformCache(0)
    chunks, paths = [], []
    root = stage.GetPrimAtPath('/World')
    if not stage.GetPrimAtPath('/World/Environment'):
        raise ValueError('expected /World/Environment in apartment simview USDA')
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or any('/' + name + '/' in path.lower() for name in ('floor', 'ceiling', 'cambody', 'annotations')):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility(0) == UsdGeom.Tokens.invisible:
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(0), float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = points @ matrix[:3, :3] + matrix[3, :3]
        if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
            points = points[:, [0, 2, 1]] * [1, -1, 1]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(0))
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(0))
        if np.all(counts == 3):
            faces = indices.reshape(-1, 3)
        else:
            offset = np.r_[0, np.cumsum(counts)]
            faces = np.array([(indices[s], indices[s+k], indices[s+k+1])
                              for s, count in zip(offset, counts) for k in range(1, count-1)]).reshape(-1, 3)
        triangles = points[faces]
        mask = np.all(triangles.max(axis=1) >= low, axis=1) & np.all(triangles.min(axis=1) <= high, axis=1)
        triangles = triangles[mask]
        if not len(triangles):
            continue
        interior = np.all((triangles >= low) & (triangles <= high), axis=(1, 2))
        pieces = [piece for triangle in triangles[~interior] for piece in clipped_triangle(triangle, low, high)]
        triangles = np.concatenate([triangles[interior], np.asarray(pieces).reshape(-1, 3, 3)])
        paths.append(path)
        longest = np.max(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2), axis=1)
        counts = np.maximum(1, np.ceil(longest / spacing).astype(int))
        # Every triangle edge in the barycentric grid is at most spacing.
        for n in np.unique(counts):
            uv = np.array([(i/n, j/n) for i in range(n+1) for j in range(n+1-i)])
            selected = triangles[counts == n]
            batch = max(1, 100000 // len(uv))
            for start in range(0, len(selected), batch):
                t = selected[start:start+batch]
                p = (t[:, None, 0] + uv[None, :, :1]*(t[:, None, 1]-t[:, None, 0])
                     + uv[None, :, 1:]*(t[:, None, 2]-t[:, None, 0]))
                chunks.append(np.round(p.reshape(-1, 3)/(spacing/4)).astype(np.int32))
    points = np.unique(np.concatenate(chunks), axis=0).astype(float)*(spacing/4)
    # The SDF is 1-Lipschitz. Subtract the mesh-grid spacing and rounding bound.
    return points, spacing + np.sqrt(3)*spacing/8, paths


def preflight(points, coverage, trajectory, scene_path):
    spawn = json.loads(trajectory.read_text())['initial_base_pose']
    scene = DynamicScene(scene_path)
    rotation = Rotation.from_euler('z', spawn['yaw'] - np.pi/2).as_matrix()
    origin = np.array([spawn['x'], -spawn['z'], spawn['height']])
    tree = cKDTree(points)
    reports = []
    for obstacle in scene.obstacles:
        model, name = obstacle['model'], obstacle['name']
        size = (np.asarray(model['size']) if model['type'] == 'box' else
                np.array([model['radius'], model['height']]) if model['type'] == 'cylinder' else
                np.array([model['radius']]))
        radius = (np.linalg.norm(size)/2 if model['type'] == 'box' else
                  np.hypot(size[0], size[1]/2) if model['type'] == 'cylinder' else size[0])
        best, when, speed = np.inf, 0., 0.
        for t in np.arange(0, scene.duration + .01, .02):
            position, velocity = scene.sample(name, t)
            center = origin + rotation @ scene.to_initial_base(position, spawn)
            speed = max(speed, np.linalg.norm(velocity))
            indices = tree.query_ball_point(center, radius + .5)
            if indices:
                gap = primitive_sdf((points[indices]-center) @ rotation @ model_axis_rotation(model), model['type'], size).min() - coverage
                if gap < best:
                    best, when = gap, t
        lower = best - speed*.01
        reports.append(dict(obstacle=name, minimum_static_clearance_lower_m=float(lower),
                            nearest_time_s=float(when), max_speed_m_s=float(speed),
                            pass_clearance_025=bool(lower >= .025)))
    return dict(trajectory=trajectory.name, obstacle_trajectory_file=scene_path.name,
                trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                obstacle_trajectory_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(), obstacles=reports,
                pass_clearance_025=all(r['pass_clearance_025'] for r in reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--reuse-reference', action='store_true', help='reuse only when every composed USD layer hash matches')
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.stage.resolve()))
    hashes = {str(Path(layer.realPath).resolve()): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
              for layer in stage.GetUsedLayers() if layer.realPath}
    low, high = np.array([2.3, -4.7, .03]), np.array([7.8, 1.8, 1.6])
    if args.reuse_reference:
        with np.load(args.output/'static_reference.npz') as cached:
            metadata = json.loads(str(cached['metadata']))
            if metadata['usd_layer_sha256'] != hashes or metadata.get('format') != 'jvs_dynamic_usd_reference/v2':
                raise ValueError('composed USD changed; regenerate the static reference')
            points, coverage = cached['points'], float(cached['coverage'])
    else:
        points, coverage, paths = reference(stage, low, high)
        metadata = dict(format='jvs_dynamic_usd_reference/v2', stage_view=str(args.stage.resolve()), usd_layer_sha256=hashes,
                        frame='Isaac world Z-up: (scene_x, -scene_z, scene_y)',
                        roi_low=low.tolist(), roi_high=high.tolist(), coverage_radius_m=coverage,
                        mesh_count=len(paths), points=len(points), excluded='supporting floor, ceiling, invisible geometry and shot preview helpers')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.reuse_reference:
        np.savez_compressed(args.output/'static_reference.npz', points=points.astype(np.float32),
                            coverage=np.float64(coverage), metadata=np.asarray(json.dumps(metadata)))
    reports = [preflight(points, coverage, args.package/'trajectories'/f'shot_combo{k}.json',
                         args.package/'scenes'/f'dynamic_{name}.json')
               for name, k in [('crossing', 1), ('overtake', 3), ('two_movers', 4)]]
    result = dict(**metadata, scenes=reports, pass_clearance_025=all(r['pass_clearance_025'] for r in reports))
    (args.output/'scene_preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(points=len(points), scenes=reports, pass_clearance_025=result['pass_clearance_025'])), flush=True)
    return 0 if result['pass_clearance_025'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
