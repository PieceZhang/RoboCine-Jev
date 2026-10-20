#!/usr/bin/env python3
"""Audit saved raw PhysX rays and RTX masks without rerunning simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_agent_wbc.robot_sensor_rig import lidar_directions


def audit(folder):
    summary = json.loads((folder/'summary.json').read_text())
    manifest = json.loads((folder/'sensor_mounts.json').read_text())
    poses = [json.loads(s) for s in (folder/'postures.jsonl').read_text().splitlines()]
    scans = [json.loads(s) for s in (folder/'detection.jsonl').read_text().splitlines()]
    errors = []
    max_line_error = 0.
    max_environment_line_error = 0.
    max_self_line_error = 0.
    checked_points = 0
    checked_targets = 0
    if len(poses) != summary['posture_count']:
        errors.append('posture count does not match summary')
    for row in poses:
        if 'self_mask' not in row:
            errors.append('missing saved segmentation mask')
            continue
        mask = np.asarray(Image.open(folder/row['self_mask'])) != 0
        if float(mask.mean()) != row['camera_self_occlusion_fraction']:
            errors.append('saved self mask disagrees with fraction')
        digest = hashlib.sha256((folder/row['rgb']).read_bytes()).hexdigest()
        if digest != row['rgb_sha256']:
            errors.append('RGB image hash mismatch')
    rays = [lidar_directions(item) for item in manifest['lidars']]
    for row in scans:
        if 'raw_scan' not in row:
            errors.append('missing raw scan evidence')
            continue
        with np.load(folder/row['raw_scan'], allow_pickle=False) as data:
            points = data['points_world']
            sensor_ids, all_ray_indices = data['sensor'], data['ray_index']
            path_ids, paths = data['path_id'], data['paths']
            path_is_self = np.asarray([str(path).startswith(manifest.get('robot_root', '/World/Robot')+'/') for path in paths])
            if float(data['stamp_s']) != row['stamp_s']:
                errors.append('acquisition time mismatch')
            seen = set()
            valid_range = np.zeros(len(points), dtype=bool)
            for index, matrix in enumerate(data['T_world_lidars']):
                selected = sensor_ids == index
                ray_indices = all_ray_indices[selected]
                direction = rays[index][ray_indices] @ matrix[:3, :3].T
                relative = points[selected]-matrix[:3, 3]
                valid_range[selected] = np.linalg.norm(relative, axis=1) >= manifest['lidars'][index]['min_range_m']
                residual = np.linalg.norm(np.cross(relative, direction), axis=1)
                if len(residual):
                    max_line_error = max(max_line_error, float(residual.max()))
                    self_hits = path_is_self[path_ids[selected]]
                    if self_hits.any():
                        max_self_line_error = max(max_self_line_error, float(residual[self_hits].max()))
                    if (~self_hits).any():
                        max_environment_line_error = max(max_environment_line_error, float(residual[~self_hits].max()))
                    checked_points += len(residual)
                    if np.any((relative*direction).sum(axis=1) < -1e-5):
                        errors.append('hit is behind the scanner')
            for path_id, path in enumerate(paths):
                if str(path).startswith('/World/Probes/P'):
                    target = int(str(path).rsplit('P', 1)[1])
                    selected = (path_ids == path_id) & valid_range
                    if selected.any():
                        seen.add(target)
                        centre = np.asarray(row['targets_world'][target])
                        if np.any(np.abs(points[selected]-centre) > .0602):
                            errors.append('target hit is outside the authored 12 cm cube')
                        checked_targets += int(selected.sum())
            if sorted(seen) != row['detected']:
                errors.append('raw collision identities disagree with target detections')
            if sorted(set(range(36))-seen) != row['missing']:
                errors.append('missing-target list is inconsistent')
    if max_environment_line_error > 1e-4:
        errors.append('world point is inconsistent with acquisition-time sensor TF/ray')
    if sum(len(r['detected']) for r in scans) != summary['target_detections']:
        errors.append('total detections disagree with summary')
    bands = {}
    for row in scans:
        key = f"r={row['radius_m']}m,z={row['target_centre_z_m']}m"
        band = bands.setdefault(key, dict(detected=0, total=0, missing=0))
        band['detected'] += len(row['detected']); band['total'] += 36; band['missing'] += len(row['missing'])
    result = dict(run=str(folder), robot=summary['robot'], complete_posture_count=len(poses),
                  evidence_consistent=not errors, errors=sorted(set(errors)),
                  checked_raw_points=checked_points, checked_target_points=checked_targets,
                  max_point_to_acquisition_ray_error_m=max_line_error,
                  max_environment_point_to_ray_error_m=max_environment_line_error,
                  max_robot_self_hit_point_to_ray_error_m=max_self_line_error,
                  robot_self_hits_excluded_from_environment=True,
                  detection_bands=bands, camera_obstructed_pose_indices=[r['index'] for r in poses if r['camera_self_occlusion_fraction'] > 0.],
                  camera_housing_overlap_pose_indices=[r['index'] for r in poses if r['camera_housing_other_body_overlaps']],
                  lidar_origin_overlap_pose_indices=[r['index'] for r in poses if any(r.get('lidar_origin_robot_overlaps', {}).values())],
                  visibility_clear_pose_indices=[r['index'] for r in poses if r['camera_self_occlusion_fraction']==0 and not r['camera_housing_other_body_overlaps']],
                  summary=summary)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = [audit(run.resolve()) for run in a.runs]
    (a.output/'audit.json').write_text(json.dumps(results, indent=2)+'\n')
    lines = ['# Isaac Sim 传感器实测审计', '',
             '以下结论限于记录的有限姿态和 12 cm 障碍物；几何检测率不能替代闭环避障验证。', '',
             '| 平台 | 姿态数 | 障碍检测 | 相机无遮挡姿态 | 外壳干涉姿态 | 扫描原点重叠姿态 | 原始数据一致 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for r in results:
        s = r['summary']; total = s['target_observations']; detected = s['target_detections']
        rate = f'{detected}/{total} ({detected/total:.2%})' if total else '本轮未执行'
        lines.append(f"| {r['robot']} | {s['posture_count']} | {rate} | {s['posture_count']-s['camera_self_occluded_pose_count']}/{s['posture_count']} | {s['camera_housing_overlap_pose_count']} | {s['lidar_origin_overlap_pose_count']} | {r['evidence_consistent']} |")
    lines += ['', '所有测试均保留原机械臂关节上下限。固定基座、关闭重力、直接放置关节姿态的覆盖测试，与原控制器的真实运动扫描分别记录。', '',
              '完整连续关节空间、真实相机支架／线缆／载荷动力学、避障 Agent 闭环仍未获验证。',
              '若存在相机本体遮挡、安装代理干涉或漏检，不得将该安装报告为“全姿态无遮挡”或“足以保证避障”。', '',
              '逐姿态 RGB、分割掩码、逐束命中点与安装清单见 audit.json 中对应 run 目录。']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:8]+lines[8:8+len(results)]), flush=True)
    return 0 if all(r['evidence_consistent'] for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
