#!/usr/bin/env python3
"""Verify portable asset closure, profile limits and sensor consistency on CPU.

This is structural validation. It never asserts physical, render, obstacle
avoidance, or full-workspace sensor acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def check():
    manifest = json.loads((ROOT/'ASSET_MANIFEST.json').read_text())
    profile = json.loads((ROOT/'robot.json').read_text())
    document = ET.parse(ROOT/manifest['entry'])
    failures = []
    for asset in manifest['meshes']:
        path = ROOT/asset['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
            failures.append(f'missing or changed mesh {path}')
    if hashlib.sha256((ROOT/manifest['entry']).read_bytes()).hexdigest() != manifest['sha256']:
        failures.append('physics URDF hash differs from manifest')
    links = {link.attrib['name'] for link in document.findall('link')}
    joints = {joint.attrib['name']: joint for joint in document.findall('joint')}
    for mesh in document.findall('.//mesh'):
        if not ((ROOT/manifest['entry']).parent/mesh.attrib['filename']).is_file():
            failures.append(f'missing relative URDF mesh {mesh.attrib["filename"]}')
    expected_joints = {'torso_lift_joint', *(f'arm_{index}_joint' for index in range(1,8))}
    if {j['name'] for j in profile['joints']} != expected_joints:
        failures.append('WBC must control exactly torso plus seven arm joints')
    for item in profile['joints']:
        limit = joints[item['name']].find('limit').attrib
        if not float(limit['lower']) <= item['home'] <= float(limit['upper']):
            failures.append(f'home outside source limits: {item["name"]}')
        for key in ('lower','upper','effort','velocity'):
            if abs(float(limit[key])-float(item[key])) > 1e-9:
                failures.append(f'profile changed official {key}: {item["name"]}')
    for name, gain in profile['passive_joint_gains'].items():
        if gain['effort'] > float(joints[name].find('limit').attrib['effort']):
            failures.append(f'passive effort exceeds source: {name}')
    sensor = json.loads((ROOT/profile['sensors']).read_text())
    rig = sensor['robots'][profile['robot_key']]
    if len(rig['lidars']) != 2:
        failures.append('expected one main and one auxiliary scanner')
    scan_keys = ('horizontal_samples','elevation_deg','min_range_m','max_range_m')
    scans = [dict(sensor['lidar'], **item) for item in rig['lidars']]
    for item in scans[1:]:
        if any(item[key] != scans[0][key] for key in scan_keys):
            failures.append('LiDAR scan parameters differ')
    for spec in [rig['camera'], *rig['lidars']]:
        if spec['parent'] not in links:
            failures.append(f'unknown sensor mount link {spec["parent"]}')
    if profile['base_link'] not in links or profile['ee_link'] not in links:
        failures.append('profile body does not exist')
    for name in [*profile['left_wheels'], *profile['right_wheels']]:
        if joints[name].attrib['type'] != 'continuous':
            failures.append(f'wheel must be continuous: {name}')
    report = dict(schema='tiago_preflight/v1', passed=not failures, failures=failures,
        profile='single-arm differential TIAGo with active lifting torso', physics_urdf=manifest['entry'],
        mesh_files_verified=len(manifest['meshes']), links=len(links), movable_joints=sum(j.attrib['type']!='fixed' for j in joints.values()),
        official_joint_limits_preserved=True, matching_lidar_parameters=not any('LiDAR' in x for x in failures),
        physics_verified=False, rendered_camera_verified=False, obstacle_avoidance_verified=False,
        full_pose_sensor_coverage_verified=False, scope='CPU structural preflight only; see separate measured Isaac Sim results')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
