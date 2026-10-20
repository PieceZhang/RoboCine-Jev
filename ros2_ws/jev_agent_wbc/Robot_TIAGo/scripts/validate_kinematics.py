#!/usr/bin/env python3
"""Check that PhysX frame preparation preserves official link kinematics."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x,y,z = axis
    cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)


def fk(path, q):
    doc = ET.parse(path)
    links = {x.attrib['name'] for x in doc.findall('link')}
    joints = doc.findall('joint')
    root = next(iter(links-{j.find('child').attrib['link'] for j in joints}))
    poses = {root: np.eye(4)}
    while joints:
        available = [j for j in joints if j.find('parent').attrib['link'] in poses]
        if not available:
            raise ValueError('invalid or disconnected URDF tree')
        for joint in available:
            origin = joint.find('origin')
            xyz = np.fromstring(origin.attrib.get('xyz','0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get('rpy','0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            matrix = np.eye(4)
            matrix[:3,3] = xyz
            matrix[:3,:3] = rotation([0,0,1],rpy[2])@rotation([0,1,0],rpy[1])@rotation([1,0,0],rpy[0])
            position = q.get(joint.attrib['name'], 0.)
            if joint.attrib['type'] != 'fixed':
                axis = np.fromstring(joint.find('axis').attrib.get('xyz','1 0 0'), sep=' ')
                motion = np.eye(4)
                if joint.attrib['type']=='prismatic':
                    motion[:3,3] = axis*position
                else:
                    motion[:3,:3] = rotation(axis, position)
                matrix = matrix@motion
            poses[joint.find('child').attrib['link']] = poses[joint.find('parent').attrib['link']]@matrix
            joints.remove(joint)
    base_inv = np.linalg.inv(poses['base_link'])
    return {name:base_inv@pose for name,pose in poses.items()}


def main():
    rng = np.random.default_rng(20261020)
    spec = json.loads((ROOT/'robot.json').read_text())
    configurations = [{j['name']:j['home'] for j in spec['joints']}]
    for joint in spec['joints']:
        for key in ('lower','upper'):
            q = configurations[0].copy()
            q[joint['name']] = joint[key]
            configurations.append(q)
    configurations += [{j['name']:rng.uniform(j['lower'],j['upper']) for j in spec['joints']} for _ in range(256)]
    worst = 0.
    for q in configurations:
        original, physical = fk(ROOT/'urdf/tiago.urdf',q), fk(ROOT/'urdf/tiago_physics.urdf',q)
        for name, pose in physical.items():
            worst = max(worst, float(np.max(np.abs(pose-original[name]))))
    report = dict(schema='tiago_kinematic_preservation/v1', configurations=len(configurations),
        common_links=len(physical), maximum_transform_entry_error=worst, passed=worst<1e-12,
        original_urdf_sha256=hashlib.sha256((ROOT/'urdf/tiago.urdf').read_bytes()).hexdigest(),
        physics_urdf_sha256=hashlib.sha256((ROOT/'urdf/tiago_physics.urdf').read_bytes()).hexdigest(),
        scope='CPU geometric preservation of massive chassis root and removal of empty fixed leaf frames; no physical-motion or sensor guarantee')
    (ROOT/'KINEMATIC_PRESERVATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()
