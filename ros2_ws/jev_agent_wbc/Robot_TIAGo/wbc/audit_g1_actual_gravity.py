#!/usr/bin/env python3
"""Independent mass/COM and potential-difference audit of G1 gravity-only FF.

The reference parses URDF XML and evaluates rotations/kinematics directly. It
never calls the reviewed calibrate(), potential(), Pin FK or Pin Jacobians.
Only ActualGravity.torque() and its resulting mass container are under test.
This is static CPU validation, with no physical steps or convergence claim.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np


def vector(text, default=(0., 0., 0.)):
    return np.array([float(x) for x in text.split()]) if text else np.array(default)


def quaternion(q):
    x, y, z, w = np.asarray(q, float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rpy(v):
    r, p, y = v
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def axis_rotation(axis, angle):
    x, y, z = axis/np.linalg.norm(axis)
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)


class XmlReference:
    def __init__(self, urdf, dump):
        self.tree = ET.parse(urdf).getroot()
        self.links = {l.get('name'): l for l in self.tree.findall('link')}
        self.joints, self.parent = {}, {}
        for element in self.tree.findall('joint'):
            origin = element.find('origin')
            axis = element.find('axis')
            item = dict(name=element.get('name'), kind=element.get('type'),
                parent=element.find('parent').get('link'), child=element.find('child').get('link'),
                translation=vector(origin.get('xyz') if origin is not None else None),
                rotation=rpy(vector(origin.get('rpy') if origin is not None else None)),
                axis=vector(axis.get('xyz') if axis is not None else '1 0 0'))
            self.joints[item['name']] = item
            self.parent[item['child']] = item
        roots = set(self.links)-set(self.parent)
        assert roots == {'pelvis'}, roots
        self.order = []
        known = {'pelvis'}
        while len(known) < len(self.links):
            added = [j for j in self.joints.values() if j['parent'] in known and j['child'] not in known]
            assert added, 'disconnected URDF'
            self.order.extend(added)
            known.update(j['child'] for j in added)
        self.names = dump['body_names']
        self.masses = np.asarray(dump['masses']['values'], float)[0]
        actors = np.asarray(dump['actor_world_poses_xyzw']['values'], float)[0]
        centers = np.asarray(dump['coms']['values'], float)[0, :, :3]
        assert len(set(self.names)) == len(self.names) == len(self.masses) == 30
        assert np.isfinite(self.masses).all() and np.all(self.masses > 0)
        self.q0 = dict(zip(dump['joint_names'], dump['joint_positions']['values'][0]))
        self.base0 = np.eye(4)
        base_actor = actors[self.names.index('pelvis')]
        self.base0[:3, :3] = quaternion(base_actor[3:])
        self.base0[:3, 3] = base_actor[:3]
        frames = self.frames(self.q0, self.base0)
        self.centers_link = {}
        self.native_world_centers = {}
        for name, actor, center in zip(self.names, actors, centers):
            actual_world = actor[:3]+quaternion(actor[3:])@center
            rotation, position = frames[name]
            # Direct point conversion from independent XML link world pose.
            # No principal COM quaternion enters the position of its origin.
            self.centers_link[name] = rotation.T@(actual_world-position)
            self.native_world_centers[name] = actual_world

    def frames(self, q, base):
        frames = {'pelvis': (base[:3, :3], base[:3, 3])}
        for joint in self.order:
            Rp, pp = frames[joint['parent']]
            rotation = Rp@joint['rotation']
            position = pp+Rp@joint['translation']
            if joint['kind'] in ('revolute', 'continuous'):
                rotation = rotation@axis_rotation(joint['axis'], q[joint['name']])
            elif joint['kind'] == 'prismatic':
                position = position+rotation@joint['axis']*q[joint['name']]
            else:
                assert joint['kind'] == 'fixed'
            frames[joint['child']] = (rotation, position)
        return frames

    def potential(self, q, base):
        frames = self.frames(q, base)
        return float(sum(m*9.81*(frames[n][1]+frames[n][0]@self.centers_link[n])[2]
                         for n, m in zip(self.names, self.masses)))

    def fused_original(self):
        frames = self.frames(self.q0, self.base0)
        result = {n: dict(mass_kg=0., first_moment=np.zeros(3), source_links=[]) for n in self.names}
        for name, element in self.links.items():
            inertial = element.find('inertial')
            if inertial is None:
                continue
            mass = float(inertial.find('mass').get('value'))
            origin = inertial.find('origin')
            local = vector(origin.get('xyz') if origin is not None else None)
            rotation, position = frames[name]
            point = position+rotation@local
            ancestor = name
            while ancestor not in result:
                ancestor = self.parent[ancestor]['parent']
            group = result[ancestor]
            group['mass_kg'] += mass
            group['first_moment'] += mass*point
            group['source_links'].append(name)
        for name, group in result.items():
            group['center_world_m'] = (group.pop('first_moment')/group['mass_kg']).tolist()
            group['actual_mass_kg'] = float(self.masses[self.names.index(name)])
            group['actual_minus_original_mass_kg'] = group['actual_mass_kg']-group['mass_kg']
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--implementation', type=Path, required=True)
    parser.add_argument('--recorded', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    dump = json.loads(args.dump.read_text())
    reference = XmlReference(args.urdf, dump)
    metadata = json.loads(args.metadata.read_text())
    frozen_implementation = args.output/'gravity_model_under_review.py'
    shutil.copy2(args.implementation, frozen_implementation)
    shutil.copy2(args.metadata, args.output/'metadata_under_review.json')
    spec = importlib.util.spec_from_file_location('gravity_under_independent_review', frozen_implementation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ActualGravity(args.urdf, metadata)
    assert set(metadata['bodies']) == set(reference.names)
    body_checks = []
    for name, actual_mass in zip(reference.names, reference.masses):
        body = metadata['bodies'][name]
        offset = np.asarray(body['T_pin_body_from_actor'])
        center_link = offset[:3, 3]+offset[:3, :3]@np.asarray(body['center_actor_m'])
        fid = model.model.getFrameId(name)
        frame = model.model.frames[fid]
        joint_inertia = model.model.inertias[frame.parent]
        # Every actual body here is an original active-joint child or pelvis.
        # Confirm this instead of borrowing Pin placement as the reference.
        placement_error = float(np.max(np.abs(frame.placement.np-np.eye(4))))
        expected_parent = reference.parent[name]['name'] if name != 'pelvis' else 'universe'
        body_checks.append(dict(body=name, actual_mass_kg=float(actual_mass),
            metadata_mass_kg=body['mass_kg'], rebuilt_parent_mass_kg=float(joint_inertia.mass),
            independent_center_link_m=reference.centers_link[name].tolist(),
            metadata_center_link_error_m=float(np.linalg.norm(center_link-reference.centers_link[name])),
            rebuilt_parent_com_error_m=float(np.linalg.norm(joint_inertia.lever-reference.centers_link[name])),
            frame_placement_identity_max_error=placement_error,
            expected_parent=expected_parent, actual_parent=model.model.names[frame.parent],
            gravity_only_rotational_inertia_max_abs=float(np.max(np.abs(joint_inertia.inertia)))))
    upper = [n for n in dump['joint_names'] if n == 'waist_yaw_joint'
             or any(t in n for t in ('shoulder', 'elbow', 'wrist'))]
    assert len(upper) == 15
    cases = [('native_calibration', reference.q0.copy(), reference.base0.copy())]
    if args.recorded:
        rows = [json.loads(x) for x in args.recorded.open()]
        selected = np.linspace(0, len(rows)-1, 10, dtype=int)
        for index in selected:
            row = rows[index]
            cases.append((f'actual_recorded_{index}', dict(zip(row['joint_names'], row['joint_positions'])),
                          np.asarray(row['T_world_base'])))
    rng = np.random.default_rng(940)
    for index in range(8):
        q = reference.q0.copy()
        for name in upper:
            limit = reference.joints[name]
            element = next(x for x in reference.tree.findall('joint') if x.get('name') == name)
            lo, hi = (float(element.find('limit').get(k)) for k in ('lower', 'upper'))
            q[name] = float(np.clip(q[name]+rng.uniform(-.35, .35), lo+.02, hi-.02))
        base = reference.base0.copy()
        base[:3, :3] = rpy([rng.uniform(-.25, .25), rng.uniform(-.25, .25), rng.uniform(-3., 3.)])
        base[:3, 3] += rng.uniform(-.2, .2, 3)
        cases.append((f'static_synthetic_{index}', q, base))
    gravity_checks = []
    with (args.output/'gravity_trace.jsonl').open('x') as stream:
        for name, q, base in cases:
            actual = model.torque(q, base)
            by_epsilon = {}
            for eps in (1e-5, 1e-6):
                gradients = {}
                for joint in dump['joint_names']:
                    plus, minus = q.copy(), q.copy()
                    plus[joint] += eps; minus[joint] -= eps
                    gradients[joint] = (reference.potential(plus, base)-reference.potential(minus, base))/(2*eps)
                by_epsilon[str(eps)] = gradients
            errors = {joint: abs(actual[joint]-by_epsilon['1e-05'][joint]) for joint in dump['joint_names']}
            error2 = {joint: abs(actual[joint]-by_epsilon['1e-06'][joint]) for joint in dump['joint_names']}
            check = dict(configuration=name, compensation_torque_Nm=actual,
                independent_potential_gradient_Nm=by_epsilon,
                upper15_max_abs_error_Nm=max(errors[n] for n in upper),
                all29_max_abs_error_Nm=max(errors.values()),
                all29_max_abs_error_epsilon1e6_Nm=max(error2.values()),
                finite_difference_convergence_max_Nm=max(abs(by_epsilon['1e-05'][n]-by_epsilon['1e-06'][n]) for n in dump['joint_names']))
            gravity_checks.append(check)
            stream.write(json.dumps(check, allow_nan=False)+'\n')
    total = float(sum(i.mass for i in model.model.inertias))
    startup_energy = sum(m*9.81*reference.native_world_centers[n][2] for n, m in zip(reference.names, reference.masses))
    body_pass = all(abs(r['metadata_mass_kg']-r['actual_mass_kg']) < 1e-12
        and abs(r['rebuilt_parent_mass_kg']-r['actual_mass_kg']) < 1e-10
        and r['metadata_center_link_error_m'] < 1e-10 and r['rebuilt_parent_com_error_m'] < 1e-10
        and r['frame_placement_identity_max_error'] < 1e-10 and r['expected_parent'] == r['actual_parent']
        and r['gravity_only_rotational_inertia_max_abs'] < 1e-12 for r in body_checks)
    source_paths = [args.urdf, args.dump, args.metadata, args.implementation,
                    frozen_implementation, Path(__file__).resolve()]
    if args.recorded:
        source_paths.append(args.recorded)
    report = dict(schema='independent_g1_actual_gravity_audit/v1',
        scope=__doc__, physical_validation=False, dynamic_inertia_certification=False,
        native_raw_jacobian_used=False, independent_reference='manual XML URDF FK, explicit RPY/Rodrigues/quaternion matrices, native actor+COM point positions, independent sum(m*9.81*world_z) central differences',
        gravity_sign='Actuator compensation g(q)=+dU/dq Nm; physical gravitational generalized torque is -dU/dq. Prismatic terms would be N; all29 here are revolute radians.',
        native_body_count=len(reference.names), old_urdf_link_count=len(reference.links),
        actual_native_total_mass_kg=float(reference.masses.sum()), rebuilt_model_total_mass_kg=total,
        original_fused_masses=reference.fused_original(), body_checks=body_checks,
        no_original_fixed_inertias_counted_twice=body_pass,
        independent_startup_potential_J=float(startup_energy),
        reference_startup_potential_J=reference.potential(reference.q0, reference.base0),
        configurations=len(cases), all29_derivatives_checked=len(cases)*29,
        upper15_derivatives_checked=len(cases)*15,
        maximum_upper15_gravity_error_Nm=max(r['upper15_max_abs_error_Nm'] for r in gravity_checks),
        maximum_all29_gravity_error_Nm=max(r['all29_max_abs_error_Nm'] for r in gravity_checks),
        maximum_all29_gravity_error_epsilon1e6_Nm=max(r['all29_max_abs_error_epsilon1e6_Nm'] for r in gravity_checks),
        numerical_tolerance_Nm=2e-6,
        interpretation='Confirms mass/COM accounting and static generalized gravity of the calibrated URDF topology. Actor/COM calibration is at one native posture; this does not independently certify physical joint axes over all postures, inertial dynamics, feedforward tracking, contact response or stable parking.',
        evidence_files_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        static_cpu_wall_seconds=time.monotonic()-started)
    report['passed'] = bool(body_pass and abs(total-reference.masses.sum()) < 1e-10
        and max(r['all29_max_abs_error_epsilon1e6_Nm'] for r in gravity_checks) < 2e-6
        and max(r['all29_max_abs_error_Nm'] for r in gravity_checks) < 2e-6)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('body_checks', 'original_fused_masses', 'evidence_files_sha256')}, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
