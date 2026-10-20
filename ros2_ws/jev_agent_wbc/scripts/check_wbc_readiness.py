#!/usr/bin/env python3
"""Audit five WBC integrations without confusing historical motion with readiness.

Read-only by default. --output creates a new evidence file. --require-ready
returns 2 when any live integration gate remains unverified; no simulator runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from jev_agent_wbc.wbc_evidence import GATES, inspect_evidence
ROBOTS = frozenset(('Robot_G1', 'Robot_Go2', 'Robot_Ridgeback_UR5',
                    'Robot_Husky_UR5', 'Robot_TIAGo'))
CONTROL_GATES = ('backend_runtime_ready', 'continuous_camera_tracking_verified',
                 'representation_ingress_verified')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_basic_control(folder, document):
    """An explicit reduced-scope attestation cannot upgrade precision gates."""
    basic = document.get('basic_control', {})
    if not isinstance(basic, dict) or basic.get('qualified') is not True:
        return dict(qualified=False, audit_errors=['basic_control_not_qualified']), {}
    errors, sources, payloads = [], {}, {}
    for key, hash_key in (('entrypoint', 'entrypoint_sha256'),
                          ('audit_file', 'audit_sha256'),
                          ('validation_file', 'validation_sha256')):
        try:
            path = (folder/basic[key]).resolve()
            if (not path.is_relative_to(folder.parent) or not path.is_file()
                    or sha256(path) != basic[hash_key]):
                raise ValueError(key+'_missing_or_changed')
            sources[str(path)] = sha256(path)
            if key != 'entrypoint':
                payloads[key] = json.loads(path.read_text())
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append(str(error))
    report = payloads.get('audit_file', {})
    if (report.get('schema') != 'independent_wbc_basic_control_audit/v1'
            or report.get('basic_control_pass') is not True
            or report.get('environment_constraints_verified') is not False
            or report.get('sensors_avoidance_verified') is not False
            or report.get('precision_velocity_tracking_qualified') is not False):
        errors.append('invalid_or_overstated_basic_control_audit')
    validation = payloads.get('validation_file', {})
    if validation.get('run_complete') is not True:
        errors.append('basic_physical_validation_incomplete')
    return dict(qualified=not errors, audit_errors=errors,
                scope='basic_stable_control_and_interface_adaptation',
                precision_and_avoidance_qualified=False), sources


def audit(package=PACKAGE):
    package = Path(package).resolve()
    registry_path = package/'ROBOT_DEPLOYMENT.json'
    registry = json.loads(registry_path.read_text())
    results, errors, sources = [], [], {str(registry_path): sha256(registry_path)}
    entries = registry['robots']
    if len(entries) != 5 or {x['directory'] for x in entries} != ROBOTS:
        errors.append('deployment registry must contain exactly the five installed robots')
    for entry in entries:
        folder = package/entry['directory']
        source = folder/'WBC_READINESS.json'
        row = dict(robot=entry['robot'], directory=entry['directory'], audit_errors=[])
        evidence = package/entry['validation']
        expected = entry['validation_sha256']
        row['historical_motion_evidence'] = dict(path=str(evidence), sha256_expected=expected,
            sha256_actual=sha256(evidence) if evidence.is_file() else None,
            scope='historical_controller_only_not_recommended_backend_acceptance')
        if row['historical_motion_evidence']['sha256_actual'] != expected:
            row['audit_errors'].append('historical_motion_evidence_missing_or_changed')
        document = {}
        if source.is_file():
            sources[str(source)] = sha256(source)
            try:
                document = json.loads(source.read_text())
                if not isinstance(document, dict):
                    raise ValueError('readiness manifest must be an object')
            except (ValueError, TypeError) as exc:
                row['audit_errors'].append(str(exc))
                document = {}
        else:
            row['audit_errors'].append('WBC_READINESS.json_missing')
        # Missing gates mean unverified. Non-boolean truthy strings never pass.
        row['gates'] = {gate: document.get(gate) is True for gate in GATES}
        evidence_errors, row['evidence_files'], bound_sources = inspect_evidence(folder, document, evidence)
        row['audit_errors'].extend(evidence_errors)
        sources.update(bound_sources)
        row['recommended_backend'] = document.get('recommended_backend', document.get('backend'))
        row['blockers'] = document.get('blockers', [])
        row['unverified_gates'] = [key for key, value in row['gates'].items() if not value]
        row['control_integration_ready'] = bool(all(row['gates'][g] for g in CONTROL_GATES)
                                               and not row['audit_errors'])
        basic, basic_sources = inspect_basic_control(folder, document)
        row['basic_control'] = basic
        sources.update(basic_sources)
        basic_declaration = document.get('basic_control', {})
        if isinstance(basic_declaration, dict) and basic_declaration.get('qualified') is True and basic['audit_errors']:
            row['audit_errors'].extend('basic_attestation:'+error for error in basic['audit_errors'])
        row['basic_control_ready'] = bool(not row['audit_errors'] and (
            row['control_integration_ready'] or basic['qualified']))
        row['avoidance_agent_ready'] = bool(all(row['gates'].values()) and not row['audit_errors'])
        if row['audit_errors']:
            errors.append(entry['directory'])
        results.append(row)
    return dict(schema='jev_wbc_readiness_audit/v1', created_utc=datetime.now(timezone.utc).isoformat(),
                audit_complete=not errors, audit_errors=errors, robots=results,
                all_recommended_backends_ready=not errors and bool(results) and all(
                    r['gates']['backend_runtime_ready'] and r['gates']['continuous_camera_tracking_verified']
                    for r in results),
                all_control_integration_gates_passed=not errors and bool(results) and all(
                    r['control_integration_ready'] for r in results),
                all_basic_control_gates_passed=not errors and bool(results) and all(
                    r['basic_control_ready'] for r in results),
                all_avoidance_agent_gates_passed=not errors and bool(results) and all(r['avoidance_agent_ready'] for r in results),
                sources_sha256=sources,
                interpretation='inspection completion is separate from controller and avoidance readiness; '
                    'bound files detect changes; a manifest is an attestation, '
                    'not a replacement for independent physical acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    parser.add_argument('--output', type=Path)
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument('--require-ready', action='store_true', help='Require control, environment and sensor avoidance gates')
    gate.add_argument('--require-control-ready', action='store_true', help='Require all three control gates; does not authorize avoidance')
    gate.add_argument('--require-basic-control-ready', action='store_true',
        help='Accept explicit reduced G1/Go2 basic qualification; precision and avoidance remain separate')
    args = parser.parse_args()
    result = audit(args.package)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Historical inspection reports cannot be silently replaced.
        with args.output.open('x') as stream:
            stream.write(payload)
    print(payload, end='')
    if not result['audit_complete']:
        return 1
    if args.require_ready and not result['all_avoidance_agent_gates_passed']:
        return 2
    if args.require_control_ready and not result['all_control_integration_gates_passed']:
        return 2
    if args.require_basic_control_ready and not result['all_basic_control_gates_passed']:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
