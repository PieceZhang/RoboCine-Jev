"""Readiness cannot inherit a pass from an unrelated historical controller."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('check_wbc_readiness', ROOT/'scripts/check_wbc_readiness.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GATEWAY_SPEC = importlib.util.spec_from_file_location('wbc_task_gateway', ROOT/'scripts/wbc_task_gateway.py')
GATEWAY = importlib.util.module_from_spec(GATEWAY_SPEC)
GATEWAY_SPEC.loader.exec_module(GATEWAY)


def create_package(tmp_path):
    entries = []
    for folder in sorted(MODULE.ROBOTS):
        path = tmp_path/folder
        path.mkdir()
        raw = b'{"historical_pass": true}\n'
        (path/'acceptance.json').write_bytes(raw)
        (path/'adapter.py').write_text('# current native adapter\n')
        (path/'new_probe.json').write_text('{"new_backend_probe":true}\n')
        (path/'WBC_READINESS.json').write_text(json.dumps(dict(
            schema='jev_wbc_readiness/v1',
            recommended_backend='test native backend', blockers=[],
            evidence_files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (path/'acceptance.json', path/'adapter.py', path/'new_probe.json')},
            runtime_source_files=['adapter.py'], runtime_validation_files=['new_probe.json'],
            **dict.fromkeys(MODULE.GATES, True))))
        entries.append(dict(robot=folder, directory=folder, validation=folder+'/acceptance.json',
                            validation_sha256=hashlib.sha256(raw).hexdigest()))
    (tmp_path/'ROBOT_DEPLOYMENT.json').write_text(json.dumps(dict(robots=entries)))
    return tmp_path


def test_historical_evidence_does_not_turn_missing_gate_into_pass(tmp_path):
    package = create_package(tmp_path)
    path = package/'Robot_TIAGo/WBC_READINESS.json'
    document = json.loads(path.read_text())
    del document['environment_constraints_verified']
    path.write_text(json.dumps(document))
    report = MODULE.audit(package)
    assert report['audit_complete']
    assert report['all_recommended_backends_ready']
    assert not report['all_avoidance_agent_gates_passed']
    row = next(r for r in report['robots'] if r['directory'] == 'Robot_TIAGo')
    assert row['unverified_gates'] == ['environment_constraints_verified']


@pytest.mark.parametrize('value', ['true', 1, [], None])
def test_nonboolean_claims_never_pass(tmp_path, value):
    package = create_package(tmp_path)
    path = package/'Robot_G1/WBC_READINESS.json'
    document = json.loads(path.read_text())
    document['backend_runtime_ready'] = value
    path.write_text(json.dumps(document))
    report = MODULE.audit(package)
    assert not report['audit_complete']
    assert not report['all_recommended_backends_ready']
    assert not report['all_avoidance_agent_gates_passed']


def test_changed_historical_evidence_is_reported(tmp_path):
    package = create_package(tmp_path)
    (package/'Robot_Go2/acceptance.json').write_text('{"different":true}')
    report = MODULE.audit(package)
    assert not report['audit_complete']
    assert not report['all_avoidance_agent_gates_passed']


def test_invalid_readiness_manifest_leaves_auditable_failure(tmp_path):
    package = create_package(tmp_path)
    (package/'Robot_Go2/WBC_READINESS.json').write_text('[]')
    report = MODULE.audit(package)
    assert not report['audit_complete']
    row = next(r for r in report['robots'] if r['directory'] == 'Robot_Go2')
    assert 'readiness manifest must be an object' in row['audit_errors']


def test_incomplete_fleet_cannot_report_all_robots_ready(tmp_path):
    package = create_package(tmp_path)
    path = package/'ROBOT_DEPLOYMENT.json'
    data = json.loads(path.read_text())
    data['robots'].pop()
    path.write_text(json.dumps(data))
    report = MODULE.audit(package)
    assert not report['audit_complete']
    assert not report['all_recommended_backends_ready']
    assert not report['all_avoidance_agent_gates_passed']


def test_runtime_source_changes_invalidate_readiness_claim(tmp_path):
    package = create_package(tmp_path)
    folder = package/'Robot_Go2'
    source = folder/'adapter.py'
    source.write_text('# tested source\n')
    manifest = folder/'WBC_READINESS.json'
    data = json.loads(manifest.read_text())
    data['evidence_files_sha256']['adapter.py'] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    assert MODULE.audit(package)['all_avoidance_agent_gates_passed']
    source.write_text('# subsequently changed source\n')
    report = MODULE.audit(package)
    assert not report['audit_complete'] and not report['all_avoidance_agent_gates_passed']
    row = next(r for r in report['robots'] if r['directory'] == 'Robot_Go2')
    assert 'evidence_missing_or_changed:adapter.py' in row['audit_errors']


def test_positive_claim_without_bound_evidence_rejected(tmp_path):
    package = create_package(tmp_path)
    path = package/'Robot_G1/WBC_READINESS.json'
    data = json.loads(path.read_text())
    del data['evidence_files_sha256']
    path.write_text(json.dumps(data))
    assert not MODULE.audit(package)['all_avoidance_agent_gates_passed']


def test_unknown_readiness_version_rejected(tmp_path):
    package = create_package(tmp_path)
    path = package/'Robot_G1/WBC_READINESS.json'
    data = json.loads(path.read_text())
    data['schema'] = 'future/unreviewed'
    path.write_text(json.dumps(data))
    report = MODULE.audit(package)
    assert not report['audit_complete']
    assert not report['all_avoidance_agent_gates_passed']


def test_gateway_requires_actual_live_ingress_not_just_backend_installation(tmp_path):
    from test_wbc_protocol import fixture
    observation, task = fixture()
    folder = create_package(tmp_path)/'Robot_Go2'
    manifest = json.loads((folder/'WBC_READINESS.json').read_text())
    manifest.update(recommended_backend='installed-framework', backend_runtime_ready=True,
                    protocol_features=['timed_camera_se3'], representation_ingress_verified=False)
    result = GATEWAY.probe(task.to_dict(), observation.to_dict(), manifest, 10., robot_dir=folder)
    assert not result['supported'] and 'backend_unavailable' in result['reasons']
    assert result['probe_only'] and not result['motion_dispatched']
    manifest['representation_ingress_verified'] = True
    accepted = GATEWAY.probe(task.to_dict(), observation.to_dict(), manifest, 10., robot_dir=folder)
    assert accepted['supported'] and accepted['feasible'] is None


def test_gateway_rejects_geometry_request_outside_native_command_space(tmp_path):
    from test_wbc_protocol import fixture
    observation, task = fixture(geometry=True)
    folder = create_package(tmp_path)/'Robot_Go2'
    manifest = json.loads((folder/'WBC_READINESS.json').read_text())
    manifest.update(recommended_backend='EE-only-policy', backend_runtime_ready=True,
                    representation_ingress_verified=True, protocol_features=['timed_camera_se3'])
    result = GATEWAY.probe(task.to_dict(), observation.to_dict(), manifest, 10., robot_dir=folder)
    assert not result['supported']
    assert result['unsupported_features'] == ['platform_avoid_region']


def test_old_motion_json_alone_cannot_attest_new_backend(tmp_path):
    package = create_package(tmp_path)
    path = package/'Robot_Go2/WBC_READINESS.json'
    data = json.loads(path.read_text())
    data['evidence_files_sha256'] = {'acceptance.json': data['evidence_files_sha256']['acceptance.json']}
    data['runtime_source_files'] = []
    data['runtime_validation_files'] = ['acceptance.json']
    path.write_text(json.dumps(data))
    report = MODULE.audit(package)
    assert not report['all_avoidance_agent_gates_passed']
    errors = next(r for r in report['robots'] if r['directory'] == 'Robot_Go2')['audit_errors']
    assert 'historical_motion_is_not_new_backend_validation:acceptance.json' in errors
    assert 'positive_readiness_requires_runtime_source_files' in errors


def test_gateway_and_fleet_share_evidence_invalidation(tmp_path):
    from test_wbc_protocol import fixture
    observation, task = fixture()
    package = create_package(tmp_path)
    folder = package/'Robot_Go2'
    manifest = json.loads((folder/'WBC_READINESS.json').read_text())
    manifest['protocol_features'] = ['timed_camera_se3']
    (folder/'adapter.py').write_text('# changed after verification\n')
    assert not MODULE.audit(package)['all_avoidance_agent_gates_passed']
    result = GATEWAY.probe(task.to_dict(), observation.to_dict(), manifest, 10., robot_dir=folder)
    assert not result['supported']
    assert 'evidence_missing_or_changed:adapter.py' in result['readiness_errors']


def test_control_readiness_does_not_certify_missing_sensing_or_environment(tmp_path):
    package = create_package(tmp_path)
    for folder in MODULE.ROBOTS:
        path = package/folder/'WBC_READINESS.json'
        data = json.loads(path.read_text())
        data.update(environment_constraints_verified=False, sensors_avoidance_verified=False)
        path.write_text(json.dumps(data))
    report = MODULE.audit(package)
    assert report['all_control_integration_gates_passed']
    assert not report['all_avoidance_agent_gates_passed']


def test_solver_and_camera_passes_cannot_replace_missing_actual_task_ingress(tmp_path):
    package = create_package(tmp_path)
    path = package/'Robot_G1/WBC_READINESS.json'
    data = json.loads(path.read_text())
    data['representation_ingress_verified'] = False
    path.write_text(json.dumps(data))
    report = MODULE.audit(package)
    assert report['all_recommended_backends_ready']
    assert not report['all_control_integration_gates_passed']


def attach_basic(folder):
    (folder/'run_basic.sh').write_text('# tested basic entry\n')
    (folder/'basic_validation.json').write_text(json.dumps(dict(run_complete=True)))
    (folder/'basic_audit.json').write_text(json.dumps(dict(
        schema='independent_wbc_basic_control_audit/v1', basic_control_pass=True,
        environment_constraints_verified=False, sensors_avoidance_verified=False,
        precision_velocity_tracking_qualified=False)))
    path = folder/'WBC_READINESS.json'; data = json.loads(path.read_text())
    data.update(dict.fromkeys(MODULE.GATES, False))
    data['basic_control'] = dict(qualified=True,
        entrypoint='run_basic.sh', entrypoint_sha256=MODULE.sha256(folder/'run_basic.sh'),
        audit_file='basic_audit.json', audit_sha256=MODULE.sha256(folder/'basic_audit.json'),
        validation_file='basic_validation.json', validation_sha256=MODULE.sha256(folder/'basic_validation.json'))
    path.write_text(json.dumps(data))


def test_reduced_basic_qualification_never_upgrades_precision_or_avoidance(tmp_path):
    package = create_package(tmp_path)
    for name in ('Robot_G1', 'Robot_Go2'):
        attach_basic(package/name)
    report = MODULE.audit(package)
    assert report['audit_complete'] and report['all_basic_control_gates_passed']
    assert not report['all_control_integration_gates_passed']
    assert not report['all_avoidance_agent_gates_passed']


@pytest.mark.parametrize('bad', ['changed_entry','changed_audit','changed_validation','overclaim','truthy'])
def test_basic_claim_needs_its_bound_completed_physical_report(tmp_path, bad):
    package = create_package(tmp_path); folder = package/'Robot_G1'; attach_basic(folder)
    path = folder/'WBC_READINESS.json'; data = json.loads(path.read_text())
    if bad == 'changed_entry': (folder/'run_basic.sh').write_text('# changed\n')
    elif bad == 'changed_audit': (folder/'basic_audit.json').write_text('{}')
    elif bad == 'changed_validation': (folder/'basic_validation.json').write_text('{}')
    elif bad == 'truthy': data['basic_control']['qualified'] = 'true'
    else:
        report = json.loads((folder/'basic_audit.json').read_text())
        report['sensors_avoidance_verified'] = True
        (folder/'basic_audit.json').write_text(json.dumps(report))
        data['basic_control']['audit_sha256'] = MODULE.sha256(folder/'basic_audit.json')
    path.write_text(json.dumps(data))
    assert not MODULE.audit(package)['all_basic_control_gates_passed']
