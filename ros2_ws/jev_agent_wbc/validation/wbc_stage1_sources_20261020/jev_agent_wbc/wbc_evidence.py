"""Read-only source/evidence bindings shared by fleet inspection and task probes."""
import hashlib
import json
from pathlib import Path

GATES = ('backend_runtime_ready', 'continuous_camera_tracking_verified',
         'environment_constraints_verified', 'representation_ingress_verified',
         'sensors_avoidance_verified')


def inspect_evidence(folder, document, historical_validation=None):
    folder = Path(folder).resolve()
    package = folder.parent
    errors, files, sources = [], {}, {}
    if document.get('schema') != 'jev_wbc_readiness/v1':
        errors.append('unsupported_readiness_schema')
    for gate in GATES:
        if gate in document and not isinstance(document[gate], bool):
            errors.append(gate+'_must_be_boolean')
    if historical_validation is None:
        registry = package/'ROBOT_DEPLOYMENT.json'
        if registry.is_file():
            for entry in json.loads(registry.read_text()).get('robots', []):
                if entry.get('directory') == folder.name:
                    historical_validation = (package/entry['validation']).resolve()
    elif historical_validation is not None:
        historical_validation = Path(historical_validation).resolve()
    bindings = document.get('evidence_files_sha256', {})
    if not isinstance(bindings, dict):
        errors.append('evidence_files_sha256_must_be_mapping')
        bindings = {}
    positive = any(document.get(gate) is True for gate in GATES)
    if positive and not bindings:
        errors.append('positive_readiness_requires_bound_source_and_evidence')
    for relative, expected in bindings.items():
        target = (folder/relative).resolve()
        if Path(relative).is_absolute() or not target.is_relative_to(package):
            errors.append('evidence_path_outside_package:'+relative)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        files[relative] = dict(expected=expected, actual=actual)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected) or actual != expected):
            errors.append('evidence_missing_or_changed:'+relative)
        if actual:
            sources[str(target)] = actual
    for role in ('runtime_source_files', 'runtime_validation_files'):
        paths = document.get(role, [])
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            errors.append(role+'_must_be_path_list')
            continue
        if positive and not paths:
            errors.append('positive_readiness_requires_'+role)
        for relative in paths:
            if relative not in files or files[relative]['actual'] != files[relative]['expected']:
                errors.append(role+'_not_bound:'+relative)
                continue
            target = (folder/relative).resolve()
            if role == 'runtime_source_files' and target.suffix not in ('.py', '.cpp', '.cc', '.c', '.h', '.hpp', '.so', '.a'):
                errors.append('runtime_source_must_include_executable_code:'+relative)
            if role == 'runtime_validation_files':
                if target == historical_validation:
                    errors.append('historical_motion_is_not_new_backend_validation:'+relative)
                try:
                    if target.suffix != '.json' or not isinstance(json.loads(target.read_text()), dict):
                        raise ValueError('expected result JSON object')
                except (ValueError, OSError):
                    errors.append('runtime_validation_must_be_result_json:'+relative)
    return errors, files, sources
