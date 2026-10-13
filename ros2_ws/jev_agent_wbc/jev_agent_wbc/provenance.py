"""Small reproducibility manifest; no mutable results enter the source digest."""
import hashlib
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

from .contracts import digest, native
from .jev import POLICY, CANDIDATE_POLICY, SELECTION_POLICY, INTENT_POLICY


def source_manifest(root, configuration=None):
    root = Path(root)
    hashes = {}
    for folder in ('jev_agent_wbc', 'scripts', 'urdf', 'meshes', 'config', 'scenarios', 'launch'):
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for name in ('numpy', 'scipy', 'osqp', 'torch', 'isaacsim', 'sam2'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return dict(source_sha256=digest(hashes), files=hashes, python=sys.executable,
        created_at_utc=datetime.now(timezone.utc).isoformat(), dependency_versions=versions,
        python_version=platform.python_version(), platform=platform.platform(),
        semantic_model=(configuration or {}).get('model', 'perplexity'), semantic_policy='screened_candidate_ranking',
        selection_policy_sha256=hashlib.sha256(SELECTION_POLICY.encode()).hexdigest(),
        intent_policy_sha256=hashlib.sha256(INTENT_POLICY.encode()).hexdigest(),
        policy_sha256=hashlib.sha256(CANDIDATE_POLICY.encode()).hexdigest(),
        legacy_policy_sha256=hashlib.sha256(POLICY.encode()).hexdigest(),
        configuration=native(configuration or {}))
