"""Resolve historical trajectories by content, never by a reused combo number."""
import hashlib
from pathlib import Path


def resolve_recorded_trajectory(package, metadata, fallback_name, record_folder=None):
    package = Path(package)
    candidates = []
    if record_folder is not None:
        candidates.append(Path(record_folder)/'trajectory.json')
    if metadata.get('trajectory'):
        candidates.append(Path(metadata['trajectory']))
    candidates.append(package/'trajectories'/f'{fallback_name}.json')
    expected = metadata.get('trajectory_sha256')
    if expected:
        candidates.extend(sorted((package/'trajectories').glob('*.json')))
        candidates.extend(sorted((package/'test/fixtures').glob('*.json')))
        for path in candidates:
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                return path
        raise FileNotFoundError(f'No retained trajectory matches {fallback_name} SHA-256 {expected}')
    # The original baseline predates per-run metadata. Its negative-control
    # fixture is outside the selectable, numbered trajectory catalogue.
    if fallback_name == 'shot_combo':
        candidates.append(package/'test/fixtures/combo_unreachable.json')
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'No trajectory snapshot available for {fallback_name}')
