#!/usr/bin/env python3
"""Read-only G1 bounded module recovery audit; physical gates remain separate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from jev_agent_wbc.wbc_recovery_audit import audit_recovery_rows


def audit(folder):
    folder = Path(folder).resolve()
    paths = [folder/name for name in ('summary.json', 'timed_tasks.json', 'measurements.jsonl')]
    try:
        summary = json.loads(paths[0].read_text())
        if summary.get('robot') != 'g1' or summary.get('run_complete') is not True:
            raise ValueError('complete measured G1 run required')
        tasks = json.loads(paths[1].read_text())
        rows = [json.loads(line) for line in paths[2].read_text().splitlines()]
        result = audit_recovery_rows(rows, tasks)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = dict(passed=False, audit_errors=[str(error)])
    paths.extend([Path(__file__), PACKAGE/'jev_agent_wbc/wbc_recovery_audit.py'])
    result['sources_sha256'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in paths if path.is_file()}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['passed'] else 2)
