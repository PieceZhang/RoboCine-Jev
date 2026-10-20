#!/usr/bin/env python3
"""Read-only source closure check; no physics or new qualification claim."""
import hashlib
import json
from pathlib import Path

ROBOT=Path(__file__).resolve().parents[1]

def main():
    profile=json.loads((ROBOT/'basic_control_profile.json').read_text())
    errors=[]
    for relative,value in profile['runtime_sources_sha256'].items():
        path=ROBOT/relative
        if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=value:
            errors.append(relative)
    print(json.dumps(dict(schema='go2_basic_source_preflight/v1',source_closure_pass=not errors,
        checked_files=len(profile['runtime_sources_sha256']),mismatches=errors,
        new_physics_run=False,precision_qualification=False),indent=2))
    return 2 if errors else 0

if __name__=='__main__':
    raise SystemExit(main())
