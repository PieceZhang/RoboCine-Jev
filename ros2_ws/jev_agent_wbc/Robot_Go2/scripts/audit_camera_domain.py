#!/usr/bin/env python3
"""Read-only audit of native commands actually issued in a saved Go2 rollout."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('measurements', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    spec = importlib.util.spec_from_file_location('go2_domain_audit', ROOT/'wbc_backend.py')
    backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
    rows = [json.loads(line) for line in a.measurements.read_text().splitlines()]
    rows = [r for r in rows if r.get('T_world_camera_optical_reference') is not None]
    if not rows:
        raise ValueError('no actual optical target/native command rows')
    statuses, violations, outside = Counter(), Counter(), []
    for row in rows:
        check = backend.check_upstream_domain(row['ee_command_mixed'])
        statuses[check['status']] += 1; violations.update(check.get('violations', []))
        if check['status'] == 'out_of_policy_domain':
            outside.append(row['t'])
    command = np.array([r['ee_command_mixed'] for r in rows])
    sources = [a.measurements, ROOT/'wbc_backend.py', Path(__file__),
               backend.UPSTREAM_MDP/'config/go2_piper/wbc_env_cfg.py', backend.UPSTREAM_MDP/'mdp/pose_command_wbc.py']
    result = dict(schema='go2_saved_native_domain_audit/v1', audit_complete=True,
                  physical_reexecution=False, checked_actual_issued_commands=len(rows),
                  native_xyz_min=command[:, :3].min(axis=0).tolist(), native_xyz_max=command[:, :3].max(axis=0).tolist(),
                  statuses=dict(statuses), violations=dict(violations),
                  first_outside_time_s=min(outside) if outside else None,
                  checkpoint_training_coverage='unknown', feasibility_certified=False,
                  interpretation='exceeding pinned configuration outer bounds excludes support; inside does not certify learned feasibility',
                  sources_sha256={str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
