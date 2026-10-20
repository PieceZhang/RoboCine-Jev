#!/usr/bin/env python3
"""CPU-only native policy/optical mixed-frame checks; no physical certification."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT/'vendor/LeggedManip_Lab/mujoco/deploy/policy/go2_piper/wbc/policy.pt'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    spec = importlib.util.spec_from_file_location('go2_backend_preflight', ROOT/'wbc_backend.py')
    backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
    torch.set_num_threads(1)
    policy = torch.jit.load(str(POLICY), map_location='cpu').eval()
    with torch.inference_mode():
        action = policy(torch.zeros(1, 210))
    assert action.shape == (1, 18) and torch.isfinite(action).all()
    source = json.loads((ROOT/'SOURCES.json').read_text())
    policy_sha = hashlib.sha256(POLICY.read_bytes()).hexdigest()
    expected = next(x['sha256'] for x in source['policies'] if x['path'].endswith('/wbc/policy.pt'))
    assert policy_sha == expected
    rng = np.random.default_rng(20261020)
    maximum_error = 0.
    for _ in range(200):
        transforms = []
        for index in range(3):
            T = np.eye(4); T[:3, 3] = rng.normal(size=3)
            angles = rng.uniform(-.6, .6, 3) if index == 1 else rng.uniform(-2., 2., 3)
            T[:3, :3] = Rotation.from_euler('xyz', angles).as_matrix(); transforms.append(T)
        camera, link0, extrinsic = transforms
        command = backend.camera_to_mixed(camera, link0, extrinsic)
        reconstructed = backend.mixed_to_world_camera(command, link0, extrinsic)
        maximum_error = max(maximum_error, float(np.max(np.abs(reconstructed-camera))))
    assert maximum_error < 1e-12
    inside = backend.check_upstream_domain([.5, 0., .5, 1., 0., 0., 0.])
    outside = backend.check_upstream_domain([-.236, 0., .5, 1., 0., 0., 0.])
    assert inside['supported'] is None and inside['feasible'] is None
    assert outside['supported'] is False and outside['status'] == 'out_of_policy_domain'
    sources = [ROOT/'wbc_backend.py', Path(__file__), POLICY, ROOT/'SOURCES.json',
               backend.UPSTREAM_MDP/'config/go2_piper/wbc_env_cfg.py', backend.UPSTREAM_MDP/'mdp/pose_command_wbc.py']
    result = dict(schema='go2_wbc_cpu_preflight/v1', all_pass=True, physical_execution=False,
                  selected_robot='Go2-Piper', selected_policy='GO2-PIPER-WBC',
                  policy_observation_dimension=210, policy_action_dimension=18,
                  optical_camera_axes='x_right_y_down_z_forward',
                  native_command='xy_link0_z_world_orientation_link0_wxyz',
                  full_extrinsic_roundtrips=200, maximum_roundtrip_matrix_error=maximum_error,
                  inside_outer_domain_result=inside, outside_outer_domain_result=outside,
                  physical_readiness_inferred=False, d1_policy_validated=False,
                  interpreter=sys.executable,
                  sources_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
