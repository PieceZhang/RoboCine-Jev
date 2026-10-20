#!/usr/bin/env python3
"""CPU-only checks of installed genuine Pink, height IO, and bounded transport.

No Physics/Kit startup. A passing result cannot establish physical acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import select
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    backend = load('g1_backend_preflight', ROOT/'wbc_backend.py')
    rig = load('g1_rig_preflight', ROOT.parent/'jev_agent_wbc/robot_sensor_rig.py')
    io = yaml.safe_load((ROOT/'config/height_policy_io.yaml').read_text())
    names = io['articulations']['robot']['joint_names']
    policy_names = io['actions'][0]['joint_names']
    q = dict(zip(names, io['articulations']['robot']['default_joint_pos']))
    camera_spec, _ = rig.load_config('g1', ROOT.parent/'config/robot_sensor_mounts.json')
    extrinsic = rig.optical_transform(camera_spec['camera'])
    torch.set_num_threads(1)
    policy = torch.jit.load(str(ROOT/'policies/unitree_g1_velocity_height_recurrent_student.pt'), map_location='cpu').eval()
    policy.reset_flat()
    with torch.inference_mode():
        action = policy(torch.zeros(80))
    assert action.shape == (12,) and torch.isfinite(action).all()
    assert sum(x['shape'][0] for x in io['observations']['policy']) == 80
    upper = backend.PinkUpperBody(policy_names, names, extrinsic)
    camera = upper.predicted_camera(q, np.eye(4))
    target = camera.copy(); target[0, 3] += .01
    child = backend.PinkProcess(policy_names, names, extrinsic, ROOT/'kinematics/g1_29dof.urdf')
    statuses = []
    try:
        child.warmup(q)
        for stamp, receive_stamp in [(0., .02), (.02, .2), (.2, .22)]:
            assert child.submit(q, np.eye(4), target, stamp, camera)
            # This CPU preflight has no physical clock; waiting here tests the
            # isolated IPC before any real-time loop, never pausing PhysX.
            assert select.select([child.process.stdout], [], [], 1.)[0]
            packet = child.poll(receive_stamp); statuses.append(packet['status'])
            if receive_stamp-stamp <= .06:
                assert packet['ready'] and set(packet['velocity']) == set(child.owners['pink'])
                assert max(abs(v) for v in packet['velocity'].values()) <= .020001
                np.testing.assert_allclose(packet['predicted_camera'], camera, atol=1e-12)
            else:
                assert not packet['ready'] and child.needs_reset and 'velocity' not in packet
        assert statuses == ['fresh_command', 'stale_or_failed_result_dropped', 'fresh_command']
        runtime = child.runtime
    finally:
        child.close()
    sources = [ROOT/'wbc_backend.py', Path(__file__), ROOT/'config/height_policy_io.yaml',
               ROOT/'kinematics/g1_29dof.urdf', ROOT/'policies/unitree_g1_velocity_height_recurrent_student.pt',
               ROOT.parent/'jev_agent_wbc/robot_sensor_rig.py', ROOT.parent/'config/robot_sensor_mounts.json']
    result = dict(schema='g1_wbc_cpu_preflight/v1', all_pass=True, physical_execution=False,
                  actual_pink=True, solver_runtime=runtime, policy_observation_dimension=80,
                  policy_action_dimension=12, ownership=upper.owners, fixed_base_dof=upper.model.nv,
                  worker_status_sequence=statuses, stale_velocity_dropped=True,
                  recovered_from_fresh_state=True, physical_readiness_inferred=False,
                  interpreter=sys.executable,
                  sources_sha256={str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f):
                                  hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
