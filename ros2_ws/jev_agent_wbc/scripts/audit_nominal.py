#!/usr/bin/env python3
"""Offline counterfactual: nominal tracking with obstacle avoidance disabled.

This ideal kinematic diagnostic only checks whether the fixtures challenge the
robot. Obstacle truth is evaluated AFTER each command, never used to plan it.
It is not a PhysX run and is not included in the strict/Jev success counts.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.evaluation import save_json
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.scenarios import NAMES, RuntimeScene
from jev_agent_wbc.whole_body_qp import whole_body_kinematics
from jev_agent_wbc.nominal import nominal_rollout
from scipy.spatial.transform import Rotation


def audit(directory):
    results = []
    for name in NAMES:
        scene = RuntimeScene(directory/name/'runtime_scene.json')
        fixture = scene.document
        reference = OriginalReference(directory/name/'task.json', fixture['subject']['position'])
        times, states, failure = nominal_rollout(reference, fixture['initial_state'], fixture['base_z'])
        distances, errors = [], []
        for t, state in zip(times, states):
            truth = DynamicCollision(scene.truth_collision(t), t, uncertainty_rate=0.)
            distances.append(truth.clearance(state, fixture['base_z']).min(axis=0))
            p, r, _ = whole_body_kinematics(state, fixture['base_z'])
            target_p, target_r, _ = reference.sample(t)
            errors.append([np.linalg.norm(p-target_p), Rotation.from_matrix(r@target_r.T).magnitude()])
        distances, errors = np.array(distances), np.array(errors)
        arm_clearance = distances[:, 1:7].min(axis=1)
        results.append(dict(case=name, min_clearance_m=float(distances.min()),
            collision_samples=int(np.sum(distances.min(axis=1) < 0)),
            margin_violation_samples=int(np.sum(distances.min(axis=1) < .025)),
            min_base_clearance_m=float(distances[:, 0].min()),
            min_arm_clearance_m=float(arm_clearance.min()),
            arm_collision_samples=int(np.sum(arm_clearance < 0)),
            arm_collision_time_range_s=times[arm_clearance < 0][[0, -1]].tolist() if np.any(arm_clearance < 0) else [],
            min_clearance_by_body_m=distances.min(axis=0).tolist(),
            base_travel_m=float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()),
            arm_joint_excursion_rad=np.ptp(states[:, 3:], axis=0).tolist(),
            max_position_error_m=float(errors[:, 0].max()),
            max_orientation_error_rad=float(errors[:, 1].max()),
            elapsed_s=float(times[-1]), failure=failure))
    return dict(role='offline ideal-kinematic counterfactual; not physical acceptance',
                control='nominal WholeBodyQP; no obstacle input to control', results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', type=Path, default=ROOT/'scenarios')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.scenarios)
    save_json(args.output, result)
    for row in result['results']:
        print(row)
