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
from jev_agent_wbc.actuation import advance_command
from jev_agent_wbc.dynamic_collision import DynamicCollision
from jev_agent_wbc.evaluation import save_json
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.scenarios import NAMES, RuntimeScene
from jev_agent_wbc.whole_body_qp import Limits, WholeBodyQP


def audit(directory):
    results = []
    for name in NAMES:
        scene = RuntimeScene(directory/name/'runtime_scene.json')
        fixture = scene.document
        reference = OriginalReference(directory/name/'task.json', fixture['subject']['position'])
        state = np.array(fixture['initial_state'], dtype=float)
        q_command = state[3:].copy()
        limits = Limits(command_ramp=True, position_margin=.03)
        limits.velocity *= .95
        limits.wheel_speed *= .95
        limits.acceleration *= .4
        limits.acceleration[1] = Limits().acceleration[1]*.15
        limits.wheel_acceleration *= .4
        controller = WholeBodyQP(limits)
        rows, failure = [], ''
        for t in np.arange(0., reference.duration+.001, .02):
            truth = DynamicCollision(scene.truth_collision(t), t, uncertainty_rate=0.)
            rows.append(dict(t=float(t), clearance_m=float(truth.clearance(state, fixture['base_z']).min())))
            if t >= reference.duration:
                break
            previous = controller.previous.copy()
            controller.time_s = t
            try:
                command, _ = controller.solve(state, *reference.sample(t), .02,
                    base_z=fixture['base_z'], commanded_q=q_command)
            except RuntimeError as error:
                failure = str(error)
                break
            state = advance_command(state, previous, command, .02, True)
            q_command += .01*(previous[2:]+command[2:])
        results.append(dict(case=name, min_clearance_m=min(r['clearance_m'] for r in rows),
            collision_samples=sum(r['clearance_m'] < 0 for r in rows),
            margin_violation_samples=sum(r['clearance_m'] < .025 for r in rows),
            elapsed_s=rows[-1]['t'], failure=failure))
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
