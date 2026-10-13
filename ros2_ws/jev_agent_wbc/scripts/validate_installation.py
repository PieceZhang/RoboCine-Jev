#!/usr/bin/env python3
"""Stationary CPU installation fixture; NOT a Jev or intent-quality experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_agent_wbc.engine import CinematicEngine
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.kc_svsdf import advance


class DeterministicTestDecider:
    def __init__(self):
        self.client = SimpleNamespace(calls=0, closed=False)

    def decide(self, evidence):
        if evidence['stage'] == 'strategy':
            return dict(stage='strategy', strategy='framing')
        cards = evidence['candidates']
        chosen = next((c for c in cards if np.linalg.norm(c['trajectory']['values']) > 0), cards[0])
        return dict(stage='selection', candidate_id=chosen['id'], selected_probability=None, test_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = ROOT/'scenarios/movein'
    config = json.loads((root/'runtime_scene.json').read_text())
    task = json.loads((root/'task.json').read_text())
    first = task['reference_camera_trajectory']['samples'][0]
    for row in task['reference_camera_trajectory']['samples']:
        row['position'] = first['position']
        row['quaternion_xyzw'] = first['quaternion_xyzw']
    task['intent'] = 'Hold the original static view; deterministic pipeline verification only.'
    original = OriginalReference(task, config['subject']['position'])
    state, z = np.array(config['initial_state']), config['base_z']
    scene = dict(format='jvs_lidar_surfels/v1', surfel_centers=np.array([[3., 3., .2]]),
        surfel_radius=.025, surfel_velocities=np.zeros((1, 3)),
        observation_stamp=0., minimum_clearance_m=.025)
    engine = CinematicEngine(original, scene, args.output, mode='rule', decider=DeterministicTestDecider())
    engine.pending_feedback = [dict(error='explicit_integration_test_trigger', time_s=0.)]
    begin, failure = time.monotonic(), ''
    try:
        for i in range(401):
            t = i*.02
            scene = dict(scene, observation_stamp=t)
            command, _ = engine.step(state, scene, t, .02, z, commanded_q=state[3:])
            if t < original.duration:
                state = advance(state, command, .02)
            remaining = (i+1)*.02-(time.monotonic()-begin)
            if remaining > 0:
                time.sleep(remaining)
    except Exception as error:
        failure = repr(error)
    finally:
        summary = dict(engine.summary(), failure=failure, wall_s=time.monotonic()-begin,
            scope='stationary CPU fixture with forced event and test selector; no Jev claim')
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary))
        engine.close()
    return 0 if not failure and summary['installed_plans'] > 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
