#!/usr/bin/env python3
"""Reproducible FK-gate probes and an explicitly forced CPU installation test.

The positive installation fixture uses a distant obstacle and a deterministic
selector of a non-continuation candidate. It tests the asynchronous numerical
pipeline + real CPU WBC, not Jev quality or dynamic-obstacle shot success.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from jev_agent_wbc.candidates import SearchBudget
from jev_agent_wbc.engine import CinematicEngine
from jev_agent_wbc.evaluation import save_json, trajectory_summary
from jev_agent_wbc.jev import PhaseAwareDecider
from jev_agent_wbc.kinematic_feasibility import KinematicSearch, summary
from jev_agent_wbc.kc_svsdf import advance
from jev_agent_wbc.provenance import source_manifest
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.task_space_planning import MeasuredCameraPrediction, build
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
from jev_agent_wbc.dynamic_collision import DynamicCollision


ROOT = Path(__file__).resolve().parents[1]


class InstallationSelector(PhaseAwareDecider):
    def decide(self, evidence):
        if evidence['stage'] == 'strategy':
            return dict(stage='strategy', strategy='view')
        candidates = [c for c in evidence['candidates'] if c['trajectory']['program']['route'] != 'continuation']
        selected = (candidates or evidence['candidates'])[0]
        return dict(stage='selection', candidate_id=selected['id'], selected_probability=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'provenance.json', source_manifest(ROOT, dict(fixture='forced_cpu_installation', model='none')))
    probes = []
    for folder in sorted((ROOT/'scenarios').iterdir()):
        if not (folder/'runtime_scene.json').exists():
            continue
        runtime = json.loads((folder/'runtime_scene.json').read_text())
        state, z = np.array(runtime['initial_state']), runtime['base_z']
        original = OriginalReference(folder/'task.json', runtime['subject']['position'])
        search = KinematicSearch(state, np.zeros(8), 0., z, Limits(), measured_control=np.zeros(8))
        lifted = search.lift(MeasuredCameraPrediction(original, state, z, 0., 1.5), original.duration+.4)
        probes.append(dict(case=folder.name, **summary(lifted)))
    save_json(args.output/'reference_kinematics.json', probes)

    folder = ROOT/'scenarios/movein'
    runtime = json.loads((folder/'runtime_scene.json').read_text())
    original = OriginalReference(folder/'task.json', runtime['subject']['position'])
    state, z = np.array(runtime['initial_state']), runtime['base_z']
    scene = dict(format='jvs_lidar_surfels/v1', observation_stamp=0., surfel_centers=[[8., 8., 8.]],
        surfel_radius=.1, surfel_velocities=[[0., 0., 0.]], minimum_clearance_m=.025,
        subject_model=runtime['subject'])
    budget = SearchBudget(activation_lead_s=1.5)
    negative = build(original, original, 'view', scene, state, np.zeros(8), 0., z,
        replace(Limits(), acceleration=np.full(8, .001), wheel_acceleration=.01), budget=budget,
        measured_control=np.zeros(8))
    save_json(args.output/'low_acceleration_rejection.json', dict(error=negative.get('error'),
        candidate_count=len(negative.get('candidates', [])), evaluations=negative.get('evaluations'),
        kinematic_prior=negative.get('kinematic_prior')))

    engine = CinematicEngine(original, scene, args.output/'installation', mode='rule',
        decider=InstallationSelector(), subject_model=runtime['subject'], budget=budget)
    engine._feedback('diagnostic_forced_replan_for_installation_test', 0., finish_pipeline=False)
    rows, failure = [], ''
    start = time.monotonic()
    try:
        for i in range(401):
            t = i*.02
            observed = dict(scene, observation_stamp=t)
            command, info = engine.step(state, observed, t, .02, base_z=z, commanded_q=state[3:],
                measured_control=engine.wbc.previous.copy())
            p, r, _ = whole_body_kinematics(state, z)
            rows.append(dict(t=t, state=state.tolist(), command=command.tolist(), camera_position=p.tolist(),
                camera_quaternion_xyzw=Rotation.from_matrix(r).as_quat().tolist(), info=info,
                truth_clearance_m=float(DynamicCollision(observed, t).clearance(state, z).min())))
            if i < 400:
                state = advance(state, command, .02)
            remaining = (i+1)*.02/.2-(time.monotonic()-start)
            if remaining > 0:
                time.sleep(min(remaining, .1))
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    finally:
        result = trajectory_summary(rows, original, engine.summary(), failure)
        result.update(fixture='forced_noncontinuation_installation_distant_obstacle',
            backend='CPU_whole_body_WBC', wall_s=time.monotonic()-start,
            semantic_model='none; deterministic integration-test selector',
            original_reference_probes_pass=all(p['feasible'] for p in probes),
            low_acceleration_candidates_rejected=not negative.get('candidates'))
        save_json(args.output/'summary.json', result)
        save_json(args.output/'telemetry.json', rows)
        engine.close()
    print(json.dumps({k: result[k] for k in ('completed', 'failure', 'installed_plans',
        'original_reference_probes_pass', 'low_acceleration_candidates_rejected')}), flush=True)
    return 0 if result['completed'] and result['installed_plans'] > 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
