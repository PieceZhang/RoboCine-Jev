#!/usr/bin/env python3
"""Run the existing six CPU demos with official Jev; preserve every failed run."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_one(command, log):
    with open(log, 'w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    return {'case': command[command.index('--case')+1], 'returncode': result.returncode}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--model', default='jev-1.14.0')
    p.add_argument('--cases', nargs='+', help='Optional subset of the existing scenario catalog')
    p.add_argument('--workers',type=int,help='Concurrent demos; use 1 for isolated latency measurements')
    p.add_argument('--pace', type=float, default=.03)
    p.add_argument('--lead', type=float, default=1.2)
    p.add_argument('--candidate-seconds', type=float, default=12.)
    p.add_argument('--proposals', type=int, default=6)
    p.add_argument('--shortlist', type=int, default=2)
    p.add_argument('--single-candidate',action='store_true')
    p.add_argument('--preplan',action='store_true')
    p.add_argument('--scenarios',type=Path,default=ROOT/'scenarios')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(TYPESAFE_API_KEY_FILE=str(a.key_file.resolve()),
                      PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    sys.path.insert(0, str(ROOT))
    from jev_agent_wbc.provenance import source_manifest
    from jev_agent_wbc.scenarios import NAMES
    names = a.cases or NAMES
    if any(name not in NAMES for name in names):
        p.error('unknown scenario')
    config = dict(model=a.model, api_url='https://api.typesafe.ai/v1/systemone',
                  pace=a.pace, activation_lead_s=a.lead, candidate_wall_budget_s=a.candidate_seconds,
                  max_proposals=a.proposals,max_rollouts=a.shortlist,parallel_cases=a.workers or len(names), planner_model='floating_camera',
                  representation_boxes=3, geometry_preferences=True,
                  obstacle_avoidance_enabled=True,single_candidate=a.single_candidate,preplan=a.preplan,
                  scenarios_source=str(a.scenarios.resolve()))
    (a.output/'provenance.json').write_text(json.dumps(source_manifest(ROOT, config), indent=2))
    shutil.copytree(ROOT/'jev_agent_wbc', a.output/'frozen_code'/'jev_agent_wbc',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(a.scenarios, a.output/'frozen_code'/'scenarios')
    # Workers and their spawned numerical processes execute the same snapshot,
    # even while the next tuning iteration edits the working tree.
    os.environ['PYTHONPATH'] = str((a.output/'frozen_code').resolve())
    results = []
    with ProcessPoolExecutor(max_workers=a.workers or len(names)) as pool:
        futures = []
        for name in names:
            command = [sys.executable, '-m', 'jev_agent_wbc.runner', '--case', name,
                       '--scenarios', str((a.output/'frozen_code'/'scenarios').resolve()),
                       '--mode', 'jev', '--model', a.model, '--api-url', config['api_url'],
                       '--pace', str(a.pace), '--activation-lead', str(a.lead),
                       '--candidate-wall-budget', str(a.candidate_seconds), '--candidate-proposals',str(a.proposals),
                       '--candidate-budget',str(a.shortlist), '--output', str(a.output/name)]
            if a.single_candidate:command.append('--single-candidate')
            if a.preplan:command.append('--preplan')
            futures.append(pool.submit(run_one, command, a.output/(name+'.log')))
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps(result), flush=True)
            (a.output/'process_results.json').write_text(json.dumps(results, indent=2))
    return 0 if all(x['returncode'] == 0 for x in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
