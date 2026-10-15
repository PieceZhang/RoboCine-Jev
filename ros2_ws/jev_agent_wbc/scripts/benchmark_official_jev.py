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
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(TYPESAFE_API_KEY_FILE=str(a.key_file.resolve()),
                      PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    sys.path.insert(0, str(ROOT))
    from jev_agent_wbc.provenance import source_manifest
    from jev_agent_wbc.scenarios import NAMES
    config = dict(model=a.model, api_url='https://api.typesafe.ai/v1/systemone',
                  pace=.03, activation_lead_s=1.2, candidate_wall_budget_s=20.,
                  parallel_cases=6, planner_model='floating_camera',
                  representation_boxes=3, geometry_preferences=True,
                  obstacle_avoidance_enabled=True)
    (a.output/'provenance.json').write_text(json.dumps(source_manifest(ROOT, config), indent=2))
    shutil.copytree(ROOT/'jev_agent_wbc', a.output/'frozen_code'/'jev_agent_wbc',
                    ignore=shutil.ignore_patterns('__pycache__'))
    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = []
        for name in NAMES:
            command = [sys.executable, '-m', 'jev_agent_wbc.runner', '--case', name,
                       '--mode', 'jev', '--model', a.model, '--api-url', config['api_url'],
                       '--pace', '.03', '--activation-lead', '1.2',
                       '--candidate-wall-budget', '20', '--output', str(a.output/name)]
            futures.append(pool.submit(run_one, command, a.output/(name+'.log')))
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps(result), flush=True)
            (a.output/'process_results.json').write_text(json.dumps(results, indent=2))
    return 0 if all(x['returncode'] == 0 for x in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
