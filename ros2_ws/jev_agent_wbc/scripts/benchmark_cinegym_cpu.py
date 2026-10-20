#!/usr/bin/env python3
"""Reproducible easy/medium CPU benchmark, restricted to one/two actions."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(os.environ.get('CINEGYM_SOURCE_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))


def generate(args):
    if not args.frozen_generation:
        frozen=args.output.with_name(args.output.name+'_source')
        frozen.mkdir(parents=True,exist_ok=False)
        for name in ('jev_agent_wbc','CineGym'):
            shutil.copytree(ROOT/name,frozen/name,ignore=shutil.ignore_patterns('__pycache__'))
        env=dict(os.environ,CINEGYM_SOURCE_ROOT=str(frozen.resolve()))
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--frozen-generation'],env=env)
        if result.returncode:raise SystemExit(result.returncode)
        return
    from CineGym import GymConfig, generate_episode
    from CineGym.episode import write_json
    families = ('movein', 'truck_left', 'orbit', 'pedestal_up', 'moveout',
                'truck_right', 'pedestal_down', 'combo_01_easy', 'combo_02_easy', 'random_combo')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for i in range(args.count):
        config = GymConfig(family=families[(i // 2) % len(families)],
                           generation_style=args.generation_style,
                           ground_boxes=not args.no_ground_boxes,
                           require_avoidance_witness=args.require_avoidance_witness,
                           max_attempts=args.max_attempts,
                           difficulty=('easy', 'medium')[i % 2], split=args.split,
                           combo_actions=2,enforce_global_penetration=args.calibrate_difficulty,
                           action_duration_s=tuple(args.duration),initial_clearance_m=args.initial_clearance)
        episode = generate_episode(args.seed + i, config)
        assert len(episode.task['shot_phase']) in (1, 2)
        episode.save(args.output / episode.scene['name'])
        records.append(dict(case=episode.scene['name'], seed=args.seed + i,
                            family=config.family, difficulty=config.difficulty,
                            sha256=episode.metadata['sha256']))
        write_json(args.output / 'catalog.json', dict(version='cinegym/v3',
            seed=args.seed, split=args.split, requested_count=args.count,
            complete=len(records) == args.count, cases=[r['case'] for r in records], episodes=records,
            coverage=dict(families=dict(Counter(r['family'] for r in records)),
                          difficulties=dict(Counter(r['difficulty'] for r in records)))))
        print(f'generated {i+1}/{args.count}: {episode.scene["name"]}', flush=True)


def execute(args):
    from CineGym.episode import write_json
    from CineGym.evaluate import summarize_runs
    from CineGym.backends import execution_environment
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / 'frozen_code'
    for name in ('jev_agent_wbc', 'CineGym'):
        source = args.controller_source if name == 'jev_agent_wbc' and args.controller_source else ROOT / name
        shutil.copytree(source, frozen / name, ignore=shutil.ignore_patterns('__pycache__'))
    # Mark this as a standalone source package; scenarios are explicit inputs.
    (frozen / 'scenarios').mkdir()
    catalog = json.loads((args.suite / 'catalog.json').read_text())
    if not catalog['complete']:
        raise ValueError('incomplete suite')
    names = catalog['cases'] if args.cases == 'all' else args.cases.split(',')
    if args.limit:
        names = names[:args.limit]
    if any(name not in catalog['cases'] for name in names):
        raise ValueError('unknown case')
    environment = execution_environment('cpu')
    environment.update(PYTHONPATH=str(frozen.resolve()), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    # Frozen modules find the official credential file through this environment;
    # its contents never enter commands, manifests or stdout.
    if args.max_wall_s is not None and args.max_wall_s<=12:
        raise ValueError('max-wall-s must leave at least 12 seconds for bounded shutdown')
    batch_started=time.monotonic()
    write_json(args.output/'benchmark_manifest.json',dict(mode=args.mode,pace=1.,
        lead_s=args.lead,candidate_wall_budget_s=args.budget,case_timeout_s=args.timeout,
        max_wall_s=args.max_wall_s,case_order=names,suite=str(args.suite.resolve()),
        started_utc=datetime.now(timezone.utc).isoformat(),frozen_source=str(frozen.resolve())))
    results = []
    for name in names:
        remaining=None if args.max_wall_s is None else args.max_wall_s-(time.monotonic()-batch_started)
        if remaining is not None and remaining<=12:
            write_json(args.output/'budget_stop.json',dict(reason='batch wall-clock limit',
                completed_cases=[row['case'] for row in results],remaining_cases=names[len(results):]))
            break
        case_timeout=args.timeout if remaining is None else min(args.timeout,remaining-12.)
        write_json(args.output/'checkpoint.json',dict(active_case=name,completed=len(results),
            requested=len(names),status='running',wall_s=time.monotonic()-batch_started))
        command = [sys.executable, '-m', 'CineGym', 'run', '--suite', str(args.suite.resolve()),
                   '--case', name, '--output', str((args.output / name).resolve()),
                   '--backend', 'cpu', '--mode', args.mode, '--timeout', str(case_timeout)]
        if args.mode != 'rule' and args.candidate_mode == 'single':
            command += ['--', '--single-candidate', '--candidate-proposals', '1',
                        '--candidate-budget', '1', '--candidate-wall-budget', str(args.budget),
                        '--activation-lead', str(args.lead)]
        else:
            command += ['--', '--async-control', '--candidate-wall-budget', str(args.budget),
                        '--activation-lead', str(args.lead)]
        if args.startup != 'none':
            if args.mode != 'jev':
                raise ValueError('startup semantic planning requires Jev')
            command += ['--'+args.startup]
        with (args.output / (name + '.log')).open('w') as log:
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        results.append(dict(case=name, returncode=process.returncode))
        write_json(args.output / 'process_results.json', results)
        report = summarize_runs(args.output)
        write_json(args.output / 'benchmark.json', report)
        print(json.dumps(dict(case=name, groups=report['groups'].get('cpu/all', {}))), flush=True)
    write_json(args.output/'checkpoint.json',dict(completed=len(results),requested=len(names),
        status='complete' if len(results)==len(names) else 'budget_stopped',wall_s=time.monotonic()-batch_started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--count', type=int, default=20)
    gen.add_argument('--seed', type=int, default=1000)
    gen.add_argument('--split', choices=('train', 'validation', 'test'), default='train')
    gen.add_argument('--generation-style',choices=('natural','legacy'),default='natural')
    gen.add_argument('--no-ground-boxes',action='store_true')
    gen.add_argument('--require-avoidance-witness',action='store_true')
    gen.add_argument('--max-attempts',type=int,default=24)
    gen.add_argument('--frozen-generation',action='store_true',help=argparse.SUPPRESS)
    gen.add_argument('--calibrate-difficulty', action='store_true',
                     help='cap sampled full-trajectory penetration by the difficulty profile')
    gen.add_argument('--duration',type=float,nargs=2,default=(8.,12.))
    gen.add_argument('--initial-clearance',type=float,default=.12)
    run = sub.add_parser('run')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--mode', choices=('strict', 'rule', 'jev'), default='jev')
    run.add_argument('--controller-source', type=Path, help='explicit frozen controller package for paired baselines')
    run.add_argument('--cases', default='all')
    run.add_argument('--limit', type=int)
    run.add_argument('--timeout', type=float, default=180.)
    run.add_argument('--max-wall-s',type=float,help='whole-batch wall-clock cap including bounded shutdown')
    run.add_argument('--budget', type=float, default=2.)
    run.add_argument('--lead', type=float, default=1.2)
    run.add_argument('--startup', choices=('none', 'preplan', 'phase-program'), default='none')
    run.add_argument('--candidate-mode', choices=('single', 'multi'), default='single')
    args = parser.parse_args()
    generate(args) if args.command == 'generate' else execute(args)


if __name__ == '__main__':
    main()
