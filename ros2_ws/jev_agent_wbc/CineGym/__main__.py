"""Generate, execute and summarize CineGym benchmarks."""
import argparse
import json
from pathlib import Path
import sys

from .backends import build_command, run_closed_loop
from .config import DIFFICULTIES, FAMILIES, GymConfig
from .episode import Episode, write_json
from .evaluate import summarize_runs
from .generator import generate_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    gen = commands.add_parser('generate', help='write a new seeded suite')
    gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--count', type=int, default=42)
    gen.add_argument('--seed', type=int, default=0)
    gen.add_argument('--family', choices=(*FAMILIES, 'mixed'), default='mixed')
    gen.add_argument('--difficulty', choices=DIFFICULTIES, default='medium',
                     help='used with --random; balanced suites cycle through all difficulties')
    gen.add_argument('--split', choices=('train', 'validation', 'test'), default='train')
    gen.add_argument('--obstacles', type=int)
    gen.add_argument('--generation-style',choices=('natural','legacy'),default='natural')
    gen.add_argument('--no-ground-boxes',action='store_true',help='sphere-only compatibility distribution')
    gen.add_argument('--require-avoidance-witness',action='store_true',
                     help='require an offline CPU safe timing witness; never passed to the Agent')
    gen.add_argument('--combo-actions', type=int, choices=(2, 3, 4), help='number of actions in random_combo')
    gen.add_argument('--enforce-global-penetration',action='store_true',
                     help='cap sampled whole-trajectory penetration by the selected difficulty')
    gen.add_argument('--duration', type=float, nargs=2, default=(8., 12.), metavar=('MIN', 'MAX'))
    gen.add_argument('--random', action='store_true', help='random families at fixed --difficulty instead of balanced coverage')
    run = commands.add_parser('run', help='existing Agent closed loop on CPU or PhysX')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--case', default='all', help='all, or comma-separated generated case names')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--backend', choices=('cpu', 'physx'), default='cpu')
    run.add_argument('--mode', choices=('jev', 'strict', 'rule'), default='jev')
    run.add_argument('--isaac-python')
    run.add_argument('--python', default=sys.executable)
    run.add_argument('--api-url')
    run.add_argument('--model')
    run.add_argument('--timeout', type=float, default=600., help='per episode wall-clock process bound')
    run.add_argument('--dry-run', action='store_true', help='verify inputs and print argv without executing')
    run.add_argument('extra_args', nargs=argparse.REMAINDER, help='backend options following --')
    report = commands.add_parser('report', help='aggregate measured results')
    report.add_argument('--runs', type=Path, required=True)
    report.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'generate':
        config = GymConfig(family=args.family, difficulty=args.difficulty, split=args.split,
                           generation_style=args.generation_style,
                           ground_boxes=not args.no_ground_boxes,
                           require_avoidance_witness=args.require_avoidance_witness,
                           obstacle_count=args.obstacles, combo_actions=args.combo_actions,
                           enforce_global_penetration=args.enforce_global_penetration,
                           action_duration_s=tuple(args.duration))
        generate_suite(args.output, args.count, args.seed, config, balanced=not args.random)
        return 0
    if args.command == 'report':
        result = summarize_runs(args.runs)
        if args.output:
            if args.output.exists():
                parser.error('report output already exists')
            write_json(args.output, result)
        print(json.dumps(result['groups'], ensure_ascii=False, indent=2))
        return 0
    catalog = json.loads((args.suite/'catalog.json').read_text())
    if args.case == 'all' and not catalog.get('complete'):
        parser.error('suite is incomplete; select existing cases explicitly or finish generation')
    names = catalog['cases'] if args.case == 'all' else args.case.split(',')
    if not names or len(set(names)) != len(names) or any(n not in catalog['cases'] or Path(n).name != n for n in names):
        parser.error('case names must be unique generated catalog entries')
    extra = args.extra_args[1:] if args.extra_args[:1] == ['--'] else args.extra_args
    options = dict(backend=args.backend, mode=args.mode, python=args.python, isaac_python=args.isaac_python,
                   api_url=args.api_url, model=args.model, extra_args=extra)
    # Validate the entire requested selection before starting the first process.
    for name in names:
        episode = Episode.load(args.suite/name)
        expected = next(r['sha256'] for r in catalog['episodes'] if r['case'] == name)
        if episode.metadata['sha256'] != expected:
            parser.error('episode no longer matches suite manifest')
        command = build_command(args.suite/name, args.output/name, **options)
        if args.dry_run:
            print(json.dumps(command))
    if args.dry_run:
        return 0
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in names:
        print(f'{args.backend}: {name}', flush=True)
        result = run_closed_loop(args.suite/name, args.output/name, timeout=args.timeout, **options)
        results.append(result)
        write_json(args.output/'runs.json', results)
        write_json(args.output/'benchmark.json', summarize_runs(args.output))
    return 0 if all(r.get('returncode') == 0 and r.get('summary', {}).get('completed') for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
