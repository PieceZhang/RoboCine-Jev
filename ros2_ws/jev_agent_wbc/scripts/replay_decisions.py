#!/usr/bin/env python3
"""Compare policies on identical recorded candidates; no simulator or robot writes.

This measures decision agreement, refusal and latency, not cinematic success.
Human labels and closed-loop execution are separate experiments.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_agent_wbc.contracts import digest, native
from jev_agent_wbc.jev import CandidateDecider, PhaseAwareDecider, SystemOneClient


def ablate(evidence, name):
    data = deepcopy(evidence)
    if name == 'no_context':
        data['phase'] = dict(name='unknown', motion='neutral')
        data['next_phase'] = None
        data['execution_history'] = []
        for card in data['candidates']:
            card['visual']['phases'] = {}
    elif name == 'no_future':
        data['next_phase'] = None
        phase = data['phase']['name']
        for card in data['candidates']:
            visual = card['visual']
            if phase not in visual['phases']:
                raise ValueError('candidate lacks current-phase prediction; cannot form no_future ablation')
            visual['summary'] = dict(visual['phases'][phase])
            visual['phases'] = {phase: visual['phases'][phase]}
            # Do not leak terminal loss/global reserve through scalar ranking scores.
            card['prediction'] = {'feasible': card['physical_valid']}
    elif name == 'no_perception':
        data.pop('subject_evidence', None)
        for card in data['candidates']:
            card['visual'].pop('measured_scale_usable', None)
    elif name != 'none':
        raise ValueError('unknown ablation')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path, nargs='+')
    parser.add_argument('--selector', choices=['rule', 'pose', 'jev'], default='rule')
    parser.add_argument('--ablation', choices=['none', 'no_context', 'no_future', 'no_perception'], default='none')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000/v1/systemone')
    parser.add_argument('--model', default='perplexity', help='same typed endpoint for model comparisons')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.selector == 'pose' and args.ablation == 'no_future':
        parser.error('no_future removes the full-horizon pose objective required by the pose selector')
    args.output.mkdir(parents=True, exist_ok=False)
    selector = (CandidateDecider(SystemOneClient(args.api_url, model=args.model,
                journal=args.output/'api.jsonl')) if args.selector == 'jev' else PhaseAwareDecider())
    rng, rows = random.Random(args.seed), []
    for path in args.events:
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event['event'] != 'selection_requested':
                continue
            source = event['evidence']
            data = ablate(source, args.ablation)
            rng.shuffle(data['candidates'])
            begin = time.monotonic()
            try:
                if args.selector == 'pose':
                    selected = min(data['candidates'], key=lambda c: c['prediction']['original_objective'])
                    result = dict(candidate_id=selected['id'])
                else:
                    result = selector.decide(data)
                if result['candidate_id'] not in {c['id'] for c in source['candidates']}:
                    raise ValueError('selector returned unshared candidate')
            except Exception as error:
                result = dict(error=f'{type(error).__name__}: {error}')
            rows.append(dict(event_file=str(path), time_s=event['time_s'],
                shared_candidate_ids=sorted(c['id'] for c in source['candidates']),
                source_evidence_sha256=digest(source), presented_evidence_sha256=digest(data),
                wall_s=time.monotonic()-begin, result=result))
    report = dict(model=args.model if args.selector == 'jev' else None, selector=args.selector, ablation=args.ablation, seed=args.seed,
        decisions=len(rows), refusals_or_errors=sum('error' in r['result'] for r in rows), rows=rows,
        scope='fixed candidate decision replay; not new closed-loop runs or success-rate evidence')
    (args.output/'replay.json').write_text(json.dumps(native(report), indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    return 0 if rows else 2


if __name__ == '__main__':
    raise SystemExit(main())
