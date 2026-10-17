#!/usr/bin/env python3
"""Audit counts, official model provenance, execution acceptance and latency."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jev_agent_wbc.scenarios import NAMES
from jev_agent_wbc.evaluation import save_json


def analyze(runs,independent=False):
    result=[]
    for name in NAMES:
        folder=runs/name/(name+'_jev')
        if not (folder/'summary.json').exists():
            result.append(dict(case=name,status='incomplete'));continue
        summary=json.loads((folder/'summary.json').read_text())
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        rows=json.loads((folder/'telemetry.json').read_text())
        api_path=folder/'jev_api.jsonl'
        api=[json.loads(line) for line in api_path.read_text().splitlines()] if api_path.exists() else []
        sets=[e for e in events if e['event']=='candidate_set']
        failed=[e for e in events if e['event']=='math_failure' and e.get('kind')=='candidates']
        evaluations=[v for e in sets+failed for v in e.get('evaluations',[])]
        cards=[c for e in sets for c in e['cards']]
        models=sorted({a['response']['model'] for a in api if a.get('response',{}).get('model')})
        one=bool(len(sets)==1 and not failed and len(cards)==len(evaluations)==1 and
                 sets[0].get('generated_proposals')==1 and not evaluations[0].get('rejected') and
                 cards[0]['physical_valid'] and summary.get('installed_plans')==1)
        acceptance=summary.get('shot_acceptance',{})
        control=np.array([r['info']['control_thread_wall_s'] for r in rows if 'control_thread_wall_s' in r['info']])
        row=dict(case=name,status='verified' if rows else 'not_executed',single_candidate=one,
            generated=sum(e.get('generated_proposals',0) or 0 for e in sets),checked=len(evaluations),
            accepted_candidates=len(cards),installed=summary.get('installed_plans'),
            completed=summary['completed'],shot_accepted=acceptance.get('accepted',False),
            failed_checks=acceptance.get('failed_checks'),failure=summary.get('failure'),
            official_models=models,api_requests=len(api),
            network_unverified_attempts=sum(a.get('status')=='transport_unverified_retry' for a in api),
            invalid_protocol_replies=sum(a.get('status')=='invalid_protocol_reply' for a in api),
            planner_wall_s=[e.get('planner_wall_s') for e in sets],
            control_wall_s=(dict(p50=float(np.percentile(control,50)),p95=float(np.percentile(control,95)),
                p99=float(np.percentile(control,99)),max=float(control.max()),over_20ms=int(sum(control>.02))) if len(control) else None))
        if independent and rows:
            from audit_state_search import audit
            full=audit(folder)
            save_json(folder/'independent_audit.json',full)
            row['independent_safety_pass']=full['independent_safety_pass']
            row['min_truth_swept_clearance_m']=full['min_truth_swept_clearance_m']
        row['stage_one_pass']=bool(one and summary['completed'] and acceptance.get('accepted') and
            models==['jev-1.14.0'] and (not independent or row.get('independent_safety_pass')))
        result.append(row)
    return dict(scope='CPU kinematic closed loop; all original shot gates retained',cases=result,
                passed=sum(r.get('stage_one_pass',False) for r in result),required=len(NAMES),
                all_passed=len(result)==len(NAMES) and all(r.get('stage_one_pass') for r in result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--independent',action='store_true')
    a=p.parse_args()
    result=analyze(a.runs,a.independent)
    save_json(a.runs/'directed_analysis.json',result)
    print(json.dumps(result,indent=2))
