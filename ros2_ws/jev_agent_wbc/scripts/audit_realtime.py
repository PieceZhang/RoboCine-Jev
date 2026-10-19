#!/usr/bin/env python3
"""Audit deadline/fault evidence separately from cinematic and physical success."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def audit(folder):
    summary = read(folder/'summary.json')
    rows = read(folder/'telemetry.json')
    timing = read(folder/'realtime_timing.json')
    fault = summary.get('actuator_fault')
    events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    api = folder/'jev_api.jsonl'
    requests = [json.loads(line) for line in api.read_text().splitlines()] if api.exists() else []
    uniform = all(abs(b['t']-a['t']-.02)<1e-8 for a,b in zip(rows,rows[1:]))
    stopped = [r for r in rows if fault and r['t'] >= fault['time_s']]
    checks = dict(pace_is_one=summary['pace']==1., no_agent_wait=summary['physics_waits_for_agent'] is False,
        continuous_simulation_samples=uniform,
        world_continues_to_original_end=rows[-1]['t'] >= summary['original_duration_s']+.4-1e-8,
        no_late_restart=all(max(map(abs,r['command']))<1e-12 for r in stopped),
        no_late_install=not any(e['event']=='plan_installed' and fault and e['time_s']>=fault['time_s'] for e in events),
        realtime_success_not_fabricated=(not summary['realtime']['success'] if fault else True))
    return dict(case=summary['case'], checks=checks, watchdog_contract_pass=all(checks.values()),
        fault=fault, emergency_stop_required=summary.get('emergency_stop_required'),
        physics_samples=summary['physical_audit']['samples'], telemetry_samples=len(rows),
        realtime=summary['realtime'], safety_pass=summary['safety_pass'],
        contacts=summary['contacts'], min_truth_clearance_m=summary['min_truth_clearance_m'],
        shot_status=summary['shot_acceptance']['status'], failed_checks=summary['shot_acceptance']['failed_checks'],
        actual_api_records=len(requests), model_responses=sum(bool(r.get('response')) for r in requests),
        installed_plans=summary['installed_plans'],
        mean_stage_wall_s={key:statistics.mean(r[key] for r in timing['stages'])
            for key in timing['stages'][0] if key!='t'} if timing['stages'] else {},
        input_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for name in ('summary.json','telemetry.json','events.jsonl','realtime_timing.json','runtime_events.jsonl')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cases=[audit(p.parent) for p in sorted(args.runs.glob('*/summary.json'))]
    if not cases:
        parser.error('no completed run summaries')
    result=dict(cases=cases, watchdog_contract_pass=all(c['watchdog_contract_pass'] for c in cases),
        realtime_successes=sum(c['realtime']['success'] for c in cases),
        safety_passes=sum(c['safety_pass'] is True for c in cases),
        semantic_calls=sum(c['actual_api_records'] for c in cases),
        model_capability_verified=all(c['model_responses']>0 for c in cases))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':
    main()
