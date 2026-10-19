#!/usr/bin/env python3
"""Summarize local closed-loop outcomes by task and phase profile.

Only previously executed camera/control outcomes are exported. No obstacle
keyframes, future states, or simulator geometry are exported. This local report
has no network or Agent-input integration.
"""
import argparse
import json
from pathlib import Path


def collect(root):
    grouped={}
    for path in sorted(root.glob('*/*/*_jev/summary.json')):
        program_path=path.parent/'phase_program.json'
        if not program_path.exists():continue
        summary=json.loads(path.read_text());program=json.loads(program_path.read_text())
        acceptance=summary.get('shot_acceptance')
        if acceptance is None:continue
        if len(program['selections'])!=1:continue
        rate=next(iter(program['selections'].values()));shape=next(iter(program.get('shapes',{'shape':'original'}).values()))
        profile=rate+'__'+shape
        checks=acceptance['checks'];curve=checks.get('natural_curve',{}).get('measurements',{})
        provenance=path.parents[2]/'provenance.json'
        source=json.loads(provenance.read_text()) if provenance.exists() else {}
        row=dict(profile=profile,completed=summary['completed'],accepted=acceptance['accepted'],
            failed_checks=acceptance['failed_checks'],executed_s=summary['elapsed_shot_s'],
            failure=summary['failure'],total_turn_rad=curve.get('total_turn_rad'),
            max_step_turn_rad=curve.get('max_step_turn_rad'),
            source_run=path.parents[2].name,source_sha256=source.get('source_sha256'),
            created_at_utc=source.get('created_at_utc',''))
        grouped.setdefault(summary['original_sha256'],{}).setdefault(profile,[]).append(row)
    result={}
    for task,profiles in grouped.items():
        result[task]=[]
        for profile,rows in profiles.items():
            rows.sort(key=lambda r:r['created_at_utc'])
            passed=[r for r in rows if r['accepted'] and r['completed']]
            selected=([passed[-1]] if passed else [])+rows[-2:]
            unique={r['source_run']:r for r in selected}
            result[task].extend(unique.values())
    return dict(format='phase_execution_feedback/v1',tasks=result,
        scope='Past execution outcomes; controller versions can differ. Re-screen and revalidate every new execution. No future obstacle trajectories.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=collect(args.runs)
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(dict(tasks=len(result['tasks']),output=str(args.output))))


if __name__=='__main__':main()
