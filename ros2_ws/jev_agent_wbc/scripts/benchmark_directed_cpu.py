#!/usr/bin/env python3
"""Replay saved official construction decisions to profile CPU numerics only.

This makes no Jev calls and is NOT new semantic or closed-loop acceptance.
Inputs, decision provenance and the numerical source snapshot are preserved.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,nargs='+',required=True,help='Saved benchmark_official_jev run directories')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/'jev_agent_wbc',a.output/'frozen_code/jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    sys.path.insert(0,str((a.output/'frozen_code').resolve()))
    import numpy as np
    from jev_agent_wbc.reference import OriginalReference
    from jev_agent_wbc.task_space_planning import build
    from jev_agent_wbc.candidates import SearchBudget
    from jev_agent_wbc.whole_body_qp import Limits
    from jev_agent_wbc.scenarios import RuntimeScene,NAMES
    from jev_agent_wbc.sensors import RuntimeLidar
    from jev_agent_wbc.evaluation import save_json
    from jev_agent_wbc.provenance import source_manifest
    save_json(a.output/'provenance.json',source_manifest(ROOT,dict(
        mode='saved_official_decisions_numerical_replay_no_new_api_or_execution',sources=[str(s.resolve()) for s in a.sources])))
    rows=[]
    for source in a.sources:
        fixture_root=a.output/'inputs'/source.name
        shutil.copytree(source/'frozen_code/scenarios',fixture_root)
        for name in NAMES:
            case=fixture_root/name
            runtime=RuntimeScene(case/'runtime_scene.json');d=runtime.document
            state=np.asarray(d['initial_state']);z=d['base_z']
            original=OriginalReference(case/'task.json',d['subject']['position'])
            events=[json.loads(line) for line in (source/name/(name+'_jev')/'events.jsonl').read_text().splitlines()]
            decision=next(e for e in events if e['event']=='search_configuration')
            plan=decision['construction_plan'];save_json(case/'saved_official_construction.json',decision)
            lidar=RuntimeLidar()
            for stamp in np.linspace(-.8,0.,9):scene=lidar.synthetic_scan(runtime,state,z,float(stamp))
            fy=.5/np.tan(np.deg2rad(27.5));intrinsics=[fy*288/384,fy,.5,.5]
            scene=dict(scene,subject_model=d['subject'],intrinsics=intrinsics)
            result=build(original,original,plan['family'],scene,state,np.zeros(8),0.,z,Limits(),
                budget=SearchBudget(single_candidate=True,candidate_wall_budget_s=30.),measured_control=np.zeros(8),
                commanded_q=state[3:],search_config=decision['search_config'],construction_plan=plan,
                initial_boundary=True,intrinsics=intrinsics)
            candidates=result.pop('candidates',[]);result['cards']=[c.card() for c in candidates]
            save_json(a.output/(source.name+'_'+name+'.json'),result)
            row=dict(source=source.name,case=name,accepted=len(candidates),wall_s=result['wall_s'],
                generated=result.get('generated_proposals'),checked=len(result.get('evaluations',[])),
                failures=[e.get('rejected') for e in result.get('evaluations',[])])
            rows.append(row);save_json(a.output/'summary.json',rows)
            print(json.dumps(row),flush=True)
    return 0 if all(r['accepted']==r['generated']==r['checked']==1 for r in rows) else 2


if __name__=='__main__':raise SystemExit(main())
