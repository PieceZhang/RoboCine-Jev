#!/usr/bin/env python3
"""Separate main-action experiments from connector experiments.

Each action uses its authored nominal starting configuration and the exact
original obstacle time window. Resets are explicit diagnostic fixtures, never
counted as a continuous combo execution.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenarios',type=Path,default=Path(__file__).resolve().parents[1]/'scenarios')
    parser.add_argument('--cases',nargs='+',default=['combo_01_easy','combo_02_easy','combo_03_medium','combo_04_medium'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for name in args.cases:
        if Path(name).name!=name:parser.error('invalid case name')
        folder=args.scenarios/name
        task=json.loads((folder/'task.json').read_text())
        scene=json.loads((folder/'runtime_scene.json').read_text())
        nominal=json.loads((args.scenarios/'video_combo'/f'{name}_nominal.json').read_text())
        expected={Path(k).name:v for k,v in nominal['source_sha256'].items()}
        if expected['task.json']!=hashlib.sha256((folder/'task.json').read_bytes()).hexdigest():
            raise ValueError('nominal action initialization belongs to a different authored task')
        for action in nominal['actions']:
            phase=action['phase'];a,b=phase['start_s'],phase['end_s']
            local=deepcopy(task)
            local['shot_phase']=[dict(phase,start_s=0.,end_s=b-a)]
            local['reference_camera_trajectory']['samples']=[dict(r,t=round(r['t']-a,8))
                for r in task['reference_camera_trajectory']['samples'] if a<=r['t']<=b]
            runtime=deepcopy(scene);runtime['name']=name+'__'+phase['name']
            runtime['initial_state']=action['states'][0]
            for obstacle in runtime['obstacles']:
                for knot in obstacle['keyframes']:knot['t']-=a
            out=args.output/runtime['name'];out.mkdir()
            for filename,document in [('task.json',local),('runtime_scene.json',runtime)]:
                (out/filename).write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
            record=dict(case=runtime['name'],source_combo=name,original_phase=phase,
                obstacle_physical_time_offset_s=a,initialization=action['start_state_source'],
                scope='independent main-action diagnostic; reset between actions; not a continuous combo result')
            (out/'diagnostic_scope.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            records.append(record)
    (args.output/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(' '.join(r['case'] for r in records))


if __name__=='__main__':main()
