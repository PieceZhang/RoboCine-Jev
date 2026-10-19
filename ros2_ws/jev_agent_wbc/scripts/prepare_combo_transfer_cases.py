#!/usr/bin/env python3
"""Independent free-camera connector fixtures, with authored state boundaries.

No endpoint reset occurs within a connector. These cases do reset between
connectors and therefore cannot establish continuous combo success.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenarios',type=Path,default=Path(__file__).resolve().parents[1]/'scenarios')
    p.add_argument('--main-runs',type=Path,help='Use measured terminal states/velocities from successful independent main-action runs.')
    p.add_argument('--cases',nargs='+',default=['combo_01_easy','combo_02_easy','combo_03_medium','combo_04_medium'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);manifest=[]
    for name in a.cases:
        if Path(name).name!=name:p.error('invalid case name')
        source=a.scenarios/name;task=json.loads((source/'task.json').read_text())
        runtime=json.loads((source/'runtime_scene.json').read_text())
        nominal=json.loads((a.scenarios/'video_combo'/f'{name}_nominal.json').read_text())
        hashes={Path(k).name:v for k,v in nominal['source_sha256'].items()}
        if hashes['task.json']!=hashlib.sha256((source/'task.json').read_bytes()).hexdigest():raise ValueError('nominal boundary source mismatch')
        for i,(before,after) in enumerate(zip(nominal['actions'],nominal['actions'][1:]),1):
            start=before['phase']['end_s'];end=after['phase']['end_s'];next_start=after['phase']['start_s']
            case=name+f'__transfer_{i:02d}';folder=a.output/case;folder.mkdir()
            scene=deepcopy(runtime);scene['name']=case;scene['initial_state']=before['states'][-1]
            scope='independent connector from authored previous-end to next-start configuration; not continuous combo execution'
            boundary_source=None
            if a.main_runs is not None:
                action=name+f'__action_{i:02d}_'+before['phase']['motion']
                source_run=a.main_runs/action/(action+'_jev')
                result=json.loads((source_run/'summary.json').read_text())
                if not result['completed'] or not result['shot_acceptance']['accepted']:raise ValueError('main action did not pass: '+action)
                rows=json.loads((source_run/'telemetry.json').read_text())
                start=before['phase']['start_s']+rows[-1]['t']
                scene['initial_state']=rows[-1]['state'];scene['initial_control']=rows[-2]['command']
                scene['initial_observation_history']=[dict(t=row['t']-rows[-1]['t'],state=row['state']) for j,row in enumerate(rows) if j%5==0 and rows[-1]['t']-.8<=row['t']]
                boundary_source=dict(telemetry=str((source_run/'telemetry.json').resolve()),sha256=hashlib.sha256((source_run/'telemetry.json').read_bytes()).hexdigest())
                scope='independent boundary replay from measured successful main-action terminal state and velocity; not continuous combo execution'
            for obstacle in scene['obstacles']:
                for knot in obstacle['keyframes']:knot['t']-=start
            next_task=deepcopy(task);next_task['shot_phase']=[dict(after['phase'],start_s=0.,end_s=end-next_start)]
            next_task['reference_camera_trajectory']['samples']=[dict(r,t=round(r['t']-next_start,8)) for r in task['reference_camera_trajectory']['samples'] if next_start<=r['t']<=end]
            goal=dict(format='free_combo_transfer/v1',name=case,goal_state=after['states'][0],
                next_motion=after['phase']['motion'],source_phase=before['phase'],next_phase=after['phase'],
                source_task_sha256=hashes['task.json'],physical_time_offset_s=start,
                boundary_source=boundary_source,
                camera_metrics_required=False,deadline_s=None,
                scope=scope)
            for filename,value in [('task.json',goal),('runtime_scene.json',scene),('next_task.json',next_task)]:
                (folder/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
            manifest.append(goal)
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(' '.join(row['name'] for row in manifest))


if __name__=='__main__':main()
