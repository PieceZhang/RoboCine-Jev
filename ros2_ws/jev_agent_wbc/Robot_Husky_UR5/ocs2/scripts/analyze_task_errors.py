#!/usr/bin/env python3
"""Recompute RMS, pointwise intervals and final error from preserved raw poses."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE))
from jev_agent_wbc.wbc_protocol import TimedTask,optical_camera


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve previous analysis: '+str(args.output))
    task=TimedTask.from_dict(json.loads((args.run/'task.json').read_text()))
    measured=[];final=None
    for line in (args.run/'samples.jsonl').open():
        row=json.loads(line)
        if 'task_feedback' not in row:continue
        target=optical_camera(task.trajectory.sample(min(row['stamp_s'],task.trajectory.end_s)))
        actual=np.array(row['T_world_camera_optical'])
        item=dict(stamp_s=row['stamp_s'],task_relative_s=row['stamp_s']-task.trajectory.start_s,
                  position_error_m=float(np.linalg.norm(actual[:3,3]-target.translation_m)),
                  orientation_error_rad=float(Rotation.from_matrix(actual[:3,:3]@target.rotation.T).magnitude()))
        final=item
        if row['case']=='ocs2_camera':measured.append(item)
    bad=[r for r in measured if r['position_error_m']>task.position_tolerance_m];groups=[]
    for row in bad:
        if not groups or row['stamp_s']-groups[-1][-1]['stamp_s']>.010001:groups.append([])
        groups[-1].append(row)
    result=dict(scope='Post hoc original public Task pose errors; RMS is not a claim that every point is within tolerance',sample_count=len(measured),position_tolerance_m=task.position_tolerance_m,orientation_tolerance_rad=task.orientation_tolerance_rad,
                position_rmse_m=float(np.sqrt(np.mean([r['position_error_m']**2 for r in measured]))),
                position_max_m=max(r['position_error_m'] for r in measured),
                position_exceeding_samples=len(bad),position_exceeding_fraction=len(bad)/len(measured),
                position_exceeding_sample_duration_s=.01*len(bad),orientation_exceeding_samples=sum(r['orientation_error_rad']>task.orientation_tolerance_rad for r in measured),
                intervals=[dict(start_s=g[0]['stamp_s'],end_s=g[-1]['stamp_s'],start_task_relative_s=g[0]['task_relative_s'],end_task_relative_s=g[-1]['task_relative_s'],samples=len(g),peak_m=max(r['position_error_m'] for r in g)) for g in groups],
                final=final,final_task_tolerance_pass=bool(final['position_error_m']<task.position_tolerance_m and final['orientation_error_rad']<task.orientation_tolerance_rad))
    files=[Path(__file__),args.run/'samples.jsonl',args.run/'task.json',args.run/'acceptance.json']
    result['files_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
