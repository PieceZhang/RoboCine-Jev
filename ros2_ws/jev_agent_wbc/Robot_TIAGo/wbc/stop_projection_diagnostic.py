#!/usr/bin/env python3
"""Paced genuine-QP replay of actual native stop proposals; no new physics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'wbc')]
from native_solver import NativeGiskardSolver
from source_state import source_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((args.input/'sources/2_robot.json').read_text())
    metadata=json.loads((args.input/'collision_geometry.json').read_text())
    samples=[json.loads(line) for line in (args.input/'samples.jsonl').open()]
    samples=[row for row in samples if row['case']=='measured_stop'][-51:]
    old=NativeGiskardSolver(spec,metadata);strict=NativeGiskardSolver(spec,metadata)
    old_settings=dict(eps_abs=1e-6,eps_rel=1e-7,eps_duality_gap_abs=1e-5,eps_duality_gap_rel=1e-5)
    for key,value in old_settings.items():setattr(old.solver.solver.settings,key,value)
    names=samples[0]['joint_names'];ids=[names.index(j['name']) for j in spec['joints']]
    rows=[];start=time.monotonic()
    for index,row in enumerate(samples):
        q=np.asarray(row['q']);measured_spec=copy.deepcopy(spec)
        for name in measured_spec['passive_joint_positions']:
            measured_spec['passive_joint_positions'][name]=q[names.index(name)]
        poses,js=source_state(measured_spec,q[ids],args.input/'sources/tiago_physics.urdf')
        base=np.asarray(row['T_world_base'])
        for name in poses:
            poses[name]=base@poses[name]
            js[name][:3]=base[:3,:3]@js[name][:3];js[name][3:]=base[:3,:3]@js[name][3:]
        previous=np.r_[row['command_vw'],row['command_joint_speed']]
        request=dict(q=q[ids],previous=previous,poses=poses,body_jacobians=js,
            mode='actuator',desired_rates=np.asarray(row['native']['upstream_proposal']))
        outputs={}
        for label,solver in [('upstream_defaults',old),('strict_projection',strict)]:
            t=time.perf_counter();answer=solver.solve(request)
            outputs[label]=dict(command=answer['command'].tolist(),wall_s=time.perf_counter()-t,
                hard_constraint_violation=answer['hard_constraint_violation'])
        rows.append(dict(recorded_stamp_s=row['stamp_s'],**outputs))
        remaining=start+(index+1)*.01-time.monotonic()
        if remaining>0:time.sleep(remaining)
    wall=time.monotonic()-start
    maximum=lambda key:float(np.max(np.abs([row[key]['command'] for row in rows])))
    result=dict(schema='tiago_genuine_stop_projection_diagnostic/v1',
        scope='recorded physical native proposals; original URDF FK reconstructed from recorded full14DOF/base; no new physical evidence',
        rows=len(rows),default_settings=old_settings,strict_settings=strict.projection_settings,
        default_max_command=maximum('upstream_defaults'),strict_max_command=maximum('strict_projection'),
        reference_duration_s=len(rows)*.01,wall_s=wall,timing_pass=wall<=len(rows)*.01*1.05,
        passed=bool(maximum('strict_projection')<1e-9 and all(row['strict_projection']['hard_constraint_violation']<1e-6 for row in rows)),
        physical_parking_verified=False,source_files_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),ROOT/'wbc/native_solver.py',ROOT/'wbc/source_state.py']},
        input_samples_sha256=hashlib.sha256((args.input/'samples.jsonl').read_bytes()).hexdigest())
    (args.output/'trace.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
