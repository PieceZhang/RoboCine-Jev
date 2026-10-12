#!/usr/bin/env python3
"""Identical-scene Isaac comparisons. Faults are reported, never omitted."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',default=['moveout','combo'])
    p.add_argument('--controllers',nargs='+',choices=['none','per_link','kc'],default=['none','per_link','kc'])
    p.add_argument('--timeout',type=float,default=900.)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    index=args.output/'suite.json';results=json.loads(index.read_text()) if index.exists() else {}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for name in args.scenes:
        scene=root/'scenes'/f'{name}.json';data=json.loads(scene.read_text())
        for mode in args.controllers:
            key=f'{name}_{mode}';folder=args.output/key;folder.mkdir(exist_ok=False)
            cmd=[sys.executable,str(root/'scripts/jvs_isaac_bootstrap.py'),'--isaac-python',
                 '/home/w84384651/miniconda3/envs/env_isaaclab/bin/python','--runner',str(root/'scripts/test_wbc_isaac.py'),'--',
                 '--xacro',str(root/'urdf/jvs_mock.urdf.xacro'),'--headless','true','--enable-lidar','false',
                 '--solver-type','PGS','--trajectory',str(root/'trajectories'/data['trajectory']),
                 '--obstacles',str(scene),'--output',str(folder.resolve()),'--controller',mode,
                 '--initial-q',*map(str,data['initial_q'])]
            with (folder/'isaac.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    code=124
            report=folder/'summary.json'
            summary=json.loads(report.read_text()) if report.exists() else dict(completed=False,failure='no report')
            results[key]=dict(returncode=code,summary=summary)
            index.write_text(json.dumps(results,indent=2)+'\n')
            print(key,summary.get('completed'),summary.get('failure'),summary.get('collision_audit'),flush=True)
    return 0 if all(r['summary'].get('completed') for r in results.values()) else 2

if __name__=='__main__':raise SystemExit(main())
