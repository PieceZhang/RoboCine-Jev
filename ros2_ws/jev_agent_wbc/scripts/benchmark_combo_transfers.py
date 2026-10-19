#!/usr/bin/env python3
"""Freeze and run independent free-camera connector cases with official Jev."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def execute(command,log):
    with log.open('w') as handle:r=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
    return dict(case=command[command.index('--case')+1],returncode=r.returncode)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenarios',type=Path,required=True);p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--cases',nargs='+');p.add_argument('--workers',type=int,default=2);p.add_argument('--watchdog',type=float,default=45.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=a.cases or sorted(x.name for x in a.scenarios.glob('combo_*') if x.is_dir())
    if any(Path(n).name!=n or not all((a.scenarios/n/f).is_file() for f in ('task.json','next_task.json','runtime_scene.json')) for n in names):p.error('invalid transfer case')
    sys.path.insert(0,str(ROOT));from jev_agent_wbc.provenance import source_manifest
    config=dict(cases=names,parallel_cases=a.workers,pace=1,model='jev-1.14.0',api_url='https://api.typesafe.ai/v1/systemone',
        execution_mode='free_camera_configuration_transfer',test_watchdog_s=a.watchdog,deadline_s=None)
    (a.output/'provenance.json').write_text(json.dumps(source_manifest(ROOT,config),indent=2))
    shutil.copytree(ROOT/'jev_agent_wbc',a.output/'frozen_code/jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(a.scenarios,a.output/'frozen_code/scenarios')
    os.environ.update(TYPESAFE_API_KEY_FILE=str(a.key_file.resolve()),PYTHONPATH=str((a.output/'frozen_code').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[]
        for name in names:
            command=[sys.executable,'-m','jev_agent_wbc.free_transfer','--case',name,'--scenarios',str((a.output/'frozen_code/scenarios').resolve()),
                '--output',str(a.output),'--api-url',config['api_url'],'--model',config['model'],'--watchdog',str(a.watchdog)]
            futures.append(pool.submit(execute,command,a.output/(name+'.log')))
        for f in as_completed(futures):
            result=f.result();results.append(result);print(json.dumps(result),flush=True)
            (a.output/'process_results.json').write_text(json.dumps(results,indent=2))
    return 0 if all(r['returncode']==0 for r in results) else 2


if __name__=='__main__':raise SystemExit(main())
