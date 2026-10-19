#!/usr/bin/env python3
"""Frozen-source official Jev tests with continuous robot and obstacle state."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def execute(command,log):
    with log.open('w') as stream:
        try:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200);code=r.returncode
        except subprocess.TimeoutExpired:code=124
    return dict(case=command[command.index('--case')+1],returncode=code)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenarios',type=Path,default=ROOT/'scenarios');p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['combo_01_easy','combo_02_easy','combo_03_medium','combo_04_medium','combo_05_hard','combo_06_hard'])
    p.add_argument('--workers',type=int,default=2);p.add_argument('--watchdog',type=float,default=35.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if any(Path(name).name!=name or not(a.scenarios/name/'task.json').exists() for name in a.cases):p.error('invalid case')
    sys.path.insert(0,str(ROOT));from jev_agent_wbc.provenance import source_manifest
    config=dict(cases=a.cases,parallel_cases=a.workers,pace=1,model='jev-1.14.0',api_url='https://api.typesafe.ai/v1/systemone',continuous=True,test_watchdog_s=a.watchdog)
    (a.output/'provenance.json').write_text(json.dumps(source_manifest(ROOT,config),indent=2))
    frozen=a.output/'frozen_code';shutil.copytree(ROOT/'jev_agent_wbc',frozen/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
    # Only source tasks and nominal configurations are needed; preserve hashes
    # in the manifest without copying historical scenario films every iteration.
    for name in a.cases:shutil.copytree(a.scenarios/name,frozen/'scenarios'/name)
    (frozen/'scenarios/video_combo').mkdir(parents=True,exist_ok=True)
    for path in (a.scenarios/'video_combo').glob('*_nominal.json'):shutil.copy2(path,frozen/'scenarios/video_combo'/path.name)
    os.environ.update(TYPESAFE_API_KEY_FILE=str(a.key_file.resolve()),PYTHONPATH=str(frozen.resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[]
        for name in a.cases:
            command=[sys.executable,'-m','jev_agent_wbc.continuous_combo','--case',name,'--scenarios',str((frozen/'scenarios').resolve()),'--output',str(a.output),'--watchdog',str(a.watchdog)]
            futures.append(pool.submit(execute,command,a.output/(name+'.log')))
        for f in as_completed(futures):
            result=f.result();results.append(result);print(json.dumps(result),flush=True)
            (a.output/'process_results.json').write_text(json.dumps(results,indent=2))
    return 0 if all(r['returncode']==0 for r in results) else 2


if __name__=='__main__':raise SystemExit(main())
