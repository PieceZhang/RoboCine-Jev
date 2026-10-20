#!/usr/bin/env python3
"""Actual native USD CPU audit of the SDK double-floor repair; no Kit/physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    python=Path('/home/w84384651/miniconda3/envs/env_isaaclab/bin/python')
    site=python.parent.parent/'lib/python3.11/site-packages/isaacsim';usd=next((site/'extscache').glob('omni.usd.libs-*'))
    if not os.environ.get('TIAGO_FLOOR_USD_READY'):
        env=dict(os.environ,TIAGO_FLOOR_USD_READY='1')
        env['LD_LIBRARY_PATH']=':'.join([str(python.parent.parent/'lib'),str(usd/'bin'),str(site/'kit'),env.get('LD_LIBRARY_PATH','')])
        os.execve(str(python),[str(python),__file__,*sys.argv[1:]],env)
    sys.path.insert(0,str(usd));from pxr import Usd
    from floor_model import configure_single_ground
    stage=Usd.Stage.Open(str(args.scene));report=configure_single_ground(stage,'/World/Ground','/World/GroundMaterial')
    report.update(scope='actual saved USD; in-memory repair only, original scene bytes preserved; no physical stability inferred',
        source_scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        source_files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'wbc/floor_model.py']})
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
