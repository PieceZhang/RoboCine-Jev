#!/usr/bin/env python3
"""Read-only native USD geometry export; no physics or GPU execution."""
import argparse
import json
import os
from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--robot-dir',type=Path,default=ROOT)
    args = parser.parse_args()
    python = Path(os.environ.get('ISAAC_PYTHON','/home/w84384651/miniconda3/envs/env_isaaclab/bin/python'))
    site = python.resolve().parents[1]/'lib/python3.11/site-packages/isaacsim'
    usd = next((site/'extscache').glob('omni.usd.libs-*'))
    if not os.environ.get('TIAGO_COLLISION_USD_READY'):
        env = dict(os.environ,TIAGO_COLLISION_USD_READY='1')
        env['LD_LIBRARY_PATH'] = ':'.join([str(python.resolve().parents[1]/'lib'),str(usd/'bin'),str(site/'kit'),env.get('LD_LIBRARY_PATH','')])
        os.execve(str(python),[str(python),__file__,*sys.argv[1:]],env)
    sys.path.insert(0,str(usd))
    from pxr import Usd
    from collision_model import extract_collision_geometry
    model=args.robot_dir/'assets/usd/tiago.usd'
    stage = Usd.Stage.Open(str(model))
    metadata = extract_collision_geometry(stage,'/tiago')
    metadata['asset_sha256']=hashlib.sha256(model.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(metadata,stream,indent=2)
    print(json.dumps(dict(shapes=len(metadata['shapes']),allowed_pairs=len(metadata['allowed_pairs']))))


if __name__=='__main__':
    main()
