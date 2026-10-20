#!/usr/bin/env python3
"""Fetch immutable upstream sources without modifying an existing checkout."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def git(*args,cwd=None):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-legacy',action='store_true',help='Optional source audit only; not installed')
    args = parser.parse_args()
    lock = json.loads((ROOT/'FRAMEWORK_SOURCE_LOCK.json').read_text())
    (ROOT/'vendor').mkdir(exist_ok=True)
    for entry in lock['repositories']:
        if entry['name']!='cognitive_robot_abstract_machine' and not args.include_legacy:
            continue
        path = ROOT/'vendor'/entry['name']
        if not path.exists():
            git('clone','--filter=blob:none','--no-checkout',entry['repository'],str(path))
            if entry['name']=='cognitive_robot_abstract_machine':
                git('sparse-checkout','init','--cone',cwd=path)
                git('sparse-checkout','set','giskardpy','krrood','semantic_digital_twin',cwd=path)
            git('checkout','--detach',entry['commit'],cwd=path)
        current = git('rev-parse','HEAD',cwd=path)
        modified = git('status','--porcelain','--untracked-files=no',cwd=path)
        if current!=entry['commit'] or modified:
            raise RuntimeError(f'{path}: unexpected commit or modified tracked files; refusing reset')
        print(f'{entry["name"]} {current} verified')


if __name__=='__main__':
    main()
