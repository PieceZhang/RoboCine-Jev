#!/usr/bin/env python3
"""Audit a native suite without changing recorded decisions or acceptance files."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.multi_robot_audit import audit_run

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--filename',default='independent_audit.json')
a=p.parse_args()
directories=[a.directory] if (a.directory/'avoidance').is_dir() else [
    f.parent.parent for f in sorted(a.directory.glob('*/avoidance/summary.json'))]
for directory in directories:
    try:
        r=audit_run(directory,a.filename)
        print(json.dumps({k:r.get(k) for k in ('robot','case','mode','complete','physical_acceptance',
            'exact_agent_plan_installed','successful','timing_pass')}),flush=True)
    except Exception as exc:
        print(json.dumps(dict(directory=str(directory),error=f'{type(exc).__name__}: {exc}')),flush=True)
