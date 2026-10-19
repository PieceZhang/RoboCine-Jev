#!/usr/bin/env python3
"""Keep an Isaac/Kit startup exit distinct from a completed validation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output", type=Path)
known, _ = parser.parse_known_args()
arguments = sys.argv[1:]
if "--help" in arguments or "-h" in arguments:
    raise SystemExit(subprocess.call([sys.executable, str(ROOT / "scripts/validate_wbc.py"), *arguments]))
output = known.output
if output is None:
    output = ROOT / "results" / f"run_{time.time_ns()}_{os.getpid()}"
    arguments += ["--output", str(output)]
output = output.resolve()
if output.exists():
    raise SystemExit(f"Refusing to overwrite existing results: {output}")
result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_wbc.py"), *arguments])
if result.returncode:
    raise SystemExit(result.returncode)
try:
    summary = json.loads((output / "summary.json").read_text())
    complete = summary["run_complete"] and summary["cases"] and all(case["complete"] and case["finite"] for case in summary["cases"])
except (OSError, KeyError, ValueError):
    complete = False
if not complete:
    raise SystemExit("Isaac/Kit exited before writing a complete measured run; validation is not accepted.")
raise SystemExit(0 if summary["all_pass"] else 2)
