#!/usr/bin/env python3
"""Propagate validation result and bound Isaac Kit's post-run shutdown only."""
from pathlib import Path
import argparse,json,subprocess,sys,time,signal
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
if a.result.exists():p.error('result directory already exists; choose a fresh path: '+str(a.result))
if not command:p.error('child command is required')
child=subprocess.Popen(command)
finished_at=None;terminated=False
try:
    while child.poll() is None:
        if (a.result/'run_complete.json').is_file() or (a.result/'error.txt').is_file():
            if finished_at is None:finished_at=time.monotonic()
            if time.monotonic()-finished_at>15 and not terminated:
                print('Kit shutdown exceeded 15s after completion/failure evidence; terminating only owned child',child.pid,flush=True)
                child.terminate();terminated=True
            if terminated and time.monotonic()-finished_at>25:child.kill()
        time.sleep(0.2)
except KeyboardInterrupt:
    child.send_signal(signal.SIGINT);child.wait(timeout=10);raise SystemExit(130)
if (a.result/'error.txt').is_file():
    print('FAILED: original child error retained at '+str(a.result/'error.txt'),file=sys.stderr)
    raise SystemExit(1)
summary=a.result/'summary.json'
if not summary.is_file():raise SystemExit('FAILED: simulation did not produce '+str(summary))
marker=a.result/'run_complete.json'
if not marker.is_file():raise SystemExit('FAILED: child did not finish this run: '+str(marker))
completion=json.loads(marker.read_text())
if completion.get('returncode')!=0:raise SystemExit('FAILED: validator reported '+str(completion))
if child.returncode!=0 and not (terminated and child.returncode in (-signal.SIGTERM,-signal.SIGKILL)):
    raise SystemExit('FAILED: unexpected child exit '+str(child.returncode))
s=json.loads(summary.read_text())
(a.result/'shutdown.json').write_text(json.dumps({'child_returncode':child.returncode,'terminated_after_physics_finished':terminated},indent=2)+'\n')
passed=s['stable_multidirection_pass'] if s['full_suite'] else s.get('selected_commands_pass',False) or (s.get('diagnostic_prefix',False) and s['complete'] and s['stability_pass'] and not s['failures'])
raise SystemExit(0 if passed else 2)
