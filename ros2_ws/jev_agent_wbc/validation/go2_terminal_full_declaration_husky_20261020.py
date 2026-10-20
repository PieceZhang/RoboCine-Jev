#!/usr/bin/env python3
"""Freeze unchanged yaw-probed terminal profile for original complete nine cases."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'Robot_Go2/results/terminal_yaw_preparation_01'
PROBE=ROOT/'Robot_Go2/results/adaptive_yaw_left_terminal_01'
OUTPUT=ROOT/'Robot_Go2/results/terminal_full_preparation_01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous=json.loads((PREP/'declaration.json').read_text());probe=json.loads((PROBE/'summary.json').read_text())
    assert probe['all_pass']is True and probe['run_complete']is True
    assert all(sha(ROOT/name)==value for name,value in previous['source_sha256'].items())
    for path in (PREP/'sources/jev_agent_wbc').rglob('*.py'):
        assert sha(path)==sha(ROOT/'jev_agent_wbc'/path.relative_to(PREP/'sources/jev_agent_wbc'))
    probe_provenance=json.loads((PROBE/'provenance.json').read_text())
    matches={name:sha(ROOT/'Robot_Go2'/name)==probe_provenance['frozen_sha256'][name]for name in
        ['adaptive_runner.py','camera_coordinator.py','terminal_allocator.py','world_z_feedback.py','optical_sampler.py']}
    assert all(matches.values())
    OUTPUT.mkdir(exist_ok=False);shutil.copytree(PREP/'sources',OUTPUT/'sources')
    shutil.copy2(Path(__file__),OUTPUT/'sources'/Path(__file__).name)
    cases='forward,backward,left,right,diagonal_left,diagonal_right,yaw_left,yaw_right,camera_shot'
    command=previous['command'].copy();command[command.index('--cases')+1]=cases
    command[-1]='Robot_Go2/results/adaptive_suite_terminal_01'
    previous.update(schema='go2_terminal_full_declaration/v1',command=command,cases=cases.split(','),total_sim_s=92,
        diagnostic_only=False,source_equal_to_yaw_probe=matches,
        physical_validation_scope='Original complete9case5move+5stop with one initial2s settle. Each new WORLD TimedTask starts from its actual measured case boundary; original actual initialZ accumulation retained, no within-task retargeting.',
        known_risk='One initial yaw-left passing does not certify other directions or accumulated state. Native18 stability and all original parking gates must pass all nine cases independently; no hold/threshold changes.',
        frozen_sha256={str(path.relative_to(OUTPUT/'sources')):sha(path)for path in(OUTPUT/'sources').rglob('*')if path.is_file()},
        probe_evidence_sha256={str((PROBE/name).relative_to(ROOT)):sha(PROBE/name)for name in
            ['summary.json','provenance.json','timed_tasks.json','camera_render_check.json']},
        physical_verified=False,formal_readiness_changed=False)
    previous['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    path=OUTPUT/'declaration.json';path.write_text(json.dumps(previous,indent=2)+'\n')
    print(path);print(sha(path));print(' '.join(command))


if __name__=='__main__':main()
