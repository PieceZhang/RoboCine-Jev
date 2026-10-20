#!/usr/bin/env python3
"""Freeze final original-slew brake priority after any later body recovery."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'Robot_Go2/results/terminal_pose_full_preparation_02'
PROOF=ROOT/'Robot_Go2/results/terminal_pose_final_priority_cpu_01'
OUT=ROOT/'Robot_Go2/results/terminal_pose_full_preparation_03'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    declaration=json.loads((PREVIOUS/'declaration.json').read_text());proof=json.loads((PROOF/'equivalence.json').read_text())
    assert proof['passed']is True and proof['cpu_tests_passed']==136
    assert all(sha(ROOT/name)==value for name,value in proof['files_sha256'].items())
    mismatches={name:dict(previous=value,current=sha(ROOT/name))for name,value in declaration['source_sha256'].items()if sha(ROOT/name)!=value}
    assert {'Robot_Go2/adaptive_runner.py','test/test_wbc_go2_terminal_integration.py'}<=set(mismatches)
    assert set(mismatches)<={'Robot_Go2/adaptive_runner.py','test/test_wbc_go2_terminal_integration.py','Robot_G1/wbc_backend.py'}
    for path in(PREVIOUS/'sources/jev_agent_wbc').rglob('*.py'):
        assert sha(path)==sha(ROOT/'jev_agent_wbc'/path.relative_to(PREVIOUS/'sources/jev_agent_wbc'))
    OUT.mkdir(exist_ok=False);shutil.copytree(PREVIOUS/'sources',OUT/'sources')
    files=[ROOT/'Robot_Go2/adaptive_runner.py',ROOT/'test/test_wbc_go2_terminal_integration.py',
        ROOT/'validation/go2_terminal_final_priority_equivalence_husky_20261020.py',Path(__file__)]
    for path in files:shutil.copy2(path,OUT/'sources'/path.name)
    shutil.copy2(ROOT/'Robot_G1/wbc_backend.py',OUT/'sources/g1_backend.py')
    declaration['source_sha256'].update({name:sha(ROOT/name)for name in declaration['source_sha256']})
    declaration['source_sha256'].update({str(path.relative_to(ROOT)):sha(path)for path in files})
    evidence=[PROOF/'equivalence.json',PROOF/'cpu_tests.xml',
        ROOT/'Robot_TIAGo/wbc/results/go2_v2_prediction_ode01/summary.json',
        ROOT/'Robot_TIAGo/wbc/results/go2_v2_prediction_ode01/sources/audit_go2_v2_prediction.py']
    declaration['cpu_evidence_sha256'].update({str(path.relative_to(ROOT)):sha(path)for path in evidence})
    declaration.update(schema='go2_terminal_pose_full_declaration/v3',cpu_tests_passed=136,cpu_tests_skipped=0,
        final_rejected_context_body_priority='After any later native-domain recovery/FCL result, terminal_error or latched deadlinefailure forces original safe_slew toward zero. Legal native7 recovery retained, publicFalse and ZI rollback retained. No unknown forecast can restart body motion.',
        current_source_change_from_previous_declaration=mismatches,
        normal_path='Verified same context.guard_applied on every fresh normal terminal route; one diagnosticFalse field added. Coordinator and all prediction/integral/sampler helpers unchanged.',
        frozen_sha256={str(path.relative_to(OUT/'sources')):sha(path)for path in(OUT/'sources').rglob('*')if path.is_file()},
        physical_verified=False,formal_readiness_changed=False)
    p=OUT/'declaration.json';p.write_text(json.dumps(declaration,indent=2)+'\n');print(p);print(sha(p));print(' '.join(declaration['command']))


if __name__=='__main__':main()
