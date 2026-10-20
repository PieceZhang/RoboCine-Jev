#!/usr/bin/env python3
"""Freeze corrected rejected-history path; normal bounded-plan evidence retained."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'Robot_Go2/results/terminal_pose_full_preparation_01'
FALLBACK=ROOT/'Robot_Go2/results/terminal_pose_fallback_cpu_01'
OUT=ROOT/'Robot_Go2/results/terminal_pose_full_preparation_02'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous=json.loads((PREVIOUS/'declaration.json').read_text());proof=json.loads((FALLBACK/'equivalence.json').read_text())
    assert proof['passed']is True and proof['cpu_tests_passed']==132
    assert all(sha(ROOT/name)==value for name,value in proof['files_sha256'].items())
    mismatches={name:dict(previous=value,current=sha(ROOT/name))for name,value in previous['source_sha256'].items()if sha(ROOT/name)!=value}
    assert set(mismatches)=={'Robot_Go2/adaptive_runner.py','Robot_Go2/camera_coordinator.py','Robot_G1/wbc_backend.py'}
    for path in (PREVIOUS/'sources/jev_agent_wbc').rglob('*.py'):
        assert sha(path)==sha(ROOT/'jev_agent_wbc'/path.relative_to(PREVIOUS/'sources/jev_agent_wbc'))
    OUT.mkdir(exist_ok=False);shutil.copytree(PREVIOUS/'sources',OUT/'sources')
    files=[ROOT/'Robot_Go2/adaptive_runner.py',ROOT/'Robot_Go2/camera_coordinator.py',
        ROOT/'test/test_wbc_go2_terminal_integration.py',ROOT/'validation/go2_terminal_fallback_equivalence_husky_20261020.py',Path(__file__)]
    for path in files:shutil.copy2(path,OUT/'sources'/path.name)
    # Concurrent G1 work changed its backend, copied by the shared runner but
    # loaded only in robot=='g1'. Keep actual closure honest without presenting
    # it as an executed Go2 algorithm change.
    shutil.copy2(ROOT/'Robot_G1/wbc_backend.py',OUT/'sources/g1_backend.py')
    previous['source_sha256'].update({name:sha(ROOT/name)for name in previous['source_sha256']})
    previous['source_sha256'].update({str(path.relative_to(ROOT)):sha(path)for path in files})
    previous.update(schema='go2_terminal_pose_full_declaration/v2',cpu_tests_passed=132,cpu_tests_skipped=0,
        source_matches_cpu=False,source_matches_normal_cpu_via_exact_dispatch_and_module_equivalence=True,
        current_source_change_from_prior_declaration=mismatches,
        unexecuted_g1_backend_change='Concurrent G1-owned backend updated. The shared runner copies it but loads it only under robot==g1; this declaration runs robot==go2 and loads go2_backend.py. Current G1 bytes are retained for exact future run provenance.',
        fallback='Terminal context construction error skips invalid forecast; fresh original current optical/native conversion and original slew brake only; publicFalse/command_stale or tracking_limit,forecast unavailable,marginNone. Original native correction/recovery/FCL/ZIrollback still execute.',
        unchanged_normal_path='Normal coordinator module AST identical after removing only new reject method. Normal runner AST identical after verified dispatch normalization/removing only helper; exact dense-route output regression passed.',
        frozen_sha256={str(path.relative_to(OUT/'sources')):sha(path)for path in(OUT/'sources').rglob('*')if path.is_file()},
        physical_verified=False,formal_readiness_changed=False)
    for path in [FALLBACK/'equivalence.json',FALLBACK/'cpu_tests.xml']:
        previous['cpu_evidence_sha256'][str(path.relative_to(ROOT))]=sha(path)
    p=OUT/'declaration.json';p.write_text(json.dumps(previous,indent=2)+'\n');print(p);print(sha(p));print(' '.join(previous['command']))


if __name__=='__main__':main()
