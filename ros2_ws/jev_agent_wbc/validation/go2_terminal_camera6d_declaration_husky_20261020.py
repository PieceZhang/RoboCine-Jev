#!/usr/bin/env python3
"""Freeze original meaningful sixteen-second6D task using complete-suite profile."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'Robot_Go2/results/terminal_full_preparation_01'
FULL=ROOT/'Robot_Go2/results/adaptive_suite_terminal_01'
OUTPUT=ROOT/'Robot_Go2/results/terminal_camera6d_preparation_01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    declaration=json.loads((PREP/'declaration.json').read_text());summary=json.loads((FULL/'summary.json').read_text())
    assert summary['all_pass'] is True and summary['run_complete'] is True
    assert all(sha(ROOT/name)==value for name,value in declaration['source_sha256'].items())
    matches={name:sha(ROOT/'Robot_Go2'/name)==sha(FULL/'sources'/name)for name in
        ['adaptive_runner.py','camera_coordinator.py','terminal_allocator.py','world_z_feedback.py','optical_sampler.py']}
    assert all(matches.values())
    OUTPUT.mkdir(exist_ok=False);shutil.copytree(PREP/'sources',OUTPUT/'sources')
    shutil.copy2(Path(__file__),OUTPUT/'sources'/Path(__file__).name)
    command=declaration['command'].copy()
    command[command.index('--cases')+1]='camera_6d';command[command.index('--move-seconds')+1]='16'
    command[command.index('--camera-excursion')+1]='nontrivial'
    command[-1]='Robot_Go2/results/adaptive_camera6d_terminal_01'
    declaration.update(schema='go2_terminal_camera6d_declaration/v1',command=command,cases=['camera_6d'],move_s=16,total_sim_s=23,
        source_equal_to_complete_suite=matches,
        physical_validation_scope='Original meaningful6D camera trajectory: all3translation/3rotation axes,16s immutable WORLD cameraTask,5s original stop and2s settle. Original zero preferred base input; subordinate native posture and measured terminal allocation retain full learned18 authority.',
        known_risk='Complete nine-case passing does not alone certify meaningful6D trajectory or stop under larger angular/translation perturbations. Exact original6D task/tolerances and all original constraints remain authoritative.',
        full_evidence_sha256={str((FULL/name).relative_to(ROOT)):sha(FULL/name)for name in ['summary.json','provenance.json','timed_tasks.json','camera_render_check.json']},
        frozen_sha256={str(path.relative_to(OUTPUT/'sources')):sha(path)for path in(OUTPUT/'sources').rglob('*')if path.is_file()},
        physical_verified=False,formal_readiness_changed=False)
    declaration['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    path=OUTPUT/'declaration.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    print(path);print(sha(path));print(' '.join(command))


if __name__=='__main__':main()
