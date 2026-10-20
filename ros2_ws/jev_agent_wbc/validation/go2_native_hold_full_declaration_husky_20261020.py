#!/usr/bin/env python3
"""Only add the already-existing native7 parking hold to the same full profile."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous=ROOT/'Robot_Go2/results/world_z_full_preparation_01'
    eligibility=ROOT/'Robot_Go2/results/parking_hold_candidate_preparation_01'
    output=ROOT/'Robot_Go2/results/native_hold_full_preparation_01'
    cpu=json.loads((eligibility/'cpu_replay.json').read_text());assert cpu['cpu_gate_pass']
    declaration=json.loads((previous/'declaration.json').read_text())
    run=ROOT/'Robot_Go2/results/adaptive_suite_world_z_01'
    source_match={name:sha(ROOT/'Robot_Go2'/name)==sha(run/'sources'/name) for name in
        ['adaptive_runner.py','camera_coordinator.py','optical_sampler.py','world_z_feedback.py']}
    assert all(source_match.values())
    output.mkdir(exist_ok=False);shutil.copytree(previous/'sources',output/'sources')
    shutil.copy2(Path(__file__),output/'sources'/Path(__file__).name)
    declaration['schema']='go2_native_parking_hold_full_declaration/v1'
    command=declaration['command'];command.insert(command.index('--rgb'),'--parking-camera-hold')
    command[-1]='Robot_Go2/results/adaptive_suite_native_hold_01'
    declaration.update(command=command,parking_camera_hold=True,source_equal_to_previous_full=source_match,
        change='Only enable existing native7 parking hold; original capture/release hysteresis and all18 learned actions unchanged. No source changes.',
        original_capture=dict(position_m=.030,orientation_rad=.09,base_linear_mps=.02,base_yaw_radps=.03),
        original_release=dict(position_m=.040,orientation_rad=.12,base_position_m=.020,base_rotation_rad=.04),
        known_risk='Frozen trace eligibility does not predict physical improvement. Parking hold rolls back subsequent world-Z integral updates and may retain larger height bias. Original Task/errors and all pass gates remain unchanged.',
        frozen_sha256={str(path.relative_to(output/'sources')):sha(path) for path in (output/'sources').rglob('*')if path.is_file()},
        cpu_evidence_sha256={str((eligibility/name).relative_to(ROOT)):sha(eligibility/name)for name in ['cpu_replay.json','declaration.json']},
        physical_full_verified=False,readiness_changed=False)
    declaration['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    path=output/'declaration.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    print(path);print(sha(path));print(' '.join(command))


if __name__=='__main__':main()
