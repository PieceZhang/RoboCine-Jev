#!/usr/bin/env python3
"""Same frozen controller, only original source leg damping during parking."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous=ROOT/'Robot_Go2/results/world_z_full_preparation_01'
    output=ROOT/'Robot_Go2/results/native_damping_full_preparation_01'
    run=ROOT/'Robot_Go2/results/adaptive_suite_world_z_01'
    declaration=json.loads((previous/'declaration.json').read_text())
    matches={name:sha(ROOT/'Robot_Go2'/name)==sha(run/'sources'/name)for name in
        ['adaptive_runner.py','camera_coordinator.py','optical_sampler.py','world_z_feedback.py']}
    assert all(matches.values())
    output.mkdir(exist_ok=False);shutil.copytree(previous/'sources',output/'sources')
    shutil.copy2(Path(__file__),output/'sources'/Path(__file__).name)
    command=declaration['command'];index=command.index('--rgb');command[index:index]=['--park-damping-scale','1']
    command[-1]='Robot_Go2/results/adaptive_suite_native_damping_01'
    declaration.update(schema='go2_native_parking_damping_full_declaration/v1',command=command,
        park_damping_scale=1.,parking_camera_hold=False,terminal_allocator_enabled=False,
        source_equal_to_previous_full=matches,
        change='Only --park-damping-scale1: use original source lower actuator PD damping throughout parking instead of historical multiplier3. All native learned18 actions and original effort/velocity/q caps remain. No source modifications or added terminal allocator.',
        known_risk='Native PD1 under this corrected predictor/worldZ/contact profile is physically unverified until this complete run. The prior source-profile6D was park3. Do not combine evidence as a same-profile complete pass without new6D and independent audit.',
        frozen_sha256={str(path.relative_to(output/'sources')):sha(path)for path in (output/'sources').rglob('*')if path.is_file()},
        physical_full_verified=False,readiness_changed=False)
    declaration['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    path=output/'declaration.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    print(path);print(sha(path));print(' '.join(command))


if __name__=='__main__':main()
