#!/usr/bin/env python3
"""Declare the same exact physical controller for all original nine cases."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'Robot_Go2/results/lateral_stand_brake_preparation_02'
SINGLE=ROOT/'Robot_Go2/results/adaptive_yaw_left_lateral_brake_01'
OUT=ROOT/'Robot_Go2/results/lateral_stand_brake_full_preparation_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=json.loads((PRIOR/'declaration.json').read_text());s=json.loads((SINGLE/'summary.json').read_text());assert s['all_pass']is True and s['run_complete']is True
 assert all(sha(ROOT/name)==value for name,value in d['source_sha256'].items())
 provenance=json.loads((SINGLE/'provenance.json').read_text())
 for name in ['adaptive_runner.py','lateral_stand_brake.py','bounded_body_prediction.py','world_z_bumpless.py','world_z_feedback.py','camera_coordinator.py']:
  assert sha(ROOT/'Robot_Go2'/name)==provenance['frozen_sha256'][name]
 OUT.mkdir(exist_ok=False);shutil.copytree(PRIOR/'sources',OUT/'sources');shutil.copy2(Path(__file__),OUT/'sources'/Path(__file__).name)
 cases=['forward','backward','left','right','diagonal_left','diagonal_right','yaw_left','yaw_right','camera_shot']
 argv=d['command'].copy();argv[argv.index('--cases')+1]=','.join(cases);argv[-1]='Robot_Go2/results/adaptive_suite_lateral_brake_01'
 d.update(schema='go2_lateral_stand_brake_full_declaration/v1',command=argv,cases=cases,total_sim_s=92,
  exact_same_runtime_as_single_yaw_left=True,single_positive_scope='Only isolated yawL independently reviewed separately; no combination with previous7/9 into a fullpass.',
  existing_single_result_sha256={str(p.relative_to(ROOT)):sha(p)for p in [SINGLE/'summary.json',SINGLE/'provenance.json',SINGLE/'measurements.jsonl',SINGLE/'timed_tasks.json']},
  physical_verified=False,formal_readiness_changed=False)
 d['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 d['frozen_sha256']={str(p.relative_to(OUT/'sources')):sha(p)for p in(OUT/'sources').rglob('*')if p.is_file()}
 d['known_limitations'][-1]='Original continuous nine cases and meaningful6D must both pass with this exact current profile and independent audits; formal3 control gates remainFalse.'
 p=OUT/'declaration.json';p.write_text(json.dumps(d,indent=2)+'\n');print(p);print(sha(p));print(' '.join(argv))
if __name__=='__main__':main()
