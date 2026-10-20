#!/usr/bin/env python3
"""Freeze explicit fresh actual-output continuity; original hard limits retained."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'Robot_Go2/results/lateral_stand_brake_full_preparation_01'
CPU=ROOT/'Robot_Go2/results/output_continuity_cpu_01'
OUT=ROOT/'Robot_Go2/results/output_continuity_full_preparation_01'
TESTS=[ROOT/'Robot_Go2/results/output_continuity_tests01.xml',ROOT/'Robot_Go2/results/output_continuity_wiring_tests01.xml']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=json.loads((PRIOR/'declaration.json').read_text());cpu=json.loads((CPU/'summary.json').read_text())
 assert cpu['cpu_gate_pass']is True and cpu['new_eligible_count']==8
 for name,value in cpu['source_sha256'].items():assert sha(ROOT/name)==value
 count=0
 for test in TESTS:
  suites=list(ET.parse(test).getroot().iter('testsuite'));count+=sum(int(s.get('tests','0'))for s in suites)
  assert sum(int(s.get(k,'0'))for s in suites for k in ['failures','errors','skipped'])==0
 assert count==103
 changed={'Robot_Go2/adaptive_runner.py','Robot_Go2/world_z_bumpless.py'}
 for name,value in d['source_sha256'].items():
  if name not in changed:assert sha(ROOT/name)==value
 OUT.mkdir(exist_ok=False);f=OUT/'sources';shutil.copytree(PRIOR/'sources',f)
 support=[ROOT/'test/test_wbc_go2_output_continuity.py',ROOT/'test/test_wbc_go2_continuity_wiring.py',ROOT/'validation/go2_output_continuity_preflight_husky_20261020.py',Path(__file__)]
 for p in [ROOT/name for name in changed]+support:shutil.copy2(p,f/p.name)
 for name in d['archived_unexecuted_g1_sources_sha256']:
  p=ROOT/name;shutil.copy2(p,f/('g1_backend.py'if p.name=='wbc_backend.py'else p.name))
 argv=d['command'].copy();argv[argv.index('--go2-world-z-seed')+1]='fresh_actual_output_continuity_v2';argv[-1]='Robot_Go2/results/adaptive_suite_output_continuity_01'
 for key in ['exact_same_runtime_as_single_yaw_left','single_positive_scope','existing_single_result_sha256']:d.pop(key,None)
 d.update(schema='go2_fresh_actual_output_continuity_full_declaration/v1',command=argv,
  admission_seed_profile='fresh_actual_output_continuity_v2',
  admission_seed=dict(purpose='Fresh actual control-output continuity only; no steady bias calibration or learned/physical certificate.',
   formula='I0=previous actually consumed float32 nativeZ minus newTask rawEEworldZ; newTask first reference is same-stamp actual optical camera, so original P is0.',
   full_immutable_task_sha=True,same_stamp_camera_link_extrinsic=True,previous_actual_input_max_age_s=.06,
   original_full3D_position_correction_cap_m=.025,first_task_initialization_m=0.,
   diagnostic_window_samples=26,diagnostic_window_s=.5,spans_gate_eligibility=False,
   legacy_steady_profile_1mm_gate_changed=False,steady_bias_or_training_claim=False,
   fresh_guard='Complete actual50Hz window, FCL/config/actuation/domain pass, no recovery/infeasible/other veto; fresh actual last consumed native7/time binding still mandatory.',
   transaction='Same actual first-policy-consumption commit, exact f32/native token and global clock; any guard/late/unknown/unconfirmed first tick rollback to0 retained.'),
  source_sha256={name:sha(ROOT/name)for name in d['source_sha256']},
  archived_unexecuted_g1_sources_sha256={name:sha(ROOT/name)for name in d['archived_unexecuted_g1_sources_sha256']},
  cpu_evidence_sha256={str(p.relative_to(ROOT)):sha(p)for p in [CPU/'summary.json',*TESTS]},
  cpu_tests_passed=count,cpu_tests_skipped=0,historical_prior_controller_test_count=214,
  cpu_scope='Eight independent actual newTask boundaries from the previous failed complete run. Two former resets24.232mm/23.652mm become0; clip-limited left1.240mm/yawL.713mm residual remains. No new NN/physics/stop efficacy inferred.',
  unchanged_active_lateral_brake='Same f9385857 helper,−.5actualvy/assistcap.03/originalcaps-slew and nonzero actual-history forecast; only nativeZ initialization selection changes.',
  known_limitations=['Output continuity may retain dynamic operating error; it is not a steady bias/stability or training-domain certificate.',
   'Original25mm full3Dcap prevents exact continuity where raw correction is larger; diagnostics retain residual rather than increasing cap.',
   'Earlier active-brake FULL failed two strictstops; neither CPU nor isolated yawL pass proves this full92s candidate.',
   'All original Task/error/5sstop/18q/camera/base/v/w/FCL/RT criteria must pass again, followed by sameprofile meaningful6D. Formal3 remainFalse.'],
  physical_verified=False,formal_readiness_changed=False)
 d['source_sha256'].update({str(p.relative_to(ROOT)):sha(p)for p in support})
 d['frozen_sha256']={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()}
 p=OUT/'declaration.json';p.write_text(json.dumps(d,indent=2)+'\n');print(p);print(sha(p));print(' '.join(argv))
if __name__=='__main__':main()
