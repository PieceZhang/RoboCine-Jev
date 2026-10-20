#!/usr/bin/env python3
"""Freeze fixed end+2 nativeEE hold and equivalent private pose/query fast paths."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'Robot_Go2/results/bumpless_z_full_preparation_01'
OUT=ROOT/'Robot_Go2/results/terminal_hold_full_preparation_01'
CPU=ROOT/'Robot_Go2/results/terminal_native_hold_cpu_03'
REVIEW=ROOT/'Robot_TIAGo/wbc/results/go2_terminal_hold_seed01'
CACHE=ROOT/'Robot_TIAGo/wbc/results/go2_prediction_query_cache01'
TESTS=ROOT/'Robot_Go2/results/terminal_native_hold_cpu_tests01.xml'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=json.loads((PRIOR/'declaration.json').read_text());cpu=json.loads((CPU/'summary.json').read_text());review=json.loads((REVIEW/'summary.json').read_text());cache=json.loads((CACHE/'summary.json').read_text())
 assert cpu['cpu_gate_pass']is True and cpu['pose_comparisons']==1377 and cpu['clock_negative_comparisons']==18
 assert review['passed']is True
 suites=list(ET.parse(TESTS).getroot().iter('testsuite'));count=sum(int(s.get('tests','0'))for s in suites)
 assert count==188 and sum(int(s.get(k,'0'))for s in suites for k in ['failures','errors','skipped'])==0
 for name,value in cpu['source_sha256'].items():assert sha(ROOT/name)==value
 assert sha(ROOT/'Robot_Go2/bounded_body_prediction.py')=='b0cfa3b064139274d156da0d820463377aecaf6ad366aef2ea1e730167d9c7f5'
 for name in ['terminal_native_hold.py','world_z_bumpless.py','adaptive_runner.py']:
  assert sha(ROOT/'Robot_Go2'/name)==review['frozen_sha256'][name]
 changed={'Robot_Go2/adaptive_runner.py','Robot_Go2/world_z_bumpless.py','Robot_Go2/bounded_body_prediction.py'}
 for name,value in old['source_sha256'].items():
  if name not in changed:assert sha(ROOT/name)==value
 OUT.mkdir(exist_ok=False);f=OUT/'sources';shutil.copytree(PRIOR/'sources',f)
 actual=[ROOT/name for name in old['source_sha256']if not name.startswith('validation/')and not name.startswith('test/')]
 actual.append(ROOT/'Robot_Go2/terminal_native_hold.py')
 support=[ROOT/'test/test_wbc_go2_terminal_native_hold.py',ROOT/'test/test_wbc_go2_bumpless_z_seed.py',ROOT/'validation/go2_terminal_native_hold_preflight03_husky_20261020.py',Path(__file__)]
 archived=[ROOT/name for name in old['archived_unexecuted_g1_sources_sha256']]
 for p in actual+archived+support:
  name='g1_backend.py'if p==ROOT/'Robot_G1/wbc_backend.py'else'go2_backend.py'if p==ROOT/'Robot_Go2/wbc_backend.py'else p.name
  shutil.copy2(p,f/name)
 shutil.rmtree(f/'jev_agent_wbc');shutil.copytree(ROOT/'jev_agent_wbc',f/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
 argv=old['command'].copy();argv[-1]='Robot_Go2/results/adaptive_suite_terminal_hold_01'
 index=argv.index('--go2-terminal-profile')+2;argv[index:index]=['--go2-terminal-native-hold','end2_actual_v1']
 evidence=[CPU/'summary.json',TESTS,REVIEW/'summary.json',REVIEW/'pytest.xml',CACHE/'summary.json',CACHE/'pytest.xml',CACHE/'zero_horizon.json',
  ROOT/'Robot_Go2/results/bumpless_z_timing_cpu_01/summary.json']
 evidence += [p for p in(REVIEW/'sources').iterdir()if p.is_file()]+[p for p in(CACHE/'sources').iterdir()if p.is_file()]
 result=old.copy();result.update(schema='go2_terminal_actual_hold_full_declaration/v1',command=argv,
  source_sha256={str(p.relative_to(ROOT)):sha(p)for p in actual+support},
  archived_unexecuted_g1_sources_sha256={str(p.relative_to(ROOT)):sha(p)for p in archived},
  frozen_sha256={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()},
  cpu_evidence_sha256={str(p.relative_to(ROOT)):sha(p)for p in evidence},
  cpu_tests_passed=count,cpu_tests_skipped=0,independent_transaction_checks=31,
  terminal_native_hold_profile='end2_actual_v1',
  native_hold=dict(fixed_window_offset_s=[2.,5.],source='previous successfully consumed float32 nativeEE7 feature',
   body_input='all3 actual inputs literal0 at capture and every held policy consumption',
   activation='first original end+2 tick only, at most one50Hz discretization step; fresh current observation/FCL/support/domain and original Task error checks',
   commit='after actual policy consumption, matching same report identity/TaskSHA/stamp/seq/native7 bits; all18 learned actions remain active',
   veto='any later fault/body/native override cancels hold for this Task; invalid/late confirmation never commits',
   integral='integration_allowed=False in fixed late window; selected hold rolls back same tick I token, logs intentional reason and native_correction_accepted=False',
   next_admission='Only this fullTask-bound actual hold may calibrate: all26 rows same first consumption token, current perrow seq/stamp, bit-identical native7/body0, sole terminal_native_hold rollback reason, fresh FCL/native safety/originaldomain and all3steady spans<=1mm. Other parking/recovery/veto stays rejected.'),
  pose_fast_path=dict(scope='Go2-only two private pose uses; public admission/feedback/twist unchanged',
   source='already frozen OpticalSampler original linear/Slerp/optical conversion',comparisons=1377,max_matrix_abs_error=cpu['max_pose_matrix_abs_error'],absolute_clock_negative_cases=18),
  query_cache_fast_path='Only exact explicit query→precomputed last np.isclose-matching index; nonexact original fallback. Independent645bit-exact comparisons/255sameerror paths/zero horizon and69tests retained.',
  cpu_scope='Nine unchanged measured end+2 snapshots all legally select and confirm original consumed7; no counterfactual learned-policy or PhysX propagation.',
  performance_scope='Paced450sample paired CPU costs and independent16-block cache benchmark; actual pace must pass again. No physics frequency/guard reduction.',
  known_limitations=['Holding a legal mixed native7 can permit WORLD camera drift; original Task measured error/RMS/final and every strictstop criterion remain unchanged.',
   'Old measured body-path desired-target re-expression reaches48.64mm in left case against original50mm tolerance; not a physical forecast/certificate.',
   'Stable intentional hold bias may carry an observed plant operating bias into a freshTask; cap25mm/freshness/fullSHA and original safety still apply, no physical guarantee.',
   'Formal3 control gates stayFalse until full9case and meaningful6D sameprofile actual independently pass. Environment/sensor avoidance remains uncertified.'],
  physical_verified=False,formal_readiness_changed=False)
 result['unchanged'] += ['Original Task/knot/clock/caps/reference/error feedback untouched by native7 hold; all18 learned policy outputs execute every50Hz.']
 p=OUT/'declaration.json';p.write_text(json.dumps(result,indent=2)+'\n');print(p);print(sha(p));print(' '.join(argv))
if __name__=='__main__':main()
