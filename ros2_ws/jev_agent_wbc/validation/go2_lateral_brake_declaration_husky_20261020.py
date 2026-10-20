#!/usr/bin/env python3
"""Freeze a single original yawL trial of measured-only lateral braking."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'Robot_Go2/results/terminal_hold_full_preparation_01'
CPU=ROOT/'Robot_Go2/results/lateral_stand_brake_cpu_01'
OUT=ROOT/'Robot_Go2/results/lateral_stand_brake_preparation_01'
TESTS=ROOT/'Robot_Go2/results/lateral_stand_brake_tests01.xml'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=json.loads((PRIOR/'declaration.json').read_text());cpu=json.loads((CPU/'summary.json').read_text())
 assert cpu['cpu_gate_pass']is True and cpu['snapshots']==150 and cpu['forecast_rejections']==0
 for name,value in cpu['source_sha256'].items():assert sha(ROOT/name)==value
 suites=list(ET.parse(TESTS).getroot().iter('testsuite'));count=sum(int(s.get('tests','0'))for s in suites)
 assert count==214 and sum(int(s.get(k,'0'))for s in suites for k in ['failures','errors','skipped'])==0
 for name,value in old['source_sha256'].items():
  if name!='Robot_Go2/adaptive_runner.py':assert sha(ROOT/name)==value
 OUT.mkdir(exist_ok=False);f=OUT/'sources';shutil.copytree(PRIOR/'sources',f)
 (f/'terminal_native_hold.py').unlink()
 actual=[ROOT/name for name in old['source_sha256']if not name.startswith(('validation/','test/'))and name!='Robot_Go2/terminal_native_hold.py']
 actual.append(ROOT/'Robot_Go2/lateral_stand_brake.py')
 support=[ROOT/'test/test_wbc_go2_lateral_stand_brake.py',ROOT/'validation/go2_lateral_brake_preflight_husky_20261020.py',Path(__file__)]
 archived=[ROOT/name for name in old['archived_unexecuted_g1_sources_sha256']]
 for p in actual+support+archived:
  name='g1_backend.py'if p==ROOT/'Robot_G1/wbc_backend.py'else'go2_backend.py'if p==ROOT/'Robot_Go2/wbc_backend.py'else p.name
  shutil.copy2(p,f/name)
 shutil.rmtree(f/'jev_agent_wbc');shutil.copytree(ROOT/'jev_agent_wbc',f/'jev_agent_wbc',ignore=shutil.ignore_patterns('__pycache__'))
 argv=old['command'].copy();argv[argv.index('--cases')+1]='yaw_left';index=argv.index('--go2-terminal-native-hold');argv[index:index+2]=['--go2-active-stand-brake','vy_p05_v1'];argv[-1]='Robot_Go2/results/adaptive_yaw_left_lateral_brake_01'
 result=old.copy();result.pop('native_hold',None)
 result.update(schema='go2_lateral_stand_brake_declaration/v1',command=argv,
  source_sha256={str(p.relative_to(ROOT)):sha(p)for p in actual+support},
  archived_unexecuted_g1_sources_sha256={str(p.relative_to(ROOT)):sha(p)for p in archived},
  frozen_sha256={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()},
  cpu_evidence_sha256={str(p.relative_to(ROOT)):sha(p)for p in [CPU/'summary.json',CPU/'rows.json',TESTS,
   ROOT/'Robot_Go2/results/bumpless_stop_axes_cpu_01/summary.json']},
  cases=['yaw_left'],total_sim_s=12,settle_s=2,move_s=5,stop_s=5,
  terminal_native_hold_profile='none',active_stand_brake_profile='vy_p05_v1',cpu_tests_passed=count,cpu_tests_skipped=0,
  separate_measured_late_snapshots=150,forecast_rejections=0,minimum_reserved_margin_m=cpu['minimum_reserved_margin_m'],
  control_only_wall_ms=cpu['control_only_wall_ms'],
  active_brake=dict(window_offset_s=[2.,5.],actual_measurement='Same-stamp measured body vy; no I or planned body trajectory.',
   raw_feedback=[0,'-0.5*actual_body_vy',0],assist_vy_cap_mps=.03,original_body_caps=[.2,.2,.2],original_body_slew=[.5,.5,.8],
   first_input='raw→assistcap→original50Hz sourcecap/slew; X/yaw remain0',
   future_assumption='Same fresh measured feedback target frozen through original1.2s horizon with original50Hz slew and actual150ms held input history; next real tick remeasures. Explicit nonzero operating approximation, not frozen-zero or learned-policy prediction.',
   forecast='Original0,.2,.4,.6,.8,1,1.2s nodes when horizon permits;35/35/15mm and.015rad reserves/.8ramp unchanged; locked actual first postchecked nonlinear margins.',
   final_veto='Any forecast rejection, FCL freshness, native-domain recovery, prior deadline failure or changed later body input forces original safe_slew0, publicFalse and same-tick I rollback.',
   requested_stop='Original requested bodyvelocity0 and original immutable WORLD camera endpoint remain; actual bounded feedback brake may be nonzero afterend+2.',
   audit='New profile requires independent raw formula/time/saturation/slew/final-veto audit. Old literalzero claim/audit applies only to old profiles; no previous proof transferred.'),
  cpu_scope='150 independent original yawL late-stop states, all current sourcecaps/slew and original reserved workspace check pass. No hypothetical commands propagated into measurements; no NN/PhysX braking effect claim.',
  performance_scope='Uses previously physically timed exact-cache and pose-only fast paths. Actual12s pace checked anew, not inferred from CPU mean.',
  known_limitations=['Body lateral command may affect learned policy/roll balance; physical damping efficacy unknown.',
   'The short3s model freezes this tick feedback target and approximates pending velocity; it does not predict future learned18/contact/stability.',
   'Single isolated yawL does not prove continuous9case or meaningful6D; formal3 control gates remainFalse until all original physical criteria independently pass.'],
  physical_verified=False,formal_readiness_changed=False)
 result['unchanged']=[x for x in old['unchanged']if '1.6s allocation+.4s brake+3s literal0'not in x]
 result['unchanged']+=['End+2 before active-brake window keeps original1.6s XYyaw allocation+.4s braking. Stop5s and actual full.5s18q/camera/base/v/w/timing thresholds unchanged.',
  'No native7hold; original bounded P+ZI and fresh measured bumpless admission enabled. Every18 learned policy action still executes.']
 p=OUT/'declaration.json';p.write_text(json.dumps(result,indent=2)+'\n');print(p);print(sha(p));print(' '.join(argv))
if __name__=='__main__':main()
