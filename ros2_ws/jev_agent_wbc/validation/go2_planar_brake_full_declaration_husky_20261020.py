#!/usr/bin/env python3
"""Freeze fresh two-axis late-stop feedback; no physical pass inferred."""
import hashlib,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'Robot_Go2/results/source_pd_full_preparation_01'
CPU=ROOT/'Robot_Go2/results/planar_stand_brake_cpu_02/summary.json'
TEST=ROOT/'Robot_Go2/results/planar_stand_brake_tests03.xml'
OUT=ROOT/'Robot_Go2/results/planar_stand_brake_full_preparation_02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=json.loads((PRIOR/'declaration.json').read_text());cpu=json.loads(CPU.read_text())
    assert cpu['cpu_gate_pass']is True and cpu['snapshots']==300 and cpu['forecast_rejections']==0
    changed={'Robot_Go2/adaptive_runner.py','Robot_Go2/lateral_stand_brake.py'}
    for name,value in d['source_sha256'].items():
        if name not in changed:assert sha(ROOT/name)==value,name
    for name in changed:assert sha(ROOT/name)==cpu['source_sha256'][name]
    suites=list(ET.parse(TEST).getroot().iter('testsuite'))
    count=sum(int(s.get('tests','0'))for s in suites)
    assert count==53 and sum(int(s.get(k,'0'))for s in suites for k in ['failures','errors','skipped'])==0
    OUT.mkdir(exist_ok=False);f=OUT/'sources';shutil.copytree(PRIOR/'sources',f)
    support=[ROOT/'test/test_wbc_go2_planar_stand_brake.py',ROOT/'validation/go2_planar_brake_preflight_husky_20261020.py',Path(__file__)]
    for p in [ROOT/name for name in changed]+support:shutil.copy2(p,f/p.name)
    for name in d['archived_unexecuted_g1_sources_sha256']:
        p=ROOT/name;shutil.copy2(p,f/('g1_backend.py'if p.name=='wbc_backend.py'else p.name))
    argv=d['command'].copy();argv[argv.index('--park-damping-scale')+1]='3'
    argv[argv.index('--go2-active-stand-brake')+1]='vxy_p05_v1'
    argv[-1]='Robot_Go2/results/adaptive_suite_planar_brake_01'
    d.pop('only_control_change',None)
    d['startup_readback'].pop('runtime_control_ast_equivalent',None)
    d['startup_readback']['historical_ast_proof_scope']='The sourceD1 startup-only logging addition was AST-equivalent after removing that helper/call. Current XY control change is separate and tested explicitly.'
    d.update(schema='go2_planar_velocity_brake_full_declaration/v1',command=argv,park_damping_scale=3.,
        active_stand_brake_profile='vxy_p05_v1',
        only_control_change_from_best_d3='Original late-stop vy feedback also applies same bounded negative fresh actual vx feedback. Restore best originalD3, not failed sourceD1.',
        rationale='Original D3 front-calf common velocity is strongly correlated with measured pitch gyro; translational vx relation varies in phase. Use direct negative measured planar velocity feedback, not an uncalibrated pitch/gyro gain. Neural/contact efficacy remains unverified.',
        active_brake=dict(window_offset_s=[2.,5.],raw_feedback=['-.5*actual_body_vx','-.5*actual_body_vy',0.],gain=.5,
            assist_caps_mps=[.03,.03,0.],original_body_caps=[.2,.2,.2],original_body_slew=[.5,.5,.8],
            no_integral=True,no_gyro_feedback=True,same_stamp_actual_measurement=True,
            first_input='Raw actual-velocity feedback→per-axis assistcap→original50Hz slew/caps; same actual first is locked into native-domain forecast.',
            future_assumption='Current fresh feedback target frozen with original50Hz slew and actual150ms input history through original1.2s seven-node forecast; next real tick remeasures. Not static-zero or learned/contact certification.',
            final_veto='Same native domain recovery/FCL/freshness/forecast/deadline/later override→original safe_slew0, publicFalse and same-tick originalZI rollback.',
            requested_stop='Unchanged0 and immutable original WORLD camera endpoint; only feedback braking input may be nonzero.'),
        cpu_evidence_sha256={**d['cpu_evidence_sha256'],str(CPU.relative_to(ROOT)):sha(CPU),str(TEST.relative_to(ROOT)):sha(TEST)},
        cpu_tests_passed=count,cpu_tests_skipped=0,
        cpu_scope='300 independent actual late-stop D3 snapshots from two failed cases. All original nonlinear domain forecast reserves/caps/slew pass; no neural-policy/contact propagation or efficacy inferred. Snapshot tests used earlier test file before two additional CLI assertions; physical helper/runner SHA unchanged and latest53 tests independently pass.',
        separate_measured_late_snapshots=300,forecast_rejections=0,
        minimum_reserved_margin_m=cpu['minimum_reserved_margin_m'],control_only_wall_ms=cpu['control_only_wall_ms'],
        source_sha256={name:sha(ROOT/name)for name in d['source_sha256']},
        archived_unexecuted_g1_sources_sha256={name:sha(ROOT/name)for name in d['archived_unexecuted_g1_sources_sha256']},
        unchanged_active_lateral_brake='Legacy vy_p05_v1 remains exact default path. Current vxy_p05_v1 extends feedback axes only.',
        performance_scope='Snapshot control-only wall costs are measured but do not predict full live realtime. New full92 must independently satisfy wall<=sim*1.05.',
        unchanged=[
            'Same v2 actual-output continuity seed, Ki.2, original25mm total3D correction cap, immutable fullTaskSHA and actual-policy-consumption transaction/rollback.',
            'Same original WORLD Task/knot/time/tolerances, nine5s move+5s stop cases and2s settle.',
            'Same originalD3 parking, source materials/collision geometry/effort/speed caps and18 learned policy actions at200Hz physics/50Hz policy/four writes.',
            'Same original1.6s XYyaw allocation+.4s braking, then fresh measured active feedback in originalend+2..5 window.',
            'Same1.2s seven-node forecast/actual history/35,35,15mm and15mrad reserves/.8s reserve ramp and all final safety vetoes.',
            'Same complete last.5s all18q<.01rad, camera/base5mm/10mrad, finalv<.02/w<.03 and5percent realtime bound.',
            'No native7hold; no liveRGB, posttimer raw camera/probe only.'
        ],
        known_limitations=[
            'SourceD1 sameprofile physical full test passed only2/9stops, so originalD3 restored; no mass/contact/physics changes.',
            'Correlation does not establish causal pitch dynamics. Fresh vx damping may be insufficient or adverse; complete actual all18q stop/camera/base limits determine acceptance.',
            'CPU snapshot forecast reserves and timing are not physical efficacy or realtime guarantees; full92 timing remains mandatory.',
            'Formal three control gates remainFalse until complete sameprofile FULL plus original meaningful6D pass independent audits.'
        ],physical_verified=False,formal_readiness_changed=False)
    d['source_sha256'].update({str(p.relative_to(ROOT)):sha(p)for p in support})
    d['frozen_sha256']={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()}
    p=OUT/'declaration.json';p.write_text(json.dumps(d,indent=2)+'\n');print(p);print(sha(p));print(' '.join(argv))
if __name__=='__main__':main()
