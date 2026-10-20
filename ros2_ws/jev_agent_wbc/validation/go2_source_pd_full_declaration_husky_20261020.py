#!/usr/bin/env python3
"""Freeze one control-factor source-D1 trial; startup reads are separately bound."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'Robot_Go2/results/output_continuity_full_preparation_01'
ACTUAL = ROOT / 'Robot_Go2/results/adaptive_suite_output_continuity_01'
SCREEN = ROOT / 'Robot_Go2/results/source_pd_screen_cpu_01/summary.json'
OUT = ROOT / 'Robot_Go2/results/source_pd_full_preparation_01'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class StripReadback(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        return None if node.name == 'save_native_plant_readback' else self.generic_visit(node)

    def visit_If(self, node):
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Call)
                and isinstance(node.body[0].value.func, ast.Name)
                and node.body[0].value.func.id == 'save_native_plant_readback'):
            return None
        return self.generic_visit(node)

def main():
    d = json.loads((PRIOR / 'declaration.json').read_text())
    for name, value in d['source_sha256'].items():
        if name != 'Robot_Go2/adaptive_runner.py':
            assert sha(ROOT / name) == value, name
    old = ACTUAL / 'sources/adaptive_runner.py'
    live = ROOT / 'Robot_Go2/adaptive_runner.py'
    assert sha(old) == d['source_sha256']['Robot_Go2/adaptive_runner.py']
    assert ast.dump(ast.parse(old.read_text())) == ast.dump(StripReadback().visit(ast.parse(live.read_text())))
    screen = json.loads(SCREEN.read_text())
    assert screen['source_native_damping'] == .6 and screen['proposed_park_damping'] == .6
    OUT.mkdir(exist_ok=False)
    frozen = OUT / 'sources'
    shutil.copytree(PRIOR / 'sources', frozen)
    support = [Path(__file__), ROOT / 'validation/go2_source_pd_screen_husky_20261020.py']
    for path in [live, *support]:
        shutil.copy2(path, frozen / path.name)
    for name in d['archived_unexecuted_g1_sources_sha256']:
        path = ROOT / name
        shutil.copy2(path, frozen / ('g1_backend.py' if path.name == 'wbc_backend.py' else path.name))
    proof = dict(schema='go2_readonly_startup_ast_equivalence/v1',
        prior_frozen_runner_sha256=sha(old), candidate_runner_sha256=sha(live),
        removed_only=['top-level save_native_plant_readback function',
                      'one Go2-only call before start_wall'],
        remaining_complete_ast_equal=True,
        scope='All timed control/physics/reference/guard/logging code is AST-identical; CLI damping factor is the only control change.')
    proof_path = OUT / 'startup_ast_equivalence.json'
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    argv = d['command'].copy()
    argv[argv.index('--park-damping-scale') + 1] = '1'
    argv[-1] = 'Robot_Go2/results/adaptive_suite_source_pd_01'
    d.update(schema='go2_source_pd_onefactor_full_declaration/v1', command=argv,
        park_damping_scale=1.,
        only_control_change=dict(parameter='--park-damping-scale', old=3., new=1.,
            actual_native_leg_damping_old=1.8, actual_native_leg_damping_new=.6,
            stiffness_unchanged=30., source_training_damping_randomized_range=[.48,.72]),
        rationale='Match source-trained native leg PD during parking. Correlation implicates front-leg common/pitch motion, but D3 is dissipative and no causal or stability claim is inferred from the isolated scalar screen.',
        startup_readback=dict(before_timing=True, read_only=True,
            file='startup_native_plant.json',
            reads=['actual explicit actuator K/D/caps', 'already sampled delay buffers; no RNG draws',
                   'native body mass/COM/inertia tensors with frame conventions',
                   'actor world poses and source-default/measured joints', 'USD/layer/API source SHA'],
            physics_parameters_changed=False, runtime_control_ast_equivalent=True),
        source_sha256={name:sha(ROOT/name) for name in d['source_sha256']},
        archived_unexecuted_g1_sources_sha256={name:sha(ROOT/name) for name in d['archived_unexecuted_g1_sources_sha256']},
        cpu_evidence_sha256={**d['cpu_evidence_sha256'], str(SCREEN.relative_to(ROOT)):sha(SCREEN),
            str(proof_path.relative_to(ROOT)):sha(proof_path)},
        cpu_scope='Existing103 seed/wiring tests retained as historical evidence; no control-source changes. New read-only startup code removed from AST exactly reproduces prior physically executed runner. Scalar PD screen is illustrative, not a contact/learned-loop certificate.',
        unchanged=[
            'Same v2 actual-output continuity seed, Ki .2, original25mm full3D correction cap, fullTaskSHA, actual policy-consumption transaction and all veto rollback.',
            'Same active vy-only feedback, XYyaw allocation, all-axis bounded history forecast, original7nodes/reserves/caps/slew.',
            'Same original nine5s move+5s stop tasks and2s settle; original WORLD reference/clock/tolerances are not retargeted.',
            'All18 learned actions, source collision/contact/material/effort/velocity limits and200Hz physics/50Hz policy/four writes unchanged.',
            'Original complete last.5s all18q<.01rad, camera/base5mm and10mrad, finalv<.02/w<.03 and wall<=sim*1.05 remain.',
            'No native7hold; no liveRGB. RawRGB/probe only after timed physics.'
        ],
        known_limitations=[
            'Previous v2 FULL completed but failed yaw_left and camera_shot strict stops; its8/8 continuity commits are not proof of this candidate.',
            'Body-response predictor is an explicit operating approximation; matching sourceD1 does not certify the learned/contact dynamics.',
            'Current source/control gates stayFalse until this complete sameprofile FULL and subsequent meaningful6D physically pass independent audits.'
        ], physical_verified=False, formal_readiness_changed=False)
    d['source_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in support})
    d['frozen_sha256'] = {str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()}
    path = OUT / 'declaration.json'
    path.write_text(json.dumps(d, indent=2) + '\n')
    print(path); print(sha(path)); print(' '.join(argv))

if __name__ == '__main__':
    main()
