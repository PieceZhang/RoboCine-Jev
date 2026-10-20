#!/usr/bin/env python3
"""Read-only source/measurement audit; never imports Isaac or changes a run."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'Robot_G1/results/default_upper_yaw_01'


def main():
    rows = [json.loads(line) for line in (RUN/'measurements.jsonl').read_text().splitlines()]
    audit = json.loads((RUN/'actuator_audit.json').read_text())
    names = audit['joint_names']
    upper = [i for i,n in enumerate(names) if not any(part in n for part in ('hip','knee','ankle'))]
    caps = np.asarray(audit['effort_limits'])
    cases = {}
    for case in ('yaw_left','yaw_right'):
        active = [r for r in rows if r['case']==case and r['phase']=='move']
        q = np.asarray([r['joint_positions'] for r in active])
        effort = np.asarray([r['applied_joint_effort_peak_abs'] for r in active])
        cases[case] = dict(samples=len(active),
            measured_body_velocity_mean=np.mean([r['measured_body_velocity'] for r in active],axis=0).tolist(),
            policy_input_mean=np.mean([r['applied_policy_velocity'] for r in active],axis=0).tolist(),
            upper_measured_source_default_error_rad={names[i]:dict(mean=float(q[:,i].mean()),
                max_abs=float(abs(q[:,i]).max())) for i in upper},
            action_max_abs=float(np.max(abs(np.asarray([r['native_policy_action'] for r in active])))),
            estimated_pd_effort_peak_fraction={names[i]:float(effort[:,i].max()/caps[i]) for i in range(len(names))},
            actual_contact_or_implicit_drive_torque_certified=False)
    sources = [RUN/'sources/adaptive_runner.py', RUN/'measurements.jsonl',RUN/'actuator_audit.json',
        RUN/'startup_physx_mass_properties.json',ROOT/'Robot_G1/config/height_policy_io.yaml',
        ROOT/'Robot_G1/upstream/WBC-AGILE/agile/rl_env/assets/robots/unitree_g1.py',
        ROOT/'Robot_G1/upstream/WBC-AGILE/agile/rl_env/tasks/locomotion_height/g1/velocity_height_env_cfg.py']
    report = dict(schema='g1_native_source_yaw_readonly/v1',diagnostic_only=True,
        physical_changes=False,upstream_commit='ad4ff6ee6b38da207592b3e71dcee4c118049f6d',
        official_source='https://github.com/nvidia-isaac/WBC-AGILE/tree/ad4ff6ee6b38da207592b3e71dcee4c118049f6d',
        cases=cases,verified_source_findings=[
            'Recurrent student training and eval delete harness. Its absence is not a mismatch.',
            'Checkpoint IO descriptor matches 4 command + root-body gyro 3 + projected gravity 3 + all29 q + .1*all29 qd + previous12 actions; no external observation history.',
            'Native G1_29DOF is None/None hand USD, 200Hz physics/50Hz policy, 8/4 iterations; current diagnostic preserves these.',
            'Leg/feet delayed implicit actuator range is source 0..4 physics steps. Actual sampled delay is not saved; matching ranges do not certify the sampled delay.',
            'Current hip-roll effort operating cap88Nm differs from training139Nm. Logged implicit-PD estimates remain far below88; they do not certify actual contact/drive torque.',
            'Upper control differs: explicit bounded PD + Pin gravity, damping3x source, source-default target with +/-0.05rad measured target lead. Measured wrists drift substantially although other upper joints stay near zero.',
            'Training source last_action is raw policy output clipped at observation +/-10 while current loop stores motor-clipped +/-6. Recorded motor outputs never approach6, so no evidence this difference activated in this run.',
            'Ground source combine multiply and friction1 matches current ground; robot shape scene-default .5 friction is explicitly resolved in current run. Student eval disables randomization; training randomizes robot material/gains/mass/COM.'
        ],limitations=[
            'Not a perfect source-native upper baseline: old Pin gravity/model mismatch remains and wrists are not at source default.',
            'Root gyro semantic/static source matches; no standalone replay of recurrent network/individual feature tensors completed in this audit.',
            'Exact remote Nucleus asset version at checkpoint training is not pinned by official config; same USD logical URL does not prove byte-identical historical training asset.',
            'No actual contact impulse or sampled delay telemetry; no causality established for low yaw.',
            'Mass/COM/inertia and Pin gravity comparisons owned by validation/tiago; avoid direct COM-principal-frame vs actor-inertia comparison.'
        ],minimal_next_candidate='First correct/verify upper model gravity against actual PhysX and quantify wrist default hold before another native baseline. Preserve 12 learned leg actions and all caps; do not raise feedforward to compensate an unverified plant discrepancy.',
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    output=Path(__file__).with_suffix('.json');output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),samples=len(rows),cases={k:dict(action_max=v['action_max_abs'],
        wrist_pitch={n:d for n,d in v['upper_measured_source_default_error_rad'].items() if 'wrist_pitch' in n}) for k,v in cases.items()}),indent=2))


if __name__=='__main__':main()
