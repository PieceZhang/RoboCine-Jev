#!/usr/bin/env python3
"""Fit one-step jvs_mock response from measured PhysX logs, with held-out audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.execution_prediction import ExecutionModel, motor_step
from jev_agent_wbc.whole_body_qp import Limits
from jev_agent_wbc.kc_svsdf import advance


def samples(files):
    result, ages = [], []
    for file in files:
        summary=json.loads((Path(file).parent/'summary.json').read_text())
        contract=dict(backend='isaac_physx',physics_hz=100.,motor_targets='physical_slew',base_support='upright')
        if any(summary.get(key)!=value for key,value in contract.items()):
            raise ValueError('calibration run does not match the jvs_mock 100 Hz physical drive contract: '+str(file))
        rows = [json.loads(line) for line in Path(file).read_text().splitlines()]
        last_sequence=None
        for row, following in zip(rows, rows[1:]):
            info = row['info']
            age = None if info.get('command_source_time_s') is None else row['t']-info['command_source_time_s']
            if (abs(following['t']-row['t']-.02)>1e-7 or info.get('runtime_fault') or
                    info.get('waiting_for_controller') or info.get('waiting_for_sensor') or
                    info.get('waiting_for_certificate') or row['truth_clearance_m'] < .025 or
                    age is None or not 0<=age<=.1):
                continue
            result.append((np.array(row['state']), np.array(row['measured']),
                           np.array(row['command']), np.array(following['measured'])))
            sequence=info.get('command_source_sequence')
            if sequence is None:sequence=info.get('command_source_time_s')
            if sequence!=last_sequence:
                arrival_delay=info.get('command_delivery_delay_s')
                ages.append(age if arrival_delay is None else float(arrival_delay))
                last_sequence=sequence
    if not result:
        raise ValueError('no fresh, non-braking physical transitions')
    return result, np.asarray(ages)


def residuals(rows, tau):
    limits = Limits()
    gain = np.ones(8)
    positive = np.asarray(tau)>0
    gain[positive] = -np.expm1(-.01/np.asarray(tau)[positive])
    errors = []
    for state, initial, command, actual in rows:
        speed = initial.copy()
        predicted_state=state.copy()
        for _ in range(2):
            target = motor_step(command, speed, predicted_state, limits, .01)
            speed += gain*(target-speed)
            predicted_state=advance(predicted_state,speed,.01)
        errors.append(actual-speed)
    return np.asarray(errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', nargs='+', type=Path, required=True)
    p.add_argument('--holdout', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if {f.resolve() for f in a.train} & {f.resolve() for f in a.holdout}:
        raise ValueError('training and holdout runs must be disjoint')
    provenance = {str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.train+a.holdout}
    if {provenance[str(f.resolve())] for f in a.train} & {provenance[str(f.resolve())] for f in a.holdout}:
        raise ValueError('training and holdout runs must have distinct execution content')
    train, ages = samples(a.train)
    holdout, _ = samples(a.holdout)
    grid = np.array([0., .0025, .005, .01, .02, .04])
    losses = np.array([np.mean(residuals(train, np.full(8,t))**2,axis=0) for t in grid])
    tau = grid[np.argmin(losses,axis=0)]
    errors = residuals(train,tau)
    # A per-step empirical maximum is not a joint path coverage probability.
    envelope = np.maximum(1.2*np.max(abs(errors),axis=0), [.0005,.001,.0005,.0005,.0005,.0005,.0005,.0005])
    model = ExecutionModel(response_s=tau.tolist(), residual_velocity=envelope.tolist(),
        bias_velocity=np.mean(errors,axis=0).tolist(), delay_s=float(np.median(ages)),
        calibrated_velocity_abs_max=np.max(abs(np.array([v for row in train for v in (row[1],row[3])])),axis=0).tolist(),
        delay_jitter_s=max(.01,float(np.max(ages)-np.median(ages))), samples=len(train),
        evidence='disjoint_run_one_step_physical_calibration; trajectory_coverage_unverified')
    held = residuals(holdout,tau)
    report = dict(model=model.to_dict(), training_runs=[str(f.resolve()) for f in a.train],
        heldout_runs=[str(f.resolve()) for f in a.holdout], source_sha256=provenance,
        delay_measurement='First observed application of each fresh command sequence; held-command age is not transport latency.',
        training_samples=len(train),heldout_samples=len(holdout),
        heldout_velocity_rms=np.sqrt(np.mean(held**2,axis=0)).tolist(),
        heldout_component_coverage=np.mean(abs(held)<=envelope,axis=0).tolist(),
        heldout_joint_step_coverage=float(np.mean(np.all(abs(held)<=envelope,axis=1))),
        scope='No fitted collision/shot acceptance thresholds. One-step calibration is not a whole-trajectory guarantee.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('model','source_sha256','training_runs','heldout_runs')},indent=2))


if __name__ == '__main__':
    main()
