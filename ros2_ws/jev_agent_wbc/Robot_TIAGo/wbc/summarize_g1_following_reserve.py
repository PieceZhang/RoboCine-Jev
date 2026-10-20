#!/usr/bin/env python3
"""Summarize conditional CPU QP experiments without changing physical evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short', type=Path, required=True)
    parser.add_argument('--early', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = output / 'sources'
    sources.mkdir()
    shutil.copy2(__file__, sources / Path(__file__).name)
    reports = {key: json.loads((path / 'summary.json').read_text())
               for key, path in [('short', args.short), ('early', args.early)]}
    experiments = []
    for label, folder in [('short', args.short), ('early', args.early)]:
        report = reports[label]
        for trace in report['traces']:
            factor, lag = trace['assumed_deceleration_fraction'], trace['latency_s']
            same = [s for s in report['solutions']
                    if s['assumption_fraction'] == factor and s['latency_s'] == lag]
            first = next((row for row in trace['states']
                          if row['original_command_extra_inequality_residual_m'] is not None
                          and row['original_command_extra_inequality_residual_m'] < 0), None)
            first_unviable = next((row for row in trace['states']
                                   if row['physical_stopping_margin_m'] < 0), None)
            first_solution = None if first is None else next(
                (s for s in same if abs(s['stamp_s'] - first['stamp_s']) < 1e-8), None)
            numeric = []
            for solution in same:
                if not solution['genuine_QP_solved']:
                    continue
                with np.load(folder / solution['matrix_file']) as qp:
                    # Frozen native backend emits velocity in reduced-model
                    # joint order; its ordered dictionary preserves that order.
                    velocity = np.asarray(list(solution['velocity'].values()))
                    residual = qp['h'] - qp['G'] @ (.02 * velocity)
                    numeric.append(float(np.min(residual)))
            details = None
            if first_solution is not None and first_solution['genuine_QP_solved']:
                # Pink appends barriers after limits: the new Limit need not
                # be the last G row. The experiment independently records its
                # residual h_extra + J*dt*v, so recover J*v from that value.
                guarded_jv = (first_solution['extra_inequality_residual_m']
                              - first['committed_reference_linear_margin_m']) / .02
                details = dict(
                    first_active_state=first,
                    original_J_v_m_s=(first['original_command_extra_inequality_residual_m']
                                      - first['committed_reference_linear_margin_m']) / .02,
                    guarded_J_v_m_s=guarded_jv,
                    guarded_QP_solved=True,
                    guarded_LP_feasible=first_solution['guarded_linear_feasible'])
            experiments.append(dict(
                name=label, assumed_deceleration_fraction=factor, latency_s=lag,
                extra_lookahead_s=report['extra_lookahead_s'],
                first_active_state=first, first_active_solution=details,
                first_negative_assumed_physical_margin=first_unviable,
                QP_calls=len(same), QP_solved=sum(s['genuine_QP_solved'] for s in same),
                minimum_solved_native_inequality_residual=float(min(numeric)),
                original_P_q_unchanged=all(s['original_P_q_preserved'] for s in same),
                static_zero_rate_fixture_pass=all(s['genuine_QP_solved'] for s in same
                                                  if s['static_zero_rate_fixture'])))
    report = dict(
        schema='g1_conditional_following_reserve_design/v1',
        physical_execution=False, runtime_modified=False, physical_safety_certified=False,
        scope='Native Pink15 original FrameTask, original closest20 SelfCollisionBarrier and native q/v/issued-acceleration limits are preserved. Add one genuine native linear Limit for critical pair377 only; no solver-output projection or previous reset.',
        retained_limits=dict(d_min_m=.0065, physical_FCL_abort_m=.005,
                             joint_velocity_rad_s=.5, issued_acceleration_rad_s2=1.,
                             collision_class_K_gain_per_s=1.),
        state_definitions=dict(
            q='Measured upper15 joints at request acquisition time.',
            r='The actual already-committed physical position-drive target at the same acquisition stamp. Never a later application-time reference.',
            J='Same-acquisition native distance Jacobian, m/rad; independently FD-checked in g1_physical_collision_abort01.',
            c='max(0,-J*qd_actual), m/s.',
            closing_reference_lead='max(0,-J*(r-q)), metres. Opening reference error gives no positive credit.'),
        tested_extra_native_inequality='-J*Delta_q <= d - .0065 - (L+T_guard)*c - c^2/(2*a_assumed) - closing_reference_lead',
        tested_physical_viability='H=d-.0065-L*c-c^2/(2*a_assumed). H<0 is already outside the assumed stoppable region even if the QP is feasible.',
        conditional_early_trigger='Brake/replan before d-.0065 <= (L+T_guard)*c+c^2/(2*a_assumed)+closing_reference_lead; keep the extra inequality always active rather than adding it for the first time at the FCL threshold.',
        provisional_parameter_example=dict(
            L_s=.06, T_guard_s=.06,
            a_assumed='0.5 * sum(abs(J)) * 1rad/s^2; conditional hypothesis only, not a measured braking lower bound.'),
        numerical_findings=experiments,
        important_negative_results=[
            'With L=.06, a_assumed=.5*sum(abs(J)), and only20ms lookahead, first intervention is25.92s with12.34micrometre assumed physical margin. At25.94s the guarded QP and LP are infeasible; original QP remains feasible. This is too late for a robust implementation.',
            'With60ms lookahead, first intervention is25.44s: d6.80914mm, c2.46622mm/s, hypothetical physical stopping margin127.47micrometre. The extra native QP is feasible and changes J*v from closing to opening without changing the original objective.',
            'The original recorded state sequence still reaches later states where the extra inequalities are infeasible. No online trajectory was rolled out, so earlier counterfactual commands are not evidence that the physical abort would disappear.',
            'The physical actuation/reference-lead problem cannot be certified by increasing gamma, checking all473 pairs, or demonstrating LP feasibility.'
        ],
        necessary_assumptions_and_missing_bounds=[
            'The tested stopping-distance expression assumes actual closing speed cannot increase during latency L and actual relative deceleration is at least a_assumed afterwards. Neither assumption follows from an issued acceleration upper cap of1rad/s^2.',
            'For a certified approaching acceleration upper bound a_close, replace L*c+c^2/(2*a_brake) with L*c+0.5*a_close*L^2+(c+a_close*L)^2/(2*a_brake), and add measured-distance/J-linearization/tracking uncertainty margins. None of these actual dynamic bounds has been identified or validated here.',
            'The local linear J inequality is a first-order proxy. Native exact FCL checks remain authoritative; curvature and tracking errors need bounded margins.',
            'The prototype only constrains one upper-controlled pair. Generalization requires correct full-joint measured relative velocities for each affected pair and must handle genuine native-limit infeasibility explicitly.'
        ],
        asynchronous_application_contract=[
            'Bind each request to its actual measured-q stamp, already-committed-r token, previous issued native velocity, Task context and deadline.',
            'Track the newly planned target separately from the target actually committed by PhysX. A submit-before-write target is not already committed.',
            'At reply application, require that the target evolution matches a verified bounded advance schedule or reject/re-solve with fresh state. Never substitute application-time r into an old QP and call it the original request.',
            'All physical ticks continue while bounded asynchronous work runs. Infeasible/stale evidence must produce explicit hold/braking/recovery status; no relaxed limits or automatic previous reset.'
        ],
        acceptance_boundary='Conditional CPU design and counterfactual feasibility only. Requires fresh-state runtime integration plus independently measured actual braking/tracking/latency bounds and new full physical tracking/clearance/stop/pace validation before any safety gate changes.',
    )
    paths = [args.short / 'summary.json', args.early / 'summary.json',
             sources / Path(__file__).name]
    report['evidence_sha256'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in paths}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(conditional_experiments=len(experiments),
                         genuine_QP_calls=sum(x['QP_calls'] for x in experiments),
                         all_objectives_unchanged=all(x['original_P_q_unchanged'] for x in experiments),
                         all_static_fixtures_solved=all(x['static_zero_rate_fixture_pass'] for x in experiments))))


if __name__ == '__main__':
    main()
