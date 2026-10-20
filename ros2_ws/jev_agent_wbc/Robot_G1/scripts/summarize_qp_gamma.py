#!/usr/bin/env python3
"""Original failed matrices and diagnostic LP gain interval; never executed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linprog
from qpsolvers import Problem, solve_problem


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    failures=json.loads((a.result/'exact_qp_failures.json').read_text())
    events=json.loads((a.result/'park_posture_refresh_events.json').read_text());results=[]
    for failure in failures:
        recovery=failure.get('pink_runtime_recovery',failure.get('recovery'));request=recovery['failed_request'];proof=recovery['qp_failure']
        group=next(g for g in proof['inequality_groups'] if g['name']=='native_pink_self_collision')
        G=np.array(proof['qp']['G']);h=np.array(proof['qp']['h']);start=group['row_start'];count=group['row_count']
        rows=slice(start,start+count);original_gain=proof['collision_class_K_gain_per_s']
        def feasibility(gain):
            rhs=h.copy();rhs[rows]*=gain/original_gain
            lp=linprog(np.zeros(G.shape[1]),A_ub=G,b_ub=rhs,bounds=[(None,None)]*G.shape[1],
                method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            return lp.status
        tested={str(g):feasibility(g) for g in (.1,.2,.3,.4,.5,1.)};interval=None
        native_qp={}
        for gain in (.1,.4):
            rhs=h.copy();rhs[rows]*=gain/original_gain
            problem=Problem(P=np.asarray(proof['qp']['P']),q=np.asarray(proof['qp']['q']),G=G,h=rhs,
                A=None if proof['qp']['A'] is None else np.asarray(proof['qp']['A']),
                b=None if proof['qp']['b'] is None else np.asarray(proof['qp']['b']))
            solution=solve_problem(problem,solver='quadprog')
            native_qp[str(gain)]=dict(found=bool(solution.found),
                maximum_inequality_residual=None if not solution.found else float(np.max(G@solution.x-rhs)),
                original_P_q_G_unchanged=True,only_native_barrier_rhs_scaled=True)
        if tested['0.1']==2 and tested['1.0']==0 and np.all(h[rows]>=0):
            lo,hi=.1,1.
            for _ in range(16):
                mid=(lo+hi)/2
                if feasibility(mid)==0:hi=mid
                else:lo=mid
            interval=dict(infeasible_below=lo,feasible_at=hi,monotonic_rhs_nonnegative=True)
        pairs=group['pairs'];closest=min(pairs,key=lambda pair:pair['distance_m'])
        results.append(dict(request_stamp_s=request['stamp_s'],failure_observed_row_s=failure['stamp_s'],
            case=failure['case'],phase='move' if (request['stamp_s']-2)%10<5 else 'stop',
            first_PARK_refresh_stamp_s=events[failure['case']]['stamp_s'],
            before_own_Task_refresh=request['stamp_s']<events[failure['case']]['stamp_s'],
            qp_capture=proof['qp_capture'],original_inequality_count=len(h),original_gain_per_s=original_gain,
            native_minimum_upper_clearance_m=closest['distance_m'],closest_pair=closest,
            diagnostic_LP_status_by_gain=tested,diagnostic_gain_interval=interval,
            diagnostic_original_quadprog=native_qp,
            native_q_velocity_acceleration_rows_unchanged=True,physical_execution_retried=False))
    output=dict(schema='g1_exact_QP_gain_diagnostic/v1',physical_execution=False,results=results,
        scope='Original exception-frame G/h. Only diagnostic nativebarrier RHS scaled, all other rows fixed. No gain/limit/Task changes are executed or proposed by this report.',
        interpretation='LP feasibility is not physical collision braking or solver robustness; changing posturedamping may change the future state/history.',
        evidence_sha256={})
    for path in (Path(__file__),a.result/'exact_qp_failures.json',a.result/'park_posture_refresh_events.json'):
        output['evidence_sha256'][str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__),a.output/'executed_source.py');(a.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
