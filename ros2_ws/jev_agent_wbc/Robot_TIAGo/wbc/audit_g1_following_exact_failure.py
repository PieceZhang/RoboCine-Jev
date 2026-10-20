#!/usr/bin/env python3
"""Exact captured native Problem IIS; no runtime/helper imports or weaker execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linprog
import qpsolvers


def lp(G,h,c=None):
    return linprog(np.zeros(G.shape[1])if c is None else c,A_ub=G,b_ub=h,
        bounds=[(None,None)]*G.shape[1],method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recorded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();folder=a.recorded.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir();shutil.copy2(__file__,out/'sources'/Path(__file__).name)
    failures=json.loads((folder/'exact_qp_failures.json').read_text())
    rows=[json.loads(line)for line in(folder/'measurements.jsonl').open()];by_stamp={round(r['stamp_s'],8):r for r in rows}
    external={round(r['fine_collision_evidence']['stamp_s'],8):r['fine_collision_evidence']for r in rows if r.get('fine_collision_evidence',{}).get('stamp_s')is not None}
    result=[]
    for index,item in enumerate(failures):
        packet=item['pink_runtime_recovery'];request=packet['failed_request'];diag=packet['qp_failure'];stamp=request['stamp_s']
        assert diag['qp_capture']=='native_pink_exception_frame'and not diag['reconstructed_after_solver_failure']
        P,q,G,h=[np.asarray(diag['qp'][k],float)for k in('P','q','G','h')];groups=diag['inequality_groups'];n=len(q)
        reserve=next(g for g in groups if g['name']=='native_following_reserve')
        native=next(g for g in groups if g['name']=='native_pink_self_collision')
        omit=set(range(reserve['row_start'],reserve['row_start']+reserve['row_count']))
        keep=np.array([i for i in range(len(h))if i not in omit]);oldG,oldh=G[keep],h[keep]
        full_lp=lp(G,h);old_lp=lp(oldG,oldh)
        # The exact same native task objective. Removal is a diagnostic
        # counterfactual only; never proposed or applied to a robot.
        full_qp=qpsolvers.solve_problem(qpsolvers.Problem(P,q,G,h),solver='quadprog')
        old_qp=qpsolvers.solve_problem(qpsolvers.Problem(P,q,oldG,oldh),solver='quadprog')
        active=list(range(len(h)))
        for i in list(active):
            trial=[j for j in active if j!=i]
            if lp(G[trial],h[trial]).status==2:active=trial
        iis=[];pair_details=[]
        for i in active:
            group=next(g for g in groups if g['row_start']<=i<g['row_start']+g['row_count']);local=i-group['row_start']
            info=dict(row=i,group=group['name'],rhs=float(h[i]),coefficients=G[i].tolist())
            if group['name']=='owned_velocity_acceleration':info.update(joint=diag['joint_names'][local%n],bound='upper'if local<n else'lower')
            if 'pairs'in group:info['pair']=group['pairs'][local]
            iis.append(info)
            if group['name']=='native_following_reserve':
                metric=group['pairs'][local];pair=next(p for p in native['pairs']if p['index']==metric['pair_index'])
                bodies=[path.split('/')[3]for path in pair['shapes']]
                J=-G[i];maximum=lp(oldG,oldh,c=-J)
                maximum_rate=None if not maximum.success else float(J@maximum.x/.02)
                future=by_stamp.get(round(stamp+.02,8));linear_rate=None
                if future is not None:
                    q0=np.array([request['joints'][name]for name in diag['joint_names']])
                    q1=np.array([future['joint_positions'][future['joint_names'].index(name)]for name in diag['joint_names']])
                    linear_rate=float(J@(q1-q0)/.02)
                qd=np.asarray(request['following_evidence']['measured_qd_rad_s'])
                ext0=external.get(round(stamp,8));ext1=external.get(round(stamp+.02,8));actual_d=[]
                for ext in(ext0,ext1):
                    matches=[]if ext is None else [p['distance_m']for p in ext['close_pairs']if set([p['first'],p['second']])==set(bodies)]
                    actual_d.append(min(matches)if matches else None)
                pair_details.append(dict(pair_index=metric['pair_index'],bodies=bodies,
                    both_upper_bodies=not any(any(key in b for key in('hip','knee','ankle','pelvis'))for b in bodies),
                    metrics=metric,required_opening_rate_m_s=float(-h[i]/.02),
                    maximum_opening_rate_under_all_original_limits_and_CBF_m_s=maximum_rate,
                    deficit_per_native20ms_m=None if maximum_rate is None else float(-h[i]-.02*maximum_rate),
                    J_actual_qd_m_s=float(J@qd),J_actual_next_q_difference_m_s=linear_rate,
                    external_same_body_pair_distance_m=actual_d,
                    external_next20ms_distance_rate_m_s=None if None in actual_d else float((actual_d[1]-actual_d[0])/.02),
                    note='Body-pair FCL values are matched by raw names; no new query/FD or physics. Instantaneous qd is not replaced by averaged pose differences.'))
        np.savez_compressed(out/f'exact_{stamp:.2f}.npz',P=P,q=q,G=G,h=h,diagnostic_original_only_G=oldG,diagnostic_original_only_h=oldh)
        result.append(dict(failed_request_stamp_s=stamp,post_stamp_s=item['stamp_s'],exact_capture=diag['qp_capture'],
            full130_LP_status=int(full_lp.status),full130_quadprog_found=bool(full_qp.found),
            same_Pq_remove_only_new20_LP_status=int(old_lp.status),same_Pq_remove_only_new20_quadprog_found=bool(old_qp.found),
            old_native_rows_preserved=110,diagnostic_removed_rows=sorted(omit),irreducible_rows=iis,
            irreducibility_verified=all(lp(G[[j for j in active if j!=i]],h[[j for j in active if j!=i]]).success for i in active),
            conflicting_new_pairs=pair_details,
            original_only_diagnostic_velocity=None if old_qp.x is None else (old_qp.x/.02).tolist(),
            new20_removed_in_actual_runtime=False))
    paths=[folder/'exact_qp_failures.json',folder/'measurements.jsonl',out/'sources'/Path(__file__).name]
    report=dict(schema='g1_following_exact130_independent/v1',physical_execution=False,imports_runtime_helpers=False,
        scope='Original native exception-frame matrices only. Diagnostic row removal holds original P/q/all remaining inequalities exactly fixed; no relaxation is executed or proposed. Greedy IIS is irreducible, not claimed minimum-cardinality.',
        original_d_min_m=.0065,original_FCL_threshold_m=.005,original_qdot_cap_rad_s=.5,issued_acceleration_cap_rad_s2=1.,
        results=result,evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths})
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(stamp=r['failed_request_stamp_s'],IIS=[x['row']for x in r['irreducible_rows']],
        old_LP=r['same_Pq_remove_only_new20_LP_status'],pair=r['conflicting_new_pairs'])for r in result],indent=2))


if __name__=='__main__':main()
