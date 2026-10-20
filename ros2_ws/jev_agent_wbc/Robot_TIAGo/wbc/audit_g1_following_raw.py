#!/usr/bin/env python3
"""Independent raw-only binding/write audit; never imports runtime helpers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

CANONICAL = ['waist_yaw_joint'] + [side+'_'+joint+'_joint' for side in ['left','right']
    for joint in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recorded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();folder=a.recorded.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir();shutil.copy2(__file__,out/'sources'/Path(__file__).name)
    rows=[json.loads(line)for line in(folder/'measurements.jsonl').open()]
    tasks=json.loads((folder/'timed_tasks.json').read_text());contexts={t['task_id']:dict(
        schema='g1_solver_task_context/v1',task_id=t['task_id'],task_sha256=digest(t),camera_path_id=digest(t['camera_trajectory']))for t in tasks}
    by_stamp={round(row['stamp_s'],8):row for row in rows};requests={};errors=[];counts=Counter();maxima=Counter()
    def check(ok,stamp,family,detail):
        counts[family+'_checked']+=1
        if not bool(ok):errors.append(dict(stamp_s=stamp,family=family,detail=detail))
    def same(x,y):return np.array_equal(np.asarray(x),np.asarray(y))
    def vector(row,key):return np.asarray(row[key],float)[[row['joint_names'].index(n)for n in CANONICAL]]
    def ref_ok(ref,stamp):
        check(ref['joint_names']==CANONICAL,stamp,'reference','canonical15')
        expected=digest(dict(task_id=ref['task_id'],task_sha256=ref['task_sha256'],
            joint_names=ref['joint_names'],positions_rad=ref['positions_rad']))
        check(ref['source']=='physically_committed'and ref['token']==expected,stamp,'reference','content token')
        check(ref['commit_stamp_s']<=ref['snapshot_stamp_s']+1e-9,stamp,'reference','no future write')
        if ref['task_id'] is not None:
            check(ref['task_id']in contexts and ref['task_sha256']==contexts[ref['task_id']]['task_sha256'],stamp,'reference','full Task SHA')
    for row in rows:
        record=row.get('following_reference_execution');stamp=row['stamp_s']
        if not record:continue
        counts['raw_rows']+=1;check(record['optimized_joint_names']==CANONICAL,stamp,'order','native canonical15')
        ctx=row['current_solver_task_context']
        if ctx is not None:check(ctx==contexts[ctx['task_id']],stamp,'Task','original complete Task/path digest')
        evidence=record.get('request_evidence')
        if record['request_submitted']and evidence is not None:
            acq=evidence['acquisition_stamp_s'];prior=by_stamp.get(round(acq,8))
            check(prior is not None,stamp,'request','same-stamp measured row exists')
            check(abs(acq-(stamp-.02))<1e-9 and evidence['velocity_stamp_s']==acq,stamp,'request','outer acquisition/velocity timestamp')
            check(evidence['joint_names']==CANONICAL and evidence['measurement_source']=='measured_physx',stamp,'request','measured canonical ownership')
            check(evidence['task_id']==ctx['task_id']and evidence['task_sha256']==ctx['task_sha256'],stamp,'request','same full admitted Task')
            if prior is not None:
                check(same(evidence['measured_q_rad'],vector(prior,'joint_positions')),stamp,'request','owned q equals actual29 at acquisition')
                check(same(evidence['measured_qd_rad_s'],vector(prior,'measured_joint_velocity')),stamp,'request','owned qd equals actual29 at acquisition')
                check(evidence['committed_reference']==prior['following_reference_execution']['final_reference_snapshot'],stamp,'request','r is prior actual final-write ledger, not planned')
            old=evidence['committed_reference'];ref_ok(old,stamp)
            check(abs(old['snapshot_stamp_s']-acq)<1e-9,stamp,'request','reference snapshot same acquisition')
            plan=evidence['planned_advance_bound'];delta=np.asarray(plan['delta_rad']);r=np.asarray(old['positions_rad']);planned=r+delta
            check(plan['source']=='planned_not_committed'and plan['from_reference_token']==old['token']and
                plan['task_id']==ctx['task_id']and plan['task_sha256']==ctx['task_sha256']and plan['request_stamp_s']==acq,stamp,'request','separate planned context')
            check(same(planned,planned.astype(np.float32).astype(float)),stamp,'request','planned target is actualfloat32 embedding')
            ulp=.5*(abs(np.spacing(r.astype(np.float32)).astype(float))+abs(np.spacing(planned.astype(np.float32)).astype(float)))
            check(same(ulp,plan['native_float32_roundoff_rad']),stamp,'request','explicit bounded float32 allowance')
            check(np.all(abs(delta)<=.01+ulp+4*np.finfo(float).eps),stamp,'request','planned delta original speed cap plus explicit rounding only')
            requests[round(acq,8)]=dict(evidence=evidence,sha=digest(evidence),row=row)
            counts['submitted_guarded_requests']+=1
    previous_write=None
    for row in rows:
        record=row.get('following_reference_execution');stamp=row['stamp_s'];start=stamp-.02
        if not record:continue
        ctx=row['current_solver_task_context'];packet=row['pink_runtime_recovery'];auth=record.get('authorization')
        native=record.get('native_report');lease=row.get('upper_execution_lease');writes=record['physical_reference_writes']
        payload=None
        if native is not None:
            acq=native['acquisition_stamp_s'];original=requests.get(round(acq,8))
            check(original is not None,stamp,'native_report','captured submitted original request')
            if original is not None:
                check(native['evidence_sha256']==original['sha'],stamp,'native_report','canonical evidence SHA')
                binding=native['application_binding']
                check(native['application_binding_sha256']==digest(binding),stamp,'native_report','application binding digest')
                check(binding['evidence_sha256']==original['sha']and binding['committed_reference']==original['evidence']['committed_reference'],stamp,'native_report','same frozen original reference/evidence')
                check(same(binding['planned_delta_rad'],original['evidence']['planned_advance_bound']['delta_rad']),stamp,'native_report','same frozen planned delta')
                worker=native.get('worker_request_binding')
                if worker:
                    check(worker['evidence_sha256']==original['sha']and worker['stamp_s']==acq and
                        worker['task_sha256']==original['evidence']['task_sha256'],stamp,'native_report','worker request bound to original submission')
                if auth is not None:
                    check(worker is not None and worker['seq']==packet['seq']and worker['stamp_s']==packet['stamp_s'],stamp,'native_report','fresh seq/acquisition binding')
                    current=by_stamp[round(start,8)]['following_reference_execution']['final_reference_snapshot']
                    old=np.asarray(binding['committed_reference']['positions_rad']);diff=np.asarray(current['positions_rad'])-old
                    delta=np.asarray(binding['planned_delta_rad'])
                    check(np.all(diff>=np.minimum(0,delta)-1e-10)and np.all(diff<=np.maximum(0,delta)+1e-10),stamp,'before_first_use','real prior ledger inside original declared planned box')
                    check(current['task_sha256']==ctx['task_sha256']and start-acq+.02<=.06+1e-9,stamp,'before_first_use','same Task and whole-step lease')
                    check(record['before_first_use']['evidence_sha256']==original['sha']and
                        record['before_first_use']['reference_rows_rebased']is False,stamp,'before_first_use','report matches and no rebase')
        if auth is not None:
            payload=auth['payload'];counts['native_step_authorizations']+=1
            check(auth['sha256']==digest(payload),stamp,'authorization','exact payload digest')
            check(payload['task_context']==ctx and payload['packet_seq']==packet['seq'],stamp,'authorization','full Task/seq')
            check(abs(payload['application_stamp_s']-start)<1e-9 and payload['acquisition_stamp_s']==packet['stamp_s'],stamp,'authorization','acquisition/application times')
            before=np.asarray(payload['integrator_before_rad']);v=np.asarray(payload['native_velocity_rad_s']);proposal=before+v*.02
            check(same(proposal,payload['float64_proposal_rad'])and payload['native_step_was_clipped']is False,stamp,'authorization','native vdt unchanged')
            check(np.all(abs(v)<=.5+1e-10),stamp,'authorization','original native speed cap')
            check(same(proposal.astype(np.float32).astype(float),payload['authorized_target_rad']),stamp,'authorization','sole known float32 embedding')
            check(same(before.astype(np.float32).astype(float),payload['start_reference']['positions_rad']),stamp,'authorization','internal-before equals actual committed start after known quantization')
            check(same(v,[lease['original_requested_velocity'][n]for n in CANONICAL]),stamp,'authorization','original native velocity source')
            check(payload['start_reference']==by_stamp[round(start,8)]['following_reference_execution']['final_reference_snapshot'],stamp,'authorization','actual start is prior end ledger')
        check(len(writes)==4,stamp,'physical_writes','four actual substep ledger events')
        for index,write in enumerate(writes):
            now=start+index*.005;counts['physical_write_events']+=1;ref_ok(write,stamp)
            check(abs(write['commit_stamp_s']-now)<1e-9 and abs(write['snapshot_stamp_s']-now)<1e-9,stamp,'physical_writes','4x5ms write timestamps')
            check(write['task_id']==(None if ctx is None else ctx['task_id']),stamp,'physical_writes','owner changes only after actual current-Task write')
            if previous_write is not None:check(write['commit_seq']==previous_write['commit_seq']+1,stamp,'physical_writes','physical event sequence increments once')
            if payload is not None:
                old=np.asarray(payload['start_reference']['positions_rad']);new=np.asarray(payload['authorized_target_rad']);actual=np.asarray(write['positions_rad'])
                check(same(actual,old)or same(actual,new),stamp,'physical_writes','atomic complete start/new, no partial mixing')
                if previous_write is not None and same(previous_write['positions_rad'],new)and not same(old,new):
                    check(not same(actual,old),stamp,'physical_writes','no reversion after own new native target')
                if lease and index<len(lease['checks']):
                    item=lease['checks'][index];sim_age=now-payload['acquisition_stamp_s']
                    check(abs(item['sim_age_s']-sim_age)<1e-9,stamp,'lease','independent sim age')
                    truth=(0<=sim_age+1e-9 and sim_age+item['execution_horizon_s']<=.06+1e-9 and
                        0<=item['wall_age_s']and item['wall_age_s']+item['execution_horizon_s']<=.06+1e-9)
                    check(item['valid']==truth,stamp,'lease','60ms wall and sim including substep horizon')
                    if item['valid']:maxima['fresh_command_sim_end_age_s']=max(maxima['fresh_command_sim_end_age_s'],sim_age+.005)
                    elif previous_write is not None:check(same(actual,previous_write['positions_rad']),stamp,'hold','expired packet cannot issue new target')
            elif previous_write is not None:
                check(same(write['positions_rad'],previous_write['positions_rad']),stamp,'hold','no authorized packet means upper reference held')
            previous_write=write
        final=record['final_reference_snapshot'];ref_ok(final,stamp)
        check(final['commit_seq']==writes[-1]['commit_seq']and final['token']==writes[-1]['token']and
            final['commit_stamp_s']==writes[-1]['commit_stamp_s']and same(final['positions_rad'],writes[-1]['positions_rad']),stamp,'final_reference','actual last write ledger')
        check(abs(final['snapshot_stamp_s']-stamp)<1e-9 and same(final['positions_rad'],vector(row,'applied_joint_position_target')),stamp,'final_reference','raw actual applied target canonical15')
        if row['current_solver_task_context']is not None and row['controller_status']!='fresh_command':
            check(not row['pink_upper_velocity']and row['task_feedback']['reason']!='accepted',stamp,'hold','waiting/error/expiry not presented as fresh accepted motion')
    summary=json.loads((folder/'summary.json').read_text())
    paths=[folder/'measurements.jsonl',folder/'timed_tasks.json',folder/'summary.json',out/'sources'/Path(__file__).name]
    report=dict(schema='g1_following_raw_independent/v1',imports_runtime_helpers=False,errors=errors,
        interface_binding_pass=not errors,physical_run_complete=summary['run_complete'],physical_all_pass=summary['all_pass'],
        scope='Raw request→fresh report→actual written target logging and independent algebra only. Does not infer physical convergence or collision safety from reference logs.',
        limitations=['Write ledger records physical position targets, not per-substep velocity/gravity target arrays; their exact zeros cannot be independently reconstructed from raw numeric fields alone.',
            'Actual PhysX integration/actuation is observed through the recorded measurements and after-write ledger, not independently replayed here.'],
        counts=dict(counts),maxima=dict(maxima),first_stamp_s=rows[0]['stamp_s'],last_stamp_s=rows[-1]['stamp_s'],
        failures=summary['failures'],evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in paths})
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(errors=len(errors),counts=dict(counts),first_errors=errors[:6]),indent=2))


if __name__=='__main__':main()
