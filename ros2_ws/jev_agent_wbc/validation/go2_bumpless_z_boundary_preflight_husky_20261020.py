#!/usr/bin/env python3
"""Admission-only measured seed arithmetic on the eight unchanged Task boundaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_terminal_pose_01'
OUT=ROOT/'Robot_Go2/results/bumpless_z_seed_cpu_01'
sys.path.insert(0,str(RUN/'sources'))
from jev_agent_wbc.wbc_protocol import TimedTask


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False);frozen=OUT/'sources';frozen.mkdir()
    paths=[ROOT/'Robot_Go2/world_z_bumpless.py',ROOT/'Robot_Go2/world_z_feedback.py',ROOT/'test/test_wbc_go2_bumpless_z_seed.py',Path(__file__)]
    for path in paths:shutil.copy2(path,frozen/path.name)
    wrapper=load('fresh_bumpless_wrapper',frozen/'world_z_bumpless.py')
    backend=load('original_bumpless_backend',RUN/'sources/go2_backend.py');backend.UPSTREAM_MDP=RUN/'sources/go2_upstream'
    sampler=load('original_bumpless_sampler',RUN/'sources/optical_sampler.py');domain=backend.PolicyDomain()
    rows=[json.loads(line)for line in(RUN/'measurements.jsonl').read_text().splitlines()]
    summary=json.loads((RUN/'summary.json').read_text());tasks=[TimedTask.from_dict(value)for value in json.loads((RUN/'timed_tasks.json').read_text())]
    extrinsic=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    results=[]
    for number,case in enumerate(summary['cases']):
        index=next(index for index,row in enumerate(rows)if row['case']==case['case']);before=rows[index-1];now=before['stamp_s'];task=tasks[number]
        body=np.asarray(before['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07])
        camera=np.asarray(before['T_world_camera_optical']);reference=sampler.OpticalSampler(task).sample(now)
        controller=wrapper.BumplessWorldZFeedback(enabled=True,ki_per_s=.2)
        previous=np.asarray(before['actual_native_command'],dtype=np.float32)if number else None
        decision=controller.admit(backend,task,now_s=now,observation_stamp_s=now,reference_optical=reference,
            measured_optical=camera,link0=link,extrinsic=extrinsic,previous_applied_native=previous,
            previous_issue_stamp_s=now-.02 if number else None,history_rows=rows[max(0,index-26):index],domain=domain)
        native,check=controller.feedback_native_command(backend,reference,camera,link,extrinsic,
            task_id=task.task_id,stamp_s=now,now_s=now,dt_s=.02,sample_seq=round(now/.02),fresh=True,integration_allowed=True,domain=domain)
        report=check['world_z_feedback']
        if report.get('seed_pending_commit'):controller.finish_seed_update(report['update_id'],applied_native=native)
        old_jump=rows[index]['actual_native_command'][2]-float(previous[2])if previous is not None else None
        new_jump=float(np.float32(native[2])-previous[2])if previous is not None else None
        results.append(dict(case=case['case'],calibration=decision,old_native_z_jump_m=old_jump,
            proposed_bounded_native_z_jump_m=new_jump,remaining_I_m=controller.integral_m,
            original_total_correction_norm_m=report['proposed_position_correction_norm_m'],
            native_domain=check['status'],task_sha_preserved=decision['task_sha256']==wrapper.canonical_task(task)[1]))
    report=dict(schema='go2_bumpless_boundary_cpu/v1',boundaries=len(results),eligible_count=sum(row['calibration']['eligible']for row in results),
        cpu_gate_pass=all(row['task_sha_preserved']and row['original_total_correction_norm_m']<=.025+1e-12 and
            row['native_domain']=='inside_outer_domain_unverified'for row in results),records=results,
        scope='Separate frozen measured admissions only. No old Task/log edits, no propagation or physical policy trial. FirstTask remains0; unstable/unsafe/missing evidence stays reset0.',
        physical_verified=False,existing_world_z_source_unchanged=sha(ROOT/'Robot_Go2/world_z_feedback.py')=='3d495f4cfbd81ed45ba744ead7d831c53253d4e597163e0319a44871b9d37a6a',
        files_sha256={str(path.relative_to(ROOT)):sha(path)for path in paths+[RUN/'measurements.jsonl',RUN/'timed_tasks.json']})
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
