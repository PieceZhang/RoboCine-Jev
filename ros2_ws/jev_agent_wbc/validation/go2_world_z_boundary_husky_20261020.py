#!/usr/bin/env python3
"""Read-only measured-command jumps across fresh Task-local Z-integral resets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_terminal_01'
OUT=ROOT/'Robot_Go2/results/world_z_boundary_readonly_01'
sys.path.insert(0,str(RUN/'sources'))
from jev_agent_wbc.wbc_protocol import TimedTask


def main():
    spec=importlib.util.spec_from_file_location('boundary_frozen_sampler',RUN/'sources/optical_sampler.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows=[json.loads(line)for line in(RUN/'measurements.jsonl').read_text().splitlines()]
    tasks=[TimedTask.from_dict(value)for value in json.loads((RUN/'timed_tasks.json').read_text())];samplers=[m.OpticalSampler(task)for task in tasks]
    cases=[value['case']for value in json.loads((RUN/'summary.json').read_text())['cases']]
    results=[]
    for n,case in enumerate(cases):
        i=next(i for i,row in enumerate(rows)if row['case']==case);records=[]
        for j in range(max(0,i-2),i+2):
            row=rows[j];prev=rows[j-1]if j else row;index=cases.index(row['case'])if row['case']in cases else None
            if index is None:continue
            issue=row['stamp_s']-.02;reference=samplers[index].sample(min(issue,tasks[index].trajectory.end_s))
            before=np.asarray(prev['T_world_camera_optical']);zi=row['actual_native_domain']['world_z_feedback']
            record=dict(measured_stamp_s=row['stamp_s'],issue_stamp_s=issue,case=row['case'],task_id=zi['task_id'],
                actual_native_ee_world_z_m=row['actual_native_command'][2],uncorrected_reference_ee_world_z_m=row['requested_native_command'][2],
                total_actual_native_z_correction_m=row['actual_native_command'][2]-row['requested_native_command'][2],
                zi_integral_before_m=zi['integral_before_m'],zi_integral_after_m=zi['integral_after_m'],
                p_world_camera_z_correction_m=.35*(reference[2,3]-before[2,3]),
                proposed_projected_world_z_correction_m=zi['proposed_position_correction_m'][2],
                before_control_camera_world_z_m=before[2,3],after_control_camera_world_z_m=np.asarray(row['T_world_camera_optical'])[2,3],
                original_task_reference_camera_world_z_m=reference[2,3],zi_reset_inferred=zi['integral_before_m']==0.,
                native_z_includes_orientation_extrinsic_effect=True)
            records.append(record)
        first=next(row for row in records if row['case']==case)
        old=next((row for row in reversed(records)if row['case']!=case),None)
        results.append(dict(new_case=case,boundary_issue_stamp_s=first['issue_stamp_s'],records=records,
            native_z_jump_m=first['actual_native_ee_world_z_m']-old['actual_native_ee_world_z_m']if old else None,
            raw_reference_ee_z_jump_m=first['uncorrected_reference_ee_world_z_m']-old['uncorrected_reference_ee_world_z_m']if old else None,
            zi_before_jump_m=first['zi_integral_before_m']-old['zi_integral_after_m']if old else None,
            prospective_bumpless_world_z_bias_m=old['actual_native_ee_world_z_m']-first['uncorrected_reference_ee_world_z_m']if old else None))
    report=dict(schema='go2_task_boundary_z_readonly/v1',records=results,
        note='No changes to runtime,Ki,Task,clock or reset. Prospective bias is scalar continuity arithmetic only: not an implemented controller or a justified blind carry across Tasks; full3D projection/domain/freshness/outerrollback remain required.',
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),RUN/'measurements.jsonl',RUN/'timed_tasks.json']})
    OUT.mkdir(exist_ok=False);(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
