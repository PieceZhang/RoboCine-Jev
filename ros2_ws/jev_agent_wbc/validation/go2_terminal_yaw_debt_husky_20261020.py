#!/usr/bin/env python3
"""Read-only optical/physical/native yaw and Z debt in original full recording."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_terminal_01'
OUT=ROOT/'Robot_Go2/results/terminal_yaw_debt_cpu_01'


def yaw(T):return float(Rotation.from_matrix(np.asarray(T)[:3,:3]).as_euler('xyz')[2])
def native_yaw(value):return float(Rotation.from_quat(np.asarray(value)[[4,5,6,3]]).as_euler('xyz')[2])
def delta(start,end):return float(np.arctan2(np.sin(end-start),np.cos(end-start)))


def main():
    rows=[json.loads(line)for line in(RUN/'measurements.jsonl').read_text().splitlines()]
    summary=json.loads((RUN/'summary.json').read_text());results=[]
    for case in summary['cases']:
        r=[(index,row)for index,row in enumerate(rows)if row['case']==case['case']];first=rows[r[0][0]-1]
        stop=[(index,row)for index,row in r if row['phase']=='stop'];before_stop=rows[stop[0][0]-1]
        end=stop[0][1]['terminal_allocation']['reference_stamp_s'];after2=next(row for _,row in stop if row['stamp_s']-.02>=end+2-1e-9);last=r[-1][1]
        boundaries=[]
        for name,row in [('first_measured',first),('pre_stop',before_stop),('post2_measured',after2),('last_measured',last)]:
            record=dict(boundary=name,stamp_s=row['stamp_s'],base_yaw_rad=yaw(row['T_world_base']),camera_yaw_rad=yaw(row['T_world_camera_optical']),
                original_reference_yaw_rad=yaw(row['T_world_reference_optical']),
                actual_native_yaw_rad=native_yaw(row['actual_native_command'])if 'actual_native_command'in row else None,
                uncorrected_native_yaw_rad=native_yaw(row['requested_native_command'])if 'requested_native_command'in row else None,
                actual_native_z_m=row.get('actual_native_command',[None,None,None])[2],
                position_error_mm=row['position_error_m']*1000,
                world_camera_z_m=np.asarray(row['T_world_camera_optical'])[2,3],reference_z_m=np.asarray(row['T_world_reference_optical'])[2,3])
            record['world_z_error_mm']=1000*(record['world_camera_z_m']-record['reference_z_m']);boundaries.append(record)
        command_integral=np.sum([row['applied_policy_velocity']for _,row in stop],axis=0)*.02
        actual_yaw_delta=delta(yaw(before_stop['T_world_base']),yaw(last['T_world_base']))
        results.append(dict(case=case['case'],boundaries=boundaries,
            moving_task_world_yaw_delta_rad=delta(yaw(first['T_world_camera_optical']),yaw(before_stop['T_world_reference_optical'])),
            moving_actual_base_yaw_delta_rad=delta(yaw(first['T_world_base']),yaw(before_stop['T_world_base'])),
            moving_actual_camera_yaw_delta_rad=delta(yaw(first['T_world_camera_optical']),yaw(before_stop['T_world_camera_optical'])),
            stop_original_endpoint_world_yaw_delta_rad=delta(yaw(stop[0][1]['T_world_reference_optical']),yaw(last['T_world_reference_optical'])),
            stop_actual_base_yaw_delta_rad=actual_yaw_delta,
            stop_integrated_applied_body_input=command_integral.tolist(),
            stop_integrated_yaw_command_to_actual_ratio=(actual_yaw_delta/command_integral[2]if abs(command_integral[2])>1e-5 else None),
            last_task_reference_z_m=np.asarray(last['T_world_reference_optical'])[2,3],
            zi_integral_end_mm=1000*last['actual_native_domain']['world_z_feedback']['integral_after_m']))
    out=dict(schema='go2_terminal_yaw_debt_readonly/v1',records=results,
        note='Euler yaw is diagnostic; native yaw is full link-relative source convention. Body twist includes roll/pitch, so integral wz is not exactly Euler yaw delta. Original raw/source remain untouched.',
        files_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()for path in[Path(__file__),RUN/'measurements.jsonl',RUN/'timed_tasks.json']})
    OUT.mkdir(exist_ok=False);(OUT/'diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
