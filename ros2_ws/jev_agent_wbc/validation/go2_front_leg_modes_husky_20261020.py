#!/usr/bin/env python3
"""Read-only front-leg common/differential motion versus actual body gyro."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.signal import detrend
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_output_continuity_01'
OUT=ROOT/'Robot_Go2/results/front_leg_modes_cpu_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def correlation(a,b):
 a=detrend(a);b=detrend(b)
 return float(np.corrcoef(a,b)[0,1])if np.std(a)*np.std(b)>1e-12 else None

def main():
 rows=[json.loads(x)for x in(RUN/'measurements.jsonl').read_text().splitlines()];result=[]
 for name in ['yaw_left','camera_shot']:
  allrows=[r for r in rows if r['case']==name];late=[r for r in allrows if r['phase']=='stop'and r['stamp_s']-.02>=r['terminal_allocation']['reference_stamp_s']+2-1e-9]
  for window,part in [('last3s',late),('last_half_second',late[-26:])]:
   names=part[0]['joint_names'];ids=[names.index(n)for n in ['FR_calf_joint','FL_calf_joint','RR_calf_joint','RL_calf_joint']]
   q=np.asarray([r['joint_positions']for r in part])[:,ids];qd=np.asarray([r['measured_joint_velocity']for r in part])[:,ids];target=np.asarray([r['applied_joint_position_target']for r in part])[:,ids]
   gyro=np.asarray([np.asarray(r['T_world_base'])[:3,:3].T@r['base_angular_velocity_world']for r in part]);vel=np.asarray([np.asarray(r['T_world_base'])[:3,:3].T@r['base_linear_velocity_world']for r in part])
   R0=np.asarray(part[0]['T_world_base'])[:3,:3];angles=np.asarray([Rotation.from_matrix(R0.T@np.asarray(r['T_world_base'])[:3,:3]).as_rotvec()for r in part])
   common=q[:,:2].mean(1);diff=(q[:,1]-q[:,0])/2;common_qd=qd[:,:2].mean(1);diff_qd=(qd[:,1]-qd[:,0])/2
   result.append(dict(case=name,window=window,stamps_s=[part[0]['stamp_s'],part[-1]['stamp_s']],
    front_calf_common_span_rad=float(np.ptp(common)),front_calf_differential_span_rad=float(np.ptp(diff)),
    front_target_common_span_rad=float(np.ptp(target[:,:2].mean(1))),
    front_calf_pair_correlation=correlation(q[:,0],q[:,1]),
    common_position_vs_body_rotvec_corr=[correlation(common,angles[:,i])for i in range(3)],
    common_qd_vs_body_gyro_corr=[correlation(common_qd,gyro[:,i])for i in range(3)],
    differential_qd_vs_body_gyro_corr=[correlation(diff_qd,gyro[:,i])for i in range(3)],
    body_velocity_rms_mps=np.sqrt(np.mean(vel*vel,axis=0)).tolist(),body_gyro_rms_radps=np.sqrt(np.mean(gyro*gyro,axis=0)).tolist(),
    all_actual_brake_veto_count=sum(bool(r['terminal_allocation']['veto_reasons'])for r in part),
    scope='Correlation and short measured-window decomposition only; closed-loop common causes prevent causal plant/control certification.'))
 report=dict(schema='go2_front_calf_modes/v1',records=result,controller_changed=False,physical_verified=False,
  source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),RUN/'measurements.jsonl',RUN/'provenance.json',RUN/'timed_tasks.json']})
 OUT.mkdir(exist_ok=False);(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
