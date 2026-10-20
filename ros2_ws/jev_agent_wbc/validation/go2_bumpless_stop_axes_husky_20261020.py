#!/usr/bin/env python3
"""Read-only axes/short-window spectra of the two best-profile failed stops."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import detrend
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_bumpless_z_01'
OUT=ROOT/'Robot_Go2/results/bumpless_stop_axes_cpu_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(a):
 a=np.asarray(a);return dict(mean=a.mean(0).tolist(),span=np.ptp(a,axis=0).tolist(),first=a[0].tolist(),last=a[-1].tolist(),rms=np.sqrt(np.mean(a*a,axis=0)).tolist())
def spectrum(a,dt=.02):
 a=np.asarray(a);v=detrend(a,axis=0);n=len(v);f=np.fft.rfftfreq(n,dt);s=np.abs(np.fft.rfft(v*np.hanning(n).reshape(-1,*([1]*(v.ndim-1))),axis=0))**2
 ok=(f>=.2)&(f<=5.);power=s[ok];freq=f[ok];idx=np.argmax(power,axis=0)
 return dict(samples=n,duration_s=(n-1)*dt,bin_spacing_hz=1/(n*dt),dominant_detrended_bin_hz=freq[idx].tolist(),caution='Short nonstationary stop; bins are diagnostic, not a validated oscillator frequency.')
def main():
 rows=[json.loads(x)for x in(RUN/'measurements.jsonl').read_text().splitlines()];summary=json.loads((RUN/'summary.json').read_text());records=[]
 for case in ['yaw_left','camera_shot']:
  actual=[r for r in rows if r['case']==case];stop=[r for r in actual if r['phase']=='stop'];tail=stop[-26:];late=[r for r in stop if r['stamp_s']-.02>=r['terminal_allocation']['reference_stamp_s']+2-1e-9]
  result=dict(case=case)
  for name,part in [('last_half_second',tail),('last_three_seconds',late)]:
   bases=np.asarray([r['T_world_base']for r in part]);cams=np.asarray([r['T_world_camera_optical']for r in part]);R=bases[0,:3,:3]
   base_delta=(bases[:,:3,3]-bases[0,:3,3])@R
   cam_delta=(cams[:,:3,3]-cams[0,:3,3])@R
   base_rot=np.asarray([Rotation.from_matrix(R.T@T[:3,:3]).as_rotvec()for T in bases])
   camR=cams[0,:3,:3];cam_rot=np.asarray([R.T@Rotation.from_matrix(T[:3,:3]@camR.T).as_rotvec()for T in cams])
   v=np.asarray([bases[i,:3,:3].T@np.asarray(r['base_linear_velocity_world'])for i,r in enumerate(part)])
   gyro=np.asarray([bases[i,:3,:3].T@np.asarray(r['base_angular_velocity_world'])for i,r in enumerate(part)])
   q=np.asarray([r['joint_positions']for r in part]);target=np.asarray([r['applied_joint_position_target']for r in part]);qd=np.asarray([r['measured_joint_velocity']for r in part]);span=np.ptp(q,axis=0);worst=np.argsort(span)[-4:][::-1]
   selected=['FR_calf_joint','FL_calf_joint','RR_calf_joint','RL_calf_joint'];ids=[part[0]['joint_names'].index(n)for n in selected]
   result[name]=dict(stamps_s=[part[0]['stamp_s'],part[-1]['stamp_s']],
    base_translation_in_initial_base_m=stat(base_delta),camera_translation_in_initial_base_m=stat(cam_delta),
    base_rotation_local_rotvec_rad=stat(base_rot),camera_rotation_in_initial_base_rotvec_rad=stat(cam_rot),
    actual_velocity_in_current_base_mps=stat(v),actual_gyro_in_current_base_radps=stat(gyro),
    actual_native7=stat([r['actual_native_command']for r in part]),
    actual_applied_body_input=stat([r['applied_policy_velocity']for r in part]),
    worst_joint_names=[part[0]['joint_names'][i]for i in worst],worst_joint_span_rad=span[worst].tolist(),
    worst_joint_net_change_rad=(q[-1]-q[0])[worst].tolist(),worst_target_span_rad=np.ptp(target,axis=0)[worst].tolist(),
    worst_joint_velocity_radps=stat(qd[:,worst]),
    body_gyro_spectrum=spectrum(gyro),selected_calf_joint_names=selected,calf_q_spectrum=spectrum(q[:,ids]),calf_target_spectrum=spectrum(target[:,ids]),
    base_velocity_spectrum=spectrum(v))
  records.append(result)
 result=dict(schema='go2_failed_stop_axes_readonly/v1',run=str(RUN.relative_to(ROOT)),records=records,
  scope='Actual physical trace axes in measured base frames; no dynamics identification or new feedback controller executed. Full3D translation/rotation and all18q stop criteria remain.',
  physical_run_complete=summary['run_complete'],original_all_pass=summary['all_pass'],
  source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),RUN/'summary.json',RUN/'measurements.jsonl',RUN/'timed_tasks.json',RUN/'provenance.json']})
 OUT.mkdir(exist_ok=False);(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
