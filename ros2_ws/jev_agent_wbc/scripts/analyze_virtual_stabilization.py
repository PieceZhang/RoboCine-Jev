#!/usr/bin/env python3
"""Independent physical-pose and image-projection audit of the recorded DDS output."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.virtual_stabilizer import se3_log


def rms(a):return float(np.sqrt(np.mean(np.sum(a*a,axis=1))))

def audit(folder):
 folder=Path(folder);rows=json.loads((folder/'virtual_camera_physics.json').read_text())
 t=np.array([x['t'] for x in rows]);fs=1/np.median(np.diff(t));mask=(t>=1.)&(t<=t[-1]-.5)
 rp=np.array([x['rp'] for x in rows]);rr=np.array([x['rr'] for x in rows]);ap=np.array([x['ap'] for x in rows]);ar=np.array([x['ar'] for x in rows]);sp=np.array([x['stabilized']['p'] for x in rows]);sr=np.array([x['stabilized']['r'] for x in rows])
 if np.max(abs(np.diff(t)-1/fs))>1e-7:raise ValueError('physics samples are not uniform')
 raw=np.array([se3_log(r.T@(a-p),r.T@b) for p,r,a,b in zip(rp,rr,ap,ar)])
 stab=np.array([se3_log(r.T@(a-p),r.T@b) for p,r,a,b in zip(rp,rr,sp,sr)])
 subject=np.array(json.loads((folder/'runtime_scene.json').read_text())['subject']['position'])
 focal=240/np.tan(np.deg2rad(55)/2)
 def pixels(p,r):
  local=np.einsum('nji,nj->ni',r,subject-p)
  return -focal*local[:,1:]/local[:,:1]
 raw_pixel=pixels(ap,ar)-pixels(rp,rr);stab_pixel=pixels(sp,sr)-pixels(rp,rr)
 bands={}
 for cutoff in (1.,3.,6.):
  sos=butter(4,cutoff,fs=fs,btype='highpass',output='sos')
  a,b=(sosfiltfilt(sos,x,axis=0)[mask] for x in (raw,stab));pa,pb=(sosfiltfilt(sos,x,axis=0)[mask] for x in (raw_pixel,stab_pixel))
  bands[str(cutoff)]=dict(raw_translation_rms_mm=1000*rms(a[:,:3]),stabilized_translation_rms_mm=1000*rms(b[:,:3]),translation_reduction_pct=100*(1-rms(b[:,:3])/max(rms(a[:,:3]),1e-15)),raw_rotation_rms_deg=np.rad2deg(rms(a[:,3:])),stabilized_rotation_rms_deg=np.rad2deg(rms(b[:,3:])),rotation_reduction_pct=100*(1-rms(b[:,3:])/max(rms(a[:,3:]),1e-15)),raw_subject_jitter_rms_px=rms(pa),stabilized_subject_jitter_rms_px=rms(pb),projected_jitter_reduction_pct=100*(1-rms(pb)/max(rms(pa),1e-15)))
 info=[x['stabilized']['info'] for x in rows]
 tele=json.loads((folder/'telemetry.json').read_text());state=np.array([x['state'] for x in tele]);measured=np.array([x['measured'] for x in tele])
 result=dict(source='live PhysX rigid-body poses and timestamp-matched independent ROS stabilizer outputs; no synthetic shake',folder=str(folder.resolve()),physics_samples=len(t),sample_hz=fs,duration_s=t[-1],evaluation_window_s=[1.,t[-1]-.5],bands_above_hz=bands,
 correction=dict(max_translation_mm=1000*max(x['translation_correction_m'] for x in info),max_rotation_deg=max(x['rotation_correction_deg'] for x in info),translation_limit_fraction=np.mean([x['translation_limited'] for x in info]),rotation_limit_fraction=np.mean([x['rotation_limited'] for x in info]),reset_count=sum(x['reset'] for x in info)),
 robot_motion=dict(base_path_m=float(np.linalg.norm(np.diff(state[:,:2],axis=0),axis=1).sum()),base_yaw_span_deg=float(np.rad2deg(np.ptp(np.unwrap(state[:,2])))),joint_range_deg=np.rad2deg(np.ptp(state[:,3:],axis=0)).tolist(),max_linear_speed_m_s=float(abs(measured[:,0]).max()),max_base_yaw_rate_rad_s=float(abs(measured[:,1]).max()),max_joint_speed_rad_s=float(abs(measured[:,2:]).max())),
 tracking=dict(raw_position_rms_mm=1000*rms(ap-rp),stabilized_position_rms_mm=1000*rms(sp-rp),raw_rotation_rms_deg=float(np.rad2deg(rms(raw[:,3:]))),stabilized_rotation_rms_deg=float(np.rad2deg(rms(stab[:,3:])))),wbc_summary=json.loads((folder/'summary.json').read_text()))
 if (folder/'video/frames.json').exists():
  frames=json.loads((folder/'video/frames.json').read_text());result['video']=dict(frames=len(frames),same_pass=all(x['capture_protocol']=='same_render_pass_same_simulation_timestamp' for x in frames),times_strictly_increasing=all(b['t']>a['t'] for a,b in zip(frames,frames[1:])))
 (folder/'stabilization_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',nargs='+');a=p.parse_args()
 for f in a.folders:
  r=audit(f);print(json.dumps({k:v for k,v in r.items() if k not in ['wbc_summary']},indent=2))
